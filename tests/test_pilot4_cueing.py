import asyncio
import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from muse_tmr.audio import (
    AudioBackendFallbackError,
    AudioCuePlayer,
    AudioPlaybackConfig,
    CueLibrary,
    CueMetadata,
    MockAudioBackend,
    VolumeCalibration,
)
from muse_tmr.data.replay import ReplayConfig, ReplaySession
from muse_tmr.data.sample_types import EEGSample, HeartRateSample, IMUSample, MuseFrame, PPGSample
from muse_tmr.features import EpochConfig
from muse_tmr.models import RemGateConfig
from muse_tmr.protocol import (
    ArousalGuardConfig,
    NightArmContext,
    NightPuzzleSession,
    PuzzleCatalog,
    PuzzleCueAssignment,
    PuzzleTask,
    TlrBlockConfig,
    TmrSchedulerConfig,
    TmrSchedulerEvent,
    default_tlr_cue,
    load_night_schedule,
    plan_tlr_block,
)
from muse_tmr.cli import main as cli_main
from muse_tmr.protocol import get_protocol_preset
from muse_tmr.reports import MorningRetestResult, build_morning_retest
from muse_tmr.sources.base_source import MuseDeviceInfo, MuseSourceMetadata
from muse_tmr.testing import SyntheticNightConfig, generate_night
from muse_tmr.validation import (
    AwakeningEvent,
    Pilot4CueingConfig,
    append_awakening_event,
    run_pilot4_cueing_night,
)
from muse_tmr.validation import pilot4 as pilot4_module
from tests.rem_event_synthetic import rem_deflection_train

COARSE_CLOCK = sys.platform == "win32" and sys.version_info < (3, 13)
COARSE_CLOCK_REASON = (
    "on Windows before Python 3.13 time.monotonic() ticks every 15.6 ms and asyncio runs a timer up to one "
    "tick early, so after the last wait the deadline can read as not reached and the night ends as "
    "no_data_timeout"
)


EEG_RATE_HZ = 256
IMU_RATE_HZ = 52
PPG_RATE_HZ = 64


class FakeMuseSource:
    def __init__(self, frames):
        self.frames = tuple(frames)
        self.stop_calls = 0

    async def discover(self):
        return (MuseDeviceInfo(name="Fake Muse", address="fake", rssi=0),)

    async def connect(self, device=None):
        return MuseSourceMetadata(
            source_name="fake",
            device_name="Fake Muse",
            device_id="fake-device",
            capabilities={"eeg": True, "imu": True, "ppg": True, "raw_packets": True},
        )

    async def stream(self):
        for frame in self.frames:
            await asyncio.sleep(0)
            yield frame

    async def stop(self):
        self.stop_calls += 1


class LoopWatchingBackend(MockAudioBackend):
    """Mock audio that, while it plays a cue, hands the event loop a callback from the
    thread it plays on, as bleak hands it each BLE notification, and waits up to 1 s for
    the loop to run it."""

    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.loop_ran = []

    def play_test_cue(self, request):
        ran = threading.Event()
        self.loop.call_soon_threadsafe(ran.set)
        self.loop_ran.append(ran.wait(timeout=1.0))
        return super().play_test_cue(request)


class HangingMuseSource(FakeMuseSource):
    def __init__(self, *, sleep_seconds=3600.0):
        super().__init__(())
        self.sleep_seconds = sleep_seconds

    async def stream(self):
        await asyncio.sleep(self.sleep_seconds)
        raise AssertionError("hanging source should be cancelled by the pilot timeout")
        yield


class TestPilot4Cueing(unittest.IsolatedAsyncioTestCase):
    async def test_low_volume_cueing_plays_only_cued_puzzles_with_calibration_cap(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        backend = MockAudioBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(4)),
                config=pilot4_config(Path(tmpdir)),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=backend,
            )

            audio_events = [
                json.loads(line)
                for line in Path(summary.audio_log_path).read_text(encoding="utf-8").splitlines()
            ]
            scheduler_events = [
                json.loads(line)
                for line in Path(summary.scheduler_events_path).read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(summary.passed)
        self.assertEqual(summary.cue_play_count, 2)
        self.assertEqual(summary.uncued_puzzle_play_count, 0)
        self.assertEqual([event["puzzle_id"] for event in scheduler_events if event["event_type"] == "play"], ["p1", "p3"])
        self.assertEqual(len(backend.requests), 2)
        self.assertLessEqual(summary.max_effective_volume, 0.04)
        self.assertTrue(all(event["effective_volume"] <= 0.04 for event in audio_events))
        self.assertTrue(all(event["backend_name"] == "mock" for event in audio_events))

    async def test_the_event_loop_keeps_running_while_a_cue_plays(self):
        # bleak hands each BLE notification to the event loop and the stream client stamps
        # it there, so a tone played on the loop would give every packet that came during
        # it the time the tone ended (#227).
        backend = LoopWatchingBackend(asyncio.get_running_loop())
        catalog, session, assignment, cue_library = protocol_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(4)),
                config=replace(pilot4_config(Path(tmpdir), duration_seconds=60.0), score_cue_arousal=False),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=backend,
            )

        self.assertEqual(summary.cue_play_count, 2)
        self.assertEqual(backend.loop_ran, [True, True])

    async def test_the_night_logs_its_gate_decisions_and_scores_its_cues(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(4)),
                config=pilot4_config(Path(tmpdir), duration_seconds=60.0),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
            )
            payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))
            gate_log = [
                json.loads(line)
                for line in Path(summary.rem_gate_decisions_path).read_text(encoding="utf-8").splitlines()
            ]
            written = sorted(path.name for path in Path(tmpdir).glob("cue_arousal*"))

        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["first_epoch_start"], 1000.0)
        # A night without an arm records none.
        self.assertEqual((payload["arm"], payload["night_id"], payload["schedule_commitment_sha256"]), (None, None, None))
        self.assertEqual([row["index"] for row in gate_log], [0, 1, 2, 3])
        self.assertEqual([row["gate_open"] for row in gate_log], [False, True, True, True])
        self.assertEqual([row["decision_seconds"] for row in gate_log], [30.0, 60.0, 90.0, 120.0])
        block = payload["cue_arousal"]
        self.assertEqual(block["status"], "scored")
        self.assertEqual((block["cue_count"], block["played_cue_count"]), (2, 2))
        # The frames of this night carry no decodable packets, so its own raw file gives
        # the windows nothing to score; the scorer says so instead of guessing.
        self.assertIsNone(block["arousal_proxy_per_cue"]["any"])
        self.assertIn("eeg_gap", block["limitations"])
        self.assertEqual(written, ["cue_arousal_events.jsonl", "cue_arousal_report.json", "cue_arousal_report.md"])
        self.assertEqual(Path(block["report_path"]).name, "cue_arousal_report.json")

    async def test_a_failing_scorer_leaves_the_summary_and_says_why(self):
        catalog, session, assignment, cue_library = protocol_fixture()

        async def broken(*args, **kwargs):
            raise RuntimeError("scorer broke")

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "muse_tmr.reports.cue_arousal.score_cue_arousal", broken
        ):
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(4)),
                config=pilot4_config(Path(tmpdir), duration_seconds=60.0),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
            )
            payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))

        self.assertTrue(summary.passed)
        self.assertEqual(payload["cue_arousal"], {"status": "failed", "error": "RuntimeError: scorer broke"})
        self.assertEqual(payload["cue_play_count"], 2)

    async def test_summary_reports_stride_and_first_cue_times(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(4)),
                config=pilot4_config(Path(tmpdir)),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
            )
            payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))

        # The second REM epoch opens the gate; its window ends 60 s after the first began.
        self.assertEqual(summary.stride_seconds, 30.0)
        self.assertEqual(summary.first_gate_open_seconds, 60.0)
        self.assertEqual(summary.first_play_seconds, 60.0)
        self.assertEqual(payload["stride_seconds"], 30.0)
        self.assertEqual(payload["first_gate_open_seconds"], 60.0)
        self.assertEqual(payload["first_play_seconds"], 60.0)

    async def test_stride_10_gate_needs_60_s_of_real_time(self):
        catalog, session, assignment, cue_library = protocol_fixture()

        async def run_at_stride_10(tmpdir):
            return await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(6)),
                config=replace(
                    pilot4_config(Path(tmpdir)),
                    epoch_config=EpochConfig(epoch_seconds=30, stride_seconds=10),
                ),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await run_at_stride_10(tmpdir)
            payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            pilot4_module, "epoch_step_seconds", lambda *args: 30.0
        ):
            before = await run_at_stride_10(tmpdir)

        # Six REM windows 10 s apart, the sixth starting at 50 s, so its decision is at
        # 80 s. Fed the 30 s epoch length, as before #225 1c, it opened on the second.
        self.assertEqual(summary.stride_seconds, 10.0)
        self.assertEqual(summary.first_gate_open_seconds, 80.0)
        self.assertEqual(summary.first_play_seconds, 80.0)
        self.assertEqual(payload["stride_seconds"], 10.0)
        self.assertEqual(summary.uncued_puzzle_play_count, 0)
        self.assertEqual(before.first_gate_open_seconds, 40.0)

    async def test_one_rem_event_tracker_for_the_whole_night(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        trackers = []
        extract = pilot4_module.extract_eeg_features

        def recording_extract(epoch, *args, **kwargs):
            trackers.append(kwargs.get("rem_tracker"))
            return extract(epoch, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            pilot4_module, "extract_eeg_features", recording_extract
        ):
            await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(4)),
                config=pilot4_config(Path(tmpdir)),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
            )

        self.assertGreaterEqual(len(trackers), 4)
        self.assertIsNotNone(trackers[0])
        self.assertTrue(all(tracker is trackers[0] for tracker in trackers))

    async def test_full_night_tlr_block_runs_before_puzzle_cues(self):
        catalog, session, assignment, cue_library = protocol_fixture(include_tlr=True)
        tlr_plan = plan_tlr_block(
            default_tlr_cue(),
            config=TlrBlockConfig(
                repetitions=2,
                interval_seconds=0.0,
                post_block_pause_seconds=0.0,
            ),
        )
        backend = MockAudioBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(5)),
                config=pilot4_config(
                    Path(tmpdir),
                    enable_tlr_block=True,
                    pilot_id="m8_pilot5_full_night_tlr_puzzle_cueing",
                    summary_filename="pilot5_summary.json",
                    require_tlr_block=True,
                ),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=backend,
                tlr_block_plan=tlr_plan,
            )
            scheduler_events = [
                json.loads(line)
                for line in Path(summary.scheduler_events_path).read_text(encoding="utf-8").splitlines()
            ]
            summary_payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))

        play_events = [event for event in scheduler_events if event["event_type"] == "play"]

        self.assertTrue(summary.passed)
        self.assertEqual(summary.pilot_id, "m8_pilot5_full_night_tlr_puzzle_cueing")
        self.assertEqual(Path(summary.summary_path).name, "pilot5_summary.json")
        self.assertTrue(summary.tlr_block_required)
        self.assertEqual(summary.tlr_cue_play_count, 2)
        self.assertGreaterEqual(summary.puzzle_cue_play_count, 1)
        self.assertEqual(summary.uncued_puzzle_play_count, 0)
        self.assertEqual([event["protocol"] for event in play_events[:2]], ["tlr", "tlr"])
        self.assertIn("puzzle", [event["protocol"] for event in play_events[2:]])
        self.assertTrue(
            all("rem_gate_open" in event["reason_codes"] for event in play_events),
        )
        self.assertIn("tlr_block_played", [criterion.name for criterion in summary.criteria])
        self.assertEqual(summary_payload["tlr_cue_play_count"], 2)
        self.assertTrue(summary_payload["tlr_block_required"])
        self.assertEqual(len(backend.requests), summary.cue_play_count)

    async def test_emergency_stop_file_blocks_audio_backend_requests(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        backend = MockAudioBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            emergency_stop_path = output_dir / "STOP_AUDIO"
            emergency_stop_path.touch()
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(4)),
                config=pilot4_config(output_dir, emergency_stop_path=emergency_stop_path),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=backend,
            )

        self.assertTrue(summary.passed)
        self.assertTrue(summary.emergency_stop_triggered)
        self.assertEqual(backend.stop_calls, 1)
        self.assertEqual(backend.requests, [])
        self.assertEqual(summary.audio_status_counts.get("blocked"), 2)

    async def test_awakening_events_append_jsonl_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "awakening_events.jsonl"
            append_awakening_event(
                output_path,
                AwakeningEvent(
                    event_type="awakening",
                    timestamp_utc="2026-05-13T22:00:00+00:00",
                    notes="woke briefly",
                ),
            )
            event = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(event["event_type"], "awakening")
        self.assertEqual(event["notes"], "woke briefly")
        # Without the two fields of #227 a marker is written as before.
        self.assertEqual(set(event), {"event_type", "timestamp_utc", "notes"})

    @unittest.skipIf(COARSE_CLOCK, COARSE_CLOCK_REASON)
    async def test_deadline_timeout_reports_duration_complete(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await run_pilot4_cueing_night(
                HangingMuseSource(),
                config=pilot4_config(
                    Path(tmpdir),
                    duration_seconds=0.01,
                    no_data_timeout_seconds=10.0,
                ),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
            )
            recording_events = [
                json.loads(line)
                for line in Path(summary.recording_events_path).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary.stop_reason, "duration_complete")
        self.assertEqual(recording_events[-1]["reason"], "duration_complete")

    async def test_no_data_timeout_before_deadline_is_preserved(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await run_pilot4_cueing_night(
                HangingMuseSource(),
                config=pilot4_config(
                    Path(tmpdir),
                    duration_seconds=10.0,
                    no_data_timeout_seconds=0.01,
                ),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
            )

        self.assertEqual(summary.stop_reason, "no_data_timeout")

    async def test_guard_follows_gate_on_eeg_only_preset(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        frames = [build_rem_frames(index, with_optics=False)[0] for index in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pilot4_config(Path(tmpdir))
            config = replace(config, gate_config=replace(config.gate_config, optics_capable=False))
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(frames),
                config=config,
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
            )
            guard_events = [
                json.loads(line)
                for line in Path(summary.arousal_guard_events_path).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([event["action"] for event in guard_events], ["allow"] * 4)
        self.assertTrue(all(event["metadata"]["ppg_absent_by_design"] for event in guard_events))
        self.assertGreaterEqual(summary.cue_play_count, 1)
        self.assertEqual(summary.uncued_puzzle_play_count, 0)

    async def test_real_audio_without_afplay_refuses_before_connect_and_any_file(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        for name in ("system", "afplay"):
            with self.subTest(backend=name), tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / "night"
                source = FakeMuseSource(rem_frames(1))
                source.connect = mock.AsyncMock(side_effect=AssertionError("connect() must not run"))
                with mock.patch("muse_tmr.audio.audio_player.shutil.which", return_value=None):
                    with self.assertRaisesRegex(AudioBackendFallbackError, "--allow-dry-run"):
                        await run_pilot4_cueing_night(
                            source,
                            config=replace(pilot4_config(output_dir), audio_backend_name=name),
                            catalog=catalog,
                            session=session,
                            assignment=assignment,
                            cue_library=cue_library,
                            calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                        )
                source.connect.assert_not_awaited()
                self.assertFalse(output_dir.exists())

    async def test_allowed_fallback_runs_silently_and_records_both_backend_names(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                pilot4_config(Path(tmpdir)),
                audio_backend_name="system",
                allow_dry_run_fallback=True,
            )
            with mock.patch("muse_tmr.audio.audio_player.shutil.which", return_value=None):
                summary = await run_pilot4_cueing_night(
                    FakeMuseSource(rem_frames(4)),
                    config=config,
                    catalog=catalog,
                    session=session,
                    assignment=assignment,
                    cue_library=cue_library,
                    calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                )
            audio_metadata = json.loads(Path(summary.metadata_path).read_text(encoding="utf-8"))["audio"]
            summary_payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))
            audio_events = [
                json.loads(line)
                for line in Path(summary.audio_log_path).read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(summary.passed)
        self.assertEqual(summary.cue_play_count, 2)
        self.assertEqual(audio_metadata["requested_backend_name"], "system")
        self.assertEqual(audio_metadata["resolved_backend_name"], "dry-run")
        self.assertEqual(audio_metadata["backend_name"], "system")
        self.assertEqual(summary_payload["requested_audio_backend_name"], "system")
        self.assertEqual(summary_payload["resolved_audio_backend_name"], "dry-run")
        self.assertEqual(summary_payload["audio_backend_name"], "dry-run")
        self.assertEqual([event["backend_name"] for event in audio_events], ["dry-run", "dry-run"])


# A generated p1034 night, synthetic input, not sleep: 4 min of N2, then 7 min of REM,
# without arousals or BLE gaps. From 240 s to 540 s (epochs 8 to 17) its optics carry
# noise at three times the pulse, a still stretch the heart-rate series cannot read.
NOISY_NIGHT_STAGES = ("N2",) * 8 + ("R",) * 14
NOISY_OPTICS = ((240.0, 300.0, 3.0),)
NOISY_EPOCHS = set(range(8, 18))


class TestBlindOpticsLowerTheVolume(unittest.TestCase):
    """Pilot 5 over a replay of that night (#234). The guard lowers the volume on the
    epochs its heart-rate rule is blind on, and the TLR and puzzle cues there play at
    half the volume they play at otherwise, through the scheduler, the player and the
    logs that every lower_volume reason goes through. With hr_rule_blind_action allow
    the same cues play at full volume. The calibration cap of 0.06 sits above the TLR
    hint (0.05) and below the hint of cue-p1 (0.10), and cue-p3 has no hint, so it plays
    at the default volume (0.02): the guard's multiplier comes after the cap (#255)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        night = generate_night(
            replace(
                SyntheticNightConfig(),
                preset="p1034",
                scenario="noisy_optics",
                hours=len(NOISY_NIGHT_STAGES) * 30 / 3600,
                stages=NOISY_NIGHT_STAGES,
                arousals_per_hour=0.0,
                ble_gaps_per_hour=0.0,
                noisy_optics=NOISY_OPTICS,
            ),
            root / "night",
        )
        cls.night_dir = night.recording_dir
        cls.lowered = asyncio.run(_replayed_pilot5(night.recording_dir, root / "default"))
        cls.allowed = asyncio.run(
            _replayed_pilot5(
                night.recording_dir,
                root / "allow",
                arousal_guard_config=ArousalGuardConfig(hr_rule_blind_action="allow"),
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_guard_lowers_the_volume_where_the_series_is_blind(self):
        guard = self.lowered["guard"]
        self.assertEqual(len(guard), len(NOISY_NIGHT_STAGES))
        blind = {index for index, event in enumerate(guard) if event["metadata"].get("hr_rule_blind")}
        self.assertEqual(blind, NOISY_EPOCHS)
        for index, event in enumerate(guard):
            if index in NOISY_EPOCHS:
                self.assertEqual(event["action"], "lower_volume", index)
                self.assertEqual(event["reason_codes"], ["hr_rule_blind"], index)
                self.assertEqual(event["volume_multiplier"], 0.5, index)
            else:
                self.assertEqual(event["action"], "allow", index)

    def test_cues_in_the_stretch_play_at_half_their_capped_volume(self):
        summary = self.lowered["summary"]
        plays = _plays(self.lowered["scheduler"])
        self.assertTrue(summary.passed)
        self.assertEqual(summary.tlr_cue_play_count, 2)
        self.assertEqual(summary.puzzle_cue_play_count, 2)
        self.assertEqual(summary.uncued_puzzle_play_count, 0)
        for play in plays:
            metadata = play["metadata"]
            self.assertIn(int(play["timestamp_seconds"] // 30), NOISY_EPOCHS)
            self.assertEqual(metadata["arousal_guard_action"], "lower_volume")
            self.assertEqual(metadata["arousal_guard_reason_codes"], ["hr_rule_blind"])
            self.assertNotIn("original_volume_hint", metadata)
        # requested, capped, multiplier and effective volume of each play
        self.assertEqual(
            [(play["cue_id"], _volumes(play["metadata"])) for play in plays],
            [
                ("tlr_soft_tone", (0.05, 0.05, 0.5, 0.025)),
                ("tlr_soft_tone", (0.05, 0.05, 0.5, 0.025)),
                ("cue-p1", (0.10, 0.06, 0.5, 0.03)),
                ("cue-p3", (0.02, 0.02, 0.5, 0.01)),
            ],
        )
        # The audio log says the same, and the volume_hint stays the cue's own.
        played = [event for event in self.lowered["audio"] if event["status"] == "played"]
        self.assertEqual([_volumes(event) for event in played], [_volumes(play["metadata"]) for play in plays])
        self.assertEqual([play["metadata"].get("volume_hint") for play in plays], [0.05, 0.05, 0.10, None])
        self.assertTrue(all("volume_lowered" in event["reason_codes"] for event in played))

    def test_lowered_cues_play_at_exactly_half_of_what_allow_plays(self):
        lowered, allowed = _plays(self.lowered["scheduler"]), _plays(self.allowed["scheduler"])
        self.assertEqual([_play_key(play) for play in allowed], [_play_key(play) for play in lowered])
        for before in allowed:
            self.assertNotIn("arousal_guard_action", before["metadata"])
            self.assertEqual(before["metadata"]["volume_multiplier"], 1.0)
        for index in NOISY_EPOCHS:
            event = self.allowed["guard"][index]
            self.assertEqual((event["action"], event["reason_codes"]), ("allow", []), index)
            self.assertIs(event["metadata"]["hr_rule_blind"], True)
        normal = [event for event in self.allowed["audio"] if event["status"] == "played"]
        quieter = [event for event in self.lowered["audio"] if event["status"] == "played"]
        self.assertEqual(len(normal), 4)
        self.assertEqual(
            [event["effective_volume"] for event in quieter],
            [0.5 * event["effective_volume"] for event in normal],
        )
        self.assertEqual([_volumes(event) for event in normal], [_volumes(play["metadata"]) for play in allowed])

    def test_the_scorer_reads_each_cues_volume_and_guard_action(self):
        # Pilot 5 replayed a synthetic night, so the scorer cut its windows from that
        # night's own raw file; the dose tables read audio_playback.jsonl (#256) and the
        # guard action from the play events.
        for logs, action, volumes in (
            (self.lowered, "lower_volume", [0.025, 0.025, 0.03, 0.01]),
            (self.allowed, "allow", [0.05, 0.05, 0.06, 0.02]),
        ):
            with self.subTest(action=action):
                block = logs["summary"].cue_arousal
                report = json.loads(Path(block["report_path"]).read_text(encoding="utf-8"))
                cues = [window for window in report["windows"] if window["kind"] == "cue"]
                self.assertEqual(block["status"], "scored")
                self.assertEqual(block["played_cue_count"], 4)
                self.assertEqual([cue["cue"]["effective_volume"] for cue in cues], volumes)
                self.assertEqual(
                    [cue["cue"]["effective_volume"] for cue in cues],
                    [event["effective_volume"] for event in logs["audio"] if event["status"] == "played"],
                )
                self.assertEqual({cue["cue"]["guard_action"] for cue in cues}, {action})
                self.assertEqual(
                    {cue["cue"]["guard_volume_multiplier"] for cue in cues},
                    {0.5 if action == "lower_volume" else 1.0},
                )
                self.assertEqual({cue["cue"]["volume_source"] for cue in cues}, {"audio_playback.jsonl"})
                self.assertEqual(
                    [row["bin"] for row in report["dose_tables"]["cue_index"]],
                    ["tlr 1", "tlr 2", "puzzle 1", "puzzle 2"],
                )
                self.assertEqual(sum(row["cues"] for row in report["dose_tables"]["effective_volume"]), 4)
                for code in ("synthetic_input", "no_audio_played", "thresholds_unfitted", "small_n"):
                    self.assertIn(code, block["limitations"])
                self.assertIs(report["synthetic"], True)
                # Every cue sits in a gate-open run; the guard decisions are the same in both.
                self.assertTrue(all(cue["rem_period_index"] is not None for cue in cues))
                # The optics are noise from 240 to 540 s, where the series uses no value:
                # no window has a heart-rate flag, and none is read from unused values.
                self.assertEqual(len(report["windows"]), 8)
                for window in report["windows"]:
                    self.assertIsNone(window["flags"]["hr"], (window["kind"], window["index"]))
                    self.assertIn("hr_unavailable", window["quality_flags"])
                    self.assertGreater(window["values"]["hr_values_skipped"], 0)
                self.assertIn("hr_unavailable", block["limitations"])

    def test_a_schema_1_summary_still_scores(self):
        # A summary from before #227 has no first_epoch_start: the scorer takes the first
        # frame, where the first epoch of a whole replay starts, and says so.
        from muse_tmr.reports.cue_arousal import score_cue_arousal

        summary_path = Path(self.lowered["summary"].summary_path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        old = {
            key: value
            for key, value in payload.items()
            if key not in ("first_epoch_start", "rem_gate_decisions_path", "cue_arousal")
        }
        old["schema_version"] = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = Path(tmpdir) / "pilot5_summary.json"
            old_path.write_text(json.dumps(old), encoding="utf-8")
            report = asyncio.run(
                score_cue_arousal(summary_path.parent, summary_path=old_path, replay_input=self.night_dir)
            )
        logged = json.loads(Path(payload["cue_arousal"]["report_path"]).read_text(encoding="utf-8"))
        self.assertIn("time_base_inferred", report.limitation_codes)
        self.assertEqual(logged["metadata"]["time_base"], "logged")
        self.assertEqual(report.metadata["time_base"], "inferred_from_first_frame")
        self.assertEqual(
            [round(window.time, 3) for window in report.windows],
            [round(window["time"], 3) for window in logged["windows"]],
        )

    def test_allow_logs_what_it_logged_before(self):
        # The volume fields of the allow plays in audio_playback.jsonl as main wrote them
        # for this replay before #255 (602386c): the TLR hint under the cap, cue-p1's
        # hint above it, the default volume of cue-p3.
        normal = [event for event in self.allowed["audio"] if event["status"] == "played"]
        old_fields = ("cue_id", "requested_volume", "effective_volume", "max_volume", "volume_capped", "reason_codes")
        before = (
            ("tlr_soft_tone", 0.05, 0.05, 0.06, False, ["mock_playback", "device_selected"]),
            ("tlr_soft_tone", 0.05, 0.05, 0.06, False, ["mock_playback", "device_selected"]),
            ("cue-p1", 0.1, 0.06, 0.06, True, ["volume_capped", "mock_playback", "device_selected"]),
            ("cue-p3", 0.02, 0.02, 0.06, False, ["mock_playback", "device_selected"]),
        )
        self.assertEqual(
            [json.dumps({key: event[key] for key in old_fields}, sort_keys=True) for event in normal],
            [json.dumps(dict(zip(old_fields, values)), sort_keys=True) for values in before],
        )


# Any relative alpha lowers the volume and none pauses, so every epoch is lower_volume.
ALPHA_LOWERS_EVERY_EPOCH = ArousalGuardConfig(
    alpha_lower_volume_relative_power=0.0,
    alpha_pause_relative_power=1.0,
)


class TestLowerVolumeAfterTheCaps(unittest.IsolatedAsyncioTestCase):
    """Pilot 4 nights where the guard lowers the volume of every epoch, against the same
    nights under allow (#255). Before, the scheduler halved the hint and the player capped
    it afterwards; now a lowered cue plays at exactly half of what it plays at under
    allow, after the calibration and hard caps, with or without a volume hint."""

    async def test_lowered_cues_play_at_half_the_allowed_volume_after_every_cap(self):
        cases = (
            # hint of both cued cues, comfortable volume, --hard-max-volume, allowed, lowered
            (0.10, 0.06, 0.20, 0.06, 0.03),  # hint above the calibration cap: 0.05 before
            (0.05, 0.10, 0.20, 0.05, 0.025),  # hint below it: 0.025 before too
            (None, 0.06, 0.20, 0.02, 0.01),  # no hint, --default-volume 0.02: not lowered before
            (0.10, 0.50, 0.04, 0.04, 0.02),  # the hard cap below the calibration: 0.04 before
        )
        for volume_hint, comfortable, hard_max, allowed, lowered in cases:
            with self.subTest(volume_hint=volume_hint, comfortable=comfortable, hard_max=hard_max):
                normal = await self._night(volume_hint, comfortable, hard_max, arousal_guard_config=None)
                quieter = await self._night(
                    volume_hint, comfortable, hard_max, arousal_guard_config=ALPHA_LOWERS_EVERY_EPOCH
                )

                for logs, action in ((normal, "allow"), (quieter, "lower_volume")):
                    plays = _plays(logs["scheduler"])
                    played = [event for event in logs["audio"] if event["status"] == "played"]
                    self.assertTrue(logs["summary"].passed)
                    self.assertEqual([play["puzzle_id"] for play in plays], ["p1", "p3"])
                    for play in plays:
                        guard = logs["guard"][int(play["timestamp_seconds"] // 30)]
                        self.assertEqual(guard["action"], action)
                    # The scheduler logs what the player then plays.
                    self.assertEqual(
                        [_volumes(event) for event in played], [_volumes(play["metadata"]) for play in plays]
                    )
                self.assertEqual(
                    [_play_key(play) for play in _plays(quieter["scheduler"])],
                    [_play_key(play) for play in _plays(normal["scheduler"])],
                )
                normal_played = [event for event in normal["audio"] if event["status"] == "played"]
                quieter_played = [event for event in quieter["audio"] if event["status"] == "played"]
                self.assertEqual([event["effective_volume"] for event in normal_played], [allowed] * 2)
                self.assertEqual([event["effective_volume"] for event in quieter_played], [lowered] * 2)
                self.assertEqual(
                    [event["effective_volume"] for event in quieter_played],
                    [0.5 * event["effective_volume"] for event in normal_played],
                )
                self.assertEqual(
                    [event["volume_capped"] for event in quieter_played],
                    [event["volume_capped"] for event in normal_played],
                )
                self.assertLessEqual(quieter["summary"].max_effective_volume, min(comfortable, hard_max))

    async def _night(self, volume_hint, comfortable, hard_max, *, arousal_guard_config):
        catalog, session, assignment, cue_library = protocol_fixture()
        cue_library = _with_volume_hint(cue_library, volume_hint, cue_ids=("cue-p1", "cue-p3"))
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(4)),
                config=replace(
                    # The end of the frames ends the night, not the clock.
                    pilot4_config(Path(tmpdir), duration_seconds=60.0),
                    hard_max_volume=hard_max,
                    arousal_guard_config=arousal_guard_config,
                ),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, comfortable),
                backend=MockAudioBackend(),
            )
            return _night_logs(summary)


async def _replayed_pilot5(recording_dir, output_dir, *, arousal_guard_config=None):
    """Pilot 5 with mock audio over a replay of ``recording_dir``, the p1034 gate and the
    guard default unless ``arousal_guard_config`` is given; the summary and its logs."""
    catalog, session, assignment, cue_library = protocol_fixture(include_tlr=True)
    cue_library = _with_volume_hint(cue_library, None, cue_ids=("cue-p3",))
    tlr_plan = plan_tlr_block(
        default_tlr_cue(),
        config=TlrBlockConfig(repetitions=2, interval_seconds=0.0, post_block_pause_seconds=0.0),
    )
    summary = await run_pilot4_cueing_night(
        ReplaySession(ReplayConfig(input_path=recording_dir, speed=0.0)),
        config=Pilot4CueingConfig(
            output_dir=output_dir,
            duration_seconds=600.0,
            allow_short=True,
            audio_backend_name="mock",
            gate_config=RemGateConfig(),
            scheduler_config=TmrSchedulerConfig(enable_tlr_block=True),
            arousal_guard_config=arousal_guard_config,
            pilot_id="m8_pilot5_full_night_tlr_puzzle_cueing",
            summary_filename="pilot5_summary.json",
            require_tlr_block=True,
        ),
        catalog=catalog,
        session=session,
        assignment=assignment,
        cue_library=cue_library,
        # Between the TLR hint (0.05) and cue-p1's (0.10).
        calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.06),
        backend=MockAudioBackend(),
        tlr_block_plan=tlr_plan,
    )
    return _night_logs(summary)


class TestOneProtocolHash(unittest.TestCase):
    """metadata.json, the summary, the Pilot 3 report and the analysis carry one
    protocol.content_hash: the preset's (#231)."""

    def test_the_night_its_replay_and_its_analysis_name_one_preset(self):
        from tests.test_pilot3_replay_simulation import write_eeg_only_recording

        preset = get_protocol_preset("konkoly2026_puzzle")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recording"
            write_eeg_only_recording(recording, ["nrem"] * 2 + ["rem"] * 4)
            catalog, session, assignment, cue_library = protocol_fixture()
            cue_library = replace(
                cue_library,
                cues=cue_library.cues + tuple(
                    CueMetadata(f"cue-{puzzle}", "generated_tone", 0.01, protocol="puzzle", frequency_hz=550.0)
                    for puzzle in ("p2", "p4")
                ),
            )
            paths = {
                "catalog": catalog.save(root / "catalog.json"),
                "session": session.save(root / "session.json"),
                "cues": cue_library.save(root / "cues.json"),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main.main([
                    "assign-puzzle-cues", str(paths["session"]), "--output", str(root / "assignment.json"),
                    "--seed", "3", "--protocol-preset", preset.name,
                ]), 0)
                self.assertEqual(cli_main.main([
                    "calibrate-volume", "--device-name", "Speaker", "--output", str(root / "calibration.json"),
                    "--detectable-volume", "0.01", "--identifiable-volume", "0.02", "--comfortable-volume", "0.04",
                ]), 0)
                inputs = [
                    "--catalog", str(paths["catalog"]), "--session", str(paths["session"]),
                    "--assignment", str(root / "assignment.json"), "--cue-library", str(paths["cues"]),
                    "--preset", "p21", "--protocol-preset", preset.name,
                ]
                self.assertEqual(cli_main.main([
                    "simulate-replay-cues", str(recording), *inputs, "--output", str(root / "pilot3.json"),
                ]), 0)
                args = cli_main.build_parser().parse_args([
                    "run-pilot4-cueing", "--output-dir", str(root / "night"), *inputs,
                    "--calibration", str(root / "calibration.json"), "--backend", "mock",
                    "--duration-seconds", "120", "--allow-short",
                ])
                replay = ReplaySession(ReplayConfig(input_path=recording, speed=0.0))
                with mock.patch.object(cli_main, "_build_source", return_value=replay):
                    asyncio.run(cli_main._run_pilot4_cueing(args))
            assignment_loaded = PuzzleCueAssignment.load(root / "assignment.json")
            build_morning_retest(
                session,
                tuple(
                    MorningRetestResult(puzzle, "", False, 30.0, 0.5) for puzzle in session.puzzle_ids
                ),
                catalog=catalog,
                assignment=assignment_loaded,
            ).save(root / "retest.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main.main([
                    "analyze-cued-uncued", str(paths["session"]), "--assignment", str(root / "assignment.json"),
                    "--retest", str(root / "retest.json"), "--output", str(root / "analysis.json"),
                    "--protocol-metadata", str(root / "night" / "metadata.json"),
                ]), 0)
            hashes = {
                "metadata.json": json.loads((root / "night" / "metadata.json").read_text(encoding="utf-8"))["protocol"]["content_hash"],
                "summary": json.loads((root / "night" / "pilot4_summary.json").read_text(encoding="utf-8"))["protocol"]["content_hash"],
                "pilot3": json.loads((root / "pilot3.json").read_text(encoding="utf-8"))["protocol"]["content_hash"],
                "analysis": json.loads((root / "analysis.json").read_text(encoding="utf-8"))["metadata"]["protocol"]["content_hash"],
                "assignment": assignment_loaded.metadata["protocol"]["content_hash"],
            }
            night_block = json.loads((root / "night" / "metadata.json").read_text(encoding="utf-8"))["protocol"]

        self.assertEqual(set(hashes.values()), {preset.content_hash()})
        self.assertEqual(night_block["inputs"]["assignment"]["name"], preset.name)
        self.assertEqual(night_block["settings"]["cued_puzzle_count"], 2)

    def test_a_pilot5_night_keeps_its_tlr_plan_for_the_fidelity_report(self):
        from muse_tmr.audio import load_cue_library
        from muse_tmr.reports.protocol_fidelity import build_protocol_fidelity
        from tests.test_pilot3_replay_simulation import write_eeg_only_recording

        preset = get_protocol_preset("konkoly2021_northwestern")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recording"
            write_eeg_only_recording(recording, ["nrem"] * 2 + ["rem"] * 6)
            catalog, session, assignment, cue_library = protocol_fixture()
            paths = {
                "catalog": catalog.save(root / "catalog.json"),
                "session": session.save(root / "session.json"),
                "assignment": assignment.save(root / "assignment.json"),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main.main([
                    "create-tlr-cue", "--output", str(root / "tlr_cue.json"), "--protocol-preset", preset.name,
                ]), 0)
                self.assertEqual(cli_main.main([
                    "plan-tlr-block", str(root / "tlr_cue.json"), "--output", str(root / "tlr_plan.json"),
                    "--protocol-preset", preset.name,
                ]), 0)
                self.assertEqual(cli_main.main([
                    "calibrate-volume", "--device-name", "Speaker", "--output", str(root / "calibration.json"),
                    "--detectable-volume", "0.01", "--identifiable-volume", "0.02", "--comfortable-volume", "0.04",
                ]), 0)
            tlr_cues = load_cue_library(root / "tlr_cue.json").cues
            replace(cue_library, cues=cue_library.cues + tlr_cues).save(root / "cues.json")
            args = cli_main.build_parser().parse_args([
                "run-pilot5-full-night", "--output-dir", str(root / "night"),
                "--catalog", str(paths["catalog"]), "--session", str(paths["session"]),
                "--assignment", str(paths["assignment"]), "--cue-library", str(root / "cues.json"),
                "--tlr-block", str(root / "tlr_plan.json"), "--calibration", str(root / "calibration.json"),
                "--backend", "mock", "--preset", "p21", "--protocol-preset", preset.name,
                "--duration-seconds", "300", "--allow-short",
            ])
            replay = ReplaySession(ReplayConfig(input_path=recording, speed=0.0))
            with mock.patch.object(cli_main, "_build_source", return_value=replay), \
                    contextlib.redirect_stdout(io.StringIO()):
                asyncio.run(cli_main._run_pilot5_full_night(args))
            plan = json.loads((root / "tlr_plan.json").read_text(encoding="utf-8"))
            metadata = json.loads((root / "night" / "metadata.json").read_text(encoding="utf-8"))
            tlr_block = build_protocol_fidelity(root / "night").checks["tlr_block"]

        # The night keeps the plan as it was made, its preset block included, and names it.
        self.assertEqual(metadata["tlr_block_plan"], plan)
        self.assertEqual(metadata["tlr_block_plan"]["metadata"]["protocol"]["name"], preset.name)
        self.assertEqual(metadata["protocol"]["inputs"]["tlr_block_plan"]["content_hash"], preset.content_hash())
        # The fidelity report compares the TLR cues with that plan.
        self.assertEqual(tlr_block["plan"], {"repetitions": 10, "interval_seconds": 30.0})
        self.assertGreaterEqual(tlr_block["observed"]["cues"], 1)
        self.assertIn(tlr_block["status"], ("as_planned", "differs"))


ARMS = ("cue", "sham", "no_cue")
SHAM_CUE = CueMetadata(
    "sham_soft_tone", "generated_tone", 1.0, protocol="sham", frequency_hz=330.0, volume_hint=0.05,
    tags=("sham", "untrained"),
)


def _arm_context(arm):
    return NightArmContext(arm=arm, night_id="night-007", sham_cue_id="sham_soft_tone")


class TestPilot5Arms(unittest.IsolatedAsyncioTestCase):
    """A Pilot 5 night under cue, sham and no_cue (#228), mock audio, no speakers."""

    async def _night(self, tmpdir, arm, *, tlr_cue=None):
        catalog, session, assignment, cue_library = protocol_fixture(include_tlr=True)
        cue_library = replace(cue_library, cues=cue_library.cues + (SHAM_CUE,))
        tlr_plan = plan_tlr_block(
            tlr_cue or default_tlr_cue(),
            config=TlrBlockConfig(repetitions=2, interval_seconds=0.0, post_block_pause_seconds=0.0),
        )
        backend = MockAudioBackend()
        config = pilot4_config(
            Path(tmpdir) / (arm or "none"),
            duration_seconds=60.0,
            enable_tlr_block=True,
            pilot_id="m8_pilot5_full_night_tlr_puzzle_cueing",
            summary_filename="pilot5_summary.json",
            require_tlr_block=True,
        )
        if arm is not None:
            config = replace(config, night_arm=_arm_context(arm))
        summary = await run_pilot4_cueing_night(
            FakeMuseSource(rem_frames(5)),
            config=config,
            catalog=catalog,
            session=session,
            assignment=assignment,
            cue_library=cue_library,
            calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
            backend=backend,
            tlr_block_plan=tlr_plan,
        )
        logs = _night_logs(summary)
        logs["requests"] = backend.requests
        logs["metadata"] = json.loads((Path(summary.output_dir) / "metadata.json").read_text(encoding="utf-8"))
        logs["payload"] = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))
        return logs

    async def test_the_arms_play_alike_but_for_the_sound(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nights = {arm: await self._night(tmpdir, arm) for arm in (None,) + ARMS}

        cue, sham, quiet = nights["cue"], nights["sham"], nights["no_cue"]
        plays = _plays(cue["scheduler"])
        self.assertEqual([play["protocol"] for play in plays][:2], ["tlr", "tlr"])
        self.assertIn("puzzle", [play["protocol"] for play in plays])
        # --arm cue is the night without an arm.
        self.assertEqual(cue["scheduler"], nights[None]["scheduler"])
        self.assertEqual(
            [{k: v for k, v in row.items() if k not in ("started_at", "ended_at")} for row in cue["audio"]],
            [{k: v for k, v in row.items() if k not in ("started_at", "ended_at")} for row in nights[None]["audio"]],
        )
        # sham: the same times and volumes; another cue_id, the arm, the sham_arm reason.
        self.assertEqual(len(sham["scheduler"]), len(cue["scheduler"]))
        for original, stand_in in zip(cue["scheduler"], sham["scheduler"]):
            self.assertEqual(stand_in.pop("arm"), "sham")
            if original["event_type"] == "play":
                self.assertEqual(stand_in.pop("cue_id"), "sham_soft_tone")
                self.assertEqual(stand_in["reason_codes"].pop(), "sham_arm")
                original = {key: value for key, value in original.items() if key != "cue_id"}
            self.assertEqual(stand_in, original)
        self.assertEqual(len(sham["audio"]), len(cue["audio"]))
        for original, stand_in in zip(cue["audio"], sham["audio"]):
            self.assertEqual((stand_in["arm"], stand_in["cue_id"], stand_in["status"]), ("sham", "sham_soft_tone", "played"))
            self.assertNotIn("arm", original)
            self.assertEqual(_volumes(stand_in), _volumes(original))
            self.assertEqual(stand_in["max_volume"], original["max_volume"])
        self.assertEqual({request.frequency_hz for request in sham["requests"]}, {330.0})
        # Each sham sound lasts as long as the cue it stands in for (1 s TLR, 0.01 s puzzle).
        self.assertEqual(
            [request.duration_seconds for request in sham["requests"]],
            [request.duration_seconds for request in cue["requests"]],
        )
        self.assertEqual(
            [request.effective_volume for request in sham["requests"]],
            [request.effective_volume for request in cue["requests"]],
        )
        # no_cue: nothing reaches the player; a virtual_play skip wherever the cue arm plays.
        self.assertEqual(_plays(quiet["scheduler"]), [])
        self.assertEqual(quiet["audio"], [])
        self.assertEqual(quiet["requests"], [])
        virtual = [event for event in quiet["scheduler"] if event["metadata"].get("virtual_play")]
        self.assertEqual(
            [(event["timestamp_seconds"], event["protocol"], event.get("puzzle_id")) for event in virtual],
            [(play["timestamp_seconds"], play["protocol"], play.get("puzzle_id")) for play in plays],
        )
        self.assertTrue(all(event["reason_codes"][-1] == "no_cue_arm" for event in virtual))

        for arm, night in nights.items():
            payload, criteria = night["payload"], {c["name"]: c for c in night["payload"]["criteria"]}
            self.assertTrue(payload["passed"], arm)
            self.assertEqual(payload["schema_version"], 4)
            self.assertEqual(payload["uncued_puzzle_play_count"], 0)
            self.assertTrue(criteria["no_uncued_puzzle_cues"]["passed"])
            self.assertTrue(criteria["tlr_block_played"]["passed"], arm)
            if arm is None:
                self.assertEqual((payload["arm"], payload["night_id"]), (None, None))
                self.assertNotIn("no_play_in_no_cue_arm", criteria)
                self.assertNotIn("night_arm", night["metadata"])
                continue
            self.assertEqual((payload["arm"], payload["night_id"]), (arm, "night-007"))
            self.assertIsNone(payload["schedule_commitment_sha256"])
            self.assertTrue(criteria["no_play_in_no_cue_arm"]["passed"])
            self.assertNotIn("arm_matches_sealed_schedule", criteria)
            self.assertEqual(night["metadata"]["night_arm"]["arm"], arm)
            self.assertEqual(night["metadata"]["night_arm"]["sham_cue_id"], "sham_soft_tone")
            self.assertEqual(night["metadata"]["protocol"], nights["cue"]["metadata"]["protocol"])
        self.assertEqual(quiet["payload"]["cue_play_count"], 0)
        self.assertEqual(quiet["payload"]["virtual_play_count"], len(plays))
        self.assertEqual(quiet["payload"]["max_effective_volume"], 0.0)
        self.assertEqual(
            {c["name"]: c["observed"] for c in quiet["payload"]["criteria"]}["no_play_in_no_cue_arm"],
            {"scheduler_plays": 0, "played_audio_rows": 0},
        )
        self.assertEqual(sham["payload"]["cue_play_count"], cue["payload"]["cue_play_count"])
        self.assertEqual(sham["payload"]["max_effective_volume"], cue["payload"]["max_effective_volume"])
        self.assertIn("sham_arm", sham["payload"]["cue_arousal"]["limitations"])
        self.assertIn("no_cue_arm", quiet["payload"]["cue_arousal"]["limitations"])
        self.assertNotIn("no_cues", quiet["payload"]["cue_arousal"]["limitations"])

    async def test_a_sham_sound_lasts_as_long_as_the_cue_the_cue_arm_plays(self):
        # A TLR plan made from another file's tlr_soft_tone says 0.65 s, while the cue arm
        # plays the night's library cue of that ID for its own 1 s.
        short = replace(default_tlr_cue(), duration_seconds=0.65)
        with tempfile.TemporaryDirectory() as tmpdir:
            nights = {arm: await self._night(tmpdir, arm, tlr_cue=short) for arm in ("cue", "sham")}

        cue, sham = nights["cue"]["requests"], nights["sham"]["requests"]
        self.assertEqual([request.duration_seconds for request in cue][:2], [1.0, 1.0])
        self.assertEqual(
            [request.duration_seconds for request in sham],
            [request.duration_seconds for request in cue],
        )
        self.assertEqual(
            [request.effective_volume for request in sham],
            [request.effective_volume for request in cue],
        )
        self.assertEqual({request.frequency_hz for request in sham}, {330.0})

    def test_only_the_cue_and_sham_arms_reach_the_player(self):
        catalog, session, assignment, cue_library = protocol_fixture(include_tlr=True)
        cue_library = replace(cue_library, cues=cue_library.cues + (SHAM_CUE,))
        backend = MockAudioBackend()
        player = AudioCuePlayer(AudioPlaybackConfig(max_volume=0.04), backend=backend)
        play = TmrSchedulerEvent("play", 0.0, cue_id="cue-p1", protocol="puzzle", puzzle_id="p1")
        for event, arm in (
            (replace(play, arm="no_cue"), "no_cue"),
            (play, "no_cue"),
            (play, "sham"),
            (replace(play, arm="sham", cue_id="sham_soft_tone"), "cue"),
        ):
            with self.assertRaises(ValueError):
                pilot4_module._play_scheduler_event(event, cue_library, player, arm=arm)
        sham_play = replace(
            play, arm="sham", cue_id="sham_soft_tone", metadata={"duration_seconds": 5.0, "volume_hint": 0.5}
        )
        with self.assertRaises(ValueError):
            pilot4_module._play_scheduler_event(sham_play, cue_library, player, arm="sham")
        self.assertEqual(backend.requests, [])
        # The length and hint of the cue it stands in for (cue-p1: 0.01 s, 0.10), as the
        # cue arm plays it, not the event's.
        result = pilot4_module._play_scheduler_event(
            sham_play, cue_library, player, arm="sham", stands_in_for="cue-p1"
        )
        self.assertEqual((result.arm, result.status, result.requested_volume, result.effective_volume), ("sham", "played", 0.10, 0.04))
        self.assertEqual((backend.requests[0].frequency_hz, backend.requests[0].duration_seconds), (330.0, 0.01))
        self.assertEqual(result.to_dict()["arm"], "sham")
        self.assertNotIn("arm", pilot4_module._play_scheduler_event(play, cue_library, player).to_dict())

    async def test_a_trained_sham_cue_stops_the_night_before_it_starts(self):
        catalog, session, assignment, cue_library = protocol_fixture(include_tlr=True)
        for sham_cue_id in ("cue-p1", "tlr_soft_tone"):
            for arm in ARMS:
                with self.subTest(sham_cue_id=sham_cue_id, arm=arm), tempfile.TemporaryDirectory() as tmpdir:
                    source = FakeMuseSource(rem_frames(2))
                    with self.assertRaises(ValueError):
                        await run_pilot4_cueing_night(
                            source,
                            config=replace(
                                pilot4_config(Path(tmpdir) / "night"),
                                night_arm=NightArmContext(arm=arm, sham_cue_id=sham_cue_id),
                            ),
                            catalog=catalog,
                            session=session,
                            assignment=assignment,
                            cue_library=cue_library,
                            calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                            backend=MockAudioBackend(),
                        )
                    self.assertFalse((Path(tmpdir) / "night").exists())
                    self.assertEqual(source.stop_calls, 0)

    def test_a_config_names_one_arm(self):
        config = pilot4_config(Path("unused"))
        with self.assertRaises(ValueError):
            replace(
                config,
                scheduler_config=replace(config.scheduler_config, arm="no_cue"),
                night_arm=_arm_context("sham"),
            ).validate()
        replace(config, scheduler_config=replace(config.scheduler_config, arm="no_cue")).validate()


class TestPilot5ArmsFromTheCommandLine(unittest.TestCase):
    """run-pilot5-full-night with --night-schedule on a replayed EEG-only night, then the
    fidelity report and the cue arousal scorer on each arm."""

    @classmethod
    def setUpClass(cls):
        from muse_tmr.audio import load_cue_library
        from tests.test_pilot3_replay_simulation import write_eeg_only_recording

        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.root = root = Path(cls._tmpdir.name)
        write_eeg_only_recording(root / "recording", ["nrem"] * 2 + ["rem"] * 6)
        catalog, session, assignment, cue_library = protocol_fixture()
        catalog.save(root / "catalog.json")
        session.save(root / "session.json")
        assignment.save(root / "assignment.json")
        with contextlib.redirect_stdout(io.StringIO()):
            assert cli_main.main(["create-tlr-cue", "--output", str(root / "tlr_cue.json")]) == 0
            assert cli_main.main(["plan-tlr-block", str(root / "tlr_cue.json"), "--output", str(root / "tlr_plan.json")]) == 0
            assert cli_main.main([
                "calibrate-volume", "--device-name", "Speaker", "--output", str(root / "calibration.json"),
                "--detectable-volume", "0.01", "--identifiable-volume", "0.02", "--comfortable-volume", "0.04",
            ]) == 0
            assert cli_main.main([
                "plan-night-schedule", "--blocks", "1", "--seed", "21", "--output", str(root / "series.json"),
            ]) == 0
        tlr_cues = load_cue_library(root / "tlr_cue.json").cues
        replace(cue_library, cues=cue_library.cues + tlr_cues + (SHAM_CUE,)).save(root / "cues.json")
        cls.schedule = load_night_schedule(root / "series.json")

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _run(self, name, *flags):
        root = self.root
        args = cli_main.build_parser().parse_args([
            "run-pilot5-full-night", "--output-dir", str(root / name),
            "--catalog", str(root / "catalog.json"), "--session", str(root / "session.json"),
            "--assignment", str(root / "assignment.json"), "--cue-library", str(root / "cues.json"),
            "--tlr-block", str(root / "tlr_plan.json"), "--calibration", str(root / "calibration.json"),
            "--backend", "mock", "--preset", "p21", "--duration-seconds", "300", "--allow-short", *flags,
        ])
        replay = ReplaySession(ReplayConfig(input_path=root / "recording", speed=0.0))
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cli_main, "_build_source", return_value=replay), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = asyncio.run(cli_main._run_pilot5_full_night(args))
        return code, out.getvalue() + err.getvalue()

    def test_each_night_runs_its_sealed_arm_and_never_prints_it(self):
        from muse_tmr.reports.protocol_fidelity import build_protocol_fidelity

        printed = {}
        nights = {}
        for night in self.schedule.nights:
            code, text = self._run(night.night_id, "--night-schedule", str(self.root / "series.json"), "--night-id", night.night_id)
            self.assertEqual(code, 0, text)
            printed[night.arm] = text.replace(night.night_id, "NIGHT")
            nights[night.arm] = self.root / night.night_id
        self.assertEqual(set(printed), set(ARMS))
        self.assertEqual(len(set(printed.values())), 1, printed)
        tokens = set(printed["cue"].replace("=", " ").replace(",", " ").replace("(", " ").split())
        self.assertFalse(tokens & set(ARMS), printed["cue"])
        self.assertNotIn("sham", printed["cue"])
        for arm, night_dir in nights.items():
            summary = json.loads((night_dir / "pilot5_summary.json").read_text(encoding="utf-8"))
            metadata = json.loads((night_dir / "metadata.json").read_text(encoding="utf-8"))
            criteria = {c["name"]: c for c in summary["criteria"]}
            self.assertEqual(summary["arm"], arm)
            self.assertEqual(summary["schedule_commitment_sha256"], self.schedule.commitment_sha256())
            self.assertTrue(criteria["arm_matches_sealed_schedule"]["passed"])
            self.assertTrue(criteria["no_play_in_no_cue_arm"]["passed"])
            self.assertEqual(metadata["night_arm"]["arm"], arm)
            self.assertTrue(metadata["night_arm"]["sealed_schedule"])
            self.assertIsNone(metadata["night_arm"]["schedule_unsealed_at_utc"])
            # The fidelity report and the scorer run on every arm and say what the arm did.
            fidelity = build_protocol_fidelity(night_dir)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli_main.main(["score-cue-arousal", str(night_dir)]), 0)
            arousal = json.loads((night_dir / "cue_arousal_report.json").read_text(encoding="utf-8"))
            codes = [item["code"] for item in arousal["limitations"]]
            self.assertEqual(fidelity.summary["arm"], arm)
            if arm == "no_cue":
                self.assertEqual(sum(fidelity.summary["plays"].values()), 0)
                self.assertGreater(sum(fidelity.summary["virtual_plays"].values()), 0)
                self.assertIn("no_cue_arm", fidelity.limitations)
                self.assertIn("no_cue_arm", codes)
                self.assertIn("nothing played, by design", fidelity.to_markdown())
            else:
                self.assertGreater(sum(fidelity.summary["plays"].values()), 0)
                self.assertEqual(sum(fidelity.summary["virtual_plays"].values()), 0)
                self.assertEqual("sham_arm" in fidelity.limitations, arm == "sham")
                self.assertEqual("sham_arm" in codes, arm == "sham")
        self.assertEqual(
            {name: check["status"] for name, check in build_protocol_fidelity(nights["sham"]).checks.items()},
            {name: check["status"] for name, check in build_protocol_fidelity(nights["cue"]).checks.items()},
        )

    def test_without_an_arm_the_night_logs_what_the_cue_arm_logs(self):
        code_plain, plain_text = self._run("plain")
        code_cue, _ = self._run("open_cue", "--arm", "cue")
        self.assertEqual((code_plain, code_cue), (0, 0))
        self.assertIn("scheduler_play=", plain_text)
        self.assertEqual(
            (self.root / "plain" / "scheduler_events.jsonl").read_bytes(),
            (self.root / "open_cue" / "scheduler_events.jsonl").read_bytes(),
        )
        summary = json.loads((self.root / "plain" / "pilot5_summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["arm"], summary["virtual_play_count"]), (None, 0))

    def test_a_changed_schedule_stops_the_night_before_any_file(self):
        changed = self.root / "changed.json"
        payload = self.schedule.to_dict()
        payload["seed"] = payload["seed"] + 1
        changed.write_text(json.dumps(payload), encoding="utf-8")
        (self.root / "changed.sealed.json").write_text(
            json.dumps(self.schedule.sealed_companion()), encoding="utf-8"
        )
        code, text = self._run("never", "--night-schedule", str(changed), "--night-id", self.schedule.nights[0].night_id)
        self.assertEqual(code, 2)
        self.assertIn("does not match the commitment", text)
        self.assertFalse((self.root / "never").exists())


def _night_logs(summary):
    return {
        "summary": summary,
        "guard": _jsonl(summary.arousal_guard_events_path),
        "scheduler": _jsonl(summary.scheduler_events_path),
        "audio": _jsonl(summary.audio_log_path),
    }


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _plays(scheduler_events):
    return [event for event in scheduler_events if event["event_type"] == "play"]


def _volumes(payload):
    """requested, capped, multiplier and effective volume of a play event's metadata or
    an audio log line."""
    return tuple(
        payload[key] for key in ("requested_volume", "capped_volume", "volume_multiplier", "effective_volume")
    )


def _with_volume_hint(cue_library, volume_hint, *, cue_ids):
    return replace(
        cue_library,
        cues=tuple(
            replace(cue, volume_hint=volume_hint) if cue.cue_id in cue_ids else cue
            for cue in cue_library.cues
        ),
    )


def _play_key(play):
    return play["timestamp_seconds"], play["protocol"], play["cue_id"], play.get("puzzle_id")


def pilot4_config(
    output_dir: Path,
    *,
    emergency_stop_path=None,
    duration_seconds=1.0,
    no_data_timeout_seconds=30.0,
    enable_tlr_block=False,
    pilot_id="m8_pilot4_low_volume_rem_gated_cueing",
    summary_filename="pilot4_summary.json",
    require_tlr_block=False,
):
    return Pilot4CueingConfig(
        output_dir=output_dir,
        duration_seconds=duration_seconds,
        allow_short=True,
        no_data_timeout_seconds=no_data_timeout_seconds,
        audio_backend_name="mock",
        hard_max_volume=0.20,
        default_volume=0.02,
        emergency_stop_path=emergency_stop_path,
        epoch_config=EpochConfig(epoch_seconds=30, stride_seconds=30),
        gate_config=RemGateConfig(
            enter_threshold=0.60,
            exit_threshold=0.45,
            min_stable_seconds=60.0,
            epoch_seconds=30.0,
            cooldown_seconds=0.0,
        ),
        scheduler_config=TmrSchedulerConfig(
            puzzle_cue_interval_seconds=30.0,
            cooldown_seconds=30.0,
            max_puzzle_cues_per_block=4,
            enable_tlr_block=enable_tlr_block,
        ),
        pilot_id=pilot_id,
        summary_filename=summary_filename,
        require_tlr_block=require_tlr_block,
    )


def protocol_fixture(*, include_tlr=False):
    catalog = PuzzleCatalog(
        puzzles=(
            PuzzleTask("p1", "one", "one", cue_id="cue-p1"),
            PuzzleTask("p2", "two", "two", cue_id="cue-p2"),
            PuzzleTask("p3", "three", "three", cue_id="cue-p3"),
            PuzzleTask("p4", "four", "four", cue_id="cue-p4"),
        )
    )
    session = NightPuzzleSession(
        session_id="night-001",
        puzzle_ids=("p1", "p2", "p3", "p4"),
        puzzle_count=4,
    )
    assignment = PuzzleCueAssignment(
        session_id="night-001",
        cued_puzzle_ids=("p1", "p3"),
        uncued_puzzle_ids=("p2", "p4"),
        seed=17,
    )
    cues = (
        CueMetadata(
            "cue-p1",
            "generated_tone",
            0.01,
            protocol="puzzle",
            frequency_hz=440.0,
            volume_hint=0.10,
        ),
        CueMetadata(
            "cue-p3",
            "generated_tone",
            0.01,
            protocol="puzzle",
            frequency_hz=660.0,
            volume_hint=0.10,
        ),
    )
    if include_tlr:
        cues = cues + (default_tlr_cue(),)
    cue_library = CueLibrary(
        library_id="pilot4-test",
        cues=cues,
    )
    return catalog, session, assignment, cue_library


def rem_frames(count):
    frames = []
    for index in range(count):
        frames.extend(build_rem_frames(index))
    return frames


def build_rem_frames(index, *, with_optics=True):
    timestamp = 1000.0 + index * 30.0
    frames = [
        MuseFrame(
            timestamp=timestamp,
            eeg=EEGSample(timestamp, rem_like_eeg_channels(), source="test"),
            imu=IMUSample(
                timestamp,
                accelerometer_g=still_accel_rows(),
                gyroscope_dps=still_gyro_rows(),
                source="test",
            ),
            ppg=(
                PPGSample(timestamp, {"LO_NIR": synthetic_ppg(72.0)}, source="test")
                if with_optics
                else None
            ),
            source="test",
            raw_packet=b"pilot4",
        )
    ]
    for sample_index in range(30):
        frames.append(
            MuseFrame(
                timestamp=timestamp + sample_index,
                heart_rate=HeartRateSample(
                    timestamp + sample_index,
                    bpm=68.0 + (sample_index % 5),
                    source="test",
                ),
                source="test",
                raw_packet=b"pilot4-hr",
            )
        )
    return frames


def sine_values(frequency_hz, *, seconds=30.0, sample_rate_hz=EEG_RATE_HZ, amplitude=10.0):
    timestamps = np.arange(0, seconds, 1 / sample_rate_hz)
    return amplitude * np.sin(2 * np.pi * frequency_hz * timestamps)


def rem_like_eeg_channels():
    theta = sine_values(6.0, amplitude=12.0)
    alpha = sine_values(10.0, amplitude=1.0)
    eye = rem_deflection_train()
    return {
        "TP9": (theta + alpha).tolist(),
        "AF7": (theta + alpha + eye).tolist(),
        "AF8": (theta + alpha - eye).tolist(),
        "TP10": (theta + alpha).tolist(),
    }


def synthetic_ppg(heart_rate_bpm, *, seconds=30.0):
    timestamps = np.arange(0, seconds, 1 / PPG_RATE_HZ)
    heart_hz = heart_rate_bpm / 60.0
    pulse = np.sin(2 * np.pi * heart_hz * timestamps)
    harmonic = 0.25 * np.sin(4 * np.pi * heart_hz * timestamps - np.pi / 4)
    return (50000.0 + 1000.0 * (pulse + harmonic)).tolist()


def still_accel_rows():
    return [{"x": 0.0, "y": 0.0, "z": 1.0}] * (30 * IMU_RATE_HZ)


def still_gyro_rows():
    return [{"x": 0.0, "y": 0.0, "z": 0.0}] * (30 * IMU_RATE_HZ)


if __name__ == "__main__":
    unittest.main()
