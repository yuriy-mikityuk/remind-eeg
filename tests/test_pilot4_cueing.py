import asyncio
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from muse_tmr.audio import (
    AudioBackendFallbackError,
    CueLibrary,
    CueMetadata,
    MockAudioBackend,
    VolumeCalibration,
)
from muse_tmr.data.sample_types import EEGSample, HeartRateSample, IMUSample, MuseFrame, PPGSample
from muse_tmr.features import EpochConfig
from muse_tmr.models import RemGateConfig
from muse_tmr.protocol import (
    NightPuzzleSession,
    PuzzleCatalog,
    PuzzleCueAssignment,
    PuzzleTask,
    TlrBlockConfig,
    TmrSchedulerConfig,
    default_tlr_cue,
    plan_tlr_block,
)
from muse_tmr.sources.base_source import MuseDeviceInfo, MuseSourceMetadata
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
