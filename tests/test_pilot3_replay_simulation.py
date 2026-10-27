import asyncio
import contextlib
import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from muse_tmr.ble import muse_athena_protocol as proto
import numpy as np
from muse_tmr.ble.muse_raw_stream import MuseRawStream

from muse_tmr.audio import CueLibrary, CueMetadata
from muse_tmr.cli.main import main as cli_main
from muse_tmr.data.sample_types import EEGSample, HeartRateSample, IMUSample, MuseFrame, PPGSample
from muse_tmr.features import EpochBuilder, EpochConfig
from muse_tmr.models import RemGateConfig, RemPrediction
from muse_tmr.presets import preset_gate_thresholds
from muse_tmr.protocol import (
    ArousalGuardConfig,
    NightPuzzleSession,
    PuzzleCatalog,
    PuzzleCueAssignment,
    PuzzleTask,
    TmrSchedulerConfig,
    get_protocol_preset,
    load_night_schedule,
    unseal_night_schedule,
)
from muse_tmr.testing import generate_night
from muse_tmr.validation import pilot3 as pilot3_module
from muse_tmr.validation import simulate_cue_plan_from_epochs, simulate_replay_cue_plan
from tests.rem_event_synthetic import rem_deflection_train


EEG_RATE_HZ = 256
IMU_RATE_HZ = 52
PPG_RATE_HZ = 64


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


def build_rem_epoch(index, *, with_optics=True, stride_seconds=30.0):
    timestamp = 1000.0 + index * stride_seconds
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
        )
    ]
    for sample_index in range(30 if with_optics else 0):
        frames.append(
            MuseFrame(
                timestamp=timestamp + sample_index,
                heart_rate=HeartRateSample(
                    timestamp + sample_index,
                    bpm=68.0 + (sample_index % 5),
                    source="test",
                ),
                source="test",
            )
        )
    return EpochBuilder(EpochConfig(epoch_seconds=30, stride_seconds=30))._build_epoch(
        index,
        timestamp,
        tuple(frames),
    )


def protocol_fixture():
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
    cue_library = CueLibrary(
        library_id="pilot3-test",
        cues=(
            CueMetadata("cue-p1", "generated_tone", 1.0, protocol="puzzle", frequency_hz=440.0),
            CueMetadata("cue-p3", "generated_tone", 1.0, protocol="puzzle", frequency_hz=660.0),
        ),
    )
    return catalog, session, assignment, cue_library


def pack_eeg_4ch(rows_uv):
    """Pack 4 samples x 4 channels of microvolts into one 28-byte EEG subpacket."""
    counts = np.clip(
        np.round(np.asarray(rows_uv) / proto.EEG_SCALE_UV_PER_COUNT) + proto.EEG_MID_COUNT,
        0,
        proto.EEG_MAX_COUNT,
    ).astype(int)
    packed = 0
    for index, count in enumerate(counts.ravel()):
        packed |= int(count) << (proto.EEG_BITS * index)
    return packed.to_bytes(28, "little")


def still_accgyro_subpacket():
    rows = [[0, 0, round(1.0 / proto.ACC_SCALE), 0, 0, 0]] * 3
    return np.asarray(rows, dtype="<i2").tobytes()


def eeg_only_epoch_uv(kind):
    timestamps = np.arange(0, 30.0, 1 / EEG_RATE_HZ)
    if kind == "rem":
        background = sine_values(6.0, amplitude=12.0) + sine_values(10.0, amplitude=1.0)
        eye = sine_values(1.0, amplitude=8.0)
        return np.stack([background, background + eye, background - eye, background], axis=1)
    slow = 40.0 * np.sin(2 * np.pi * 1.5 * timestamps) + 4.0 * np.sin(2 * np.pi * 6.0 * timestamps)
    return np.stack([slow] * 4, axis=1)


def write_eeg_only_recording(recording_dir: Path, epoch_kinds) -> Path:
    """A p21-style recording: EEG and IMU notifications, no TAG_OPTICS_* packets."""
    recording_dir.mkdir(parents=True, exist_ok=True)
    raw_path = recording_dir / "raw_amused.bin"
    channels_uv = np.concatenate([eeg_only_epoch_uv(kind) for kind in epoch_kinds])
    stream = MuseRawStream(str(raw_path))
    stream.open_write()
    base_time = stream.session_start
    eeg_counter = 0
    for notification, start in enumerate(range(0, channels_uv.shape[0], 16)):
        header = bytearray(14)
        header[9] = proto.TAG_EEG_4CH
        header[10] = eeg_counter % 256
        payload = bytes(header) + pack_eeg_4ch(channels_uv[start:start + 4])
        eeg_counter += 1
        for offset in (4, 8, 12):
            payload += bytes([proto.TAG_EEG_4CH, eeg_counter % 256, 0, 0, 0])
            payload += pack_eeg_4ch(channels_uv[start + offset:start + offset + 4])
            eeg_counter += 1
        payload += bytes([proto.TAG_ACCGYRO, notification % 256, 0, 0, 0])
        payload += still_accgyro_subpacket()
        stream.write_packet(payload, base_time + dt.timedelta(seconds=start / EEG_RATE_HZ))
    stream.close()
    return raw_path


class TestPilot3ReplaySimulation(unittest.TestCase):
    def test_replay_cue_plan_uses_mock_audio_and_only_cued_puzzles(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        report = simulate_cue_plan_from_epochs(
            [build_rem_epoch(index) for index in range(4)],
            catalog=catalog,
            session=session,
            assignment=assignment,
            cue_library=cue_library,
            recording_input="synthetic-replay",
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
                enable_tlr_block=False,
            ),
        )

        self.assertTrue(report.passed)
        self.assertFalse(report.audio_playback_executed)
        self.assertEqual(report.audio_backend, "mock")
        self.assertEqual(report.metrics["uncued_puzzle_play_count"], 0)
        self.assertEqual([event.puzzle_id for event in report.cue_plan], ["p1", "p3"])
        self.assertGreaterEqual(report.metrics["scheduler_event_count"], 4)

    def test_report_and_scheduler_events_are_inspectable_files(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        report = simulate_cue_plan_from_epochs(
            [build_rem_epoch(0), build_rem_epoch(1)],
            catalog=catalog,
            session=session,
            assignment=assignment,
            cue_library=cue_library,
            gate_config=RemGateConfig(
                enter_threshold=0.60,
                exit_threshold=0.45,
                min_stable_seconds=60.0,
                epoch_seconds=30.0,
                cooldown_seconds=0.0,
            ),
            scheduler_config=TmrSchedulerConfig(enable_tlr_block=False),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = report.save(Path(tmpdir) / "pilot3_report.json")
            events_path = report.save_scheduler_events(Path(tmpdir) / "events.jsonl")

            saved = json.loads(report_path.read_text(encoding="utf-8"))
            event_lines = events_path.read_text(encoding="utf-8").splitlines()

        self.assertTrue(saved["passed"])
        self.assertEqual(saved["audio_backend"], "mock")
        self.assertFalse(saved["audio_playback_executed"])
        self.assertGreaterEqual(len(saved["scheduler_events"]), 2)
        self.assertGreaterEqual(len(event_lines), 2)
        # The time base of every timestamp_seconds, and no injected cue without the flag (#227).
        self.assertEqual(saved["metrics"]["first_epoch_start"], 1000.0)
        self.assertEqual(saved["injected_cues"], [])
        self.assertEqual(saved["metrics"]["injected_cue_count"], 0)
        self.assertIsNone(saved["metrics"]["inject_cues_every_seconds"])

    def test_one_rem_event_tracker_for_the_whole_replay(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        trackers = []
        extract = pilot3_module.extract_eeg_features

        def recording_extract(epoch, *args, **kwargs):
            trackers.append(kwargs.get("rem_tracker"))
            return extract(epoch, *args, **kwargs)

        with mock.patch.object(pilot3_module, "extract_eeg_features", recording_extract):
            report = simulate_cue_plan_from_epochs(
                [build_rem_epoch(index) for index in range(4)],
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
            )

        self.assertEqual(len(trackers), 4)
        self.assertIsNotNone(trackers[0])
        self.assertTrue(all(tracker is trackers[0] for tracker in trackers))
        counts = [result.prediction.feature_values["rem_event_count"] for result in report.epoch_results]
        self.assertEqual(counts, [8.0] * 4)

    def test_empty_replay_epochs_fail_without_calling_audio(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        report = simulate_cue_plan_from_epochs(
            [],
            catalog=catalog,
            session=session,
            assignment=assignment,
            cue_library=cue_library,
        )

        self.assertFalse(report.passed)
        self.assertIn("epochs_present", report.failed_criteria)
        self.assertFalse(report.audio_playback_executed)


class ScriptedDetector:
    """P_REM by epoch index, so a test decides what the gate sees."""

    def __init__(self, probabilities=None, *, default=0.95, arousal_indexes=()):
        self.probabilities = dict(probabilities or {})
        self.default = default
        self.arousal_indexes = frozenset(arousal_indexes)

    def predict_features(self, *, eeg=None, imu=None, ppg=None):
        index = eeg.epoch_index
        return RemPrediction(
            probability=self.probabilities.get(index, self.default),
            reason_codes=("motion_arousal_proxy",) if index in self.arousal_indexes else (),
            source="scripted",
        )


def scripted_run(stride_seconds, count, detector, *, pass_stride=True):
    """p21 gate (0.90/0.85, 60 s stable, 120 s cooldown) over REM epochs at a stride."""
    catalog, session, assignment, cue_library = protocol_fixture()
    return simulate_cue_plan_from_epochs(
        [
            build_rem_epoch(index, with_optics=False, stride_seconds=stride_seconds)
            for index in range(count)
        ],
        catalog=catalog,
        session=session,
        assignment=assignment,
        cue_library=cue_library,
        detector=detector,
        gate_config=RemGateConfig(enter_threshold=0.90, exit_threshold=0.85, optics_capable=False),
        scheduler_config=TmrSchedulerConfig(enable_tlr_block=False),
        stride_seconds=stride_seconds if pass_stride else None,
    )


def epoch_length_wiring(start_time, previous_start_time, stride_seconds):
    """What the pilots gave the gate before #225 1c: the 30 s epoch length every time."""
    return 30.0


class TestPilot3GateClock(unittest.TestCase):
    """The gate keeps real time: the stride between epoch starts, not the epoch length."""

    def test_stride_10_gate_needs_60_s_of_real_time(self):
        for pass_stride in (True, False):
            with self.subTest(pass_stride=pass_stride):
                report = scripted_run(10.0, 8, ScriptedDetector(), pass_stride=pass_stride)
                gate = [result.gate_decision for result in report.epoch_results]

                self.assertEqual(
                    [decision.stable_seconds for decision in gate[:6]],
                    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                )
                self.assertEqual([decision.gate_open for decision in gate[:6]], [False] * 5 + [True])
                self.assertEqual(report.epoch_results[5].timestamp_seconds, 50.0)
                self.assertEqual(report.metrics["stride_seconds"], 10.0)
                self.assertEqual(report.metrics["first_gate_open_seconds"], 80.0)
                self.assertEqual(report.metrics["first_play_seconds"], 80.0)

        with mock.patch.object(pilot3_module, "epoch_step_seconds", epoch_length_wiring):
            before = scripted_run(10.0, 8, ScriptedDetector())
        self.assertEqual(before.metrics["first_gate_open_seconds"], 40.0)

    def test_stride_10_cooldown_takes_120_s_of_real_time(self):
        report = scripted_run(10.0, 30, ScriptedDetector(arousal_indexes={8}))
        results = report.epoch_results
        cooling = [
            result.timestamp_seconds
            for result in results
            if "cooldown_active" in result.gate_decision.reason_codes
        ]

        self.assertIn("motion_arousal_block", results[8].gate_decision.reason_codes)
        self.assertEqual(results[8].gate_decision.cooldown_remaining_seconds, 120.0)
        self.assertEqual(cooling, [90.0 + 10.0 * step for step in range(11)])
        self.assertEqual(results[20].timestamp_seconds, 200.0)
        self.assertEqual(results[20].gate_decision.state, "warming")
        reopened = next(result for result in results[9:] if result.gate_decision.gate_open)
        self.assertEqual(reopened.timestamp_seconds, 250.0)

        with mock.patch.object(pilot3_module, "epoch_step_seconds", epoch_length_wiring):
            before = scripted_run(10.0, 30, ScriptedDetector(arousal_indexes={8}))
        before_cooling = [
            result.timestamp_seconds
            for result in before.epoch_results
            if "cooldown_active" in result.gate_decision.reason_codes
        ]
        self.assertEqual(before_cooling, [90.0, 100.0, 110.0])

    def test_stride_30_output_is_the_same_as_with_the_epoch_length(self):
        def run(stride_seconds):
            # REM, a drop to NREM, a motion arousal, a dip into the hysteresis band.
            detector = ScriptedDetector(
                {4: 0.30, 5: 0.30, 15: 0.86, 16: 0.80},
                arousal_indexes={9},
            )
            return scripted_run(stride_seconds, 24, detector)

        def outputs(report):
            return (
                [event.to_dict() for event in report.scheduler_events],
                [result.gate_decision.to_dict() for result in report.epoch_results],
                [result.arousal_guard_decision.to_dict() for result in report.epoch_results],
            )

        after = run(30.0)
        with mock.patch.object(pilot3_module, "epoch_step_seconds", epoch_length_wiring):
            before = run(30.0)
        self.assertEqual(outputs(after), outputs(before))
        event_types = {event.event_type for event in after.scheduler_events}
        self.assertTrue({"play", "pause", "skip"} <= event_types)
        self.assertTrue(
            any("cooldown_active" in r.gate_decision.reason_codes for r in after.epoch_results)
        )

        after_10 = run(10.0)
        with mock.patch.object(pilot3_module, "epoch_step_seconds", epoch_length_wiring):
            before_10 = run(10.0)
        self.assertNotEqual(outputs(after_10)[0], outputs(before_10)[0])

    def test_first_times_are_none_when_the_gate_never_opens(self):
        report = scripted_run(30.0, 4, ScriptedDetector(default=0.30))

        self.assertEqual(report.metrics["stride_seconds"], 30.0)
        self.assertIsNone(report.metrics["first_gate_open_seconds"])
        self.assertIsNone(report.metrics["first_play_seconds"])
        self.assertTrue(report.passed)

    def test_epoch_starts_must_increase(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        with self.assertRaises(ValueError):
            simulate_cue_plan_from_epochs(
                [build_rem_epoch(1), build_rem_epoch(0)],
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
            )


class TestPilot3ReplayPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_replay_recording_generates_mocked_plan_report(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            recording_dir = Path(tmpdir) / "recording"
            generate_night(output_dir=recording_dir, scenario="short_allow_short", hours=2 / 60)

            report = await simulate_replay_cue_plan(
                recording_dir,
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                scheduler_config=TmrSchedulerConfig(enable_tlr_block=False),
            )

        self.assertTrue(report.passed)
        self.assertEqual(report.metrics["epoch_count"], 4)
        self.assertGreater(report.metrics["scheduler_event_count"], 0)
        self.assertEqual(report.metrics["uncued_puzzle_play_count"], 0)
        self.assertFalse(report.audio_playback_executed)


class TestPilot3EegOnlyPreset(unittest.TestCase):
    """A p21 night has no optics by design; the guard must not pause it for that."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmpdir.name)
        write_eeg_only_recording(cls.root / "recording", ["nrem"] * 3 + ["rem"] * 5)
        catalog, session, assignment, cue_library = protocol_fixture()
        catalog.save(cls.root / "catalog.json")
        session.save(cls.root / "session.json")
        assignment.save(cls.root / "assignment.json")
        cue_library.save(cls.root / "cues.json")

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _simulate(self, preset, *extra_args):
        output = self.root / f"report_{preset}_{len(extra_args)}.json"
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main([
                "simulate-replay-cues", str(self.root / "recording"),
                "--catalog", str(self.root / "catalog.json"),
                "--session", str(self.root / "session.json"),
                "--assignment", str(self.root / "assignment.json"),
                "--cue-library", str(self.root / "cues.json"),
                "--output", str(output),
                "--preset", preset,
                *extra_args,
            ])
        self.assertEqual(exit_code, 0)
        return json.loads(output.read_text(encoding="utf-8"))

    def test_p21_replay_reports_stride_and_first_cue_times(self):
        # REM-like EEG from 90 s on. At a 30 s stride the gate opens on the second REM
        # epoch, decided at 150 s; at 10 s it needs six REM windows, the first ending at
        # 120 s, so it opens at 170 s.
        for stride, first_open in (("30", 150.0), ("10", 170.0)):
            with self.subTest(stride=stride):
                metrics = self._simulate("p21", "--stride-seconds", stride)["metrics"]
                self.assertEqual(metrics["stride_seconds"], float(stride))
                self.assertEqual(metrics["first_gate_open_seconds"], first_open)
                self.assertEqual(metrics["first_play_seconds"], first_open)
                self.assertEqual(metrics["uncued_puzzle_play_count"], 0)

    def test_recording_has_no_optics_packets(self):
        stream = MuseRawStream(str(self.root / "recording" / "raw_amused.bin"))
        stream.open_read()
        try:
            tags = set()
            for packet in stream.read_packets():
                tags.update(proto.inspect_payload(packet.data)["tags"])
        finally:
            stream.close()
        self.assertEqual(tags, {proto.TAG_EEG_4CH, proto.TAG_ACCGYRO})

    def test_p21_replay_plays_cues_without_repeated_artifact_pauses(self):
        report = self._simulate("p21")
        guard = [epoch["arousal_guard_decision"] for epoch in report["epochs"]]

        repeated_artifact_pauses = [
            decision for decision in guard
            if decision["action"] == "pause"
            and "repeated_artifact_quality" in decision["reason_codes"]
        ]
        self.assertEqual(repeated_artifact_pauses, [])
        self.assertTrue(all(decision["action"] == "allow" for decision in guard))
        self.assertTrue(all(decision["metadata"]["ppg_absent_by_design"] for decision in guard))
        self.assertGreaterEqual(report["metrics"]["cue_plan_count"], 1)
        self.assertEqual(report["metrics"]["uncued_puzzle_play_count"], 0)
        self.assertTrue(report["passed"])

    def test_optics_preset_still_pauses_when_optics_are_missing(self):
        report = self._simulate("p1034")
        actions = [epoch["arousal_guard_decision"]["action"] for epoch in report["epochs"]]

        self.assertEqual(actions, ["lower_volume"] * 2 + ["pause"] * 6)
        for epoch in report["epochs"][2:]:
            self.assertIn(
                "repeated_artifact_quality", epoch["arousal_guard_decision"]["reason_codes"]
            )
        self.assertEqual(report["metrics"]["uncued_puzzle_play_count"], 0)

    def _events_bytes(self, *extra_args):
        events = self.root / f"events_{len(extra_args)}_{abs(hash(extra_args))}.jsonl"
        self._simulate("p21", "--scheduler-events-output", str(events), *extra_args)
        return events.read_bytes()

    def test_without_a_preset_the_scheduler_events_are_mains(self):
        """main's simulate-replay-cues built these configs from its flag defaults."""
        catalog, session, assignment, cue_library = protocol_fixture()
        enter, exit_ = preset_gate_thresholds("p21")
        report = asyncio.run(
            simulate_replay_cue_plan(
                self.root / "recording",
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                epoch_config=EpochConfig(epoch_seconds=30.0, stride_seconds=30.0),
                gate_config=RemGateConfig(
                    enter_threshold=enter,
                    exit_threshold=exit_,
                    min_stable_seconds=60.0,
                    epoch_seconds=30.0,
                    cooldown_seconds=120.0,
                    optics_capable=False,
                ),
                scheduler_config=TmrSchedulerConfig(
                    puzzle_cue_interval_seconds=30.0,
                    cooldown_seconds=120.0,
                    max_puzzle_cues_per_block=4,
                    enable_tlr_block=False,
                ),
                arousal_guard_config=ArousalGuardConfig(enabled=True, optics_capable=False),
            )
        )
        expected = report.save_scheduler_events(self.root / "main_events.jsonl").read_bytes()
        self.assertGreaterEqual(len(report.cue_plan), 1)
        self.assertEqual(self._events_bytes(), expected)
        self.assertEqual(self._events_bytes("--protocol-preset", "remind_default_v1"), expected)

    def test_the_report_records_the_preset(self):
        report = self._simulate("p21")
        block = report["protocol"]
        default = get_protocol_preset("remind_default_v1")
        self.assertEqual((block["name"], block["content_hash"]), ("remind_default_v1", default.content_hash()))
        self.assertEqual(block["overrides"], {})
        flagged = self._simulate("p21", "--min-stable-seconds", "90")["protocol"]
        self.assertEqual(flagged["overrides"], {"gate_min_stable_seconds": 90.0})
        self.assertEqual(flagged["content_hash"], block["content_hash"])
        self.assertNotEqual(flagged["effective_hash"], block["effective_hash"])

    def test_a_preset_changes_what_the_replay_plans(self):
        # 5 min of stable REM before the first cue: this recording has 150 s of it.
        report = self._simulate("p21", "--protocol-preset", "borghese2022_rem_home")
        self.assertEqual(report["protocol"]["name"], "borghese2022_rem_home")
        self.assertEqual(report["metrics"]["cue_plan_count"], 0)
        self.assertGreaterEqual(self._simulate("p21")["metrics"]["cue_plan_count"], 1)

    def test_an_explicit_flag_changes_what_the_replay_plans(self):
        # The flag alone asks for the 300 s this recording does not have.
        self.assertEqual(self._simulate("p21", "--min-stable-seconds", "300")["metrics"]["cue_plan_count"], 0)
        # And it wins over the preset: borghese2022_rem_home brought back to 60 s plans cues.
        report = self._simulate("p21", "--protocol-preset", "borghese2022_rem_home", "--min-stable-seconds", "60")
        self.assertGreaterEqual(report["metrics"]["cue_plan_count"], 1)
        self.assertEqual(report["protocol"]["overrides"], {"gate_min_stable_seconds": 60.0})

    def test_guard_default_follows_gate_optics_capable(self):
        catalog, session, assignment, cue_library = protocol_fixture()
        report = simulate_cue_plan_from_epochs(
            [build_rem_epoch(index, with_optics=False) for index in range(4)],
            catalog=catalog,
            session=session,
            assignment=assignment,
            cue_library=cue_library,
            gate_config=RemGateConfig(
                enter_threshold=0.60,
                exit_threshold=0.45,
                min_stable_seconds=60.0,
                epoch_seconds=30.0,
                cooldown_seconds=0.0,
                optics_capable=False,
            ),
            scheduler_config=TmrSchedulerConfig(enable_tlr_block=False),
        )

        decisions = [result.arousal_guard_decision for result in report.epoch_results]
        self.assertEqual([decision.action for decision in decisions], ["allow"] * 4)
        self.assertTrue(all(decision.metadata["ppg_absent_by_design"] for decision in decisions))


ARMS = ("cue", "sham", "no_cue")
SHAM_CUE = CueMetadata(
    "sham_soft_tone", "generated_tone", 1.0, protocol="sham", frequency_hz=330.0, tags=("sham", "untrained")
)


class TestPilot3Arms(unittest.TestCase):
    """simulate-replay-cues with --arm and --night-schedule (#228); mock audio only."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmpdir.name)
        write_eeg_only_recording(cls.root / "recording", ["nrem"] * 3 + ["rem"] * 5)
        catalog, session, assignment, cue_library = protocol_fixture()
        catalog.save(cls.root / "catalog.json")
        session.save(cls.root / "session.json")
        assignment.save(cls.root / "assignment.json")
        CueLibrary(library_id="arms", cues=cue_library.cues + (SHAM_CUE,)).save(cls.root / "cues.json")

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _run(self, name, *extra_args):
        output = self.root / name / "report.json"
        events = self.root / name / "events.jsonl"
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main([
                "simulate-replay-cues", str(self.root / "recording"),
                "--catalog", str(self.root / "catalog.json"),
                "--session", str(self.root / "session.json"),
                "--assignment", str(self.root / "assignment.json"),
                "--cue-library", str(self.root / "cues.json"),
                "--output", str(output),
                "--scheduler-events-output", str(events),
                "--preset", "p21",
                *extra_args,
            ])
        report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
        return code, report, events, out.getvalue() + err.getvalue()

    def test_the_three_arms_plan_at_the_same_times(self):
        _, plain, plain_events, _ = self._run("plain")
        runs = {arm: self._run(arm, "--arm", arm) for arm in ARMS}

        self.assertEqual(plain["schema_version"], 2)
        self.assertEqual((plain["arm"], plain["night_id"], plain["schedule_commitment_sha256"]), (None, None, None))
        self.assertNotIn("no_play_in_no_cue_arm", [c["name"] for c in plain["criteria"]])
        # --arm cue logs what a run without an arm logs, byte for byte.
        self.assertEqual(runs["cue"][2].read_bytes(), plain_events.read_bytes())
        cue_plays = runs["cue"][1]["cue_plan"]
        self.assertEqual([play["puzzle_id"] for play in cue_plays], ["p1", "p3"])
        for arm, (code, report, _, _) in runs.items():
            criteria = {c["name"]: c for c in report["criteria"]}
            self.assertEqual(code, 0, arm)
            self.assertEqual(report["arm"], arm)
            self.assertTrue(report["passed"], arm)
            self.assertTrue(criteria["no_play_in_no_cue_arm"]["passed"], arm)
            self.assertTrue(criteria["no_uncued_puzzle_cues"]["passed"], arm)
            self.assertNotIn("arm_matches_sealed_schedule", criteria)
            self.assertEqual(report["metrics"]["uncued_puzzle_play_count"], 0)
            self.assertEqual(report["protocol"], runs["cue"][1]["protocol"])
            self.assertFalse(report["audio_playback_executed"])
        sham = runs["sham"][1]
        self.assertEqual(
            [(p["timestamp_seconds"], p["puzzle_id"], p["metadata"]) for p in sham["cue_plan"]],
            [(p["timestamp_seconds"], p["puzzle_id"], p["metadata"]) for p in cue_plays],
        )
        self.assertEqual({p["cue_id"] for p in sham["cue_plan"]}, {"sham_soft_tone"})
        self.assertTrue(all(p["reason_codes"][-1] == "sham_arm" for p in sham["cue_plan"]))
        quiet = runs["no_cue"][1]
        self.assertEqual(quiet["cue_plan"], [])
        self.assertEqual(quiet["metrics"]["cue_plan_count"], 0)
        self.assertEqual(quiet["metrics"]["virtual_play_count"], 2)
        virtual = [e for e in quiet["scheduler_events"] if e["metadata"].get("virtual_play")]
        self.assertEqual(
            [(e["timestamp_seconds"], e["puzzle_id"], e["cue_id"]) for e in virtual],
            [(p["timestamp_seconds"], p["puzzle_id"], p["cue_id"]) for p in cue_plays],
        )

    def test_a_preset_runs_its_timing_in_every_arm(self):
        reports = {arm: self._run(f"preset_{arm}", "--arm", arm, "--protocol-preset", "konkoly2021_nijmegen")[1] for arm in ARMS}
        plans = {arm: [(e["timestamp_seconds"], e.get("puzzle_id")) for e in report["scheduler_events"]
                       if e["event_type"] == "play" or e["metadata"].get("virtual_play")]
                 for arm, report in reports.items()}
        self.assertTrue(plans["cue"])
        self.assertEqual(plans["sham"], plans["cue"])
        self.assertEqual(plans["no_cue"], plans["cue"])
        # The arm is no override of the preset: one protocol block for the three.
        blocks = {json.dumps(report["protocol"], sort_keys=True) for report in reports.values()}
        self.assertEqual(len(blocks), 1)
        self.assertEqual(reports["sham"]["protocol"]["name"], "konkoly2021_nijmegen")
        self.assertEqual(reports["sham"]["protocol"]["overrides"], {})

    def test_a_sealed_night_never_prints_its_arm(self):
        schedule_path = self.root / "schedules" / "series.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main([
                "plan-night-schedule", "--blocks", "1", "--seed", "8", "--output", str(schedule_path),
            ]), 0)
        schedule = load_night_schedule(schedule_path)
        printed = {}
        for night in schedule.nights:
            code, report, _, text = self._run(
                f"sealed_{night.night_id}", "--night-schedule", str(schedule_path), "--night-id", night.night_id
            )
            criteria = {c["name"]: c for c in report["criteria"]}
            self.assertEqual(code, 0)
            self.assertEqual(report["arm"], night.arm)
            self.assertEqual(report["night_id"], night.night_id)
            self.assertEqual(report["schedule_commitment_sha256"], schedule.commitment_sha256())
            self.assertTrue(criteria["arm_matches_sealed_schedule"]["passed"])
            self.assertEqual(
                criteria["arm_matches_sealed_schedule"]["observed"],
                {"commitment_matches": True, "arm_matches": True},
            )
            printed[night.arm] = text.replace(night.night_id, "NIGHT").replace(f"sealed_{night.night_id}", "DIR")
        self.assertEqual(set(printed), set(ARMS))
        self.assertEqual(len(set(printed.values())), 1, printed)
        tokens = set(printed["cue"].replace("=", " ").replace(",", " ").split())
        self.assertFalse(tokens & set(ARMS), printed["cue"])
        self.assertNotIn("sham", printed["cue"])
        self.assertNotIn("cue_plan", printed["cue"])

    def test_the_sham_cue_is_checked_in_every_arm(self):
        schedule_path = self.root / "schedules" / "trained_sham.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main([
                "plan-night-schedule", "--blocks", "1", "--seed", "8", "--sham-cue-id", "cue-p1",
                "--output", str(schedule_path),
            ]), 0)
        errors = set()
        for night in load_night_schedule(schedule_path).nights:
            code, report, _, text = self._run(
                f"trained_{night.night_id}", "--night-schedule", str(schedule_path), "--night-id", night.night_id
            )
            self.assertEqual((code, report), (2, None))
            errors.add(text.replace(night.night_id, "NIGHT"))
        self.assertEqual(len(errors), 1)
        self.assertIn("the cue of puzzle p1", errors.pop())

    def test_arm_flags_that_cannot_run_exit_before_anything_is_written(self):
        schedule_path = self.root / "schedules" / "washout.json"
        with contextlib.redirect_stdout(io.StringIO()):
            cli_main([
                "plan-night-schedule", "--blocks", "2", "--washout-nights", "1", "--seed", "3",
                "--output", str(schedule_path),
            ])
        cases = {
            "no_night_id": (("--night-schedule", str(schedule_path)), "needs --night-id"),
            "unknown_night": (("--night-schedule", str(schedule_path), "--night-id", "night-099"), "not in night schedule"),
            "washout": (("--night-schedule", str(schedule_path), "--night-id", "night-004"), "washout night"),
            "label_alone": (("--night-id", "night-001"), "go with --arm or --night-schedule"),
            "missing_sham_cue": (("--arm", "sham", "--sham-cue-id", "nowhere"), "sham cue"),
        }
        for name, (flags, message) in cases.items():
            with self.subTest(name):
                code, report, _, text = self._run(f"bad_{name}", *flags)
                self.assertEqual((code, report), (2, None))
                self.assertIn(message, text)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self._run("both", "--arm", "cue", "--night-schedule", str(schedule_path), "--night-id", "night-001")
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        payload["nights"][0]["arm"], payload["nights"][1]["arm"] = payload["nights"][1]["arm"], payload["nights"][0]["arm"]
        schedule_path.write_text(json.dumps(payload), encoding="utf-8")
        code, report, _, text = self._run("tampered", "--night-schedule", str(schedule_path), "--night-id", "night-001")
        self.assertEqual((code, report), (2, None))
        self.assertIn("does not match the commitment", text)

    def test_analyze_nof1_reads_the_reports_of_a_replayed_series(self):
        """The Pilot 3 reports of a sealed series, as the runner writes them, stand for its
        nights in analyze-nof1 once the series is unsealed (#228): replay, not sleep."""
        schedule_path = self.root / "schedules" / "replayed.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main([
                "plan-night-schedule", "--blocks", "1", "--seed", "8", "--output", str(schedule_path),
            ]), 0)
        protocol_dir = self.root / "replayed_protocol"
        protocol_dir.mkdir()
        (self.root / "replayed_reports").mkdir()
        summaries = []
        for night in load_night_schedule(schedule_path).nights:
            session = NightPuzzleSession(session_id=night.night_id, puzzle_ids=("p1", "p2", "p3", "p4"), puzzle_count=4)
            assignment = PuzzleCueAssignment(
                session_id=night.night_id, cued_puzzle_ids=("p1", "p3"), uncued_puzzle_ids=("p2", "p4"), seed=17
            )
            session.save(protocol_dir / f"{night.night_id}_puzzles.json")
            assignment.save(protocol_dir / f"{night.night_id}_assignment.json")
            code, report, _, _ = self._run(
                f"replayed_{night.night_id}",
                "--night-schedule", str(schedule_path), "--night-id", night.night_id,
                "--session", str(protocol_dir / f"{night.night_id}_puzzles.json"),
                "--assignment", str(protocol_dir / f"{night.night_id}_assignment.json"),
            )
            self.assertEqual(code, 0)
            summaries += ["--summary", str(self.root / f"replayed_{night.night_id}" / "report.json")]
        command = [
            "analyze-nof1", str(schedule_path),
            "--protocol-dir", str(protocol_dir),
            "--reports-dir", str(self.root / "replayed_reports"),
            "--recordings-dir", str(self.root / "replayed_recordings"),
            *summaries,
            "--output", str(self.root / "replayed_nof1.json"),
            "--markdown-output", str(self.root / "replayed_nof1.md"),
        ]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli_main(command), 2)  # still sealed
        unseal_night_schedule(schedule_path)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main(command), 0)
        payload = json.loads((self.root / "replayed_nof1.json").read_text(encoding="utf-8"))
        for night in payload["nights"]:
            self.assertEqual((night["summary_kind"], night["summary_arm"]), ("pilot3_report", night["arm"]))
            # A replay played nothing: its cues count as planned, never as plays.
            self.assertTrue(night["replay"])
            self.assertEqual(
                (night["plays"], night["planned_plays"], night["virtual_plays"]),
                (None, 0, 2) if night["arm"] == "no_cue" else (None, 2, 0),
                night["arm"],
            )
        self.assertEqual({arm: counts["plays"] for arm, counts in payload["cue_counts"].items()},
                         {"cue": None, "sham": None, "no_cue": None})
        codes = [item["code"] for item in payload["limitations"]]
        self.assertNotIn("arm_mismatch", codes)
        self.assertIn("missing_night_data", codes)
        replayed = next(item for item in payload["limitations"] if item["code"] == "replay_nights")
        self.assertEqual(replayed["metadata"]["night_ids"], [night.night_id for night in load_night_schedule(schedule_path).nights])
        # No sound reached the sleeper, so neither the sham's sound nor its scoring applies.
        self.assertNotIn("sham_detectable", codes)
        self.assertNotIn("missing_arousal_scoring", codes)
        self.assertEqual(payload["schedule"]["seal"], "matches")
        markdown = (self.root / "replayed_nof1.md").read_text(encoding="utf-8")
        self.assertEqual(markdown.count(" (replay) |"), 3)
        self.assertIn("`replay_nights`", markdown)


if __name__ == "__main__":
    unittest.main()
