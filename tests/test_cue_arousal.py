"""Cue-locked arousal proxies with paired quiet controls (#227).

Synthetic input, not sleep: every frame stream and recording here is generated, and
nothing measured on them says anything about arousals after real cues.
"""

import ast
import asyncio
import contextlib
import csv
import io
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from muse_tmr.audio import MockAudioBackend, VolumeCalibration
from muse_tmr.cli.main import main as cli_main
from muse_tmr.data.sample_types import EEGSample, IMUSample, MuseFrame, PPGSample
from muse_tmr.features import EpochBuilder, EpochConfig
from muse_tmr.models import RemGateConfig
from muse_tmr.protocol import TmrSchedulerConfig
from muse_tmr.reports.cue_arousal import (
    AROUSAL_FLAGS,
    CueArousalConfig,
    CueArousalReport,
    _control_targets,
    _Target,
    load_cue_arousal_report,
    score_cue_arousal,
    score_frame_windows,
    score_windows,
)
from muse_tmr.testing import SyntheticNightConfig, generate_night
from muse_tmr.testing.synthetic import START_EPOCH_SECONDS, pink_noise
from muse_tmr.validation import (
    REM_GATE_DECISIONS_FILENAME,
    GateDecisionRecord,
    gate_open_runs,
    load_gate_open_runs,
    load_rem_gate_decisions,
    run_pilot4_cueing_night,
    simulate_cue_plan_from_epochs,
    simulate_replay_cue_plan,
    write_rem_gate_decisions,
)
from tests.test_pilot4_cueing import (
    FakeMuseSource,
    build_rem_frames,
    pilot4_config,
    protocol_fixture,
    rem_frames,
    still_accel_rows,
    still_gyro_rows,
    synthetic_ppg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FS = 256
PPG_FS = 64
IMU_FS = 52
T0 = 1_700_000_000.0
CHANNELS = ("TP9", "AF7", "AF8", "TP10")
FRAME_SECONDS = 0.125


def band_noise(n, low_hz, high_hz, rng, fs=FS):
    """Unit-RMS noise with a flat spectrum between low_hz and high_hz."""
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0.0
    values = np.fft.irfft(spectrum, n)
    return values / np.std(values)


def pulse(seconds, heart_rate):
    """An optics pulse at 64 Hz whose rate is heart_rate(t) bpm, t in seconds from T0."""
    t = np.arange(int(seconds * PPG_FS)) / PPG_FS
    phase = 2 * np.pi * np.cumsum(heart_rate(t) / 60.0) / PPG_FS
    return 50000.0 + 1000.0 * (np.sin(phase) + 0.25 * np.sin(2 * phase - np.pi / 4))


def stream(
    seconds=150.0,
    *,
    seed=0,
    alpha=(),
    emg=(),
    line=None,
    heart_rate=None,
    movements=(),
    gaps=(),
):
    """Frames from T0, eight a second: 12 µV 1/f EEG with 1 µV of 20-100 Hz on every
    channel, a still IMU, optics only when heart_rate is given. alpha (10 Hz) and emg
    (55-95 Hz) are (start, stop, µV) bursts in seconds from T0 on every channel, line(t)
    the µV of a 63.93 Hz line on AF7, movements (start, stop) head turns of 0.6 g and
    80 deg/s, gaps (start, stop) stretches whose frames never arrive."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    eeg = {}
    for channel in CHANNELS:
        values = 12.0 * pink_noise(n, 1.0, rng, fs=FS, low_hz=0.3) + band_noise(n, 20.0, 100.0, rng)
        for start, stop, amplitude in alpha:
            inside = (t >= start) & (t < stop)
            values[inside] += amplitude * np.sin(2 * np.pi * 10.0 * t[inside])
        for start, stop, amplitude in emg:
            inside = (t >= start) & (t < stop)
            values[inside] += amplitude * band_noise(n, 55.0, 95.0, rng)[inside]
        eeg[channel] = values
    if line is not None:
        eeg["AF7"] = eeg["AF7"] + line(t) * np.sin(2 * np.pi * 63.93 * t)
    optics = None if heart_rate is None else pulse(seconds, heart_rate)
    frames = []
    per_frame = int(FRAME_SECONDS * FS)
    imu_done = 0
    for first in range(0, n, per_frame):
        offset = first / FS
        timestamp = T0 + offset
        imu_rows = int(round((first + per_frame) / FS * IMU_FS)) - imu_done
        imu_times = offset + np.arange(imu_rows) / IMU_FS
        imu_done += imu_rows
        if any(start <= offset < stop for start, stop in gaps):
            continue
        moving = [any(start <= moment < stop for start, stop in movements) for moment in imu_times]
        accel = [{"x": 0.6 if move else 0.0, "y": 0.0, "z": 1.0} for move in moving]
        gyro = [{"x": 80.0 if move else 0.0, "y": 0.0, "z": 0.0} for move in moving]
        ppg = None
        if optics is not None:
            low = int(round(offset * PPG_FS))
            ppg = PPGSample(timestamp, {"LO_NIR": optics[low:low + int(FRAME_SECONDS * PPG_FS)].tolist()}, source="test")
        frames.append(
            MuseFrame(
                timestamp=timestamp,
                eeg=EEGSample(
                    timestamp,
                    {channel: values[first:first + per_frame].tolist() for channel, values in eeg.items()},
                    source="test",
                ),
                imu=IMUSample(timestamp, accelerometer_g=accel, gyroscope_dps=gyro, source="test"),
                ppg=ppg,
                source="test",
            )
        )
    return frames


def stamped_late(frames, start, seconds):
    """frames with each one from start to start + seconds stamped at start + seconds, as
    the BLE client stamps the packets a busy event loop held; none is lost."""
    moment = start + seconds
    late = []
    for frame in frames:
        if start <= frame.timestamp < moment:
            frame = replace(
                frame,
                timestamp=moment,
                eeg=replace(frame.eeg, timestamp=moment),
                imu=replace(frame.imu, timestamp=moment),
            )
        late.append(frame)
    return late


CUE = T0 + 40.0
CONTROL = T0 + 100.0


def score(frames, *, optics_capable=False, config=None):
    cue, control = score_frame_windows(frames, [CUE, CONTROL], optics_capable=optics_capable, config=config)
    return cue, control


class WindowFlagTest(unittest.TestCase):
    """Each proxy on a stream with a known burst 2 to 8 s after the cue at 40 s."""

    def test_an_alpha_burst_after_the_cue_flags_alpha_and_its_control_does_not(self):
        cue, control = score(stream(alpha=[(42.0, 48.0, 25.0)]))
        self.assertIs(cue.flags["alpha"], True)
        self.assertIs(cue.flags["any"], True)
        self.assertGreater(cue.values["alpha_rise"], 5.0)
        self.assertIs(control.flags["alpha"], False)
        self.assertIs(control.flags["any"], False)
        for flag in ("beta", "emg", "imu"):
            self.assertIs(cue.flags[flag], False, flag)

    def test_an_alpha_burst_before_the_baseline_ends_is_not_a_rise(self):
        cue, _ = score(stream(alpha=[(25.0, 31.0, 25.0)]))
        self.assertIs(cue.flags["alpha"], False)

    def test_a_55_to_95_hz_burst_flags_emg_through_a_40_uv_line(self):
        constant = lambda t: np.full(t.shape, 40.0)  # noqa: E731
        cue, control = score(stream(emg=[(42.0, 48.0, 8.0)], line=constant))
        self.assertIs(cue.flags["emg"], True)
        self.assertIs(control.flags["emg"], False)

    def test_a_64_hz_line_alone_does_not_flag_emg(self):
        lines = {
            "constant 40 µV": lambda t: np.full(t.shape, 40.0),
            "rising 30 to 40 µV": lambda t: 30.0 + 10.0 * t / 150.0,
            "switched on at the cue": lambda t: np.where(t >= 40.0, 40.0, 0.0),
        }
        for name, line in lines.items():
            with self.subTest(line=name):
                cue, control = score(stream(line=line))
                self.assertIs(cue.flags["emg"], False)
                self.assertIs(control.flags["emg"], False)
                self.assertLess(cue.values["emg_rise"], 1.5)

    def test_without_the_line_handling_the_line_hides_an_emg_burst(self):
        # The same burst with the line fitted at a frequency far from it and no bridge:
        # the line dominates the band before and after the cue.
        config = replace(CueArousalConfig(), line_hz=(30.0,), line_half_width_hz=0.0)
        constant = lambda t: np.full(t.shape, 40.0)  # noqa: E731
        cue, _ = score(stream(emg=[(42.0, 48.0, 8.0)], line=constant), config=config)
        self.assertIs(cue.flags["emg"], False)

    def test_a_movement_after_the_cue_flags_imu(self):
        cue, control = score(stream(movements=[(43.0, 44.0)]))
        self.assertIs(cue.flags["imu"], True)
        self.assertGreaterEqual(cue.values["arousal_movement_events"], 1)
        self.assertIs(control.flags["imu"], False)

    def test_a_heart_rate_jump_after_the_cue_flags_hr(self):
        jump = lambda t: np.where((t >= 41.0) & (t < 60.0), 78.0, 60.0)  # noqa: E731
        cue, control = score(stream(heart_rate=jump), optics_capable=True)
        self.assertIs(cue.flags["hr"], True)
        self.assertGreaterEqual(cue.values["hr_jump_bpm"], 10.0)
        self.assertIs(control.flags["hr"], False)
        self.assertGreaterEqual(control.values["hr_values_baseline"], 3)

    def test_missing_ppg_gives_ppg_absent_not_a_flag(self):
        for optics_capable in (True, False):
            with self.subTest(optics_capable=optics_capable):
                cue, control = score(stream(), optics_capable=optics_capable)
                for window in (cue, control):
                    self.assertIsNone(window.flags["hr"])
                    self.assertIn("ppg_absent", window.quality_flags)
                    self.assertIsNotNone(window.flags["any"])

    def test_a_ble_gap_is_missing_signal_not_zeros(self):
        cue, control = score(stream(gaps=[(43.0, 47.0)]))
        for flag in ("alpha", "beta", "emg"):
            self.assertIsNone(cue.flags[flag], flag)
        self.assertIn("eeg_gap", cue.quality_flags)
        self.assertLess(cue.values["eeg_coverage_post"], 0.8)
        self.assertIs(control.flags["alpha"], False)
        self.assertNotIn("eeg_gap", control.quality_flags)

    def test_a_short_gap_is_still_scored(self):
        cue, _ = score(stream(alpha=[(42.0, 48.0, 25.0)], gaps=[(36.0, 36.25)]))
        self.assertIs(cue.flags["alpha"], True)
        self.assertNotIn("eeg_gap", cue.quality_flags)

    def test_frames_that_came_late_but_whole_keep_their_place(self):
        # A busy event loop stamps every packet it held with the moment it is free again.
        # Up to 127 subpackets (1.98 s of 4-channel EEG) the decoder would have reported a
        # loss, so frames that late with none reported lost nothing; later than that the
        # counter cannot tell, and the scorer counts a gap.
        frames = stream(alpha=[(42.0, 48.0, 25.0)])
        for seconds in (1.3, 1.9):
            with self.subTest(seconds=seconds):
                cue, control = score(stamped_late(frames, CUE + 0.2, seconds))
                self.assertIs(cue.flags["alpha"], True)
                self.assertEqual(cue.values["eeg_coverage_post"], 1.0)
                self.assertNotIn("eeg_gap", cue.quality_flags)
                self.assertEqual(control.values["eeg_coverage_post"], 1.0)
        cue, _ = score(stamped_late(frames, CUE + 0.2, 2.5))
        self.assertIn("eeg_gap", cue.quality_flags)
        self.assertLess(cue.values["eeg_coverage_post"], 0.8)

    def test_no_frames_scores_nothing(self):
        (window,) = score_frame_windows([], [CUE])
        self.assertTrue(all(window.flags[flag] is None for flag in AROUSAL_FLAGS))
        self.assertIsNone(window.flags["any"])
        self.assertIn("eeg_gap", window.quality_flags)


def records(states, *, epoch_seconds=30.0):
    """Gate records of consecutive epochs; states holds True for an open gate."""
    return tuple(
        GateDecisionRecord(
            index=index,
            timestamp_seconds=index * epoch_seconds,
            duration_seconds=epoch_seconds,
            gate_open=bool(state),
            state="open" if state else "closed",
        )
        for index, state in enumerate(states)
    )


class ControlSelectionTest(unittest.TestCase):
    def test_controls_are_quiet_and_in_the_cues_gate_state(self):
        # The gate is open from epoch 2 to epoch 29, closed before and after.
        gate = records([False] * 2 + [True] * 28 + [False] * 10)
        cue_offsets = [120.0, 150.0, 180.0, 450.0, 1080.0]
        cues = [
            _Target(kind="cue", index=index, time=T0 + offset, offset_seconds=offset, gate_open=offset < 900.0)
            for index, offset in enumerate(cue_offsets)
        ]
        config = CueArousalConfig()
        controls = _control_targets(cues, gate, T0, config)

        self.assertEqual(len(controls), len(cues))
        open_offsets = {record.decision_seconds for record in gate if record.gate_open}
        for control in controls:
            cue = cues[control.paired_index]
            self.assertEqual(control.gate_open, cue.gate_open)
            self.assertEqual(control.offset_seconds in open_offsets, cue.gate_open)
            self.assertGreater(min(abs(control.offset_seconds - offset) for offset in cue_offsets), 60.0)
            self.assertEqual(control.time, T0 + control.offset_seconds)
        offsets = sorted(control.offset_seconds for control in controls)
        self.assertTrue(all(later - earlier >= 30.0 for earlier, later in zip(offsets, offsets[1:])))

    def test_no_gate_log_means_no_controls(self):
        cues = [_Target(kind="cue", index=0, time=T0 + 300.0, offset_seconds=300.0, gate_open=True)]
        self.assertEqual(_control_targets(cues, (), T0, CueArousalConfig()), [])

    def test_a_cue_without_a_quiet_moment_stays_unpaired(self):
        gate = records([True] * 8)
        cues = [
            _Target(kind="cue", index=index, time=T0 + offset, offset_seconds=offset, gate_open=True)
            for index, offset in enumerate((60.0, 120.0, 180.0, 240.0))
        ]
        self.assertEqual(_control_targets(cues, gate, T0, CueArousalConfig()), [])


def nrem_frames(index):
    """One 30 s epoch of slow waves (40 µV at 1.5 Hz), a still IMU and a 72 bpm pulse."""
    timestamp = 1000.0 + index * 30.0
    t = np.arange(0, 30.0, 1 / FS)
    slow = (40.0 * np.sin(2 * np.pi * 1.5 * t) + 4.0 * np.sin(2 * np.pi * 6.0 * t)).tolist()
    return [
        MuseFrame(
            timestamp=timestamp,
            eeg=EEGSample(timestamp, {channel: slow for channel in CHANNELS}, source="test"),
            imu=IMUSample(timestamp, accelerometer_g=still_accel_rows(), gyroscope_dps=still_gyro_rows(), source="test"),
            ppg=PPGSample(timestamp, {"LO_NIR": synthetic_ppg(72.0)}, source="test"),
            source="test",
            raw_packet=b"pilot4",
        )
    ]


async def _as_stream(frames):
    for frame in frames:
        yield frame


class GateLogTest(unittest.IsolatedAsyncioTestCase):
    def test_records_round_trip_through_rem_gate_decisions_jsonl(self):
        logged = records([False, True, True, False, True])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_rem_gate_decisions(logged[:2], Path(tmpdir) / REM_GATE_DECISIONS_FILENAME)
            write_rem_gate_decisions(logged[2:], path, append=True)
            self.assertEqual(load_rem_gate_decisions(Path(tmpdir)), logged)
            self.assertEqual(load_rem_gate_decisions(path), logged)
            self.assertEqual(load_rem_gate_decisions(Path(tmpdir) / "missing.jsonl"), ())
        runs = gate_open_runs(logged)
        self.assertEqual(
            [(run.start_seconds, run.end_seconds, run.closed, run.epoch_count) for run in runs],
            [(60.0, 120.0, True, 2), (150.0, 150.0, False, 1)],
        )
        self.assertTrue(runs[0].contains(60.0))
        self.assertFalse(runs[0].contains(120.0))

    async def test_pilot4_log_and_pilot3_report_give_the_same_runs(self):
        kinds = ["nrem", "rem", "rem", "rem", "nrem", "nrem", "rem", "rem", "rem"]
        frames = []
        for index, kind in enumerate(kinds):
            frames.extend(build_rem_frames(index) if kind == "rem" else nrem_frames(index))
        catalog, session, assignment, cue_library = protocol_fixture()
        config = replace(pilot4_config(Path("unused"), duration_seconds=600.0), score_cue_arousal=False)
        # The slow-wave epochs score P_REM 0.5 and the REM-like ones 0.8: an exit of 0.55
        # closes the gate on them.
        config = replace(config, gate_config=replace(config.gate_config, exit_threshold=0.55))
        with tempfile.TemporaryDirectory() as tmpdir:
            night_dir = Path(tmpdir) / "night"
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(frames),
                config=replace(config, output_dir=night_dir),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
            )
            epochs = [epoch async for epoch in EpochBuilder(EpochConfig()).build(_as_stream(frames))]
            report = simulate_cue_plan_from_epochs(
                epochs,
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                gate_config=config.gate_config,
                scheduler_config=config.scheduler_config,
            )
            report_path = report.save(Path(tmpdir) / "pilot3.json")
            from_pilot4 = load_gate_open_runs(night_dir)
            from_pilot3 = load_gate_open_runs(report_path)
            pilot4_records = load_rem_gate_decisions(night_dir)
            payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))

        self.assertEqual(summary.epoch_count, len(kinds))
        self.assertEqual(from_pilot4, from_pilot3)
        self.assertEqual(from_pilot3, load_gate_open_runs(report.to_dict()))
        self.assertEqual(
            [(run.first_epoch_index, run.last_epoch_index, run.closed) for run in from_pilot3],
            [(2, 3, True), (7, 8, False)],
        )
        self.assertEqual((from_pilot3[0].start_seconds, from_pilot3[0].end_seconds), (90.0, 150.0))
        self.assertEqual(
            [(record.gate_open, record.state) for record in pilot4_records],
            [(result.gate_decision.gate_open, result.gate_decision.state) for result in report.epoch_results],
        )
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["first_epoch_start"], 1000.0)
        self.assertEqual(report.metrics["first_epoch_start"], 1000.0)
        self.assertTrue(payload["rem_gate_decisions_path"].endswith(REM_GATE_DECISIONS_FILENAME))
        self.assertEqual(payload["cue_arousal"], {})


class InjectedCueTest(unittest.TestCase):
    def test_injected_cues_leave_the_scheduler_events_alone(self):
        from tests.test_pilot3_replay_simulation import build_rem_epoch
        from tests.test_pilot3_replay_simulation import protocol_fixture as pilot3_fixture

        catalog, session, assignment, cue_library = pilot3_fixture()
        epochs = [build_rem_epoch(index) for index in range(10)]
        gate = RemGateConfig(enter_threshold=0.60, exit_threshold=0.45, cooldown_seconds=0.0)

        def run(every):
            return simulate_cue_plan_from_epochs(
                epochs,
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                gate_config=gate,
                scheduler_config=TmrSchedulerConfig(enable_tlr_block=False),
                inject_cues_every_seconds=every,
            )

        plain, injected = run(None), run(90.0)
        self.assertEqual(
            [event.to_dict() for event in plain.scheduler_events],
            [event.to_dict() for event in injected.scheduler_events],
        )
        self.assertEqual(plain.injected_cues, ())
        self.assertEqual(plain.metrics["injected_cue_count"], 0)
        decided = [event.metadata["decision_seconds"] for event in injected.injected_cues]
        open_decisions = [
            result.timestamp_seconds + result.duration_seconds
            for result in injected.epoch_results
            if result.gate_decision.gate_open
        ]
        self.assertEqual(decided, [60.0, 150.0, 240.0])
        self.assertTrue(set(decided) <= set(open_decisions))
        self.assertEqual(injected.metrics["injected_cue_count"], 3)
        self.assertEqual(injected.metrics["uncued_puzzle_play_count"], 0)
        for event in injected.injected_cues:
            self.assertEqual((event.event_type, event.cue_id, event.protocol), ("play", "synthetic_cue", "synthetic"))
            self.assertIn("synthetic_cue", event.reason_codes)
        self.assertNotIn("synthetic_cue", [event.cue_id for event in injected.cue_plan])
        self.assertEqual(len(injected.to_dict()["injected_cues"]), 3)
        with self.assertRaises(ValueError):
            run(0.0)


# A short synthetic p21 night, synthetic input, not sleep: 4 min of N2 and 9 of REM
# without arousals, so the gate opens and nothing in it is an arousal.
PLACEMENT_STAGES = ("N2",) * 8 + ("R",) * 18


class WindowPlacementTest(unittest.TestCase):
    """Cues injected by Pilot 3 into a replayed night land where the scorer cuts them."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.night = generate_night(
            replace(
                SyntheticNightConfig(),
                preset="p21",
                scenario="placement",
                hours=len(PLACEMENT_STAGES) * 30 / 3600,
                stages=PLACEMENT_STAGES,
                arousals_per_hour=0.0,
                ble_gaps_per_hour=0.0,
            ),
            root / "night",
        )
        catalog, session, assignment, cue_library = protocol_fixture()
        for name, item in (("catalog", catalog), ("session", session), ("assignment", assignment), ("cues", cue_library)):
            item.save(root / f"{name}.json")
        cls.report_path = root / "reports" / "pilot3.json"
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            cls.exit_code = cli_main([
                "simulate-replay-cues", str(cls.night.recording_dir),
                "--catalog", str(root / "catalog.json"),
                "--session", str(root / "session.json"),
                "--assignment", str(root / "assignment.json"),
                "--cue-library", str(root / "cues.json"),
                "--output", str(cls.report_path),
                "--preset", "p21",
                # Below the p21 pair, so the gate stays open through the REM epochs and
                # leaves quiet moments for controls between the cues.
                "--enter-threshold", "0.7",
                "--exit-threshold", "0.6",
                "--inject-cues-every-seconds", "180",
            ])
        cls.simulate_output = printed.getvalue()
        cls.pilot3 = json.loads(cls.report_path.read_text(encoding="utf-8"))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            cls.score_exit_code = cli_main([
                "score-cue-arousal", str(cls.night.recording_dir),
                "--pilot3-report", str(cls.report_path),
            ])
        cls.score_output = printed.getvalue()
        cls.report = load_cue_arousal_report(cls.report_path.parent / "cue_arousal_report.json")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_commands_ran(self):
        self.assertEqual(self.exit_code, 0)
        self.assertEqual(self.score_exit_code, 0)
        self.assertIn("injected_cues=", self.simulate_output)
        self.assertIn("synthetic input, not sleep", self.score_output)
        for name in ("cue_arousal_events.jsonl", "cue_arousal_report.json", "cue_arousal_report.md"):
            self.assertTrue((self.report_path.parent / name).is_file(), name)

    def test_each_injected_cue_is_scored_at_the_end_of_its_epoch(self):
        first_epoch_start = self.pilot3["metrics"]["first_epoch_start"]
        injected = self.pilot3["injected_cues"]
        self.assertGreaterEqual(len(injected), 2)
        synthetic = [window for window in self.report.cues if window.cue.get("synthetic_cue")]
        self.assertEqual(
            [window.offset_seconds for window in synthetic],
            [event["metadata"]["decision_seconds"] for event in injected],
        )
        for window in synthetic:
            self.assertAlmostEqual(window.time, first_epoch_start + window.offset_seconds, places=6)
            self.assertIs(window.gate_open, True)
            self.assertIsNotNone(window.rem_period_index)
            self.assertGreaterEqual(window.values["eeg_coverage_post"], 0.8)

    def test_controls_are_quiet_and_in_the_same_gate_state(self):
        cue_offsets = [window.offset_seconds for window in self.report.cues]
        self.assertGreaterEqual(len(self.report.controls), 1)
        for control in self.report.controls:
            self.assertGreater(min(abs(control.offset_seconds - offset) for offset in cue_offsets), 60.0)
            paired = self.report.cues[control.paired_index]
            self.assertEqual(control.gate_open, paired.gate_open)
            self.assertEqual(paired.paired_index, control.index)

    def test_every_window_is_scored_from_the_replayed_night(self):
        # Eye movements and theta in synthetic REM raise frontal power now and then, so a
        # window can flag without an arousal; the controls carry that rate.
        for window in self.report.windows:
            self.assertIsNotNone(window.flags["any"], (window.kind, window.index))
            self.assertNotIn("eeg_gap", window.quality_flags)
            self.assertIn("ppg_absent", window.quality_flags)
            self.assertIsNone(window.flags["hr"])

    def test_the_report_says_what_it_is(self):
        codes = self.report.limitation_codes
        for code in ("thresholds_unfitted", "arousal_proxy_only", "synthetic_input", "no_audio_played", "ppg_absent"):
            self.assertIn(code, codes)
        self.assertNotIn("time_base_inferred", codes)
        self.assertTrue(self.report.synthetic)
        saved = json.loads((self.report_path.parent / "cue_arousal_report.json").read_text(encoding="utf-8"))
        self.assertIs(saved["synthetic"], True)
        self.assertEqual(saved["note"], "synthetic input, not sleep")
        markdown = (self.report_path.parent / "cue_arousal_report.md").read_text(encoding="utf-8")
        self.assertIn("Synthetic input, not sleep", markdown)
        self.assertIn("UNFITTED", markdown)
        self.assertEqual(self.report.dose_tables["cue_index"][-1]["bin"], "synthetic")
        lines = (self.report_path.parent / "cue_arousal_events.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), len(self.report.windows))

    def test_the_report_round_trips(self):
        again = CueArousalReport.from_dict(self.report.to_dict())
        self.assertEqual(again.to_dict(), self.report.to_dict())

    def test_without_the_metric_the_time_base_comes_from_the_first_frame(self):
        older = dict(self.pilot3)
        older["metrics"] = {key: value for key, value in self.pilot3["metrics"].items() if key != "first_epoch_start"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pilot3_old.json"
            path.write_text(json.dumps(older), encoding="utf-8")
            report = asyncio.run(score_cue_arousal(self.night.recording_dir, pilot3_report=path))
        self.assertIn("time_base_inferred", report.limitation_codes)
        self.assertEqual(
            [round(window.time, 3) for window in report.windows],
            [round(window.time, 3) for window in self.report.windows],
        )


class CleanNightCorpusTest(unittest.TestCase):
    """clean_night, 45 min, seed 0 (synthetic input, not sleep): cues placed at the
    arousals of its arousals.csv against cues at quiet N2 and N3 moments."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        night = generate_night(output_dir=Path(cls._tmp.name) / "night", scenario="clean_night", hours=0.75, seed=0)
        duration = float(night.generation["duration_seconds"])
        with (night.recording_dir / "arousals.csv").open(newline="", encoding="utf-8") as handle:
            arousals = list(csv.DictReader(handle))
        with (night.recording_dir / "hypnogram.csv").open(newline="", encoding="utf-8") as handle:
            hypnogram = list(csv.DictReader(handle))
        spans = [(float(row["onset_s"]), float(row["onset_s"]) + float(row["duration_s"])) for row in arousals]
        cls.arousals = [
            row for row in arousals
            if row["stage"] != "W" and float(row["onset_s"]) >= 40.0 and float(row["onset_s"]) + 20.0 <= duration
        ]
        quiet = []
        for row in hypnogram:
            moment = float(row["start_s"]) + 15.0
            if row["stage"] not in ("N2", "N3"):
                continue
            if any(start - 90.0 < moment < end + 90.0 for start, end in spans):
                continue
            if quiet and moment - quiet[-1] < 60.0:
                continue
            quiet.append(moment)
        cls.quiet = quiet
        times = [START_EPOCH_SECONDS + float(row["onset_s"]) for row in cls.arousals]
        times += [START_EPOCH_SECONDS + moment for moment in quiet]
        windows = asyncio.run(score_windows(night.recording_dir, times, optics_capable=False))
        cls.at_arousals = windows[: len(cls.arousals)]
        cls.at_quiet = windows[len(cls.arousals):]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_cues_at_arousals_flag_far_more_often_than_cues_at_quiet_moments(self):
        self.assertGreaterEqual(len(self.at_arousals), 5)
        self.assertGreaterEqual(len(self.at_quiet), 10)
        arousal_rate = np.mean([window.flags["any"] for window in self.at_arousals])
        quiet_rate = np.mean([window.flags["any"] for window in self.at_quiet])
        self.assertGreaterEqual(arousal_rate, 0.8)
        self.assertLessEqual(quiet_rate, 0.2)
        self.assertGreaterEqual(arousal_rate - quiet_rate, 0.6)
        self.assertTrue(all(window.flags["alpha"] for window in self.at_arousals))

    def test_the_imu_flags_the_arousals_with_movement(self):
        for row, window in zip(self.arousals, self.at_arousals):
            self.assertEqual(window.flags["imu"], row["movement"] == "1", row)


class AwakeningFieldsTest(unittest.TestCase):
    def test_old_rows_load_and_new_fields_round_trip(self):
        from muse_tmr.validation import AwakeningEvent, append_awakening_event, load_awakening_events

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "awakening_events.jsonl"
            path.write_text(
                json.dumps({"event_type": "awakening", "timestamp_utc": "2026-05-13T22:00:00+00:00", "notes": "old"}) + "\n",
                encoding="utf-8",
            )
            append_awakening_event(
                path,
                AwakeningEvent(timestamp_utc="2026-05-13T23:00:00+00:00", timestamp_seconds=3610.0, attributed_to_cue="Yes"),
            )
            old, new = load_awakening_events(path)
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual((old.notes, old.timestamp_seconds, old.attributed_to_cue), ("old", None, None))
        self.assertEqual((new.timestamp_seconds, new.attributed_to_cue), (3610.0, "yes"))
        self.assertNotIn("timestamp_seconds", lines[0])
        self.assertEqual(lines[1]["attributed_to_cue"], "yes")
        with self.assertRaises(ValueError):
            AwakeningEvent(attributed_to_cue="maybe")
        with self.assertRaises(ValueError):
            AwakeningEvent(timestamp_seconds=-1.0)

    def test_the_command_takes_both_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmpdir) / "awakening_events.jsonl"
            exit_code = cli_main([
                "log-pilot4-awakening", str(path),
                "--timestamp-seconds", "1250.5",
                "--attributed-to-cue", "unsure",
                "--notes", "woke after a tone",
            ])
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertEqual((row["timestamp_seconds"], row["attributed_to_cue"]), (1250.5, "unsure"))


class MorningRescoreTest(unittest.TestCase):
    def test_scoring_the_night_again_rewrites_its_summary_block(self):
        # The night scores itself when it ends, before any marker is logged; the runbooks
        # say to score it again after logging awakenings in the morning.
        from muse_tmr.validation import AwakeningEvent, append_awakening_event

        catalog, session, assignment, cue_library = protocol_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            night_dir = Path(tmpdir)
            summary = asyncio.run(
                run_pilot4_cueing_night(
                    FakeMuseSource(rem_frames(4)),
                    config=pilot4_config(night_dir, duration_seconds=60.0),
                    catalog=catalog,
                    session=session,
                    assignment=assignment,
                    cue_library=cue_library,
                    calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                    backend=MockAudioBackend(),
                )
            )
            summary_path = Path(summary.summary_path)
            at_night_end = json.loads(summary_path.read_text(encoding="utf-8"))
            # 30 s after the first cue (60 s), as the sleeper marks it in the morning.
            append_awakening_event(
                night_dir / "awakening_events.jsonl",
                AwakeningEvent(timestamp_seconds=90.0, attributed_to_cue="yes"),
            )
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                exit_code = cli_main(["score-cue-arousal", str(night_dir)])
            rescored = json.loads(summary_path.read_text(encoding="utf-8"))
            report = json.loads((night_dir / "cue_arousal_report.json").read_text(encoding="utf-8"))

            # A summary from before the block existed is left as it is.
            old = {key: value for key, value in rescored.items() if key != "cue_arousal"}
            old["schema_version"] = 1
            summary_path.write_text(json.dumps(old), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as printed_old:
                old_exit_code = cli_main(["score-cue-arousal", str(night_dir)])
            old_after = summary_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(at_night_end["cue_arousal"]["cue_attributed_awakening_rate"], 0.0)
        block = rescored["cue_arousal"]
        self.assertEqual(block["status"], "scored")
        self.assertEqual(block["cue_attributed_awakening_rate"], 0.5)
        self.assertEqual(report["outcomes"]["cue_attributed_awakening_rate"], 0.5)
        self.assertEqual(report["outcomes"]["awakening_attribution_counts"], {"yes": 1})
        self.assertEqual(block["report_path"], str(night_dir / "cue_arousal_report.json"))
        self.assertIn(f"summary={summary_path}", printed.getvalue())
        self.assertEqual(
            {key: value for key, value in rescored.items() if key != "cue_arousal"},
            {key: value for key, value in at_night_end.items() if key != "cue_arousal"},
        )
        self.assertEqual(old_exit_code, 0)
        self.assertEqual(json.loads(old_after), old)
        self.assertNotIn("summary=", printed_old.getvalue())


class BoundaryTest(unittest.TestCase):
    def test_the_scorer_imports_nothing_from_audio(self):
        source = (REPO_ROOT / "src" / "muse_tmr" / "reports" / "cue_arousal.py").read_text(encoding="utf-8")
        imported = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        self.assertFalse([name for name in imported if name.startswith("muse_tmr.audio")], imported)
        self.assertFalse([name for name in imported if name.startswith("muse_tmr.protocol")], imported)

    def test_the_wording_stays_with_proxies(self):
        for relative in ("src/muse_tmr/reports/cue_arousal.py", "src/muse_tmr/validation/gate_log.py", "docs/safety_outcomes.md"):
            text = (REPO_ROOT / relative).read_text(encoding="utf-8").lower()
            with self.subTest(file=relative):
                self.assertNotIn("arousal detected", text)
                self.assertNotIn("detected arousal", text)
                self.assertIsNone(re.search(r"\bsafe\b", text))
                self.assertIsNone(re.search(r"\blucid", text))
                self.assertEqual(re.findall(r"\beffective\b(?![ _]volume)", text), [])

    def test_the_thresholds_are_marked_unfitted(self):
        config = CueArousalConfig()
        self.assertIn("UNFITTED", CueArousalConfig.__doc__)
        self.assertEqual(
            {name: getattr(config, name) for name in config.to_dict()["unfitted_thresholds"]},
            {"alpha_rise_ratio": 2.0, "beta_rise_ratio": 2.0, "emg_rise_ratio": 2.0, "hr_jump_bpm": 10.0},
        )


if __name__ == "__main__":
    unittest.main()
