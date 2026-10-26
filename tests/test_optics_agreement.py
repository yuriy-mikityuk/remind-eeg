"""The optics heart-rate agreement report against a Polar H10 ECG (#234).

Muse optics frames and a Polar H10 session are built from the same scripted beats.
Synthetic input, not sleep: the agreement here checks the code, not the headband.
"""

import ast
import asyncio
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from muse_tmr.cli.main import main as cli_main
from muse_tmr.data.polar_session import load_polar_session
from muse_tmr.data.sample_types import EEGSample, MuseFrame, PPGSample
from muse_tmr.features import EpochBuilder, EpochConfig
from muse_tmr.features.ppg_features import extract_ppg_features
from muse_tmr.protocol import ArousalGuard, ArousalGuardConfig
from muse_tmr.reports.optics_agreement import (
    JSON_FILENAME,
    MARKDOWN_FILENAME,
    OpticsAgreementConfig,
    OpticsAgreementInputError,
    OpticsAgreementReport,
    load_optics_agreement_report,
    optics_agreement_from_epochs,
)
from muse_tmr.testing import SyntheticNightConfig, generate_night, heartbeat_onsets
from muse_tmr.testing.synthetic import OPTICS_CHANNELS, START_EPOCH_SECONDS, beat_volume
from tests.polar_synthetic import beat_times, write_raw_session

FS = 64.0
CLEAN = "LI_NIR"
# A channel without a pulse: a 63.93 Hz line, which 64 Hz sampling folds to 0.07 Hz, and
# white noise. Its raw SD stays under the pulse channels', so it is never the primary one.
NOISY = "RO_IR"
REPO_ROOT = Path(__file__).resolve().parents[1]


def beat_phase(beats_s, times):
    """Beats since the first scripted beat at each time, the phase beat_volume() takes."""
    index = np.clip(np.searchsorted(beats_s, times, side="right") - 1, 0, len(beats_s) - 2)
    return index + (times - beats_s[index]) / (beats_s[index + 1] - beats_s[index])


def optics_frames(beats_s, seconds, rng, *, wall0, noisy=NOISY, frame_seconds=0.5):
    """Frames of 8 optics channels at 64 Hz from wall0: a pulse after each beat on every
    channel but ``noisy``."""
    times = np.arange(int(seconds * FS)) / FS
    volume = beat_volume(beat_phase(np.asarray(beats_s, dtype=float), times))
    channels = {}
    for name in OPTICS_CHANNELS:
        if name == noisy:
            line = 0.01 * np.sin(2 * np.pi * 63.93 * times + 1.0)
            channels[name] = 10.0 + line + 0.01 * rng.standard_normal(times.size)
        else:
            channels[name] = 12.0 * (1.0 - 0.012 * volume) + 0.0007 * rng.standard_normal(times.size)
    rows = int(frame_seconds * FS)
    frames = []
    for first in range(0, times.size - rows + 1, rows):
        stamp = wall0 + first / FS
        values = {name: column[first:first + rows].tolist() for name, column in channels.items()}
        frames.append(MuseFrame(timestamp=stamp, ppg=PPGSample(stamp, values, source="test"), source="test"))
    return frames


def epochs_of(frames):
    async def build():
        async def stream():
            for frame in frames:
                yield frame

        return [epoch async for epoch in EpochBuilder(EpochConfig()).build(stream())]

    return asyncio.run(build())


def frames_report(beats_s, seconds, seed, tmp, *, config=None, no_contact=None):
    """Polar session and Muse frames from the same beats, and their report. The H10 has
    no skin contact between the seconds ``no_contact``."""
    rng = np.random.default_rng(seed)
    truth = write_raw_session(tmp, seconds, rng, beats_s=beats_s, no_contact=no_contact)
    session = load_polar_session(Path(tmp))
    epochs = epochs_of(optics_frames(beats_s, seconds, rng, wall0=truth["wall0"]))
    return optics_agreement_from_epochs(epochs, session, config), epochs


def held_beats(seconds, seed, *rates, base_bpm=60.0):
    """Beats at ``base_bpm``, and at ``bpm`` for each (start, end, bpm) in ``rates``."""
    rng = np.random.default_rng(seed)
    beats = [1.0]
    while beats[-1] < seconds - 1.0:
        bpm = next((rate for start, end, rate in rates if start <= beats[-1] < end), base_bpm)
        beats.append(beats[-1] + 60.0 / bpm + rng.normal(0.0, 0.005))
    return np.asarray(beats)


def guard_pauses(epochs):
    """The epochs ArousalGuard pauses for a sudden heart-rate change and the ones it logs
    as hr_rule_blind, from their PPG rows alone."""
    guard = ArousalGuard()
    paused, blind = set(), set()
    for epoch in epochs:
        decision = guard.evaluate(timestamp_seconds=0.0, ppg=extract_ppg_features(epoch))
        if "sudden_heart_rate_change" in decision.reason_codes:
            paused.add(epoch.index)
        if decision.metadata.get("hr_rule_blind"):
            blind.add(epoch.index)
    return paused, blind


class SteadyHeartTest(unittest.TestCase):
    """Ten minutes around 60 bpm with sinus arrhythmia, no step."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        beats = beat_times(600.0, rng, breathing_hz=0.25)
        with tempfile.TemporaryDirectory() as tmp:
            cls.report, cls.epochs = frames_report(beats, 600.0, 12, tmp)

    def test_the_clean_channel_agrees_with_the_ecg(self):
        for estimator, channel in (("channel", CLEAN), ("ppg_estimated_hr_bpm", ""), ("hr_series", "")):
            with self.subTest(estimator=estimator):
                row = self.report.overall(estimator, channel)
                self.assertGreater(row.n, 0)
                self.assertLess(abs(row.bias_bpm), 1.0)
                self.assertGreater(row.loa_low_bpm, -3.0)
                self.assertLess(row.loa_high_bpm, 3.0)
        self.assertEqual(self.report.overall("channel", CLEAN).n, 20)
        self.assertGreaterEqual(self.report.overall("hr_series").coverage, 0.95)

    def test_the_line_and_noise_channel_is_reported_apart(self):
        noisy = self.report.overall("channel", NOISY)
        self.assertEqual(noisy.n, 20)
        self.assertGreater(noisy.mae_bpm, 10.0)
        periodicity = {row.level: row.n for row in self.report.rows("channel", channel=NOISY, stratum="periodicity")}
        self.assertEqual(periodicity, {"0-0.5": 20})
        clean = {row.level: row.n for row in self.report.rows("channel", channel=CLEAN, stratum="periodicity")}
        self.assertEqual(sum(n for level, n in clean.items() if level != "0-0.5"), 20)
        # The peak-interval confidence behind ppg_signal_quality does not flag it: band-passed
        # noise has peaks about as regular as a pulse's. Only the periodicity does.
        qualities = {row.level for row in self.report.rows("channel", channel=NOISY, stratum="ppg_signal_quality")}
        self.assertNotIn("Poor", qualities)

    def test_the_primary_channel_is_never_the_noisy_one(self):
        for epoch in self.epochs:
            self.assertNotEqual(extract_ppg_features(epoch).primary_ppg_channel, NOISY)

    def test_no_decoder_stream_no_decoder_estimate(self):
        row = self.report.overall("decoder_heart_rate")
        self.assertEqual((row.n_reference, row.n), (20, 0))
        self.assertTrue(math.isnan(row.bias_bpm))

    def test_a_steady_heart_neither_jumps_nor_fires(self):
        overall = self.report.jumps[0]
        self.assertEqual((overall.stratum, overall.epochs), ("all", 20))
        self.assertEqual((overall.fired, overall.false_jumps, overall.ecg_jumps, overall.missed), (0, 0, 0, 0))
        self.assertEqual(self.report.events, ())

    def test_limitations_and_counts(self):
        self.assertEqual(self.report.limitation_codes, ("n_epochs_small", "few_jump_events"))
        self.assertFalse(self.report.synthetic)
        self.assertEqual(self.report.counts["compared_epochs"], 20)
        self.assertEqual(self.report.counts["no_contact_seconds"], 0.0)
        self.assertEqual(self.report.counts["ecg_intervals_corrected"], 0)

    def test_saved_report_loads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.report.write(Path(tmp))
            self.assertEqual(paths["json"].name, JSON_FILENAME)
            self.assertEqual(paths["markdown"].name, MARKDOWN_FILENAME)
            loaded = load_optics_agreement_report(paths["json"])
            markdown = paths["markdown"].read_text(encoding="utf-8")
        self.assertEqual(loaded.to_dict(), self.report.to_dict())
        self.assertEqual(loaded.schema_version, 1)
        self.assertIn("| channel | RO_IR |", markdown)
        self.assertIn("a proxy for heart rate; no PSG takes part", markdown)
        self.assertNotIn("Synthetic input", markdown)


class HeldStepTest(unittest.TestCase):
    """A 15 bpm step held 60 s, 15 s into epoch 4, back down 15 s into epoch 6. The rule
    pauses epochs 4 and 6 and, from the window over the step into epoch 5's values,
    epoch 5: a fire in the epoch after an ECG jump counts for that jump."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(21)
        beats = [1.0]
        while beats[-1] < 479.0:
            fast = 135.0 <= beats[-1] < 195.0
            beats.append(beats[-1] + (0.8 if fast else 1.0) + rng.normal(0.0, 0.01))
        cls.beats = np.asarray(beats)
        with tempfile.TemporaryDirectory() as tmp:
            cls.report, cls.epochs = frames_report(cls.beats, 480.0, 22, tmp)

    def test_the_step_is_an_ecg_jump_the_rule_catches(self):
        overall = self.report.jumps[0]
        self.assertEqual(overall.ecg_jumps, 2)
        self.assertEqual((overall.fired, overall.false_jumps, overall.missed), (3, 0, 0))
        self.assertEqual({event.epoch_index for event in self.report.events}, {4, 5, 6})
        for event in self.report.events:
            # A window over the step averages both rates on the ECG, while the series reads
            # its longer run of regular beats, so only the direction has to agree.
            self.assertEqual(event.optics_delta_bpm > 0, event.ecg_delta_bpm > 0)
        self.assertAlmostEqual(
            max(abs(event.ecg_delta_bpm) for event in self.report.events), 15.0, delta=2.0
        )

    def test_the_report_fires_where_the_guard_pauses(self):
        paused, _blind = guard_pauses(self.epochs)
        self.assertEqual(paused, {event.epoch_index for event in self.report.events})

    def test_gates_keep_or_drop_the_steps(self):
        gates = {(row.field, row.threshold): row for row in self.report.gates}
        self.assertEqual((gates[("none", 0.0)].fired, gates[("none", 0.0)].caught), (3, 2))
        series = gates[("hr_series_median_quality", 0.5)]
        self.assertEqual((series.gated_epochs, series.caught), (0, 2))
        # A gate of 1 on either field would hold back every epoch.
        strict = replace(OpticsAgreementConfig(), gate_thresholds=(1.0,))
        with tempfile.TemporaryDirectory() as tmp:
            report, _epochs = frames_report(self.beats, 480.0, 22, tmp, config=strict)
        for row in report.gates[1:]:
            self.assertEqual((row.fired, row.caught), (0, 0), row.field)
        self.assertEqual(report.separation["ppg_confidence"]["true_fires"], 3)
        self.assertEqual(report.separation["ppg_confidence"]["false_fires"], 0)

    def test_the_jump_fields_mirror_the_guard_defaults(self):
        config, guard = OpticsAgreementConfig(), ArousalGuardConfig()
        self.assertEqual(config.jump_bpm, guard.sudden_hr_change_pause_bpm)
        self.assertEqual(config.jump_margin_bpm, guard.sudden_hr_change_margin_bpm)
        self.assertEqual(config.jump_pause_count, guard.sudden_hr_change_pause_count)
        self.assertEqual(config.jump_link_seconds, guard.sudden_hr_change_link_seconds)
        self.assertEqual(config.hr_rule_blind_skipped_share, guard.hr_rule_blind_skipped_share)


class StepScoringTest(unittest.TestCase):
    """How the report scores held steps against the ECG, next to the epochs ArousalGuard
    pauses on the same rows. Five minutes around 60 bpm in epochs of 30 s from 0 s."""

    def score(self, *rates, no_contact=None):
        beats = held_beats(300.0, 21, *rates)
        with tempfile.TemporaryDirectory() as tmp:
            report, epochs = frames_report(beats, 300.0, 22, tmp, no_contact=no_contact)
        overall = report.jumps[0]
        counts = (overall.ecg_jumps, overall.fired, overall.false_jumps, overall.missed)
        return report, counts, guard_pauses(epochs)[0]

    def test_a_rise_the_next_epoch_sees_again_counts_once(self):
        # The ECG window over the rise averages both rates, so epoch 5's windows, paired with
        # it, change by 10 bpm or more as well; the series reads that window at the new rate
        # and the rule does not fire again.
        for start, bpm in ((135.0, 100.0), (134.0, 85.0)):
            with self.subTest(start=start, bpm=bpm):
                report, counts, paused = self.score((start, 300.0, bpm))
                self.assertEqual(counts, (1, 1, 0, 0))
                self.assertEqual(paused, {4})
                self.assertEqual({event.epoch_index for event in report.events}, paused)

    def test_a_step_at_an_epoch_boundary(self):
        # Up as epoch 4 starts, down as epoch 6 starts: only the previous epoch's windows,
        # paired with the next one's, see each step, on the ECG as in the rule.
        report, counts, paused = self.score((120.0, 180.0, 75.0))
        self.assertEqual(counts, (2, 2, 0, 0))
        self.assertEqual(paused, {4, 6})
        self.assertEqual([(event.epoch_index, event.kind) for event in report.events], [(4, "across"), (6, "across")])

    def test_a_step_the_row_and_the_link_both_see_is_one_change(self):
        # 5 s into epoch 4 the row's own values and the link from epoch 3 both hold the step;
        # the link's change, in the same direction over the same time, is not listed again.
        report, counts, paused = self.score((125.0, 300.0, 75.0))
        self.assertEqual(counts, (1, 1, 0, 0))
        self.assertEqual(paused, {4})
        self.assertEqual([(event.epoch_index, event.kind) for event in report.events], [(4, "row")])

    def test_a_change_between_the_two_thresholds(self):
        # 9.4 bpm up and down: the rule counts changes from 9 bpm, so it fires, and neither
        # fire is false; an ECG jump takes 10.
        _report, counts, paused = self.score((135.0, 195.0, 69.4))
        self.assertEqual(counts, (0, 2, 0, 0))
        self.assertEqual(paused, {4, 6})

    def test_a_pause_in_the_next_epoch_catches_a_late_step(self):
        # 25 s into epoch 4 the ECG window over the step already changes by 10 bpm or more,
        # while the series reads that window at the old rate: the rule sees the step from
        # epoch 5. The H10 loses contact there, so epoch 5 has no ECG change and is not
        # scored (fired 0), and its pause still catches epoch 4's jump.
        report, counts, paused = self.score((145.0, 300.0, 90.0), no_contact=(165.0, 170.0))
        self.assertEqual(paused, {5})
        self.assertEqual(counts, (1, 0, 0, 0))
        self.assertIn("no_contact_seconds", report.limitation_codes)


class GeneratedNightCliTest(unittest.TestCase):
    """analyze-optics-agreement on a generated p1034 night whose H10 session shares its
    scripted beats: replay, the decoder, EpochBuilder, the files."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        config = replace(
            SyntheticNightConfig(),
            preset="p1034",
            scenario="agreement",
            hours=8 * 30 / 3600,
            stages=("N2",) * 8,
            arousals_per_hour=0.0,
            ble_gaps_per_hour=0.0,
        )
        night = generate_night(config, root / "night")
        seconds = float(night.generation["duration_seconds"])
        write_raw_session(
            night.recording_dir,
            seconds,
            np.random.default_rng(3),
            wall0=START_EPOCH_SECONDS,
            beats_s=heartbeat_onsets(night),
        )
        cls.output_dir = root / "report"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cls.exit_code = cli_main(
                ["analyze-optics-agreement", str(night.recording_dir), "--output-dir", str(cls.output_dir)]
            )
        cls.stdout = stdout.getvalue()
        cls.payload = json.loads((cls.output_dir / JSON_FILENAME).read_text(encoding="utf-8"))
        cls.markdown = (cls.output_dir / MARKDOWN_FILENAME).read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_command_writes_both_files_labelled_synthetic(self):
        self.assertEqual(self.exit_code, 0)
        self.assertIn("(synthetic input, not sleep)", self.stdout)
        self.assertTrue(self.payload["synthetic"])
        self.assertEqual(self.payload["note"], "synthetic input, not sleep")
        self.assertIn("Synthetic input, not sleep.", self.markdown)
        codes = [item["code"] for item in self.payload["limitations"]]
        self.assertEqual(codes[:2], ["synthetic_input", "n_epochs_small"])
        self.assertEqual(self.payload["metadata"]["preset"], "p1034")

    def test_every_estimator_of_the_replayed_night_is_compared(self):
        report = OpticsAgreementReport.from_dict(self.payload)
        self.assertEqual(report.counts["compared_epochs"], 8)
        for estimator in ("decoder_heart_rate", "ppg_estimated_hr_bpm", "hr_series"):
            with self.subTest(estimator=estimator):
                row = report.overall(estimator)
                self.assertGreater(row.n, 0)
                self.assertLess(abs(row.bias_bpm), 2.0)
        channels = {row.channel for row in report.rows("channel")}
        self.assertEqual(channels, set(OPTICS_CHANNELS))

    def test_the_report_holds_no_device_name(self):
        self.assertNotIn("Synthetic Muse S Athena", json.dumps(self.payload))
        self.assertNotIn("synthetic-agreement-0", json.dumps(self.payload))


class InputErrorTest(unittest.TestCase):
    def _run(self, recording_dir, *extra):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = cli_main(["analyze-optics-agreement", str(recording_dir), *extra])
        return code, stderr.getvalue()

    @staticmethod
    def _metadata(directory, preset):
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"source": {"metadata": {"preset": preset}}}
        (directory / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_a_recording_without_polar_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._metadata(Path(tmp) / "rec", "p1034")
            code, message = self._run(Path(tmp) / "rec", "--output-dir", str(Path(tmp) / "out"))
            self.assertFalse((Path(tmp) / "out").exists())
        self.assertEqual(code, 2)
        self.assertIn("no polar/ folder", message)

    def test_a_p21_recording_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._metadata(Path(tmp) / "rec", "p21")
            code, message = self._run(Path(tmp) / "rec", "--output-dir", str(Path(tmp) / "out"))
        self.assertEqual(code, 2)
        self.assertIn("p21, which streams no optics", message)

    def test_an_epoch_shorter_than_a_series_window_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, message = self._run(Path(tmp), "--epoch-seconds", "5")
        self.assertEqual(code, 2)
        self.assertIn("epoch_seconds must hold one heart-rate series window", message)

    def test_epochs_without_optics_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = write_raw_session(tmp, 60.0, np.random.default_rng(1))
            session = load_polar_session(Path(tmp))
        frames = [
            MuseFrame(
                timestamp=truth["wall0"] + second,
                eeg=EEGSample(truth["wall0"] + second, {"AF7": [0.0] * 256}, source="test"),
                source="test",
            )
            for second in range(60)
        ]
        with self.assertRaisesRegex(OpticsAgreementInputError, "no optics"):
            optics_agreement_from_epochs(epochs_of(frames), session)

    def test_a_polar_session_without_ecg_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_raw_session(tmp, 60.0, np.random.default_rng(2), with_ecg=False)
            session = load_polar_session(Path(tmp))
        with self.assertRaisesRegex(OpticsAgreementInputError, "no ECG R-peaks"):
            optics_agreement_from_epochs([], session)


class ImportBoundaryTest(unittest.TestCase):
    PATH = REPO_ROOT / "src/muse_tmr/reports/optics_agreement.py"
    FORBIDDEN = ("muse_tmr.audio", "muse_tmr.models", "muse_tmr.protocol")

    def test_the_report_imports_no_audio_models_or_protocol(self):
        imported = set()
        for node in ast.walk(ast.parse(self.PATH.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for name in imported:
            for package in self.FORBIDDEN:
                self.assertFalse(name == package or name.startswith(package + "."), name)

    def test_nothing_it_imports_loads_them_either(self):
        # Loaded on its own, without muse_tmr.reports.__init__, which imports the analysis.
        script = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('optics_agreement_alone', {str(self.PATH)!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules['optics_agreement_alone'] = module\n"
            "spec.loader.exec_module(module)\n"
            f"print(sorted(n for n in sys.modules if n.startswith({self.FORBIDDEN!r})))\n"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(REPO_ROOT / "src"), str(REPO_ROOT), environment.get("PYTHONPATH", "")])
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=environment, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
