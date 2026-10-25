import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from muse_tmr.data.sample_types import HeartRateSample, MuseFrame, PPGSample
from muse_tmr.features.epochs import EpochBuilder, EpochConfig, SleepEpoch
from muse_tmr.features.ppg_features import (
    HeartRatePoint,
    PPGFeatureConfig,
    _series_hr_changes,
    export_ppg_feature_rows,
    extract_ppg_feature_rows,
    extract_ppg_features,
    heart_rate_series,
)
from muse_tmr.testing.synthetic import band_limited_noise, beat_volume


SAMPLE_RATE_HZ = 64


def synthetic_ppg(
    heart_rate_bpm: float,
    *,
    seconds: float = 30.0,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    baseline: float = 50000.0,
    amplitude: float = 1000.0,
):
    timestamps = np.arange(0, seconds, 1 / sample_rate_hz)
    heart_hz = heart_rate_bpm / 60.0
    pulse = np.sin(2 * np.pi * heart_hz * timestamps)
    harmonic = 0.25 * np.sin(4 * np.pi * heart_hz * timestamps - np.pi / 4)
    respiratory = 0.05 * np.sin(2 * np.pi * 0.25 * timestamps)
    return (baseline + amplitude * (pulse + harmonic + respiratory)).tolist()


def ppg_hr_epoch(
    *,
    ppg_channels=None,
    hr_values=None,
    epoch_index: int = 0,
) -> SleepEpoch:
    frames = []
    if ppg_channels is not None:
        frames.append(
            MuseFrame(
                timestamp=1000.0,
                ppg=PPGSample(timestamp=1000.0, channels=ppg_channels, source="test"),
                source="test",
            )
        )
    if hr_values is not None:
        for index, bpm in enumerate(hr_values):
            frames.append(
                MuseFrame(
                    timestamp=1000.0 + index,
                    heart_rate=HeartRateSample(
                        timestamp=1000.0 + index,
                        bpm=bpm,
                        source="test",
                    ),
                    source="test",
                )
            )
    return EpochBuilder(EpochConfig(epoch_seconds=30, stride_seconds=30))._build_epoch(
        epoch_index,
        1000.0,
        tuple(frames),
    )


def pulse_light(
    heart_rate_bpm,
    *,
    seconds: float = 30.0,
    seed: int = 0,
    missed_beats=(),
    noise=None,
    dc: float = 9.0,
    pulse_share: float = 0.006,
):
    """Optics light like LO_NIR on a synthetic night: a pulse of the generator's beat
    shape at ``heart_rate_bpm`` (bpm, or a function of seconds), 1.5 bpm of sinus
    arrhythmia, wander between 0.1 and 0.5 Hz at 0.85 of the pulse and sensor noise.
    ``missed_beats`` leaves those beats out; ``noise=(start, stop, scale)`` adds white
    noise of ``scale`` pulse heights. Synthetic input, not sleep."""
    rng = np.random.default_rng(seed)
    times = np.arange(int(seconds * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
    rate = heart_rate_bpm(times) if callable(heart_rate_bpm) else np.full(times.size, float(heart_rate_bpm))
    rate = rate + 1.5 * np.sin(2 * np.pi * 0.25 * times + rng.uniform(0, 2 * np.pi))
    cycles = np.cumsum(rate / 60.0 / SAMPLE_RATE_HZ)
    pulse = beat_volume(cycles)
    for beat in missed_beats:
        pulse[np.floor(cycles) == beat] = 0.0
    spectrum = np.fft.rfft(pulse - np.mean(pulse))
    freqs = np.fft.rfftfreq(pulse.size, 1.0 / SAMPLE_RATE_HZ)
    cardiac_rms = np.sqrt(2.0 * np.sum(np.abs(spectrum[(freqs >= 0.7) & (freqs <= 3.0)]) ** 2)) / pulse.size
    wander = 0.85 * cardiac_rms * band_limited_noise(times.size, 0.1, 0.5, rng, fs=SAMPLE_RATE_HZ)
    light = dc * (1.0 + pulse_share * (wander - pulse))
    light += 0.00006 * dc * rng.standard_normal(times.size)
    if noise is not None:
        start, stop, scale = noise
        span = slice(int(start * SAMPLE_RATE_HZ), int(stop * SAMPLE_RATE_HZ))
        light[span] += scale * pulse_share * dc * rng.standard_normal(span.stop - span.start)
    return light


def optics_epoch(channels, *, rows_per_frame=None, lost_frames=(), start=1000.0):
    """An epoch of optics frames; ``rows_per_frame`` splits the channels into frames as
    notifications do, and ``lost_frames`` drops those frames and reports their samples
    lost in the next one's ``gaps``, as the decoder does."""
    length = len(next(iter(channels.values())))
    rows = rows_per_frame or length
    frames = []
    lost = 0
    for number, first in enumerate(range(0, length, rows)):
        span = slice(first, min(length, first + rows))
        if number in lost_frames:
            lost += span.stop - span.start
            continue
        timestamp = start + first / SAMPLE_RATE_HZ
        sample = PPGSample(
            timestamp,
            {name: list(values[span]) for name, values in channels.items()},
            source="test",
            gaps=((0, lost),) if lost else (),
        )
        frames.append(MuseFrame(timestamp=timestamp, ppg=sample, source="test"))
        lost = 0
    return EpochBuilder(EpochConfig(epoch_seconds=30, stride_seconds=30))._build_epoch(
        0, start, tuple(frames)
    )


def held_step(onset_seconds, before_bpm, after_bpm):
    return lambda times: np.where(times >= onset_seconds, after_bpm, before_bpm)


def surge(onset_seconds, seconds, before_bpm, delta_bpm):
    """A rate that jumps by ``delta_bpm`` at ``onset_seconds`` and back ``seconds`` later."""
    return lambda times: np.where(
        (times >= onset_seconds) & (times < onset_seconds + seconds),
        before_bpm + delta_bpm,
        before_bpm,
    )


class TestHeartRateSeries(unittest.TestCase):
    """The series the sudden heart-rate change rule reads (#247). Synthetic input, not sleep."""

    def test_steady_pulse_gives_a_used_value_every_second(self):
        row = extract_ppg_features(optics_epoch({"LO_NIR": pulse_light(60.0)}))

        self.assertEqual(row.sudden_hr_change_source, "optics")
        self.assertEqual(row.hr_series_channel, "LO_NIR")
        self.assertEqual(len(row.hr_series), 21)
        self.assertEqual(len(row.hr_series_used), 21)
        times = [point.time for point in row.hr_series]
        self.assertAlmostEqual(times[0], 1005.0)
        np.testing.assert_allclose(np.diff(times), 1.0)
        for point in row.hr_series:
            self.assertAlmostEqual(point.bpm, 60.0, delta=1.0)
            self.assertAlmostEqual(point.peak_bpm, 60.0, delta=2.0)
            self.assertGreater(point.quality, 0.8)
        self.assertEqual(row.sudden_hr_change_count, 0)
        self.assertTrue(math.isnan(row.max_sudden_hr_change_bpm))

    def test_missed_beats_do_not_fire(self):
        for seed in range(3):
            for missed in ((12,), (9, 21)):
                with self.subTest(seed=seed, missed=missed):
                    light = pulse_light(60.0, seed=seed, missed_beats=missed)
                    row = extract_ppg_features(optics_epoch({"LO_NIR": light}))

                    self.assertEqual(row.sudden_hr_change_count, 0)
                    self.assertEqual(row.sudden_hr_changes, ())
                    for point in row.hr_series_used:
                        self.assertAlmostEqual(point.bpm, 60.0, delta=2.0)

    def test_held_steps_of_15_and_40_bpm_fire(self):
        for before, after in ((60.0, 75.0), (60.0, 100.0), (75.0, 60.0)):
            with self.subTest(before=before, after=after):
                light = pulse_light(held_step(15.0, before, after), seed=2)
                row = extract_ppg_features(optics_epoch({"LO_NIR": light}))

                self.assertEqual(row.sudden_hr_change_count, 1)
                change = row.sudden_hr_changes[0]
                self.assertAlmostEqual(change.delta_bpm, after - before, delta=1.5)
                self.assertAlmostEqual(row.max_sudden_hr_change_bpm, abs(after - before), delta=1.5)
                self.assertLess(change.start_time, 1015.0)
                self.assertGreater(change.end_time, 1015.0)

    def test_short_surges_of_25_to_40_bpm_fire(self):
        # No window holds only the surge, and one that holds both rates peaks at neither
        # period; its run of regular fast beats gives the value instead.
        for delta, seconds in ((25.0, 5.0), (30.0, 6.0), (40.0, 5.0), (40.0, 7.0)):
            with self.subTest(delta=delta, seconds=seconds):
                light = pulse_light(surge(10.0, seconds, 60.0, delta), seed=4)
                row = extract_ppg_features(optics_epoch({"LO_NIR": light}))

                fast = [
                    point
                    for point in row.hr_series_used
                    if point.run_seconds > 0 and abs(point.bpm - (60.0 + delta)) < 3.0
                ]
                self.assertTrue(fast)
                for point in fast:
                    self.assertGreaterEqual(point.run_seconds, 4.0)
                    self.assertLess(point.run_seconds, seconds + 0.5)
                self.assertGreaterEqual(row.sudden_hr_change_count, 1)
                self.assertGreaterEqual(row.max_sudden_hr_change_bpm, delta - 3.0)
                logged = json.loads(row.to_dict()["hr_series_json"])
                self.assertTrue(any(point["run_seconds"] > 0 for point in logged))

    def test_changes_are_logged_from_the_bpm_less_the_margin(self):
        points = tuple(
            HeartRatePoint(time=1005.0 + second, bpm=bpm, peak_bpm=bpm, quality=0.9, used=True)
            for second, bpm in enumerate([60.0] * 10 + [69.5] * 11)
        )

        changes = _series_hr_changes(points, PPGFeatureConfig())
        exact = _series_hr_changes(points, PPGFeatureConfig(hr_series_margin_bpm=0.0))

        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(changes[0].delta_bpm, 9.5)
        self.assertEqual(exact, ())

    def test_low_quality_values_are_skipped_and_logged(self):
        burst = pulse_light(60.0, noise=(10.0, 18.0, 5.0))
        row = extract_ppg_features(optics_epoch({"LO_NIR": burst}))

        skipped = [point for point in row.hr_series if not point.used]
        self.assertGreaterEqual(row.hr_series_skipped_count, 10)
        self.assertEqual(len(skipped), row.hr_series_skipped_count)
        for point in skipped:
            # Windows that reach the burst, or come within the 2 s the zero-phase
            # band-pass spreads it by.
            self.assertLess(point.quality, 0.5)
            self.assertLess(point.time - 5.0, 1018.0 + 2.0)
            self.assertGreater(point.time + 5.0, 1010.0 - 2.0)
        self.assertEqual(row.sudden_hr_change_count, 0)
        logged = json.loads(row.to_dict()["hr_series_json"])
        self.assertEqual(sum(1 for point in logged if not point["used"]), row.hr_series_skipped_count)
        self.assertEqual(row.to_dict()["hr_series_skipped_count"], row.hr_series_skipped_count)

        noise_only = extract_ppg_features(
            optics_epoch({"LO_NIR": pulse_light(60.0, noise=(0.0, 30.0, 20.0))})
        )
        self.assertEqual(noise_only.hr_series_used, ())
        self.assertEqual(noise_only.hr_series_skipped_count, len(noise_only.hr_series))
        self.assertEqual(noise_only.sudden_hr_change_count, 0)

    def test_the_most_periodic_channel_feeds_the_series(self):
        rng = np.random.default_rng(5)
        channels = {
            "A_noise": 9.0 + 0.05 * rng.standard_normal(1920),
            "B_pulse": pulse_light(72.0, seed=5),
        }

        series = heart_rate_series(optics_epoch(channels))

        self.assertEqual(series.channel, "B_pulse")
        self.assertEqual(len(series.used_points), 21)
        self.assertAlmostEqual(float(np.median([p.bpm for p in series.used_points])), 72.0, delta=1.0)

    def test_lost_samples_keep_the_series_on_the_clock(self):
        light = pulse_light(60.0, seed=1)
        # Notifications of two rows; 32 of them (1 s) lost in a row, reported in the next one.
        row = extract_ppg_features(
            optics_epoch({"LO_NIR": light}, rows_per_frame=2, lost_frames=range(400, 432))
        )

        self.assertEqual(len(row.hr_series), 21)
        self.assertAlmostEqual(row.hr_series[-1].time, 1025.0)
        self.assertEqual(row.sudden_hr_change_count, 0)
        for point in row.hr_series_used:
            self.assertAlmostEqual(point.bpm, 60.0, delta=2.0)

    def test_a_late_frame_without_reported_loss_keeps_the_clock(self):
        light = pulse_light(60.0, seed=1)
        first = optics_epoch({"LO_NIR": light[:960]})
        later = optics_epoch({"LO_NIR": light[1088:]}, start=1017.0)
        epoch = EpochBuilder(EpochConfig())._build_epoch(0, 1000.0, first.frames + later.frames)

        series = heart_rate_series(epoch)

        self.assertEqual(len(series.points), 21)
        self.assertAlmostEqual(series.points[-1].time, 1025.0)

    def test_heart_rate_samples_without_optics_keep_the_consecutive_comparison(self):
        row = extract_ppg_features(ppg_hr_epoch(hr_values=[60.0, 61.0, 80.0, 79.0, 55.0]))

        self.assertEqual(row.sudden_hr_change_source, "heart_rate")
        self.assertEqual(row.hr_series, ())
        self.assertEqual(row.sudden_hr_change_count, 2)

    def test_optics_feed_the_rule_even_with_a_jumping_heart_rate_stream(self):
        # The decoder's stream jumps by 19 and 24 bpm; the optics hold a steady pulse.
        epoch = ppg_hr_epoch(
            ppg_channels={"LO_NIR": pulse_light(60.0, seed=3).tolist()},
            hr_values=[60.0, 61.0, 80.0, 79.0, 55.0],
        )

        row = extract_ppg_features(epoch)

        self.assertEqual(row.sudden_hr_change_source, "optics")
        self.assertEqual(row.sudden_hr_change_count, 0)
        self.assertEqual(row.hr_source, "heart_rate")

    def test_no_optics_and_no_heart_rate(self):
        row = extract_ppg_features(ppg_hr_epoch())

        self.assertEqual(row.sudden_hr_change_source, "missing")
        self.assertEqual(row.sudden_hr_change_count, 0)

    def test_config_rejects_bad_series_settings(self):
        for bad in (
            {"hr_series_window_seconds": 3.0},
            {"hr_series_step_seconds": 0.0},
            {"hr_series_compare_seconds": 0.5},
            {"hr_series_max_compare_seconds": 5.0},
            {"hr_series_min_quality": 1.5},
            {"hr_series_max_disagreement_bpm": 0.0},
            {"hr_series_max_missing_fraction": 1.0},
            {"hr_series_peak_prominence": 0.0},
            {"hr_series_min_run_seconds": 0.0},
            {"hr_series_min_run_seconds": 11.0},
            {"hr_series_margin_bpm": -1.0},
            {"hr_series_margin_bpm": 10.0},
        ):
            with self.subTest(**bad), self.assertRaises(ValueError):
                PPGFeatureConfig(**bad).validate()


class TestPPGFeatures(unittest.TestCase):
    def test_ppg_signal_estimates_heart_rate_when_hr_series_missing(self):
        epoch = ppg_hr_epoch(
            ppg_channels={
                "LO_NIR": synthetic_ppg(72.0),
                "RO_NIR": synthetic_ppg(72.0, amplitude=500.0),
            }
        )

        row = extract_ppg_features(epoch)

        self.assertAlmostEqual(row.ppg_estimated_hr_bpm, 72.0, delta=4.0)
        self.assertAlmostEqual(row.mean_hr_bpm, row.ppg_estimated_hr_bpm)
        self.assertEqual(row.hr_source, "ppg")
        self.assertEqual(row.hrv_source, "ppg")
        self.assertGreater(row.ppg_peak_count, 20)

    def test_heart_rate_summary_and_trend_use_hr_samples(self):
        hr_values = [60.0 + index for index in range(30)]
        epoch = ppg_hr_epoch(
            ppg_channels={"LO_NIR": synthetic_ppg(70.0)},
            hr_values=hr_values,
        )

        row = extract_ppg_features(epoch)

        self.assertAlmostEqual(row.mean_hr_bpm, sum(hr_values) / len(hr_values))
        self.assertEqual(row.hr_source, "heart_rate")
        self.assertGreater(row.hr_trend_bpm_per_min, 50.0)
        self.assertEqual(row.min_hr_bpm, 60.0)
        self.assertEqual(row.max_hr_bpm, 89.0)

    def test_hrv_proxy_uses_variable_hr_when_ppg_peaks_unavailable(self):
        hr_values = [60.0, 62.0, 58.0, 63.0, 59.0, 61.0] * 5
        epoch = ppg_hr_epoch(hr_values=hr_values)

        row = extract_ppg_features(epoch)

        self.assertEqual(row.hrv_source, "heart_rate")
        self.assertGreater(row.rmssd_ms, 0.0)
        self.assertGreaterEqual(row.pnn50_percent, 0.0)
        self.assertIn("ppg_missing", row.artifact_flags)

    def test_sudden_heart_rate_changes_are_logged(self):
        epoch = ppg_hr_epoch(hr_values=[60.0, 61.0, 80.0, 79.0, 55.0])

        row = extract_ppg_features(epoch, PPGFeatureConfig(sudden_hr_change_bpm=10.0))

        self.assertEqual(row.sudden_hr_change_count, 2)
        self.assertEqual(row.max_sudden_hr_change_bpm, 24.0)
        self.assertEqual(row.sudden_hr_changes[0].before_bpm, 61.0)
        self.assertEqual(row.sudden_hr_changes[0].after_bpm, 80.0)

    def test_missing_ppg_with_hr_does_not_crash(self):
        epoch = ppg_hr_epoch(hr_values=[62.0] * 30)

        row = extract_ppg_features(epoch)

        self.assertEqual(row.ppg_sample_count, 0)
        self.assertEqual(row.hr_source, "heart_rate")
        self.assertAlmostEqual(row.mean_hr_bpm, 62.0)
        self.assertIn("ppg_missing", row.artifact_flags)

    def test_missing_ppg_and_hr_returns_flags_without_crashing(self):
        epoch = SleepEpoch(
            index=0,
            start_time=1000.0,
            end_time=1030.0,
            frames=(),
            modality_counts={},
            sample_counts={},
            coverage={"ppg": 0.0, "heart_rate": 0.0},
            quality_flags=("missing_ppg", "missing_heart_rate"),
        )

        row = extract_ppg_features(epoch)

        self.assertTrue(math.isnan(row.mean_hr_bpm))
        self.assertEqual(row.hr_source, "missing")
        self.assertIn("ppg_missing", row.artifact_flags)
        self.assertIn("heart_rate_missing", row.artifact_flags)

    def test_nonfinite_ppg_and_out_of_range_hr_are_flagged(self):
        ppg_values = synthetic_ppg(72.0)
        ppg_values[10] = math.nan
        epoch = ppg_hr_epoch(
            ppg_channels={"LO_NIR": ppg_values},
            hr_values=[20.0, 60.0, math.nan, 300.0],
        )

        row = extract_ppg_features(epoch)

        self.assertIn("ppg_nonfinite_LO_NIR", row.artifact_flags)
        self.assertIn("heart_rate_nonfinite", row.artifact_flags)
        self.assertIn("heart_rate_out_of_range", row.artifact_flags)

    def test_feature_rows_export_to_csv(self):
        rows = extract_ppg_feature_rows([
            ppg_hr_epoch(
                ppg_channels={"LO_NIR": synthetic_ppg(72.0)},
                hr_values=[72.0] * 30,
            )
        ])

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "ppg_features.csv"
            export_ppg_feature_rows(rows, output_path)
            frame = pd.read_csv(output_path)

        self.assertEqual(len(frame), 1)
        self.assertIn("mean_hr_bpm", frame.columns)
        self.assertIn("rmssd_ms", frame.columns)
        self.assertIn("sudden_hr_changes_json", frame.columns)
        # The rule's columns keep their names (#247), next to what fed it.
        for column in (
            "sudden_hr_change_count",
            "max_sudden_hr_change_bpm",
            "sudden_hr_change_source",
            "hr_series_channel",
            "hr_series_used_count",
            "hr_series_skipped_count",
            "hr_series_median_quality",
            "hr_series_json",
        ):
            self.assertIn(column, frame.columns)
        self.assertEqual(frame.loc[0, "sudden_hr_change_source"], "optics")

    @unittest.skipIf(
        importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None,
        "pandas parquet engine is not installed",
    )
    def test_feature_rows_export_to_parquet_when_engine_available(self):
        rows = extract_ppg_feature_rows([
            ppg_hr_epoch(ppg_channels={"LO_NIR": synthetic_ppg(72.0)})
        ])

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "ppg_features.parquet"
            export_ppg_feature_rows(rows, output_path)
            frame = pd.read_parquet(output_path)

        self.assertEqual(len(frame), 1)
        self.assertIn("ppg_estimated_hr_bpm", frame.columns)


if __name__ == "__main__":
    unittest.main()
