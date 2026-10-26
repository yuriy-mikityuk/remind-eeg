import dataclasses
import unittest

import numpy as np

from muse_tmr.data.sample_types import EEGSample, HeartRateSample, IMUSample, MuseFrame, PPGSample
from muse_tmr.features.eeg_features import extract_eeg_features
from muse_tmr.features.epochs import EpochBuilder, EpochConfig
from muse_tmr.features.imu_features import extract_imu_features
from muse_tmr.features.ppg_features import extract_ppg_features
from muse_tmr.models import HeuristicRemConfig, HeuristicRemDetector, RemPrediction
from muse_tmr.models.heuristic_rem_detector import DEFAULT_REM_WEIGHTS
from tests.rem_event_synthetic import rem_deflection_train


EEG_RATE_HZ = 256
IMU_RATE_HZ = 52
PPG_RATE_HZ = 64


async def async_frames(frames):
    for frame in frames:
        yield frame


def sine_values(frequency_hz, *, seconds=30.0, sample_rate_hz=EEG_RATE_HZ, amplitude=10.0):
    timestamps = np.arange(0, seconds, 1 / sample_rate_hz)
    return amplitude * np.sin(2 * np.pi * frequency_hz * timestamps)


def synthetic_ppg(heart_rate_bpm, *, seconds=30.0):
    timestamps = np.arange(0, seconds, 1 / PPG_RATE_HZ)
    heart_hz = heart_rate_bpm / 60.0
    pulse = np.sin(2 * np.pi * heart_hz * timestamps)
    harmonic = 0.25 * np.sin(4 * np.pi * heart_hz * timestamps - np.pi / 4)
    return (50000.0 + 1000.0 * (pulse + harmonic)).tolist()


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


def nrem_like_eeg_channels():
    delta = sine_values(2.0, amplitude=20.0)
    alpha = sine_values(10.0, amplitude=1.0)
    return {
        "TP9": (delta + alpha).tolist(),
        "AF7": (delta + alpha).tolist(),
        "AF8": (delta + alpha).tolist(),
        "TP10": (delta + alpha).tolist(),
    }


def still_accel_rows():
    return [{"x": 0.0, "y": 0.0, "z": 1.0}] * (30 * IMU_RATE_HZ)


def still_gyro_rows():
    return [{"x": 0.0, "y": 0.0, "z": 0.0}] * (30 * IMU_RATE_HZ)


def motion_gyro_rows():
    rows = still_gyro_rows()
    for index in range(10 * IMU_RATE_HZ, 11 * IMU_RATE_HZ):
        rows[index] = {"x": 90.0, "y": 0.0, "z": 0.0}
    return rows


def build_epoch(
    *,
    eeg_channels=None,
    accel_rows=None,
    gyro_rows=None,
    ppg_channels=None,
    hr_values=None,
):
    frames = []
    sensor_kwargs = {}
    if eeg_channels is not None:
        sensor_kwargs["eeg"] = EEGSample(1000.0, eeg_channels, source="test")
    if accel_rows is not None or gyro_rows is not None:
        sensor_kwargs["imu"] = IMUSample(
            1000.0,
            accelerometer_g=accel_rows,
            gyroscope_dps=gyro_rows,
            source="test",
        )
    if ppg_channels is not None:
        sensor_kwargs["ppg"] = PPGSample(1000.0, ppg_channels, source="test")
    if sensor_kwargs:
        frames.append(MuseFrame(timestamp=1000.0, source="test", **sensor_kwargs))
    if hr_values is not None:
        for index, bpm in enumerate(hr_values):
            frames.append(
                MuseFrame(
                    timestamp=1000.0 + index,
                    heart_rate=HeartRateSample(1000.0 + index, bpm=bpm, source="test"),
                    source="test",
                )
            )
    return EpochBuilder(EpochConfig(epoch_seconds=30, stride_seconds=30))._build_epoch(
        0,
        1000.0,
        tuple(frames),
    )


class TestHeuristicRemDetector(unittest.IsolatedAsyncioTestCase):
    def test_rem_like_epoch_scores_high_with_reason_codes(self):
        epoch = build_epoch(
            eeg_channels=rem_like_eeg_channels(),
            accel_rows=still_accel_rows(),
            gyro_rows=still_gyro_rows(),
            ppg_channels={"LO_NIR": synthetic_ppg(72.0)},
            hr_values=[70.0, 72.0, 68.0, 74.0, 69.0, 73.0] * 5,
        )
        detector = HeuristicRemDetector()

        prediction = detector.predict_epoch(epoch)

        self.assertGreater(prediction.probability, 0.65)
        self.assertLessEqual(prediction.probability, 1.0)
        self.assertEqual(prediction.source, "heuristic")
        self.assertIn("eeg_low_delta_support", prediction.reason_codes)
        self.assertIn("eeg_eye_movement_support", prediction.reason_codes)
        self.assertIn("eeg_rem_events_support", prediction.reason_codes)
        self.assertEqual(prediction.feature_values["rem_event_count"], 8.0)
        self.assertIn("imu_stillness_support", prediction.reason_codes)
        self.assertIn("low_delta_power", prediction.feature_scores)

    def test_nrem_like_epoch_scores_lower(self):
        epoch = build_epoch(
            eeg_channels=nrem_like_eeg_channels(),
            accel_rows=still_accel_rows(),
            gyro_rows=still_gyro_rows(),
            hr_values=[58.0] * 30,
        )
        detector = HeuristicRemDetector()

        prediction = detector.predict_epoch(epoch)

        self.assertLess(prediction.probability, 0.45)
        self.assertIn("eeg_delta_power_high", prediction.reason_codes)
        self.assertIn("eeg_eye_movement_low", prediction.reason_codes)
        self.assertIn("eeg_rem_events_low", prediction.reason_codes)
        self.assertEqual(prediction.feature_values["rem_event_count"], 0.0)

    def test_motion_arousal_caps_rem_probability_without_audio_decision(self):
        epoch = build_epoch(
            eeg_channels=rem_like_eeg_channels(),
            accel_rows=still_accel_rows(),
            gyro_rows=motion_gyro_rows(),
            ppg_channels={"LO_NIR": synthetic_ppg(72.0)},
            hr_values=[70.0, 72.0, 68.0, 74.0, 69.0, 73.0] * 5,
        )
        detector = HeuristicRemDetector()

        prediction = detector.predict_epoch(epoch)

        self.assertLessEqual(prediction.probability, detector.config.motion_arousal_probability_cap)
        self.assertIn("motion_arousal_proxy", prediction.reason_codes)
        self.assertFalse(hasattr(prediction, "should_play"))

    def test_missing_modalities_return_zero_probability_without_crash(self):
        epoch = build_epoch()
        detector = HeuristicRemDetector()

        prediction = detector.predict_epoch(epoch)

        self.assertEqual(prediction.probability, 0.0)
        self.assertIn("insufficient_features", prediction.reason_codes)
        self.assertIn("low_eeg_coverage", prediction.reason_codes)
        self.assertIn("low_imu_coverage", prediction.reason_codes)
        self.assertIn("low_ppg_hr_coverage", prediction.reason_codes)

    async def test_detector_accepts_epochs_from_live_or_replay_style_stream(self):
        frame = MuseFrame(
            timestamp=1000.0,
            eeg=EEGSample(1000.0, rem_like_eeg_channels(), source="test"),
            imu=IMUSample(
                1000.0,
                accelerometer_g=still_accel_rows(),
                gyroscope_dps=still_gyro_rows(),
                source="test",
            ),
            ppg=PPGSample(1000.0, {"LO_NIR": synthetic_ppg(72.0)}, source="test"),
            source="test",
        )
        builder = EpochBuilder(EpochConfig(epoch_seconds=30, stride_seconds=30))
        epochs = [epoch async for epoch in builder.build(async_frames([frame]))]
        detector = HeuristicRemDetector()

        predictions = detector.predict_epochs(epochs)

        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], RemPrediction)
        self.assertGreaterEqual(predictions[0].probability, 0.0)
        self.assertLessEqual(predictions[0].probability, 1.0)


class TestHeuristicRemWeights(unittest.TestCase):
    """The saturated eye-movement proxy carries no weight (#225, #116)."""

    def test_weights_sum_to_one_with_eye_proxy_at_zero(self):
        self.assertEqual(DEFAULT_REM_WEIGHTS["eye_movement_proxy"], 0.0)
        self.assertAlmostEqual(sum(DEFAULT_REM_WEIGHTS.values()), 1.0, places=12)
        earlier = {
            "low_delta_power": 0.25,
            "theta_alpha_ratio": 0.15,
            "stillness": 0.20,
            "hr_variability": 0.10,
            "hr_trend": 0.05,
        }
        for name, weight in earlier.items():
            self.assertAlmostEqual(DEFAULT_REM_WEIGHTS[name], weight / 0.75, places=12)
        self.assertEqual(dict(HeuristicRemConfig().weights), DEFAULT_REM_WEIGHTS)

    def test_p_rem_ignores_eye_movement_proxy(self):
        epoch = build_epoch(
            eeg_channels=nrem_like_eeg_channels(),
            accel_rows=still_accel_rows(),
            gyro_rows=still_gyro_rows(),
        )
        eeg = extract_eeg_features(epoch)
        imu = extract_imu_features(epoch)
        detector = HeuristicRemDetector()

        quiet = detector.predict_features(
            eeg=dataclasses.replace(eeg, eye_movement_proxy=0.0), imu=imu
        )
        saturated = detector.predict_features(
            eeg=dataclasses.replace(eeg, eye_movement_proxy=10.0), imu=imu
        )

        self.assertEqual(quiet.feature_scores["eye_movement_proxy"], 0.0)
        self.assertEqual(saturated.feature_scores["eye_movement_proxy"], 1.0)
        self.assertEqual(saturated.feature_values["eye_movement_proxy"], 10.0)
        self.assertAlmostEqual(quiet.probability, saturated.probability, places=12)
        self.assertGreater(quiet.probability, 0.0)
        self.assertLess(quiet.probability, 1.0)

    def test_rem_event_density_is_scored_at_weight_zero(self):
        self.assertEqual(DEFAULT_REM_WEIGHTS["rem_event_density"], 0.0)
        epoch = build_epoch(
            eeg_channels=nrem_like_eeg_channels(),
            accel_rows=still_accel_rows(),
            gyro_rows=still_gyro_rows(),
        )
        eeg = extract_eeg_features(epoch)
        imu = extract_imu_features(epoch)
        detector = HeuristicRemDetector()

        quiet = detector.predict_features(eeg=dataclasses.replace(eeg, rem_density_per_min=0.0), imu=imu)
        dense = detector.predict_features(eeg=dataclasses.replace(eeg, rem_density_per_min=30.0), imu=imu)

        self.assertEqual(quiet.feature_scores["rem_event_density"], 0.0)
        self.assertEqual(dense.feature_scores["rem_event_density"], 1.0)
        self.assertIn("eeg_rem_events_support", dense.reason_codes)
        self.assertAlmostEqual(quiet.probability, dense.probability, places=12)

    def test_p21_eeg_imu_epoch_does_not_trip_limited_feature_support(self):
        epoch = build_epoch(
            eeg_channels=rem_like_eeg_channels(),
            accel_rows=still_accel_rows(),
            gyro_rows=still_gyro_rows(),
        )
        detector = HeuristicRemDetector()

        prediction = detector.predict_epoch(epoch)

        self.assertIn("low_ppg_hr_coverage", prediction.reason_codes)
        self.assertNotIn("limited_feature_support", prediction.reason_codes)
        self.assertNotIn("hr_variability", prediction.feature_scores)
        self.assertGreater(prediction.probability, detector.config.limited_feature_probability_cap)


class TestLowPpgQualityReason(unittest.TestCase):
    """HRV from irregular PPG peaks is flagged, not rescored (#234)."""

    def setUp(self):
        epoch = build_epoch(
            eeg_channels=rem_like_eeg_channels(),
            accel_rows=still_accel_rows(),
            gyro_rows=still_gyro_rows(),
            ppg_channels={"LO_NIR": synthetic_ppg(72.0)},
        )
        self.eeg = extract_eeg_features(epoch)
        self.imu = extract_imu_features(epoch)
        self.ppg = extract_ppg_features(epoch)
        self.detector = HeuristicRemDetector()

    def test_low_confidence_peaks_add_the_reason_and_change_nothing_else(self):
        self.assertEqual(self.ppg.hrv_source, "ppg")
        steady = self.detector.predict_features(
            eeg=self.eeg, imu=self.imu, ppg=dataclasses.replace(self.ppg, ppg_confidence=0.9)
        )
        irregular = self.detector.predict_features(
            eeg=self.eeg, imu=self.imu, ppg=dataclasses.replace(self.ppg, ppg_confidence=0.3)
        )

        self.assertNotIn("low_ppg_quality", steady.reason_codes)
        self.assertIn("low_ppg_quality", irregular.reason_codes)
        self.assertEqual(irregular.probability, steady.probability)
        self.assertEqual(irregular.feature_scores, steady.feature_scores)
        self.assertEqual(irregular.feature_values, steady.feature_values)
        self.assertEqual(
            tuple(code for code in irregular.reason_codes if code != "low_ppg_quality"),
            steady.reason_codes,
        )

    def test_only_hrv_from_ppg_peaks_and_only_below_the_threshold(self):
        at_threshold = dataclasses.replace(self.ppg, ppg_confidence=0.4)
        from_heart_rate = dataclasses.replace(self.ppg, ppg_confidence=0.1, hrv_source="heart_rate")
        for ppg in (at_threshold, from_heart_rate):
            with self.subTest(hrv_source=ppg.hrv_source, confidence=ppg.ppg_confidence):
                prediction = self.detector.predict_features(eeg=self.eeg, imu=self.imu, ppg=ppg)
                self.assertNotIn("low_ppg_quality", prediction.reason_codes)
        stricter = HeuristicRemDetector(HeuristicRemConfig(low_ppg_quality_confidence=0.6))
        self.assertIn(
            "low_ppg_quality",
            stricter.predict_features(eeg=self.eeg, imu=self.imu, ppg=at_threshold).reason_codes,
        )
        with self.assertRaises(ValueError):
            HeuristicRemConfig(low_ppg_quality_confidence=1.5).validate()


class TestRemPrediction(unittest.TestCase):
    def test_probability_must_be_bounded(self):
        with self.assertRaises(ValueError):
            RemPrediction(probability=1.1)

    def test_to_dict_exports_prediction_metadata(self):
        prediction = RemPrediction(
            probability=0.25,
            reason_codes=("limited_feature_support",),
            feature_scores={"stillness": 1.0},
            feature_values={"stillness_score": 0.99},
            source="heuristic",
        )

        payload = prediction.to_dict()

        self.assertEqual(payload["probability"], 0.25)
        self.assertEqual(payload["reason_codes"], ["limited_feature_support"])
        self.assertEqual(payload["source"], "heuristic")

    def test_config_rejects_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            HeuristicRemConfig(rem_delta_relative_power=0.7, nrem_delta_relative_power=0.6).validate()
        with self.assertRaises(ValueError):
            HeuristicRemConfig(rem_event_density_min=6.0, rem_event_density_strong=1.0).validate()


if __name__ == "__main__":
    unittest.main()
