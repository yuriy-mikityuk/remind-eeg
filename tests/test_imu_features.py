import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from muse_tmr.data.sample_types import IMUSample, MuseFrame
from muse_tmr.features.epochs import EpochBuilder, EpochConfig, SleepEpoch
from muse_tmr.features.imu_features import (
    IMUFeatureConfig,
    export_imu_feature_rows,
    extract_imu_feature_rows,
    extract_imu_features,
)
from muse_tmr.protocol.tmr_scheduler import arousal_guard_decision


SAMPLE_RATE_HZ = 52


def still_accel_rows(seconds: float = 30.0):
    return [{"x": 0.0, "y": 0.0, "z": 1.0}] * int(seconds * SAMPLE_RATE_HZ)


def still_gyro_rows(seconds: float = 30.0):
    return [{"x": 0.0, "y": 0.0, "z": 0.0}] * int(seconds * SAMPLE_RATE_HZ)


def imu_epoch(
    accel_rows=None,
    gyro_rows=None,
    *,
    epoch_index: int = 0,
) -> SleepEpoch:
    frame = MuseFrame(
        timestamp=1000.0,
        imu=IMUSample(
            timestamp=1000.0,
            accelerometer_g=accel_rows,
            gyroscope_dps=gyro_rows,
            source="test",
        ),
        source="test",
    )
    return EpochBuilder(EpochConfig(epoch_seconds=30, stride_seconds=30))._build_epoch(
        epoch_index,
        1000.0,
        (frame,),
    )


class TestIMUFeatures(unittest.TestCase):
    def test_still_epoch_has_low_motion_and_high_stillness(self):
        epoch = imu_epoch(still_accel_rows(), still_gyro_rows())

        row = extract_imu_features(epoch)

        self.assertAlmostEqual(row.motion_level, 0.0)
        self.assertAlmostEqual(row.stillness_score, 1.0)
        self.assertEqual(row.movement_event_count, 0)
        self.assertEqual(row.arousal_event_count, 0)
        self.assertEqual(row.arousal_guard_reason_codes, ())

    def test_accel_burst_creates_movement_and_arousal_proxy(self):
        accel_rows = still_accel_rows()
        for index in range(10 * SAMPLE_RATE_HZ, 11 * SAMPLE_RATE_HZ):
            accel_rows[index] = {"x": 0.0, "y": 0.0, "z": 1.35}
        epoch = imu_epoch(accel_rows, still_gyro_rows())

        row = extract_imu_features(epoch)

        self.assertEqual(row.movement_event_count, 1)
        self.assertEqual(row.arousal_event_count, 1)
        self.assertGreater(row.accel_peak_delta_g, 0.30)
        self.assertLess(row.stillness_score, 1.0)
        self.assertIn("motion_arousal_proxy", row.arousal_guard_reason_codes)

    def test_separated_gyro_bursts_create_multiple_events(self):
        gyro_rows = still_gyro_rows()
        for index in range(5 * SAMPLE_RATE_HZ, 6 * SAMPLE_RATE_HZ):
            gyro_rows[index] = {"x": 20.0, "y": 0.0, "z": 0.0}
        for index in range(8 * SAMPLE_RATE_HZ, 9 * SAMPLE_RATE_HZ):
            gyro_rows[index] = {"x": 80.0, "y": 0.0, "z": 0.0}
        epoch = imu_epoch(still_accel_rows(), gyro_rows)

        row = extract_imu_features(epoch)

        self.assertEqual(row.movement_event_count, 2)
        self.assertEqual(row.arousal_event_count, 1)
        self.assertGreater(row.gyro_peak_dps, 70.0)

    def test_rows_after_a_loss_inside_the_frame_keep_their_time(self):
        # Two ACCGYRO subpackets in one notification, one lost before it and one between them.
        moving = {"x": 80.0, "y": 0.0, "z": 0.0}
        frame = MuseFrame(
            timestamp=1000.0,
            imu=IMUSample(
                timestamp=1000.0,
                accelerometer_g=still_accel_rows()[:6],
                gyroscope_dps=still_gyro_rows()[:3] + [moving] * 3,
                source="test",
                gaps=((0, 3), (3, 3)),
            ),
            source="test",
        )
        epoch = EpochBuilder(EpochConfig())._build_epoch(0, 1000.0, (frame,))

        event = extract_imu_features(epoch).movement_events[0]

        # The moving rows 3 to 5 are samples 6 to 8 after the frame's first row.
        self.assertAlmostEqual(event.start_time, 1000.0 + 6 / SAMPLE_RATE_HZ)
        self.assertAlmostEqual(event.end_time, 1000.0 + 9 / SAMPLE_RATE_HZ)
        self.assertEqual(event.sample_count, 3)

    def test_cue_related_movement_logs_are_windowed(self):
        accel_rows = still_accel_rows()
        for index in range(10 * SAMPLE_RATE_HZ, 11 * SAMPLE_RATE_HZ):
            accel_rows[index] = {"x": 0.0, "y": 0.0, "z": 1.30}
        epoch = imu_epoch(accel_rows, still_gyro_rows())

        row = extract_imu_features(epoch, cue_timestamps=[1010.0, 1025.0])

        self.assertEqual(len(row.cue_movement_logs), 2)
        self.assertEqual(row.cue_movement_logs[0].movement_event_count, 1)
        self.assertEqual(row.cue_movement_logs[0].arousal_event_count, 1)
        self.assertEqual(row.cue_movement_logs[1].movement_event_count, 0)
        self.assertEqual(row.cue_related_movement_count, 1)

    def test_missing_imu_flags_and_blocks_guard(self):
        epoch = SleepEpoch(
            index=0,
            start_time=1000.0,
            end_time=1030.0,
            frames=(),
            modality_counts={},
            sample_counts={},
            coverage={"imu": 0.0},
            quality_flags=("missing_imu",),
        )

        row = extract_imu_features(epoch)
        decision = arousal_guard_decision(row.arousal_guard_reason_codes)

        self.assertEqual(row.sample_count, 0)
        self.assertTrue(math.isnan(row.motion_level))
        self.assertIn("imu_missing", row.artifact_flags)
        self.assertFalse(decision.should_play)
        self.assertEqual(decision.reason_codes, ("imu_missing",))

    def test_low_imu_coverage_is_flagged_for_guard(self):
        epoch = imu_epoch(
            still_accel_rows(seconds=2.0),
            still_gyro_rows(seconds=2.0),
        )

        row = extract_imu_features(epoch)

        self.assertLess(row.imu_coverage, 0.5)
        self.assertIn("low_imu_coverage", row.artifact_flags)
        self.assertIn("low_imu_coverage", row.arousal_guard_reason_codes)

    def test_nonfinite_accelerometer_rows_are_flagged(self):
        accel_rows = still_accel_rows()
        accel_rows[0] = {"x": math.nan, "y": 0.0, "z": 1.0}
        epoch = imu_epoch(accel_rows, still_gyro_rows())

        row = extract_imu_features(epoch)

        self.assertIn("imu_accelerometer_nonfinite", row.artifact_flags)
        self.assertEqual(row.movement_event_count, 0)

    def test_feature_rows_export_to_csv(self):
        row = extract_imu_features(imu_epoch(still_accel_rows(), still_gyro_rows()))

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "imu_features.csv"
            export_imu_feature_rows([row], output_path)
            frame = pd.read_csv(output_path)

        self.assertEqual(len(frame), 1)
        self.assertIn("motion_level", frame.columns)
        self.assertIn("cue_movement_logs_json", frame.columns)

    @unittest.skipIf(
        importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None,
        "pandas parquet engine is not installed",
    )
    def test_feature_rows_export_to_parquet_when_engine_available(self):
        rows = extract_imu_feature_rows([imu_epoch(still_accel_rows(), still_gyro_rows())])

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "imu_features.parquet"
            export_imu_feature_rows(rows, output_path)
            frame = pd.read_parquet(output_path)

        self.assertEqual(len(frame), 1)
        self.assertIn("stillness_score", frame.columns)


if __name__ == "__main__":
    unittest.main()
