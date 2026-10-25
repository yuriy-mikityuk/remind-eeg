"""The sudden heart-rate change rule from BLE notifications to the guard (#247).

A generated p1034 night goes through replay, the decoder, EpochBuilder,
extract_ppg_features and ArousalGuard, with heart-rate steps scripted into it.
Synthetic input, not sleep.
"""

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from muse_tmr.data.replay import ReplayConfig, ReplaySession
from muse_tmr.features import EpochBuilder, EpochConfig
from muse_tmr.features.ppg_features import (
    PPGFeatureConfig,
    _collect_heart_rate_series,
    _sudden_hr_changes,
    extract_ppg_features,
)
from muse_tmr.protocol import ArousalGuard
from muse_tmr.testing import SyntheticNightConfig, generate_night

# 13 min of N2 without arousals or BLE gaps. The heart rate (58 bpm in N2) steps up by
# 15 bpm 15 s into epoch 4 and back down 15 s into epoch 6, and up by 40 bpm at the
# boundary of epochs 9 and 10 and back down at that of epochs 11 and 12. Then three
# short surges: 40 bpm for 5 s from 8 s into epoch 15, 40 bpm for 10 s from 25 s into
# epoch 19 across the boundary with epoch 20, and 25 bpm for 6 s from 17 s into epoch 23.
STEPS = (
    (135.0, 60.0, 15.0),
    (300.0, 60.0, 40.0),
    (458.0, 5.0, 40.0),
    (595.0, 10.0, 40.0),
    (707.0, 6.0, 25.0),
)
STEP_EPOCHS = {4, 6, 10, 12}
SURGE_EPOCHS = {15: 40.0, 19: 40.0, 23: 25.0}
EPOCHS = 26
# Every jump of the scripted rate, up or down; each may pause its epoch or the next.
JUMP_EPOCHS = {
    int(time // 30) + later
    for onset, seconds, _delta in STEPS
    for time in (onset, onset + seconds)
    for later in (0, 1)
}


async def _epochs(recording_dir):
    replay = ReplaySession(ReplayConfig(input_path=recording_dir, speed=0.0))
    await replay.connect()
    try:
        return [epoch async for epoch in EpochBuilder(EpochConfig()).build(replay.stream())]
    finally:
        await replay.stop()


class ReplayedNightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = replace(
            SyntheticNightConfig(),
            preset="p1034",
            scenario="hr_steps",
            hours=EPOCHS * 30 / 3600,
            stages=("N2",) * EPOCHS,
            arousals_per_hour=0.0,
            ble_gaps_per_hour=0.0,
            heart_rate_steps=STEPS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            night = generate_night(config, Path(tmp) / "night")
            cls.epochs = asyncio.run(_epochs(night.recording_dir))[:EPOCHS]
        guard = ArousalGuard()
        first = cls.epochs[0].start_time
        cls.rows = [extract_ppg_features(epoch) for epoch in cls.epochs]
        cls.decisions = [
            guard.evaluate(timestamp_seconds=epoch.start_time - first, ppg=row)
            for epoch, row in zip(cls.epochs, cls.rows)
        ]

    def test_the_optics_feed_the_rule(self):
        self.assertEqual(len(self.rows), EPOCHS)
        for row in self.rows:
            self.assertEqual(row.sudden_hr_change_source, "optics")
            self.assertGreaterEqual(len(row.hr_series_used), 15, row.epoch_index)

    def test_the_guard_pauses_on_every_scripted_step_and_nowhere_else(self):
        fired = {
            index
            for index, decision in enumerate(self.decisions)
            if "sudden_heart_rate_change" in decision.reason_codes
        }
        self.assertEqual(fired & set(range(14)), STEP_EPOCHS)
        for index in STEP_EPOCHS:
            self.assertEqual(self.decisions[index].action, "pause")
            self.assertGreaterEqual(self.decisions[index].metadata["sudden_hr_change_pause_steps"], 1)
        for index in SURGE_EPOCHS:
            self.assertTrue(fired & {index, index + 1}, index)
        self.assertLessEqual(fired, JUMP_EPOCHS)

    def test_short_surges_show_in_runs_of_fast_beats(self):
        # No window holds only the surge; the ones over it read its run of fast beats.
        for index, delta in SURGE_EPOCHS.items():
            fast = [
                point
                for row in self.rows[index:index + 2]
                for point in row.hr_series_used
                if point.run_seconds > 0 and point.bpm > 58.0 + delta - 5.0
            ]
            self.assertTrue(fast, index)

    def test_steps_inside_an_epoch_show_in_its_row(self):
        for index, delta in ((4, 15.0), (6, -15.0)):
            row = self.rows[index]
            self.assertEqual(row.sudden_hr_change_count, 1, index)
            self.assertAlmostEqual(row.sudden_hr_changes[0].delta_bpm, delta, delta=2.0)

    def test_steps_at_an_epoch_start_show_across_the_boundary(self):
        for index, delta in ((10, 40.0), (12, -40.0)):
            self.assertEqual(self.rows[index].sudden_hr_change_count, 0, index)
            across = self.decisions[index].metadata["sudden_hr_change_across_epochs_bpm"]
            self.assertAlmostEqual(across, delta, delta=3.0)

    def test_the_decoder_stream_jumps_where_nothing_changed(self):
        # The old input: consecutive decoder heart rates 10 bpm apart in epochs without
        # a scripted step, the beat-in-beat-out jumps #247 describes.
        steady = [
            index
            for index, epoch in enumerate(self.epochs)
            if index not in JUMP_EPOCHS
            and _sudden_hr_changes(*_collect_heart_rate_series(epoch), PPGFeatureConfig())
        ]
        self.assertTrue(steady)
        for index in steady:
            self.assertNotIn("sudden_heart_rate_change", self.decisions[index].reason_codes)


if __name__ == "__main__":
    unittest.main()
