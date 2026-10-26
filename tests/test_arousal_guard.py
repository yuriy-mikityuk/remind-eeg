import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from muse_tmr.data.sample_types import EEGSample, MuseFrame
from muse_tmr.features import EpochBuilder, EpochConfig
from muse_tmr.features.eeg_features import EEGFeatureRow
from muse_tmr.features.imu_features import IMUFeatureRow
from muse_tmr.features.ppg_features import (
    HeartRatePoint,
    PPGFeatureConfig,
    PPGFeatureRow,
    SuddenHeartRateChange,
    _series_hr_changes,
    extract_ppg_features,
)
from muse_tmr.protocol import (
    ArousalGuard,
    ArousalGuardConfig,
    load_arousal_guard_decisions,
)


class TestArousalGuard(unittest.TestCase):
    def test_default_allows_quiet_features(self):
        decision = ArousalGuard().evaluate(
            timestamp_seconds=0.0,
            eeg=_eeg(alpha=0.10),
            imu=_imu(stillness_score=0.99),
            ppg=_ppg(),
        )

        self.assertEqual(decision.action, "allow")
        self.assertEqual(decision.reason_codes, ())
        self.assertEqual(decision.volume_multiplier, 1.0)

    def test_motion_arousal_pauses_and_logs_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "arousal_guard.jsonl"
            guard = ArousalGuard(event_log_path=log_path)

            decision = guard.evaluate(
                timestamp_seconds=30.0,
                imu=_imu(
                    arousal_guard_reason_codes=("motion_arousal_proxy",),
                    arousal_event_count=1,
                ),
            )
            logged = load_arousal_guard_decisions(log_path)

        self.assertEqual(decision.action, "pause")
        self.assertEqual(decision.pause_seconds, 120.0)
        self.assertIn("motion_arousal_proxy", decision.reason_codes)
        self.assertEqual([item.action for item in logged], ["pause"])

    def test_alpha_proxy_lowers_volume_before_stronger_pause_threshold(self):
        config = ArousalGuardConfig(
            alpha_lower_volume_relative_power=0.20,
            alpha_pause_relative_power=0.40,
            lower_volume_multiplier=0.40,
        )
        guard = ArousalGuard(config)

        mild = guard.evaluate(timestamp_seconds=0.0, eeg=_eeg(alpha=0.30))
        strong = guard.evaluate(timestamp_seconds=30.0, eeg=_eeg(alpha=0.50))

        self.assertEqual(mild.action, "lower_volume")
        self.assertEqual(mild.volume_multiplier, 0.40)
        self.assertIn("alpha_arousal_proxy_mild", mild.reason_codes)
        self.assertEqual(strong.action, "pause")
        self.assertIn("alpha_arousal_proxy", strong.reason_codes)

    def test_sudden_heart_rate_jump_pauses_cueing(self):
        decision = ArousalGuard().evaluate(
            timestamp_seconds=60.0,
            ppg=_ppg(sudden_hr_change_count=1, max_sudden_hr_change_bpm=18.0),
        )

        self.assertEqual(decision.action, "pause")
        self.assertIn("sudden_heart_rate_change", decision.reason_codes)

    def test_out_of_range_heart_rate_artifact_pauses_cueing(self):
        decision = ArousalGuard().evaluate(
            timestamp_seconds=60.0,
            ppg=_ppg(artifact_flags=("heart_rate_out_of_range",)),
        )

        self.assertEqual(decision.action, "pause")
        self.assertIn("heart_rate_out_of_range", decision.reason_codes)
        self.assertIn("ppg_artifact_flags", decision.reason_codes)

    def test_repeated_single_channel_eeg_artifact_lowers_volume_without_pause_or_stop(self):
        guard = ArousalGuard(
            ArousalGuardConfig(stop_after_consecutive_artifact_epochs=2)
        )

        decisions = [
            guard.evaluate(
                timestamp_seconds=index * 30.0,
                eeg=_eeg(
                    artifact_flags=("eeg_clipping_AF8",),
                    bad_channels=("AF8",),
                    usable_channel_count=3,
                ),
            )
            for index in range(3)
        ]

        self.assertEqual([decision.action for decision in decisions], ["lower_volume"] * 3)
        self.assertTrue(
            all("single_channel_eeg_artifact" in decision.reason_codes for decision in decisions)
        )
        self.assertEqual(guard.consecutive_artifact_epochs, 0)
        self.assertEqual(guard.consecutive_critical_artifact_epochs, 0)

    def test_repeated_noncritical_artifact_pauses_recoverably(self):
        guard = ArousalGuard(
            ArousalGuardConfig(stop_after_consecutive_artifact_epochs=2)
        )

        first = guard.evaluate(
            timestamp_seconds=0.0,
            eeg=_eeg(artifact_flags=("eeg_missing",)),
        )
        second = guard.evaluate(
            timestamp_seconds=30.0,
            eeg=_eeg(artifact_flags=("eeg_missing",)),
        )
        clean = guard.evaluate(
            timestamp_seconds=60.0,
            eeg=_eeg(),
        )

        self.assertEqual(first.action, "lower_volume")
        self.assertEqual(second.action, "pause")
        self.assertIn("repeated_artifact_quality", second.reason_codes)
        self.assertIn("eeg_artifact_flags", second.reason_codes)
        self.assertEqual(clean.action, "allow")
        self.assertEqual(guard.consecutive_artifact_epochs, 0)

    def test_legacy_repeated_artifact_stop_option_is_still_available(self):
        guard = ArousalGuard(
            ArousalGuardConfig(
                repeated_artifact_action="stop",
                stop_after_consecutive_artifact_epochs=2,
            )
        )

        guard.evaluate(timestamp_seconds=0.0, eeg=_eeg(artifact_flags=("eeg_missing",)))
        second = guard.evaluate(timestamp_seconds=30.0, eeg=_eeg(artifact_flags=("eeg_missing",)))

        self.assertEqual(second.action, "stop")
        self.assertIn("repeated_artifact_quality", second.reason_codes)

    def test_critical_multi_channel_eeg_artifact_stops_after_configured_streak(self):
        guard = ArousalGuard(
            ArousalGuardConfig(
                critical_eeg_bad_channel_count=3,
                stop_after_consecutive_critical_artifact_epochs=2,
            )
        )

        first = guard.evaluate(
            timestamp_seconds=0.0,
            eeg=_eeg(
                artifact_flags=("eeg_clipping_AF7", "eeg_clipping_AF8", "eeg_clipping_TP9"),
                bad_channels=("AF7", "AF8", "TP9"),
                usable_channel_count=1,
            ),
        )
        second = guard.evaluate(
            timestamp_seconds=30.0,
            eeg=_eeg(
                artifact_flags=("eeg_clipping_AF7", "eeg_clipping_AF8", "eeg_clipping_TP9"),
                bad_channels=("AF7", "AF8", "TP9"),
                usable_channel_count=1,
            ),
        )

        self.assertEqual(first.action, "pause")
        self.assertIn("critical_eeg_artifact", first.reason_codes)
        self.assertEqual(second.action, "stop")
        self.assertIn("critical_eeg_artifact", second.reason_codes)

    def test_motion_and_hr_arousal_pause_then_recover_after_clean_epoch(self):
        guard = ArousalGuard()

        motion = guard.evaluate(
            timestamp_seconds=0.0,
            imu=_imu(arousal_guard_reason_codes=("motion_arousal_proxy",), arousal_event_count=1),
        )
        hr = guard.evaluate(
            timestamp_seconds=30.0,
            ppg=_ppg(sudden_hr_change_count=1, max_sudden_hr_change_bpm=18.0),
        )
        clean = guard.evaluate(
            timestamp_seconds=60.0,
            eeg=_eeg(alpha=0.10),
            imu=_imu(stillness_score=0.99),
            ppg=_ppg(),
        )

        self.assertEqual(motion.action, "pause")
        self.assertEqual(hr.action, "pause")
        self.assertEqual(clean.action, "allow")
        self.assertEqual(guard.consecutive_pause_epochs, 0)

    def test_eeg_channel_diagnostics_are_logged_for_artifacts(self):
        decision = ArousalGuard().evaluate(
            timestamp_seconds=0.0,
            eeg=_eeg(
                artifact_flags=("eeg_clipping_AF7",),
                channel_diagnostics={
                    "AF7": {
                        "max_uv": 760.0,
                        "median_uv": 725.0,
                        "centered_abs_max_uv": 35.0,
                    }
                },
            ),
        )

        self.assertEqual(decision.action, "lower_volume")
        self.assertIn("eeg_channel_diagnostics", decision.metadata)
        self.assertEqual(
            decision.metadata["eeg_channel_diagnostics"]["AF7"]["centered_abs_max_uv"],
            35.0,
        )

    def test_config_rejects_unordered_alpha_thresholds(self):
        with self.assertRaises(ValueError):
            ArousalGuardConfig(
                alpha_lower_volume_relative_power=0.50,
                alpha_pause_relative_power=0.40,
            ).validate()

    def test_eeg_only_row_is_allowed_when_preset_has_no_optics(self):
        ppg = _eeg_only_ppg()
        self.assertEqual(
            ppg.artifact_flags,
            (
                "heart_rate_missing",
                "low_heart_rate_coverage",
                "missing_heart_rate",
                "missing_ppg",
                "ppg_missing",
            ),
        )
        guard = ArousalGuard(ArousalGuardConfig(optics_capable=False))

        decisions = [
            guard.evaluate(
                timestamp_seconds=index * 30.0,
                eeg=_eeg(alpha=0.10),
                imu=_imu(stillness_score=0.99),
                ppg=ppg,
            )
            for index in range(4)
        ]

        self.assertEqual([decision.action for decision in decisions], ["allow"] * 4)
        for decision in decisions:
            self.assertEqual(decision.reason_codes, ())
            self.assertEqual(decision.volume_multiplier, 1.0)
            self.assertIs(decision.metadata["ppg_absent_by_design"], True)
        self.assertEqual(guard.consecutive_artifact_epochs, 0)

    def test_eeg_only_row_on_optics_preset_keeps_lower_lower_pause(self):
        guard = ArousalGuard()

        decisions = [
            guard.evaluate(
                timestamp_seconds=index * 30.0,
                eeg=_eeg(alpha=0.10),
                imu=_imu(stillness_score=0.99),
                ppg=_eeg_only_ppg(),
            )
            for index in range(3)
        ]

        self.assertEqual(
            [decision.action for decision in decisions],
            ["lower_volume", "lower_volume", "pause"],
        )
        self.assertEqual(
            decisions[0].reason_codes, ("low_ppg_hr_coverage", "ppg_artifact_flags")
        )
        self.assertEqual(
            decisions[2].reason_codes,
            ("repeated_artifact_quality", "low_ppg_hr_coverage", "ppg_artifact_flags"),
        )
        self.assertTrue(
            all("ppg_absent_by_design" not in decision.metadata for decision in decisions)
        )

    def test_hr_rules_are_skipped_when_preset_has_no_optics(self):
        guard = ArousalGuard(ArousalGuardConfig(optics_capable=False))

        jump = guard.evaluate(
            timestamp_seconds=60.0,
            ppg=_ppg(sudden_hr_change_count=1, max_sudden_hr_change_bpm=18.0),
        )
        out_of_range = guard.evaluate(
            timestamp_seconds=90.0,
            ppg=_ppg(artifact_flags=("heart_rate_out_of_range",)),
        )

        self.assertEqual(jump.action, "allow")
        self.assertIs(jump.metadata["ppg_absent_by_design"], True)
        self.assertNotEqual(out_of_range.action, "pause")
        self.assertNotIn("heart_rate_out_of_range", out_of_range.reason_codes)
        self.assertNotIn("sudden_heart_rate_change", out_of_range.reason_codes)


class TestSuddenHeartRateRule(unittest.TestCase):
    """The HR rule over the heart-rate series of the PPG rows (#247)."""

    def test_pause_counts_only_changes_at_or_above_its_bpm(self):
        self.assertEqual(ArousalGuard().evaluate(timestamp_seconds=0.0, ppg=_ppg_with_changes(15.0)).action, "pause")

        raised = ArousalGuard(ArousalGuardConfig(sudden_hr_change_pause_bpm=20.0))
        below = raised.evaluate(timestamp_seconds=0.0, ppg=_ppg_with_changes(15.0))
        above = raised.evaluate(timestamp_seconds=30.0, ppg=_ppg_with_changes(15.0, -25.0))
        self.assertEqual(below.action, "allow")
        self.assertEqual(above.action, "pause")
        self.assertIn("sudden_heart_rate_change", above.reason_codes)
        self.assertEqual(above.metadata["sudden_hr_change_pause_steps"], 1)

        twice = ArousalGuard(ArousalGuardConfig(sudden_hr_change_pause_count=2))
        once = twice.evaluate(timestamp_seconds=0.0, ppg=_ppg_with_changes(15.0))
        two = twice.evaluate(timestamp_seconds=30.0, ppg=_ppg_with_changes(15.0, -12.0))
        self.assertEqual(once.action, "allow")
        self.assertEqual(two.action, "pause")

    def test_a_step_at_the_epoch_boundary_pauses_the_next_epoch(self):
        before = _ppg_with_series(60.0, start=0.0)
        after = _ppg_with_series(75.0, start=30.0)
        guard = ArousalGuard()

        first = guard.evaluate(timestamp_seconds=0.0, ppg=before)
        second = guard.evaluate(timestamp_seconds=30.0, ppg=after)

        self.assertEqual(first.action, "allow")
        self.assertEqual(second.action, "pause")
        self.assertIn("sudden_heart_rate_change", second.reason_codes)
        self.assertAlmostEqual(second.metadata["sudden_hr_change_across_epochs_bpm"], 15.0)
        self.assertEqual(second.metadata["sudden_hr_change_pause_steps"], 1)
        self.assertEqual(second.metadata["sudden_hr_change_source"], "optics")
        self.assertEqual(second.metadata["hr_series_used_count"], 21)

        # On its own the second row holds no change; nor after a reset, without the
        # link, or more than sudden_hr_change_link_seconds after the first.
        self.assertEqual(ArousalGuard().evaluate(timestamp_seconds=30.0, ppg=after).action, "allow")
        reset = ArousalGuard()
        reset.evaluate(timestamp_seconds=0.0, ppg=before)
        reset.reset()
        self.assertEqual(reset.evaluate(timestamp_seconds=30.0, ppg=after).action, "allow")
        unlinked = ArousalGuard(ArousalGuardConfig(sudden_hr_change_link_seconds=0.0))
        unlinked.evaluate(timestamp_seconds=0.0, ppg=before)
        self.assertEqual(unlinked.evaluate(timestamp_seconds=30.0, ppg=after).action, "allow")
        later = ArousalGuard()
        later.evaluate(timestamp_seconds=0.0, ppg=before)
        self.assertEqual(
            later.evaluate(timestamp_seconds=60.0, ppg=_ppg_with_series(75.0, start=60.0)).action,
            "allow",
        )

    def test_skipped_hr_values_are_logged_and_never_compared(self):
        guard = ArousalGuard()
        skipped = guard.evaluate(
            timestamp_seconds=0.0,
            eeg=_eeg(alpha=0.50),
            ppg=_ppg_with_series(60.0, start=0.0, used=False),
        )
        after = guard.evaluate(timestamp_seconds=30.0, ppg=_ppg_with_series(75.0, start=30.0))

        self.assertEqual(skipped.action, "pause")
        self.assertNotIn("sudden_heart_rate_change", skipped.reason_codes)
        self.assertEqual(skipped.metadata["hr_series_used_count"], 0)
        self.assertEqual(skipped.metadata["hr_series_skipped_count"], 21)
        self.assertEqual(after.action, "allow")

    def test_allow_decisions_log_skipped_hr_values(self):
        clean = ArousalGuard().evaluate(timestamp_seconds=0.0, ppg=_ppg_with_series(60.0, start=0.0))
        blind = ArousalGuard().evaluate(
            timestamp_seconds=0.0, ppg=_ppg_with_series(60.0, start=0.0, used=False)
        )

        self.assertEqual(clean.action, "allow")
        self.assertEqual(clean.metadata, {})
        self.assertEqual(blind.action, "allow")
        self.assertEqual(blind.metadata, {"hr_series_used_count": 0, "hr_series_skipped_count": 21})

    def test_the_margin_counts_a_change_just_under_the_bpm(self):
        self.assertEqual(ArousalGuard().evaluate(timestamp_seconds=0.0, ppg=_ppg_with_changes(9.5)).action, "pause")
        self.assertEqual(ArousalGuard().evaluate(timestamp_seconds=0.0, ppg=_ppg_with_changes(8.5)).action, "allow")
        exact = ArousalGuard(ArousalGuardConfig(sudden_hr_change_margin_bpm=0.0))
        self.assertEqual(exact.evaluate(timestamp_seconds=0.0, ppg=_ppg_with_changes(9.5)).action, "allow")

        across = ArousalGuard()
        across.evaluate(timestamp_seconds=0.0, ppg=_ppg_with_series(60.0, start=0.0))
        second = across.evaluate(timestamp_seconds=30.0, ppg=_ppg_with_series(69.5, start=30.0))
        self.assertEqual(second.action, "pause")
        self.assertAlmostEqual(second.metadata["sudden_hr_change_across_epochs_bpm"], 9.5)

    def test_one_step_seen_from_both_sides_of_a_boundary_counts_once(self):
        # A 15 bpm step 8 s into the second epoch shows in that row's own comparison and in
        # the one from the first epoch's values: one step, not two.
        before = _ppg_with_series(60.0, start=0.0)
        after = _ppg_with_step_series(start=30.0, step_time=38.0, before_bpm=60.0, after_bpm=75.0)

        once = ArousalGuard()
        once.evaluate(timestamp_seconds=0.0, ppg=before)
        single = once.evaluate(timestamp_seconds=30.0, ppg=after)
        twice = ArousalGuard(ArousalGuardConfig(sudden_hr_change_pause_count=2))
        twice.evaluate(timestamp_seconds=0.0, ppg=before)

        self.assertEqual(single.action, "pause")
        self.assertAlmostEqual(single.metadata["sudden_hr_change_across_epochs_bpm"], 15.0)
        self.assertEqual(single.metadata["sudden_hr_change_pause_steps"], 1)
        self.assertEqual(twice.evaluate(timestamp_seconds=30.0, ppg=after).action, "allow")

    def test_config_rejects_a_negative_link(self):
        with self.assertRaises(ValueError):
            ArousalGuardConfig(sudden_hr_change_link_seconds=-1.0).validate()

    def test_config_rejects_a_margin_outside_the_bpm(self):
        for margin in (-1.0, 10.0):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                ArousalGuardConfig(sudden_hr_change_margin_bpm=margin).validate()


def _ppg_with_changes(*deltas):
    changes = tuple(
        SuddenHeartRateChange(
            start_time=10.0 + index,
            end_time=20.0 + index,
            before_bpm=60.0,
            after_bpm=60.0 + delta,
            delta_bpm=delta,
        )
        for index, delta in enumerate(deltas)
    )
    return replace(
        _ppg(len(changes), max(abs(delta) for delta in deltas)),
        sudden_hr_changes=changes,
        sudden_hr_change_source="optics",
    )


def _ppg_with_series(bpm, *, start, used=True):
    points = tuple(
        HeartRatePoint(
            time=start + 5.0 + second,
            bpm=bpm if used else math.nan,
            peak_bpm=bpm if used else math.nan,
            quality=0.95 if used else 0.0,
            used=used,
        )
        for second in range(21)
    )
    return replace(
        _ppg(),
        start_time=start,
        end_time=start + 30.0,
        sudden_hr_change_source="optics",
        hr_series_channel="LO_NIR",
        hr_series=points,
    )


def _ppg_with_step_series(*, start, step_time, before_bpm, after_bpm):
    """A row whose series steps from before_bpm to after_bpm at step_time, with the
    changes extract_ppg_features would log for it."""
    points = tuple(
        HeartRatePoint(
            time=start + 5.0 + second,
            bpm=before_bpm if start + 5.0 + second < step_time else after_bpm,
            peak_bpm=before_bpm if start + 5.0 + second < step_time else after_bpm,
            quality=0.95,
            used=True,
        )
        for second in range(21)
    )
    changes = _series_hr_changes(points, PPGFeatureConfig())
    return replace(
        _ppg(len(changes), max((abs(change.delta_bpm) for change in changes), default=math.nan)),
        start_time=start,
        end_time=start + 30.0,
        sudden_hr_changes=changes,
        sudden_hr_change_source="optics",
        hr_series_channel="LO_NIR",
        hr_series=points,
    )


def _eeg_only_ppg():
    frame = MuseFrame(
        timestamp=0.0,
        eeg=EEGSample(0.0, {"AF7": [0.0] * 7680, "AF8": [0.0] * 7680}, source="test"),
        source="test",
    )
    epoch = EpochBuilder(EpochConfig(epoch_seconds=30, stride_seconds=30))._build_epoch(
        0, 0.0, (frame,)
    )
    return extract_ppg_features(epoch)


def _eeg(
    alpha=0.10,
    artifact_flags=(),
    channel_diagnostics=None,
    bad_channels=(),
    usable_channel_count=2,
):
    return EEGFeatureRow(
        epoch_index=1,
        start_time=0.0,
        end_time=30.0,
        eeg_coverage=1.0,
        sample_count=256,
        channel_count=2,
        channel_sample_counts={"AF7": 256, "AF8": 256},
        band_powers={},
        relative_band_powers={"alpha": alpha},
        ratios={},
        asymmetry={},
        eye_movement_proxy=0.0,
        artifact_flags=artifact_flags,
        quality_flags=(),
        channel_diagnostics=channel_diagnostics or {},
        bad_channels=bad_channels,
        usable_channel_count=usable_channel_count,
    )


def _imu(
    stillness_score=1.0,
    arousal_guard_reason_codes=(),
    arousal_event_count=0,
):
    return IMUFeatureRow(
        epoch_index=1,
        start_time=0.0,
        end_time=30.0,
        imu_coverage=1.0,
        sample_count=100,
        accelerometer_sample_count=100,
        gyroscope_sample_count=100,
        motion_level=0.0,
        stillness_score=stillness_score,
        accel_rms_delta_g=0.0,
        accel_peak_delta_g=0.0,
        gyro_rms_dps=0.0,
        gyro_peak_dps=0.0,
        movement_event_count=arousal_event_count,
        arousal_event_count=arousal_event_count,
        arousal_proxy=0.0,
        movement_events=(),
        cue_movement_logs=(),
        artifact_flags=(),
        quality_flags=(),
        arousal_guard_reason_codes=arousal_guard_reason_codes,
    )


def _ppg(sudden_hr_change_count=0, max_sudden_hr_change_bpm=math.nan, artifact_flags=()):
    return PPGFeatureRow(
        epoch_index=1,
        start_time=0.0,
        end_time=30.0,
        ppg_coverage=1.0,
        heart_rate_coverage=1.0,
        ppg_sample_count=64,
        heart_rate_sample_count=10,
        ppg_channel_count=1,
        ppg_channel_sample_counts={"ambient": 64},
        primary_ppg_channel="ambient",
        ppg_estimated_hr_bpm=60.0,
        ppg_confidence=0.9,
        ppg_peak_count=30,
        ppg_signal_quality="Good",
        mean_hr_bpm=60.0,
        median_hr_bpm=60.0,
        min_hr_bpm=58.0,
        max_hr_bpm=62.0,
        hr_trend_bpm_per_min=0.0,
        hr_source="heart_rate",
        mean_rr_ms=1000.0,
        sdnn_ms=20.0,
        rmssd_ms=25.0,
        pnn50_percent=0.0,
        hrv_source="heart_rate",
        sudden_hr_change_count=sudden_hr_change_count,
        max_sudden_hr_change_bpm=max_sudden_hr_change_bpm,
        sudden_hr_changes=(),
        artifact_flags=artifact_flags,
        quality_flags=(),
    )


if __name__ == "__main__":
    unittest.main()
