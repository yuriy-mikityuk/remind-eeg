"""REM agreement metrics moved out of scripts/validate_rem_detection.py (#233).

The expected values were produced by the script's own functions before the move, on
the fixed arrays below, so the move changes no output.
"""

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

from muse_tmr.models.rem_detector import RemPrediction
from muse_tmr.validation import rem_metrics
from muse_tmr.validation.rem_metrics import (
    GRID_ENTER_THRESHOLDS,
    _cohen_kappa,
    best_f1_threshold,
    binary_metrics,
    false_open_minutes,
    feature_columns,
    gate_grid_pairs,
    gate_open_fraction,
    latency_summary,
    per_feature_auc,
    precision_at_recall,
    rem_onset_latency_seconds,
    rem_runs,
    roc_auc,
    simulate_gate,
    threshold_sweep,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_rem_detection.py"

Y_TRUE = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=int)
SCORES = np.array([0.10, 0.40, 0.35, 0.80, 0.40, 0.90, 0.20, 0.65, 0.65, 0.55, 0.95, 0.05])
NAN = float("nan")


def _load_script():
    spec = importlib.util.spec_from_file_location("validate_rem_detection_for_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(tp, fp, fn, tn, precision, recall, f1, accuracy, kappa):
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "kappa": kappa,
    }


class SameOutputAsTheScriptTest(unittest.TestCase):
    def assertSameMetrics(self, observed, expected):
        self.assertEqual(list(observed), list(expected))
        for key, value in expected.items():
            if isinstance(value, float) and math.isnan(value):
                self.assertTrue(math.isnan(observed[key]), key)
            else:
                self.assertEqual(observed[key], value, key)

    def test_the_script_uses_these_functions(self):
        script = _load_script()
        for name in (
            "binary_metrics", "roc_auc", "threshold_sweep", "best_f1_threshold", "gate_open_fraction",
            # Moved out of the script by #226:
            "simulate_gate", "feature_columns", "per_feature_auc", "grid_exit_thresholds",
            "GRID_ENTER_THRESHOLDS", "GRID_LOWEST_EXIT",
        ):
            self.assertIs(getattr(script, name), getattr(rem_metrics, name), name)
        self.assertFalse(hasattr(script, "_cohen_kappa"))
        self.assertFalse(hasattr(script, "feature_metrics"))

    def test_binary_metrics(self):
        self.assertSameMetrics(
            binary_metrics(Y_TRUE, (SCORES >= 0.5).astype(int)),
            _metrics(5, 1, 1, 5, 5 / 6, 5 / 6, 0.8333333333333334, 0.8333333333333334, 0.6666666666666667),
        )
        self.assertSameMetrics(
            binary_metrics(Y_TRUE, np.zeros_like(Y_TRUE)),
            _metrics(0, 0, 6, 6, NAN, 0.0, NAN, 0.5, 0.0),
        )
        self.assertSameMetrics(
            binary_metrics(np.zeros_like(Y_TRUE), (SCORES >= 0.5).astype(int)),
            _metrics(0, 6, 0, 6, 0.0, NAN, NAN, 0.5, 0.0),
        )

    def test_kappa(self):
        self.assertEqual(_cohen_kappa(4, 1, 2, 5), 0.5)
        self.assertTrue(math.isnan(_cohen_kappa(0, 0, 0, 0)))

    def test_roc_auc(self):
        self.assertEqual(roc_auc(Y_TRUE, SCORES), 0.875)
        self.assertTrue(math.isnan(roc_auc(np.ones_like(Y_TRUE), SCORES)))

    def test_threshold_sweep(self):
        rows = threshold_sweep(Y_TRUE, SCORES, step=0.25)
        expected = [
            (0.25, _metrics(6, 3, 0, 3, 0.6666666666666666, 1.0, 0.8, 0.75, 0.5)),
            (0.5, _metrics(5, 1, 1, 5, 0.8333333333333334, 0.8333333333333334,
                           0.8333333333333334, 0.8333333333333334, 0.6666666666666667)),
            (0.75, _metrics(3, 0, 3, 6, 1.0, 0.5, 0.6666666666666666, 0.75, 0.5)),
        ]
        self.assertEqual(len(rows), len(expected))
        for row, (threshold, metrics) in zip(rows, expected):
            self.assertSameMetrics(row, {"threshold": threshold, **metrics})

        default = threshold_sweep(Y_TRUE, SCORES)
        self.assertEqual(len(default), 19)
        self.assertSameMetrics(
            default[0], {"threshold": 0.05, **_metrics(6, 6, 0, 0, 0.5, 1.0, 0.6666666666666666, 0.5, 0.0)}
        )
        # The threshold adds up 0.05 steps, so the last one is a hair above 0.95 and the
        # 0.95 score falls under it. The move keeps that.
        self.assertSameMetrics(
            default[-1], {"threshold": 0.95, **_metrics(0, 0, 6, 6, NAN, 0.0, NAN, 0.5, 0.0)}
        )

    def test_best_f1_threshold(self):
        threshold, metrics = best_f1_threshold(Y_TRUE, SCORES)
        self.assertEqual(threshold, 0.5)
        self.assertSameMetrics(
            metrics,
            _metrics(5, 1, 1, 5, 0.8333333333333334, 0.8333333333333334,
                     0.8333333333333334, 0.8333333333333334, 0.6666666666666667),
        )


class GateOpenFractionTest(unittest.TestCase):
    def test_share_of_open_epochs(self):
        self.assertEqual(gate_open_fraction([True, False, False, True]), 0.5)
        self.assertEqual(gate_open_fraction(np.array([1, 0, 0, 0])), 0.25)

    def test_no_epochs_is_nan(self):
        self.assertTrue(math.isnan(gate_open_fraction([])))

    def test_pairs_are_refused(self):
        with self.assertRaises(ValueError):
            gate_open_fraction([(30.0, True), (60.0, False)])


class RemOnsetLatencyTest(unittest.TestCase):
    def test_first_open_decision_inside_each_run(self):
        # REM 600-900 s and 1800-1950 s; decisions every 30 s at the end of each epoch.
        decisions = [(30.0 * k, False) for k in range(1, 80)]
        opened = {660.0, 690.0, 720.0, 1860.0}
        decisions = [(seconds, seconds in opened) for seconds, _ in decisions]

        self.assertEqual(
            rem_onset_latency_seconds([(600.0, 900.0), (1800.0, 1950.0)], decisions),
            [60.0, 60.0],
        )

    def test_run_without_an_open_gate_is_none(self):
        decisions = [(630.0, False), (660.0, False), (960.0, True)]
        self.assertEqual(rem_onset_latency_seconds([(600.0, 900.0)], decisions), [None])

    def test_decision_at_the_run_start_belongs_to_the_epoch_before(self):
        decisions = [(600.0, True), (630.0, True)]
        self.assertEqual(rem_onset_latency_seconds([(600.0, 900.0)], decisions), [30.0])

    def test_decision_at_the_run_end_still_counts(self):
        self.assertEqual(rem_onset_latency_seconds([(600.0, 660.0)], [(660.0, True)]), [60.0])

    def test_decisions_need_not_be_sorted(self):
        decisions = [(720.0, True), (690.0, True), (660.0, False)]
        self.assertEqual(rem_onset_latency_seconds([(600.0, 900.0)], decisions), [90.0])


class PrecisionAtRecallTest(unittest.TestCase):
    """Known answers for precision at recall 0.80 (#226)."""

    def test_the_best_precision_among_thresholds_that_reach_the_recall(self):
        # Six positives; descending scores 0.95 P, 0.90 P, 0.80 P, 0.65 N, 0.65 P, 0.55 P,
        # 0.40 N, 0.40 N, 0.35 P, ... Five of six (0.83) are reached first at 0.55, where
        # 5 of the 6 epochs at or above it are REM.
        result = precision_at_recall(Y_TRUE, SCORES, 0.80)
        self.assertEqual(result, {"precision": 5 / 6, "recall": 5 / 6, "threshold": 0.55})

    def test_recall_one_takes_the_lowest_threshold_that_holds_every_positive(self):
        # At 0.35 nine epochs are called REM, six of them are.
        self.assertEqual(precision_at_recall(Y_TRUE, SCORES, 1.0), {"precision": 6 / 9, "recall": 1.0, "threshold": 0.35})

    def test_a_later_threshold_with_a_higher_precision_wins(self):
        y = np.array([1, 0, 0, 1, 1, 1])
        s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        # Recall 0.75 is first reached at 0.5 (3 of 5); at 0.4 all four positives are in, 4 of 6.
        self.assertEqual(precision_at_recall(y, s, 0.75), {"precision": 4 / 6, "recall": 1.0, "threshold": 0.4})

    def test_nan_scores_are_left_out_and_no_positive_is_nan(self):
        y = np.array([1, 0, 1, 0])
        s = np.array([0.9, float("nan"), 0.8, 0.1])
        self.assertEqual(precision_at_recall(y, s, 0.8), {"precision": 1.0, "recall": 1.0, "threshold": 0.8})
        empty = precision_at_recall(np.zeros(3, dtype=int), np.array([0.1, 0.2, 0.3]))
        self.assertTrue(all(math.isnan(value) for value in empty.values()))

    def test_a_hard_label_score_has_one_operating_point(self):
        stager = np.array([1, 1, 0, 1, 0, 0, 0, 1])
        y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        self.assertEqual(precision_at_recall(y, stager.astype(float), 0.5), {"precision": 0.75, "recall": 0.75, "threshold": 1.0})


class FalseOpenMinutesTest(unittest.TestCase):
    def test_open_epochs_outside_reference_rem(self):
        y = np.array([0, 1, 1, 0, 0])
        g = np.array([1, 1, 0, 1, 0])
        self.assertEqual(false_open_minutes(y, g), 1.0)
        self.assertEqual(false_open_minutes(y, g, epoch_seconds=10.0), 20.0 / 60.0)
        self.assertEqual(false_open_minutes(y, np.zeros(5, dtype=int)), 0.0)

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            false_open_minutes(np.array([0, 1]), np.array([1]))


class RemRunsAndLatencyTest(unittest.TestCase):
    def test_runs_end_at_a_non_rem_epoch_or_a_gap(self):
        epochs = [(0, 30, False), (30, 60, True), (60, 90, True), (90, 120, False), (120, 150, True), (180, 210, True)]
        self.assertEqual(rem_runs(epochs), [(30.0, 90.0), (120.0, 150.0), (180.0, 210.0)])
        self.assertEqual(rem_runs([]), [])

    def test_summary_of_the_runs_with_an_open_gate(self):
        summary = latency_summary([30.0, None, 60.0, 90.0, 150.0])
        self.assertEqual(summary["runs"], 5)
        self.assertEqual(summary["runs_with_open"], 4)
        self.assertEqual((summary["q1_s"], summary["median_s"], summary["q3_s"]), (52.5, 75.0, 105.0))
        none = latency_summary([None])
        self.assertEqual(none["runs_with_open"], 0)
        self.assertTrue(math.isnan(none["median_s"]))


class PerFeatureAucTest(unittest.TestCase):
    def test_known_aucs_saturation_and_missing_columns(self):
        predictions = [
            RemPrediction(
                probability=0.5,
                feature_scores={"rising": float(k), "flat": 1.0},
                feature_values={"falling": -float(k), **({"half": float(k)} if k % 2 else {})},
            )
            for k in range(8)
        ]
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        columns = feature_columns(predictions)
        metrics = per_feature_auc(columns, y)

        self.assertEqual(list(metrics), ["score_flat", "score_rising", "value_falling", "value_half"])
        self.assertEqual(metrics["score_rising"]["auc"], 1.0)
        self.assertEqual(metrics["value_falling"]["auc"], 0.0)
        self.assertEqual(metrics["score_flat"], {"n": 8, "auc": 0.5, "min": 1.0, "max": 1.0, "saturated": True})
        # value_half exists on the odd epochs only: 1 and 3 below REM, 5 and 7 in it.
        self.assertEqual(metrics["value_half"]["n"], 4)
        self.assertEqual(metrics["value_half"]["auc"], 1.0)
        self.assertTrue(np.isnan(columns["value_half"][0]))


class SimulateGateTest(unittest.TestCase):
    def _predictions(self, codes):
        return [RemPrediction(probability=0.95, reason_codes=codes) for _ in range(4)]

    def test_missing_imu_caps_the_gate_unless_it_is_left_out_by_design(self):
        predictions = self._predictions(("low_imu_coverage", "low_ppg_hr_coverage"))

        self.assertEqual(simulate_gate(predictions, 30.0, optics_capable=False), [False] * 4)
        self.assertEqual(
            simulate_gate(predictions, 30.0, optics_capable=False, imu_capable=False), [False, True, True, True]
        )
        # Missing optics still caps on an optics preset, whatever the IMU.
        self.assertEqual(simulate_gate(predictions, 30.0, imu_capable=False), [False] * 4)

    def test_the_grid_pairs(self):
        pairs = list(gate_grid_pairs())
        self.assertEqual(len(pairs), 51)
        self.assertEqual(pairs[0], (GRID_ENTER_THRESHOLDS[0], 0.45))
        self.assertIn((0.90, 0.85), pairs)
        wide = list(gate_grid_pairs(tuple(round(0.05 * k, 2) for k in range(1, 20)), 0.05))
        self.assertEqual(len(wide), 190)
        self.assertTrue(all(exit_ <= enter for enter, exit_ in wide))


if __name__ == "__main__":
    unittest.main()
