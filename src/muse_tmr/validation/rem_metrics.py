"""REM agreement metrics against a reference hypnogram.

One implementation of each metric. ``scripts/validate_rem_detection.py`` scores the
detector against an automated stager or external labels with them,
``muse_tmr.validation.external_labels`` against the PSG consensus of the BOAS dataset,
and ``muse_tmr.testing.agreement`` against the hypnogram a synthetic night was generated
from. The κ here is binary (REM against not REM), not five-class. ``simulate_gate()``
runs the real ``StableRemGate`` over a series of predictions, as the pilots do.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from muse_tmr.models.rem_detector import RemPrediction
from muse_tmr.models.rem_gate import RemGateConfig, StableRemGate

# The enter/exit grid of the REM check: enter 0.70 to 0.95, exit 0.45 up to enter, in
# steps of 0.05. It holds the historical 0.70/0.45 and both preset pairs (0.90/0.85,
# 0.80/0.70). validate-external fits over a wider one (external_labels.FIT_*).
GRID_ENTER_THRESHOLDS = tuple(round(0.70 + 0.05 * step, 2) for step in range(6))
GRID_LOWEST_EXIT = 0.45
# The gate's low-confidence reason codes that come from missing IMU. When the IMU is
# left out by design (simulate_gate's imu_capable=False), as the BOAS harness leaves out
# a headband IMU whose units are not the ones its file names, they do not cap confidence,
# as RemGateConfig.optics_capable does for the cardiac codes.
IMU_DEPENDENT_REASON_CODES = ("low_imu_coverage", "imu_features_missing")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if math.isfinite(precision) and math.isfinite(recall):
        # Conventional F1: a total miss (P=R=0) is F1=0, not undefined.
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        f1 = float("nan")
    accuracy = (tp + tn) / total if total else float("nan")
    kappa = _cohen_kappa(tp, fp, fn, tn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "kappa": kappa,
    }


def _cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    total = tp + fp + fn + tn
    if total == 0:
        return float("nan")
    po = (tp + tn) / total
    p_pred_pos = (tp + fp) / total
    p_true_pos = (tp + fn) / total
    pe = p_pred_pos * p_true_pos + (1 - p_pred_pos) * (1 - p_true_pos)
    return (po - pe) / (1 - pe) if (1 - pe) else float("nan")


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from scipy.stats import rankdata

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    sum_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, step: float = 0.05) -> List[Dict[str, float]]:
    rows = []
    thr = step
    while thr < 1.0:
        m = binary_metrics(y_true, (scores >= thr).astype(int))
        rows.append({"threshold": round(thr, 3), **m})
        thr += step
    return rows


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_thr, best = 0.5, binary_metrics(y_true, (scores >= 0.5).astype(int))
    for thr in np.round(np.arange(0.01, 1.0, 0.01), 2):
        m = binary_metrics(y_true, (scores >= thr).astype(int))
        f1 = m["f1"]
        if math.isfinite(f1) and (not math.isfinite(best["f1"]) or f1 > best["f1"]):
            best_thr, best = float(thr), m
    return best_thr, best


def gate_open_fraction(gate_open: Sequence[bool]) -> float:
    """Share of epochs with the gate open, one flag per epoch; NaN when there are none."""
    values = np.asarray(gate_open, dtype=bool)
    if values.ndim != 1:
        raise ValueError("gate_open must be one flag per epoch")
    return float(values.mean()) if values.size else float("nan")


def rem_onset_latency_seconds(
    reference_rem_runs: Sequence[Tuple[float, float]],
    gate_open: Sequence[Tuple[float, bool]],
) -> List[Optional[float]]:
    """Seconds from the start of each reference REM run to the first open gate in it.

    ``reference_rem_runs`` are ``(start, end)`` seconds of the reference's REM periods.
    ``gate_open`` holds ``(decision_seconds, open)`` per epoch on the same clock, where
    a decision exists once its epoch has ended (start plus epoch length). A decision
    counts for a run when ``start < decision_seconds <= end``; a run with no open gate
    among them gives None. A gate that was already open when the run began counts from
    its first decision inside the run.
    """
    decisions = sorted((float(seconds), bool(is_open)) for seconds, is_open in gate_open)
    latencies: List[Optional[float]] = []
    for start, end in reference_rem_runs:
        start, end = float(start), float(end)
        first = next(
            (seconds for seconds, is_open in decisions if is_open and start < seconds <= end),
            None,
        )
        latencies.append(None if first is None else first - start)
    return latencies


def precision_at_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    recall: float = 0.80,
) -> Dict[str, float]:
    """The highest precision of ``scores >= t`` over the thresholds ``t`` whose recall
    reaches ``recall``: the interpolated precision-recall curve at ``recall``.

    Returns ``precision``, the ``recall`` reached there and the ``threshold``. Epochs
    with a NaN score are left out. All three are NaN without a positive epoch. Among
    thresholds with the same precision the highest is returned.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    positives = int(np.sum(y == 1))
    if positives == 0:
        return {"precision": float("nan"), "recall": float("nan"), "threshold": float("nan")}
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    # One threshold per distinct score: every epoch at or above it is called REM.
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], s_sorted.size - 1]
    recalls = tp[last] / positives
    precisions = tp[last] / (tp[last] + fp[last])
    reaching = recalls >= recall - 1e-12
    index = int(np.argmax(np.where(reaching, precisions, -np.inf)))
    return {
        "precision": float(precisions[index]),
        "recall": float(recalls[index]),
        "threshold": float(s_sorted[last[index]]),
    }


def false_open_minutes(y_true: np.ndarray, gate_open: np.ndarray, epoch_seconds: float = 30.0) -> float:
    """Minutes the gate was open on epochs the reference does not call REM."""
    y = np.asarray(y_true, dtype=int)
    g = np.asarray(gate_open, dtype=bool)
    if y.shape != g.shape:
        raise ValueError("y_true and gate_open must hold one value per epoch")
    return float(np.sum(g & (y == 0)) * float(epoch_seconds) / 60.0)


def rem_runs(epochs: Sequence[Tuple[float, float, bool]]) -> List[Tuple[float, float]]:
    """(start, end) seconds of each run of REM epochs, from ``(start, end, is_rem)`` per
    epoch in time order. A run ends at an epoch that is not REM or at a gap between
    epochs."""
    runs: List[Tuple[float, float]] = []
    previous_end: Optional[float] = None
    for start, end, is_rem in epochs:
        start, end = float(start), float(end)
        if is_rem:
            joined = runs and previous_end is not None and runs[-1][1] == previous_end
            if joined and math.isclose(previous_end, start):
                runs[-1] = (runs[-1][0], end)
            else:
                runs.append((start, end))
        previous_end = end
    return runs


def latency_summary(latencies: Sequence[Optional[float]]) -> Dict[str, object]:
    """Median and interquartile range of REM-onset latencies, as
    ``rem_onset_latency_seconds()`` gives them; a run without an open gate counts in
    ``runs`` and not in the figures."""
    found = np.asarray([value for value in latencies if value is not None], dtype=float)
    summary: Dict[str, object] = {"runs": len(latencies), "runs_with_open": int(found.size)}
    if found.size:
        q1, median, q3 = (float(value) for value in np.percentile(found, [25, 50, 75]))
    else:
        q1 = median = q3 = float("nan")
    summary.update({"median_s": median, "q1_s": q1, "q3_s": q3})
    return summary


# --------------------------------------------------------------------------- #
# Per-feature separation, moved here from scripts/validate_rem_detection.py (#226)
# --------------------------------------------------------------------------- #
def feature_columns(predictions: Sequence[RemPrediction]) -> Dict[str, np.ndarray]:
    """The per-epoch score_<name> and value_<name> columns, NaN where an epoch lacks one."""
    rows = [
        {
            **{f"score_{name}": value for name, value in pred.feature_scores.items()},
            **{f"value_{name}": value for name, value in pred.feature_values.items()},
        }
        for pred in predictions
    ]
    names = sorted({name for row in rows for name in row})
    return {
        name: np.array([_as_float(row.get(name)) for row in rows], dtype=float)
        for name in names
    }


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def per_feature_auc(columns: Mapping[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Dict[str, object]]:
    """ROC-AUC against the reference's REM, min, max and count of finite values per column.

    Epochs where a column is NaN are left out of its numbers. ``saturated`` means every
    finite value is the same, so the column cannot rank epochs at all, as the
    eye-movement proxy on a real night (#116). An AUC below 0.5 means the column runs
    against REM. score_* columns come first.
    """
    metrics: Dict[str, Dict[str, object]] = {}
    for name in sorted(columns, key=lambda column: (not column.startswith("score_"), column)):
        values = np.asarray(columns[name], dtype=float)
        finite = np.isfinite(values)
        count = int(finite.sum())
        if count:
            low, high = float(values[finite].min()), float(values[finite].max())
            auc = roc_auc(y_true[finite], values[finite])
        else:
            low = high = auc = float("nan")
        metrics[name] = {
            "n": count,
            "auc": auc,
            "min": low,
            "max": high,
            "saturated": bool(count and low == high),
        }
    return metrics


# --------------------------------------------------------------------------- #
# The gate over a series of predictions, moved here from the REM check script (#226)
# --------------------------------------------------------------------------- #
def simulate_gate(
    predictions: Sequence[RemPrediction],
    epoch_seconds: float,
    *,
    optics_capable: bool = True,
    enter_threshold: Optional[float] = None,
    exit_threshold: Optional[float] = None,
    imu_capable: bool = True,
) -> List[bool]:
    """Run the real StableRemGate over the p_rem series, in time order.

    ``optics_capable`` mirrors the session preset: on an EEG-only preset (p21)
    the missing-cardiac reason codes must not cap confidence, or the gate never
    opens (see issue #112). ``enter_threshold``/``exit_threshold`` default to the
    recalibrated RemGateConfig defaults when omitted. ``imu_capable`` False does the
    same for the IMU codes (``IMU_DEPENDENT_REASON_CODES``), for input whose IMU is left
    out by design; the default runs the gate as the pilots do.
    """
    base = RemGateConfig()
    low_confidence = base.low_confidence_reason_codes
    if not imu_capable:
        low_confidence = tuple(code for code in low_confidence if code not in IMU_DEPENDENT_REASON_CODES)
    cfg = RemGateConfig(
        epoch_seconds=epoch_seconds,
        optics_capable=optics_capable,
        enter_threshold=base.enter_threshold if enter_threshold is None else enter_threshold,
        exit_threshold=base.exit_threshold if exit_threshold is None else exit_threshold,
        low_confidence_reason_codes=low_confidence,
    )
    gate = StableRemGate(cfg)
    return [gate.update(pred, duration_seconds=epoch_seconds).gate_open for pred in predictions]


def grid_exit_thresholds(enter_threshold: float, lowest_exit: float = GRID_LOWEST_EXIT) -> Tuple[float, ...]:
    steps = int(round((enter_threshold - lowest_exit) / 0.05))
    return tuple(round(lowest_exit + 0.05 * step, 2) for step in range(steps + 1))


def gate_grid_pairs(
    enter_thresholds: Sequence[float] = GRID_ENTER_THRESHOLDS,
    lowest_exit: float = GRID_LOWEST_EXIT,
) -> Iterator[Tuple[float, float]]:
    """Every (enter, exit) pair of a grid, by enter and then exit, both increasing, with
    exits from ``lowest_exit`` up to the enter value in steps of 0.05. The defaults are
    the REM check's grid."""
    for enter in enter_thresholds:
        for exit_ in grid_exit_thresholds(enter, lowest_exit):
            yield enter, exit_
