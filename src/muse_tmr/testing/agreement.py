"""Gate decisions against the hypnogram a synthetic night was generated from.

Synthetic input, not sleep. This only turns ``hypnogram.csv`` into reference REM epochs
and runs, and scores with ``muse_tmr.validation.rem_metrics``. The numbers say whether
the replay, detector and gate path still behaves the same on the same synthetic night;
they are never evidence of REM accuracy.
"""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from muse_tmr.testing.synthetic import SYNTHETIC_NOTE
from muse_tmr.validation.rem_metrics import (
    binary_metrics,
    gate_open_fraction,
    rem_onset_latency_seconds,
    roc_auc,
)

REM_STAGE = "R"
KAPPA_DEFINITION = "binary REM-vs-not κ of the gate against the generated hypnogram"


@dataclass(frozen=True)
class HypnogramEpoch:
    epoch: int
    start_seconds: float
    end_seconds: float
    stage: str


@dataclass(frozen=True)
class GateEpoch:
    """One detector epoch: start in seconds from the recording start, length, decision."""

    start_seconds: float
    duration_seconds: float
    gate_open: bool
    p_rem: float = math.nan

    @property
    def decision_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds


def load_hypnogram(path: Path) -> Tuple[HypnogramEpoch, ...]:
    """Rows of a ``hypnogram.csv``, or of the one in a recording directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "hypnogram.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(
            HypnogramEpoch(int(row["epoch"]), float(row["start_s"]), float(row["end_s"]), row["stage"])
            for row in csv.DictReader(handle)
        )


def reference_rem_runs(hypnogram: Sequence[HypnogramEpoch]) -> List[Tuple[float, float]]:
    """(start, end) seconds of each run of consecutive REM epochs."""
    runs: List[Tuple[float, float]] = []
    for row in hypnogram:
        if row.stage != REM_STAGE:
            continue
        if runs and math.isclose(runs[-1][1], row.start_seconds):
            runs[-1] = (runs[-1][0], row.end_seconds)
        else:
            runs.append((row.start_seconds, row.end_seconds))
    return runs


def gate_epochs_from_pilot3(report) -> List[GateEpoch]:
    """Detector epochs of a Pilot 3 report.

    Its timestamps count from the first replayed epoch, which starts with the first
    notification, a few tens of milliseconds into the recording; that offset is left out.
    """
    return [
        GateEpoch(
            start_seconds=result.timestamp_seconds,
            duration_seconds=result.duration_seconds,
            gate_open=bool(result.gate_decision.gate_open),
            p_rem=float(result.prediction.probability),
        )
        for result in report.epoch_results
    ]


def synthetic_agreement(
    hypnogram: Sequence[HypnogramEpoch],
    epochs: Sequence[GateEpoch],
) -> Dict[str, object]:
    """κ, REM recall and precision of the gate, gate-open fraction and REM-onset latency.

    Each detector epoch takes the generated stage at its middle; epochs past the end of
    the hypnogram are left out.
    """
    starts = [row.start_seconds for row in hypnogram]
    labelled = []
    for epoch in epochs:
        stage = _stage_at(hypnogram, starts, epoch.start_seconds + epoch.duration_seconds / 2)
        if stage is not None:
            labelled.append((epoch, stage))
    y_true = np.array([1 if stage == REM_STAGE else 0 for _, stage in labelled], dtype=int)
    gate = np.array([1 if epoch.gate_open else 0 for epoch, _ in labelled], dtype=int)
    p_rem = np.array([epoch.p_rem for epoch, _ in labelled], dtype=float)
    metrics = binary_metrics(y_true, gate)
    runs = reference_rem_runs(hypnogram)
    latencies = rem_onset_latency_seconds(
        runs, [(epoch.decision_seconds, epoch.gate_open) for epoch in epochs]
    )
    return {
        "synthetic": True,
        "note": SYNTHETIC_NOTE,
        "kappa_definition": KAPPA_DEFINITION,
        "epochs": len(labelled),
        "reference_rem_epochs": int(y_true.sum()),
        "gate_open_epochs": int(gate.sum()),
        "gate_open_fraction": gate_open_fraction(gate),
        "kappa": metrics["kappa"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "confusion": {key: metrics[key] for key in ("tp", "fp", "fn", "tn")},
        "p_rem_roc_auc": roc_auc(y_true, p_rem) if np.isfinite(p_rem).all() else math.nan,
        "rem_periods": [
            {"start_s": start, "end_s": end, "first_gate_open_latency_s": latency}
            for (start, end), latency in zip(runs, latencies)
        ],
    }


def stage_at(hypnogram: Sequence[HypnogramEpoch], seconds: float) -> Optional[str]:
    """The generated stage at ``seconds`` from the start; None past the hypnogram's end."""
    return _stage_at(hypnogram, [row.start_seconds for row in hypnogram], seconds)


def _stage_at(hypnogram: Sequence[HypnogramEpoch], starts: Sequence[float], seconds: float) -> Optional[str]:
    index = bisect.bisect_right(starts, seconds) - 1
    if index >= 0 and seconds < hypnogram[index].end_seconds:
        return hypnogram[index].stage
    return None
