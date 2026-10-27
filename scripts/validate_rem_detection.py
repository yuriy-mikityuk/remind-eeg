#!/usr/bin/env python3
"""Validate the wired REM detector against an offline reference stager (YASA or GSSC).

The project ships one wired REM detector -- the fixed-threshold, non-ML
``HeuristicRemDetector`` -- but nothing in the repo measures whether it fires REM
at the *right time*: there is no PSG/hypnogram ground truth. This script gives a
first, cheap, independent second opinion by comparing the heuristic against YASA's
automated stager on a recording you already have.

Pipeline
    recording/
      +-- raw_amused.bin -> ReplaySession --+--> continuous EEG (uV), muse_tmr.data.continuous
                                            |                                   |
                                            |             YASA SleepStaging (.venv-yasa)
                                            v                                   |
                       EpochBuilder(30s) -> HeuristicRemDetector -> p_rem        |
                                         -> StableRemGate -> gate_open          |
                                                                                v
                          align 30s epochs by time  ->  agreement metrics

What it reports
    * REM precision / recall / F1 / Cohen's kappa of the heuristic vs the
      reference's REM (YASA or GSSC, --reference), at RemGateConfig's default
      thresholds (enter=0.80, exit=0.70), at 0.50, and at the best-F1 threshold;
      plus a full threshold sweep and ROC-AUC. The kappa is binary (REM vs not)
      against an automated stager, not a five-class kappa against PSG.
    * The *system* view: the actual StableRemGate simulated over the p_rem series
      with the --preset thresholds (p21 0.90/0.85, optics presets 0.80/0.70)
      -- when the gate would open (i.e. fire a cue), how often is the reference
      calling REM?
    * For a refit: ROC-AUC, min, max and a `saturated` flag (one value in every
      epoch) for each per-epoch score_*/value_* column, and an enter/exit grid of
      the gate (gate-open fraction, REM recall and precision per pair). --weights
      name=w,... replays with other detector weights.
    * provenance: the command line, versions (muse_tmr, git commit, Python,
      numpy, scipy, the stager package, torch, mne), the resample rate, the gate
      thresholds, the detector weights and which Muse channels the stager was and
      was not given.

Honest caveats (printed in the summary too)
    YASA is a PROXY, not PSG truth. Its model wants central electrodes (C3/C4);
    the Muse gives only frontal (AF7/AF8) + temporal (TP9/TP10) referenced near
    Fpz, and there is no EMG -- so its own REM calls are imperfect, especially
    W/N1/REM boundaries. Read agreement as directional evidence, not a grade.

Usage
    python scripts/validate_rem_detection.py data/recordings/overnight_20260707
    # --eog defaults to none for gssc and AF8 for yasa; override channels / venv / outputs:
    python scripts/validate_rem_detection.py <dir> --eeg AF7 --eog AF8 \
        --yasa-python .venv-yasa/bin/python --output out.json --html out.html
    # other detector weights; the rest keep their defaults, then all are scaled to sum 1:
    python scripts/validate_rem_detection.py <dir> --reference gssc --preset p21 \
        --weights theta_alpha_ratio=0,rem_event_density=0.2
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import platform
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from muse_tmr import __version__ as MUSE_TMR_VERSION  # noqa: E402
from muse_tmr.data.continuous import ContinuousBuilder, ContinuousConfig  # noqa: E402
from muse_tmr.data.replay import ReplayConfig, ReplaySession  # noqa: E402
from muse_tmr.features import new_rem_event_tracker  # noqa: E402
from muse_tmr.features.epochs import EpochBuilder, EpochConfig  # noqa: E402
from muse_tmr.models import HeuristicRemDetector  # noqa: E402
from muse_tmr.models.heuristic_rem_detector import (  # noqa: E402
    DEFAULT_REM_WEIGHTS,
    HeuristicRemConfig,
)
from muse_tmr.models.rem_detector import RemPrediction  # noqa: E402
from muse_tmr.models.rem_gate import RemGateConfig, StableRemGate  # noqa: E402
from muse_tmr.presets import preset_gate_thresholds, preset_provides_optics  # noqa: E402
from muse_tmr.validation.rem_metrics import (  # noqa: E402
    best_f1_threshold,
    binary_metrics,
    gate_open_fraction,
    roc_auc,
    threshold_sweep,
)

EEG_CHANNELS = ("AF7", "AF8", "TP9", "TP10")
REM_STAGES = ("R", "REM")
NOMINAL_EEG_RATE_HZ = 256.0
GAP_FILL_THRESHOLD_SECONDS = 0.5
# The reconstruction the stager has always been given: frames without AF7 are dropped,
# a dropout over 0.5 s is zero-filled by host time, and lost subpackets are not filled
# (an export fills them; keeping them out keeps these metrics comparable with #115/#116).
STAGER_CONTINUOUS_CONFIG = ContinuousConfig(
    gap_policy="zero_fill",
    gap_threshold_seconds=GAP_FILL_THRESHOLD_SECONDS,
    nominal_rate_hz=NOMINAL_EEG_RATE_HZ,
    channels=EEG_CHANNELS,
    reference_channel="AF7",
    fill_lost_samples=False,
)

# --eog when it is not given. GSSC's README documents horizontal or left/right EOG
# only, and AF8 is a second EEG channel, so GSSC stages from AF7 alone unless asked;
# YASA keeps AF8 as its EOG derivation.
DEFAULT_EOG = {"gssc": "none", "yasa": "AF8"}
KAPPA_DEFINITION = "binary REM-vs-not κ against an automated stager; not five-class"

# The enter/exit grid: enter 0.70 to 0.95, exit 0.45 up to enter, in steps of 0.05. It
# holds the historical 0.70/0.45 and both preset pairs (0.90/0.85, 0.80/0.70).
GRID_ENTER_THRESHOLDS = tuple(round(0.70 + 0.05 * step, 2) for step in range(6))
GRID_LOWEST_EXIT = 0.45


def parse_weights(text: str) -> Dict[str, float]:
    """``--weights name=w,name=w`` as {name: w}; names from DEFAULT_REM_WEIGHTS, w >= 0."""
    overrides: Dict[str, float] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        name, separator, value = (part.strip() for part in item.partition("="))
        if not separator or not name:
            raise argparse.ArgumentTypeError(f"expected name=weight, got {item.strip()!r}")
        if name not in DEFAULT_REM_WEIGHTS:
            raise argparse.ArgumentTypeError(
                f"unknown weight {name!r}; known: {', '.join(DEFAULT_REM_WEIGHTS)}"
            )
        if name in overrides:
            raise argparse.ArgumentTypeError(f"{name} is given twice")
        try:
            weight = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"weight for {name} is not a number: {value!r}") from None
        if not math.isfinite(weight) or weight < 0:
            raise argparse.ArgumentTypeError(f"weight for {name} must be a number >= 0")
        overrides[name] = weight
    if not overrides:
        raise argparse.ArgumentTypeError("--weights needs at least one name=weight")
    return overrides


def resolve_weights(overrides: Optional[Mapping[str, float]]) -> Dict[str, float]:
    """The detector weights for this run: DEFAULT_REM_WEIGHTS as they are without
    overrides; with them, the defaults updated and divided by their sum, so they add up
    to 1 like the defaults and the 0.40 limited-feature threshold keeps its meaning."""
    if not overrides:
        return dict(DEFAULT_REM_WEIGHTS)
    weights = dict(DEFAULT_REM_WEIGHTS)
    weights.update(overrides)
    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("--weights leaves every weight at 0")
    return {name: weight / total for name, weight in weights.items()}


# --------------------------------------------------------------------------- #
# One replay pass: heuristic p_rem per epoch AND continuous EEG for the stager
# --------------------------------------------------------------------------- #
async def _collect(
    recording_dir: Path,
    epoch_seconds: float,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """Single pass over raw_amused.bin.

    Feeds every MuseFrame to the EpochBuilder (-> heuristic p_rem per epoch)
    while side-accumulating per-channel EEG samples (uV) for the reference
    stager. The heuristic epochs are anchored to wall-clock frame timestamps, so
    to keep YASA's sample-indexed 30s epochs on the SAME clock, we zero-fill any
    BLE-dropout / AF7-loss gap: array position then maps to real elapsed time and
    YASA epoch j covers real [t0 + j*30, ...). Without the fill, a mid-night gap
    would slide every later YASA epoch earlier and mispair the two grids.
    ``weights`` replaces the detector's DEFAULT_REM_WEIGHTS (see resolve_weights).
    The reconstruction is muse_tmr.data.continuous with STAGER_CONTINUOUS_CONFIG.
    """
    continuous = ContinuousBuilder(STAGER_CONTINUOUS_CONFIG)

    async def teed(stream):
        async for frame in stream:
            continuous.add(frame)
            yield frame

    session = ReplaySession(ReplayConfig(input_path=recording_dir, speed=0.0))
    await session.connect()
    detector = HeuristicRemDetector(
        None if weights is None else HeuristicRemConfig(weights=dict(weights))
    )
    rem_tracker = new_rem_event_tracker()
    builder = EpochBuilder(EpochConfig(epoch_seconds=epoch_seconds, stride_seconds=epoch_seconds))
    predictions: List[Tuple[float, RemPrediction]] = []
    try:
        async for epoch in builder.build(teed(session.stream())):
            prediction = detector.predict_epoch(epoch, rem_tracker=rem_tracker)
            predictions.append((epoch.start_time, prediction))
    finally:
        await session.stop()

    try:
        eeg = continuous.result().eeg
    except ValueError:
        eeg = None
    if eeg is None or eeg.n_samples == 0 or eeg.t_last <= eeg.t0:
        raise SystemExit("no usable EEG samples reconstructed from the recording")
    return {
        "predictions": predictions,
        "channels": {name: eeg.channels[name] for name in EEG_CHANNELS},
        "sfreq": float(eeg.measured_rate_hz),
        "t0": float(eeg.t0),
        "n_samples": int(eeg.n_samples),
    }


def collect(
    recording_dir: Path,
    epoch_seconds: float,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    return asyncio.run(_collect(recording_dir, epoch_seconds, weights))


def resample_for_stager(channels: Dict[str, np.ndarray], sfreq: float, target: float = 100.0) -> Tuple[Dict[str, np.ndarray], float]:
    """Resample to ~target Hz with a fast polyphase FIR, preserving real duration.

    The Muse's empirical rate is a non-integer, measured per night as ``sfreq``. YASA resamples to 100
    Hz internally, and doing that from an irrational ratio on a multi-million
    sample array triggers a pathologically slow full-length FFT resample. Pre-
    resampling here (polyphase, O(n)) shrinks the array to ~target*duration
    samples and makes YASA's own step trivial. Duration is preserved, so YASA
    epoch j still maps to real time [t0 + j*30, ...).
    """
    from fractions import Fraction

    from scipy.signal import resample_poly

    frac = Fraction(target / sfreq).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    resampled = {
        name: resample_poly(arr.astype(np.float64), up, down).astype(np.float32)
        for name, arr in channels.items()
    }
    # Declare exactly `target` Hz. The polyphase result is ~target (e.g. 99.999),
    # but a near-unity sfreq makes YASA's internal resample-to-100 pathologically
    # slow; declaring exactly 100 makes it a no-op. The <0.01% rate error is ~0.4s
    # of drift over a full night -- far below the 30s epoch grid.
    return resampled, float(target)


def simulate_gate(
    predictions: Sequence[RemPrediction],
    epoch_seconds: float,
    *,
    optics_capable: bool = True,
    enter_threshold: Optional[float] = None,
    exit_threshold: Optional[float] = None,
) -> List[bool]:
    """Run the real StableRemGate over the p_rem series, in time order.

    ``optics_capable`` mirrors the session preset: on an EEG-only preset (p21)
    the missing-cardiac reason codes must not cap confidence, or the gate never
    opens (see issue #112). ``enter_threshold``/``exit_threshold`` default to the
    recalibrated RemGateConfig defaults when omitted.
    """
    base = RemGateConfig()
    cfg = RemGateConfig(
        epoch_seconds=epoch_seconds,
        optics_capable=optics_capable,
        enter_threshold=base.enter_threshold if enter_threshold is None else enter_threshold,
        exit_threshold=base.exit_threshold if exit_threshold is None else exit_threshold,
    )
    gate = StableRemGate(cfg)
    return [gate.update(pred, duration_seconds=epoch_seconds).gate_open for pred in predictions]


# --------------------------------------------------------------------------- #
# Reference stager (subprocess into the isolated YASA venv)
# --------------------------------------------------------------------------- #
# Reference stagers, each shelled into its own isolated venv (schemas identical).
REFERENCE_SCRIPTS = {"yasa": "_yasa_stage.py", "gssc": "_gssc_stage.py"}


def run_reference(
    reference: str, npz_path: Path, eeg: str, eog: str, python_path: Path
) -> Dict[str, object]:
    stager = Path(__file__).resolve().parent / REFERENCE_SCRIPTS[reference]
    out_path = npz_path.with_suffix(".hypno.json")
    cmd = [
        str(python_path), str(stager),
        "--npz", str(npz_path),
        "--eeg", eeg,
        "--eog", eog,
        "--out", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"{reference} staging failed (exit {result.returncode}).\n"
            f"cmd: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(out_path.read_text())


def resolve_eog(reference: str, eog: Optional[str]) -> str:
    """The --eog value for ``reference``: the one given, or that reference's default."""
    return DEFAULT_EOG[reference] if eog is None else eog


def _eog_channel(eog: Optional[str]) -> Optional[str]:
    """The channel the stager gets as EOG; None for 'none' or '' (as the stagers read it)."""
    return None if eog is None or str(eog).lower() in ("", "none") else str(eog)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def build_provenance(
    *,
    argv: Sequence[str],
    reference: str,
    hypno: Mapping[str, object],
    eeg: str,
    eog: Optional[str],
    preset: Optional[str],
    enter_threshold: float,
    exit_threshold: float,
    resample_hz: float,
    epoch_seconds: float,
    weights: Optional[Mapping[str, float]] = None,
    weight_overrides: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """What produced these metrics: command, versions, channels, thresholds, detector
    weights (``weight_overrides`` as given to --weights, ``weights`` as used), stager.

    The stager versions come from the ``stager`` block the stager script writes into
    its hypnogram JSON, since it runs in its own venv. They are None when the
    hypnogram has no such block.
    """
    import scipy

    eog_channel = _eog_channel(eog)
    passed = {eeg} if eog_channel is None else {eeg, eog_channel}
    stager = hypno.get("stager") or {}
    return {
        "command_line": shlex.join(str(arg) for arg in argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "muse_tmr_version": MUSE_TMR_VERSION,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "resample_hz": float(resample_hz),
        "epoch_seconds": float(epoch_seconds),
        "preset": preset,
        "enter_threshold": float(enter_threshold),
        "exit_threshold": float(exit_threshold),
        "weights": {name: float(w) for name, w in (weights or DEFAULT_REM_WEIGHTS).items()},
        "weight_overrides": {name: float(w) for name, w in (weight_overrides or {}).items()},
        "eeg": eeg,
        "eog": eog_channel,
        "channels_not_passed": [name for name in EEG_CHANNELS if name not in passed],
        "stager": {
            "tool": reference,
            "package_version": stager.get("package_version"),
            "torch_version": stager.get("torch_version"),
            "mne_version": stager.get("mne_version"),
            "returns_probabilities": stager.get("returns_probabilities"),
        },
    }


# --------------------------------------------------------------------------- #
# Metrics (binary_metrics, roc_auc, threshold_sweep, best_f1_threshold and
# gate_open_fraction live in muse_tmr.validation.rem_metrics)
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


def feature_metrics(columns: Mapping[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Dict[str, object]]:
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


def grid_exit_thresholds(enter_threshold: float) -> Tuple[float, ...]:
    steps = int(round((enter_threshold - GRID_LOWEST_EXIT) / 0.05))
    return tuple(round(GRID_LOWEST_EXIT + 0.05 * step, 2) for step in range(steps + 1))


def gate_grid(
    predictions: Sequence[RemPrediction],
    matched_positions: Sequence[int],
    y_true: np.ndarray,
    epoch_seconds: float,
    *,
    optics_capable: bool,
    preset_pair: Tuple[float, float],
) -> List[Dict[str, object]]:
    """The real StableRemGate for every enter/exit pair of the grid, over the whole p_rem
    series, scored on the matched epochs: gate-open fraction, REM recall and precision
    (with F1 and κ) against the reference."""
    rows: List[Dict[str, object]] = []
    for enter in GRID_ENTER_THRESHOLDS:
        for exit_ in grid_exit_thresholds(enter):
            opened = simulate_gate(
                predictions, epoch_seconds, optics_capable=optics_capable,
                enter_threshold=enter, exit_threshold=exit_,
            )
            g = np.array([1 if opened[i] else 0 for i in matched_positions], dtype=int)
            m = binary_metrics(y_true, g)
            rows.append(
                {
                    "enter": enter,
                    "exit": exit_,
                    "preset_pair": (enter, exit_) == tuple(round(t, 2) for t in preset_pair),
                    "gate_open_epochs": int(g.sum()),
                    "gate_open_fraction": gate_open_fraction(g),
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "f1": m["f1"],
                    "kappa": m["kappa"],
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _write_epoch_dump(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write per-epoch rows to CSV (union of columns; blanks where a key is absent)."""
    fixed = ["epoch", "start_time", "p_rem", "reason_codes", "gate_open", "ref_stage", "ref_rem"]
    extra = sorted({k for r in rows for k in r if k not in fixed})
    header = fixed + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def run(
    recording_dir: Path,
    eeg: str,
    eog: str,
    reference_python: Path,
    epoch_seconds: float,
    preset: Optional[str] = None,
    reference: str = "yasa",
    dump_epochs: Optional[Path] = None,
    argv: Optional[Sequence[str]] = None,
    weight_overrides: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    eog = resolve_eog(reference, eog)
    weights = resolve_weights(weight_overrides)
    print(f"[1/4] replaying {recording_dir.name}: heuristic p_rem + EEG reconstruction ...", flush=True)
    if weight_overrides:
        print(
            "      detector weights: "
            + ", ".join(f"{name}={weight:.3f}" for name, weight in weights.items()),
            flush=True,
        )
    collected = collect(recording_dir, epoch_seconds, weights=weights if weight_overrides else None)
    t0 = float(collected["t0"])
    sfreq = float(collected["sfreq"])
    channels: Dict[str, np.ndarray] = collected["channels"]  # type: ignore[assignment]
    predictions: List[Tuple[float, RemPrediction]] = collected["predictions"]  # type: ignore[assignment]
    print(f"      {collected['n_samples']} samples/ch  eff_sfreq={sfreq:.2f} Hz  epochs={len(predictions)}", flush=True)

    print("[2/4] live-gate simulation over p_rem ...", flush=True)
    optics_capable = preset_provides_optics(preset)
    enter_thr, exit_thr = preset_gate_thresholds(preset)
    preds_only = [p for _, p in predictions]
    gate_open = simulate_gate(
        preds_only, epoch_seconds, optics_capable=optics_capable,
        enter_threshold=enter_thr, exit_threshold=exit_thr,
    )
    n_open = sum(gate_open)
    n_min = round(n_open * epoch_seconds / 60, 1)
    print(
        f"      preset={preset or 'unknown'} optics_capable={optics_capable} "
        f"enter={enter_thr:g} exit={exit_thr:g}  "
        f"gate opened on {n_open} epochs ({n_min} min)",
        flush=True,
    )
    if not optics_capable:
        # One replay, two policies: show the pre-fix counterfactual so the
        # EEG-only cueing gain is visible in a single run (issue #112).
        cf_open = sum(simulate_gate(
            preds_only, epoch_seconds, optics_capable=True,
            enter_threshold=enter_thr, exit_threshold=exit_thr,
        ))
        cf_min = round(cf_open * epoch_seconds / 60, 1)
        print(
            f"      counterfactual optics-capable cap (pre-#112-fix): "
            f"{cf_open} epochs ({cf_min} min)",
            flush=True,
        )

    print(f"[3/4] {reference} reference staging (eeg={eeg}, eog={eog}) ...", flush=True)
    channels, stage_sfreq = resample_for_stager(channels, sfreq)
    print(f"      resampled to {stage_sfreq:.2f} Hz for the stager", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "eeg.npz"
        np.savez(
            npz_path,
            sfreq=np.float64(stage_sfreq),
            ch_names=np.array(list(EEG_CHANNELS)),
            **{name: channels[name] for name in EEG_CHANNELS},
        )
        hypno = run_reference(reference, npz_path, eeg, eog, reference_python)
    provenance = build_provenance(
        argv=sys.argv if argv is None else argv,
        reference=reference,
        hypno=hypno,
        eeg=eeg,
        eog=eog,
        preset=preset,
        enter_threshold=enter_thr,
        exit_threshold=exit_thr,
        resample_hz=stage_sfreq,
        epoch_seconds=epoch_seconds,
        weights=weights,
        weight_overrides=weight_overrides,
    )
    hypno_by_index = {int(e["epoch"]): e for e in hypno["epochs"]}
    print(f"      staged {len(hypno_by_index)} epochs; {hypno.get('stage_counts')}", flush=True)

    print("[4/4] aligning epochs and computing metrics ...", flush=True)
    matched = []
    matched_positions = []
    epoch_rows = []
    for position, ((start, pred), opened) in enumerate(zip(predictions, gate_open)):
        k = round((start - t0) / epoch_seconds)
        ref = hypno_by_index.get(k)
        if ref is None:
            continue
        matched_positions.append(position)
        ref_rem = 1 if ref["stage"] in REM_STAGES else 0
        matched.append(
            {
                "epoch": k,
                "p_rem": float(pred.probability),
                "gate_open": bool(opened),
                "ref_stage": ref["stage"],
                "ref_rem": ref_rem,
                "ref_proba_rem": ref.get("proba_rem"),
            }
        )
        if dump_epochs is not None:
            row = {
                "epoch": k,
                "start_time": float(start),
                "p_rem": float(pred.probability),
                "reason_codes": "|".join(pred.reason_codes),
                "gate_open": int(bool(opened)),
                "ref_stage": ref["stage"],
                "ref_rem": ref_rem,
            }
            row.update({f"score_{name}": v for name, v in pred.feature_scores.items()})
            row.update({f"value_{name}": v for name, v in pred.feature_values.items()})
            epoch_rows.append(row)

    if not matched:
        raise SystemExit("no epochs aligned between heuristic and reference -- check timestamps")

    if dump_epochs is not None:
        _write_epoch_dump(dump_epochs, epoch_rows)
        print(f"      per-epoch dump written: {dump_epochs} ({len(epoch_rows)} rows)", flush=True)

    y = np.array([m["ref_rem"] for m in matched], dtype=int)
    p = np.array([m["p_rem"] for m in matched], dtype=float)
    g = np.array([1 if m["gate_open"] else 0 for m in matched], dtype=int)

    reference_flags = _reference_flags(hypno.get("stage_counts", {}), int(y.sum()), len(matched))
    enter, exit_ = RemGateConfig().enter_threshold, RemGateConfig().exit_threshold
    best_thr, best = best_f1_threshold(y, p)
    result = {
        "recording": recording_dir.name,
        "reference": {"tool": reference, "eeg": eeg, "eog": hypno.get("eog"), "note": "proxy, not PSG"},
        "kappa_definition": KAPPA_DEFINITION,
        "provenance": provenance,
        "epochs_matched": len(matched),
        "epoch_seconds": epoch_seconds,
        "ref_rem_epochs": int(y.sum()),
        "ref_rem_fraction": float(y.mean()),
        "reference_flags": reference_flags,
        "mean_p_rem": float(p.mean()),
        "roc_auc": roc_auc(y, p),
        "threshold_metrics": {
            f"enter_{enter:g}": binary_metrics(y, (p >= enter).astype(int)),
            f"exit_{exit_:g}": binary_metrics(y, (p >= exit_).astype(int)),
            "at_0.5": binary_metrics(y, (p >= 0.5).astype(int)),
            f"best_f1_{best_thr:g}": best,
        },
        "gate_vs_ref": {
            **binary_metrics(y, g),
            "gate_open_epochs": int(g.sum()),
            "gate_open_minutes": round(int(g.sum()) * epoch_seconds / 60, 1),
            "enter_threshold": enter_thr,
            "exit_threshold": exit_thr,
            "optics_capable": optics_capable,
            "preset": preset,
        },
        "threshold_sweep": threshold_sweep(y, p),
        "feature_metrics": feature_metrics(
            feature_columns([preds_only[i] for i in matched_positions]), y
        ),
        "gate_grid": gate_grid(
            preds_only, matched_positions, y, epoch_seconds,
            optics_capable=optics_capable, preset_pair=(enter_thr, exit_thr),
        ),
        "matched": matched,
    }
    return result


def _reference_flags(stage_counts: Mapping[str, int], rem_epochs: int, matched: int) -> List[str]:
    """Flag a physiologically implausible YASA hypnogram (a degenerate reference).

    A full night of real sleep has ~15-25% REM and meaningful N3. If the
    reference is far off that, its REM labels are not trustworthy and the
    agreement metrics below cannot validate (or invalidate) the detector.
    """
    total = sum(stage_counts.values()) or 1
    rem_frac = stage_counts.get("REM", stage_counts.get("R", 0)) / total
    deep_frac = stage_counts.get("N3", 0) / total
    flags: List[str] = []
    if matched >= 240:  # only judge plausibility on a full night (>=2h), not a short nap
        if len([s for s, c in stage_counts.items() if c]) < 3:
            flags.append("fewer than 3 sleep stages scored")
        if rem_frac < 0.05:
            flags.append(f"implausibly low REM ({rem_frac * 100:.1f}%; a real night is ~15-25%)")
        if deep_frac < 0.02:
            flags.append(f"implausibly low N3 deep sleep ({deep_frac * 100:.1f}%)")
    if rem_epochs == 0:
        flags.append("zero REM epochs in the reference -- REM precision/recall are undefined")
    return flags


def _fmt(m: Dict[str, float]) -> str:
    def n(x):
        return "  n/a" if (isinstance(x, float) and not math.isfinite(x)) else f"{x:5.2f}"
    return (
        f"P={n(m['precision'])} R={n(m['recall'])} F1={n(m['f1'])} "
        f"kappa={n(m['kappa'])} acc={n(m['accuracy'])} "
        f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})"
    )


REFERENCE_CAVEATS = {
    "yasa": (
        "  CAVEAT: YASA is a proxy, not PSG. Its model expects central electrodes\n"
        "  (C3/C4); the Muse gives only frontal/temporal channels and no EMG, so YASA\n"
        "  mis-scores this montage badly -- treat it as a weak proxy, not a grade."
    ),
    "gssc": (
        "  CAVEAT: GSSC was trained mostly on C3, C4, F3, F4 (NSRR PSG) and stages\n"
        "  from one EEG channel, but there is no published evaluation on forehead-only\n"
        "  EEG. It is an automated proxy, not PSG. Validate against a few\n"
        "  manually-scored nights before treating its REM calls as ground truth."
    ),
}


def stager_label(provenance: Mapping[str, object]) -> str:
    """One line on the stager run: tool, versions, the channels it got and the ones it did not."""
    stager = provenance.get("stager") or {}
    not_passed = ", ".join(provenance.get("channels_not_passed") or []) or "none"
    return (
        f"{stager.get('tool')} {stager.get('package_version') or 'version unknown'} "
        f"(torch {stager.get('torch_version') or 'n/a'}, mne {stager.get('mne_version') or 'n/a'}); "
        f"eeg {provenance.get('eeg')}, eog {provenance.get('eog') or 'none'}; not passed: {not_passed}"
    )


def print_summary(result: Dict[str, object]) -> None:
    tm = result["threshold_metrics"]
    gate = result["gate_vs_ref"]
    tool = str(result.get("reference", {}).get("tool", "reference")).upper()
    print("\n" + "=" * 78)
    print(f"REM detection vs {tool} reference -- {result['recording']}")
    print("=" * 78)
    print(
        f"matched epochs: {result['epochs_matched']}  |  "
        f"{tool} REM: {result['ref_rem_epochs']} epochs "
        f"({result['ref_rem_fraction']*100:.1f}% of night)  |  "
        f"mean p_rem: {result['mean_p_rem']:.2f}"
    )
    flags = result.get("reference_flags") or []
    if flags:
        print("!" * 78)
        print("  DEGENERATE REFERENCE -- the metrics below CANNOT be trusted:")
        for f in flags:
            print(f"    - {f}")
        print(f"  {tool} mis-staged this night (see caveat); the numbers are not a verdict")
        print("  on the detector. A trustworthy reference (manual scoring / PSG) is required.")
        print("!" * 78)
    auc = result["roc_auc"]
    print(f"ROC-AUC (p_rem ranks {tool}-REM): {auc:.3f}" if math.isfinite(auc) else "ROC-AUC: n/a")
    print(f"kappa: {result.get('kappa_definition', KAPPA_DEFINITION)}")
    provenance = result.get("provenance")
    if provenance:
        print(f"stager: {stager_label(provenance)}")
        if provenance.get("weight_overrides"):
            weights = provenance.get("weights") or {}
            print("detector weights (--weights): " + ", ".join(f"{k}={v:.3f}" for k, v in weights.items()))
    print("-" * 78)
    for label, m in tm.items():
        print(f"  p_rem >= {label:<14} {_fmt(m)}")
    print("-" * 78)
    g_enter = gate.get("enter_threshold", 0.80)
    g_exit = gate.get("exit_threshold", 0.70)
    g_preset = gate.get("preset") or "unknown"
    g_optics = gate.get("optics_capable", True)
    print(
        f"  SYSTEM VIEW - real StableRemGate (preset={g_preset} optics_capable={g_optics} "
        f"enter {g_enter:g} / exit {g_exit:g} / 60s stable / 120s cooldown):"
    )
    print(
        f"    gate opened on {gate['gate_open_epochs']} epochs "
        f"({gate['gate_open_minutes']} min).  vs {tool}-REM: {_fmt(gate)}"
    )
    features = result.get("feature_metrics") or {}
    if features:
        print("-" * 78)
        print(f"  PER FEATURE vs {tool}-REM (AUC < 0.5 runs against REM; saturated = one value in every epoch):")
        print(f"    {'column':<34} {'n':>5} {'AUC':>6} {'min':>9} {'max':>9}")
        for name, m in features.items():
            flag = "  saturated" if m["saturated"] else ""
            print(
                f"    {name:<34} {m['n']:>5} {_num(m['auc'], 6, 3)} "
                f"{_num(m['min'], 9, 3)} {_num(m['max'], 9, 3)}{flag}"
            )
    grid = result.get("gate_grid") or []
    if grid:
        print("-" * 78)
        print("  ENTER/EXIT GRID of the gate over p_rem, on the matched epochs (* = this preset's pair):")
        _print_grid(grid, "gate open, % of epochs", "gate_open_fraction", 100.0, 0)
        _print_grid(grid, f"REM recall vs {tool}", "recall", 1.0, 2)
        _print_grid(grid, f"REM precision vs {tool}", "precision", 1.0, 2)
    print("-" * 78)
    print(REFERENCE_CAVEATS.get(tool.lower(), REFERENCE_CAVEATS["yasa"]))
    print("=" * 78 + "\n")


def _num(value: object, width: int, decimals: int) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return f"{'n/a':>{width}}"
    return f"{float(value):>{width}.{decimals}f}"  # type: ignore[arg-type]


def _print_grid(grid: Sequence[Mapping[str, object]], title: str, key: str, scale: float, decimals: int) -> None:
    exits = sorted({float(row["exit"]) for row in grid})  # type: ignore[arg-type]
    print(f"    {title}")
    print("    enter\\exit " + "".join(f"{x:>6.2f} " for x in exits).rstrip())
    for enter in sorted({float(row["enter"]) for row in grid}):  # type: ignore[arg-type]
        cells = {float(row["exit"]): row for row in grid if float(row["enter"]) == enter}  # type: ignore[arg-type]
        line = ""
        for x in exits:
            row = cells.get(x)
            if row is None:
                line += " " * 7
                continue
            value = float(row[key]) * scale  # type: ignore[arg-type]
            text = "n/a" if not math.isfinite(value) else f"{value:.{decimals}f}"
            line += f"{text:>6}{'*' if row['preset_pair'] else ' '}"
        print(f"    {enter:>10.2f} {line}".rstrip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--eeg", default="AF7", help="EEG channel for the stager (default AF7)")
    parser.add_argument(
        "--eog",
        default=None,
        help="EOG channel for the stager, or 'none'. Default: none for gssc (its README "
        "documents horizontal or left/right EOG only), AF8 for yasa.",
    )
    parser.add_argument("--epoch-seconds", type=float, default=30.0)
    parser.add_argument(
        "--preset",
        default=None,
        help="Session preset (e.g. p21, p1034). EEG-only presets (p21) tell the "
        "gate no optics were streamed, so absent PPG does not cap confidence.",
    )
    parser.add_argument(
        "--reference",
        choices=("yasa", "gssc"),
        default="yasa",
        help="Offline reference stager. gssc was trained mostly on C3, C4, F3, F4 "
        "(NSRR PSG) and stages from one EEG channel; no published evaluation on "
        "forehead-only EEG. Both are automated proxies, not PSG.",
    )
    parser.add_argument(
        "--yasa-python",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".venv-yasa" / "bin" / "python",
        help="Python of the isolated venv that has YASA installed",
    )
    parser.add_argument(
        "--gssc-python",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".venv-gssc" / "bin" / "python",
        help="Python of the isolated venv that has GSSC installed",
    )
    parser.add_argument("--output", type=Path, help="Write full metrics JSON here")
    parser.add_argument("--html", type=Path, help="Write an HTML overlay report here")
    parser.add_argument(
        "--dump-epochs",
        type=Path,
        help="Write a per-epoch CSV (p_rem, feature sub-scores/values, reason codes, "
        "reference stage) for calibration analysis.",
    )
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=None,
        help="Replay with other detector weights, e.g. theta_alpha_ratio=0,rem_event_density=0.2. "
        "The weights not named keep their DEFAULT_REM_WEIGHTS value, then all are divided by "
        "their sum. For trying a refit; the pilots always use the defaults.",
    )
    args = parser.parse_args()

    if args.epoch_seconds != 30.0:
        raise SystemExit("YASA stages in fixed 30 s epochs; --epoch-seconds must be 30")

    recording_dir = args.recording_dir.resolve()
    if not recording_dir.exists():
        raise SystemExit(f"recording directory not found: {recording_dir}")

    reference_python = args.gssc_python if args.reference == "gssc" else args.yasa_python
    if not reference_python.exists():
        venv = ".venv-gssc" if args.reference == "gssc" else ".venv-yasa"
        raise SystemExit(
            f"{args.reference} venv python not found: {reference_python}\n"
            f"Create it with:  python -m venv {venv} && {venv}/bin/pip install {args.reference}"
        )

    # NB: do NOT resolve() the venv python -- it is a symlink to the base
    # interpreter, and resolving it would drop the venv's site-packages (numpy,
    # yasa/gssc). Invoke the venv path directly so venv detection kicks in.
    result = run(
        recording_dir, args.eeg, args.eog, reference_python, args.epoch_seconds,
        args.preset, args.reference, args.dump_epochs,
        weight_overrides=args.weights,
    )
    print_summary(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({k: v for k, v in result.items() if k != "matched"}, indent=2))
        print(f"metrics written: {args.output}")
    if args.html:
        from _rem_validation_html import build_html  # local sibling module

        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(build_html(result))
        print(f"html overlay written: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
