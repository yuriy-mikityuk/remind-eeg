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
      thresholds, the detector weights, which Muse channels the stager was and
      was not given, and the labels file, its sha256 and dataset version.

External labels (#226)
    --labels <file or BOAS night> scores the detector and the gate against
    labels instead of a stager: annotate-template rows (.csv/.json), a hypnogram
    muse_tmr.data.night_events.read_hypnogram reads, or a BOAS night directory
    (OpenNeuro ds005555 sub-<n>), whose PSG consensus stage_hum is read. The
    recording can be a BOAS night too; its headband EEG is replayed (EEG only,
    see docs/validation_external.md). reference.tool is then "external", and with
    --reference the stager is also scored against the labels (stager_vs_labels).

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
    # a BOAS night against its PSG consensus, and GSSC against the same labels:
    python scripts/validate_rem_detection.py ds005555/sub-3 --labels ds005555/sub-3 \
        --preset p21 --reference gssc
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from muse_tmr.data.boas import is_boas_night  # noqa: E402
from muse_tmr.models.heuristic_rem_detector import DEFAULT_REM_WEIGHTS  # noqa: E402
from muse_tmr.models.rem_detector import RemPrediction  # noqa: E402
from muse_tmr.models.rem_gate import RemGateConfig  # noqa: E402
from muse_tmr.presets import preset_gate_thresholds, preset_provides_optics  # noqa: E402
from muse_tmr.validation.external_labels import (  # noqa: E402
    EXTERNAL_KAPPA_DEFINITION,
    display_stage,
    load_external_labels,
)
from muse_tmr.validation.rem_metrics import (  # noqa: E402,F401
    GRID_ENTER_THRESHOLDS,
    GRID_LOWEST_EXIT,
    best_f1_threshold,
    binary_metrics,
    false_open_minutes,
    feature_columns,
    gate_open_fraction,
    grid_exit_thresholds,
    latency_summary,
    per_feature_auc,
    precision_at_recall,
    rem_onset_latency_seconds,
    rem_runs,
    roc_auc,
    simulate_gate,
    threshold_sweep,
)
from muse_tmr.validation.rem_reference import (  # noqa: E402,F401
    EEG_CHANNELS,
    StagerError,
    build_provenance,
    collect_from_source,
    night_source,
    resolve_eog,
    run_reference,
    stage_eeg,
)

REM_STAGES = ("R", "REM")
KAPPA_DEFINITION = "binary REM-vs-not κ against an automated stager; not five-class"


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
    """Single pass over raw_amused.bin, or the headband EDF of a BOAS night.

    The heuristic p_rem per epoch and the continuous EEG the stager reads, from
    muse_tmr.validation.rem_reference.collect_from_source with STAGER_CONTINUOUS_CONFIG.
    ``weights`` replaces the detector's DEFAULT_REM_WEIGHTS (see resolve_weights).
    """
    try:
        return await collect_from_source(night_source(recording_dir), epoch_seconds, weights)
    except (ImportError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


def collect(
    recording_dir: Path,
    epoch_seconds: float,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    return asyncio.run(_collect(recording_dir, epoch_seconds, weights))


# --------------------------------------------------------------------------- #
# Metrics (binary_metrics, roc_auc, threshold_sweep, best_f1_threshold,
# gate_open_fraction, per_feature_auc and simulate_gate live in
# muse_tmr.validation.rem_metrics; the stager and the provenance block in
# muse_tmr.validation.rem_reference)
# --------------------------------------------------------------------------- #
def gate_grid(
    predictions: Sequence[RemPrediction],
    matched_positions: Sequence[int],
    y_true: np.ndarray,
    epoch_seconds: float,
    *,
    optics_capable: bool,
    preset_pair: Tuple[float, float],
    imu_capable: bool = True,
) -> List[Dict[str, object]]:
    """The real StableRemGate for every enter/exit pair of the grid, over the whole p_rem
    series, scored on the matched epochs: gate-open fraction, REM recall and precision
    (with F1 and κ) against the reference."""
    rows: List[Dict[str, object]] = []
    for enter in GRID_ENTER_THRESHOLDS:
        for exit_ in grid_exit_thresholds(enter):
            opened = simulate_gate(
                predictions, epoch_seconds, optics_capable=optics_capable,
                enter_threshold=enter, exit_threshold=exit_, imu_capable=imu_capable,
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
    reference: Optional[str] = "yasa",
    dump_epochs: Optional[Path] = None,
    argv: Optional[Sequence[str]] = None,
    weight_overrides: Optional[Mapping[str, float]] = None,
    labels: Optional[Path] = None,
) -> Dict[str, object]:
    """Replay ``recording_dir`` and score the detector and the gate against ``reference``
    (a stager) or, with ``labels``, against those labels; with both, the stager too is
    scored against the labels. ``reference`` None runs no stager (labels only)."""
    if reference is None and labels is None:
        raise SystemExit("give a reference stager, labels, or both")
    if reference is not None:
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
    # A BOAS night is replayed with its EEG alone (its IMU is not in the units it names),
    # so missing IMU is by design there and does not cap the gate's confidence.
    imu_capable = not is_boas_night(recording_dir)
    enter_thr, exit_thr = preset_gate_thresholds(preset)
    preds_only = [p for _, p in predictions]
    gate_open = simulate_gate(
        preds_only, epoch_seconds, optics_capable=optics_capable,
        enter_threshold=enter_thr, exit_threshold=exit_thr, imu_capable=imu_capable,
    )
    n_open = sum(gate_open)
    n_min = round(n_open * epoch_seconds / 60, 1)
    print(
        f"      preset={preset or 'unknown'} optics_capable={optics_capable} "
        + ("" if imu_capable else "imu_capable=False ")
        + f"enter={enter_thr:g} exit={exit_thr:g}  "
        f"gate opened on {n_open} epochs ({n_min} min)",
        flush=True,
    )
    if not optics_capable:
        # One replay, two policies: show the pre-fix counterfactual so the
        # EEG-only cueing gain is visible in a single run (issue #112).
        cf_open = sum(simulate_gate(
            preds_only, epoch_seconds, optics_capable=True,
            enter_threshold=enter_thr, exit_threshold=exit_thr, imu_capable=imu_capable,
        ))
        cf_min = round(cf_open * epoch_seconds / 60, 1)
        print(
            f"      counterfactual optics-capable cap (pre-#112-fix): "
            f"{cf_open} epochs ({cf_min} min)",
            flush=True,
        )

    hypno: Mapping[str, object] = {}
    hypno_by_index: Dict[int, Mapping[str, object]] = {}
    stage_sfreq: Optional[float] = None
    if reference is not None:
        print(f"[3/4] {reference} reference staging (eeg={eeg}, eog={eog}) ...", flush=True)
        try:
            hypno, stage_sfreq = stage_eeg(channels, sfreq, reference, eeg, eog, reference_python, runner=run_reference)
        except StagerError as exc:
            raise SystemExit(str(exc)) from None
        print(f"      resampled to {stage_sfreq:.2f} Hz for the stager", flush=True)
        hypno_by_index = {int(e["epoch"]): e for e in hypno["epochs"]}  # type: ignore[union-attr]
        print(f"      staged {len(hypno_by_index)} epochs; {hypno.get('stage_counts')}", flush=True)
    external = None
    if labels is not None:
        print(("      and " if reference is not None else "[3/4] ") + f"external labels from {labels} ...", flush=True)
        try:
            external = load_external_labels(labels)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--labels: {exc}") from None
        print(f"      {external.source}: {len(external.intervals)} intervals; {external.stage_counts()}", flush=True)
    provenance = build_provenance(
        argv=sys.argv if argv is None else argv,
        reference=reference,
        hypno=hypno,
        eeg=eeg if reference is not None else None,
        eog=eog if reference is not None else None,
        preset=preset,
        enter_threshold=enter_thr,
        exit_threshold=exit_thr,
        resample_hz=stage_sfreq,
        epoch_seconds=epoch_seconds,
        weights=weights,
        weight_overrides=weight_overrides,
        labels_source=external.source if external else None,
        labels_file_sha256=external.sha256 if external else None,
        dataset_version=external.dataset_version if external else None,
        stride_seconds=epoch_seconds,
    )

    print("[4/4] aligning epochs and computing metrics ...", flush=True)
    matched = []
    matched_positions = []
    epoch_rows = []
    stager_pairs: List[Tuple[int, Mapping[str, object]]] = []
    for position, ((start, pred), opened) in enumerate(zip(predictions, gate_open)):
        k = round((start - t0) / epoch_seconds)
        if external is not None:
            interval = external.interval_for(start, t0=t0, epoch_seconds=epoch_seconds)
            if interval is None or interval.label == "unknown":
                continue
            ref_stage, ref_rem, ref_proba = display_stage(interval), int(interval.label == "probable_rem"), None
            if k in hypno_by_index:
                stager_pairs.append((ref_rem, hypno_by_index[k]))
        else:
            ref = hypno_by_index.get(k)
            if ref is None:
                continue
            ref_stage, ref_rem, ref_proba = ref["stage"], 1 if ref["stage"] in REM_STAGES else 0, ref.get("proba_rem")
        matched_positions.append(position)
        matched.append(
            {
                "epoch": k,
                "p_rem": float(pred.probability),
                "gate_open": bool(opened),
                "ref_stage": ref_stage,
                "ref_rem": ref_rem,
                "ref_proba_rem": ref_proba,
            }
        )
        if dump_epochs is not None:
            row = {
                "epoch": k,
                "start_time": float(start),
                "p_rem": float(pred.probability),
                "reason_codes": "|".join(pred.reason_codes),
                "gate_open": int(bool(opened)),
                "ref_stage": ref_stage,
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

    if external is not None:
        reference_flags = _label_flags(int(y.sum()))
        reference_block = {"tool": "external", "labels": external.source, "eeg": None, "eog": None, "note": external.note}
    else:
        reference_flags = _reference_flags(hypno.get("stage_counts", {}), int(y.sum()), len(matched))  # type: ignore[arg-type]
        reference_block = {"tool": reference, "eeg": eeg, "eog": hypno.get("eog"), "note": "proxy, not PSG"}
    enter, exit_ = RemGateConfig().enter_threshold, RemGateConfig().exit_threshold
    best_thr, best = best_f1_threshold(y, p)
    gate_block = {
        **binary_metrics(y, g),
        "gate_open_epochs": int(g.sum()),
        "gate_open_minutes": round(int(g.sum()) * epoch_seconds / 60, 1),
        "enter_threshold": enter_thr,
        "exit_threshold": exit_thr,
        "optics_capable": optics_capable,
        "preset": preset,
    }
    if not imu_capable:
        gate_block["imu_capable"] = False
    result = {
        "recording": recording_dir.name,
        "reference": reference_block,
        "kappa_definition": EXTERNAL_KAPPA_DEFINITION if external is not None else KAPPA_DEFINITION,
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
        "gate_vs_ref": gate_block,
        "threshold_sweep": threshold_sweep(y, p),
        "feature_metrics": per_feature_auc(
            feature_columns([preds_only[i] for i in matched_positions]), y
        ),
        "gate_grid": gate_grid(
            preds_only, matched_positions, y, epoch_seconds,
            optics_capable=optics_capable, preset_pair=(enter_thr, exit_thr), imu_capable=imu_capable,
        ),
        "matched": matched,
    }
    if external is not None:
        result["labels_metrics"] = labels_metrics(
            [(start, opened) for (start, _), opened in zip(predictions, gate_open)],
            [(m["epoch"], m["ref_rem"], m["gate_open"], m["p_rem"]) for m in matched],
            t0=t0,
            epoch_seconds=epoch_seconds,
        )
        if reference is not None:
            result["stager_vs_labels"] = stager_vs_labels(reference, stager_pairs)
    return result


def _label_flags(rem_epochs: int) -> List[str]:
    """External labels are taken as they are; only a night without REM is flagged."""
    if rem_epochs == 0:
        return ["zero REM epochs in the labels -- REM precision/recall are undefined"]
    return []


def labels_metrics(
    decisions: Sequence[Tuple[float, bool]],
    matched: Sequence[Tuple[int, int, bool, float]],
    *,
    t0: float,
    epoch_seconds: float,
) -> Dict[str, object]:
    """Against external labels: precision at recall 0.80 of p_rem, the minutes the gate
    was open outside the labels' REM, and the latency from the start of each REM run of
    the labels to the gate's first open decision in it."""
    y = np.array([row[1] for row in matched], dtype=int)
    g = np.array([1 if row[2] else 0 for row in matched], dtype=int)
    p = np.array([row[3] for row in matched], dtype=float)
    runs = rem_runs([(k * epoch_seconds, (k + 1) * epoch_seconds, bool(rem)) for k, rem, _, _ in matched])
    latencies = rem_onset_latency_seconds(
        runs, [(start - t0 + epoch_seconds, is_open) for start, is_open in decisions]
    )
    return {
        "precision_at_recall_0.80": precision_at_recall(y, p, 0.80),
        "false_open_minutes": false_open_minutes(y, g, epoch_seconds),
        "rem_onset_latency": {**latency_summary(latencies), "seconds": latencies},
    }


def stager_vs_labels(reference: str, pairs: Sequence[Tuple[int, Mapping[str, object]]]) -> Dict[str, object]:
    """The stager's REM epochs against the labels' REM, on the epochs both have."""
    y = np.array([label for label, _ in pairs], dtype=int)
    s = np.array([1 if epoch["stage"] in REM_STAGES else 0 for _, epoch in pairs], dtype=int)
    proba = np.array([_proba(epoch.get("proba_rem")) for _, epoch in pairs], dtype=float)
    finite = np.isfinite(proba)
    return {
        "tool": reference,
        "epochs": int(y.size),
        **binary_metrics(y, s),
        "roc_auc": roc_auc(y[finite], proba[finite]) if finite.any() else float("nan"),
        "precision_at_recall_0.80": precision_at_recall(y, proba, 0.80),
    }


def _proba(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


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
    "external": (
        "  CAVEAT: the labels are taken as they are. A BOAS night is a Bitbrain Ikon\n"
        "  headband referenced to the mastoid, scored against the PSG consensus of three\n"
        "  scorers: a result on a frontal montage, not on a Muse. One night says little;\n"
        "  muse-tmr validate-external scores participant-grouped report folds."
    ),
}


def stager_label(provenance: Mapping[str, object]) -> str:
    """One line on the stager run: tool, versions, the channels it got and the ones it did not."""
    stager = provenance.get("stager") or {}
    if not stager.get("tool"):
        return "none; scored against external labels"
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
    if flags and tool == "EXTERNAL":
        print("!" * 78)
        for f in flags:
            print(f"  {f}")
        print("!" * 78)
    elif flags:
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
        if provenance.get("labels_source"):
            version = provenance.get("dataset_version")
            print(
                f"labels: {provenance['labels_source']}"
                + (f", dataset version {version}" if version else "")
                + f", sha256 {str(provenance.get('labels_file_sha256') or '')[:12]}"
            )
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
    g_imu = "" if gate.get("imu_capable", True) else " imu_capable=False"
    print(
        f"  SYSTEM VIEW - real StableRemGate (preset={g_preset} optics_capable={g_optics}{g_imu} "
        f"enter {g_enter:g} / exit {g_exit:g} / 60s stable / 120s cooldown):"
    )
    print(
        f"    gate opened on {gate['gate_open_epochs']} epochs "
        f"({gate['gate_open_minutes']} min).  vs {tool}-REM: {_fmt(gate)}"
    )
    extra = result.get("labels_metrics")
    if extra:
        at_recall = extra["precision_at_recall_0.80"]
        latency = extra["rem_onset_latency"]
        print(
            f"    p_rem precision at recall 0.80: {_num(at_recall['precision'], 4, 2)} "
            f"(p_rem >= {_num(at_recall['threshold'], 4, 2)});  gate open outside REM: "
            f"{extra['false_open_minutes']:.1f} min;  REM-onset latency median "
            f"{_num(latency['median_s'], 4, 0)} s over {latency['runs_with_open']} of {latency['runs']} REM runs"
        )
    stager_block = result.get("stager_vs_labels")
    if stager_block:
        print("-" * 78)
        print(f"  {str(stager_block['tool']).upper()} vs the labels ({stager_block['epochs']} epochs): {_fmt(stager_block)}")
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
        default=None,
        help="Offline reference stager (default yasa, or none with --labels). gssc was trained "
        "mostly on C3, C4, F3, F4 (NSRR PSG) and stages from one EEG channel; no published "
        "evaluation on forehead-only EEG. Both are automated proxies, not PSG. With --labels "
        "the stager is scored against the labels too.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Score against these labels instead of a stager: annotate-template rows (.csv or "
        ".json), a hypnogram (.edf, BIDS events .tsv, .csv) or a BOAS night directory "
        "(OpenNeuro ds005555 sub-<n>), whose PSG consensus stage_hum is read.",
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
    labels = None
    if args.labels is not None:
        labels = args.labels.resolve()
        if not labels.exists():
            raise SystemExit(f"--labels not found: {labels}")

    reference = args.reference if args.reference is not None or labels is not None else "yasa"
    reference_python = args.gssc_python if reference == "gssc" else args.yasa_python
    if reference is not None and not reference_python.exists():
        venv = ".venv-gssc" if reference == "gssc" else ".venv-yasa"
        raise SystemExit(
            f"{reference} venv python not found: {reference_python}\n"
            f"Create it with:  python -m venv {venv} && {venv}/bin/pip install {reference}"
        )

    # NB: do NOT resolve() the venv python -- it is a symlink to the base
    # interpreter, and resolving it would drop the venv's site-packages (numpy,
    # yasa/gssc). Invoke the venv path directly so venv detection kicks in.
    result = run(
        recording_dir, args.eeg, args.eog, reference_python, args.epoch_seconds,
        args.preset, reference, args.dump_epochs,
        weight_overrides=args.weights,
        labels=labels,
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
