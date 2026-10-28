"""REM detection against external labels, and the BOAS harness of validate-external (#226).

Two uses:

- ``load_external_labels()`` gives ``scripts/validate_rem_detection.py --labels`` the
  labels of a file (annotate-template rows or a hypnogram, through
  ``night_events.read_hypnogram()``) or of a BOAS night, whose PSG consensus
  ``stage_hum`` it reads.
- ``run_external_validation()`` is ``muse-tmr validate-external --dataset boas``. It puts
  the participants of BOAS (OpenNeuro ds005555) into a ``fit`` and a ``report`` fold with
  every night of a ``pid`` in the same fold (``make_splits()``); replays each selected
  night's headband EEG through the heuristic detector once and keeps the epochs under
  ``nights/<recording>/`` of the output directory; fits the ``boas_v1`` gate pair and a
  personal model on the fit fold; and scores the report fold, and only it, against the
  consensus: the heuristic with each threshold set, the personal model, GSSC and YASA.

The detector gets the EEG alone (``muse_tmr.data.boas.DETECTOR_MODALITIES``): the
headband IMU is not in the units its files name, so the stillness term and the motion
blocks are left out and the gate runs with ``imu_capable`` and ``optics_capable`` off.
A night is cached once its epochs and stager hypnograms are written, each file
atomically; a run skips a night cached with the same splits hash and code version. The
results are agreement on a frontal montage (Bitbrain Ikon, mastoid reference), not on a
Muse, and say nothing about cueing.
"""

from __future__ import annotations

import asyncio
import bisect
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from muse_tmr import __version__
from muse_tmr.data.boas import (
    BOAS_DATASET_ID,
    BOAS_DATASET_NAME,
    BOAS_EPOCH_SECONDS,
    BOAS_URL,
    DETECTOR_MODALITIES,
    BoasEpochLabel,
    BoasHeadbandSession,
    BoasNight,
    boas_night,
    dataset_doi,
    dataset_version,
    is_boas_night,
    is_synthetic_dataset,
    iter_boas_nights,
    load_boas_hypnogram,
    subject_number,
)
from muse_tmr.data.night_events import SYNTHETIC_NOTE, StageInterval, read_hypnogram
from muse_tmr.models.ml_rem_detector import (
    EEG_PERSONAL_REM_FEATURES,
    PersonalRemClassifierConfig,
    PersonalRemModel,
    train_personal_rem_classifier,
)
from muse_tmr.models.rem_detector import RemPrediction
from muse_tmr.presets import named_gate_threshold_sets, named_gate_thresholds, preset_gate_thresholds
from muse_tmr.validation.rem_metrics import (
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
)
from muse_tmr.validation.rem_reference import (
    REFERENCES,
    StagerError,
    build_provenance,
    collect_from_source,
    file_sha256,
    git_commit,
    resolve_eog,
    stage_eeg,
)

EXTERNAL_KAPPA_DEFINITION = "binary REM-vs-not κ against external labels; not five-class"
CONSENSUS_KAPPA_DEFINITION = "binary REM-vs-not κ against the PSG consensus (BOAS stage_hum); not five-class"
BOAS_LABELS_SOURCE = "boas:stage_hum"
BOAS_LABELS_NOTE = "PSG consensus of three AASM scorers (BOAS stage_hum)"
SCORED_LABELS = ("wake", "nrem", "probable_rem")
REM_LABEL = "probable_rem"
STAGER_REM_STAGES = ("R", "REM")
DATASETS = ("boas",)
FOLDS = ("fit", "report")
SPLITS_SCHEMA_VERSION = 1
HARNESS_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1
DEFAULT_FIT_FRACTION = 0.5
EPOCH_SECONDS = BOAS_EPOCH_SECONDS
STAGER_EEG = "AF7"
P21 = "p21"
GATE_THRESHOLD_CHOICES = (P21,) + named_gate_threshold_sets()
NIGHTS_DIRNAME = "nights"
EPOCHS_FILENAME = "epochs.csv"
NIGHT_FILENAME = "night.json"
REPORT_FILENAME = "report.json"
REPORT_EPOCHS_FILENAME = "report_epochs.csv"
FIT_FILENAME = "fit.json"
METRICS_FILENAME = "metrics.json"
PER_NIGHT_FILENAME = "per_night_metrics.json"
PERSONAL_MODEL_FILENAME = "boas_v1_personal_rem_model.json"
# The grid a fit searches: enter 0.05 to 0.95 and exit 0.05 up to enter, in steps of 0.05
# (190 pairs), wider than the REM check's 0.70 to 0.95: p_rem need not run as high on
# another montage as on a Muse night, and a personal model's probabilities are calibrated.
FIT_ENTER_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
FIT_LOWEST_EXIT = 0.05
FIT_RULE = (
    "the pair of the fit grid (enter 0.05 to 0.95, exit 0.05 up to enter, steps of 0.05) with the highest "
    "pooled binary κ of the gate against stage_hum REM over the scored epochs of the fit nights; ties go to "
    "the higher enter, then the higher exit"
)
SYSTEM_DESCRIPTIONS = {
    "heuristic_p21": "heuristic detector, gate at the p21 pair",
    "heuristic_boas_v1": "heuristic detector, gate at boas_v1 (fitted on the fit fold)",
    "personal": "personal model trained on the fit fold, gate at its own pair (fitted on the fit fold)",
    "gssc": "GSSC on HB_1 (AF7), an offline stager that reads the whole night",
    "yasa": "YASA on HB_1 (AF7) with HB_2 (AF8) as EOG, an offline stager that reads the whole night",
    "stage_ai": "Bitbrain's network on the headband (BOAS stage_ai), trained on BOAS itself; context only",
}
LIMITATIONS = (
    "transfer_not_muse: a Bitbrain Ikon headband referenced to the mastoid, not a Muse referenced near Fpz",
    "eeg_only: the headband IMU is not in the units its files name, so the detector runs without its "
    "stillness term and the gate without motion blocks",
    "binary_kappa: REM against not REM, not five-class",
    "offline_stagers: GSSC and YASA read the whole night; their latency is not that of a causal gate",
    "stage_ai_trained_on_boas: stage_ai comes from Bitbrain's network, trained on BOAS itself in "
    "cross-validation folds; it is context, not a baseline",
)


# What leaves one night out of a run instead of ending it: a file that cannot be read or
# does not hold what BOAS describes, or a stager that fails on it.
NIGHT_ERRORS = (OSError, ValueError, StagerError)


class SplitOverlapError(ValueError):
    """A participant would be fitted on and scored, or a split does not fit the dataset."""


class NotFittedError(ValueError):
    """A threshold set or model that needs the fit step has no values."""


# --------------------------------------------------------------------------- #
# External labels for the REM check script
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExternalLabels:
    """Scored stretches of a night from outside the detector, with where they came from.

    ``intervals`` are relative to the recording's first frame, or in host time when
    ``absolute`` (annotate-template rows). ``source`` is ``boas:stage_hum`` for a BOAS
    night, else ``labels:<file name>``; ``sha256`` is of the file read.
    """

    source: str
    path: Path
    sha256: str
    dataset_version: Optional[str]
    intervals: Tuple[StageInterval, ...]
    note: str

    def interval_for(self, start_time: float, *, t0: float, epoch_seconds: float) -> Optional[StageInterval]:
        """The interval that holds the middle of the epoch starting at ``start_time``."""
        middle = float(start_time) + float(epoch_seconds) / 2.0
        absolute = [interval for interval in self.intervals if interval.absolute]
        relative = [interval for interval in self.intervals if not interval.absolute]
        return _interval_at(absolute, middle) or _interval_at(relative, middle - float(t0))

    def stage_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for interval in self.intervals:
            name = display_stage(interval)
            counts[name] = counts.get(name, 0) + 1
        return counts


def _interval_at(intervals: Sequence[StageInterval], seconds: float) -> Optional[StageInterval]:
    starts = [interval.onset_s for interval in intervals]
    position = bisect.bisect_right(starts, seconds) - 1
    if position >= 0 and seconds < intervals[position].end_s:
        return intervals[position]
    return None


def display_stage(interval: StageInterval) -> str:
    """The stage as the REM check reports it: N1, N2 and N3 kept, else WAKE, REM, NREM
    or UNKNOWN from the label."""
    stage = str(interval.stage).strip().upper()
    if stage in ("N1", "N2", "N3"):
        return stage
    return {"wake": "WAKE", "probable_rem": "REM", "nrem": "NREM"}.get(interval.label, "UNKNOWN")


def load_external_labels(path: Path) -> ExternalLabels:
    """The labels of ``path``: a BOAS night directory, or a file ``read_hypnogram`` reads."""
    path = Path(path).expanduser()
    if is_boas_night(path) or (path.is_dir() and (path / "eeg").is_dir()):
        night = boas_night(path)
        rows = load_boas_hypnogram(night)
        intervals = tuple(
            StageInterval(row.onset_s, row.duration_s, row.stage_name, row.label) for row in rows
        )
        return ExternalLabels(
            source=BOAS_LABELS_SOURCE,
            path=night.psg_events,
            sha256=file_sha256(night.psg_events),
            dataset_version=dataset_version(night.root),
            intervals=intervals,
            note=BOAS_LABELS_NOTE,
        )
    if path.is_dir():
        raise ValueError(f"{path} is a directory but not a BOAS night")
    intervals = read_hypnogram(path)
    return ExternalLabels(
        source=f"labels:{path.name}",
        path=path,
        sha256=file_sha256(path),
        dataset_version=None,
        intervals=tuple(intervals),
        note=f"labels from {path.name}",
    )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_splits(
    nights: Sequence[BoasNight],
    *,
    seed: int = 0,
    fit_fraction: float = DEFAULT_FIT_FRACTION,
    dataset_version_text: Optional[str] = None,
    doi: Optional[str] = None,
    participants_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Participant-grouped fit and report folds.

    The participants (``pid``) are shuffled with ``random.Random(seed)`` and go to the fit
    fold until it holds at least ``fit_fraction`` of the nights; the rest go to the
    report fold. Every night of a participant is in its fold, so no fold separates a
    ``pid``. ValueError for a night without a ``pid``.
    """
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1")
    missing = [night.recording_id for night in nights if not night.pid]
    if missing:
        raise ValueError(f"no pid for {', '.join(missing)}: every night needs its participant")
    by_pid: Dict[str, List[BoasNight]] = {}
    for night in nights:
        by_pid.setdefault(str(night.pid), []).append(night)
    order = sorted(by_pid, key=_pid_sort_key)
    random.Random(int(seed)).shuffle(order)
    target = fit_fraction * len(nights)
    fit_pids: List[str] = []
    fit_count = 0
    for pid in order:
        if fit_count >= target:
            break
        fit_pids.append(pid)
        fit_count += len(by_pid[pid])
    fit_set = set(fit_pids)

    def fold(members: Iterable[BoasNight]) -> List[Dict[str, str]]:
        return [
            {"recording_id": night.recording_id, "pid": str(night.pid)}
            for night in sorted(members, key=lambda night: night.number)
        ]

    return {
        "schema_version": SPLITS_SCHEMA_VERSION,
        "dataset": "boas",
        "dataset_id": BOAS_DATASET_ID,
        "dataset_version": dataset_version_text,
        "dataset_doi": doi,
        "grouping": "pid",
        "seed": int(seed),
        "fit_fraction": float(fit_fraction),
        "participants_tsv_sha256": participants_sha256,
        "folds": {
            "fit": fold(night for night in nights if str(night.pid) in fit_set),
            "report": fold(night for night in nights if str(night.pid) not in fit_set),
        },
    }


def _pid_sort_key(pid: str) -> Tuple[int, int, str]:
    text = str(pid)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def splits_text(splits: Mapping[str, object]) -> str:
    """The splits as written: sorted keys, two-space indent, one trailing newline, so
    the same splits always have the same sha256."""
    return json.dumps(splits, indent=2, sort_keys=True) + "\n"


def splits_sha256(splits: Mapping[str, object]) -> str:
    """The sha256 of ``splits`` as ``splits_text()`` writes them: the sha256 of the
    splits.json it wrote, and the same for a checkout that turned its line ends into CRLF."""
    return hashlib.sha256(splits_text(splits).encode("utf-8")).hexdigest()


def write_splits(splits: Mapping[str, object], path: Path) -> str:
    """Write ``splits`` to ``path`` unless an equal file is there; their sha256
    (``splits_sha256()``). ValueError when ``path`` holds other splits."""
    path = Path(path).expanduser()
    text = splits_text(splits)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(
                f"{path} holds other splits; give another --splits path, or drop --make-splits to use it"
            )
    else:
        _atomic_write(path, text)
    return splits_sha256(splits)


def load_splits(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SPLITS_SCHEMA_VERSION:
        raise ValueError(f"{path} is not a splits file of schema {SPLITS_SCHEMA_VERSION}")
    folds = payload.get("folds")
    if not isinstance(folds, dict) or set(folds) != set(FOLDS):
        raise ValueError(f"{path}: folds must be exactly {', '.join(FOLDS)}")
    return payload


def check_splits(
    splits: Mapping[str, object],
    nights: Sequence[BoasNight],
    *,
    version: Optional[str],
) -> Dict[str, str]:
    """The fold of each recording in ``splits``.

    SplitOverlapError when a participant has nights in both folds, a recording is in
    both or twice, a recording's ``pid`` differs from ``participants.tsv``, a recording
    is not in the dataset, or the splits name another dataset version.
    """
    if splits.get("dataset") != "boas":
        raise SplitOverlapError(f"the splits are for {splits.get('dataset')!r}, not boas")
    if version is not None and splits.get("dataset_version") not in (None, version):
        raise SplitOverlapError(
            f"the splits are for dataset version {splits.get('dataset_version')}, the dataset is {version}"
        )
    known = {night.recording_id: night.pid for night in nights}
    folds: Dict[str, str] = {}
    pid_folds: Dict[str, set] = {}
    for fold_name in FOLDS:
        for entry in splits["folds"][fold_name]:  # type: ignore[index]
            recording_id = str(entry["recording_id"])
            pid = str(entry["pid"])
            if recording_id in folds:
                raise SplitOverlapError(f"{recording_id} is in the splits twice")
            if recording_id not in known:
                raise SplitOverlapError(f"{recording_id} of the splits is not in this dataset")
            if str(known[recording_id]) != pid:
                raise SplitOverlapError(
                    f"{recording_id} has pid {known[recording_id]} in participants.tsv, not {pid}"
                )
            folds[recording_id] = fold_name
            pid_folds.setdefault(pid, set()).add(fold_name)
    split_pids = sorted((pid for pid, names in pid_folds.items() if len(names) > 1), key=_pid_sort_key)
    if split_pids:
        raise SplitOverlapError(
            "participants with nights in both the fit and the report fold: pid "
            + ", ".join(split_pids)
            + "; their report nights would be scored with what was fitted on them"
        )
    return folds


# --------------------------------------------------------------------------- #
# One night: the detector's epochs and the stagers' hypnograms, cached
# --------------------------------------------------------------------------- #
def code_version() -> Dict[str, object]:
    """What a cached night was computed with: the package version, the git commit and
    the harness schema."""
    return {"muse_tmr_version": __version__, "git_commit": git_commit(), "harness_schema": HARNESS_SCHEMA_VERSION}


@dataclass(frozen=True)
class NightJob:
    """One night for ``score_night()``: what to compute and where to keep it.
    ``synthetic`` labels its JSON files ``synthetic input, not sleep``."""

    night: BoasNight
    fold: str
    output_dir: Path
    key: Mapping[str, object]
    references: Tuple[str, ...] = ()
    reference_pythons: Mapping[str, str] = field(default_factory=dict)
    rescore: bool = False
    synthetic: bool = False


def night_dir(output_dir: Path, recording_id: str) -> Path:
    return Path(output_dir) / NIGHTS_DIRNAME / recording_id


def score_night(job: NightJob, *, stager_runner: Optional[Callable] = None) -> Dict[str, object]:
    """Replay a night once and write what is missing: ``epochs.csv`` (labels, ``p_rem``,
    reason codes, feature scores and values per 30 s epoch; no EEG and no clock time)
    and ``<reference>.json`` per stager. ``night.json``, written last, records the key
    and the seconds each part took; a part already there with the same key is kept.
    Returns ``night.json``'s content."""
    out = night_dir(job.output_dir, job.night.recording_id)
    manifest_path = out / NIGHT_FILENAME
    manifest = _read_json(manifest_path) if manifest_path.is_file() and not job.rescore else None
    if manifest is None or manifest.get("key") != dict(job.key):
        manifest = None
    parts: Dict[str, Dict[str, object]] = dict(manifest["parts"]) if manifest else {}
    stager_settings = {reference: _stager_settings(reference) for reference in job.references}
    missing = [
        part
        for part in ("heuristic",) + tuple(job.references)
        if part not in parts
        or not (out / str(parts[part].get("file", ""))).is_file()
        or (part != "heuristic" and parts[part].get("settings") != stager_settings[part])
    ]
    if not missing:
        return {**manifest, "skipped": True}  # type: ignore[dict-item]

    labels = load_boas_hypnogram(job.night)
    started = time.perf_counter()
    stagers = [reference for reference in job.references if reference in missing]
    collected = asyncio.run(
        collect_from_source(
            BoasHeadbandSession(job.night, modalities=DETECTOR_MODALITIES),
            EPOCH_SECONDS,
            keep_eeg=bool(stagers),
        )
    )
    rows = epoch_rows(collected["predictions"], labels, t0=float(collected["t0"]))  # type: ignore[arg-type]
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write(out / EPOCHS_FILENAME, _rows_csv(rows))
    parts["heuristic"] = {
        "file": EPOCHS_FILENAME,
        "epochs": len(rows),
        "seconds": round(time.perf_counter() - started, 3),
    }
    for reference in stagers:
        started = time.perf_counter()
        settings = stager_settings[reference]
        hypno, rate = stage_eeg(
            collected["channels"],  # type: ignore[arg-type]
            float(collected["sfreq"]),  # type: ignore[arg-type]
            reference,
            settings["eeg"],
            settings["eog"],
            Path(job.reference_pythons[reference]),
            runner=stager_runner,
        )
        payload = {
            "recording_id": job.night.recording_id,
            "reference": reference,
            "resample_hz": rate,
            **{key: hypno.get(key) for key in ("eeg", "eog", "epoch_seconds", "stager", "stage_counts")},
            "epochs": [
                {"epoch": int(epoch["epoch"]), "stage": epoch["stage"], "proba_rem": epoch.get("proba_rem")}
                for epoch in hypno["epochs"]  # type: ignore[union-attr]
            ],
        }
        _atomic_write(
            out / f"{reference}.json", json.dumps(_labelled(payload, job.synthetic), indent=1, sort_keys=True) + "\n"
        )
        parts[reference] = {
            "file": f"{reference}.json",
            "settings": settings,
            "seconds": round(time.perf_counter() - started, 3),
        }
    manifest = _labelled(
        {
            "schema_version": HARNESS_SCHEMA_VERSION,
            "recording_id": job.night.recording_id,
            "pid": job.night.pid,
            "fold": job.fold,
            "key": dict(job.key),
            "labels": {"epochs": len(labels), "scored": sum(1 for row in labels if row.label in SCORED_LABELS)},
            "parts": parts,
        },
        job.synthetic,
    )
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "skipped": False}


def _stager_settings(reference: str) -> Dict[str, str]:
    return {"eeg": STAGER_EEG, "eog": resolve_eog(reference, None)}


def epoch_rows(
    predictions: Sequence[Tuple[float, RemPrediction]],
    labels: Sequence[BoasEpochLabel],
    *,
    t0: float,
) -> List[Dict[str, object]]:
    """One row per detector epoch, in time order, with the consensus of the BOAS epoch it
    falls on (``unknown`` past the last scored epoch)."""
    by_index = {row.epoch_index: row for row in labels}
    rows: List[Dict[str, object]] = []
    for start, prediction in predictions:
        index = int(round((float(start) - t0) / EPOCH_SECONDS))
        label = by_index.get(index)
        row: Dict[str, object] = {
            "epoch": index,
            "onset_s": index * EPOCH_SECONDS,
            "stage_hum": "" if label is None or label.stage_hum is None else label.stage_hum,
            "label": "unknown" if label is None else label.label,
            "reference_stage_ai": "" if label is None or label.reference_stage_ai is None else label.reference_stage_ai,
            "psg_stage_ai": "" if label is None or label.psg_stage_ai is None else label.psg_stage_ai,
            "p_rem": float(prediction.probability),
            "reason_codes": ";".join(prediction.reason_codes),
        }
        row.update({f"feature_score_{name}": value for name, value in sorted(prediction.feature_scores.items())})
        row.update({f"feature_value_{name}": value for name, value in sorted(prediction.feature_values.items())})
        rows.append(row)
    return rows


def read_epoch_rows(output_dir: Path, recording_id: str) -> List[Dict[str, object]]:
    """The cached ``epochs.csv`` of a night, numbers as floats and labels as text."""
    path = night_dir(output_dir, recording_id) / EPOCHS_FILENAME
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: Dict[str, object] = {}
            for key, value in raw.items():
                if key in ("label", "reason_codes"):
                    row[key] = value or ""
                elif key in ("epoch", "stage_hum", "reference_stage_ai", "psg_stage_ai"):
                    row[key] = int(float(value)) if value not in ("", None) else None
                else:
                    row[key] = float(value) if value not in ("", None) else math.nan
            rows.append(row)
    return rows


def row_prediction(row: Mapping[str, object]) -> RemPrediction:
    """The heuristic's prediction of a cached row, as the gate and the feature AUC read it."""
    codes = str(row.get("reason_codes") or "")
    return RemPrediction(
        probability=float(row["p_rem"]),  # type: ignore[arg-type]
        reason_codes=tuple(code for code in codes.split(";") if code),
        feature_scores={
            key[len("feature_score_"):]: float(value)  # type: ignore[arg-type]
            for key, value in row.items()
            if key.startswith("feature_score_")
        },
        feature_values={
            key[len("feature_value_"):]: float(value)  # type: ignore[arg-type]
            for key, value in row.items()
            if key.startswith("feature_value_")
        },
        source="heuristic",
    )


# --------------------------------------------------------------------------- #
# Scoring one system on one night
# --------------------------------------------------------------------------- #
@dataclass
class SystemNight:
    """What one system did on one night, on the scored epochs (and every decision)."""

    recording_id: str
    y: np.ndarray
    open: np.ndarray
    score: np.ndarray
    latencies: List[Optional[float]]
    false_open_min: float

    def metrics(self) -> Dict[str, object]:
        return system_metrics(self.y, self.open, self.score, self.latencies, [self.false_open_min])


def score_system_night(
    recording_id: str,
    labels: Sequence[str],
    opened: Sequence[Optional[bool]],
    scores: Sequence[float],
) -> SystemNight:
    """``labels``, ``opened`` and ``scores`` hold one value per detector epoch (epoch k
    at k * 30 s); an epoch whose label is not scored, or whose system has no decision,
    is left out of the metrics. REM runs come from consecutive scored REM epochs, and a
    decision counts at the end of its epoch."""
    scored = [index for index, label in enumerate(labels) if label in SCORED_LABELS and opened[index] is not None]
    y = np.array([1 if labels[index] == REM_LABEL else 0 for index in scored], dtype=int)
    g = np.array([1 if opened[index] else 0 for index in scored], dtype=int)
    s = np.array([float(scores[index]) for index in scored], dtype=float)
    runs = rem_runs(
        [(index * EPOCH_SECONDS, (index + 1) * EPOCH_SECONDS, labels[index] == REM_LABEL) for index in scored]
    )
    decisions = [
        ((index + 1) * EPOCH_SECONDS, bool(value)) for index, value in enumerate(opened) if value is not None
    ]
    return SystemNight(
        recording_id=recording_id,
        y=y,
        open=g,
        score=s,
        latencies=rem_onset_latency_seconds(runs, decisions),
        false_open_min=false_open_minutes(y, g, EPOCH_SECONDS),
    )


def system_metrics(
    y: np.ndarray,
    opened: np.ndarray,
    scores: np.ndarray,
    latencies: Sequence[Optional[float]],
    false_open_per_night: Sequence[float],
) -> Dict[str, object]:
    """κ, F1, precision and recall of the open epochs, precision at recall 0.80 and
    ROC-AUC of the score, REM-onset latency and false-open minutes per night."""
    metrics = binary_metrics(y, opened)
    finite = np.isfinite(scores)
    per_night = np.asarray(false_open_per_night, dtype=float)
    return {
        "epochs": int(y.size),
        "reference_rem_epochs": int(np.sum(y)),
        "open_epochs": int(np.sum(opened)),
        "open_fraction": gate_open_fraction(opened.astype(bool)) if opened.size else float("nan"),
        **metrics,
        "roc_auc": roc_auc(y[finite], scores[finite]) if finite.any() else float("nan"),
        "precision_at_recall_0.80": precision_at_recall(y, scores, 0.80),
        "rem_onset_latency": latency_summary(latencies),
        "false_open_minutes_per_night": {
            "nights": int(per_night.size),
            "mean": float(per_night.mean()) if per_night.size else float("nan"),
            "median": float(np.median(per_night)) if per_night.size else float("nan"),
            "q1": float(np.percentile(per_night, 25)) if per_night.size else float("nan"),
            "q3": float(np.percentile(per_night, 75)) if per_night.size else float("nan"),
        },
    }


def pooled_metrics(nights: Sequence[SystemNight]) -> Dict[str, object]:
    """Every night's scored epochs as one set, the latencies of every REM run, and the
    false-open minutes and κ per night as distributions."""
    if not nights:
        return {"nights": 0}
    y = np.concatenate([night.y for night in nights])
    g = np.concatenate([night.open for night in nights])
    s = np.concatenate([night.score for night in nights])
    latencies = [value for night in nights for value in night.latencies]
    pooled = system_metrics(y, g, s, latencies, [night.false_open_min for night in nights])
    kappas = np.array([binary_metrics(night.y, night.open)["kappa"] for night in nights], dtype=float)
    finite = kappas[np.isfinite(kappas)]
    pooled["nights"] = len(nights)
    pooled["kappa_per_night"] = {
        "nights_defined": int(finite.size),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "q1": float(np.percentile(finite, 25)) if finite.size else float("nan"),
        "q3": float(np.percentile(finite, 75)) if finite.size else float("nan"),
    }
    return pooled


# --------------------------------------------------------------------------- #
# The fit fold: boas_v1 and the personal model
# --------------------------------------------------------------------------- #
def gate_kwargs() -> Dict[str, bool]:
    """How the gate runs on a BOAS night: no optics and no IMU, by design."""
    return {"optics_capable": False, "imu_capable": False}


def fit_gate_pair(series: Sequence[Tuple[Sequence[str], Sequence[RemPrediction]]]) -> Dict[str, object]:
    """The pair of the fit grid with the highest pooled κ of the gate against the
    consensus REM, per ``FIT_RULE``, and every pair's numbers. ``series`` holds each fit
    night's labels and predictions, one per epoch."""
    grid = []
    for enter, exit_ in gate_grid_pairs(FIT_ENTER_THRESHOLDS, FIT_LOWEST_EXIT):
        ys: List[int] = []
        gs: List[int] = []
        for labels, predictions in series:
            opened = simulate_gate(
                predictions, EPOCH_SECONDS, enter_threshold=enter, exit_threshold=exit_, **gate_kwargs()
            )
            for label, is_open in zip(labels, opened):
                if label in SCORED_LABELS:
                    ys.append(1 if label == REM_LABEL else 0)
                    gs.append(1 if is_open else 0)
        metrics = binary_metrics(np.array(ys, dtype=int), np.array(gs, dtype=int))
        grid.append(
            {
                "enter": enter,
                "exit": exit_,
                "kappa": metrics["kappa"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "open_fraction": float(np.mean(gs)) if gs else float("nan"),
            }
        )
    best = max(
        grid,
        key=lambda row: (math.isfinite(row["kappa"]), row["kappa"] if math.isfinite(row["kappa"]) else -1.0,
                         row["enter"], row["exit"]),
    )
    return {"enter": best["enter"], "exit": best["exit"], "kappa": best["kappa"], "rule": FIT_RULE, "grid": grid}


def personal_training_rows(
    rows_by_night: Mapping[str, Sequence[Mapping[str, object]]],
    pids: Mapping[str, Optional[str]],
) -> List[Dict[str, object]]:
    """Training rows of the fit nights: the cached features with ``label``,
    ``recording_id`` and ``pid``; unknown epochs are skipped by the trainer."""
    training = []
    for recording_id, rows in rows_by_night.items():
        for row in rows:
            training.append({**row, "recording_id": recording_id, "pid": pids.get(recording_id) or recording_id})
    return training


def personal_predictions(model: PersonalRemModel, rows: Sequence[Mapping[str, object]]) -> List[RemPrediction]:
    return [model.predict_mapping(row) for row in rows]


def resolve_named_pair(
    name: str,
    fit: Optional[Mapping[str, object]],
    splits_sha256: str,
) -> Dict[str, object]:
    """The (enter, exit) of a threshold set and where it came from.

    ``p21`` is the preset's pair. A named set (``boas_v1``) comes from ``presets.py`` once
    it holds values, and then only for the splits it was fitted on; before that from the
    fit step of this output directory. NotFittedError when neither has it,
    SplitOverlapError when ``presets.py`` has it for other splits.
    """
    if name == P21:
        enter, exit_ = preset_gate_thresholds(P21)
        return {"enter": enter, "exit": exit_, "source": "presets.preset_gate_thresholds('p21')"}
    stored = named_gate_thresholds(name)
    if stored is not None:
        if stored.splits_sha256 != splits_sha256:
            raise SplitOverlapError(
                f"{name} in presets.py was fitted on the fit fold of splits {stored.splits_sha256[:12]}, "
                f"not of these ({splits_sha256[:12]}); its report nights may hold participants it was fitted on"
            )
        return {"enter": stored.enter_threshold, "exit": stored.exit_threshold, "source": "presets.py"}
    fitted = (fit or {}).get("gate_thresholds", {}).get(name) if fit else None  # type: ignore[union-attr]
    if not fitted:
        raise NotFittedError(
            f"{name} has no values yet: presets.py does not hold them and no fit step has run in this output "
            "directory; run validate-external with fit nights selected first"
        )
    return {"enter": fitted["enter"], "exit": fitted["exit"], "source": FIT_FILENAME}


def usable_fit(output_dir: Path, splits_sha: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """``fit.json`` of ``output_dir`` when a report may use it: fitted on the splits of
    sha256 ``splits_sha`` with this code version, and its personal model the file beside
    it. Else None and why not; (None, None) when there is no ``fit.json``."""
    path = Path(output_dir) / FIT_FILENAME
    if not path.is_file():
        return None, None
    fit = _read_json(path)
    if fit.get("splits_sha256") != splits_sha:
        return None, "it was fitted on other splits"
    if fit.get("code") != code_version():
        return None, "it was fitted with another code version"
    personal = fit.get("personal")
    model = Path(output_dir) / PERSONAL_MODEL_FILENAME
    if personal and (not model.is_file() or file_sha256(model) != personal.get("sha256")):  # type: ignore[union-attr]
        return None, f"{PERSONAL_MODEL_FILENAME} is not the model it records"
    return fit, None


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExternalValidationConfig:
    """What ``validate-external`` runs. ``nights`` restricts the run to those recordings
    (a smoke run); ``references`` are the stagers scored on the report fold, each with
    its venv's python in ``reference_pythons``."""

    root: Path
    output_dir: Path
    splits_path: Path
    dataset: str = "boas"
    make_splits: bool = False
    seed: int = 0
    fit_fraction: float = DEFAULT_FIT_FRACTION
    gate_thresholds: Tuple[str, ...] = GATE_THRESHOLD_CHOICES
    references: Tuple[str, ...] = ()
    reference_pythons: Mapping[str, Path] = field(default_factory=dict)
    nights: Optional[Tuple[str, ...]] = None
    jobs: int = 1
    rescore: bool = False
    argv: Tuple[str, ...] = ()

    def validate(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of: {', '.join(DATASETS)}")
        unknown = sorted(set(self.gate_thresholds) - set(GATE_THRESHOLD_CHOICES))
        if unknown or not self.gate_thresholds:
            raise ValueError(f"--gate-thresholds must name some of: {', '.join(GATE_THRESHOLD_CHOICES)}")
        unknown_references = sorted(set(self.references) - set(REFERENCES))
        if unknown_references:
            raise ValueError(f"unknown reference stagers {unknown_references}; known: {', '.join(REFERENCES)}")
        for reference in self.references:
            python = self.reference_pythons.get(reference)
            if python is None or not Path(python).exists():
                raise ValueError(f"{reference} venv python not found: {python}")
        if self.jobs < 1:
            raise ValueError("--jobs must be at least 1")


def run_external_validation(
    config: ExternalValidationConfig,
    *,
    log: Callable[[str], None] = print,
    stager_runner: Optional[Callable] = None,
) -> Dict[str, object]:
    """Splits, the per-night pass, the fit and the report (see the module docstring).

    Returns what was written: the splits and their sha256, the nights scored and their
    seconds, and, with report nights, the pooled metrics. ValueError (SplitOverlapError,
    NotFittedError among them) before any night is replayed when the inputs do not fit.
    """
    config.validate()
    root = Path(config.root).expanduser()
    output_dir = Path(config.output_dir).expanduser()
    nights = list(iter_boas_nights(root))
    version = dataset_version(root)
    synthetic = is_synthetic_dataset(root)
    splits, splits_sha = _resolve_splits(config, root, nights, version)
    folds = check_splits(splits, nights, version=version)
    by_id = {night.recording_id: night for night in nights}

    selected = sorted(folds, key=subject_number)
    if config.nights is not None:
        unknown = [name for name in config.nights if name not in folds]
        if unknown:
            raise ValueError(f"--nights names recordings that are not in the splits: {', '.join(unknown)}")
        selected = sorted(set(config.nights), key=subject_number)
    absent = {name: by_id[name].missing_files() for name in selected if by_id[name].missing_files()}
    if absent:
        listed = "; ".join(f"{name}: {', '.join(files)}" for name, files in list(absent.items())[:5])
        raise ValueError(
            f"{len(absent)} of the {len(selected)} selected nights are not on disk ({listed}"
            + ("; ..." if len(absent) > 5 else "")
            + "); finish the download or pass --nights"
        )
    fit_nights = [name for name in selected if folds[name] == "fit"]
    report_nights = [name for name in selected if folds[name] == "report"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # A run without fit nights reports with the fit already here, and only with one made
    # on these splits by this code, beside the model it records.
    prior_fit, unusable = usable_fit(output_dir, splits_sha)
    for name in config.gate_thresholds:
        if name != P21 and not fit_nights and prior_fit is None and named_gate_thresholds(name) is None:
            raise NotFittedError(
                f"--gate-thresholds {name} needs the fit step: no fit-fold night is selected and "
                + (f"{output_dir / FIT_FILENAME} cannot be used: {unusable}" if unusable
                   else f"{output_dir / FIT_FILENAME} does not exist")
            )
    if unusable and not fit_nights:
        log(f"{FIT_FILENAME} is not used, so the report has no personal model: {unusable}")

    key = {"splits_sha256": splits_sha, **code_version()}
    jobs = [
        NightJob(
            night=by_id[name],
            fold=folds[name],
            output_dir=output_dir,
            key=key,
            references=tuple(config.references) if folds[name] == "report" else (),
            reference_pythons={reference: str(path) for reference, path in config.reference_pythons.items()},
            rescore=config.rescore,
            synthetic=synthetic,
        )
        for name in selected
    ]
    manifests = _run_jobs(jobs, config.jobs, log, stager_runner)
    # A night that could not be read or staged is left out of the fit and the report and
    # named in metrics.json, so that one bad file does not end an unattended run.
    failed = {name: str(manifest["error"]) for name, manifest in manifests.items() if manifest.get("error")}
    fit_nights = [name for name in fit_nights if name not in failed]
    report_nights = [name for name in report_nights if name not in failed]

    fit = _fit(output_dir, fit_nights, by_id, splits, splits_sha, synthetic, log) if fit_nights else prior_fit
    result: Dict[str, object] = {
        "synthetic": synthetic,
        "splits": str(config.splits_path),
        "splits_sha256": splits_sha,
        "output_dir": str(output_dir),
        "nights": {name: _timing(manifests[name]) for name in selected},
        "failed": failed,
        "fit": str(output_dir / FIT_FILENAME) if fit_nights else None,
    }
    if report_nights:
        metrics = _report(
            config, output_dir, report_nights, by_id, fit, splits, splits_sha, version, synthetic, log, failed
        )
        result["metrics"] = str(output_dir / METRICS_FILENAME)
        result["systems"] = {name: block["pooled"] for name, block in metrics["systems"].items()}  # type: ignore[index]
    return _labelled(result, synthetic)


def _resolve_splits(config, root: Path, nights, version) -> Tuple[Dict[str, object], str]:
    splits_path = Path(config.splits_path).expanduser()
    if config.make_splits:
        participants = root / "participants.tsv"
        splits = make_splits(
            nights,
            seed=config.seed,
            fit_fraction=config.fit_fraction,
            dataset_version_text=version,
            doi=dataset_doi(root),
            participants_sha256=file_sha256(participants),
        )
        if is_synthetic_dataset(root):
            splits = _labelled(splits, True)
        return splits, write_splits(splits, splits_path)
    if not splits_path.is_file():
        raise ValueError(f"splits not found: {splits_path}; make them with --make-splits --seed N")
    splits = load_splits(splits_path)
    return splits, splits_sha256(splits)


def _run_jobs(jobs: Sequence[NightJob], workers: int, log, stager_runner) -> Dict[str, Dict[str, object]]:
    """``score_night()`` for every job, in this process or in ``workers`` processes; a
    night that fails gets a manifest with ``error`` instead."""
    manifests: Dict[str, Dict[str, object]] = {}
    total = len(jobs)
    if workers <= 1 or total <= 1 or stager_runner is not None:
        for index, job in enumerate(jobs, 1):
            try:
                manifests[job.night.recording_id] = score_night(job, stager_runner=stager_runner)
            except NIGHT_ERRORS as exc:
                manifests[job.night.recording_id] = _failed(job, exc)
            log(_progress(index, total, manifests[job.night.recording_id]))
        return manifests
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_night, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                manifests[job.night.recording_id] = future.result()
            except NIGHT_ERRORS as exc:
                manifests[job.night.recording_id] = _failed(job, exc)
            log(_progress(index, total, manifests[job.night.recording_id]))
    return manifests


def _failed(job: NightJob, exc: BaseException) -> Dict[str, object]:
    return {"recording_id": job.night.recording_id, "fold": job.fold, "error": f"{type(exc).__name__}: {exc}"}


# metrics.json is committed, so what it says about the run names no local path.
# A path in an error text starts with "/" at the start of a word, with "~/" or with a drive.
_ABSOLUTE_PATH = re.compile(r"(?:(?<![\w.~\\/-])/|~[\\/]|\b[A-Za-z]:[\\/])[^\s'\"]*")


def public_command(config: ExternalValidationConfig) -> Tuple[str, ...]:
    """The command of ``config`` as ``metrics.json`` records it: its options, with
    ``<root>``, ``<splits>``, ``<output-dir>`` and ``<gssc-python>`` or ``<yasa-python>``
    in place of the paths. Each night's ``report.json`` keeps the command as given."""
    argv = ["muse-tmr", "validate-external", "--dataset", config.dataset, "--root", "<root>", "--splits", "<splits>"]
    if config.make_splits:
        argv += ["--make-splits", "--seed", str(config.seed), "--fit-fraction", f"{config.fit_fraction:g}"]
    for name in config.gate_thresholds:
        argv += ["--gate-thresholds", name]
    for reference in config.references:
        argv += ["--reference", reference, f"--{reference}-python", f"<{reference}-python>"]
    argv += ["--output-dir", "<output-dir>"]
    if config.nights is not None:
        argv += ["--nights", ",".join(config.nights)]
    argv += ["--jobs", str(config.jobs)] + (["--rescore"] if config.rescore else [])
    return tuple(argv)


def public_reason(error: str) -> str:
    """What ``metrics.json`` says about a night it left out: the first line of the error,
    every absolute path in it cut to its last part. The log and ``per_night_metrics.json``
    keep the whole text."""
    line = next((text.strip() for text in str(error).splitlines() if text.strip()), "")
    home = str(Path.home())
    if len(home) > 1:
        line = line.replace(home, "~")
    return _ABSOLUTE_PATH.sub(lambda match: _last_part(match.group(0)), line)


def _last_part(path: str) -> str:
    parts = [part for part in re.split(r"[\\/]", path) if part]
    return parts[-1] if parts else path


def _progress(index: int, total: int, manifest: Mapping[str, object]) -> str:
    if manifest.get("error"):
        return f"[{index}/{total}] {manifest['recording_id']} ({manifest['fold']}) failed, left out: {manifest['error']}"
    timing = _timing(manifest)
    state = "cached" if manifest.get("skipped") else "scored"
    parts = timing["seconds"].items()  # type: ignore[union-attr]
    seconds = ", ".join(f"{part} {value:.1f} s" for part, value in parts)
    return f"[{index}/{total}] {manifest['recording_id']} ({manifest['fold']}) {state}: {seconds}"


def _timing(manifest: Mapping[str, object]) -> Dict[str, object]:
    parts = manifest.get("parts") or {}
    return {
        "fold": manifest.get("fold"),
        "cached": bool(manifest.get("skipped")),
        "seconds": {part: float(info.get("seconds", 0.0)) for part, info in parts.items()},  # type: ignore[union-attr]
    }


def _fit(output_dir: Path, fit_nights, by_id, splits, splits_sha: str, synthetic: bool, log) -> Dict[str, object]:
    started = time.perf_counter()
    rows_by_night = {name: read_epoch_rows(output_dir, name) for name in fit_nights}
    series = [
        ([str(row["label"]) for row in rows], [row_prediction(row) for row in rows])
        for rows in rows_by_night.values()
    ]
    boas_v1 = fit_gate_pair(series)
    pids = {name: by_id[name].pid for name in fit_nights}
    training_data = {
        "dataset": "boas",
        "dataset_id": BOAS_DATASET_ID,
        "dataset_version": splits.get("dataset_version"),
        "dataset_doi": splits.get("dataset_doi"),
        "splits_sha256": splits_sha,
        "fold": "fit",
        "labels": BOAS_LABELS_SOURCE,
        "detector_modalities": list(DETECTOR_MODALITIES),
        "recording_ids": sorted(fit_nights, key=subject_number),
        "pids": sorted({str(pid) for pid in pids.values()}, key=_pid_sort_key),
    }
    if synthetic:
        training_data = _labelled(training_data, True)
    model = train_personal_rem_classifier(
        personal_training_rows(rows_by_night, pids),
        PersonalRemClassifierConfig(feature_names=EEG_PERSONAL_REM_FEATURES, group_column="pid"),
        training_data=training_data,
    )
    # As PersonalRemModel.save() writes it, but atomically. fit.json, written after it,
    # records its sha256, so a run that stops in between leaves a fit no report uses.
    model_path = output_dir / PERSONAL_MODEL_FILENAME
    _atomic_write(model_path, json.dumps(model.to_dict(), indent=2, sort_keys=True))
    personal_series = [
        ([str(row["label"]) for row in rows], personal_predictions(model, rows)) for rows in rows_by_night.values()
    ]
    personal_pair = fit_gate_pair(personal_series)
    fit = {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "splits_sha256": splits_sha,
        "code": code_version(),
        "fit_recordings": training_data["recording_ids"],
        "fit_pids": training_data["pids"],
        "gate": gate_kwargs(),
        "gate_thresholds": {"boas_v1": boas_v1},
        "personal": {
            "artifact": PERSONAL_MODEL_FILENAME,
            "sha256": file_sha256(model_path),
            "feature_names": list(model.feature_names),
            "gate_thresholds": personal_pair,
            "training_summary": model.training_summary.to_dict() if model.training_summary else None,
        },
    }
    fit = _labelled(fit, synthetic)
    _atomic_write(output_dir / FIT_FILENAME, json.dumps(_json_safe(fit), indent=2, sort_keys=True) + "\n")
    log(
        f"fit on {len(fit_nights)} nights in {time.perf_counter() - started:.1f} s: boas_v1 enter "
        f"{boas_v1['enter']:g} exit {boas_v1['exit']:g} (κ {boas_v1['kappa']:.3f}); personal model gate "
        f"{personal_pair['enter']:g}/{personal_pair['exit']:g}"
    )
    return fit


def _night_systems(name, rows, labels, predictions, pairs, model, personal_pair, references, output_dir):
    """Every system's outcome on one report night, and its per-epoch flags for
    ``report_epochs.csv``."""
    night_systems: Dict[str, SystemNight] = {}
    flags: Dict[str, Sequence[object]] = {}
    scores = [float(row["p_rem"]) for row in rows]  # type: ignore[arg-type]
    for set_name, pair in pairs.items():
        opened = simulate_gate(
            predictions, EPOCH_SECONDS, enter_threshold=pair["enter"], exit_threshold=pair["exit"], **gate_kwargs()
        )
        night_systems[f"heuristic_{set_name}"] = score_system_night(name, labels, opened, scores)
        flags[f"gate_{set_name}"] = opened
    if model is not None and personal_pair is not None:
        personal = personal_predictions(model, rows)
        opened = simulate_gate(
            personal, EPOCH_SECONDS, enter_threshold=personal_pair["enter"],
            exit_threshold=personal_pair["exit"], **gate_kwargs(),
        )
        probabilities = [prediction.probability for prediction in personal]
        night_systems["personal"] = score_system_night(name, labels, opened, probabilities)
        flags["gate_personal"] = opened
        flags["p_personal"] = probabilities
    for reference in references:
        stager = _read_json(night_dir(output_dir, name) / f"{reference}.json")
        by_epoch = {int(epoch["epoch"]): epoch for epoch in stager["epochs"]}
        stager_rem = [
            (str(by_epoch[index]["stage"]) in STAGER_REM_STAGES) if index in by_epoch else None
            for index in range(len(rows))
        ]
        proba = [
            _float(by_epoch[index].get("proba_rem")) if index in by_epoch else math.nan for index in range(len(rows))
        ]
        night_systems[reference] = score_system_night(name, labels, stager_rem, proba)
        flags[f"{reference}_rem"] = stager_rem
    # The network's -2 (artefact or missing data) is no decision, not a "not REM".
    stage_ai = [
        None if row.get("reference_stage_ai") in (None, -2) else row["reference_stage_ai"] == 4 for row in rows
    ]
    night_systems["stage_ai"] = score_system_night(name, labels, stage_ai, [1.0 if value else 0.0 for value in stage_ai])
    return night_systems, flags


def _report(config, output_dir: Path, report_nights, by_id, fit, splits, splits_sha: str, version, synthetic, log,
            failed=None):
    fit_pids = set(str(pid) for pid in (fit or {}).get("fit_pids", []))
    overlap = sorted({str(by_id[name].pid) for name in report_nights} & fit_pids, key=_pid_sort_key)
    if overlap:
        raise SplitOverlapError(f"report nights of participants the fit used: pid {', '.join(overlap)}")
    pairs = {name: resolve_named_pair(name, fit, splits_sha) for name in config.gate_thresholds}
    model = None
    model_sha = None
    personal_pair = None
    if fit and fit.get("personal") and (output_dir / PERSONAL_MODEL_FILENAME).is_file():
        model_sha = file_sha256(output_dir / PERSONAL_MODEL_FILENAME)
        if model_sha != fit["personal"]["sha256"]:
            raise NotFittedError(f"{PERSONAL_MODEL_FILENAME} is not the model {FIT_FILENAME} records")
        model = PersonalRemModel.load(output_dir / PERSONAL_MODEL_FILENAME)
        model_pids = set(str(pid) for pid in (model.training_data or {}).get("pids", []))
        if model_pids & {str(by_id[name].pid) for name in report_nights}:
            raise SplitOverlapError("the personal model was trained on participants of the report fold")
        personal_pair = fit["personal"]["gate_thresholds"]
    systems: Dict[str, List[SystemNight]] = {}
    per_night: Dict[str, Dict[str, object]] = {}
    feature_predictions: List[RemPrediction] = []
    feature_labels: List[int] = []
    headband = {"scored_epochs": 0, "without_eeg_scores": 0, "nights_mostly_without_eeg": 0}
    for name in report_nights:
        rows = read_epoch_rows(output_dir, name)
        labels = [str(row["label"]) for row in rows]
        predictions = [row_prediction(row) for row in rows]
        night_systems, flags = _night_systems(
            name, rows, labels, predictions, pairs, model, personal_pair, config.references, output_dir
        )
        for system, outcome in night_systems.items():
            systems.setdefault(system, []).append(outcome)
        scored = [index for index, label in enumerate(labels) if label in SCORED_LABELS]
        feature_predictions.extend(predictions[index] for index in scored)
        feature_labels.extend(1 if labels[index] == REM_LABEL else 0 for index in scored)
        # Epochs the detector could not score from the headband EEG (flat, clipped, missing).
        without_eeg = sum(1 for index in scored if "insufficient_features" in predictions[index].reason_codes)
        headband["scored_epochs"] += len(scored)
        headband["without_eeg_scores"] += without_eeg
        headband["nights_mostly_without_eeg"] += int(without_eeg * 2 > len(scored))
        night_metrics = {system: outcome.metrics() for system, outcome in night_systems.items()}
        per_night[name] = {
            "pid": by_id[name].pid,
            "fold": "report",
            "headband": {"scored_epochs": len(scored), "without_eeg_scores": without_eeg},
            "systems": night_metrics,
        }
        report = _labelled(
            {
                "schema_version": METRICS_SCHEMA_VERSION,
                "recording_id": name,
                "kappa_definition": CONSENSUS_KAPPA_DEFINITION,
                "provenance": build_provenance(
                    argv=config.argv,
                    reference=None,
                    hypno={},
                    eeg=None,
                    eog=None,
                    preset=P21,
                    enter_threshold=preset_gate_thresholds(P21)[0],
                    exit_threshold=preset_gate_thresholds(P21)[1],
                    resample_hz=None,
                    epoch_seconds=EPOCH_SECONDS,
                    labels_source=BOAS_LABELS_SOURCE,
                    labels_file_sha256=file_sha256(by_id[name].psg_events),
                    dataset_version=version,
                    stride_seconds=EPOCH_SECONDS,
                ),
                "systems": night_metrics,
            },
            synthetic,
        )
        _atomic_write(
            night_dir(output_dir, name) / REPORT_FILENAME, json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n"
        )
        _atomic_write(night_dir(output_dir, name) / REPORT_EPOCHS_FILENAME, _report_epochs_csv(rows, flags))

    feature_auc = per_feature_auc(feature_columns(feature_predictions), np.array(feature_labels, dtype=int))
    stager_blocks = {}
    for reference in config.references:
        sample = _read_json(night_dir(output_dir, report_nights[0]) / f"{reference}.json")
        settings = _stager_settings(reference)
        stager_blocks[reference] = build_provenance(
            argv=config.argv,
            reference=reference,
            hypno=sample,
            eeg=settings["eeg"],
            eog=settings["eog"],
            preset=None,
            enter_threshold=0.0,
            exit_threshold=0.0,
            resample_hz=sample.get("resample_hz"),
            epoch_seconds=EPOCH_SECONDS,
        )["stager"] | {"eeg": settings["eeg"], "eog": settings["eog"], "resample_hz": sample.get("resample_hz")}
    metrics = _labelled(
        {
            "schema_version": METRICS_SCHEMA_VERSION,
            "dataset": {
                "name": BOAS_DATASET_NAME,
                "id": BOAS_DATASET_ID,
                "version": version,
                "doi": splits.get("dataset_doi"),
                "url": BOAS_URL,
            },
            "splits": {
                "file": Path(config.splits_path).name,
                "sha256": splits_sha,
                "seed": splits.get("seed"),
                "fit_fraction": splits.get("fit_fraction"),
                "grouping": splits.get("grouping"),
            },
            "code": code_version(),
            "kappa_definition": CONSENSUS_KAPPA_DEFINITION,
            "labels": "stage_hum: 0 wake, 1 to 3 nrem, 4 REM; 8 (PSG disconnection), -2 and missing are not scored",
            "detector": {
                "modalities": list(DETECTOR_MODALITIES),
                "channels": {"HB_1": "AF7", "HB_2": "AF8"},
                "gate": gate_kwargs(),
                "epoch_seconds": EPOCH_SECONDS,
            },
            "nights": {
                "report": len(report_nights),
                "report_recordings": report_nights,
                "fit_recordings": list((fit or {}).get("fit_recordings", [])),
                "left_out": {name: public_reason(error) for name, error in (failed or {}).items()},
                "headband": headband,
            },
            "gate_thresholds": {
                **{name: dict(pair) for name, pair in pairs.items()},
                **({"personal": {"enter": personal_pair["enter"], "exit": personal_pair["exit"],
                                 "source": FIT_FILENAME}} if personal_pair else {}),
            },
            "personal_model": (
                {"artifact": PERSONAL_MODEL_FILENAME, "sha256": model_sha,
                 "feature_names": fit["personal"]["feature_names"]}
                if model is not None else {"status": "not fitted"}
            ),
            "stagers": stager_blocks,
            "provenance": build_provenance(
                argv=public_command(config),
                reference=None,
                hypno={},
                eeg=None,
                eog=None,
                preset=P21,
                enter_threshold=preset_gate_thresholds(P21)[0],
                exit_threshold=preset_gate_thresholds(P21)[1],
                resample_hz=None,
                epoch_seconds=EPOCH_SECONDS,
                labels_source=BOAS_LABELS_SOURCE,
                dataset_version=version,
                stride_seconds=EPOCH_SECONDS,
            ),
            "systems": {
                system: {"description": SYSTEM_DESCRIPTIONS.get(system, system), "pooled": pooled_metrics(outcomes)}
                for system, outcomes in systems.items()
            },
            "per_feature_auc": feature_auc,
            "limitations": list(LIMITATIONS)
            + (["small_n: fewer than 10 report nights"] if len(report_nights) < 10 else []),
        },
        synthetic,
    )
    _atomic_write(output_dir / METRICS_FILENAME, json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n")
    _atomic_write(
        output_dir / PER_NIGHT_FILENAME,
        json.dumps(
            _json_safe(_labelled({"splits_sha256": splits_sha, "nights": per_night, "left_out": dict(failed or {})},
                                 synthetic)),
            indent=2, sort_keys=True,
        ) + "\n",
    )
    log(f"report on {len(report_nights)} nights written: {output_dir / METRICS_FILENAME}")
    return metrics


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file beside ``path`` and move it into place, so a
    run that stops half-way never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return payload


def _rows_csv(rows: Sequence[Mapping[str, object]]) -> str:
    fixed = ["epoch", "onset_s", "stage_hum", "label", "reference_stage_ai", "psg_stage_ai", "p_rem", "reason_codes"]
    extra = sorted({key for row in rows for key in row if key not in fixed})
    return _csv_text(fixed + extra, rows)


def _report_epochs_csv(rows: Sequence[Mapping[str, object]], flags: Mapping[str, Sequence[object]]) -> str:
    header = ["epoch", "onset_s", "stage_hum", "label", "reference_stage_ai", "p_rem"] + sorted(flags)
    table = []
    for index, row in enumerate(rows):
        line = {key: row.get(key) for key in header if key in row}
        for name in flags:
            value = flags[name][index]
            line[name] = "" if value is None else (int(value) if isinstance(value, bool) else value)
        table.append(line)
    return _csv_text(header, table)


def _csv_text(header: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(header), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in header})
    return buffer.getvalue()


def _float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _labelled(payload: Mapping[str, object], synthetic: bool) -> Dict[str, object]:
    """``payload`` with ``synthetic`` and its note in front when the dataset is synthetic."""
    if not synthetic:
        return dict(payload)
    rest = {key: value for key, value in payload.items() if key not in ("synthetic", "note")}
    return {"synthetic": True, "note": SYNTHETIC_NOTE, **rest}


def _json_safe(value: object) -> object:
    """NaN and infinity as null, so metrics.json is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value
