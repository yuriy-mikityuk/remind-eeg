"""Personal REM classifier training and inference.

Version 2 of the artifact (#226) adds ``training_data``, what the model was trained on,
and ``train_personal_rem_classifier(init=...)``, a warm start from another artifact with
the same features: training starts from its coefficients and the L2 penalty pulls
toward them instead of toward 0, so a few labelled nights adjust that model rather than
replace it. Version 1 artifacts still load, with ``training_data`` None.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from muse_tmr.models.rem_detector import RemPrediction

PERSONAL_REM_MODEL_VERSION = 2
SUPPORTED_PERSONAL_REM_MODEL_VERSIONS = (1, 2)
PERSONAL_REM_MODEL_TYPE = "balanced_logistic_regression"

DEFAULT_PERSONAL_REM_FEATURES = (
    "p_rem",
    "feature_score_low_delta_power",
    "feature_score_theta_alpha_ratio",
    "feature_score_eye_movement_proxy",
    "feature_score_stillness",
    "feature_score_hr_variability",
    "feature_score_hr_trend",
    "feature_value_eeg_coverage",
    "feature_value_imu_coverage",
    "feature_value_ppg_coverage",
    "feature_value_heart_rate_coverage",
)

# The features of an epoch read from EEG alone, as validate-external reads a BOAS night
# (#226): the heuristic's EEG scores and the values behind them that do not depend on
# the amplifier's scale. No IMU or cardiac column, which would be missing and imputed in
# every epoch, and no eye-movement proxy, which saturates on real nights (#116). Each
# comes from its own epoch and the REM-event tracker's past, so a prediction never reads
# a later epoch.
EEG_PERSONAL_REM_FEATURES = (
    "p_rem",
    "feature_score_low_delta_power",
    "feature_score_theta_alpha_ratio",
    "feature_score_rem_event_density",
    "feature_value_relative_delta_power",
    "feature_value_theta_alpha_ratio",
    "feature_value_rem_event_density",
    "feature_value_eeg_coverage",
)

POSITIVE_REM_LABEL = "probable_rem"
NEGATIVE_REM_LABELS = ("wake", "nrem")
TRAINING_REM_LABELS = (POSITIVE_REM_LABEL,) + NEGATIVE_REM_LABELS


@dataclass(frozen=True)
class PersonalRemClassifierConfig:
    """Training knobs for the per-user REM classifier."""

    feature_names: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_PERSONAL_REM_FEATURES)
    learning_rate: float = 0.05
    epochs: int = 1200
    l2_penalty: float = 0.01
    decision_threshold: float = 0.5
    min_training_rows: int = 4
    calibration_bins: int = 5
    group_column: str = "recording_id"
    max_group_folds: int = 8
    compute_group_holdout: bool = True

    def validate(self) -> None:
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.l2_penalty < 0:
            raise ValueError("l2_penalty must be non-negative")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1")
        if self.min_training_rows < 2:
            raise ValueError("min_training_rows must be at least 2")
        if self.calibration_bins <= 0:
            raise ValueError("calibration_bins must be positive")
        if self.max_group_folds <= 0:
            raise ValueError("max_group_folds must be positive")


@dataclass(frozen=True)
class PersonalRemTrainingSummary:
    training_rows: int
    positive_rows: int
    negative_rows: int
    skipped_unknown_rows: int
    feature_names: Tuple[str, ...]
    missing_feature_counts: Mapping[str, int]
    metrics: Mapping[str, object]
    group_holdout_metrics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "training_rows": self.training_rows,
            "positive_rows": self.positive_rows,
            "negative_rows": self.negative_rows,
            "skipped_unknown_rows": self.skipped_unknown_rows,
            "feature_names": list(self.feature_names),
            "missing_feature_counts": dict(self.missing_feature_counts),
            "metrics": dict(self.metrics),
            "group_holdout_metrics": dict(self.group_holdout_metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PersonalRemTrainingSummary":
        return cls(
            training_rows=int(payload["training_rows"]),
            positive_rows=int(payload["positive_rows"]),
            negative_rows=int(payload["negative_rows"]),
            skipped_unknown_rows=int(payload.get("skipped_unknown_rows", 0)),
            feature_names=tuple(str(name) for name in payload["feature_names"]),
            missing_feature_counts={
                str(name): int(count)
                for name, count in dict(payload.get("missing_feature_counts", {})).items()
            },
            metrics=dict(payload.get("metrics", {})),
            group_holdout_metrics=dict(payload.get("group_holdout_metrics", {})),
        )


@dataclass(frozen=True)
class PersonalRemModel:
    """A versioned personal REM classifier artifact.

    ``training_data`` (version 2) says what it was trained on: the annotation file or
    the dataset, its split and fold, the recordings and participants, and the artifact
    it was warm-started from (``init``). None for a version 1 artifact.
    """

    model_version: int
    model_type: str
    feature_names: Tuple[str, ...]
    feature_means: Mapping[str, float]
    feature_scales: Mapping[str, float]
    coefficients: Mapping[str, float]
    bias: float
    calibration_intercept: float
    decision_threshold: float
    feature_importance: Mapping[str, float]
    training_summary: Optional[PersonalRemTrainingSummary]
    trained_at_utc: str
    training_data: Optional[Mapping[str, object]] = None

    def predict_mapping(self, row: Mapping[str, object]) -> RemPrediction:
        values, missing_features = _prediction_vector(row, self.feature_names)
        means = np.asarray([self.feature_means[name] for name in self.feature_names], dtype=float)
        scales = np.asarray([self.feature_scales[name] for name in self.feature_names], dtype=float)
        coefficients = np.asarray([self.coefficients[name] for name in self.feature_names], dtype=float)

        filled = np.where(np.isfinite(values), values, means)
        normalized = (filled - means) / scales
        raw_logit = float(np.dot(normalized, coefficients) + self.bias)
        calibrated_logit = raw_logit + self.calibration_intercept
        probability = _sigmoid_scalar(calibrated_logit)
        reason_codes = [
            "personal_model_positive"
            if probability >= self.decision_threshold
            else "personal_model_negative",
            "personal_model_calibrated",
        ]
        if missing_features:
            reason_codes.append("missing_features_imputed")

        return RemPrediction(
            probability=probability,
            reason_codes=tuple(reason_codes),
            feature_scores={"personal_model_probability": probability},
            feature_values={
                "personal_model_logit": calibrated_logit,
                "missing_feature_count": float(len(missing_features)),
            },
            source="personal",
        )

    def predict_rows(self, rows: Iterable[Mapping[str, object]]) -> Tuple[RemPrediction, ...]:
        return tuple(self.predict_mapping(_row_to_mapping(row)) for row in rows)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_version": self.model_version,
            "model_type": self.model_type,
            "trained_at_utc": self.trained_at_utc,
            "feature_names": list(self.feature_names),
            "feature_means": dict(self.feature_means),
            "feature_scales": dict(self.feature_scales),
            "coefficients": dict(self.coefficients),
            "bias": self.bias,
            "calibration_intercept": self.calibration_intercept,
            "decision_threshold": self.decision_threshold,
            "feature_importance": dict(self.feature_importance),
            "training_summary": (
                self.training_summary.to_dict()
                if self.training_summary is not None
                else None
            ),
            "training_data": dict(self.training_data) if self.training_data is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PersonalRemModel":
        """Load a version 1 or 2 artifact; ValueError for another version."""
        model_version = int(payload["model_version"])
        if model_version not in SUPPORTED_PERSONAL_REM_MODEL_VERSIONS:
            raise ValueError(
                f"personal REM model version {model_version} is not supported; "
                f"this version reads {', '.join(str(v) for v in SUPPORTED_PERSONAL_REM_MODEL_VERSIONS)}"
            )
        feature_names = tuple(str(name) for name in payload["feature_names"])
        summary_payload = payload.get("training_summary")
        training_data = payload.get("training_data")
        return cls(
            model_version=model_version,
            model_type=str(payload["model_type"]),
            trained_at_utc=str(payload["trained_at_utc"]),
            feature_names=feature_names,
            feature_means=_float_mapping(payload["feature_means"]),
            feature_scales=_float_mapping(payload["feature_scales"]),
            coefficients=_float_mapping(payload["coefficients"]),
            bias=float(payload["bias"]),
            calibration_intercept=float(payload.get("calibration_intercept", 0.0)),
            decision_threshold=float(payload.get("decision_threshold", 0.5)),
            feature_importance=_float_mapping(payload.get("feature_importance", {})),
            training_summary=(
                PersonalRemTrainingSummary.from_dict(summary_payload)
                if isinstance(summary_payload, MappingABC)
                else None
            ),
            training_data=dict(training_data) if isinstance(training_data, MappingABC) else None,
        )

    def save(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, input_path: Path) -> "PersonalRemModel":
        return cls.from_dict(json.loads(input_path.expanduser().read_text(encoding="utf-8")))


def train_personal_rem_classifier(
    rows: Iterable[Mapping[str, object]],
    config: Optional[PersonalRemClassifierConfig] = None,
    *,
    init: Optional[PersonalRemModel] = None,
    training_data: Optional[Mapping[str, object]] = None,
) -> PersonalRemModel:
    """Train a personal REM model from labeled annotation rows.

    `probable_rem` is the positive class. `wake` and `nrem` are the negative class.
    `unknown` rows are skipped so annotation templates can be edited incrementally.

    ``init`` warm-starts from that artifact: ValueError unless it has the same
    ``feature_names`` as ``config``. Its coefficients and bias, carried over to this
    data's feature means and scales so that the start predicts what ``init`` predicts,
    are where the descent starts and what the L2 penalty pulls toward. ``training_data``
    is stored with the artifact; an ``init`` adds its version, training time and
    ``training_data`` there as ``init``.
    """

    config = config or PersonalRemClassifierConfig()
    config.validate()
    if init is not None and tuple(init.feature_names) != tuple(config.feature_names):
        raise ValueError(
            "the init artifact has the features "
            f"{', '.join(init.feature_names)}; train with the same feature_names, in the same order"
        )

    training_rows, skipped_unknown_rows = _coerce_training_rows(rows)
    targets = np.asarray([_target_for_row(row) for row in training_rows], dtype=float)
    positive_rows = int(targets.sum())
    negative_rows = int(len(targets) - positive_rows)
    _validate_training_counts(
        training_rows=len(training_rows),
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        min_training_rows=config.min_training_rows,
    )

    matrix, means, scales, missing_counts = _training_matrix(training_rows, config.feature_names)
    start = _warm_start(init, config.feature_names, means, scales) if init is not None else None
    coefficients, bias = _fit_balanced_logistic(matrix, targets, config, start=start)
    raw_logits = matrix @ coefficients + bias
    calibration_intercept = _fit_calibration_intercept(raw_logits, targets)
    probabilities = _sigmoid_array(raw_logits + calibration_intercept)
    metrics = _binary_metrics(
        targets,
        probabilities,
        threshold=config.decision_threshold,
        calibration_bins=config.calibration_bins,
    )
    group_holdout_metrics = (
        _group_holdout_metrics(training_rows, config, init=init)
        if config.compute_group_holdout
        else {"status": "disabled"}
    )
    stored_training_data: Optional[Dict[str, object]] = dict(training_data) if training_data is not None else None
    if init is not None:
        stored_training_data = dict(stored_training_data or {})
        stored_training_data["init"] = {
            "model_version": init.model_version,
            "trained_at_utc": init.trained_at_utc,
            "training_data": dict(init.training_data) if init.training_data is not None else None,
        }

    feature_importance = _feature_importance(config.feature_names, coefficients)
    summary = PersonalRemTrainingSummary(
        training_rows=len(training_rows),
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        skipped_unknown_rows=skipped_unknown_rows,
        feature_names=config.feature_names,
        missing_feature_counts=missing_counts,
        metrics=metrics,
        group_holdout_metrics=group_holdout_metrics,
    )

    return PersonalRemModel(
        model_version=PERSONAL_REM_MODEL_VERSION,
        model_type=PERSONAL_REM_MODEL_TYPE,
        trained_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        feature_names=config.feature_names,
        feature_means=means,
        feature_scales=scales,
        coefficients={name: float(value) for name, value in zip(config.feature_names, coefficients)},
        bias=float(bias),
        calibration_intercept=float(calibration_intercept),
        decision_threshold=config.decision_threshold,
        feature_importance=feature_importance,
        training_summary=summary,
        training_data=stored_training_data,
    )


def _warm_start(
    init: PersonalRemModel,
    feature_names: Sequence[str],
    means: Mapping[str, float],
    scales: Mapping[str, float],
) -> Tuple[np.ndarray, float]:
    """``init``'s coefficients and bias on this data's normalisation.

    With ``z = (x - mean) / scale`` here and ``z0 = (x - mean0) / scale0`` in ``init``,
    ``w0 . z0 + b0 + c0`` equals ``w . z + b`` for ``w = w0 * scale / scale0`` and
    ``b = b0 + c0 + sum(w0 * (mean - mean0) / scale0)``; its calibration intercept ``c0``
    goes into the bias, as calibration is refitted afterwards.
    """
    coefficients = np.empty(len(feature_names), dtype=float)
    bias = float(init.bias) + float(init.calibration_intercept)
    for index, name in enumerate(feature_names):
        weight = float(init.coefficients[name])
        init_scale = float(init.feature_scales[name])
        coefficients[index] = weight * float(scales[name]) / init_scale
        bias += weight * (float(means[name]) - float(init.feature_means[name])) / init_scale
    return coefficients, bias


def _coerce_training_rows(
    rows: Iterable[Mapping[str, object]],
) -> Tuple[Tuple[Mapping[str, object], ...], int]:
    training_rows = []
    skipped_unknown_rows = 0
    for row in rows:
        mapping = _row_to_mapping(row)
        label = _label_for_row(mapping)
        if label == "unknown":
            skipped_unknown_rows += 1
            continue
        if label not in TRAINING_REM_LABELS:
            raise ValueError(f"label must be one of: {', '.join(TRAINING_REM_LABELS)}, unknown")
        training_rows.append(mapping)
    return tuple(training_rows), skipped_unknown_rows


def _row_to_mapping(row: object) -> Mapping[str, object]:
    if isinstance(row, MappingABC):
        return row
    to_training_dict = getattr(row, "to_training_dict", None)
    if callable(to_training_dict):
        return to_training_dict()
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError("training rows must be mappings or expose to_training_dict()/to_dict()")


def _label_for_row(row: Mapping[str, object]) -> str:
    return str(row.get("label", "unknown")).strip().lower()


def _target_for_row(row: Mapping[str, object]) -> int:
    return 1 if _label_for_row(row) == POSITIVE_REM_LABEL else 0


def _validate_training_counts(
    *,
    training_rows: int,
    positive_rows: int,
    negative_rows: int,
    min_training_rows: int,
) -> None:
    if training_rows < min_training_rows:
        raise ValueError(
            f"at least {min_training_rows} labeled rows are required; got {training_rows}"
        )
    if positive_rows <= 0 or negative_rows <= 0:
        raise ValueError("training requires at least one probable_rem row and one wake/nrem row")


def _training_matrix(
    rows: Sequence[Mapping[str, object]],
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, Mapping[str, float], Mapping[str, float], Mapping[str, int]]:
    raw_rows: List[List[float]] = []
    missing_counts = {name: 0 for name in feature_names}
    for row in rows:
        values = []
        for name in feature_names:
            value = _float_or_nan(row.get(name))
            if not math.isfinite(value):
                missing_counts[name] += 1
            values.append(value)
        raw_rows.append(values)

    raw = np.asarray(raw_rows, dtype=float)
    means_array = np.empty(raw.shape[1], dtype=float)
    scales_array = np.empty(raw.shape[1], dtype=float)
    for column_index in range(raw.shape[1]):
        column = raw[:, column_index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            means_array[column_index] = 0.0
            scales_array[column_index] = 1.0
            continue
        means_array[column_index] = float(finite.mean())
        scale = float(finite.std())
        scales_array[column_index] = scale if scale > 1e-12 else 1.0

    filled = np.where(np.isfinite(raw), raw, means_array)
    normalized = (filled - means_array) / scales_array
    means = {name: float(value) for name, value in zip(feature_names, means_array)}
    scales = {name: float(value) for name, value in zip(feature_names, scales_array)}
    return normalized, means, scales, missing_counts


def _prediction_vector(
    row: Mapping[str, object],
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    values = []
    missing_features = []
    for name in feature_names:
        value = _float_or_nan(row.get(name))
        if not math.isfinite(value):
            missing_features.append(name)
        values.append(value)
    return np.asarray(values, dtype=float), tuple(missing_features)


def _fit_balanced_logistic(
    matrix: np.ndarray,
    targets: np.ndarray,
    config: PersonalRemClassifierConfig,
    start: Optional[Tuple[np.ndarray, float]] = None,
) -> Tuple[np.ndarray, float]:
    """Gradient descent from zeros and the log prior odds, or from ``start``, which the
    L2 penalty then pulls toward instead of toward 0 (the warm start)."""
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    positive_rows = float(targets.sum())
    negative_rows = float(len(targets) - positive_rows)
    prior = np.clip(positive_rows / len(targets), 1e-6, 1 - 1e-6)
    bias = float(math.log(prior / (1.0 - prior)))
    anchor = np.zeros(matrix.shape[1], dtype=float)
    if start is not None:
        coefficients = np.array(start[0], dtype=float)
        anchor = coefficients.copy()
        bias = float(start[1])
    sample_weights = np.where(
        targets == 1.0,
        len(targets) / (2.0 * positive_rows),
        len(targets) / (2.0 * negative_rows),
    )

    for _ in range(config.epochs):
        probabilities = _sigmoid_array(matrix @ coefficients + bias)
        weighted_error = (probabilities - targets) * sample_weights
        gradient = (matrix.T @ weighted_error) / len(targets)
        gradient += config.l2_penalty * (coefficients - anchor)
        bias_gradient = float(weighted_error.mean())

        coefficients -= config.learning_rate * gradient
        bias -= config.learning_rate * bias_gradient

    return coefficients, bias


def _fit_calibration_intercept(logits: np.ndarray, targets: np.ndarray) -> float:
    intercept = 0.0
    for _ in range(300):
        probabilities = _sigmoid_array(logits + intercept)
        gradient = float((probabilities - targets).mean())
        intercept -= 0.05 * gradient
    return intercept


def _group_holdout_metrics(
    rows: Sequence[Mapping[str, object]],
    config: PersonalRemClassifierConfig,
    init: Optional[PersonalRemModel] = None,
) -> Mapping[str, object]:
    groups: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        group = str(row.get(config.group_column, "")).strip() or "unknown"
        groups.setdefault(group, []).append(row)

    if len(groups) < 2:
        return {"status": "skipped", "reason": "fewer_than_two_recordings"}

    fold_targets = []
    fold_probabilities = []
    fold_count = 0
    fold_config = replace(config, compute_group_holdout=False)
    for group in sorted(groups)[: config.max_group_folds]:
        test_rows = groups[group]
        train_rows = [
            row
            for other_group, group_rows in groups.items()
            if other_group != group
            for row in group_rows
        ]
        if not _can_train_rows(train_rows, fold_config.min_training_rows):
            continue
        model = train_personal_rem_classifier(train_rows, config=fold_config, init=init)
        for row, prediction in zip(test_rows, model.predict_rows(test_rows)):
            fold_targets.append(_target_for_row(row))
            fold_probabilities.append(prediction.probability)
        fold_count += 1

    if not fold_targets:
        return {
            "status": "skipped",
            "reason": "no_trainable_recording_folds",
            "recording_count": len(groups),
        }

    return {
        "status": "computed",
        "recording_count": len(groups),
        "fold_count": fold_count,
        "heldout_rows": len(fold_targets),
        "metrics": _binary_metrics(
            np.asarray(fold_targets, dtype=float),
            np.asarray(fold_probabilities, dtype=float),
            threshold=config.decision_threshold,
            calibration_bins=config.calibration_bins,
        ),
    }


def _can_train_rows(rows: Sequence[Mapping[str, object]], min_training_rows: int) -> bool:
    if len(rows) < min_training_rows:
        return False
    positives = sum(1 for row in rows if _target_for_row(row) == 1)
    negatives = len(rows) - positives
    return positives > 0 and negatives > 0


def _binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    calibration_bins: int,
) -> Mapping[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    target_int = targets.astype(int)
    tp = int(((predictions == 1) & (target_int == 1)).sum())
    tn = int(((predictions == 0) & (target_int == 0)).sum())
    fp = int(((predictions == 1) & (target_int == 0)).sum())
    fn = int(((predictions == 0) & (target_int == 1)).sum())
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)

    return {
        "rows": int(len(targets)),
        "positive_rows": int(target_int.sum()),
        "negative_rows": int(len(targets) - target_int.sum()),
        "threshold": float(threshold),
        "accuracy": _safe_divide(tp + tn, len(targets)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "log_loss": float(
            -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
        ),
        "predicted_positive_rate": float(predictions.mean()),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "calibration_bins": _calibration_bins(targets, probabilities, calibration_bins),
    }


def _calibration_bins(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int,
) -> Tuple[Mapping[str, object], ...]:
    bins = []
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for index in range(bin_count):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bin_count - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not bool(mask.any()):
            continue
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "mean_probability": float(probabilities[mask].mean()),
                "observed_positive_rate": float(targets[mask].mean()),
            }
        )
    return tuple(bins)


def _feature_importance(
    feature_names: Sequence[str],
    coefficients: np.ndarray,
) -> Mapping[str, float]:
    magnitudes = np.abs(coefficients)
    total = float(magnitudes.sum())
    if total <= 0:
        return {name: 0.0 for name in feature_names}
    return {
        name: float(magnitude / total)
        for name, magnitude in zip(feature_names, magnitudes)
    }


def _float_mapping(payload: object) -> Mapping[str, float]:
    return {str(key): float(value) for key, value in dict(payload).items()}


def _float_or_nan(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return math.nan
        return float(stripped)
    return float(value)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _sigmoid_array(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def _sigmoid_scalar(logit: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(min(logit, 50.0), -50.0))))
