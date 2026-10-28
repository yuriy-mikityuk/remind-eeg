"""REM detection models."""

from muse_tmr.models.heuristic_rem_detector import HeuristicRemConfig, HeuristicRemDetector
from muse_tmr.models.ml_rem_detector import (
    DEFAULT_PERSONAL_REM_FEATURES,
    EEG_PERSONAL_REM_FEATURES,
    PersonalRemClassifierConfig,
    PersonalRemModel,
    PersonalRemTrainingSummary,
    train_personal_rem_classifier,
)
from muse_tmr.models.rem_gate import (
    RemConfidence,
    RemGateConfig,
    RemGateDecision,
    StableRemGate,
    build_rem_confidence,
)
from muse_tmr.models.rem_detector import RemPrediction

__all__ = [
    "DEFAULT_PERSONAL_REM_FEATURES",
    "EEG_PERSONAL_REM_FEATURES",
    "HeuristicRemConfig",
    "HeuristicRemDetector",
    "PersonalRemClassifierConfig",
    "PersonalRemModel",
    "PersonalRemTrainingSummary",
    "RemConfidence",
    "RemGateConfig",
    "RemGateDecision",
    "RemPrediction",
    "StableRemGate",
    "build_rem_confidence",
    "train_personal_rem_classifier",
]
