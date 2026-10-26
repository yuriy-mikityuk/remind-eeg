"""Heuristic non-ML REM detector baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

from muse_tmr.features.eeg_features import (
    EEGFeatureRow,
    extract_eeg_features,
    new_rem_event_tracker,
)
from muse_tmr.features.epochs import SleepEpoch
from muse_tmr.features.imu_features import IMUFeatureRow, extract_imu_features
from muse_tmr.features.ppg_features import PPGFeatureRow, extract_ppg_features
from muse_tmr.features.rem_event_features import RemEventTracker
from muse_tmr.models.rem_detector import RemPrediction


# The eye-movement proxy is still scored and reported but carries no weight: on real
# nights it is 1.0 in every epoch (AUC 0.500 against GSSC REM, #116). The other five
# are the earlier 0.25/0.15/0.20/0.10/0.05 divided by 0.75, so they still sum to 1 and
# limited_feature_weight_threshold keeps its meaning. Set by hand; refit pending (#225).
# rem_event_density, its replacement, is scored and exported at weight 0 too until it
# has been fitted on real nights; never fit it on synthetic data.
DEFAULT_REM_WEIGHTS = {
    "low_delta_power": 5 / 15,
    "theta_alpha_ratio": 3 / 15,
    "eye_movement_proxy": 0.0,
    "rem_event_density": 0.0,
    "stillness": 4 / 15,
    "hr_variability": 2 / 15,
    "hr_trend": 1 / 15,
}


@dataclass(frozen=True)
class HeuristicRemConfig:
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_REM_WEIGHTS))
    min_eeg_coverage: float = 0.30
    min_imu_coverage: float = 0.30
    min_ppg_or_hr_coverage: float = 0.30
    rem_delta_relative_power: float = 0.35
    nrem_delta_relative_power: float = 0.65
    theta_alpha_ratio_min: float = 0.70
    theta_alpha_ratio_strong: float = 1.40
    eye_movement_proxy_min: float = 0.05
    eye_movement_proxy_strong: float = 0.20
    # UNFITTED: REM events per minute, starting values until real nights exist (#225).
    rem_event_density_min: float = 1.0
    rem_event_density_strong: float = 6.0
    stillness_min: float = 0.85
    stillness_strong: float = 0.97
    hrv_rmssd_ms_min: float = 15.0
    hrv_rmssd_ms_strong: float = 80.0
    hr_trend_abs_bpm_per_min_min: float = 5.0
    hr_trend_abs_bpm_per_min_strong: float = 40.0
    # HRV scored from PPG peaks whose ppg_confidence is below this adds the reason code
    # low_ppg_quality; scores and P_REM stay as they are (#234).
    low_ppg_quality_confidence: float = 0.40
    limited_feature_weight_threshold: float = 0.40
    limited_feature_probability_cap: float = 0.55
    motion_arousal_probability_cap: float = 0.35

    def validate(self) -> None:
        if not 0 <= self.min_eeg_coverage <= 1:
            raise ValueError("min_eeg_coverage must be between 0 and 1")
        if not 0 <= self.min_imu_coverage <= 1:
            raise ValueError("min_imu_coverage must be between 0 and 1")
        if not 0 <= self.min_ppg_or_hr_coverage <= 1:
            raise ValueError("min_ppg_or_hr_coverage must be between 0 and 1")
        if not 0 <= self.rem_delta_relative_power < self.nrem_delta_relative_power <= 1:
            raise ValueError("delta relative-power thresholds must be ordered inside 0..1")
        if self.theta_alpha_ratio_min < 0 or self.theta_alpha_ratio_strong <= self.theta_alpha_ratio_min:
            raise ValueError("theta/alpha thresholds must be ordered")
        if self.eye_movement_proxy_min < 0 or self.eye_movement_proxy_strong <= self.eye_movement_proxy_min:
            raise ValueError("eye-movement thresholds must be ordered")
        if self.rem_event_density_min < 0 or self.rem_event_density_strong <= self.rem_event_density_min:
            raise ValueError("REM-event density thresholds must be ordered")
        if not 0 <= self.stillness_min < self.stillness_strong <= 1:
            raise ValueError("stillness thresholds must be ordered inside 0..1")
        if self.hrv_rmssd_ms_min < 0 or self.hrv_rmssd_ms_strong <= self.hrv_rmssd_ms_min:
            raise ValueError("HRV thresholds must be ordered")
        if (
            self.hr_trend_abs_bpm_per_min_min < 0
            or self.hr_trend_abs_bpm_per_min_strong <= self.hr_trend_abs_bpm_per_min_min
        ):
            raise ValueError("HR trend thresholds must be ordered")
        if not 0 <= self.low_ppg_quality_confidence <= 1:
            raise ValueError("low_ppg_quality_confidence must be between 0 and 1")
        if not 0 <= self.limited_feature_weight_threshold:
            raise ValueError("limited_feature_weight_threshold must be non-negative")
        if not 0 <= self.limited_feature_probability_cap <= 1:
            raise ValueError("limited_feature_probability_cap must be between 0 and 1")
        if not 0 <= self.motion_arousal_probability_cap <= 1:
            raise ValueError("motion_arousal_probability_cap must be between 0 and 1")
        for name, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"weight for {name} must be non-negative")


class HeuristicRemDetector:
    """Non-ML REM baseline built from sleep feature rows.

    The detector only returns REM probability and reason codes. It intentionally does
    not call cue playback or scheduler code; later stable-gate and safety layers must
    decide whether a probability is actionable.
    """

    def __init__(self, config: Optional[HeuristicRemConfig] = None) -> None:
        self.config = config or HeuristicRemConfig()
        self.config.validate()

    def predict_epoch(
        self,
        epoch: SleepEpoch,
        *,
        rem_tracker: Optional[RemEventTracker] = None,
    ) -> RemPrediction:
        """Predict one epoch; pass the session's REM-event tracker when epochs follow
        one another (see ``extract_eeg_features``)."""
        return self.predict_features(
            eeg=extract_eeg_features(epoch, rem_tracker=rem_tracker),
            imu=extract_imu_features(epoch),
            ppg=extract_ppg_features(epoch),
        )

    def predict_epochs(self, epochs: Iterable[SleepEpoch]) -> Tuple[RemPrediction, ...]:
        """Predict consecutive epochs of one session with one REM-event tracker."""
        rem_tracker = new_rem_event_tracker()
        return tuple(self.predict_epoch(epoch, rem_tracker=rem_tracker) for epoch in epochs)

    def predict_features(
        self,
        *,
        eeg: Optional[EEGFeatureRow] = None,
        imu: Optional[IMUFeatureRow] = None,
        ppg: Optional[PPGFeatureRow] = None,
    ) -> RemPrediction:
        scores: Dict[str, float] = {}
        values: Dict[str, float] = {}
        reasons = []

        self._score_eeg(eeg, scores, values, reasons)
        self._score_imu(imu, scores, values, reasons)
        self._score_ppg(ppg, scores, values, reasons)

        probability, available_weight = self._weighted_probability(scores)
        if available_weight <= 0:
            reasons.append("insufficient_features")
            probability = 0.0
        elif available_weight < self.config.limited_feature_weight_threshold:
            reasons.append("limited_feature_support")
            probability = min(probability, self.config.limited_feature_probability_cap)

        if imu is not None and "motion_arousal_proxy" in imu.arousal_guard_reason_codes:
            reasons.append("motion_arousal_proxy")
            probability = min(probability, self.config.motion_arousal_probability_cap)

        return RemPrediction(
            probability=_clamp01(probability),
            reason_codes=_unique(reasons),
            feature_scores=scores,
            feature_values=values,
            source="heuristic",
        )

    def _score_eeg(
        self,
        eeg: Optional[EEGFeatureRow],
        scores: Dict[str, float],
        values: Dict[str, float],
        reasons: list,
    ) -> None:
        if eeg is None:
            reasons.append("eeg_features_missing")
            return
        values["eeg_coverage"] = eeg.eeg_coverage
        if eeg.eeg_coverage < self.config.min_eeg_coverage:
            reasons.append("low_eeg_coverage")
            return

        relative_delta = eeg.relative_band_powers.get("delta", math.nan)
        values["relative_delta_power"] = relative_delta
        if math.isfinite(relative_delta):
            scores["low_delta_power"] = _low_score(
                relative_delta,
                self.config.rem_delta_relative_power,
                self.config.nrem_delta_relative_power,
            )
            reasons.append(
                "eeg_low_delta_support"
                if scores["low_delta_power"] >= 0.5
                else "eeg_delta_power_high"
            )

        theta_alpha = eeg.ratios.get("theta_alpha_ratio", math.nan)
        values["theta_alpha_ratio"] = theta_alpha
        if math.isfinite(theta_alpha):
            scores["theta_alpha_ratio"] = _high_score(
                theta_alpha,
                self.config.theta_alpha_ratio_min,
                self.config.theta_alpha_ratio_strong,
            )
            reasons.append(
                "eeg_theta_alpha_support"
                if scores["theta_alpha_ratio"] >= 0.5
                else "eeg_theta_alpha_low"
            )

        eye_proxy = eeg.eye_movement_proxy
        values["eye_movement_proxy"] = eye_proxy
        if math.isfinite(eye_proxy):
            scores["eye_movement_proxy"] = _high_score(
                eye_proxy,
                self.config.eye_movement_proxy_min,
                self.config.eye_movement_proxy_strong,
            )
            reasons.append(
                "eeg_eye_movement_support"
                if scores["eye_movement_proxy"] >= 0.5
                else "eeg_eye_movement_low"
            )

        rem_density = eeg.rem_density_per_min
        values["rem_event_density"] = rem_density
        values["rem_event_count"] = eeg.rem_event_count
        values["rem_event_threshold_uv"] = eeg.rem_event_threshold_uv
        values["rem_event_warmup"] = 1.0 if eeg.rem_event_warmup else 0.0
        if math.isfinite(rem_density):
            scores["rem_event_density"] = _high_score(
                rem_density,
                self.config.rem_event_density_min,
                self.config.rem_event_density_strong,
            )
            reasons.append(
                "eeg_rem_events_support"
                if scores["rem_event_density"] >= 0.5
                else "eeg_rem_events_low"
            )

        if eeg.artifact_flags:
            reasons.append("eeg_artifact_flags")

    def _score_imu(
        self,
        imu: Optional[IMUFeatureRow],
        scores: Dict[str, float],
        values: Dict[str, float],
        reasons: list,
    ) -> None:
        if imu is None:
            reasons.append("imu_features_missing")
            return
        values["imu_coverage"] = imu.imu_coverage
        values["stillness_score"] = imu.stillness_score
        values["motion_level"] = imu.motion_level
        if imu.imu_coverage < self.config.min_imu_coverage:
            reasons.append("low_imu_coverage")
            return
        if math.isfinite(imu.stillness_score):
            scores["stillness"] = _high_score(
                imu.stillness_score,
                self.config.stillness_min,
                self.config.stillness_strong,
            )
            reasons.append(
                "imu_stillness_support"
                if scores["stillness"] >= 0.5
                else "imu_motion_present"
            )
        if imu.arousal_guard_reason_codes:
            reasons.extend(imu.arousal_guard_reason_codes)

    def _score_ppg(
        self,
        ppg: Optional[PPGFeatureRow],
        scores: Dict[str, float],
        values: Dict[str, float],
        reasons: list,
    ) -> None:
        if ppg is None:
            reasons.append("ppg_features_missing")
            return

        usable_coverage = max(ppg.ppg_coverage, ppg.heart_rate_coverage)
        values["ppg_coverage"] = ppg.ppg_coverage
        values["heart_rate_coverage"] = ppg.heart_rate_coverage
        values["mean_hr_bpm"] = ppg.mean_hr_bpm
        values["rmssd_ms"] = ppg.rmssd_ms
        values["hr_trend_bpm_per_min"] = ppg.hr_trend_bpm_per_min
        if usable_coverage < self.config.min_ppg_or_hr_coverage:
            reasons.append("low_ppg_hr_coverage")
            return

        if math.isfinite(ppg.rmssd_ms):
            scores["hr_variability"] = _high_score(
                ppg.rmssd_ms,
                self.config.hrv_rmssd_ms_min,
                self.config.hrv_rmssd_ms_strong,
            )
            reasons.append(
                "hr_variability_support"
                if scores["hr_variability"] >= 0.5
                else "hr_variability_low"
            )
        if ppg.hrv_source == "ppg" and ppg.ppg_confidence < self.config.low_ppg_quality_confidence:
            # The peak intervals behind that HRV are irregular; the score is kept.
            reasons.append("low_ppg_quality")

        if math.isfinite(ppg.hr_trend_bpm_per_min):
            abs_trend = abs(ppg.hr_trend_bpm_per_min)
            scores["hr_trend"] = _high_score(
                abs_trend,
                self.config.hr_trend_abs_bpm_per_min_min,
                self.config.hr_trend_abs_bpm_per_min_strong,
            )
            if scores["hr_trend"] >= 0.5:
                reasons.append("hr_trend_support")

        if ppg.artifact_flags:
            reasons.append("ppg_artifact_flags")

    def _weighted_probability(self, scores: Mapping[str, float]) -> Tuple[float, float]:
        weighted_total = 0.0
        available_weight = 0.0
        for name, score in scores.items():
            if not math.isfinite(score):
                continue
            weight = float(self.config.weights.get(name, 0.0))
            if weight <= 0:
                continue
            weighted_total += weight * _clamp01(score)
            available_weight += weight
        if available_weight <= 0:
            return 0.0, 0.0
        return weighted_total / available_weight, available_weight


def _low_score(value: float, full_score_at_or_below: float, zero_score_at_or_above: float) -> float:
    if not math.isfinite(value):
        return math.nan
    if value <= full_score_at_or_below:
        return 1.0
    if value >= zero_score_at_or_above:
        return 0.0
    return _clamp01(
        (zero_score_at_or_above - value)
        / (zero_score_at_or_above - full_score_at_or_below)
    )


def _high_score(value: float, zero_score_at_or_below: float, full_score_at_or_above: float) -> float:
    if not math.isfinite(value):
        return math.nan
    if value <= zero_score_at_or_below:
        return 0.0
    if value >= full_score_at_or_above:
        return 1.0
    return _clamp01(
        (value - zero_score_at_or_below)
        / (full_score_at_or_above - zero_score_at_or_below)
    )


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))
