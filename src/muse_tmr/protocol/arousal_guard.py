"""Conservative arousal guard over sleep feature rows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from muse_tmr.features.eeg_features import EEGFeatureRow
from muse_tmr.features.imu_features import IMUFeatureRow
from muse_tmr.features.ppg_features import (
    HeartRatePoint,
    PPGFeatureRow,
    heart_rate_change_across_epochs,
)

AROUSAL_GUARD_ACTIONS = ("allow", "lower_volume", "pause", "stop")
REPEATED_ARTIFACT_ACTIONS = ("pause", "stop")

PAUSE_IMU_REASON_CODES = (
    "motion_arousal_proxy",
    "cue_related_arousal",
    "motion_not_still",
)

# PPG flags that only say the optics signal is absent. On a preset that has no
# optics by design (see ArousalGuardConfig.optics_capable) they say nothing about
# the sleeper. low_ppg_coverage means "present but sparse", so an EEG-only epoch
# never carries it; it is listed in case optics data turns up anyway.
PPG_ABSENCE_FLAGS = (
    "ppg_missing",
    "missing_ppg",
    "heart_rate_missing",
    "missing_heart_rate",
    "low_heart_rate_coverage",
    "low_ppg_coverage",
)


@dataclass(frozen=True)
class ArousalGuardConfig:
    enabled: bool = True
    lower_volume_multiplier: float = 0.50
    pause_seconds: float = 120.0
    stop_after_consecutive_pause_epochs: int = 3
    stop_after_consecutive_artifact_epochs: int = 3
    repeated_artifact_action: str = "pause"
    critical_eeg_bad_channel_count: int = 3
    stop_after_consecutive_critical_artifact_epochs: int = 3
    min_stillness_score_for_cue: float = 0.90
    alpha_lower_volume_relative_power: float = 0.25
    alpha_pause_relative_power: float = 0.40
    # The HR rule pauses when at least sudden_hr_change_pause_count heart-rate changes of
    # sudden_hr_change_pause_bpm or more are found: the row's own (sudden_hr_changes,
    # logged from PPGFeatureConfig.sudden_hr_change_bpm up) and the largest change between
    # the previous row's heart-rate series and this one's, for values at most
    # sudden_hr_change_link_seconds apart (0 leaves that comparison out).
    sudden_hr_change_pause_count: int = 1
    sudden_hr_change_pause_bpm: float = 10.0
    sudden_hr_change_link_seconds: float = 20.0
    # Whether the session preset streams optics/PPG at all, as in RemGateConfig.
    # When False (an EEG-only preset such as p21) the PPG-absence flags are not
    # artifacts, the HR rules are skipped and decisions carry
    # metadata["ppg_absent_by_design"]. Defaults True for optics presets.
    optics_capable: bool = True

    def validate(self) -> None:
        if not 0.0 < self.lower_volume_multiplier <= 1.0:
            raise ValueError("lower_volume_multiplier must be inside (0, 1]")
        if self.pause_seconds < 0:
            raise ValueError("pause_seconds must be non-negative")
        if self.stop_after_consecutive_pause_epochs <= 0:
            raise ValueError("stop_after_consecutive_pause_epochs must be positive")
        if self.stop_after_consecutive_artifact_epochs <= 0:
            raise ValueError("stop_after_consecutive_artifact_epochs must be positive")
        if self.repeated_artifact_action not in REPEATED_ARTIFACT_ACTIONS:
            raise ValueError(
                f"repeated_artifact_action must be one of: {', '.join(REPEATED_ARTIFACT_ACTIONS)}"
            )
        if self.critical_eeg_bad_channel_count <= 0:
            raise ValueError("critical_eeg_bad_channel_count must be positive")
        if self.stop_after_consecutive_critical_artifact_epochs <= 0:
            raise ValueError("stop_after_consecutive_critical_artifact_epochs must be positive")
        if not 0.0 <= self.min_stillness_score_for_cue <= 1.0:
            raise ValueError("min_stillness_score_for_cue must be between 0 and 1")
        if not (
            0.0
            <= self.alpha_lower_volume_relative_power
            <= self.alpha_pause_relative_power
            <= 1.0
        ):
            raise ValueError("alpha thresholds must satisfy 0 <= lower <= pause <= 1")
        if self.sudden_hr_change_pause_count <= 0:
            raise ValueError("sudden_hr_change_pause_count must be positive")
        if self.sudden_hr_change_pause_bpm <= 0:
            raise ValueError("sudden_hr_change_pause_bpm must be positive")
        if self.sudden_hr_change_link_seconds < 0:
            raise ValueError("sudden_hr_change_link_seconds must be non-negative")


@dataclass(frozen=True)
class ArousalGuardDecision:
    action: str
    timestamp_seconds: float
    reason_codes: Tuple[str, ...] = ()
    volume_multiplier: float = 1.0
    pause_seconds: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in AROUSAL_GUARD_ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(AROUSAL_GUARD_ACTIONS)}")
        if self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        if not 0.0 <= self.volume_multiplier <= 1.0:
            raise ValueError("volume_multiplier must be between 0 and 1")
        if self.pause_seconds < 0:
            raise ValueError("pause_seconds must be non-negative")
        object.__setattr__(self, "reason_codes", _unique(self.reason_codes))

    @property
    def should_lower_volume(self) -> bool:
        return self.action == "lower_volume"

    @property
    def should_pause(self) -> bool:
        return self.action == "pause"

    @property
    def should_stop(self) -> bool:
        return self.action == "stop"

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "timestamp_seconds": self.timestamp_seconds,
            "reason_codes": list(self.reason_codes),
            "volume_multiplier": self.volume_multiplier,
            "pause_seconds": self.pause_seconds,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ArousalGuardDecision":
        return cls(
            action=str(payload["action"]),
            timestamp_seconds=float(payload["timestamp_seconds"]),
            reason_codes=tuple(str(item) for item in payload.get("reason_codes", ())),
            volume_multiplier=float(payload.get("volume_multiplier", 1.0)),
            pause_seconds=float(payload.get("pause_seconds", 0.0)),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


class ArousalGuard:
    """Stateful cue safety guard over motion, alpha, HR, and artifact proxies."""

    def __init__(
        self,
        config: Optional[ArousalGuardConfig] = None,
        *,
        event_log_path: Optional[Path] = None,
    ) -> None:
        self.config = config or ArousalGuardConfig()
        self.config.validate()
        self.event_log_path = event_log_path
        self._consecutive_pause_epochs = 0
        self._consecutive_artifact_epochs = 0
        self._consecutive_critical_artifact_epochs = 0
        self._previous_hr_points: Tuple[HeartRatePoint, ...] = ()

    @property
    def consecutive_pause_epochs(self) -> int:
        return self._consecutive_pause_epochs

    @property
    def consecutive_artifact_epochs(self) -> int:
        return self._consecutive_artifact_epochs

    @property
    def consecutive_critical_artifact_epochs(self) -> int:
        return self._consecutive_critical_artifact_epochs

    def reset(self) -> None:
        self._consecutive_pause_epochs = 0
        self._consecutive_artifact_epochs = 0
        self._consecutive_critical_artifact_epochs = 0
        self._previous_hr_points = ()

    def evaluate(
        self,
        *,
        timestamp_seconds: float,
        eeg: Optional[EEGFeatureRow] = None,
        imu: Optional[IMUFeatureRow] = None,
        ppg: Optional[PPGFeatureRow] = None,
    ) -> ArousalGuardDecision:
        if timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")

        if not self.config.enabled:
            return self._record(
                ArousalGuardDecision(
                    action="allow",
                    timestamp_seconds=timestamp_seconds,
                    metadata={"enabled": False},
                )
            )

        lower_reasons = []
        pause_reasons = []
        artifact_reasons = []
        critical_artifact_reasons = []
        metadata: Dict[str, object] = {}

        self._collect_imu_reasons(imu, lower_reasons, pause_reasons, artifact_reasons, metadata)
        self._collect_eeg_reasons(
            eeg,
            lower_reasons,
            pause_reasons,
            artifact_reasons,
            critical_artifact_reasons,
            metadata,
        )
        self._collect_ppg_reasons(ppg, lower_reasons, pause_reasons, artifact_reasons, metadata)

        artifact_present = bool(artifact_reasons)
        critical_artifact_present = bool(critical_artifact_reasons)
        pause_present = bool(pause_reasons)
        self._consecutive_artifact_epochs = (
            self._consecutive_artifact_epochs + 1 if artifact_present else 0
        )
        self._consecutive_critical_artifact_epochs = (
            self._consecutive_critical_artifact_epochs + 1 if critical_artifact_present else 0
        )
        self._consecutive_pause_epochs = (
            self._consecutive_pause_epochs + 1 if pause_present else 0
        )
        metadata["consecutive_artifact_epochs"] = self._consecutive_artifact_epochs
        metadata["consecutive_critical_artifact_epochs"] = (
            self._consecutive_critical_artifact_epochs
        )
        metadata["consecutive_pause_epochs"] = self._consecutive_pause_epochs

        stop_reasons = []
        if self._consecutive_pause_epochs >= self.config.stop_after_consecutive_pause_epochs:
            pause_reasons.append("repeated_arousal_guard_pause")
        if self._consecutive_artifact_epochs >= self.config.stop_after_consecutive_artifact_epochs:
            if self.config.repeated_artifact_action == "stop":
                stop_reasons.append("repeated_artifact_quality")
            else:
                pause_reasons.append("repeated_artifact_quality")
        if (
            self._consecutive_critical_artifact_epochs
            >= self.config.stop_after_consecutive_critical_artifact_epochs
        ):
            stop_reasons.append("critical_eeg_artifact")

        if stop_reasons:
            return self._record(
                ArousalGuardDecision(
                    action="stop",
                    timestamp_seconds=timestamp_seconds,
                    reason_codes=_unique(
                        stop_reasons
                        + pause_reasons
                        + lower_reasons
                        + artifact_reasons
                        + critical_artifact_reasons
                    ),
                    volume_multiplier=0.0,
                    metadata=metadata,
                )
            )
        if pause_reasons:
            return self._record(
                ArousalGuardDecision(
                    action="pause",
                    timestamp_seconds=timestamp_seconds,
                    reason_codes=_unique(
                        pause_reasons + lower_reasons + artifact_reasons + critical_artifact_reasons
                    ),
                    volume_multiplier=0.0,
                    pause_seconds=self.config.pause_seconds,
                    metadata=metadata,
                )
            )
        if lower_reasons or artifact_reasons:
            return self._record(
                ArousalGuardDecision(
                    action="lower_volume",
                    timestamp_seconds=timestamp_seconds,
                    reason_codes=_unique(lower_reasons + artifact_reasons),
                    volume_multiplier=self.config.lower_volume_multiplier,
                    metadata=metadata,
                )
            )
        allow_metadata = (
            {"ppg_absent_by_design": True} if metadata.get("ppg_absent_by_design") else {}
        )
        return self._record(
            ArousalGuardDecision(
                action="allow",
                timestamp_seconds=timestamp_seconds,
                metadata=allow_metadata,
            )
        )

    def _collect_imu_reasons(
        self,
        imu: Optional[IMUFeatureRow],
        lower_reasons: list,
        pause_reasons: list,
        artifact_reasons: list,
        metadata: Dict[str, object],
    ) -> None:
        if imu is None:
            return
        metadata["stillness_score"] = imu.stillness_score
        metadata["motion_level"] = imu.motion_level
        metadata["arousal_event_count"] = imu.arousal_event_count
        metadata["cue_related_arousal_count"] = imu.cue_related_arousal_count

        for reason in imu.arousal_guard_reason_codes:
            if reason in PAUSE_IMU_REASON_CODES:
                pause_reasons.append(reason)
            else:
                lower_reasons.append(reason)
        if (
            math.isfinite(imu.stillness_score)
            and imu.stillness_score < self.config.min_stillness_score_for_cue
        ):
            pause_reasons.append("motion_not_still")
        if imu.cue_related_arousal_count > 0:
            pause_reasons.append("cue_related_arousal")
        if imu.artifact_flags:
            artifact_reasons.append("imu_artifact_flags")
            metadata["imu_artifact_flags"] = list(imu.artifact_flags)

    def _collect_eeg_reasons(
        self,
        eeg: Optional[EEGFeatureRow],
        lower_reasons: list,
        pause_reasons: list,
        artifact_reasons: list,
        critical_artifact_reasons: list,
        metadata: Dict[str, object],
    ) -> None:
        if eeg is None:
            return
        alpha_power = float(eeg.relative_band_powers.get("alpha", math.nan))
        metadata["relative_alpha_power"] = alpha_power
        metadata["eeg_coverage"] = eeg.eeg_coverage
        bad_channels = _eeg_bad_channels(eeg)
        bad_channel_count = len(bad_channels)
        metadata["bad_eeg_channels"] = list(bad_channels)
        metadata["bad_eeg_channel_count"] = bad_channel_count
        metadata["usable_eeg_channel_count"] = eeg.usable_channel_count
        if eeg.channel_diagnostics:
            metadata["eeg_channel_diagnostics"] = {
                channel: dict(diagnostics)
                for channel, diagnostics in sorted(eeg.channel_diagnostics.items())
            }
        if math.isfinite(alpha_power):
            if alpha_power >= self.config.alpha_pause_relative_power:
                pause_reasons.append("alpha_arousal_proxy")
            elif alpha_power >= self.config.alpha_lower_volume_relative_power:
                lower_reasons.append("alpha_arousal_proxy_mild")
        if eeg.artifact_flags:
            metadata["eeg_artifact_flags"] = list(eeg.artifact_flags)
            if bad_channel_count == 1:
                lower_reasons.append("single_channel_eeg_artifact")
            elif bad_channel_count >= 2:
                pause_reasons.append("multi_channel_eeg_artifact")
                artifact_reasons.append("eeg_artifact_flags")
            else:
                artifact_reasons.append("eeg_artifact_flags")
            if bad_channel_count >= self.config.critical_eeg_bad_channel_count:
                critical_artifact_reasons.append("critical_eeg_artifact")
        if "low_eeg_coverage" in eeg.artifact_flags or "low_eeg_coverage" in eeg.quality_flags:
            lower_reasons.append("low_eeg_coverage")

    def _collect_ppg_reasons(
        self,
        ppg: Optional[PPGFeatureRow],
        lower_reasons: list,
        pause_reasons: list,
        artifact_reasons: list,
        metadata: Dict[str, object],
    ) -> None:
        if not self.config.optics_capable:
            metadata["ppg_absent_by_design"] = True
        if ppg is None:
            self._previous_hr_points = ()
            return
        metadata["sudden_hr_change_count"] = ppg.sudden_hr_change_count
        metadata["max_sudden_hr_change_bpm"] = ppg.max_sudden_hr_change_bpm
        metadata["ppg_coverage"] = ppg.ppg_coverage
        metadata["heart_rate_coverage"] = ppg.heart_rate_coverage

        artifact_flags = ppg.artifact_flags
        if not self.config.optics_capable:
            artifact_flags = tuple(
                flag for flag in artifact_flags if flag not in PPG_ABSENCE_FLAGS
            )
        else:
            metadata["sudden_hr_change_source"] = ppg.sudden_hr_change_source
            metadata["hr_series_used_count"] = len(ppg.hr_series_used)
            metadata["hr_series_skipped_count"] = ppg.hr_series_skipped_count
            steps = self._sudden_hr_steps(ppg, metadata)
            metadata["sudden_hr_change_pause_steps"] = steps
            if steps >= self.config.sudden_hr_change_pause_count:
                pause_reasons.append("sudden_heart_rate_change")
            if "heart_rate_out_of_range" in artifact_flags:
                pause_reasons.append("heart_rate_out_of_range")
        if artifact_flags:
            artifact_reasons.append("ppg_artifact_flags")
            metadata["ppg_artifact_flags"] = list(artifact_flags)
        if "low_ppg_coverage" in artifact_flags or "low_heart_rate_coverage" in artifact_flags:
            lower_reasons.append("low_ppg_hr_coverage")

    def _sudden_hr_steps(self, ppg: PPGFeatureRow, metadata: Dict[str, object]) -> int:
        """Heart-rate changes of sudden_hr_change_pause_bpm or more: the row's own, and
        the largest one from the previous row's series into this one's."""
        bpm = self.config.sudden_hr_change_pause_bpm
        if ppg.sudden_hr_changes:
            steps = sum(1 for change in ppg.sudden_hr_changes if abs(change.delta_bpm) >= bpm)
        elif math.isfinite(ppg.max_sudden_hr_change_bpm):
            # A row with the counts but not the changes: all of them reach bpm when the
            # largest does, none otherwise.
            steps = max(1, ppg.sudden_hr_change_count) if ppg.max_sudden_hr_change_bpm >= bpm else 0
        else:
            steps = ppg.sudden_hr_change_count

        previous, current = self._previous_hr_points, ppg.hr_series_used
        self._previous_hr_points = current
        if self.config.sudden_hr_change_link_seconds > 0 and previous and current:
            change = heart_rate_change_across_epochs(
                previous, current, self.config.sudden_hr_change_link_seconds
            )
            if change is not None:
                metadata["sudden_hr_change_across_epochs_bpm"] = change.delta_bpm
                if abs(change.delta_bpm) >= bpm:
                    steps += 1
        return steps

    def _record(self, decision: ArousalGuardDecision) -> ArousalGuardDecision:
        if self.event_log_path is not None:
            append_arousal_guard_decisions((decision,), self.event_log_path)
        return decision


def append_arousal_guard_decisions(
    decisions: Iterable[ArousalGuardDecision],
    output_path: Path,
) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision.to_dict(), sort_keys=True) + "\n")
    return output_path


def load_arousal_guard_decisions(input_path: Path) -> Tuple[ArousalGuardDecision, ...]:
    input_path = input_path.expanduser()
    if not input_path.exists():
        return ()
    return tuple(
        ArousalGuardDecision.from_dict(json.loads(line))
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _unique(reason_codes: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(code for code in reason_codes if code))


def _eeg_bad_channels(eeg: EEGFeatureRow) -> Tuple[str, ...]:
    if eeg.bad_channels:
        return tuple(sorted(eeg.bad_channels))

    channel_prefixes = (
        "eeg_clipping_",
        "eeg_empty_",
        "eeg_flatline_",
        "eeg_nonfinite_",
    )
    bad_channels = set()
    for flag in eeg.artifact_flags:
        for prefix in channel_prefixes:
            if flag.startswith(prefix):
                bad_channels.add(flag[len(prefix) :])
                break
    return tuple(sorted(channel for channel in bad_channels if channel))
