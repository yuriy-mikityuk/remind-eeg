"""IMU motion and arousal feature extraction for SleepEpoch windows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from muse_tmr.features.epochs import SleepEpoch


@dataclass(frozen=True)
class IMUFeatureConfig:
    sample_rate_hz: float = 52.0
    min_imu_coverage: float = 0.5
    movement_accel_delta_g_threshold: float = 0.05
    movement_gyro_dps_threshold: float = 15.0
    arousal_accel_delta_g_threshold: float = 0.20
    arousal_gyro_dps_threshold: float = 60.0
    arousal_motion_level_threshold: float = 2.0
    min_stillness_score_for_cue: float = 0.90
    max_event_gap_seconds: float = 0.50
    cue_pre_window_seconds: float = 2.0
    cue_post_window_seconds: float = 10.0

    def validate(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if not 0 <= self.min_imu_coverage <= 1:
            raise ValueError("min_imu_coverage must be between 0 and 1")
        if self.movement_accel_delta_g_threshold <= 0:
            raise ValueError("movement_accel_delta_g_threshold must be positive")
        if self.movement_gyro_dps_threshold <= 0:
            raise ValueError("movement_gyro_dps_threshold must be positive")
        if self.arousal_accel_delta_g_threshold <= 0:
            raise ValueError("arousal_accel_delta_g_threshold must be positive")
        if self.arousal_gyro_dps_threshold <= 0:
            raise ValueError("arousal_gyro_dps_threshold must be positive")
        if self.arousal_motion_level_threshold <= 0:
            raise ValueError("arousal_motion_level_threshold must be positive")
        if not 0 <= self.min_stillness_score_for_cue <= 1:
            raise ValueError("min_stillness_score_for_cue must be between 0 and 1")
        if self.max_event_gap_seconds < 0:
            raise ValueError("max_event_gap_seconds must be non-negative")
        if self.cue_pre_window_seconds < 0 or self.cue_post_window_seconds < 0:
            raise ValueError("cue windows must be non-negative")


@dataclass(frozen=True)
class MovementEvent:
    start_time: float
    end_time: float
    duration_seconds: float
    sample_count: int
    peak_motion_level: float
    peak_accel_delta_g: float
    peak_gyro_dps: float
    is_arousal_proxy: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "sample_count": self.sample_count,
            "peak_motion_level": self.peak_motion_level,
            "peak_accel_delta_g": self.peak_accel_delta_g,
            "peak_gyro_dps": self.peak_gyro_dps,
            "is_arousal_proxy": self.is_arousal_proxy,
        }


@dataclass(frozen=True)
class CueMovementLog:
    cue_time: float
    window_start: float
    window_end: float
    movement_event_count: int
    arousal_event_count: int
    peak_motion_level: float
    peak_accel_delta_g: float
    peak_gyro_dps: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "cue_time": self.cue_time,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "movement_event_count": self.movement_event_count,
            "arousal_event_count": self.arousal_event_count,
            "peak_motion_level": self.peak_motion_level,
            "peak_accel_delta_g": self.peak_accel_delta_g,
            "peak_gyro_dps": self.peak_gyro_dps,
        }


@dataclass(frozen=True)
class IMUFeatureRow:
    epoch_index: int
    start_time: float
    end_time: float
    imu_coverage: float
    sample_count: int
    accelerometer_sample_count: int
    gyroscope_sample_count: int
    motion_level: float
    stillness_score: float
    accel_rms_delta_g: float
    accel_peak_delta_g: float
    gyro_rms_dps: float
    gyro_peak_dps: float
    movement_event_count: int
    arousal_event_count: int
    arousal_proxy: float
    movement_events: Tuple[MovementEvent, ...]
    cue_movement_logs: Tuple[CueMovementLog, ...]
    artifact_flags: Tuple[str, ...]
    quality_flags: Tuple[str, ...]
    arousal_guard_reason_codes: Tuple[str, ...]

    @property
    def is_noisy(self) -> bool:
        return bool(self.artifact_flags)

    @property
    def arousal_guard_blocked(self) -> bool:
        return bool(self.arousal_guard_reason_codes)

    @property
    def cue_related_movement_count(self) -> int:
        return sum(log.movement_event_count for log in self.cue_movement_logs)

    @property
    def cue_related_arousal_count(self) -> int:
        return sum(log.arousal_event_count for log in self.cue_movement_logs)

    def to_dict(self) -> Dict[str, object]:
        return {
            "epoch_index": self.epoch_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "imu_coverage": self.imu_coverage,
            "sample_count": self.sample_count,
            "accelerometer_sample_count": self.accelerometer_sample_count,
            "gyroscope_sample_count": self.gyroscope_sample_count,
            "motion_level": self.motion_level,
            "stillness_score": self.stillness_score,
            "accel_rms_delta_g": self.accel_rms_delta_g,
            "accel_peak_delta_g": self.accel_peak_delta_g,
            "gyro_rms_dps": self.gyro_rms_dps,
            "gyro_peak_dps": self.gyro_peak_dps,
            "movement_event_count": self.movement_event_count,
            "arousal_event_count": self.arousal_event_count,
            "arousal_proxy": self.arousal_proxy,
            "arousal_guard_blocked": self.arousal_guard_blocked,
            "cue_related_movement_count": self.cue_related_movement_count,
            "cue_related_arousal_count": self.cue_related_arousal_count,
            "is_noisy": self.is_noisy,
            "artifact_flags": ";".join(self.artifact_flags),
            "quality_flags": ";".join(self.quality_flags),
            "arousal_guard_reason_codes": ";".join(self.arousal_guard_reason_codes),
            "movement_events_json": json.dumps(
                [event.to_dict() for event in self.movement_events],
                sort_keys=True,
            ),
            "cue_movement_logs_json": json.dumps(
                [log.to_dict() for log in self.cue_movement_logs],
                sort_keys=True,
            ),
        }


@dataclass(frozen=True)
class _MotionSeries:
    timestamps: np.ndarray
    accel_delta_g: np.ndarray
    gyro_dps: np.ndarray
    motion_level: np.ndarray
    movement_mask: np.ndarray
    finite_accel_count: int
    finite_gyro_count: int


def extract_imu_features(
    epoch: SleepEpoch,
    config: Optional[IMUFeatureConfig] = None,
    cue_timestamps: Optional[Sequence[float]] = None,
) -> IMUFeatureRow:
    config = config or IMUFeatureConfig()
    config.validate()

    samples = _collect_motion_series(epoch, config)
    imu_coverage = float(epoch.coverage.get("imu", 0.0))
    quality_flags = tuple(flag for flag in epoch.quality_flags if "imu" in flag)
    artifact_flags = _artifact_flags(samples, imu_coverage, quality_flags, config)
    movement_events = _movement_events(samples, config)
    arousal_event_count = sum(1 for event in movement_events if event.is_arousal_proxy)
    cue_movement_logs = _cue_movement_logs(
        movement_events=movement_events,
        cue_timestamps=tuple(cue_timestamps or ()),
        config=config,
    )
    arousal_guard_reason_codes = _arousal_guard_reason_codes(
        samples=samples,
        movement_events=movement_events,
        imu_coverage=imu_coverage,
        config=config,
    )

    return IMUFeatureRow(
        epoch_index=epoch.index,
        start_time=epoch.start_time,
        end_time=epoch.end_time,
        imu_coverage=imu_coverage,
        sample_count=int(samples.timestamps.size),
        accelerometer_sample_count=samples.finite_accel_count,
        gyroscope_sample_count=samples.finite_gyro_count,
        motion_level=_safe_rms(samples.motion_level),
        stillness_score=_stillness_score(samples),
        accel_rms_delta_g=_safe_rms(samples.accel_delta_g),
        accel_peak_delta_g=_safe_max(samples.accel_delta_g),
        gyro_rms_dps=_safe_rms(samples.gyro_dps),
        gyro_peak_dps=_safe_max(samples.gyro_dps),
        movement_event_count=len(movement_events),
        arousal_event_count=arousal_event_count,
        arousal_proxy=_safe_max(samples.motion_level),
        movement_events=movement_events,
        cue_movement_logs=cue_movement_logs,
        artifact_flags=artifact_flags,
        quality_flags=quality_flags,
        arousal_guard_reason_codes=arousal_guard_reason_codes,
    )


def extract_imu_feature_rows(
    epochs: Iterable[SleepEpoch],
    config: Optional[IMUFeatureConfig] = None,
    cue_timestamps_by_epoch: Optional[Mapping[int, Sequence[float]]] = None,
) -> Tuple[IMUFeatureRow, ...]:
    cue_timestamps_by_epoch = cue_timestamps_by_epoch or {}
    return tuple(
        extract_imu_features(
            epoch,
            config=config,
            cue_timestamps=cue_timestamps_by_epoch.get(epoch.index),
        )
        for epoch in epochs
    )


def export_imu_feature_rows(rows: Sequence[IMUFeatureRow], output_path: Path) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row.to_dict() for row in rows])

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(output_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(output_path, index=False)
    else:
        raise ValueError("IMU feature export path must end with .csv, .parquet, or .pq")
    return output_path


def _collect_motion_series(epoch: SleepEpoch, config: IMUFeatureConfig) -> _MotionSeries:
    timestamps = []
    accel_magnitude = []
    gyro_magnitude = []

    for frame in epoch.frames:
        if frame.imu is None:
            continue
        accel_rows = frame.imu.accelerometer_g or ()
        gyro_rows = frame.imu.gyroscope_dps or ()
        row_count = max(len(accel_rows), len(gyro_rows))
        offsets = _row_offsets(row_count, frame.imu.gaps)
        for index in range(row_count):
            timestamps.append(frame.imu.timestamp + offsets[index] / config.sample_rate_hz)
            accel_magnitude.append(
                _axis_magnitude(accel_rows[index]) if index < len(accel_rows) else math.nan
            )
            gyro_magnitude.append(
                _axis_magnitude(gyro_rows[index]) if index < len(gyro_rows) else math.nan
            )

    if not timestamps:
        empty = np.asarray((), dtype=float)
        return _MotionSeries(empty, empty, empty, empty, empty.astype(bool), 0, 0)

    timestamps_array = np.asarray(timestamps, dtype=float)
    accel_array = np.asarray(accel_magnitude, dtype=float)
    gyro_array = np.asarray(gyro_magnitude, dtype=float)
    accel_delta_g = _accel_delta(accel_array)
    gyro_dps = np.abs(gyro_array)
    motion_level = np.fmax(
        _normalized_or_zero(accel_delta_g, config.movement_accel_delta_g_threshold),
        _normalized_or_zero(gyro_dps, config.movement_gyro_dps_threshold),
    )
    movement_mask = (
        (np.isfinite(accel_delta_g) & (accel_delta_g >= config.movement_accel_delta_g_threshold))
        | (np.isfinite(gyro_dps) & (gyro_dps >= config.movement_gyro_dps_threshold))
    )

    return _MotionSeries(
        timestamps=timestamps_array,
        accel_delta_g=accel_delta_g,
        gyro_dps=gyro_dps,
        motion_level=motion_level,
        movement_mask=movement_mask,
        finite_accel_count=int(np.sum(np.isfinite(accel_delta_g))),
        finite_gyro_count=int(np.sum(np.isfinite(gyro_dps))),
    )


def _row_offsets(row_count: int, gaps: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Samples from the frame's first row to each row. Samples lost inside the frame
    (a gap after row 0) move the rows after them; a gap at row 0 lies before the frame."""
    offsets = np.arange(row_count)
    for index, missing in gaps:
        if 0 < index < row_count:
            offsets[index:] += missing
    return offsets


def _axis_magnitude(row: Mapping[str, float]) -> float:
    x = float(row.get("x", 0.0))
    y = float(row.get("y", 0.0))
    z = float(row.get("z", 0.0))
    return float(math.sqrt(x * x + y * y + z * z))


def _accel_delta(accel_magnitude: np.ndarray) -> np.ndarray:
    finite = accel_magnitude[np.isfinite(accel_magnitude)]
    if finite.size == 0:
        return np.full(accel_magnitude.shape, math.nan)
    baseline = float(np.median(finite))
    return np.abs(accel_magnitude - baseline)


def _normalized_or_zero(values: np.ndarray, threshold: float) -> np.ndarray:
    normalized = np.divide(values, threshold)
    return np.where(np.isfinite(normalized), normalized, 0.0)


def _movement_events(
    samples: _MotionSeries,
    config: IMUFeatureConfig,
) -> Tuple[MovementEvent, ...]:
    if samples.timestamps.size == 0 or not np.any(samples.movement_mask):
        return ()

    event_indices = np.flatnonzero(samples.movement_mask)
    groups = []
    current = [int(event_indices[0])]
    for index in event_indices[1:]:
        previous_index = current[-1]
        time_gap = samples.timestamps[int(index)] - samples.timestamps[previous_index]
        if time_gap <= config.max_event_gap_seconds:
            current.append(int(index))
        else:
            groups.append(tuple(current))
            current = [int(index)]
    groups.append(tuple(current))

    events = []
    for group in groups:
        start_time = float(samples.timestamps[group[0]])
        end_time = float(samples.timestamps[group[-1]] + 1 / config.sample_rate_hz)
        peak_motion_level = _safe_max(samples.motion_level[list(group)])
        peak_accel_delta_g = _safe_max(samples.accel_delta_g[list(group)])
        peak_gyro_dps = _safe_max(samples.gyro_dps[list(group)])
        is_arousal_proxy = (
            peak_accel_delta_g >= config.arousal_accel_delta_g_threshold
            or peak_gyro_dps >= config.arousal_gyro_dps_threshold
            or peak_motion_level >= config.arousal_motion_level_threshold
        )
        events.append(
            MovementEvent(
                start_time=start_time,
                end_time=end_time,
                duration_seconds=end_time - start_time,
                sample_count=len(group),
                peak_motion_level=peak_motion_level,
                peak_accel_delta_g=peak_accel_delta_g,
                peak_gyro_dps=peak_gyro_dps,
                is_arousal_proxy=is_arousal_proxy,
            )
        )
    return tuple(events)


def _cue_movement_logs(
    movement_events: Tuple[MovementEvent, ...],
    cue_timestamps: Sequence[float],
    config: IMUFeatureConfig,
) -> Tuple[CueMovementLog, ...]:
    logs = []
    for cue_time in cue_timestamps:
        window_start = float(cue_time) - config.cue_pre_window_seconds
        window_end = float(cue_time) + config.cue_post_window_seconds
        matching_events = tuple(
            event
            for event in movement_events
            if event.end_time >= window_start and event.start_time <= window_end
        )
        logs.append(
            CueMovementLog(
                cue_time=float(cue_time),
                window_start=window_start,
                window_end=window_end,
                movement_event_count=len(matching_events),
                arousal_event_count=sum(1 for event in matching_events if event.is_arousal_proxy),
                peak_motion_level=_safe_max(event.peak_motion_level for event in matching_events),
                peak_accel_delta_g=_safe_max(event.peak_accel_delta_g for event in matching_events),
                peak_gyro_dps=_safe_max(event.peak_gyro_dps for event in matching_events),
            )
        )
    return tuple(logs)


def _artifact_flags(
    samples: _MotionSeries,
    imu_coverage: float,
    quality_flags: Tuple[str, ...],
    config: IMUFeatureConfig,
) -> Tuple[str, ...]:
    flags = set(quality_flags)
    if samples.timestamps.size == 0:
        flags.add("imu_missing")
        return tuple(sorted(flags))
    if imu_coverage < config.min_imu_coverage:
        flags.add("low_imu_coverage")
    if samples.finite_accel_count == 0:
        flags.add("imu_accelerometer_missing")
    if samples.finite_gyro_count == 0:
        flags.add("imu_gyroscope_missing")
    if np.any(~np.isfinite(samples.accel_delta_g)) and samples.finite_accel_count > 0:
        flags.add("imu_accelerometer_nonfinite")
    if np.any(~np.isfinite(samples.gyro_dps)) and samples.finite_gyro_count > 0:
        flags.add("imu_gyroscope_nonfinite")
    return tuple(sorted(flags))


def _arousal_guard_reason_codes(
    samples: _MotionSeries,
    movement_events: Tuple[MovementEvent, ...],
    imu_coverage: float,
    config: IMUFeatureConfig,
) -> Tuple[str, ...]:
    reasons = []
    if samples.timestamps.size == 0:
        reasons.append("imu_missing")
        return tuple(reasons)
    if imu_coverage < config.min_imu_coverage:
        reasons.append("low_imu_coverage")
    if _stillness_score(samples) < config.min_stillness_score_for_cue:
        reasons.append("motion_not_still")
    if any(event.is_arousal_proxy for event in movement_events):
        reasons.append("motion_arousal_proxy")
    return tuple(reasons)


def _stillness_score(samples: _MotionSeries) -> float:
    if samples.movement_mask.size == 0:
        return math.nan
    return float(np.mean(~samples.movement_mask))


def _safe_rms(values: Iterable[float]) -> float:
    values_array = np.asarray(tuple(values), dtype=float)
    finite = values_array[np.isfinite(values_array)]
    if finite.size == 0:
        return math.nan
    return float(math.sqrt(np.mean(finite * finite)))


def _safe_max(values: Iterable[float]) -> float:
    values_array = np.asarray(tuple(values), dtype=float)
    finite = values_array[np.isfinite(values_array)]
    if finite.size == 0:
        return math.nan
    return float(np.max(finite))
