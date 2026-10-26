"""PPG, HR, and HRV feature extraction for SleepEpoch windows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, find_peaks, sosfiltfilt

from muse_tmr.features.epochs import SleepEpoch

# What fed the sudden heart-rate change rule of a row: the epoch's own heart-rate series
# from the optics, the heart-rate samples when the epoch has no optics, or neither.
SUDDEN_HR_CHANGE_SOURCES = ("optics", "heart_rate", "missing")

# A frame that arrives this much later than the optics sample clock says it should lost
# samples the decoder did not report (a loss long enough to wrap the subpacket counter);
# the clock moves on to it and the samples in between are missing.
_OPTICS_CLOCK_SLACK_SECONDS = 1.0
# The autocorrelation estimate takes the shortest lag whose peak reaches this share of the
# highest peak in the heart-rate range, so one beat wins over two or three (sub-harmonics).
_AUTOCORRELATION_PEAK_SHARE = 0.9
# A window that holds two heart rates (a step or a short surge inside it) peaks at neither
# beat period. Its longest run of at least this many beat intervals, each within this
# share of the window's median interval, then stands in for it (see _two_rate_run()).
_RUN_MIN_INTERVALS = 5
_RUN_INTERVAL_SHARE = 0.15
# Series times are Unix seconds; their differences carry float rounding.
_TIME_TOLERANCE_SECONDS = 1e-6


@dataclass(frozen=True)
class PPGFeatureConfig:
    ppg_sample_rate_hz: float = 64.0
    min_ppg_coverage: float = 0.5
    min_heart_rate_coverage: float = 0.5
    min_ppg_seconds_for_peak_hr: float = 5.0
    ppg_low_hz: float = 0.5
    ppg_high_hz: float = 4.0
    peak_prominence: float = 0.3
    sudden_hr_change_bpm: float = 10.0
    min_valid_hr_bpm: float = 30.0
    max_valid_hr_bpm: float = 240.0
    flat_ppg_std_threshold: float = 1e-9
    # The heart-rate series the sudden-change rule reads (#247; see heart_rate_series()):
    # windows of hr_series_window_seconds stepped by hr_series_step_seconds over one optics
    # channel band-passed to ppg_low_hz..ppg_high_hz. A value is used when its two
    # estimates agree within hr_series_max_disagreement_bpm and its quality reaches
    # hr_series_min_quality, or else when a run of regular beats of at least
    # hr_series_min_run_seconds inside the window does; used values
    # hr_series_compare_seconds to hr_series_max_compare_seconds apart are compared, and
    # changes from sudden_hr_change_bpm less hr_series_margin_bpm (the error of a value)
    # are logged.
    hr_series_window_seconds: float = 10.0
    hr_series_step_seconds: float = 1.0
    hr_series_compare_seconds: float = 10.0
    hr_series_max_compare_seconds: float = 20.0
    hr_series_min_quality: float = 0.5
    hr_series_max_disagreement_bpm: float = 5.0
    hr_series_max_missing_fraction: float = 0.2
    hr_series_peak_prominence: float = 0.5
    hr_series_min_run_seconds: float = 4.0
    hr_series_margin_bpm: float = 1.0

    def validate(self) -> None:
        if self.ppg_sample_rate_hz <= 0:
            raise ValueError("ppg_sample_rate_hz must be positive")
        if not 0 <= self.min_ppg_coverage <= 1:
            raise ValueError("min_ppg_coverage must be between 0 and 1")
        if not 0 <= self.min_heart_rate_coverage <= 1:
            raise ValueError("min_heart_rate_coverage must be between 0 and 1")
        if self.min_ppg_seconds_for_peak_hr <= 0:
            raise ValueError("min_ppg_seconds_for_peak_hr must be positive")
        if self.ppg_low_hz <= 0 or self.ppg_high_hz <= self.ppg_low_hz:
            raise ValueError("invalid PPG bandpass range")
        if self.ppg_high_hz >= self.ppg_sample_rate_hz / 2:
            raise ValueError("ppg_high_hz must be below Nyquist")
        if self.peak_prominence <= 0:
            raise ValueError("peak_prominence must be positive")
        if self.sudden_hr_change_bpm <= 0:
            raise ValueError("sudden_hr_change_bpm must be positive")
        if self.min_valid_hr_bpm <= 0 or self.max_valid_hr_bpm <= self.min_valid_hr_bpm:
            raise ValueError("invalid heart-rate bounds")
        if self.flat_ppg_std_threshold < 0:
            raise ValueError("flat_ppg_std_threshold must be non-negative")
        if self.hr_series_window_seconds < 2 * 60.0 / self.min_valid_hr_bpm:
            raise ValueError("hr_series_window_seconds must hold two beats at min_valid_hr_bpm")
        if self.hr_series_step_seconds <= 0:
            raise ValueError("hr_series_step_seconds must be positive")
        if self.hr_series_compare_seconds < self.hr_series_step_seconds:
            raise ValueError("hr_series_compare_seconds must be at least one step")
        if self.hr_series_max_compare_seconds < self.hr_series_compare_seconds:
            raise ValueError("hr_series_max_compare_seconds must not be below hr_series_compare_seconds")
        if not 0 <= self.hr_series_min_quality <= 1:
            raise ValueError("hr_series_min_quality must be between 0 and 1")
        if self.hr_series_max_disagreement_bpm <= 0:
            raise ValueError("hr_series_max_disagreement_bpm must be positive")
        if not 0 <= self.hr_series_max_missing_fraction < 1:
            raise ValueError("hr_series_max_missing_fraction must be in [0, 1)")
        if self.hr_series_peak_prominence <= 0:
            raise ValueError("hr_series_peak_prominence must be positive")
        if not 0 < self.hr_series_min_run_seconds <= self.hr_series_window_seconds:
            raise ValueError("hr_series_min_run_seconds must be positive and fit in a window")
        if not 0 <= self.hr_series_margin_bpm < self.sudden_hr_change_bpm:
            raise ValueError("hr_series_margin_bpm must be in [0, sudden_hr_change_bpm)")


@dataclass(frozen=True)
class HeartRatePoint:
    """One value of the heart-rate series: a window of band-passed optics.

    ``time`` is the window's centre in Unix seconds. ``bpm`` is the autocorrelation
    estimate and ``peak_bpm`` the peak-interval estimate it is checked against, NaN when
    the window gave none. ``quality`` (0 to 1) is the autocorrelation peak when the two
    agree, else 0; ``used`` says whether the sudden-change rule reads the value.
    ``run_seconds`` is 0, or the length of the run of regular beats inside the window
    that gave all three when the window's own estimates did not reach the quality.
    """

    time: float
    bpm: float
    peak_bpm: float
    quality: float
    used: bool
    run_seconds: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "time": self.time,
            "bpm": _finite_or_none(self.bpm),
            "peak_bpm": _finite_or_none(self.peak_bpm),
            "quality": self.quality,
            "used": self.used,
            "run_seconds": self.run_seconds,
        }


@dataclass(frozen=True)
class HeartRateSeries:
    """The heart-rate series of one epoch and the optics channel it came from."""

    channel: str
    points: Tuple[HeartRatePoint, ...]

    @property
    def used_points(self) -> Tuple[HeartRatePoint, ...]:
        return tuple(point for point in self.points if point.used)


@dataclass(frozen=True)
class SuddenHeartRateChange:
    start_time: float
    end_time: float
    before_bpm: float
    after_bpm: float
    delta_bpm: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "before_bpm": self.before_bpm,
            "after_bpm": self.after_bpm,
            "delta_bpm": self.delta_bpm,
        }


@dataclass(frozen=True)
class PPGFeatureRow:
    epoch_index: int
    start_time: float
    end_time: float
    ppg_coverage: float
    heart_rate_coverage: float
    ppg_sample_count: int
    heart_rate_sample_count: int
    ppg_channel_count: int
    ppg_channel_sample_counts: Mapping[str, int]
    primary_ppg_channel: str
    ppg_estimated_hr_bpm: float
    ppg_confidence: float
    ppg_peak_count: int
    ppg_signal_quality: str
    mean_hr_bpm: float
    median_hr_bpm: float
    min_hr_bpm: float
    max_hr_bpm: float
    hr_trend_bpm_per_min: float
    hr_source: str
    mean_rr_ms: float
    sdnn_ms: float
    rmssd_ms: float
    pnn50_percent: float
    hrv_source: str
    sudden_hr_change_count: int
    max_sudden_hr_change_bpm: float
    sudden_hr_changes: Tuple[SuddenHeartRateChange, ...]
    artifact_flags: Tuple[str, ...]
    quality_flags: Tuple[str, ...]
    # What fed the sudden-change rule (SUDDEN_HR_CHANGE_SOURCES). A row built without it
    # gets "heart_rate", the only source before #247.
    sudden_hr_change_source: str = "heart_rate"
    hr_series_channel: str = ""
    hr_series: Tuple[HeartRatePoint, ...] = ()

    @property
    def is_noisy(self) -> bool:
        return bool(self.artifact_flags)

    @property
    def hr_series_used(self) -> Tuple[HeartRatePoint, ...]:
        return tuple(point for point in self.hr_series if point.used)

    @property
    def hr_series_skipped_count(self) -> int:
        return sum(1 for point in self.hr_series if not point.used)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "epoch_index": self.epoch_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "ppg_coverage": self.ppg_coverage,
            "heart_rate_coverage": self.heart_rate_coverage,
            "ppg_sample_count": self.ppg_sample_count,
            "heart_rate_sample_count": self.heart_rate_sample_count,
            "ppg_channel_count": self.ppg_channel_count,
            "primary_ppg_channel": self.primary_ppg_channel,
            "ppg_estimated_hr_bpm": self.ppg_estimated_hr_bpm,
            "ppg_confidence": self.ppg_confidence,
            "ppg_peak_count": self.ppg_peak_count,
            "ppg_signal_quality": self.ppg_signal_quality,
            "mean_hr_bpm": self.mean_hr_bpm,
            "median_hr_bpm": self.median_hr_bpm,
            "min_hr_bpm": self.min_hr_bpm,
            "max_hr_bpm": self.max_hr_bpm,
            "hr_trend_bpm_per_min": self.hr_trend_bpm_per_min,
            "hr_source": self.hr_source,
            "mean_rr_ms": self.mean_rr_ms,
            "sdnn_ms": self.sdnn_ms,
            "rmssd_ms": self.rmssd_ms,
            "pnn50_percent": self.pnn50_percent,
            "hrv_source": self.hrv_source,
            "sudden_hr_change_count": self.sudden_hr_change_count,
            "max_sudden_hr_change_bpm": self.max_sudden_hr_change_bpm,
            "sudden_hr_change_source": self.sudden_hr_change_source,
            "hr_series_channel": self.hr_series_channel,
            "hr_series_used_count": len(self.hr_series_used),
            "hr_series_skipped_count": self.hr_series_skipped_count,
            "hr_series_median_quality": _safe_median(point.quality for point in self.hr_series),
            "is_noisy": self.is_noisy,
            "artifact_flags": ";".join(self.artifact_flags),
            "quality_flags": ";".join(self.quality_flags),
            "sudden_hr_changes_json": json.dumps(
                [change.to_dict() for change in self.sudden_hr_changes],
                sort_keys=True,
            ),
            "hr_series_json": json.dumps(
                [point.to_dict() for point in self.hr_series],
                sort_keys=True,
            ),
        }
        for channel, count in sorted(self.ppg_channel_sample_counts.items()):
            payload[f"ppg_channel_samples_{channel}"] = count
        return payload


@dataclass(frozen=True)
class _PPGPeakResult:
    channel: str
    estimated_hr_bpm: float
    confidence: float
    peak_times: Tuple[float, ...]
    signal_quality: str


def extract_ppg_features(
    epoch: SleepEpoch,
    config: Optional[PPGFeatureConfig] = None,
) -> PPGFeatureRow:
    config = config or PPGFeatureConfig()
    config.validate()

    ppg_channels = _collect_ppg_channels(epoch)
    hr_times, hr_values = _collect_heart_rate_series(epoch)
    ppg_peak_result = _estimate_ppg_heart_rate(ppg_channels, config)
    ppg_coverage = float(epoch.coverage.get("ppg", 0.0))
    heart_rate_coverage = float(epoch.coverage.get("heart_rate", 0.0))
    quality_flags = tuple(
        flag for flag in epoch.quality_flags
        if "ppg" in flag or "heart_rate" in flag
    )
    artifact_flags = _artifact_flags(
        ppg_channels=ppg_channels,
        hr_values=hr_values,
        ppg_coverage=ppg_coverage,
        heart_rate_coverage=heart_rate_coverage,
        quality_flags=quality_flags,
        config=config,
    )
    hr_summary = _heart_rate_summary(hr_times, hr_values, ppg_peak_result)
    hrv = _hrv_metrics(ppg_peak_result.peak_times, hr_values)
    # The rule reads the epoch's own heart-rate series from the optics, never the
    # decoder's heart_rate stream, which is re-estimated from the last 5 s on every optics
    # notification and jumps when a beat enters or leaves that window (#247). Only an
    # epoch without optics falls back to its heart-rate samples, compared as before.
    series = HeartRateSeries("", ())
    if ppg_channels:
        series = heart_rate_series(epoch, config)
        sudden_changes = _series_hr_changes(series.points, config)
        sudden_source = "optics"
    elif hr_values.size:
        sudden_changes = _sudden_hr_changes(hr_times, hr_values, config)
        sudden_source = "heart_rate"
    else:
        sudden_changes = ()
        sudden_source = "missing"

    return PPGFeatureRow(
        epoch_index=epoch.index,
        start_time=epoch.start_time,
        end_time=epoch.end_time,
        ppg_coverage=ppg_coverage,
        heart_rate_coverage=heart_rate_coverage,
        ppg_sample_count=max((values.size for values in ppg_channels.values()), default=0),
        heart_rate_sample_count=int(hr_values.size),
        ppg_channel_count=len(ppg_channels),
        ppg_channel_sample_counts={
            channel: int(values.size) for channel, values in ppg_channels.items()
        },
        primary_ppg_channel=ppg_peak_result.channel,
        ppg_estimated_hr_bpm=ppg_peak_result.estimated_hr_bpm,
        ppg_confidence=ppg_peak_result.confidence,
        ppg_peak_count=len(ppg_peak_result.peak_times),
        ppg_signal_quality=ppg_peak_result.signal_quality,
        mean_hr_bpm=hr_summary["mean_hr_bpm"],
        median_hr_bpm=hr_summary["median_hr_bpm"],
        min_hr_bpm=hr_summary["min_hr_bpm"],
        max_hr_bpm=hr_summary["max_hr_bpm"],
        hr_trend_bpm_per_min=hr_summary["hr_trend_bpm_per_min"],
        hr_source=hr_summary["hr_source"],
        mean_rr_ms=hrv["mean_rr_ms"],
        sdnn_ms=hrv["sdnn_ms"],
        rmssd_ms=hrv["rmssd_ms"],
        pnn50_percent=hrv["pnn50_percent"],
        hrv_source=hrv["hrv_source"],
        sudden_hr_change_count=len(sudden_changes),
        max_sudden_hr_change_bpm=_safe_max(
            abs(change.delta_bpm) for change in sudden_changes
        ),
        sudden_hr_changes=sudden_changes,
        artifact_flags=artifact_flags,
        quality_flags=quality_flags,
        sudden_hr_change_source=sudden_source,
        hr_series_channel=series.channel,
        hr_series=series.points,
    )


def heart_rate_series(
    epoch: SleepEpoch,
    config: Optional[PPGFeatureConfig] = None,
) -> HeartRateSeries:
    """The heart-rate series of an epoch, from the optics it holds (#247).

    Every optics channel goes on one sample clock at ``ppg_sample_rate_hz`` (lost samples
    the decoder reports, and longer losses it cannot report, stay missing and are
    bridged linearly) and is band-passed to ``ppg_low_hz``..``ppg_high_hz``. The series
    comes from the channel whose band-passed epoch is the most periodic: the highest
    autocorrelation peak between ``min_valid_hr_bpm`` and ``max_valid_hr_bpm``, the first
    channel by name on a tie. Each window of ``hr_series_window_seconds``, stepped by
    ``hr_series_step_seconds``, gives two estimates: the beat period from the
    autocorrelation (the value) and the median interval between pulse maxima (the check).
    Its quality is the autocorrelation peak when the two agree within
    ``hr_series_max_disagreement_bpm``, else 0, and 0 when more than
    ``hr_series_max_missing_fraction`` of the window is missing. A window that holds two
    rates, where a step or a short surge falls inside it, peaks at neither beat period;
    when its quality stays below ``hr_series_min_quality``, its longest run of regular
    beats of at least ``hr_series_min_run_seconds`` gives the two estimates and the
    quality instead (``run_seconds``). Values below ``hr_series_min_quality`` stay in the
    series with ``used`` false.
    """
    config = config or PPGFeatureConfig()
    config.validate()
    fs = config.ppg_sample_rate_hz
    window = int(round(config.hr_series_window_seconds * fs))
    start_time, channels, lost = _optics_samples(epoch, fs)
    if lost.size < window:
        return HeartRateSeries("", ())

    filtered: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, values in sorted(channels.items()):
        missing = lost | ~np.isfinite(values)
        signal = _band_passed(values, missing, config)
        if signal is not None:
            filtered[name] = (signal, missing)
    if not filtered:
        return HeartRateSeries("", ())

    channel = _most_periodic_channel(filtered, config)
    signal, missing = filtered[channel]
    return HeartRateSeries(channel, _series_points(signal, missing, start_time, config))


def heart_rate_change_across_epochs(
    previous: Sequence[HeartRatePoint],
    current: Sequence[HeartRatePoint],
    max_seconds: float,
) -> Optional[SuddenHeartRateChange]:
    """The largest change between a used value of one epoch's series and a used value
    of the next epoch's at most ``max_seconds`` later; None when no pair qualifies.

    An epoch's windows stay inside it, so a step in its last or first 10 s shows only
    when values on both sides of the boundary are compared.
    """
    largest: Optional[SuddenHeartRateChange] = None
    for after in current:
        if not after.used:
            continue
        for before in previous:
            if not before.used or not 0 < after.time - before.time <= max_seconds:
                continue
            delta = float(after.bpm - before.bpm)
            if largest is None or abs(delta) > abs(largest.delta_bpm):
                largest = SuddenHeartRateChange(
                    start_time=before.time,
                    end_time=after.time,
                    before_bpm=before.bpm,
                    after_bpm=after.bpm,
                    delta_bpm=delta,
                )
    return largest


def extract_ppg_feature_rows(
    epochs: Iterable[SleepEpoch],
    config: Optional[PPGFeatureConfig] = None,
) -> Tuple[PPGFeatureRow, ...]:
    return tuple(extract_ppg_features(epoch, config=config) for epoch in epochs)


def export_ppg_feature_rows(rows: Sequence[PPGFeatureRow], output_path: Path) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row.to_dict() for row in rows])

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(output_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(output_path, index=False)
    else:
        raise ValueError("PPG feature export path must end with .csv, .parquet, or .pq")
    return output_path


def _collect_ppg_channels(epoch: SleepEpoch) -> Mapping[str, np.ndarray]:
    channel_values: Dict[str, list] = {}
    for frame in epoch.frames:
        if frame.ppg is None:
            continue
        for channel, values in frame.ppg.channels.items():
            channel_values.setdefault(channel, []).extend(values)
    return {
        channel: np.asarray(values, dtype=float)
        for channel, values in channel_values.items()
    }


def _collect_heart_rate_series(epoch: SleepEpoch) -> Tuple[np.ndarray, np.ndarray]:
    times = []
    bpms = []
    for frame in epoch.frames:
        if frame.heart_rate is None:
            continue
        times.append(float(frame.heart_rate.timestamp))
        bpms.append(float(frame.heart_rate.bpm))
    return np.asarray(times, dtype=float), np.asarray(bpms, dtype=float)


def _estimate_ppg_heart_rate(
    channels: Mapping[str, np.ndarray],
    config: PPGFeatureConfig,
) -> _PPGPeakResult:
    if not channels:
        return _PPGPeakResult("", math.nan, 0.0, (), "Missing")

    channel, values = _primary_ppg_channel(channels)
    finite_values = values[np.isfinite(values)]
    min_samples = int(config.ppg_sample_rate_hz * config.min_ppg_seconds_for_peak_hr)
    if finite_values.size < min_samples:
        return _PPGPeakResult(channel, math.nan, 0.0, (), "Insufficient data")
    if float(np.std(finite_values)) <= config.flat_ppg_std_threshold:
        return _PPGPeakResult(channel, math.nan, 0.0, (), "Flatline")

    filtered = _filter_ppg(finite_values, config)
    std = float(np.std(filtered))
    if not math.isfinite(std) or std <= 0:
        return _PPGPeakResult(channel, math.nan, 0.0, (), "Poor")

    normalized = (filtered - np.mean(filtered)) / std
    peaks, _ = find_peaks(
        normalized,
        distance=max(1, int(0.4 * config.ppg_sample_rate_hz)),
        prominence=config.peak_prominence,
        height=0,
    )
    if peaks.size < 3:
        return _PPGPeakResult(channel, math.nan, 0.0, (), "Too few peaks detected")

    peak_times = peaks / config.ppg_sample_rate_hz
    ibi_seconds = np.diff(peak_times)
    valid_ibi = ibi_seconds[(ibi_seconds > 0.25) & (ibi_seconds < 2.0)]
    if valid_ibi.size < 2:
        return _PPGPeakResult(channel, math.nan, 0.0, tuple(peak_times), "Irregular rhythm")

    mean_ibi = float(np.mean(valid_ibi))
    estimated_hr_bpm = 60.0 / mean_ibi
    confidence = _clamp01(1.0 - _safe_divide(float(np.std(valid_ibi)), mean_ibi))
    return _PPGPeakResult(
        channel=channel,
        estimated_hr_bpm=float(estimated_hr_bpm),
        confidence=confidence,
        peak_times=tuple(float(value) for value in peak_times),
        signal_quality=_signal_quality(confidence),
    )


def _primary_ppg_channel(channels: Mapping[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    best_channel = ""
    best_values = np.asarray((), dtype=float)
    best_std = -1.0
    for channel, values in sorted(channels.items()):
        finite = values[np.isfinite(values)]
        channel_std = float(np.std(finite)) if finite.size else -1.0
        if channel_std > best_std:
            best_channel = channel
            best_values = values
            best_std = channel_std
    return best_channel, best_values


def _filter_ppg(values: np.ndarray, config: PPGFeatureConfig) -> np.ndarray:
    signal = detrend(values)
    nyquist = config.ppg_sample_rate_hz / 2.0
    low = config.ppg_low_hz / nyquist
    high = config.ppg_high_hz / nyquist
    b, a = butter(4, [low, high], btype="band")
    min_length_for_filter = 3 * max(len(a), len(b))
    if signal.size <= min_length_for_filter:
        return signal
    return filtfilt(b, a, signal)


def _heart_rate_summary(
    hr_times: np.ndarray,
    hr_values: np.ndarray,
    ppg_peak_result: _PPGPeakResult,
) -> Mapping[str, object]:
    finite_mask = np.isfinite(hr_times) & np.isfinite(hr_values) & (hr_values > 0)
    finite_times = hr_times[finite_mask]
    finite_values = hr_values[finite_mask]
    if finite_values.size:
        return {
            "mean_hr_bpm": float(np.mean(finite_values)),
            "median_hr_bpm": float(np.median(finite_values)),
            "min_hr_bpm": float(np.min(finite_values)),
            "max_hr_bpm": float(np.max(finite_values)),
            "hr_trend_bpm_per_min": _heart_rate_trend(finite_times, finite_values),
            "hr_source": "heart_rate",
        }
    if math.isfinite(ppg_peak_result.estimated_hr_bpm):
        hr = ppg_peak_result.estimated_hr_bpm
        return {
            "mean_hr_bpm": hr,
            "median_hr_bpm": hr,
            "min_hr_bpm": hr,
            "max_hr_bpm": hr,
            "hr_trend_bpm_per_min": math.nan,
            "hr_source": "ppg",
        }
    return {
        "mean_hr_bpm": math.nan,
        "median_hr_bpm": math.nan,
        "min_hr_bpm": math.nan,
        "max_hr_bpm": math.nan,
        "hr_trend_bpm_per_min": math.nan,
        "hr_source": "missing",
    }


def _heart_rate_trend(times: np.ndarray, values: np.ndarray) -> float:
    if values.size < 2:
        return math.nan
    relative_times = times - float(np.min(times))
    if float(np.max(relative_times)) <= 0:
        return math.nan
    slope_bpm_per_second = float(np.polyfit(relative_times, values, 1)[0])
    return slope_bpm_per_second * 60.0


def _hrv_metrics(
    ppg_peak_times: Tuple[float, ...],
    hr_values: np.ndarray,
) -> Mapping[str, object]:
    if len(ppg_peak_times) >= 3:
        rr_ms = np.diff(np.asarray(ppg_peak_times, dtype=float)) * 1000.0
        return {**_rr_metrics(rr_ms), "hrv_source": "ppg"}

    finite_hr = hr_values[np.isfinite(hr_values) & (hr_values > 0)]
    if finite_hr.size >= 3:
        rr_ms = 60000.0 / finite_hr
        return {**_rr_metrics(rr_ms), "hrv_source": "heart_rate"}

    return {
        "mean_rr_ms": math.nan,
        "sdnn_ms": math.nan,
        "rmssd_ms": math.nan,
        "pnn50_percent": math.nan,
        "hrv_source": "missing",
    }


def _rr_metrics(rr_ms: np.ndarray) -> Mapping[str, float]:
    finite_rr = rr_ms[np.isfinite(rr_ms) & (rr_ms > 0)]
    if finite_rr.size < 2:
        return {
            "mean_rr_ms": math.nan,
            "sdnn_ms": math.nan,
            "rmssd_ms": math.nan,
            "pnn50_percent": math.nan,
        }
    rr_diffs = np.diff(finite_rr)
    return {
        "mean_rr_ms": float(np.mean(finite_rr)),
        "sdnn_ms": float(np.std(finite_rr)),
        "rmssd_ms": _safe_rms(rr_diffs),
        "pnn50_percent": (
            float(np.sum(np.abs(rr_diffs) > 50.0) / rr_diffs.size * 100.0)
            if rr_diffs.size
            else math.nan
        ),
    }


def _optics_samples(
    epoch: SleepEpoch,
    fs: float,
) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
    """Every optics channel of the epoch on one sample clock, and which samples it lost.

    The clock starts at the first optics frame's timestamp and moves ``1 / fs`` a row;
    each frame's rows follow the ones before, in the order the decoder delivers them.
    Samples the decoder reports lost (``PPGSample.gaps``, also those before a frame's
    first row, but not before the epoch's first optics frame) and the time a frame
    arrives more than ``_OPTICS_CLOCK_SLACK_SECONDS`` after the clock says it should are
    NaN in every channel and True in the returned mask.
    """
    values_by_channel: Dict[str, List[float]] = {}
    positions: List[int] = []
    length = 0
    start_time: Optional[float] = None
    for frame in epoch.frames:
        ppg = frame.ppg
        if ppg is None or not ppg.channels:
            continue
        rows = max(len(values) for values in ppg.channels.values())
        if rows == 0:
            continue
        lead, offsets = _frame_row_offsets(rows, ppg.gaps)
        if start_time is None:
            start_time = float(ppg.timestamp)
            lead = 0
        else:
            late = float(ppg.timestamp) - (start_time + (length + lead) / fs)
            if late > _OPTICS_CLOCK_SLACK_SECONDS:
                lead += int(round(late * fs))
        before = len(positions)
        positions.extend((length + lead + offsets).tolist())
        for channel, values in ppg.channels.items():
            column = values_by_channel.setdefault(channel, [math.nan] * before)
            column.extend(values[:rows])
        for column in values_by_channel.values():
            if len(column) < len(positions):
                column.extend([math.nan] * (len(positions) - len(column)))
        length += lead + int(offsets[-1]) + 1

    if start_time is None:
        return math.nan, {}, np.zeros(0, dtype=bool)
    index = np.asarray(positions, dtype=int)
    lost = np.ones(length, dtype=bool)
    lost[index] = False
    channels = {}
    for channel, column in values_by_channel.items():
        values = np.full(length, np.nan)
        values[index] = column
        channels[channel] = values
    return start_time, channels, lost


def _frame_row_offsets(rows: int, gaps: Sequence[Tuple[int, int]]) -> Tuple[int, np.ndarray]:
    """Samples lost before a frame's first row, and the clock offset of each row from it."""
    offsets = np.arange(rows)
    lead = 0
    for index, missing in gaps:
        if missing <= 0:
            continue
        if index <= 0:
            lead += int(missing)
        elif index < rows:
            offsets[index:] += int(missing)
    return lead, offsets


def _band_passed(
    values: np.ndarray,
    missing: np.ndarray,
    config: PPGFeatureConfig,
) -> Optional[np.ndarray]:
    """``values`` with missing samples bridged linearly, detrended and band-passed to
    ``ppg_low_hz``..``ppg_high_hz`` without phase shift; None when fewer samples than one
    window are left or the channel is flat."""
    fs = config.ppg_sample_rate_hz
    present = ~missing
    if int(np.sum(present)) < int(round(config.hr_series_window_seconds * fs)):
        return None
    known = values[present]
    if float(np.std(known)) <= config.flat_ppg_std_threshold:
        return None
    filled = np.array(values, dtype=float)
    if not np.all(present):
        index = np.arange(values.size)
        filled[missing] = np.interp(index[missing], index[present], known)
    nyquist = fs / 2.0
    sections = butter(
        4,
        [config.ppg_low_hz / nyquist, config.ppg_high_hz / nyquist],
        btype="band",
        output="sos",
    )
    signal = detrend(filled)
    return sosfiltfilt(sections, signal, padlen=min(signal.size - 1, int(3 * fs)))


def _most_periodic_channel(
    filtered: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    config: PPGFeatureConfig,
) -> str:
    best_channel = ""
    best_peak = -math.inf
    for channel, (signal, _missing) in sorted(filtered.items()):
        _bpm, peak = _autocorrelation_bpm(signal, config)
        if peak > best_peak:
            best_channel, best_peak = channel, peak
    return best_channel


def _series_points(
    signal: np.ndarray,
    missing: np.ndarray,
    start_time: float,
    config: PPGFeatureConfig,
) -> Tuple[HeartRatePoint, ...]:
    fs = config.ppg_sample_rate_hz
    window = int(round(config.hr_series_window_seconds * fs))
    step = max(1, int(round(config.hr_series_step_seconds * fs)))
    points = []
    for first in range(0, signal.size - window + 1, step):
        time = start_time + (first + window / 2.0) / fs
        if float(np.mean(missing[first:first + window])) > config.hr_series_max_missing_fraction:
            points.append(HeartRatePoint(time, math.nan, math.nan, 0.0, False))
            continue
        segment = signal[first:first + window]
        bpm, peak = _autocorrelation_bpm(segment, config)
        maxima = _pulse_maxima(segment, config)
        peak_bpm = _interval_bpm(maxima, fs)
        quality = _clamp01(peak) if _estimates_agree(bpm, peak_bpm, config) else 0.0
        run_seconds = 0.0
        if quality < config.hr_series_min_quality:
            run = _two_rate_run(segment, maxima, config)
            if run is not None:
                bpm, peak_bpm, quality, run_seconds = run
        used = quality > 0 and quality >= config.hr_series_min_quality
        points.append(
            HeartRatePoint(float(time), float(bpm), float(peak_bpm), quality, used, run_seconds)
        )
    return tuple(points)


def _estimates_agree(bpm: float, peak_bpm: float, config: PPGFeatureConfig) -> bool:
    return bool(
        math.isfinite(bpm)
        and math.isfinite(peak_bpm)
        and abs(bpm - peak_bpm) <= config.hr_series_max_disagreement_bpm
    )


def _two_rate_run(
    segment: np.ndarray,
    maxima: np.ndarray,
    config: PPGFeatureConfig,
) -> Optional[Tuple[float, float, float, float]]:
    """The window's longest run of regular beats, for a window that holds two rates.

    A step or a short surge inside a window gives it beats at two rates, and its
    autocorrelation peaks at neither period (often at two fast beats, near one slow one).
    The run is the longest stretch of at least ``_RUN_MIN_INTERVALS`` consecutive beat
    intervals, each within ``_RUN_INTERVAL_SHARE`` of the window's median interval, that
    spans ``hr_series_min_run_seconds`` or more. It stands in for the window when the
    autocorrelation of that stretch alone agrees with the median of its intervals:
    (bpm, peak_bpm, quality, run seconds), else None. A run in noise rarely holds five
    regular intervals whose stretch also correlates with itself at their period.
    """
    fs = config.ppg_sample_rate_hz
    intervals = np.diff(maxima)
    if intervals.size < _RUN_MIN_INTERVALS:
        return None
    median = float(np.median(intervals))
    regular = np.abs(intervals - median) <= _RUN_INTERVAL_SHARE * median
    best: Optional[Tuple[int, int]] = None
    start: Optional[int] = None
    for index, flag in enumerate(regular.tolist() + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if best is None or maxima[index] - maxima[start] > maxima[best[1]] - maxima[best[0]]:
                best = (start, index)
            start = None
    if best is None:
        return None
    first, last = best
    span = float(maxima[last] - maxima[first])
    if last - first < _RUN_MIN_INTERVALS or span < config.hr_series_min_run_seconds * fs:
        return None
    stretch = segment[int(math.floor(maxima[first])):int(math.ceil(maxima[last])) + 1]
    bpm, peak = _autocorrelation_bpm(stretch, config)
    peak_bpm = 60.0 * fs / float(np.median(intervals[first:last]))
    if not _estimates_agree(bpm, peak_bpm, config):
        return None
    return bpm, peak_bpm, _clamp01(peak), span / fs


def _autocorrelation_bpm(segment: np.ndarray, config: PPGFeatureConfig) -> Tuple[float, float]:
    """Heart rate from the beat period that ``segment`` correlates best with itself at,
    and that correlation (Pearson, over the overlap at each lag); (NaN, 0) when no lag
    in the heart-rate range has a positive peak."""
    fs = config.ppg_sample_rate_hz
    size = segment.size
    shortest = max(2, int(math.floor(fs * 60.0 / config.max_valid_hr_bpm)))
    longest = min(int(math.ceil(fs * 60.0 / config.min_valid_hr_bpm)), size // 2)
    if longest - shortest < 2:
        return math.nan, 0.0
    centred = segment - float(np.mean(segment))
    fft_size = 1 << int(math.ceil(math.log2(2 * size)))
    spectrum = np.fft.rfft(centred, fft_size)
    products = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)
    energy = np.concatenate(([0.0], np.cumsum(centred * centred)))
    lags = np.arange(shortest - 1, longest + 2)
    denominator = np.sqrt(energy[size - lags] * (energy[size] - energy[lags]))
    correlation = np.full(lags.size, np.nan)
    positive = denominator > 0
    correlation[positive] = products[lags[positive]] / denominator[positive]

    inner = correlation[1:-1]
    with np.errstate(invalid="ignore"):
        peaks = np.flatnonzero(
            (inner >= correlation[:-2]) & (inner >= correlation[2:]) & (inner > 0)
        ) + 1
    if peaks.size == 0:
        return math.nan, 0.0
    heights = correlation[peaks]
    chosen = int(peaks[np.argmax(heights >= _AUTOCORRELATION_PEAK_SHARE * float(np.max(heights)))])
    left, middle, right = correlation[chosen - 1], correlation[chosen], correlation[chosen + 1]
    curvature = left - 2.0 * middle + right
    shift = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
    return 60.0 * fs / (float(lags[chosen]) + shift), float(middle)


def _pulse_maxima(segment: np.ndarray, config: PPGFeatureConfig) -> np.ndarray:
    """The light maxima of ``segment`` (the pulse foot), in samples, each placed between
    samples by a parabola."""
    fs = config.ppg_sample_rate_hz
    scale = float(np.std(segment))
    if not math.isfinite(scale) or scale <= 0:
        return np.zeros(0)
    peaks, _ = find_peaks(
        segment,
        distance=max(1, int(math.floor(fs * 60.0 / config.max_valid_hr_bpm))),
        prominence=config.hr_series_peak_prominence * scale,
    )
    left, middle, right = segment[peaks - 1], segment[peaks], segment[peaks + 1]
    curvature = left - 2.0 * middle + right
    bent = curvature < 0
    shift = np.zeros(peaks.size)
    shift[bent] = 0.5 * (left[bent] - right[bent]) / curvature[bent]
    return peaks + shift


def _interval_bpm(maxima: np.ndarray, fs: float) -> float:
    """60 over the median interval between pulse maxima; NaN with fewer than three."""
    if maxima.size < 3:
        return math.nan
    return 60.0 * fs / float(np.median(np.diff(maxima)))


def _series_hr_changes(
    points: Sequence[HeartRatePoint],
    config: PPGFeatureConfig,
) -> Tuple[SuddenHeartRateChange, ...]:
    """Changes of at least ``sudden_hr_change_bpm`` less ``hr_series_margin_bpm``
    between used values ``hr_series_compare_seconds`` to ``hr_series_max_compare_seconds``
    apart, the largest pair ending at each value. The margin is the error of a value, so
    a change of ``sudden_hr_change_bpm`` is not lost to it. A held step shows at several
    neighbouring values; those in one direction at most ``hr_series_compare_seconds``
    apart are one change, logged by its largest pair."""
    shortest = config.hr_series_compare_seconds - _TIME_TOLERANCE_SECONDS
    longest = config.hr_series_max_compare_seconds + _TIME_TOLERANCE_SECONDS
    smallest = config.sudden_hr_change_bpm - config.hr_series_margin_bpm
    used = [point for point in points if point.used]
    changes: List[SuddenHeartRateChange] = []
    last_time = -math.inf
    for index, after in enumerate(used):
        largest: Optional[SuddenHeartRateChange] = None
        for before in used[:index]:
            if not shortest <= after.time - before.time <= longest:
                continue
            delta = float(after.bpm - before.bpm)
            if largest is None or abs(delta) > abs(largest.delta_bpm):
                largest = SuddenHeartRateChange(
                    start_time=before.time,
                    end_time=after.time,
                    before_bpm=before.bpm,
                    after_bpm=after.bpm,
                    delta_bpm=delta,
                )
        if largest is None or abs(largest.delta_bpm) < smallest:
            continue
        same = (
            changes
            and after.time - last_time <= config.hr_series_compare_seconds + _TIME_TOLERANCE_SECONDS
            and (largest.delta_bpm > 0) == (changes[-1].delta_bpm > 0)
        )
        if not same:
            changes.append(largest)
        elif abs(largest.delta_bpm) > abs(changes[-1].delta_bpm):
            changes[-1] = largest
        last_time = after.time
    return tuple(changes)


def _sudden_hr_changes(
    hr_times: np.ndarray,
    hr_values: np.ndarray,
    config: PPGFeatureConfig,
) -> Tuple[SuddenHeartRateChange, ...]:
    """Consecutive heart-rate samples that differ by ``sudden_hr_change_bpm`` or more;
    the rule's input when an epoch has heart-rate samples but no optics."""
    finite_mask = np.isfinite(hr_times) & np.isfinite(hr_values)
    times = hr_times[finite_mask]
    values = hr_values[finite_mask]
    if values.size < 2:
        return ()

    changes = []
    for index in range(1, values.size):
        delta = float(values[index] - values[index - 1])
        if abs(delta) >= config.sudden_hr_change_bpm:
            changes.append(
                SuddenHeartRateChange(
                    start_time=float(times[index - 1]),
                    end_time=float(times[index]),
                    before_bpm=float(values[index - 1]),
                    after_bpm=float(values[index]),
                    delta_bpm=delta,
                )
            )
    return tuple(changes)


def _artifact_flags(
    ppg_channels: Mapping[str, np.ndarray],
    hr_values: np.ndarray,
    ppg_coverage: float,
    heart_rate_coverage: float,
    quality_flags: Tuple[str, ...],
    config: PPGFeatureConfig,
) -> Tuple[str, ...]:
    flags = set(quality_flags)
    if not ppg_channels:
        flags.add("ppg_missing")
    elif ppg_coverage < config.min_ppg_coverage:
        flags.add("low_ppg_coverage")

    for channel, values in ppg_channels.items():
        if values.size == 0:
            flags.add(f"ppg_empty_{channel}")
            continue
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            flags.add(f"ppg_nonfinite_{channel}")
            continue
        if finite.size != values.size:
            flags.add(f"ppg_nonfinite_{channel}")
        if float(np.std(finite)) <= config.flat_ppg_std_threshold:
            flags.add(f"ppg_flatline_{channel}")

    finite_hr = hr_values[np.isfinite(hr_values)]
    if hr_values.size == 0:
        flags.add("heart_rate_missing")
    elif finite_hr.size == 0:
        flags.add("heart_rate_nonfinite")
    elif finite_hr.size != hr_values.size:
        flags.add("heart_rate_nonfinite")

    if heart_rate_coverage < config.min_heart_rate_coverage:
        flags.add("low_heart_rate_coverage")
    if finite_hr.size and (
        np.any(finite_hr < config.min_valid_hr_bpm)
        or np.any(finite_hr > config.max_valid_hr_bpm)
    ):
        flags.add("heart_rate_out_of_range")
    return tuple(sorted(flags))


def _signal_quality(confidence: float) -> str:
    if confidence > 0.8:
        return "Excellent"
    if confidence > 0.6:
        return "Good"
    if confidence > 0.4:
        return "Fair"
    return "Poor"


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


def _safe_median(values: Iterable[float]) -> float:
    values_array = np.asarray(tuple(values), dtype=float)
    finite = values_array[np.isfinite(values_array)]
    if finite.size == 0:
        return math.nan
    return float(np.median(finite))


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def _safe_divide(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return float(numerator / denominator)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))
