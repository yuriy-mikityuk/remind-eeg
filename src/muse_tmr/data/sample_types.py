"""Shared sample types for Muse frames.

All timestamps are Unix epoch seconds as floats. EEG values are microvolts,
IMU acceleration is g, gyroscope values are degrees per second, PPG values are
raw optics units, heart rate is beats per minute, and battery is percent.

EEG, PPG and IMU samples may carry `gaps`, (index, missing) pairs: `missing`
samples the device produced never arrived and belong right before row `index` of
the series. The rows themselves stay joined.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

NumberSeries = Tuple[float, ...]
AxisRows = Tuple[Mapping[str, float], ...]
Gaps = Tuple[Tuple[int, int], ...]


def _timestamp_seconds(value: Any) -> float:
    if isinstance(value, _dt.datetime):
        return value.timestamp()
    return float(value)


def _series(value: Any) -> NumberSeries:
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    return (float(value),)


def _series_mapping(values: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, NumberSeries]]:
    if values is None:
        return None
    return {str(key): _series(value) for key, value in values.items()}


def _gaps(value: Optional[Iterable[Any]]) -> Gaps:
    if not value:
        return ()
    return tuple((int(index), int(missing)) for index, missing in value)


def _axis_rows(rows: Optional[Iterable[Any]], axes: Tuple[str, ...]) -> Optional[AxisRows]:
    if rows is None:
        return None

    normalized = []
    for row in rows:
        if isinstance(row, Mapping):
            normalized.append({str(key): float(value) for key, value in row.items()})
        else:
            normalized.append({
                axes[idx]: float(value)
                for idx, value in enumerate(row)
                if idx < len(axes)
            })
    return tuple(normalized)


def _rows_to_dict(rows: Optional[AxisRows]) -> Optional[Tuple[Dict[str, float], ...]]:
    if rows is None:
        return None
    return tuple(dict(row) for row in rows)


def _bytes_from_hex(value: Optional[str]) -> Optional[bytes]:
    if value in (None, ""):
        return None
    return bytes.fromhex(value)


@dataclass(frozen=True)
class EEGSample:
    timestamp: float
    channels_uv: Mapping[str, NumberSeries]
    source: str = "unknown"
    gaps: Gaps = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp_seconds(self.timestamp))
        object.__setattr__(self, "channels_uv", _series_mapping(self.channels_uv) or {})
        object.__setattr__(self, "gaps", _gaps(self.gaps))

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "channels_uv": {key: list(value) for key, value in self.channels_uv.items()},
            "source": self.source,
        }
        if self.gaps:
            data["gaps"] = [list(gap) for gap in self.gaps]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EEGSample":
        return cls(
            timestamp=data["timestamp"],
            channels_uv=data.get("channels_uv", {}),
            source=data.get("source", "unknown"),
            gaps=data.get("gaps", ()),
        )


@dataclass(frozen=True)
class IMUSample:
    timestamp: float
    accelerometer_g: Optional[AxisRows] = None
    gyroscope_dps: Optional[AxisRows] = None
    source: str = "unknown"
    gaps: Gaps = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp_seconds(self.timestamp))
        object.__setattr__(
            self,
            "accelerometer_g",
            _axis_rows(self.accelerometer_g, ("x", "y", "z")),
        )
        object.__setattr__(
            self,
            "gyroscope_dps",
            _axis_rows(self.gyroscope_dps, ("x", "y", "z")),
        )
        object.__setattr__(self, "gaps", _gaps(self.gaps))

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "accelerometer_g": _rows_to_dict(self.accelerometer_g),
            "gyroscope_dps": _rows_to_dict(self.gyroscope_dps),
            "source": self.source,
        }
        if self.gaps:
            data["gaps"] = [list(gap) for gap in self.gaps]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IMUSample":
        return cls(
            timestamp=data["timestamp"],
            accelerometer_g=data.get("accelerometer_g"),
            gyroscope_dps=data.get("gyroscope_dps"),
            source=data.get("source", "unknown"),
            gaps=data.get("gaps", ()),
        )


@dataclass(frozen=True)
class PPGSample:
    timestamp: float
    channels: Mapping[str, NumberSeries]
    source: str = "unknown"
    gaps: Gaps = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp_seconds(self.timestamp))
        object.__setattr__(self, "channels", _series_mapping(self.channels) or {})
        object.__setattr__(self, "gaps", _gaps(self.gaps))

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "channels": {key: list(value) for key, value in self.channels.items()},
            "source": self.source,
        }
        if self.gaps:
            data["gaps"] = [list(gap) for gap in self.gaps]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PPGSample":
        return cls(
            timestamp=data["timestamp"],
            channels=data.get("channels", {}),
            source=data.get("source", "unknown"),
            gaps=data.get("gaps", ()),
        )


@dataclass(frozen=True)
class HeartRateSample:
    timestamp: float
    bpm: float
    source: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp_seconds(self.timestamp))
        object.__setattr__(self, "bpm", float(self.bpm))

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "bpm": self.bpm, "source": self.source}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeartRateSample":
        return cls(
            timestamp=data["timestamp"],
            bpm=data["bpm"],
            source=data.get("source", "unknown"),
        )


@dataclass(frozen=True)
class BatterySample:
    timestamp: float
    percent: float
    source: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp_seconds(self.timestamp))
        object.__setattr__(self, "percent", float(self.percent))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "percent": self.percent,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BatterySample":
        return cls(
            timestamp=data["timestamp"],
            percent=data["percent"],
            source=data.get("source", "unknown"),
        )


@dataclass(frozen=True)
class MuseFrame:
    timestamp: float
    eeg: Optional[EEGSample] = None
    imu: Optional[IMUSample] = None
    ppg: Optional[PPGSample] = None
    heart_rate: Optional[HeartRateSample] = None
    battery: Optional[BatterySample] = None
    source: str = "unknown"
    raw_packet: Optional[bytes] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp_seconds(self.timestamp))

    def modalities(self) -> Tuple[str, ...]:
        present = []
        for name in ("eeg", "imu", "ppg", "heart_rate", "battery"):
            if getattr(self, name) is not None:
                present.append(name)
        return tuple(present)

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "eeg": self.eeg.to_dict() if self.eeg else None,
            "imu": self.imu.to_dict() if self.imu else None,
            "ppg": self.ppg.to_dict() if self.ppg else None,
            "heart_rate": self.heart_rate.to_dict() if self.heart_rate else None,
            "battery": self.battery.to_dict() if self.battery else None,
            "source": self.source,
        }
        if include_raw:
            data["raw_packet_hex"] = self.raw_packet.hex() if self.raw_packet else None
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MuseFrame":
        return cls(
            timestamp=data["timestamp"],
            eeg=EEGSample.from_dict(data["eeg"]) if data.get("eeg") else None,
            imu=IMUSample.from_dict(data["imu"]) if data.get("imu") else None,
            ppg=PPGSample.from_dict(data["ppg"]) if data.get("ppg") else None,
            heart_rate=(
                HeartRateSample.from_dict(data["heart_rate"])
                if data.get("heart_rate")
                else None
            ),
            battery=(
                BatterySample.from_dict(data["battery"])
                if data.get("battery")
                else None
            ),
            source=data.get("source", "unknown"),
            raw_packet=_bytes_from_hex(data.get("raw_packet_hex")),
        )

    def to_json(self, include_raw: bool = True) -> str:
        return json.dumps(self.to_dict(include_raw=include_raw), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "MuseFrame":
        return cls.from_dict(json.loads(payload))


def frame_from_decoded(decoded: Any, source: str = "amused") -> MuseFrame:
    """Convert a `muse_tmr.ble.muse_realtime_decoder.DecodedData`-like object to a MuseFrame."""

    timestamp = _timestamp_seconds(decoded.timestamp)
    gaps = getattr(decoded, "gaps", None) or {}
    eeg = (
        EEGSample(timestamp, decoded.eeg, source=source, gaps=gaps.get("eeg", ()))
        if decoded.eeg
        else None
    )
    ppg = (
        PPGSample(timestamp, decoded.ppg, source=source, gaps=gaps.get("ppg", ()))
        if decoded.ppg
        else None
    )

    imu = None
    if decoded.imu:
        imu = IMUSample(
            timestamp=timestamp,
            accelerometer_g=decoded.imu.get("accel"),
            gyroscope_dps=decoded.imu.get("gyro"),
            source=source,
            gaps=gaps.get("imu", ()),
        )

    heart_rate = None
    if decoded.heart_rate is not None:
        heart_rate = HeartRateSample(timestamp, decoded.heart_rate, source=source)

    battery = None
    if decoded.battery is not None:
        battery = BatterySample(timestamp, decoded.battery, source=source)

    return MuseFrame(
        timestamp=timestamp,
        eeg=eeg,
        imu=imu,
        ppg=ppg,
        heart_rate=heart_rate,
        battery=battery,
        source=source,
        raw_packet=getattr(decoded, "raw_bytes", None) or None,
    )
