"""Offline replay for recorded Muse sessions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Mapping, Optional, Sequence

from muse_tmr.ble.muse_raw_stream import MuseRawStream
from muse_tmr.ble.muse_realtime_decoder import MuseRealtimeDecoder

from muse_tmr.data.sample_types import MuseFrame, frame_from_decoded
from muse_tmr.sources.base_source import BaseMuseSource, MuseDeviceInfo, MuseSourceMetadata


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration for offline replay.

    `speed=1.0` replays in real time, `speed=10.0` is 10x faster, and
    `speed=0.0` disables sleeps for deterministic tests and batch processing.
    Time ranges are seconds relative to the raw recording start.
    """

    input_path: Path
    speed: float = 0.0
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    source_name: str = "replay"

    def validate(self) -> None:
        if self.speed < 0:
            raise ValueError("speed must be non-negative")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("end_seconds must be greater than or equal to start_seconds")


class ReplaySession(BaseMuseSource):
    """Replay recorded raw Muse packets as MuseFrames."""

    def __init__(self, config: ReplayConfig) -> None:
        config.validate()
        self.config = config
        self.raw_path = resolve_raw_path(config.input_path)
        self.recording_dir = self.raw_path.parent
        self.metadata: Optional[MuseSourceMetadata] = None
        self._stop_requested = False

    async def discover(self) -> Sequence[MuseDeviceInfo]:
        return [
            MuseDeviceInfo(
                name=f"Replay {self.raw_path.name}",
                address=str(self.raw_path),
                rssi=0,
                metadata={"recording_dir": str(self.recording_dir)},
            )
        ]

    async def connect(self, device: Optional[MuseDeviceInfo] = None) -> MuseSourceMetadata:
        self._stop_requested = False
        source_metadata = self._load_source_metadata()
        self.metadata = MuseSourceMetadata(
            source_name=self.config.source_name,
            device_name=source_metadata.get("device_name", "recorded Muse"),
            device_id=source_metadata.get("device_id", self.raw_path.stem),
            capabilities=_capabilities_from_metadata(source_metadata),
            metadata={
                "raw_path": str(self.raw_path),
                "recording_dir": str(self.recording_dir),
                "original_source": source_metadata.get("source_name", "unknown"),
                "speed": str(self.config.speed),
            },
        )
        return self.metadata

    async def stream(self) -> AsyncIterator[MuseFrame]:
        if self.metadata is None:
            await self.connect()

        decoder = MuseRealtimeDecoder()
        raw_stream = MuseRawStream(str(self.raw_path))
        previous_timestamp = None

        try:
            raw_stream.open_read()
            session_start = raw_stream.session_start
            for packet in raw_stream.read_packets():
                if self._stop_requested:
                    break

                offset_seconds = (packet.timestamp - session_start).total_seconds()
                if self.config.start_seconds is not None and offset_seconds < self.config.start_seconds:
                    continue
                if self.config.end_seconds is not None and offset_seconds > self.config.end_seconds:
                    break

                if previous_timestamp is not None and self.config.speed > 0:
                    delta_seconds = (packet.timestamp - previous_timestamp).total_seconds()
                    await asyncio.sleep(max(0.0, delta_seconds / self.config.speed))
                previous_timestamp = packet.timestamp

                decoded = decoder.decode(packet.data, packet.timestamp)
                yield frame_from_decoded(decoded, source=self.config.source_name)
        finally:
            raw_stream.close()

    async def stop(self) -> None:
        self._stop_requested = True

    def _load_source_metadata(self) -> Mapping[str, object]:
        metadata_path = self.recording_dir / "metadata.json"
        if not metadata_path.exists():
            return {}

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = payload.get("source", {})
        return source if isinstance(source, Mapping) else {}


def resolve_raw_path(input_path: Path) -> Path:
    path = input_path.expanduser()
    if path.is_dir():
        path = path / "raw_amused.bin"
    if not path.exists():
        raise FileNotFoundError(f"Replay input not found: {path}")
    if not path.is_file():
        raise ValueError(f"Replay input must be a raw file or recording directory: {path}")
    return path.resolve()


def _capabilities_from_metadata(source_metadata: Mapping[str, object]) -> Mapping[str, bool]:
    capabilities = source_metadata.get("capabilities")
    if isinstance(capabilities, Mapping):
        return {str(key): bool(value) for key, value in capabilities.items()}
    return {
        "eeg": True,
        "imu": True,
        "ppg": True,
        "heart_rate": True,
        "battery": True,
        "raw_packets": True,
    }
