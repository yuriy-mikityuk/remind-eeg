"""Adapter around the forked amused-py BLE source."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import subprocess
import sys
import time
from typing import Any, AsyncIterator, Callable, Mapping, Optional, Sequence

from muse_tmr.ble.muse_realtime_decoder import DecodedData
from muse_tmr.ble.muse_stream_client import MuseStreamClient

from muse_tmr.data.sample_types import MuseFrame, frame_from_decoded
from muse_tmr.sources.base_source import BaseMuseSource, MuseDeviceInfo, MuseSourceMetadata

_DISCOVERY_CHILD_TIMEOUT_SECONDS = 15.0
_DISCOVERY_CHILD_CODE = r"""
import asyncio
import contextlib
import json
import sys

from muse_tmr.ble.muse_discovery import find_muse_devices


async def main():
    with contextlib.redirect_stdout(sys.stderr):
        devices = await find_muse_devices()
    print(json.dumps([
        {
            "name": device.name,
            "address": device.address,
            "rssi": getattr(device, "rssi", -100),
        }
        for device in devices
    ]))


asyncio.run(main())
"""


class AmusedSource(BaseMuseSource):
    """Stream MuseFrames from the existing `MuseStreamClient` implementation."""

    strategy = "forked-source"

    def __init__(
        self,
        address: Optional[str] = None,
        name_filter: str = "Muse",
        preset: str = "p1034",
        duration_seconds: int = 0,
        stream_client_factory: Callable[..., MuseStreamClient] = MuseStreamClient,
        queue_size: int = 1000,
        verbose: bool = True,
    ) -> None:
        self.address = address
        self.name_filter = name_filter
        self.preset = preset
        self.duration_seconds = duration_seconds
        self.stream_client_factory = stream_client_factory
        self.queue_size = queue_size
        self.verbose = verbose

        self.client: Optional[MuseStreamClient] = None
        self.metadata: Optional[MuseSourceMetadata] = None
        self.packet_count = 0
        self.frame_count = 0
        self.last_packet_monotonic: Optional[float] = None
        self.disconnect_reason: Optional[str] = None

        self._queue: Optional[asyncio.Queue[MuseFrame]] = None
        self._queue_loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._stream_loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_requested = False

    async def discover(self) -> Sequence[MuseDeviceInfo]:
        devices = await _discover_muse_devices()
        if self.name_filter:
            devices = [device for device in devices if self.name_filter in device.name]
        return devices

    async def connect(self, device: Optional[MuseDeviceInfo] = None) -> MuseSourceMetadata:
        self._stop_requested = False
        self.disconnect_reason = None
        self.packet_count = 0
        self.frame_count = 0
        self.last_packet_monotonic = None
        if self._stream_task is None or self._stream_task.done():
            self._queue = None
            self._queue_loop = None

        if device is not None:
            self.address = device.address

        if self.address is None:
            devices = await self.discover()
            if not devices:
                raise RuntimeError("No Muse devices found")
            self.address = devices[0].address
            device = devices[0]

        device_name = device.name if device is not None else self.address
        self.client = self.stream_client_factory(
            save_raw=False,
            decode_realtime=True,
            verbose=self.verbose,
        )
        if self.client.decoder:
            self.client.decoder.register_callback("any", self._handle_decoded)
        self.client.on_packet(self._handle_packet)

        self.metadata = MuseSourceMetadata(
            source_name="amused",
            device_name=device_name or "Muse",
            device_id=self.address,
            capabilities={
                "eeg": True,
                "imu": True,
                "ppg": True,
                "heart_rate": True,
                "battery": True,
                "raw_packets": True,
            },
            metadata={"preset": self.preset, "strategy": self.strategy},
        )
        return self.metadata

    async def stream(self) -> AsyncIterator[MuseFrame]:
        if self.client is None:
            await self.connect()

        current_loop = asyncio.get_running_loop()
        if (
            self._stream_task is not None
            and not self._stream_task.done()
            and self._stream_loop is not current_loop
        ):
            raise RuntimeError("amused stream is already active in another event loop")

        queue = self._ensure_queue()
        if self._stream_task is None or self._stream_task.done():
            self._stream_loop = current_loop
            self._stream_task = asyncio.create_task(self._run_client())

        while not self._stop_requested:
            if self._stream_task.done() and queue.empty():
                exc = self._stream_task.exception()
                if exc is not None:
                    raise exc
                break
            try:
                yield await asyncio.wait_for(queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stop_requested = True
        if self._stream_task and not self._stream_task.done():
            current_loop = asyncio.get_running_loop()
            if self._stream_loop is current_loop:
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
            elif self._stream_loop and self._stream_loop.is_running():
                self._stream_loop.call_soon_threadsafe(self._stream_task.cancel)
        self._stream_task = None
        self._stream_loop = None

    def diagnostics(self) -> Mapping[str, Any]:
        decoder_stats = None
        if self.client is not None and self.client.decoder is not None:
            decoder_stats = _json_safe(self.client.decoder.get_stats())

        return {
            "source": "amused",
            "address": self.address,
            "packet_count": self.packet_count,
            "frame_count": self.frame_count,
            "last_packet_age_seconds": (
                time.monotonic() - self.last_packet_monotonic
                if self.last_packet_monotonic is not None
                else None
            ),
            "disconnect_reason": self.disconnect_reason,
            "decoder": decoder_stats,
        }

    async def _run_client(self) -> None:
        assert self.client is not None
        assert self.address is not None
        success = await self.client.connect_and_stream(
            self.address,
            duration_seconds=self.duration_seconds,
            preset=self.preset,
        )
        if not success and not self._stop_requested:
            self.disconnect_reason = "stream_failed"
            raise RuntimeError("amused stream failed")

    def _handle_packet(self, raw_packet: bytes) -> None:
        self.packet_count += 1
        self.last_packet_monotonic = time.monotonic()

    def _handle_decoded(self, decoded: DecodedData) -> None:
        frame = frame_from_decoded(decoded, source="amused")
        self.frame_count += 1
        queue = self._queue
        if queue is None:
            return
        queue_loop = self._queue_loop
        current_loop = _running_loop_or_none()
        if queue_loop is not None and queue_loop.is_running() and queue_loop is not current_loop:
            queue_loop.call_soon_threadsafe(self._put_frame_nowait, queue, frame)
            return
        self._put_frame_nowait(queue, frame)

    def _ensure_queue(self) -> asyncio.Queue[MuseFrame]:
        current_loop = asyncio.get_running_loop()
        if self._queue is None or self._queue_loop is not current_loop:
            self._queue = asyncio.Queue(maxsize=self.queue_size)
            self._queue_loop = current_loop
        return self._queue

    def _put_frame_nowait(self, queue: asyncio.Queue[MuseFrame], frame: MuseFrame) -> None:
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            self.disconnect_reason = "frame_queue_full"


async def _discover_muse_devices() -> Sequence[MuseDeviceInfo]:
    if sys.platform == "darwin":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _discover_muse_devices_subprocess)

    return await _discover_muse_devices_in_process()


async def _discover_muse_devices_in_process() -> Sequence[MuseDeviceInfo]:
    from muse_tmr.ble.muse_discovery import find_muse_devices

    devices = await find_muse_devices()
    return [
        MuseDeviceInfo(name=device.name, address=device.address, rssi=device.rssi)
        for device in devices
    ]


def _discover_muse_devices_subprocess() -> Sequence[MuseDeviceInfo]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _DISCOVERY_CHILD_CODE],
            capture_output=True,
            check=False,
            text=True,
            timeout=_DISCOVERY_CHILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Muse BLE discovery timed out in an isolated process. "
            "Check macOS Bluetooth permission for the terminal/Python app, "
            "then retry scan."
        ) from exc

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        if details:
            details = f": {details.splitlines()[0]}"
        raise RuntimeError(
            "Muse BLE discovery crashed in an isolated process"
            f" (exit {completed.returncode}){details}. "
            "On macOS, grant Bluetooth permission to the terminal/Python app "
            "running muse-tmr, then retry scan."
        )

    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Muse BLE discovery returned invalid device data") from exc

    return [
        MuseDeviceInfo(
            name=str(device["name"]),
            address=str(device["address"]),
            rssi=int(device.get("rssi", -100)),
        )
        for device in payload
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _running_loop_or_none() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
