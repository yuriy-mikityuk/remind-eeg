"""Overnight recording orchestration."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from muse_tmr.ble.muse_raw_stream import MuseRawStream

from muse_tmr.contact import ContactQualityConfig, ContactQualityMonitor
from muse_tmr.data.sample_types import MuseFrame
from muse_tmr.data.watchdog import RecordingWatchdog, WatchdogEvent
from muse_tmr.sources.base_source import BaseMuseSource, MuseSourceMetadata


@dataclass(frozen=True)
class RecordingConfig:
    output_dir: Path
    duration_seconds: float
    source_name: str = "amused"
    no_data_timeout_seconds: float = 30.0
    modality_timeout_seconds: float = 120.0
    max_reconnect_attempts: int = 5
    allow_short: bool = False
    # Count the duration from the first frame instead of the recorder start, so
    # a slow connect does not eat into a timed plan (guided meditation). Until
    # the first frame arrives the run may wait up to first_frame_grace_seconds.
    duration_from_first_frame: bool = False
    first_frame_grace_seconds: float = 600.0

    def validate(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not self.allow_short and not 7200 <= self.duration_seconds <= 28800:
            raise ValueError("overnight recordings must be between 2 and 8 hours")


@dataclass(frozen=True)
class RecordingSummary:
    output_dir: str
    raw_path: str
    decoded_frames_path: str
    metadata_path: str
    events_path: str
    summary_path: str
    started_at: str
    ended_at: str
    duration_seconds: float
    frame_count: int
    raw_packet_count: int
    decoded_frame_count: int
    modality_counts: Dict[str, int]
    reconnect_attempts: int
    downtime_seconds: float
    stop_reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "raw_path": self.raw_path,
            "decoded_frames_path": self.decoded_frames_path,
            "metadata_path": self.metadata_path,
            "events_path": self.events_path,
            "summary_path": self.summary_path,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "raw_packet_count": self.raw_packet_count,
            "decoded_frame_count": self.decoded_frame_count,
            "modality_counts": self.modality_counts,
            "reconnect_attempts": self.reconnect_attempts,
            "downtime_seconds": self.downtime_seconds,
            "stop_reason": self.stop_reason,
        }


class CompanionProcess:
    """A child recorder (e.g. ``record-polar``) beside the Muse recording.

    Its failures are only ever logged as Muse events: nothing here may raise
    into the Muse recorder or change its stop_reason.
    """

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        log_path: Path,
        stop_timeout_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self.command = list(command)
        self.log_path = Path(log_path)
        self.stop_timeout_seconds = stop_timeout_seconds
        self.process: Optional[subprocess.Popen] = None
        self.exit_reported = False

    def start(self) -> int:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab") as log:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
            )
        return self.process.pid

    def poll(self) -> Optional[int]:
        """Return code once, the first time the child is seen to have exited."""
        if self.process is None or self.exit_reported:
            return None
        code = self.process.poll()
        if code is not None:
            self.exit_reported = True
        return code

    def request_stop(self) -> None:
        """SIGINT for a clean stop (the child closes its own streams)."""
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)

    def wait(self) -> Optional[int]:
        """Wait for the child after request_stop; SIGTERM, then SIGKILL, if it hangs."""
        if self.process is None:
            return None
        try:
            self.process.wait(timeout=self.stop_timeout_seconds)
        except subprocess.TimeoutExpired:
            # Ask once more with SIGTERM, which record-polar also handles cleanly.
            self.process.terminate()
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        return self.process.returncode


class OvernightRecorder:
    """Record MuseFrames, raw packets, metadata, events, and a summary."""

    def __init__(
        self,
        config: RecordingConfig,
        watchdog: Optional[RecordingWatchdog] = None,
        companions: Sequence[CompanionProcess] = (),
    ) -> None:
        config.validate()
        self.config = config
        self.watchdog = watchdog or RecordingWatchdog(
            no_data_timeout_seconds=config.no_data_timeout_seconds,
            modality_timeout_seconds=config.modality_timeout_seconds,
        )
        self._last_event_name: Optional[str] = None
        self.companions: List[CompanionProcess] = list(companions)
        # The recorder holds the only BLE connection while it runs, so it also
        # publishes contact quality for the app. 128 Hz matches the app's live
        # amused monitor.
        self._contact_monitor = ContactQualityMonitor(
            source=config.source_name,
            config=ContactQualityConfig(sample_rate_hz=128.0),
        )

    async def record(self, source: BaseMuseSource) -> RecordingSummary:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        raw_path = self.config.output_dir / "raw_amused.bin"
        decoded_frames_path = self.config.output_dir / "decoded_frames.jsonl"
        metadata_path = self.config.output_dir / "metadata.json"
        events_path = self.config.output_dir / "events.jsonl"
        summary_path = self.config.output_dir / "summary.json"
        progress_path = self.config.output_dir / "progress.json"

        started_at_dt = dt.datetime.now(dt.timezone.utc)
        started_monotonic = time.monotonic()
        deadline = started_monotonic + self.config.duration_seconds
        if self.config.duration_from_first_frame:
            deadline += self.config.first_frame_grace_seconds

        frame_count = 0
        raw_packet_count = 0
        decoded_frame_count = 0
        reconnect_attempts = 0
        downtime_seconds = 0.0
        modality_counts: Dict[str, int] = {}
        stop_reason = "duration_complete"
        last_battery_percent: Optional[float] = None
        last_progress_write = 0.0
        first_frame_elapsed: Optional[float] = None

        def write_summary() -> RecordingSummary:
            ended_at_dt = dt.datetime.now(dt.timezone.utc)
            summary = RecordingSummary(
                output_dir=str(self.config.output_dir),
                raw_path=str(raw_path),
                decoded_frames_path=str(decoded_frames_path),
                metadata_path=str(metadata_path),
                events_path=str(events_path),
                summary_path=str(summary_path),
                started_at=started_at_dt.isoformat(),
                ended_at=ended_at_dt.isoformat(),
                duration_seconds=(ended_at_dt - started_at_dt).total_seconds(),
                frame_count=frame_count,
                raw_packet_count=raw_packet_count,
                decoded_frame_count=decoded_frame_count,
                modality_counts=modality_counts,
                reconnect_attempts=reconnect_attempts,
                downtime_seconds=downtime_seconds,
                stop_reason=stop_reason,
            )
            summary_path.write_text(
                json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return summary

        try:
            metadata = await source.connect()
        except asyncio.CancelledError:
            # Stop pressed while still connecting (amused discovery can take
            # several seconds): nothing was recorded, but finish like any stop.
            _uncancel_current_task()
            stop_reason = "user_stopped"
            await source.stop()
            with events_path.open("w", encoding="utf-8") as events_file:
                self._write_event(
                    events_file,
                    WatchdogEvent(
                        event="recording_stopped",
                        timestamp=time.monotonic(),
                        details={"reason": stop_reason},
                    ),
                )
            return write_summary()
        self._write_metadata(metadata_path, metadata, started_at_dt)

        self._write_progress(
            progress_path,
            elapsed_seconds=0.0,
            frame_count=0,
            decoded_frame_count=0,
            battery_percent=None,
            reconnect_attempts=0,
        )

        raw_stream = MuseRawStream(str(raw_path))
        raw_stream.open_write()

        with events_path.open("w", encoding="utf-8") as events_file, decoded_frames_path.open(
            "w", encoding="utf-8"
        ) as decoded_frames_file:
            self._write_event(
                events_file,
                WatchdogEvent(
                    event="recording_started",
                    timestamp=started_monotonic,
                    details={"source": metadata.source_name},
                ),
            )
            self._companions_start(events_file)

            try:
                stream = source.stream().__aiter__()
                while time.monotonic() < deadline:
                    if cancel_requested():
                        raise asyncio.CancelledError()
                    timeout = min(
                        self.config.no_data_timeout_seconds,
                        max(0.01, deadline - time.monotonic()),
                    )
                    try:
                        frame = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
                    except asyncio.TimeoutError:
                        now = time.monotonic()
                        if now >= deadline:
                            break
                        event = self.watchdog.no_data_event(now)
                        if event:
                            self._write_event(events_file, event)

                        new_stream, reconnect_attempts, exhausted, added_downtime = (
                            await self._reconnect_until_ready(
                                source, events_file, reconnect_attempts, deadline
                            )
                        )
                        downtime_seconds += added_downtime
                        if new_stream is None:
                            if exhausted:
                                stop_reason = "max_reconnect_attempts"
                            break
                        stream = new_stream
                        continue
                    except StopAsyncIteration:
                        stop_reason = "source_ended"
                        break
                    except Exception as exc:
                        now = time.monotonic()
                        self._write_event(
                            events_file,
                            WatchdogEvent(
                                event="stream_error",
                                timestamp=now,
                                details={"error": str(exc)},
                            ),
                        )

                        new_stream, reconnect_attempts, exhausted, added_downtime = (
                            await self._reconnect_until_ready(
                                source, events_file, reconnect_attempts, deadline
                            )
                        )
                        downtime_seconds += added_downtime
                        if new_stream is None:
                            if exhausted:
                                stop_reason = "max_reconnect_attempts"
                            break
                        stream = new_stream
                        continue

                    frame_count += 1
                    if first_frame_elapsed is None:
                        # Plans (e.g. meditation blocks) count from the first frame.
                        first_frame_elapsed = time.monotonic() - started_monotonic
                        if self.config.duration_from_first_frame:
                            deadline = time.monotonic() + self.config.duration_seconds
                    for modality in frame.modalities():
                        modality_counts[modality] = modality_counts.get(modality, 0) + 1

                    if frame.raw_packet:
                        packet_timestamp = dt.datetime.fromtimestamp(frame.timestamp)
                        if packet_timestamp < raw_stream.session_start:
                            packet_timestamp = raw_stream.session_start
                        raw_stream.write_packet(
                            frame.raw_packet,
                            packet_timestamp,
                        )
                        raw_packet_count += 1

                    decoded_frames_file.write(frame.to_json(include_raw=False) + "\n")
                    decoded_frames_file.flush()
                    decoded_frame_count += 1

                    if frame.battery is not None:
                        last_battery_percent = frame.battery.percent
                    self._contact_monitor.observe(frame)

                    now_monotonic = time.monotonic()
                    if now_monotonic - last_progress_write >= 2.0:
                        self._write_progress(
                            progress_path,
                            elapsed_seconds=now_monotonic - started_monotonic,
                            frame_count=frame_count,
                            decoded_frame_count=decoded_frame_count,
                            battery_percent=last_battery_percent,
                            reconnect_attempts=reconnect_attempts,
                            first_frame_elapsed_seconds=first_frame_elapsed,
                            contact=self._contact_monitor.snapshot(
                                now_seconds=frame.timestamp
                            ).to_dict(),
                            source_diagnostics=_source_diagnostics(source),
                        )
                        last_progress_write = now_monotonic
                        self._companions_poll(events_file)

                    for event in self.watchdog.observe_frame(frame, time.monotonic()):
                        self._write_event(events_file, event)
            except asyncio.CancelledError:
                # The app's Stop button (or Ctrl-C) sends SIGINT, and asyncio.run
                # cancels this task. Treat it as a normal stop so the summary
                # still gets written.
                _uncancel_current_task()
                stop_reason = "user_stopped"
            finally:
                try:
                    raw_stream.close()
                    await source.stop()
                finally:
                    # Even if Muse cleanup raises, the child must hear about it:
                    # it has to stop its own H10 streams.
                    self._companions_request_stop(events_file)

            self._write_event(
                events_file,
                WatchdogEvent(
                    event="recording_stopped",
                    timestamp=time.monotonic(),
                    details={"reason": stop_reason},
                ),
            )

        summary = write_summary()
        # Only now wait on children, so a slow one cannot cost the Muse summary.
        self._companions_wait(events_path)
        return summary

    def _companion_event(self, events_file, event: str, companion: CompanionProcess, **details) -> None:
        try:
            self._write_event(
                events_file,
                WatchdogEvent(
                    event=event,
                    timestamp=time.monotonic(),
                    details={"companion": companion.name, **details},
                ),
            )
        except Exception:
            pass

    def _companions_start(self, events_file) -> None:
        for companion in self.companions:
            try:
                pid = companion.start()
                self._companion_event(events_file, "companion_started", companion, pid=pid)
            except Exception as exc:
                self._companion_event(events_file, "companion_start_failed", companion, error=str(exc))

    def _companions_poll(self, events_file) -> None:
        for companion in self.companions:
            try:
                code = companion.poll()
                if code is not None:
                    self._companion_event(events_file, "companion_exited", companion, returncode=code)
            except Exception as exc:
                self._companion_event(events_file, "companion_error", companion, error=str(exc))

    def _companions_request_stop(self, events_file) -> None:
        for companion in self.companions:
            try:
                companion.request_stop()
                self._companion_event(events_file, "companion_stop_requested", companion)
            except Exception as exc:
                self._companion_event(events_file, "companion_error", companion, error=str(exc))

    def _companions_wait(self, events_path: Path) -> None:
        if not self.companions:
            return
        try:
            with events_path.open("a", encoding="utf-8") as events_file:
                for companion in self.companions:
                    try:
                        code = companion.wait()
                        self._companion_event(events_file, "companion_stopped", companion, returncode=code)
                    except Exception as exc:
                        self._companion_event(events_file, "companion_error", companion, error=str(exc))
        except Exception:
            pass

    async def _reconnect_until_ready(
        self,
        source: BaseMuseSource,
        events_file,
        reconnect_attempts: int,
        deadline: float,
    ) -> Tuple[Optional[Any], int, bool, float]:
        """Retry connecting to `source` until it succeeds, the reconnect
        budget is exhausted, or the recording deadline passes.

        A failure raised by `source.stop()`/`source.connect()` during a
        reconnect attempt (e.g. the device is genuinely gone) is treated as
        just another failed attempt instead of being allowed to propagate
        and crash the whole recording.

        Returns (new_stream_or_None, updated_reconnect_attempts,
        budget_exhausted, downtime_seconds_added).
        """
        downtime_added = 0.0
        while True:
            if reconnect_attempts >= self.config.max_reconnect_attempts:
                return None, reconnect_attempts, True, downtime_added
            if time.monotonic() >= deadline:
                return None, reconnect_attempts, False, downtime_added

            reconnect_attempts += 1
            backoff = self.watchdog.reconnect_backoff(reconnect_attempts)
            downtime_start = time.monotonic()
            self._write_event(
                events_file,
                WatchdogEvent(
                    event="reconnect_scheduled",
                    timestamp=downtime_start,
                    details={"attempt": reconnect_attempts, "backoff_seconds": backoff},
                ),
            )
            try:
                await source.stop()
                await asyncio.sleep(backoff)
                await source.connect()
                stream = source.stream().__aiter__()
            except Exception as exc:
                downtime_added += time.monotonic() - downtime_start
                self._write_event(
                    events_file,
                    WatchdogEvent(
                        event="reconnect_failed",
                        timestamp=time.monotonic(),
                        details={"attempt": reconnect_attempts, "error": str(exc)},
                    ),
                )
                continue

            downtime_added += time.monotonic() - downtime_start
            return stream, reconnect_attempts, False, downtime_added

    def _write_metadata(
        self,
        metadata_path: Path,
        metadata: MuseSourceMetadata,
        started_at: dt.datetime,
    ) -> None:
        payload = {
            "started_at": started_at.isoformat(),
            "source": {
                "source_name": metadata.source_name,
                "device_name": metadata.device_name,
                "device_id": metadata.device_id,
                "capabilities": dict(metadata.capabilities),
                "metadata": dict(metadata.metadata or {}),
            },
            "config": {
                "duration_seconds": self.config.duration_seconds,
                "source_name": self.config.source_name,
                "no_data_timeout_seconds": self.config.no_data_timeout_seconds,
                "modality_timeout_seconds": self.config.modality_timeout_seconds,
                "max_reconnect_attempts": self.config.max_reconnect_attempts,
            },
        }
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_event(self, events_file, event: WatchdogEvent) -> None:
        events_file.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        events_file.flush()
        self._last_event_name = event.event

    def _write_progress(
        self,
        path: Path,
        *,
        elapsed_seconds: float,
        frame_count: int,
        decoded_frame_count: int,
        battery_percent: Optional[float],
        reconnect_attempts: int,
        first_frame_elapsed_seconds: Optional[float] = None,
        contact: Optional[Dict[str, Any]] = None,
        source_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a small fixed-size heartbeat the app can poll cheaply.

        decoded_frames.jsonl cannot be tailed for battery (it is the first,
        usually-null key on every EEG frame), so the recorder publishes its own
        progress. Written atomically via a temp file + os.replace so a reader
        never observes a torn JSON object.
        """
        payload = {
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "frame_count": frame_count,
            "decoded_frame_count": decoded_frame_count,
            "battery_percent": battery_percent,
            "reconnect_attempts": reconnect_attempts,
            "last_event": self._last_event_name,
            "first_frame_elapsed_seconds": first_frame_elapsed_seconds,
            "contact": contact,
            "source_diagnostics": source_diagnostics,
        }
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, path)


def cancel_requested() -> bool:
    """True once this task has been cancelled, even if the cancel got lost.

    On Python 3.11 asyncio.wait_for can swallow a cancel that races with its
    inner future completing, so a Stop would be ignored until the deadline.
    Task.cancelling() still counts the request (3.11+).
    """
    cancelling = getattr(asyncio.current_task(), "cancelling", None)
    return bool(cancelling is not None and cancelling())


def _uncancel_current_task() -> None:
    """Undo the cancel we are absorbing so asyncio.run returns normally (3.11+)."""
    uncancel = getattr(asyncio.current_task(), "uncancel", None)
    if uncancel is not None:
        uncancel()


def _source_diagnostics(source: BaseMuseSource) -> Optional[Dict[str, Any]]:
    """Best-effort source stats for the app; never allowed to break a recording."""
    diagnostics = getattr(source, "diagnostics", None)
    if diagnostics is None:
        return None
    try:
        payload = dict(diagnostics())
        json.dumps(payload)
    except Exception:
        return None
    return payload
