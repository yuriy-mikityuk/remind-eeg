"""Stdlib HTTP server for the REMind app."""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import mimetypes
import os
import posixpath
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from muse_tmr import __version__
from muse_tmr.contact import (
    ContactGate,
    ContactGateConfig,
    ContactQualityConfig,
    ContactQualityMonitor,
    ContactQualitySnapshot,
    MockContactProvider,
    available_mock_contact_scenarios,
    builtin_contact_snapshots,
)
from muse_tmr.sources.polar_h10 import parse_heart_rate_measurement

CONNECTION_STATES = ("disconnected", "scanning", "connecting", "connected", "error")

CAFFEINATE = "/usr/bin/caffeinate"

# Python's own table has no .webp before 3.13, and Windows does not always register
# it, so the guide pictures went out as application/octet-stream there.
_EXTRA_CONTENT_TYPES = {".webp": "image/webp"}

# The recorder rewrites progress.json every ~2 s while frames arrive.
RECORDER_HEARTBEAT_STALE_SECONDS = 10.0

# After Stop (SIGINT to the group) the recorder gets this long before SIGKILL.
# With a Polar H10 child it first waits for the child to stop the strap's
# streams and write its summary (up to 30 s, then SIGTERM for 10 s).
STOP_GRACE_SECONDS = 5.0
STOP_GRACE_WITH_POLAR_SECONDS = 50.0
POLAR_STALE_SECONDS = 10.0

# kind -> (preset, duration_hours, allow_short)
RECORDING_KINDS: Dict[str, Tuple[str, float, bool]] = {
    "night": ("p21", 8.0, False),
    "session": ("p1034", 1.0, True),
}
# Meditation and calibration sessions need clean EEG and take heart rate and
# breathing from the H10. On p1034 the optics put a 64 Hz line into every EEG
# channel (32-47 dB above its neighbours on the 2026-10-09 line check, gone on
# p21), so these record without the optics.
EEG_ONLY_PRESET = "p21"
# The gate sees the contact only when the page asks for it, every second. The
# contact monitor judges each channel over its last 2 s, so looks at most 2 s
# apart cover every moment; a longer gap (a throttled background tab) restarts
# the stability window instead of counting the unseen time as good contact.
CONTACT_GATE_MAX_GAP_SECONDS = 2.0
# A meditation series keeps its practices and block layout, so its sessions pool
# in aggregate-meditation. Only the settle-in may change between sessions: it is
# read off the EMG timeline of the reports. Kept with the other protocol data.
SERIES_SCHEMA_VERSION = 1
# A series session counts once it has headband data up to its last block's end,
# give or take one 10 s epoch: the panel's countdown can run up to 3 s ahead of
# the recorder, so a stop right at "All blocks done" may land just short of it.
SERIES_END_SLACK_SECONDS = 10.0


class JobUnavailable(Exception):
    """A background job cannot run here (e.g. not a project checkout)."""


@dataclass
class BackgroundJob:
    """A detached helper process for one recording (report build, meditation analysis)."""

    process: Optional[Any] = None  # Popen, or a fake in tests
    launching: bool = False  # reserved under the lock until the process is spawned

    def running(self) -> bool:
        return self.launching or (self.process is not None and self.process.poll() is None)

    def status(self, output_file: Path, url: str, log_path: Path) -> Dict[str, Any]:
        exit_code = self.process.poll() if self.process is not None else None
        if self.running():
            state = "running"
        elif output_file.is_file() and (self.process is None or exit_code == 0):
            state = "ready"
        elif self.process is not None:
            state = "failed"
        else:
            state = "none"
        return {"state": state, "url": url if state == "ready" else None, "log_path": str(log_path)}


@dataclass
class RecordingHandle:
    kind: str
    output_dir: Path
    log_path: Path
    command: List[str]
    preset: str
    duration_seconds: float
    started_at_seconds: float
    pid: Optional[int] = None
    process: Optional[Any] = None  # Popen, or a fake in tests; None after app restart
    state: str = "launching"  # launching | running | stopping | completed | failed
    stop_signalled_at_seconds: Optional[float] = None
    with_polar: bool = False
    meditation: bool = False  # blocks.json from a guided meditation lives in output_dir
    calibration: bool = False  # a calibration-guide process speaks the protocol along it


def _real_launcher(command: List[str], log_path: Path):
    """Spawn a detached recording process that outlives this app.

    start_new_session=True makes the child a session/process-group leader
    detached from the app's controlling terminal, so closing the terminal that
    launched the app does not signal the recorder. Its stdout/stderr go to a log
    file the app can tail.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab", buffering=0)
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        # The child keeps its own dup'd copy of the fd; the parent does not need it.
        log_fh.close()


def _real_terminator(pid: int, sig: int) -> None:
    """Signal the whole process group led by ``pid`` (caffeinate + python)."""
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _read_json_tolerant(path: Path) -> Mapping[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _signal_numbers(progress: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The EEG numbers from the recorder's last heartbeat, for the recordings list.

    Lost and repeated subpackets are summed over the sensor types; recordings made
    before the decoder counted them carry only the rate.
    """
    decoder = (progress.get("source_diagnostics") or {}).get("decoder") or {}
    rate = decoder.get("eeg_effective_sample_rate_hz")
    if rate is None:
        return None
    lost = decoder.get("lost_subpackets")
    repeated = decoder.get("repeated_subpackets")
    return {
        "eeg_rate_hz": rate,
        "lost_subpackets": sum(int(count) for count in lost.values()) if isinstance(lost, dict) else None,
        "repeated_subpackets": sum(int(count) for count in repeated.values()) if isinstance(repeated, dict) else None,
    }


_FOLDER_STAMP = re.compile(r"\d{8}_\d{6}")


def _folder_start(output_dir: Path) -> Optional[str]:
    """Recording folders are named after their start time, sometimes behind a prefix
    (`smoke_muse_h10_20261009_012712`)."""
    match = _FOLDER_STAMP.search(output_dir.name)
    if match is None:
        return None
    try:
        return dt.datetime.strptime(match.group(0), "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        return None


def _newest_first_key(output_dir: Path) -> Tuple[str, str]:
    """A folder's start, or its last change when the name has none."""
    start = _folder_start(output_dir)
    if start is None:
        start = dt.datetime.fromtimestamp(output_dir.stat().st_mtime).isoformat()
    return start, output_dir.name


def _meditation_plan(body: Mapping[str, Any]):
    """A counterbalanced plan from the meditation form; ValueError/TypeError on bad input."""
    import random as _random

    from muse_tmr.reports.meditation_analysis import build_meditation_plan, check_drift_cancelling

    conditions = [str(item).strip() for item in body.get("conditions") or []]
    blocks = int(body.get("blocks", 4))
    block_minutes = float(body.get("block_minutes", 8))
    settle_seconds = float(body.get("settle_seconds", 60))
    seed = body.get("seed")
    seed = int(seed) if seed not in (None, "") else _random.randrange(1_000_000)
    if not 4 <= blocks <= 12 or not 0.5 <= block_minutes <= 60 or not 0 <= settle_seconds <= 600:
        raise ValueError("blocks 4, 8 or 12, block minutes 0.5-60, settle seconds 0-600")
    check_drift_cancelling(blocks)
    return build_meditation_plan(
        conditions, blocks=blocks, block_minutes=block_minutes, settle_seconds=settle_seconds, seed=seed
    )


RECORDING_HEARTBEAT_LIVE_SECONDS = 120.0


def _recording_folder_active(output_dir: Path) -> bool:
    """A recording folder still being written by some recorder process.

    The final summary.json marks it finished. Before that, it is live if the
    recorder from launch.json is still running with this folder on its command
    line (a reused PID won't have it), or if progress.json moved recently (for
    recordings started from the CLI, which write no launch.json).
    """
    if (output_dir / "summary.json").exists():
        return False
    pid = _read_json_tolerant(output_dir / "launch.json").get("pid")
    if isinstance(pid, int) and pid > 0 and _process_mentions(pid, output_dir):
        return True
    try:
        return time.time() - (output_dir / "progress.json").stat().st_mtime < RECORDING_HEARTBEAT_LIVE_SECONDS
    except OSError:
        return False


def _process_mentions(pid: int, output_dir: Path) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        command = subprocess.run(
            # -ww: no width limit, the folder sits at the end of a long recorder command line.
            ["ps", "-ww", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return True  # alive and we cannot tell more: err on the safe side
    return str(output_dir.resolve()) in command or str(output_dir) in command


def _polar_status(output_dir: Path, *, active: bool, now: Optional[float] = None) -> Dict[str, Any]:
    """What the Polar H10 child is doing, from the files it writes.

    state: starting | connected | stale | reconnecting | stopped | failed.
    """
    now = time.time() if now is None else now
    polar_dir = output_dir / "polar"
    status: Dict[str, Any] = {"state": "starting", "last_event": None, "data_age_seconds": None,
                              "heart_rate_bpm": None, "contact": None}
    events = _tail_jsonl(polar_dir / "events.jsonl")
    if events:
        status["last_event"] = events[-1].get("event")
    raw_path = polar_dir / "raw_notifications.jsonl"
    try:
        status["data_age_seconds"] = max(0.0, now - raw_path.stat().st_mtime)
    except OSError:
        pass
    for record in reversed(_tail_jsonl(raw_path)):
        if record.get("char") == "hr":
            try:
                measurement = parse_heart_rate_measurement(base64.b64decode(record["b64"]))
                status["heart_rate_bpm"] = measurement.heart_rate_bpm
                status["contact"] = measurement.sensor_contact
            except Exception:
                pass
            break

    muse_events = [event.get("event") for event in _tail_jsonl(output_dir / "events.jsonl")]
    summary = _read_json_tolerant(polar_dir / "summary.json")
    if summary:
        status["state"] = "stopped"
        status["stop_reason"] = summary.get("stop_reason")
    elif "companion_start_failed" in muse_events or (active and "companion_exited" in muse_events):
        status["state"] = "failed"
    elif status["last_event"] in ("disconnected", "connect_failed"):
        status["state"] = "reconnecting"
    elif status["last_event"] == "connected":
        age = status["data_age_seconds"]
        status["state"] = "connected" if age is not None and age < POLAR_STALE_SECONDS else "stale"
    elif status["last_event"] == "recording_stopped":
        status["state"] = "stopped"
    return status


def _tail_jsonl(path: Path, max_bytes: int = 8192) -> List[Dict[str, Any]]:
    """Complete JSON lines from the end of a file that is being appended to."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            chunk = handle.read()
    except OSError:
        return []
    lines = chunk.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line is probably cut
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def _recorder_live_view(
    progress: Mapping[str, Any], now_seconds: float
) -> Optional[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Contact snapshot and source diagnostics from the recorder's heartbeat.

    Returns None when progress.json carries no contact (older recorder, or no
    frame yet). A heartbeat older than RECORDER_HEARTBEAT_STALE_SECONDS means
    the recorder stopped getting frames, so the contact is reported stale with
    no channels rather than replaying the last good values.
    """
    contact = progress.get("contact")
    if not isinstance(contact, Mapping):
        return None
    heartbeat_age = _heartbeat_age_seconds(progress, now_seconds)
    if heartbeat_age is None or heartbeat_age > RECORDER_HEARTBEAT_STALE_SECONDS:
        contact = {**contact, "stale": True, "channels": {}}
    try:
        snapshot = ContactQualitySnapshot.from_dict(contact)
    except (KeyError, TypeError, ValueError):
        return None

    diagnostics = progress.get("source_diagnostics")
    if isinstance(diagnostics, Mapping):
        diagnostics = dict(diagnostics)
        packet_age = diagnostics.get("last_packet_age_seconds")
        if isinstance(packet_age, (int, float)) and heartbeat_age is not None:
            diagnostics["last_packet_age_seconds"] = packet_age + heartbeat_age
    else:
        diagnostics = None
    return snapshot.to_dict(), diagnostics


def _heartbeat_age_seconds(progress: Mapping[str, Any], now_seconds: float) -> Optional[float]:
    try:
        updated_at = dt.datetime.fromisoformat(str(progress["updated_at"]))
    except (KeyError, ValueError):
        return None
    return max(0.0, now_seconds - updated_at.timestamp())


def _expected_report_path(output_dir: Path) -> str:
    kind = output_dir.parent.name
    if kind in ("night", "session"):
        return str(Path("data/reports") / kind / f"{output_dir.name}.html")
    return str(Path("data/reports/nightly") / f"{output_dir.name}.html")


def _report_command(output_dir: Path) -> str:
    """A command that works pasted into a fresh terminal.

    The report script lives in the repo and writes to a cwd-relative
    data/reports/, so cd to the project root and use the app's own Python.
    """
    from muse_tmr.cli.main import _find_project_root

    project_root = _find_project_root(Path(__file__).resolve())
    parts = (
        _report_python(project_root),
        "scripts/generate_nightly_report.py",
        str(output_dir.resolve()),
    )
    command = " ".join(shlex.quote(part) for part in parts)
    if project_root is None:
        return command
    return f"cd {shlex.quote(str(project_root))} && {command}"


def _report_python(project_root: Optional[Path]) -> str:
    # Under the macOS Python.app launch (see AGENTS.md) sys.executable is the
    # base interpreter and the venv only arrives via PYTHONPATH, which a fresh
    # terminal does not have. Point at the project's venv in that case.
    if sys.prefix == sys.base_prefix and project_root is not None:
        venv_python = project_root / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
    return sys.executable


def _format_hours(hours: float) -> str:
    return str(int(hours)) if float(hours).is_integer() else str(hours)


@dataclass(frozen=True)
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    source: str = "mock"
    address: Optional[str] = None
    name_filter: str = "Muse"
    preset: str = "p1034"
    mock_scenario: str = "mixed_fair_good"
    mock_interval_seconds: float = 1.0
    gate_stability_seconds: float = 5.0
    auto_update: bool = False

    def validate(self) -> None:
        if self.source not in {"mock", "amused"}:
            raise ValueError("app source must be mock or amused")
        if self.port < 0 or self.port > 65535:
            raise ValueError("port must be between 0 and 65535")
        if self.mock_scenario not in available_mock_contact_scenarios():
            raise ValueError(f"unknown mock contact scenario: {self.mock_scenario}")
        if self.mock_interval_seconds < 0:
            raise ValueError("mock_interval_seconds must be non-negative")
        if self.gate_stability_seconds < 0:
            raise ValueError("gate_stability_seconds must be non-negative")


class LocalMuseAppState:
    def __init__(
        self,
        config: AppConfig,
        *,
        launcher: Optional[Callable[[List[str], Path], Any]] = None,
        terminator: Optional[Callable[[int, int], None]] = None,
        recordings_base: Optional[Path] = None,
        now_fn: Optional[Callable[[], dt.datetime]] = None,
        build: Optional[str] = None,
        reports_base: Optional[Path] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.build = build
        self._launcher = launcher if launcher is not None else _real_launcher
        self._terminator = terminator if terminator is not None else _real_terminator
        self._recordings_base = Path(recordings_base) if recordings_base is not None else None
        self._reports_base = Path(reports_base) if reports_base is not None else None
        self._now = now_fn if now_fn is not None else (lambda: dt.datetime.now())
        self._recording: Optional[RecordingHandle] = None
        self._lock = threading.Lock()
        self._connection_state = "disconnected"
        self._device_name: Optional[str] = None
        self._device_address: Optional[str] = config.address
        self._error_message: Optional[str] = None
        self._devices: Sequence[Mapping[str, Any]] = ()
        self._last_scan: Optional[Dict[str, Any]] = None
        self._battery_percent: Optional[float] = None
        # Report builds and meditation analyses, keyed by (recording folder, job kind),
        # so they work for any recording on disk, not just the one in memory.
        self._jobs: Dict[Tuple[str, str], BackgroundJob] = {}
        self._source = None
        self._contact_stop_requested = threading.Event()
        self._contact_thread: Optional[threading.Thread] = None
        self._contact_provider = (
            MockContactProvider.for_scenario(
                config.mock_scenario,
                interval_seconds=config.mock_interval_seconds,
                loop=True,
            )
            if config.source == "mock"
            else None
        )
        self._contact_monitor = (
            ContactQualityMonitor(
                source=config.source,
                config=ContactQualityConfig(sample_rate_hz=128.0),
            )
            if config.source != "mock"
            else None
        )
        self._contact_gate = ContactGate(
            ContactGateConfig(
                required_stability_seconds=config.gate_stability_seconds,
                max_observation_gap_seconds=CONTACT_GATE_MAX_GAP_SECONDS,
            )
        )
        self._start_when_ready_requested = False
        self._last_contact_snapshot = None
        self._connected_at_seconds: Optional[float] = None
        self._session_started_at_seconds: Optional[float] = None
        self._contact_warning_count = 0
        self._contact_warning_events: List[Dict[str, Any]] = []
        self._active_contact_warning_started_at: Optional[float] = None
        self._active_contact_warning_channels: Tuple[str, ...] = ()
        self._active_contact_warning_reasons: Tuple[str, ...] = ()

    def health(self) -> Mapping[str, Any]:
        return {
            "ok": True,
            "service": "muse-tmr-local-app",
            "source": self.config.source,
        }

    def state(self) -> Mapping[str, Any]:
        with self._lock:
            return self._state_unlocked()

    def ui_state(self) -> Mapping[str, Any]:
        with self._lock:
            source = self._source
            snapshot = self._contact_snapshot_unlocked(advance_mock=True)
            gate = self._advance_gate_unlocked(snapshot)
            generated_at_seconds = time.time()
            state = self._state_unlocked(now_seconds=generated_at_seconds)
            contact = snapshot.to_dict()
            gate_payload = gate.to_dict()

        source_diagnostics = self._source_diagnostics(source)
        recording, progress = self._recording_payload_and_progress()
        if recording.get("active"):
            # The app released BLE to the recorder, so its own monitor sees
            # nothing; show what the recorder publishes instead.
            recorder_view = _recorder_live_view(progress, generated_at_seconds)
            if recorder_view is not None:
                contact, source_diagnostics = recorder_view

        return {
            "service": "muse-tmr-local-app",
            "version": __version__,
            "build": self.build,
            "generated_at_seconds": generated_at_seconds,
            "state": state,
            "contact": contact,
            "gate": gate_payload,
            "source_diagnostics": source_diagnostics,
            "recording": recording,
        }

    def contact(self) -> Mapping[str, Any]:
        with self._lock:
            snapshot = self._contact_snapshot_unlocked(advance_mock=True)
            self._advance_gate_unlocked(snapshot)
            return snapshot.to_dict()

    def gate(self) -> Mapping[str, Any]:
        with self._lock:
            snapshot = self._contact_snapshot_unlocked(advance_mock=False)
            return self._advance_gate_unlocked(snapshot).to_dict()

    def diagnostics(self) -> Mapping[str, Any]:
        with self._lock:
            source = self._source
            state = self._state_unlocked()
            contact = (
                self._last_contact_snapshot.to_dict()
                if self._last_contact_snapshot is not None
                else None
            )

        return {
            "service": "muse-tmr-local-app",
            "state": state,
            "contact": contact,
            "source_diagnostics": self._source_diagnostics(source),
        }

    def arm_gate(self) -> Mapping[str, Any]:
        with self._lock:
            snapshot = self._contact_snapshot_unlocked(advance_mock=False)
            self._start_when_ready_requested = True
            state = self._contact_gate.arm(snapshot)
            if state.ready:
                self._start_when_ready_requested = False
                state = self._contact_gate.start(snapshot)
                self._mark_session_started_unlocked()
            return state.to_dict()

    def start_session(self) -> Mapping[str, Any]:
        with self._lock:
            snapshot = self._contact_snapshot_unlocked(advance_mock=False)
            state = self._contact_gate.start(snapshot)
            if state.ready:
                self._start_when_ready_requested = False
                self._mark_session_started_unlocked()
            return state.to_dict()

    def scan(self) -> Mapping[str, Any]:
        with self._lock:
            # A new scan replaces the old answer even if it fails.
            self._devices = ()
            self._last_scan = None
        self._set_state("scanning", error_message=None)
        try:
            if self.config.source == "mock":
                devices = (
                    {
                        "name": "Muse Mock Headband",
                        "address": self.config.address or "mock://muse-s",
                        "rssi": -42,
                    },
                )
            else:
                devices = tuple(_device_to_dict(device) for device in asyncio.run(self._amused_source().discover()))
            devices = tuple(sorted(devices, key=lambda device: -(device.get("rssi") or -999)))
            configured = (self.config.address or "").lower()
            with self._lock:
                self._devices = devices
                self._last_scan = {
                    "at_seconds": time.time(),
                    "count": len(devices),
                    "configured_address": self.config.address,
                    # None when no address is configured: any Muse will do.
                    "configured_found": (
                        any(str(device.get("address", "")).lower() == configured for device in devices)
                        if configured
                        else None
                    ),
                }
                self._connection_state = "disconnected"
                self._error_message = None if devices else "No Muse devices found"
                return self._state_unlocked(extra={"devices": list(devices)})
        except Exception as exc:
            with self._lock:
                self._last_scan = {"at_seconds": time.time(), "count": 0, "failed": True, "error": str(exc)}
            self._set_state("error", error_message=str(exc))
            return self.state()

    def connect(self) -> Mapping[str, Any]:
        with self._lock:
            if self._connection_state == "connected":
                return self._state_unlocked()

        self._set_state("connecting", error_message=None)
        try:
            if self.config.source == "mock":
                with self._lock:
                    self._device_name = "Muse Mock Headband"
                    self._device_address = self.config.address or "mock://muse-s"
                    self._connection_state = "connected"
                    self._connected_at_seconds = time.time()
                    return self._state_unlocked()

            source = self._amused_source()
            metadata = asyncio.run(source.connect())
            with self._lock:
                self._source = source
                self._device_name = self._display_name_unlocked(metadata.device_name, metadata.device_id)
                self._device_address = metadata.device_id
                self._connection_state = "connected"
                self._error_message = None
                self._connected_at_seconds = time.time()
                state = self._state_unlocked()
            self._start_contact_stream(source)
            return state
        except Exception as exc:
            with self._lock:
                self._source = None
                self._contact_thread = None
            self._set_state("error", error_message=str(exc))
            return self.state()

    def disconnect(self) -> Mapping[str, Any]:
        source = None
        thread = None
        with self._lock:
            self._contact_stop_requested.set()
            thread = self._contact_thread
            self._contact_thread = None
            source = self._source
            self._source = None
            self._connection_state = "disconnected"
            self._device_name = None
            self._device_address = self.config.address
            self._battery_percent = None
            self._connected_at_seconds = None
            self._error_message = None
            self._start_when_ready_requested = False
            self._reset_session_unlocked()
            self._contact_gate.disarm()
        if source is not None:
            asyncio.run(source.stop())
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        return self.state()

    def idle_for_update(self) -> bool:
        """True when restarting the app would not interrupt anything."""
        with self._lock:
            if self._recording is not None and self._recording_alive_unlocked():
                return False
            # A restart would lose track of running report/analysis jobs.
            if any(job.running() for job in self._jobs.values()):
                return False
            if self._connection_state in ("scanning", "connecting", "connected"):
                return False
            return self._session_started_at_seconds is None

    def shutdown(self) -> None:
        # Intentionally does NOT stop a running recording: the recorder is a
        # detached process meant to outlive the app.
        self.disconnect()

    def arm_recording_gate(self) -> Mapping[str, Any]:
        """A start button arms the contact gate; the page sends the start once it is ready."""
        with self._lock:
            snapshot = self._contact_snapshot_unlocked(advance_mock=False)
            return self._contact_gate.arm(snapshot).to_dict()

    def disarm_recording_gate(self) -> Mapping[str, Any]:
        with self._lock:
            return self._contact_gate.disarm().to_dict()

    def start_when_contact_ready(
        self, start: Callable[[], Tuple[Mapping[str, Any], HTTPStatus]]
    ) -> Tuple[Mapping[str, Any], HTTPStatus]:
        """Run a recording start only once the armed contact gate is ready.

        Ready means every required channel good for the whole stability window
        since the gate was armed. Not armed, a poor, missing or stale channel, or
        a disconnect refuses the start with the gate's reasons.
        """
        with self._lock:
            snapshot = self._contact_snapshot_unlocked(advance_mock=False)
            gate = self._contact_gate.update(snapshot)
        if gate.state != "ready":
            return {"error": "contact is not ready", "gate": gate.to_dict()}, HTTPStatus.CONFLICT
        return start()

    def start_recording(
        self,
        kind: Optional[str],
        with_polar: bool = False,
        duration_seconds: Optional[float] = None,
        meditation: bool = False,
        calibration: bool = False,
    ) -> Tuple[Mapping[str, Any], HTTPStatus]:
        if kind not in RECORDING_KINDS:
            return {"error": "kind must be night or session"}, HTTPStatus.BAD_REQUEST
        if self.config.source != "amused":
            return {"error": "recording requires the live amused source"}, HTTPStatus.CONFLICT

        preset, duration_hours, _ = RECORDING_KINDS[kind]
        if meditation or calibration:
            preset = EEG_ONLY_PRESET
        with self._lock:
            if self._recording is not None and self._recording_alive_unlocked():
                return {"error": "a recording is already running"}, HTTPStatus.CONFLICT
            output_dir = (
                self._recordings_base_resolved() / kind / self._now().strftime("%Y%m%d_%H%M%S")
            )
            command = self._build_record_command(
                kind, output_dir, with_polar=with_polar, duration_seconds=duration_seconds, preset=preset
            )
            handle = RecordingHandle(
                kind=kind,
                output_dir=output_dir,
                log_path=output_dir / "record.log",
                command=command,
                preset=preset,
                duration_seconds=duration_seconds if duration_seconds is not None else duration_hours * 3600.0,
                started_at_seconds=time.time(),
                state="launching",
                with_polar=bool(with_polar),
                meditation=meditation,
                calibration=calibration,
            )
            self._recording = handle

        # Release our own BLE grip on the headband BEFORE spawning the recorder;
        # the Muse allows only one connection at a time.
        self.disconnect()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._write_launch_json(handle)
            proc = self._launcher(command, handle.log_path)
        except Exception as exc:
            with self._lock:
                if self._recording is handle:
                    handle.state = "failed"
            return {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR

        with self._lock:
            if self._recording is handle:
                handle.process = proc
                handle.pid = getattr(proc, "pid", None)
                handle.state = "running"
        self._write_launch_json(handle)
        return self._recording_payload(), HTTPStatus.OK

    def stop_recording(self) -> Tuple[Mapping[str, Any], HTTPStatus]:
        with self._lock:
            handle = self._recording
            if handle is None or not self._recording_alive_unlocked():
                return {"error": "no recording is running"}, HTTPStatus.CONFLICT
            pid = handle.pid
            handle.state = "stopping"
            handle.stop_signalled_at_seconds = time.time()

        if pid is not None:
            # SIGINT (not SIGTERM) so the recorder's finally-block runs: it closes
            # files cleanly and releases BLE. killpg reaches caffeinate + python.
            self._terminator(pid, signal.SIGINT)
        return self._recording_payload(), HTTPStatus.OK

    def _recording_alive_unlocked(self) -> bool:
        handle = self._recording
        if handle is None or handle.state in ("completed", "failed"):
            return False
        proc = handle.process
        if proc is not None:
            # poll() reaps a finished child, avoiding a zombie reporting as alive.
            return proc.poll() is None
        if handle.pid is not None:
            try:
                os.kill(handle.pid, 0)
                return True
            except (ProcessLookupError, PermissionError):
                return False
        # Slot reserved but not spawned yet.
        return handle.state == "launching"

    def _recording_payload(self) -> Mapping[str, Any]:
        return self._recording_payload_and_progress()[0]

    def _recording_payload_and_progress(self) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        escalate_pid: Optional[int] = None
        with self._lock:
            handle = self._recording
            if handle is None:
                return {"active": False, "state": "idle"}, {}
            alive = self._recording_alive_unlocked()
            if (
                handle.state == "stopping"
                and alive
                and handle.pid is not None
                and handle.stop_signalled_at_seconds is not None
                and time.time() - handle.stop_signalled_at_seconds
                > (STOP_GRACE_WITH_POLAR_SECONDS if handle.with_polar else STOP_GRACE_SECONDS)
            ):
                escalate_pid = handle.pid
            snapshot = (
                handle.kind,
                handle.output_dir,
                handle.log_path,
                handle.pid,
                handle.preset,
                handle.duration_seconds,
                handle.started_at_seconds,
                handle.state,
                handle.with_polar,
                handle.meditation,
                handle.calibration,
            )

        if escalate_pid is not None:
            self._terminator(escalate_pid, signal.SIGKILL)

        (kind, output_dir, log_path, pid, preset, duration, started, state, with_polar, meditation, calibration) = snapshot
        progress = _read_json_tolerant(output_dir / "progress.json")
        summary = _read_json_tolerant(output_dir / "summary.json")
        summary_available = bool(summary)

        if summary_available and alive:
            # The Muse summary is written first, then the recorder waits for its
            # Polar child to stop the strap; it is not done until it exits.
            state = "finishing"
        elif summary_available:
            state = "completed"
        elif not alive and state in ("running", "stopping"):
            state = "failed"

        if state in ("completed", "failed"):
            with self._lock:
                if self._recording is handle:
                    handle.state = state

        elapsed = progress.get("elapsed_seconds")
        if elapsed is None:
            elapsed = max(0.0, time.time() - started)

        reconnects = progress.get("reconnect_attempts")
        if reconnects is None:
            reconnects = summary.get("reconnect_attempts")

        return {
            "active": state in ("launching", "running", "stopping", "finishing"),
            "kind": kind,
            "state": state,
            "pid": pid,
            "preset": preset,
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "report_path": _expected_report_path(output_dir),
            "report_command": _report_command(output_dir),
            "report": self.report_status(output_dir),
            "meditation": self._meditation_payload(output_dir, progress) if meditation else None,
            "calibration": self._calibration_payload(output_dir) if calibration else None,
            "started_at_seconds": started,
            "duration_seconds": duration,
            "elapsed_seconds": elapsed,
            "progress_fraction": min(1.0, elapsed / duration) if duration else None,
            "frame_count": progress.get("frame_count"),
            "battery_percent": progress.get("battery_percent"),
            "reconnect_attempts": reconnects,
            "last_event": progress.get("last_event") or summary.get("stop_reason"),
            "summary_available": summary_available,
            "with_polar": with_polar,
            "polar": _polar_status(output_dir, active=state in ("launching", "running", "stopping", "finishing"))
            if with_polar
            else None,
        }, progress

    def _build_record_command(
        self,
        kind: str,
        output_dir: Path,
        with_polar: bool = False,
        duration_seconds: Optional[float] = None,
        preset: Optional[str] = None,
    ) -> List[str]:
        kind_preset, duration_hours, allow_short = RECORDING_KINDS[kind]
        preset = preset or kind_preset
        command = [
            CAFFEINATE,
            "-s",
            sys.executable,
            "-m",
            "muse_tmr.cli.main",
            "record",
            "--source",
            "amused",
            "--preset",
            preset,
            *(
                ("--duration-seconds", f"{duration_seconds:g}")
                if duration_seconds is not None
                else ("--duration-hours", _format_hours(duration_hours))
            ),
            "--no-data-timeout-seconds",
            "45",
            "--max-reconnect-attempts",
            "1000",
            "--output-dir",
            str(output_dir.resolve()),
            "--quiet",
        ]
        if allow_short:
            command.append("--allow-short")
        if with_polar:
            command.append("--with-polar")
        if duration_seconds is not None:
            # Timed plans count from the first frame, so connecting must not shorten them.
            command.append("--duration-from-first-frame")
        return command

    def reports_base(self) -> Path:
        if self._reports_base is not None:
            return self._reports_base
        return self._recordings_base_resolved().parent / "reports"

    def _report_file(self, output_dir: Path) -> Path:
        kind = output_dir.parent.name if output_dir.parent.name in ("night", "session") else "nightly"
        return self.reports_base() / kind / f"{output_dir.name}.html"

    def build_report(self) -> Tuple[Mapping[str, Any], HTTPStatus]:
        """Build the REM report for the recording the app launched last."""
        output_dir, error = self._finished_handle_dir()
        if error:
            return error
        _job, status = self._start_report(output_dir)
        return (self._recording_payload(), status) if status == HTTPStatus.OK else (_job, status)

    def analyze_meditation(self) -> Tuple[Mapping[str, Any], HTTPStatus]:
        """Analyze the guided meditation the app launched last."""
        output_dir, error = self._finished_handle_dir()
        if error:
            return error
        _job, status = self._start_analysis(output_dir)
        return (self._recording_payload(), status) if status == HTTPStatus.OK else (_job, status)

    def build_report_for(self, kind: Any, name: Any) -> Tuple[Mapping[str, Any], HTTPStatus]:
        output_dir, error = self._recording_dir(kind, name)
        if error:
            return error
        payload, status = self._start_report(output_dir)
        return (self._recording_entry(output_dir), status) if status == HTTPStatus.OK else (payload, status)

    def analyze_meditation_for(self, kind: Any, name: Any) -> Tuple[Mapping[str, Any], HTTPStatus]:
        output_dir, error = self._recording_dir(kind, name)
        if error:
            return error
        payload, status = self._start_analysis(output_dir)
        return (self._recording_entry(output_dir), status) if status == HTTPStatus.OK else (payload, status)

    def _finished_handle_dir(self):
        with self._lock:
            handle = self._recording
            if handle is None:
                return None, ({"error": "no finished recording"}, HTTPStatus.CONFLICT)
            if self._recording_alive_unlocked():
                return None, ({"error": "the recording is still running"}, HTTPStatus.CONFLICT)
            return handle.output_dir, None

    def _recording_dir(self, kind: Any, name: Any):
        """A recording folder from user input, only inside the recordings base."""
        if kind not in RECORDING_KINDS or not isinstance(name, str) or not name or "/" in name or name.startswith("."):
            return None, ({"error": "unknown recording"}, HTTPStatus.BAD_REQUEST)
        base = self._recordings_base_resolved().resolve()
        output_dir = (base / kind / name).resolve()
        if not _is_relative_to(output_dir, base) or not output_dir.is_dir():
            return None, ({"error": "unknown recording"}, HTTPStatus.NOT_FOUND)
        if self._is_live(output_dir):
            return None, ({"error": "the recording is still running"}, HTTPStatus.CONFLICT)
        return output_dir, None

    def _is_live(self, output_dir: Path) -> bool:
        """Still being written: by the recorder we launched, or by one this app
        process no longer knows about (it was restarted, or the CLI started it)."""
        with self._lock:
            handle = self._recording
            if handle is not None and handle.output_dir.resolve() == output_dir.resolve():
                return self._recording_alive_unlocked()
        return _recording_folder_active(output_dir)

    def _start_report(self, output_dir: Path) -> Tuple[Mapping[str, Any], HTTPStatus]:
        def command() -> Tuple[List[str], Path]:
            report_file = self._report_file(output_dir)
            return [
                sys.executable,
                str(self._project_root() / "scripts" / "generate_nightly_report.py"),
                str(output_dir.resolve()),
                "--output",
                str(report_file.resolve()),
            ], report_file

        return self._start_job(output_dir, "report", command, "report.log")

    def _start_analysis(self, output_dir: Path) -> Tuple[Mapping[str, Any], HTTPStatus]:
        def command() -> Tuple[List[str], Path]:
            if not (output_dir / "blocks.json").is_file():
                raise JobUnavailable("this recording has no meditation plan")
            report_dir = self._meditation_report_dir(output_dir)
            return [
                sys.executable,
                "-m",
                "muse_tmr.cli.main",
                "analyze-meditation",
                str(output_dir.resolve()),
                "--blocks",
                str((output_dir / "blocks.json").resolve()),
                "--output-dir",
                str(report_dir.resolve()),
            ], report_dir / "report.html"

        return self._start_job(output_dir, "analysis", command, "meditation-analysis.log")

    def _job(self, output_dir: Path, kind: str) -> BackgroundJob:
        return self._jobs.setdefault((str(Path(output_dir).resolve()), kind), BackgroundJob())

    def _start_job(
        self,
        output_dir: Path,
        kind: str,
        build_command: Callable[[], Tuple[List[str], Path]],
        log_name: str,
    ) -> Tuple[Mapping[str, Any], HTTPStatus]:
        with self._lock:
            job = self._job(output_dir, kind)
            if job.running():
                return {}, HTTPStatus.OK
            # Reserve before releasing the lock so a second tab cannot start another one.
            job.launching = True
        try:
            command, output_file = build_command()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            process = self._launcher(command, Path(output_dir) / log_name)
        except JobUnavailable as exc:
            with self._lock:
                job.launching = False
            return {"error": str(exc)}, HTTPStatus.CONFLICT
        except Exception as exc:
            with self._lock:
                job.launching = False
            return {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR
        with self._lock:
            job.process = process
            job.launching = False
        return {}, HTTPStatus.OK

    def list_recordings(self, limit: int = 30) -> Dict[str, Any]:
        """The newest `limit` recordings, and how many there are in all."""
        base = self._recordings_base_resolved()
        folders = []
        for kind in RECORDING_KINDS:
            kind_dir = base / kind
            if kind_dir.is_dir():
                folders.extend(path for path in kind_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
        folders.sort(key=_newest_first_key, reverse=True)
        return {"recordings": [self._recording_entry(path) for path in folders[:limit]], "total": len(folders)}

    def _recording_entry(self, output_dir: Path) -> Dict[str, Any]:
        summary = _read_json_tolerant(output_dir / "summary.json")
        progress = _read_json_tolerant(output_dir / "progress.json")
        live = self._is_live(output_dir)
        meditation = (output_dir / "blocks.json").is_file()
        calibration = (output_dir / "calibration" / "cues.jsonl").is_file()
        return {
            "kind": output_dir.parent.name,
            "name": output_dir.name,
            "started_at": _folder_start(output_dir),
            "live": live,
            "duration_seconds": summary.get("duration_seconds", progress.get("elapsed_seconds")),
            "stop_reason": summary.get("stop_reason") or ("recording" if live else None),
            "frame_count": summary.get("frame_count", progress.get("frame_count")),
            "signal": _signal_numbers(progress),
            "with_polar": (output_dir / "polar").is_dir(),
            "meditation": meditation,
            "report": None if live else self.report_status(output_dir),
            "meditation_report": self._meditation_analysis_status(output_dir) if meditation and not live else None,
            "calibration": calibration,
            "calibration_report": self._calibration_report_status(output_dir) if calibration and not live else None,
        }

    def _project_root(self) -> Path:
        from muse_tmr.cli.main import _find_project_root

        root = _find_project_root(Path(__file__).resolve())
        if root is None:
            raise JobUnavailable("report script not found: not running from a project checkout")
        return root

    def report_status(self, output_dir: Path) -> Dict[str, Any]:
        report_file = self._report_file(output_dir)
        url = f"/reports/{report_file.relative_to(self.reports_base()).as_posix()}"
        return self._job(output_dir, "report").status(report_file, url, output_dir / "report.log")

    def _meditation_analysis_status(self, output_dir: Path) -> Dict[str, Any]:
        report_dir = self._meditation_report_dir(output_dir)
        url = f"/reports/{(report_dir / 'report.html').relative_to(self.reports_base()).as_posix()}"
        return self._job(output_dir, "analysis").status(
            report_dir / "report.html", url, output_dir / "meditation-analysis.log"
        )

    def _meditation_report_dir(self, output_dir: Path) -> Path:
        return self.reports_base() / "meditation" / output_dir.name

    # --- guided meditation ------------------------------------------------

    def start_meditation(self, body: Mapping[str, Any]) -> Tuple[Mapping[str, Any], HTTPStatus]:
        """Build an A/B plan, start a session recording that covers it, store blocks.json."""
        try:
            plan = _meditation_plan(body)
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}, HTTPStatus.BAD_REQUEST
        return self._start_meditation_plan(plan, with_polar=bool(body.get("with_polar")))

    def _start_meditation_plan(self, plan, *, with_polar: bool) -> Tuple[Mapping[str, Any], HTTPStatus]:
        from muse_tmr.reports.meditation_analysis import write_meditation_blocks

        # A minute of slack for connecting and the last epoch.
        duration = plan.blocks[-1].end_s + 60.0
        payload, status = self.start_recording(
            "session", with_polar=with_polar, duration_seconds=duration, meditation=True
        )
        if status != HTTPStatus.OK:
            return payload, status
        write_meditation_blocks(plan, Path(payload["output_dir"]) / "blocks.json")
        return self._recording_payload(), HTTPStatus.OK

    def save_meditation_rating(self, body: Mapping[str, Any]) -> Tuple[Mapping[str, Any], HTTPStatus]:
        from muse_tmr.reports.meditation_analysis import load_meditation_blocks, write_meditation_blocks

        def rating(name: str) -> Optional[float]:
            value = body.get(name)
            if value in (None, ""):
                return None
            number = float(value)
            if not 0 <= number <= 10:
                raise ValueError(f"{name} must be between 0 and 10")
            return number

        with self._lock:
            handle = self._recording
            if handle is None or not handle.meditation:
                return {"error": "no meditation recording"}, HTTPStatus.CONFLICT
            path = handle.output_dir / "blocks.json"
            try:
                index = int(body["block_index"])
                # Every save carries all three ratings; depth holds stability.
                ratings = {name: rating(name) for name in ("depth", "sensory_fading", "drowsiness")}
                plan = load_meditation_blocks(path)
                if not any(block.index == index for block in plan.blocks):
                    raise ValueError(f"no block {index}")
                updated = replace(
                    plan,
                    blocks=tuple(
                        replace(block, **ratings) if block.index == index else block
                        for block in plan.blocks
                    ),
                )
                tmp = path.with_suffix(".json.tmp")
                write_meditation_blocks(updated, tmp)
                os.replace(tmp, path)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                return {"error": str(exc)}, HTTPStatus.BAD_REQUEST
        return self._recording_payload(), HTTPStatus.OK

    def _meditation_payload(self, output_dir: Path, progress: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "plan": _read_json_tolerant(output_dir / "blocks.json") or None,
            # Block times count from the first Muse frame, not from the recorder start.
            "first_frame_elapsed_seconds": progress.get("first_frame_elapsed_seconds"),
            "analysis": self._meditation_analysis_status(output_dir),
        }

    # --- meditation series ---------------------------------------------------

    def meditation_series(self) -> Dict[str, Any]:
        """The current series, its sessions oldest first, and how many count."""
        from muse_tmr.reports.meditation_analysis import MIN_SESSIONS_FOR_INFERENCE

        series = self._load_series()
        sessions = self._series_sessions(series["id"]) if series else []
        return {
            "series": series,
            "sessions": sessions,
            "counted": sum(1 for session in sessions if session["state"] == "counted"),
            "target": MIN_SESSIONS_FOR_INFERENCE,
        }

    def start_meditation_series(self, body: Mapping[str, Any]) -> Tuple[Mapping[str, Any], HTTPStatus]:
        """The next session of the series, or the first one of a new series.

        The first session sets the practices, block count and block length; later
        ones keep them and take only the settle-in from the request. Always
        records the H10, whatever the checkbox says.
        """
        series = self._load_series()
        settings = dict(body)
        if series is not None:
            settings.update({key: series.get(key) for key in ("conditions", "blocks", "block_minutes")})
            if settings.get("settle_seconds") in (None, ""):
                settings["settle_seconds"] = series.get("settle_seconds")
        try:
            plan = _meditation_plan(settings)
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}, HTTPStatus.BAD_REQUEST
        if series is None:
            now = self._now()
            first = plan.blocks[0]
            series = {
                "schema_version": SERIES_SCHEMA_VERSION,
                "id": f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}",
                "created_at": now.isoformat(timespec="seconds"),
                "conditions": list(plan.conditions),
                "blocks": len(plan.blocks),
                "block_minutes": (first.end_s - first.start_s) / 60.0,
            }
        series["settle_seconds"] = plan.settle_seconds
        payload, status = self._start_meditation_plan(replace(plan, series=series["id"]), with_polar=True)
        if status == HTTPStatus.OK:
            self._write_series(series)
        return payload, status

    def new_meditation_series(self) -> Dict[str, Any]:
        """Set the current series aside; its sessions stay as they are and the
        next series start makes a new one."""
        series = self._load_series()
        if series is not None:
            path = self._series_path()
            path.replace(path.with_name(f"series_{series['id']}.json"))
        return self.meditation_series()

    def _series_path(self) -> Path:
        return self._recordings_base_resolved().parent / "protocol" / "meditation" / "series.json"

    def _load_series(self) -> Optional[Dict[str, Any]]:
        series = _read_json_tolerant(self._series_path())
        return dict(series) if isinstance(series, dict) and series.get("id") else None

    def _write_series(self, series: Mapping[str, Any]) -> None:
        path = self._series_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(series, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def _series_sessions(self, series_id: str) -> List[Dict[str, Any]]:
        session_dir = self._recordings_base_resolved() / "session"
        if not session_dir.is_dir():
            return []
        sessions = []
        for output_dir in sorted(session_dir.iterdir(), key=lambda path: path.name):
            plan = _read_json_tolerant(output_dir / "blocks.json")
            if isinstance(plan, dict) and plan.get("series") == series_id:
                sessions.append(self._series_session(output_dir, plan))
        return sessions

    def _series_session(self, output_dir: Path, plan: Mapping[str, Any]) -> Dict[str, Any]:
        """state: recording | counted | short (stopped early or lost the headband) |
        no_data (no Muse frame ever came) | unfinished (no summary)."""
        needed = max((float(block.get("end_s") or 0.0) for block in plan.get("blocks") or ()), default=0.0)
        session: Dict[str, Any] = {
            "name": output_dir.name,
            "started_at": _folder_start(output_dir),
            "needed_seconds": needed,
            "covered_seconds": None,
        }
        summary = _read_json_tolerant(output_dir / "summary.json")
        if self._is_live(output_dir):
            session["state"] = "recording"
        elif not summary:
            session["state"] = "unfinished"
        else:
            # Block times count from the first Muse frame, the recorder's duration from its
            # start. Its downtime is the time spent reconnecting, without the 45 s of silence
            # before each reconnect, so after a dropout this still overstates the data a bit.
            first_frame = _read_json_tolerant(output_dir / "progress.json").get("first_frame_elapsed_seconds")
            if first_frame is None:
                session["state"] = "no_data"
            else:
                downtime = float(summary.get("downtime_seconds") or 0.0)
                covered = float(summary.get("duration_seconds") or 0.0) - float(first_frame) - downtime
                session["covered_seconds"] = covered
                session["state"] = "counted" if covered >= needed - SERIES_END_SLACK_SECONDS else "short"
        return session

    # --- the morning after a puzzle night ----------------------------------------

    def _protocol_dir(self) -> Path:
        return self._recordings_base_resolved().parent / "protocol"

    def morning_nights(self) -> Dict[str, Any]:
        from muse_tmr.app import morning

        return morning.nights_payload(self._protocol_dir(), self.reports_base())

    def morning_night(self, name: str) -> Tuple[Mapping[str, Any], HTTPStatus]:
        from muse_tmr.app import morning

        try:
            return morning.night_payload(self._protocol_dir(), self.reports_base(), name), HTTPStatus.OK
        except morning.UnknownNight as exc:
            return {"error": str(exc)}, HTTPStatus.NOT_FOUND

    def save_morning(self, form: str, body: Mapping[str, Any]) -> Tuple[Mapping[str, Any], HTTPStatus]:
        """Save the dream report or the blind retest of a puzzle night."""
        from muse_tmr.app import morning

        save = morning.save_dream_report if form == "dream-report" else morning.save_retest
        try:
            return save(self._protocol_dir(), self.reports_base(), body), HTTPStatus.OK
        except morning.MorningFileExists as exc:
            return {"error": str(exc), "exists": True}, HTTPStatus.CONFLICT
        except morning.UnknownNight as exc:
            return {"error": str(exc)}, HTTPStatus.NOT_FOUND
        except KeyError as exc:
            return {"error": f"missing {exc}"}, HTTPStatus.BAD_REQUEST
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}, HTTPStatus.BAD_REQUEST

    # --- calibration run -----------------------------------------------------

    def start_calibration(self, body: Optional[Mapping[str, Any]] = None) -> Tuple[Mapping[str, Any], HTTPStatus]:
        """Voice-guided calibration: a session recording with the H10, plus the guide.

        The guide (``calibration-guide``) waits for the first Muse frame, speaks
        the protocol through ``say`` and writes the cue log and blocks files into
        the recording folder. It stops by itself when the recording ends. The
        body's ``language`` (en or ru) is the language it speaks.
        """
        from muse_tmr.protocol.calibration import LANGUAGES, RECORD_SECONDS

        language = (body or {}).get("language")
        if language is not None and language not in LANGUAGES:
            return {"error": f"language must be one of {', '.join(LANGUAGES)}"}, HTTPStatus.BAD_REQUEST
        payload, status = self.start_recording(
            "session", with_polar=True, duration_seconds=RECORD_SECONDS, calibration=True
        )
        if status != HTTPStatus.OK:
            return payload, status
        output_dir = Path(payload["output_dir"])
        pid = payload.get("pid")

        def command() -> Tuple[List[str], Path]:
            argv = [
                CAFFEINATE,
                "-i",
                sys.executable,
                "-m",
                "muse_tmr.cli.main",
                "calibration-guide",
                str(output_dir.resolve()),
            ]
            if pid is not None:
                argv += ["--recorder-pid", str(pid)]
            if language is not None:
                argv += ["--language", language]
            return argv, output_dir / "calibration" / "segments.json"

        guide_payload, guide_status = self._start_job(output_dir, "guide", command, "calibration-guide.log")
        if guide_status != HTTPStatus.OK:
            # A calibration run without its voice is no use; do not leave it recording.
            self.stop_recording()
            return guide_payload, guide_status
        return self._recording_payload(), HTTPStatus.OK

    def calibration_protocol(self) -> Dict[str, Any]:
        from muse_tmr.protocol.calibration import PROTOCOL, PROTOCOL_SECONDS

        return {
            "protocol_seconds": PROTOCOL_SECONDS,
            "steps": [
                {"name": item.name, "label": item.label, "start_s": item.start_s, "end_s": item.end_s}
                for item in PROTOCOL
            ],
        }

    def build_calibration_report(self) -> Tuple[Mapping[str, Any], HTTPStatus]:
        output_dir, error = self._finished_handle_dir()
        if error:
            return error
        _job, status = self._start_calibration_report(output_dir)
        return (self._recording_payload(), status) if status == HTTPStatus.OK else (_job, status)

    def calibration_report_for(self, kind: Any, name: Any) -> Tuple[Mapping[str, Any], HTTPStatus]:
        output_dir, error = self._recording_dir(kind, name)
        if error:
            return error
        payload, status = self._start_calibration_report(output_dir)
        return (self._recording_entry(output_dir), status) if status == HTTPStatus.OK else (payload, status)

    def _start_calibration_report(self, output_dir: Path) -> Tuple[Mapping[str, Any], HTTPStatus]:
        def command() -> Tuple[List[str], Path]:
            if not (output_dir / "calibration" / "cues.jsonl").is_file():
                raise JobUnavailable("this recording is not a calibration run")
            report_dir = self._calibration_report_dir(output_dir)
            return [
                sys.executable,
                "-m",
                "muse_tmr.cli.main",
                "calibration-report",
                str(output_dir.resolve()),
                "--output-dir",
                str(report_dir.resolve()),
            ], report_dir / "report.html"

        return self._start_job(output_dir, "calibration_report", command, "calibration-report.log")

    def _calibration_report_dir(self, output_dir: Path) -> Path:
        return self.reports_base() / "calibration" / output_dir.name

    def _calibration_report_status(self, output_dir: Path) -> Dict[str, Any]:
        report_dir = self._calibration_report_dir(output_dir)
        url = f"/reports/{(report_dir / 'report.html').relative_to(self.reports_base()).as_posix()}"
        return self._job(output_dir, "calibration_report").status(
            report_dir / "report.html", url, output_dir / "calibration-report.log"
        )

    def _calibration_payload(self, output_dir: Path) -> Dict[str, Any]:
        from muse_tmr.protocol.calibration import read_state

        guide = self._job(output_dir, "guide")
        return {
            "step": read_state(output_dir),
            "guide_running": guide.running(),
            "guide_log": str(output_dir / "calibration-guide.log"),
            "report": self._calibration_report_status(output_dir),
        }

    def _recordings_base_resolved(self) -> Path:
        if self._recordings_base is not None:
            return self._recordings_base
        from muse_tmr.cli.main import _default_path_base

        return _default_path_base() / "data" / "recordings"

    def _write_launch_json(self, handle: RecordingHandle) -> None:
        payload = {
            "kind": handle.kind,
            "preset": handle.preset,
            "pid": handle.pid,
            "duration_seconds": handle.duration_seconds,
            "started_at_seconds": handle.started_at_seconds,
            "command": list(handle.command),
            "log_path": str(handle.log_path),
            "output_dir": str(handle.output_dir),
            "with_polar": handle.with_polar,
            "calibration": handle.calibration,
        }
        try:
            handle.output_dir.mkdir(parents=True, exist_ok=True)
            (handle.output_dir / "launch.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    def _state_unlocked(
        self,
        extra: Optional[Mapping[str, Any]] = None,
        now_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = time.time() if now_seconds is None else float(now_seconds)
        payload: Dict[str, Any] = {
            "source": self.config.source,
            "connection_state": self._connection_state,
            "connected_at_seconds": self._connected_at_seconds,
            "connected_elapsed_seconds": (
                now - self._connected_at_seconds
                if self._connected_at_seconds is not None
                else None
            ),
            "device": (
                {
                    "name": self._device_name,
                    "address": self._device_address,
                }
                if self._device_name or self._device_address
                else None
            ),
            "error_message": self._error_message,
            "devices": list(self._devices),
            "scan": dict(self._last_scan) if self._last_scan else None,
            "battery_percent": self._battery_percent if self._connection_state == "connected" else None,
            "mock": {
                "scenario": self.config.mock_scenario,
                "interval_seconds": self.config.mock_interval_seconds,
            }
            if self.config.source == "mock"
            else None,
            "session": self._session_payload_unlocked(now),
            "available_states": list(CONNECTION_STATES),
        }
        if extra:
            payload.update(dict(extra))
        return payload

    def _source_diagnostics(self, source) -> Optional[Mapping[str, Any]]:
        return (
            source.diagnostics()
            if source is not None and hasattr(source, "diagnostics")
            else None
        )

    def _set_state(self, connection_state: str, error_message: Optional[str] = None) -> None:
        with self._lock:
            self._connection_state = connection_state
            self._error_message = error_message

    def _advance_gate_unlocked(self, snapshot: ContactQualitySnapshot):
        state = self._contact_gate.update(snapshot)
        if self._start_when_ready_requested and state.ready:
            self._start_when_ready_requested = False
            state = self._contact_gate.start(snapshot)
            self._mark_session_started_unlocked()
            return state
        if state.state == "running" and self._session_started_at_seconds is None:
            self._mark_session_started_unlocked()
        self._track_contact_warning_unlocked(snapshot, state)
        return state

    def _mark_session_started_unlocked(self) -> None:
        if self._session_started_at_seconds is not None:
            return
        self._session_started_at_seconds = time.time()
        self._contact_warning_count = 0
        self._contact_warning_events = []
        self._active_contact_warning_started_at = None
        self._active_contact_warning_channels = ()
        self._active_contact_warning_reasons = ()

    def _reset_session_unlocked(self) -> None:
        self._session_started_at_seconds = None
        self._contact_warning_count = 0
        self._contact_warning_events = []
        self._active_contact_warning_started_at = None
        self._active_contact_warning_channels = ()
        self._active_contact_warning_reasons = ()

    def _session_payload_unlocked(self, now: float) -> Dict[str, Any]:
        active_warning = None
        if self._active_contact_warning_started_at is not None:
            active_warning = {
                "started_at_seconds": self._active_contact_warning_started_at,
                "elapsed_seconds": now - self._active_contact_warning_started_at,
                "channels": list(self._active_contact_warning_channels),
                "reason_codes": list(self._active_contact_warning_reasons),
            }
        return {
            "running": self._session_started_at_seconds is not None,
            "started_at_seconds": self._session_started_at_seconds,
            "elapsed_seconds": (
                now - self._session_started_at_seconds
                if self._session_started_at_seconds is not None
                else None
            ),
            "contact_warning_count": self._contact_warning_count,
            "active_contact_warning": active_warning,
            "contact_warning_events": list(self._contact_warning_events[-20:]),
        }

    def _track_contact_warning_unlocked(self, snapshot: ContactQualitySnapshot, gate_state) -> None:
        if gate_state.state != "running" or self._session_started_at_seconds is None:
            return

        bad_channels = []
        reasons = []
        for channel in snapshot.required_channels:
            channel_state = snapshot.channels.get(channel)
            if channel_state is None or channel_state.status == "good":
                continue
            bad_channels.append(f"{channel} {channel_state.status}")
            reasons.extend(channel_state.reason_codes)

        now = time.time()
        if bad_channels:
            bad_tuple = tuple(bad_channels)
            reason_tuple = tuple(dict.fromkeys(str(reason) for reason in reasons if reason))
            if self._active_contact_warning_started_at is None:
                self._active_contact_warning_started_at = now
                self._active_contact_warning_channels = bad_tuple
                self._active_contact_warning_reasons = reason_tuple
                self._contact_warning_count += 1
                self._append_contact_warning_event_unlocked(
                    {
                        "timestamp_seconds": now,
                        "kind": "contact_drop",
                        "channels": list(bad_tuple),
                        "reason_codes": list(reason_tuple),
                        "duration_seconds": None,
                    }
                )
            else:
                self._active_contact_warning_channels = bad_tuple
                self._active_contact_warning_reasons = reason_tuple
            return

        if self._active_contact_warning_started_at is None:
            return

        duration = now - self._active_contact_warning_started_at
        for event in reversed(self._contact_warning_events):
            if event.get("kind") == "contact_drop" and event.get("duration_seconds") is None:
                event["duration_seconds"] = duration
                break
        self._append_contact_warning_event_unlocked(
            {
                "timestamp_seconds": now,
                "kind": "contact_recovered",
                "channels": [],
                "reason_codes": [],
                "duration_seconds": duration,
            }
        )
        self._active_contact_warning_started_at = None
        self._active_contact_warning_channels = ()
        self._active_contact_warning_reasons = ()

    def _append_contact_warning_event_unlocked(self, event: Dict[str, Any]) -> None:
        self._contact_warning_events.append(event)
        if len(self._contact_warning_events) > 50:
            del self._contact_warning_events[:-50]

    def _display_name_unlocked(self, name: Optional[str], address: Optional[str]) -> str:
        """amused names the device by its address when connecting by address;
        prefer the advertised name from the last scan, else plain "Muse"."""
        if name and name != address:
            return name
        for device in self._devices:
            if address and str(device.get("address", "")).lower() == str(address).lower() and device.get("name"):
                return str(device["name"])
        return "Muse"

    def _amused_source(self):
        from muse_tmr.sources.amused_source import AmusedSource

        if self._source is None:
            self._source = AmusedSource(
                address=self.config.address,
                name_filter=self.config.name_filter,
                preset=self.config.preset,
                duration_seconds=0,
                verbose=False,
            )
        return self._source

    def _contact_snapshot_unlocked(self, advance_mock: bool):
        if self._contact_provider is not None:
            if advance_mock or self._last_contact_snapshot is None:
                self._last_contact_snapshot = self._contact_provider.next_snapshot()
            if self._connection_state != "connected":
                missing = builtin_contact_snapshots("all_missing")[0].to_dict()
                missing["source"] = self.config.source
                missing["connection_state"] = self._connection_state
                return ContactQualitySnapshot.from_dict(missing)
            return self._last_contact_snapshot
        if self._contact_monitor is not None:
            self._last_contact_snapshot = self._contact_monitor.snapshot(
                connection_state=self._connection_state,
            )
            return self._last_contact_snapshot
        snapshot = builtin_contact_snapshots("all_missing")[0]
        self._last_contact_snapshot = snapshot
        return snapshot

    def _start_contact_stream(self, source) -> None:
        if self._contact_monitor is None:
            return
        with self._lock:
            if self._contact_thread is not None and self._contact_thread.is_alive():
                return
            self._contact_stop_requested.clear()
            thread = threading.Thread(
                target=self._run_contact_stream,
                args=(source,),
                daemon=True,
            )
            self._contact_thread = thread
        thread.start()

    def _run_contact_stream(self, source) -> None:
        async def consume() -> None:
            try:
                async for frame in source.stream():
                    if self._contact_stop_requested.is_set():
                        break
                    with self._lock:
                        assert self._contact_monitor is not None
                        self._contact_monitor.update(frame)
                        if frame.battery is not None:
                            self._battery_percent = float(frame.battery.percent)
            except Exception as exc:
                self._set_state("error", error_message=str(exc))

        asyncio.run(consume())


class LocalMuseAppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        app_state: LocalMuseAppState,
        static_dir: Path,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.app_state = app_state
        self.static_dir = static_dir


class LocalMuseAppHandler(BaseHTTPRequestHandler):
    server: LocalMuseAppServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._write_json(self.server.app_state.health())
            return
        if path == "/api/muse/ui-state":
            self._write_json(self.server.app_state.ui_state())
            return
        if path == "/api/muse/state":
            self._write_json(self.server.app_state.state())
            return
        if path == "/api/muse/contact":
            self._write_json(self.server.app_state.contact())
            return
        if path == "/api/muse/contact/stream":
            self._write_contact_stream(parse_qs(parsed.query))
            return
        if path == "/api/muse/gate":
            self._write_json(self.server.app_state.gate())
            return
        if path.startswith("/reports/"):
            self._serve_report(path[len("/reports/"):])
            return
        if path == "/api/recordings":
            try:
                limit = int((parse_qs(parsed.query).get("limit") or ["30"])[0])
            except ValueError:
                limit = 30
            self._write_json(self.server.app_state.list_recordings(limit=max(1, min(limit, 1000))))
            return
        if path == "/api/morning/nights":
            self._write_json(self.server.app_state.morning_nights())
            return
        if path == "/api/morning/night":
            name = (parse_qs(parsed.query).get("name") or [""])[0]
            payload, status = self.server.app_state.morning_night(name)
            self._write_json(payload, status=status)
            return
        if path == "/api/calibration/protocol":
            self._write_json(self.server.app_state.calibration_protocol())
            return
        if path == "/api/meditation/series":
            self._write_json(self.server.app_state.meditation_series())
            return
        if path == "/api/muse/diagnostics":
            self._write_json(self.server.app_state.diagnostics())
            return
        self._serve_static()

    def do_POST(self) -> None:
        state = self.server.app_state
        if self.path == "/api/muse/scan":
            self._write_json(self.server.app_state.scan())
            return
        if self.path == "/api/muse/connect":
            self._write_json(self.server.app_state.connect())
            return
        if self.path == "/api/muse/disconnect":
            self._write_json(self.server.app_state.disconnect())
            return
        if self.path == "/api/muse/start-when-ready":
            self._write_json(self.server.app_state.arm_gate())
            return
        if self.path == "/api/session/start":
            state = self.server.app_state.start_session()
            status = HTTPStatus.OK if state.get("ready") else HTTPStatus.CONFLICT
            self._write_json(state, status=status)
            return
        if self.path == "/api/session/record":
            body = self._read_json_body()
            payload, status = state.start_when_contact_ready(
                lambda: state.start_recording(body.get("kind"), with_polar=bool(body.get("with_polar")))
            )
            self._write_json(payload, status=status)
            return
        if self.path == "/api/session/record/arm":
            self._write_json(state.arm_recording_gate())
            return
        if self.path == "/api/session/record/disarm":
            self._write_json(state.disarm_recording_gate())
            return
        if self.path == "/api/session/record/stop":
            payload, status = self.server.app_state.stop_recording()
            self._write_json(payload, status=status)
            return
        if self.path == "/api/session/report":
            payload, status = self.server.app_state.build_report()
            self._write_json(payload, status=status)
            return
        if self.path == "/api/meditation/start":
            body = self._read_json_body()
            payload, status = state.start_when_contact_ready(lambda: state.start_meditation(body))
            self._write_json(payload, status=status)
            return
        if self.path == "/api/meditation/rating":
            payload, status = self.server.app_state.save_meditation_rating(self._read_json_body())
            self._write_json(payload, status=status)
            return
        if self.path == "/api/meditation/series/start":
            body = self._read_json_body()
            payload, status = state.start_when_contact_ready(lambda: state.start_meditation_series(body))
            self._write_json(payload, status=status)
            return
        if self.path == "/api/meditation/series/new":
            self._write_json(self.server.app_state.new_meditation_series())
            return
        if self.path in ("/api/recordings/report", "/api/recordings/analyze", "/api/recordings/calibration-report"):
            body = self._read_json_body()
            state = self.server.app_state
            action = {
                "/api/recordings/report": state.build_report_for,
                "/api/recordings/analyze": state.analyze_meditation_for,
                "/api/recordings/calibration-report": state.calibration_report_for,
            }[self.path]
            payload, status = action(body.get("kind"), body.get("name"))
            self._write_json(payload, status=status)
            return
        if self.path in ("/api/morning/dream-report", "/api/morning/retest"):
            payload, status = state.save_morning(self.path.rsplit("/", 1)[1], self._read_json_body())
            self._write_json(payload, status=status)
            return
        if self.path == "/api/calibration/start":
            body = self._read_json_body()
            payload, status = state.start_when_contact_ready(lambda: state.start_calibration(body))
            self._write_json(payload, status=status)
            return
        if self.path == "/api/calibration/report":
            payload, status = self.server.app_state.build_calibration_report()
            self._write_json(payload, status=status)
            return
        if self.path == "/api/meditation/analyze":
            payload, status = self.server.app_state.analyze_meditation()
            self._write_json(payload, status=status)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown app endpoint")

    def _read_json_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            return {}
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def log_message(self, format: str, *args) -> None:
        return

    def _write_json(self, payload: Mapping[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_contact_stream(self, params: Mapping[str, Sequence[str]]) -> None:
        count = int(params.get("count", ["0"])[0] or "0")
        interval_seconds = float(params.get("interval", ["1.0"])[0] or "1.0")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        sent = 0
        while count <= 0 or sent < count:
            payload = json.dumps(self.server.app_state.contact(), sort_keys=True)
            event = f"event: contact\ndata: {payload}\n\n".encode("utf-8")
            try:
                self.wfile.write(event)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
            sent += 1
            if interval_seconds > 0 and (count <= 0 or sent < count):
                time.sleep(interval_seconds)

    def _serve_report(self, relative_path: str) -> None:
        """Generated HTML reports only, from inside the reports folder."""
        root = self.server.app_state.reports_base().resolve()
        normalized = posixpath.normpath("/" + relative_path).lstrip("/")
        file_path = (root / normalized).resolve()
        if file_path.suffix != ".html" or not _is_relative_to(file_path, root) or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self) -> None:
        relative_path = self.path.split("?", 1)[0]
        if relative_path in {"", "/"}:
            relative_path = "/index.html"
        normalized = posixpath.normpath(relative_path).lstrip("/")
        if normalized == "favicon.ico":
            normalized = "icon.svg"
        if normalized.startswith("../") or normalized == "..":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        static_root = self.server.static_dir.resolve()
        file_path = (self.server.static_dir / normalized).resolve()
        if not _is_relative_to(file_path, static_root) or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = file_path.read_bytes()
        content_type = (
            mimetypes.guess_type(str(file_path))[0]
            or _EXTRA_CONTENT_TYPES.get(file_path.suffix.lower())
            or "application/octet-stream"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_local_app_server(
    config: AppConfig,
    *,
    launcher: Optional[Callable[[List[str], Path], Any]] = None,
    terminator: Optional[Callable[[int, int], None]] = None,
    recordings_base: Optional[Path] = None,
    now_fn: Optional[Callable[[], dt.datetime]] = None,
    build: Optional[str] = None,
    reports_base: Optional[Path] = None,
) -> LocalMuseAppServer:
    config.validate()
    static_dir = resources.files("muse_tmr.app").joinpath("static")
    return LocalMuseAppServer(
        (config.host, config.port),
        LocalMuseAppHandler,
        app_state=LocalMuseAppState(
            config,
            launcher=launcher,
            terminator=terminator,
            recordings_base=recordings_base,
            now_fn=now_fn,
            build=build,
            reports_base=reports_base,
        ),
        static_dir=Path(str(static_dir)),
    )


def run_local_app(config: AppConfig) -> int:
    from muse_tmr.app.auto_update import AutoUpdater, current_build
    from muse_tmr.cli.main import _find_project_root

    project_root = _find_project_root(Path(__file__).resolve())
    build = current_build(project_root) if project_root is not None else None
    server = create_local_app_server(config, build=build)
    host, port = server.server_address
    print(
        f"REMind serving at http://{host}:{port} (build {build or 'unknown'})",
        flush=True,
    )
    updater = None
    if config.auto_update:
        if project_root is None:
            print("auto-update off: not running from a project checkout")
        else:
            updater = AutoUpdater(project_root, is_idle=server.app_state.idle_for_update)
            updater.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if updater is not None:
            updater.stop()
        server.app_state.shutdown()
        server.server_close()
    return 0


def _device_to_dict(device) -> Mapping[str, Any]:
    return {
        "name": device.name,
        "address": device.address,
        "rssi": device.rssi,
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
