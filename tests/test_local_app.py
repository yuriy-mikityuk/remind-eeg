import asyncio
import datetime as dt
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import AsyncMock, patch

from muse_tmr import __version__
from muse_tmr.app import AppConfig, create_local_app_server
from muse_tmr.app.server import CAFFEINATE
from muse_tmr.contact import MockContactProvider, builtin_contact_snapshots
from muse_tmr.sources.base_source import MuseSourceMetadata

ON_WINDOWS = sys.platform == "win32"
POSIX_SHELL_LINE = "the report command is a POSIX shell line for the macOS Terminal (shlex quoting, .venv/bin/python)"
POSIX_PROCESS_CONTROL = (
    "the app stops a recorder with SIGINT and SIGKILL to its process group and checks a detached one "
    "with os.kill(pid, 0) and ps; Windows has no SIGKILL or ps, and there os.kill(pid, 0) ends the process"
)


class LoopSafeFakeAmusedSource:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.connect_calls = 0
        self.stream_calls = 0
        self.stop_requested = False
        LoopSafeFakeAmusedSource.instances.append(self)

    async def discover(self):
        return []

    async def connect(self, device=None):
        self.connect_calls += 1
        return MuseSourceMetadata(
            source_name="amused",
            device_name="Muse Test",
            device_id=self.kwargs.get("address") or "test-address",
            capabilities={"eeg": True},
        )

    async def stream(self):
        self.stream_calls += 1
        while not self.stop_requested:
            await asyncio.sleep(0.01)
        if False:
            yield

    async def stop(self):
        self.stop_requested = True

    def diagnostics(self):
        return {"connect_calls": self.connect_calls, "stream_calls": self.stream_calls}


class TestLocalMuseApp(unittest.TestCase):
    def setUp(self):
        self.server = create_local_app_server(AppConfig(port=0, source="mock"))
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.app_state.shutdown()
        self.server.server_close()

    def get_json(self, path):
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_json(self, path):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_health_and_initial_state_are_available_without_ble(self):
        health = self.get_json("/api/health")
        state = self.get_json("/api/muse/state")
        contact = self.get_json("/api/muse/contact")
        ui_state = self.get_json("/api/muse/ui-state")

        self.assertTrue(health["ok"])
        self.assertEqual(health["source"], "mock")
        self.assertEqual(state["connection_state"], "disconnected")
        self.assertEqual(state["source"], "mock")
        self.assertEqual(state["device"], None)
        self.assertEqual(set(contact["required_channels"]), {"TP9", "AF7", "AF8", "TP10"})
        self.assertEqual(set(contact["channels"]), {"TP9", "AF7", "AF8", "TP10"})
        self.assertEqual(contact["connection_state"], "disconnected")
        self.assertEqual(contact["channels"]["AF7"]["status"], "missing")
        self.assertEqual(contact["channels"]["AF7"]["quality_score"], contact["channels"]["AF7"]["fill"])
        self.assertEqual(ui_state["service"], "muse-tmr-local-app")
        self.assertEqual(ui_state["version"], __version__)
        self.assertIn("generated_at_seconds", ui_state)
        self.assertEqual(ui_state["state"]["connection_state"], "disconnected")
        self.assertEqual(ui_state["contact"]["connection_state"], "disconnected")
        self.assertEqual(ui_state["gate"]["state"], "disconnected")
        self.assertIsNone(ui_state["source_diagnostics"])

    def test_contact_stream_emits_sse_snapshots(self):
        with urllib.request.urlopen(
            f"{self.base_url}/api/muse/contact/stream?count=2&interval=0",
            timeout=2,
        ) as response:
            body = response.read().decode("utf-8")

        self.assertEqual(response.headers["Content-Type"], "text/event-stream; charset=utf-8")
        self.assertEqual(body.count("event: contact"), 2)
        self.assertIn("\"required_channels\": [\"TP9\", \"AF7\", \"AF8\", \"TP10\"]", body)

    def test_start_when_ready_arms_gate_and_direct_start_blocks(self):
        armed = self.post_json("/api/muse/start-when-ready")

        self.assertEqual(armed["state"], "armed_waiting_contact")
        self.assertTrue(armed["armed"])
        self.assertFalse(armed["ready"])
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.post_json("/api/session/start")
        self.assertEqual(raised.exception.code, 409)

    def test_mock_scan_connect_and_disconnect_states(self):
        scanned = self.post_json("/api/muse/scan")
        connected = self.post_json("/api/muse/connect")
        contact = self.get_json("/api/muse/contact")
        disconnected = self.post_json("/api/muse/disconnect")

        self.assertEqual(scanned["connection_state"], "disconnected")
        self.assertEqual(scanned["devices"][0]["address"], "mock://muse-s")
        self.assertEqual(connected["connection_state"], "connected")
        self.assertEqual(connected["device"]["name"], "Muse Mock Headband")
        self.assertEqual(contact["channels"]["AF7"]["status"], "fair")
        self.assertEqual(disconnected["connection_state"], "disconnected")

    def test_static_ui_loads_connect_muse_screen(self):
        with urllib.request.urlopen(f"{self.base_url}/", timeout=2) as response:
            body = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{self.base_url}/app.js", timeout=2) as response:
            script = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{self.base_url}/icon.svg", timeout=2) as response:
            self.assertEqual(response.headers.get_content_type(), "image/svg+xml")
        # Reports and other pages get the same icon where browsers look by default.
        with urllib.request.urlopen(f"{self.base_url}/favicon.ico", timeout=2) as response:
            self.assertEqual(response.headers.get_content_type(), "image/svg+xml")

        self.assertIn("Connect Muse", body)
        self.assertIn("headband-title", body)
        self.assertIn("session-strip", body)
        self.assertIn("device-card", body)
        self.assertIn("device-battery", body)
        self.assertIn("scan-result", body)
        self.assertIn("with-polar-checkbox", body)
        self.assertIn("warning-log", body)
        self.assertIn("diagnostics-panel", body)
        self.assertIn("source-badge", body)
        self.assertIn("data-channel=\"TP9\"", body)
        self.assertIn("data-channel=\"AF7\"", body)
        self.assertIn("data-channel=\"AF8\"", body)
        self.assertIn("data-channel=\"TP10\"", body)
        self.assertIn('id="start-session-button"', body)
        self.assertIn('id="start-night-button"', body)
        self.assertIn("Start night", body)
        self.assertIn("<title>REMind</title>", body)
        self.assertIn('href="/icon.svg"', body)
        self.assertIn('id="screen-recordings"', body)
        self.assertIn('id="mode-calibration"', body)
        self.assertIn('id="screen-morning"', body)
        self.assertIn("/api/morning/nights", script)
        self.assertIn("/api/morning/retest", script)
        self.assertIn('id="stop-recording-button"', body)
        self.assertIn('id="recording-strip"', body)
        self.assertIn('id="meditation-sound"', body)
        self.assertIn('id="back-to-menu-button"', body)
        self.assertEqual(body.count('class="voice-language"'), 3)
        self.assertIn("/api/muse/ui-state", script)
        self.assertIn('id="app-version"', body)
        self.assertIn("Version ${payload.version}", script)
        self.assertNotIn('requestJson("/api/muse/state"', script)
        self.assertNotIn('requestJson("/api/muse/contact"', script)
        self.assertNotIn('requestJson("/api/muse/gate"', script)
        self.assertNotIn('requestJson("/api/muse/diagnostics"', script)
        self.assertIn("/api/session/record", script)
        self.assertIn("/api/session/record/arm", script)
        self.assertIn("/api/session/record/disarm", script)
        self.assertIn('id="cancel-start-button"', body)
        # Cancel start and the meditation, series and calibration form Cancel all drop a waiting start.
        self.assertEqual(script.count("cancelPendingStart"), 5)
        self.assertIn("/api/session/record/stop", script)
        self.assertIn("Starting session", script)
        self.assertIn("Session running", script)
        self.assertIn("contact warnings", script)
        self.assertIn("contact-sparkline", script)
        self.assertIn("active-warning-status", script)
        self.assertIn('scanButton.hidden = connection === "connected"', script)
        # The mode cards stay in view and only their buttons wait for a connection.
        self.assertIn("modes.hidden = recordingActive || finished", script)
        self.assertIn("document.title", script)
        self.assertIn("Show older", script)
        self.assertIn("/api/recordings?limit=", script)
        self.assertNotIn("startSessionButton.hidden", script)
        # The all-good check mark is an SVG path, which has no hidden property.
        self.assertIn('allGoodCheck.toggleAttribute("hidden", !snapshot.all_good)', script)
        self.assertNotIn("allGoodCheck.hidden", script)

    def test_page_and_guide_follow_a_dark_system_setting(self):
        for name in ("styles.css", "guide.css"):
            with urllib.request.urlopen(f"{self.base_url}/{name}", timeout=2) as response:
                css = response.read().decode("utf-8")
            dark = css.split("@media (prefers-color-scheme: dark)", 1)
            self.assertEqual(len(dark), 2, name)
            self.assertIn("color-scheme: dark;", dark[1], name)
            self.assertIn("--paper:", dark[1], name)

    def test_meditation_guide_is_linked_and_quotes_the_voice(self):
        with urllib.request.urlopen(f"{self.base_url}/", timeout=2) as response:
            body = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{self.base_url}/app.js", timeout=2) as response:
            script = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{self.base_url}/guide.html", timeout=2) as response:
            self.assertEqual(response.headers.get_content_type(), "text/html")
            guide = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{self.base_url}/guide.css", timeout=2) as response:
            self.assertEqual(response.headers.get_content_type(), "text/css")

        # Linked from the top bar and from the meditation and the series form.
        self.assertEqual(body.count('href="/guide.html"'), 3)
        self.assertIn('data-lang="en"', guide)
        self.assertIn('data-lang="ru"', guide)
        # The guide quotes what the voice says, in both languages.
        for phrase in (
            "focus on the breath",
            "open awareness",
            "That's it, you can open your eyes",
            "Sound on",
            "фокус на дыхании",
            "открытое внимание",
            "Всё, можно открыть глаза",
            "Звук включён",
        ):
            self.assertIn(phrase.lower(), script.lower())
            self.assertIn(phrase.lower(), guide.lower())

    def test_meditation_guide_pictures_are_served(self):
        with urllib.request.urlopen(f"{self.base_url}/guide.html", timeout=2) as response:
            guide = response.read().decode("utf-8")
        # One picture per language, folded under the intro, and the file behind each.
        for language in ("en", "ru"):
            name = f"guide-why-{language}.webp"
            self.assertIn(f'<img src="/{name}"', guide)
            with urllib.request.urlopen(f"{self.base_url}/{name}", timeout=2) as response:
                self.assertEqual(response.headers.get_content_type(), "image/webp")
                self.assertEqual(response.read()[8:12], b"WEBP")

    def test_diagnostics_endpoint_reports_state_and_last_contact(self):
        self.post_json("/api/muse/connect")
        self.get_json("/api/muse/contact")

        diagnostics = self.get_json("/api/muse/diagnostics")

        self.assertEqual(diagnostics["service"], "muse-tmr-local-app")
        self.assertEqual(diagnostics["state"]["connection_state"], "connected")
        self.assertIsNone(diagnostics["source_diagnostics"])
        self.assertEqual(diagnostics["contact"]["channels"]["AF7"]["status"], "fair")
        self.assertIn("session", diagnostics["state"])
        self.assertFalse(diagnostics["state"]["session"]["running"])


class TestLocalMuseAppReadyGate(unittest.TestCase):
    def setUp(self):
        self.server = create_local_app_server(
            AppConfig(port=0, source="mock", mock_scenario="all_good", gate_stability_seconds=0.0)
        )
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.app_state.shutdown()
        self.server.server_close()

    def post_json(self, path):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_json(self, path):
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_start_when_ready_auto_starts_after_ready_gate(self):
        self.post_json("/api/muse/connect")
        armed = self.post_json("/api/muse/start-when-ready")
        ui_state = self.get_json("/api/muse/ui-state")

        self.assertEqual(armed["state"], "starting")
        self.assertTrue(armed["ready"])
        self.assertEqual(ui_state["contact"]["connection_state"], "connected")
        self.assertTrue(ui_state["contact"]["all_good"])
        self.assertEqual(ui_state["gate"]["state"], "running")
        self.assertTrue(ui_state["gate"]["ready"])
        self.assertTrue(ui_state["state"]["session"]["running"])
        self.assertIsNotNone(ui_state["state"]["session"]["started_at_seconds"])


class TestLocalMuseAppContactWarningLog(unittest.TestCase):
    def setUp(self):
        self.server = create_local_app_server(
            AppConfig(port=0, source="mock", mock_scenario="flapping_af7", gate_stability_seconds=0.0)
        )
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.app_state.shutdown()
        self.server.server_close()

    def post_json(self, path):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_json(self, path):
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_running_session_logs_contact_drop_and_recovery(self):
        self.post_json("/api/muse/connect")
        self.post_json("/api/muse/start-when-ready")
        dropped = self.get_json("/api/muse/ui-state")
        recovered = self.get_json("/api/muse/ui-state")

        dropped_session = dropped["state"]["session"]
        self.assertEqual(dropped["contact"]["channels"]["AF7"]["status"], "poor")
        self.assertFalse(dropped["contact"]["all_good"])
        self.assertEqual(dropped["gate"]["state"], "running")
        self.assertTrue(dropped_session["running"])
        self.assertEqual(dropped_session["contact_warning_count"], 1)
        self.assertIsNotNone(dropped_session["active_contact_warning"])
        self.assertIn("AF7 poor", dropped_session["active_contact_warning"]["channels"])

        session = recovered["state"]["session"]
        self.assertTrue(recovered["contact"]["all_good"])
        self.assertTrue(session["running"])
        self.assertEqual(session["contact_warning_count"], 1)
        self.assertIsNone(session["active_contact_warning"])
        self.assertEqual(
            [event["kind"] for event in session["contact_warning_events"]],
            ["contact_drop", "contact_recovered"],
        )
        self.assertIn("AF7 poor", session["contact_warning_events"][0]["channels"])
        self.assertIn("low_coverage", session["contact_warning_events"][0]["reason_codes"])


class TestLocalMuseAppAmusedScan(unittest.TestCase):
    def test_live_app_uses_effective_amused_contact_sample_rate(self):
        server = create_local_app_server(AppConfig(port=0, source="amused"))
        try:
            self.assertEqual(server.app_state._contact_monitor.config.sample_rate_hz, 128.0)
        finally:
            server.app_state.shutdown()
            server.server_close()

    def test_empty_live_scan_returns_disconnected_error_without_sticking_scanning(self):
        with patch(
            "muse_tmr.sources.amused_source.AmusedSource.discover",
            new_callable=AsyncMock,
        ) as discover:
            discover.return_value = []
            server = create_local_app_server(AppConfig(port=0, source="amused"))
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            host, port = server.server_address
            base_url = f"http://{host}:{port}"
            try:
                request = urllib.request.Request(
                    f"{base_url}/api/muse/scan",
                    data=b"",
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    scanned = json.loads(response.read().decode("utf-8"))

                self.assertEqual(scanned["connection_state"], "disconnected")
                self.assertEqual(scanned["devices"], [])
                self.assertEqual(scanned["error_message"], "No Muse devices found")
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.app_state.shutdown()
                server.server_close()


class BatteryFakeAmusedSource(LoopSafeFakeAmusedSource):
    """Connects by address like amused (name == address) and streams battery frames."""

    async def discover(self):
        from muse_tmr.sources.base_source import MuseDeviceInfo

        return [
            MuseDeviceInfo(name="Muse-OTHER", address="other-address", rssi=-80),
            MuseDeviceInfo(name="MuseS-1234", address="test-address", rssi=-55),
        ]

    async def connect(self, device=None):
        self.connect_calls += 1
        return MuseSourceMetadata(
            source_name="amused", device_name="test-address", device_id="test-address", capabilities={"eeg": True}
        )

    async def stream(self):
        from muse_tmr.data.sample_types import BatterySample, MuseFrame

        self.stream_calls += 1
        percent = 81.5
        while not self.stop_requested:
            yield MuseFrame(timestamp=time.time(), battery=BatterySample(timestamp=time.time(), percent=percent), source="amused")
            await asyncio.sleep(0.01)


class TestLocalMuseAppBatteryAndScan(unittest.TestCase):
    def make(self, address):
        BatteryFakeAmusedSource.instances = []
        patcher = patch("muse_tmr.sources.amused_source.AmusedSource", BatteryFakeAmusedSource)
        patcher.start()
        self.addCleanup(patcher.stop)
        server = create_local_app_server(AppConfig(port=0, source="amused", address=address))
        self.addCleanup(server.server_close)
        self.addCleanup(server.app_state.shutdown)
        return server.app_state

    def test_scan_reports_configured_headband_and_sorts_by_signal(self):
        state = self.make("test-address")
        scanned = state.scan()
        self.assertEqual([device["name"] for device in scanned["devices"]], ["MuseS-1234", "Muse-OTHER"])
        self.assertTrue(scanned["scan"]["configured_found"])
        self.assertEqual(scanned["scan"]["count"], 2)

    def test_failed_scan_replaces_the_previous_result(self):
        state = self.make("test-address")
        self.assertTrue(state.scan()["scan"]["configured_found"])
        with patch.object(BatteryFakeAmusedSource, "discover", side_effect=RuntimeError("Bluetooth is off")):
            failed = state.scan()
        self.assertEqual(failed["devices"], [])
        self.assertTrue(failed["scan"]["failed"])
        self.assertIn("Bluetooth is off", failed["scan"]["error"])

    def test_scan_says_when_the_configured_headband_is_missing(self):
        state = self.make("not-around")
        self.assertFalse(state.scan()["scan"]["configured_found"])

    def test_battery_while_connected_and_name_instead_of_address(self):
        state = self.make("test-address")
        self.assertIsNone(state.state()["battery_percent"])
        state.scan()
        state.connect()
        deadline = time.time() + 2
        while state.state()["battery_percent"] is None and time.time() < deadline:
            time.sleep(0.02)
        connected = state.state()
        self.assertEqual(connected["battery_percent"], 81.5)
        self.assertEqual(connected["device"]["name"], "MuseS-1234")
        state.disconnect()
        self.assertIsNone(state.state()["battery_percent"])

    def test_name_falls_back_to_muse_without_a_scan(self):
        state = self.make("test-address")
        self.assertEqual(state.connect()["device"]["name"], "Muse")


class TestLocalMuseAppAmusedConnect(unittest.TestCase):
    def test_repeated_live_connect_is_idempotent(self):
        LoopSafeFakeAmusedSource.instances = []
        with patch("muse_tmr.sources.amused_source.AmusedSource", LoopSafeFakeAmusedSource):
            server = create_local_app_server(
                AppConfig(
                    port=0,
                    source="amused",
                    address="2C48FFC8-A1C5-BDFD-A5A4-EEA280A7BBA6",
                )
            )
            try:
                first = server.app_state.connect()
                second = server.app_state.connect()
                time.sleep(0.05)

                self.assertEqual(first["connection_state"], "connected")
                self.assertEqual(second["connection_state"], "connected")
                self.assertEqual(len(LoopSafeFakeAmusedSource.instances), 1)
                source = LoopSafeFakeAmusedSource.instances[0]
                self.assertEqual(source.connect_calls, 1)
                self.assertEqual(source.stream_calls, 1)
                self.assertEqual(server.app_state.state()["connection_state"], "connected")
            finally:
                server.app_state.shutdown()
                server.server_close()


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class LauncherSpy:
    def __init__(self, proc=None):
        self.calls = []
        self._proc = proc if proc is not None else _FakeProc()

    def __call__(self, command, log_path):
        self.calls.append((list(command), Path(log_path)))
        return self._proc


class TerminatorSpy:
    def __init__(self):
        self.signals = []

    def __call__(self, pid, sig):
        self.signals.append((pid, sig))


_FIXED_NOW = dt.datetime(2026, 7, 7, 1, 0, 0)


class TestLocalMuseAppRecording(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.recordings_base = Path(self._tmp.name)
        self.launcher = LauncherSpy()
        self.terminator = TerminatorSpy()

    def tearDown(self):
        self._tmp.cleanup()

    def _make_state(self, source="amused", proc=None):
        server = create_local_app_server(
            AppConfig(port=0, source=source),
            launcher=LauncherSpy(proc) if proc is not None else self.launcher,
            terminator=self.terminator,
            recordings_base=self.recordings_base,
            now_fn=lambda: _FIXED_NOW,
        )
        self.addCleanup(server.server_close)
        self.addCleanup(server.app_state.shutdown)
        return server.app_state

    def test_record_requires_amused_source(self):
        state = self._make_state(source="mock")
        payload, status = state.start_recording("night")
        self.assertEqual(int(status), 409)
        self.assertIn("amused", payload["error"])
        self.assertEqual(self.launcher.calls, [])

    def test_record_rejects_unknown_kind(self):
        state = self._make_state()
        payload, status = state.start_recording("bogus")
        self.assertEqual(int(status), 400)
        self.assertEqual(self.launcher.calls, [])

    @unittest.skipIf(ON_WINDOWS, POSIX_SHELL_LINE)
    def test_record_night_builds_expected_command_and_folder(self):
        state = self._make_state()
        payload, status = state.start_recording("night")

        self.assertEqual(int(status), 200)
        self.assertEqual(len(self.launcher.calls), 1)
        command, _log = self.launcher.calls[0]
        expected_dir = (self.recordings_base / "night" / "20260707_010000").resolve()
        self.assertEqual(
            command,
            [
                CAFFEINATE,
                "-s",
                sys.executable,
                "-m",
                "muse_tmr.cli.main",
                "record",
                "--source",
                "amused",
                "--preset",
                "p21",
                "--duration-hours",
                "8",
                "--no-data-timeout-seconds",
                "45",
                "--max-reconnect-attempts",
                "1000",
                "--output-dir",
                str(expected_dir),
                "--quiet",
            ],
        )
        self.assertNotIn("--allow-short", command)
        self.assertTrue(payload["report_command"].startswith("cd "))
        self.assertIn(
            f"&& {sys.executable} scripts/generate_nightly_report.py {expected_dir}",
            payload["report_command"],
        )
        self.assertEqual(payload["kind"], "night")
        self.assertEqual(payload["preset"], "p21")
        self.assertTrue(payload["active"])
        self.assertTrue((expected_dir / "launch.json").exists())

    @unittest.skipIf(ON_WINDOWS, POSIX_SHELL_LINE)
    def test_report_command_uses_project_venv_outside_a_venv(self):
        project_root = self.recordings_base / "project"
        venv_python = project_root / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("", encoding="utf-8")
        state = self._make_state()

        # The macOS Python.app launch: base interpreter, venv only on PYTHONPATH.
        with patch.object(sys, "prefix", sys.base_prefix), patch(
            "muse_tmr.cli.main._find_project_root", return_value=project_root
        ):
            payload, _status = state.start_recording("night")

        self.assertTrue(
            payload["report_command"].startswith(
                f"cd {project_root} && {venv_python} scripts/generate_nightly_report.py "
            )
        )

    def test_record_session_uses_p1034_and_allow_short(self):
        state = self._make_state()
        payload, status = state.start_recording("session")

        self.assertEqual(int(status), 200)
        command, _log = self.launcher.calls[0]
        expected_dir = (self.recordings_base / "session" / "20260707_010000").resolve()
        self.assertIn("--preset", command)
        self.assertEqual(command[command.index("--preset") + 1], "p1034")
        self.assertEqual(command[command.index("--duration-hours") + 1], "1")
        self.assertEqual(command[-1], "--allow-short")
        self.assertEqual(command[command.index("--output-dir") + 1], str(expected_dir))
        self.assertEqual(payload["preset"], "p1034")

    def test_double_start_returns_conflict(self):
        state = self._make_state()
        first, first_status = state.start_recording("night")
        self.assertEqual(int(first_status), 200)
        payload, status = state.start_recording("session")
        self.assertEqual(int(status), 409)
        self.assertIn("already running", payload["error"])
        self.assertEqual(len(self.launcher.calls), 1)

    def test_stop_signals_sigint_to_group(self):
        state = self._make_state()
        state.start_recording("night")
        payload, status = state.stop_recording()

        self.assertEqual(int(status), 200)
        self.assertEqual(self.terminator.signals, [(4242, signal.SIGINT)])
        self.assertEqual(payload["state"], "stopping")

    def test_running_recording_blocks_auto_update_restart(self):
        state = self._make_state()
        self.assertTrue(state.idle_for_update())
        state.start_recording("night")
        self.assertFalse(state.idle_for_update())

    def test_record_with_polar_passes_flag_and_reports_h10_status(self):
        import base64

        state = self._make_state()
        payload, status = state.start_recording("session", with_polar=True)
        self.assertEqual(int(status), 200)
        command, _log = self.launcher.calls[0]
        self.assertEqual(command[-1], "--with-polar")
        self.assertTrue(payload["with_polar"])
        self.assertEqual(payload["polar"]["state"], "starting")
        output_dir = Path(payload["output_dir"])
        launch = json.loads((output_dir / "launch.json").read_text())
        self.assertTrue(launch["with_polar"])

        polar = output_dir / "polar"
        polar.mkdir(parents=True)
        (polar / "events.jsonl").write_text(
            json.dumps({"event": "recording_started"}) + "\n" + json.dumps({"event": "connected"}) + "\n"
        )
        hr = bytes([0x16, 72, 0x00, 0x04])  # contact supported + detected, RR present
        (polar / "raw_notifications.jsonl").write_text(
            json.dumps({"char": "pmd_data", "b64": "AA=="}) + "\n"
            + json.dumps({"char": "hr", "b64": base64.b64encode(hr).decode()}) + "\n"
        )
        status_now = state.ui_state()["recording"]["polar"]
        self.assertEqual((status_now["state"], status_now["heart_rate_bpm"], status_now["contact"]), ("connected", 72, True))

        (polar / "events.jsonl").write_text(json.dumps({"event": "disconnected"}) + "\n")
        self.assertEqual(state.ui_state()["recording"]["polar"]["state"], "reconnecting")

        (output_dir / "events.jsonl").write_text(json.dumps({"event": "companion_exited"}) + "\n")
        self.assertEqual(state.ui_state()["recording"]["polar"]["state"], "failed")

        (polar / "summary.json").write_text(json.dumps({"stop_reason": "user_stopped"}))
        self.assertEqual(state.ui_state()["recording"]["polar"]["state"], "stopped")

    def test_recording_stays_active_while_the_recorder_waits_for_its_polar_child(self):
        proc = _FakeProc()
        state = self._make_state(proc=proc)
        payload, _ = state.start_recording("session", with_polar=True)
        output_dir = Path(payload["output_dir"])
        (output_dir / "summary.json").write_text(json.dumps({"stop_reason": "user_stopped"}))

        finishing = state.ui_state()["recording"]
        self.assertEqual(finishing["state"], "finishing")
        self.assertTrue(finishing["active"])
        _payload, status = state.start_recording("session")
        self.assertEqual(int(status), 409)  # the old child may still hold the H10

        proc.returncode = 0
        done = state.ui_state()["recording"]
        self.assertEqual(done["state"], "completed")
        self.assertFalse(done["active"])

    def test_recording_without_polar_has_no_status_or_flag(self):
        state = self._make_state()
        payload, _status = state.start_recording("night")
        self.assertNotIn("--with-polar", self.launcher.calls[0][0])
        self.assertIsNone(payload["polar"])

    @unittest.skipIf(ON_WINDOWS, POSIX_PROCESS_CONTROL)
    def test_stop_waits_longer_before_sigkill_when_polar_is_on(self):
        for with_polar, expect_kill in ((False, True), (True, False)):
            terminator = TerminatorSpy()
            server = create_local_app_server(
                AppConfig(port=0, source="amused"),
                launcher=LauncherSpy(),
                terminator=terminator,
                recordings_base=self.recordings_base / str(with_polar),
                now_fn=lambda: _FIXED_NOW,
            )
            self.addCleanup(server.server_close)
            self.addCleanup(server.app_state.shutdown)
            state = server.app_state
            state.start_recording("session", with_polar=with_polar)
            state.stop_recording()
            state._recording.stop_signalled_at_seconds -= 10.0  # 10 s after Stop
            state.ui_state()
            killed = (4242, signal.SIGKILL) in terminator.signals
            self.assertEqual(killed, expect_kill, f"with_polar={with_polar}")

    def test_stop_without_recording_returns_conflict(self):
        state = self._make_state()
        payload, status = state.stop_recording()
        self.assertEqual(int(status), 409)
        self.assertEqual(self.terminator.signals, [])

    def test_ui_state_reports_progress_from_progress_json(self):
        state = self._make_state()
        state.start_recording("night")
        output_dir = (self.recordings_base / "night" / "20260707_010000").resolve()
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "elapsed_seconds": 120.0,
                    "frame_count": 999,
                    "battery_percent": 87.0,
                    "reconnect_attempts": 2,
                    "last_event": "reconnect_scheduled",
                }
            ),
            encoding="utf-8",
        )
        recording = state.ui_state()["recording"]
        self.assertTrue(recording["active"])
        self.assertEqual(recording["frame_count"], 999)
        self.assertEqual(recording["battery_percent"], 87.0)
        self.assertEqual(recording["reconnect_attempts"], 2)
        self.assertEqual(recording["elapsed_seconds"], 120.0)
        self.assertEqual(recording["last_event"], "reconnect_scheduled")

    def _write_progress_with_contact(self, output_dir, updated_at):
        channels = {
            channel: {
                "channel": channel,
                "status": "good",
                "fill": 1.0,
                "coverage": 1.0,
                "sample_count": 256,
                "reason_codes": [],
            }
            for channel in ("TP9", "AF7", "AF8", "TP10")
        }
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "updated_at": updated_at.isoformat(),
                    "elapsed_seconds": 30.0,
                    "frame_count": 500,
                    "contact": {
                        "source": "amused",
                        "connection_state": "connected",
                        "sequence": 500,
                        "timestamp_seconds": updated_at.timestamp(),
                        "stale": False,
                        "channels": channels,
                    },
                    "source_diagnostics": {
                        "last_packet_age_seconds": 0.5,
                        "decoder": {"eeg_rolling_sample_rate_hz": 128.0},
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_ui_state_shows_recorder_contact_while_recording(self):
        state = self._make_state()
        state.start_recording("session")
        output_dir = (self.recordings_base / "session" / "20260707_010000").resolve()
        self._write_progress_with_contact(output_dir, dt.datetime.now(dt.timezone.utc))

        payload = state.ui_state()

        self.assertEqual(payload["state"]["connection_state"], "disconnected")
        self.assertTrue(payload["contact"]["all_good"])
        self.assertEqual(payload["contact"]["channels"]["AF7"]["status"], "good")
        diagnostics = payload["source_diagnostics"]
        self.assertEqual(diagnostics["decoder"]["eeg_rolling_sample_rate_hz"], 128.0)
        self.assertGreaterEqual(diagnostics["last_packet_age_seconds"], 0.5)
        self.assertLess(diagnostics["last_packet_age_seconds"], 5.0)
        self.assertNotIn("contact", payload["recording"])

    def test_ui_state_marks_recorder_contact_stale_when_heartbeat_stops(self):
        state = self._make_state()
        state.start_recording("session")
        output_dir = (self.recordings_base / "session" / "20260707_010000").resolve()
        updated_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=60)
        self._write_progress_with_contact(output_dir, updated_at)

        payload = state.ui_state()

        contact = payload["contact"]
        self.assertTrue(contact["stale"])
        self.assertFalse(contact["all_good"])
        self.assertEqual(contact["channels"], {})
        self.assertGreater(payload["source_diagnostics"]["last_packet_age_seconds"], 60.0)

    def test_ui_state_falls_back_without_recorder_contact(self):
        state = self._make_state()
        state.start_recording("session")
        output_dir = (self.recordings_base / "session" / "20260707_010000").resolve()
        (output_dir / "progress.json").write_text(
            json.dumps({"elapsed_seconds": 5.0, "frame_count": 10}), encoding="utf-8"
        )

        payload = state.ui_state()

        self.assertEqual(payload["contact"]["connection_state"], "disconnected")
        self.assertFalse(payload["contact"]["all_good"])
        self.assertIsNone(payload["source_diagnostics"])

    def test_ui_state_marks_completed_when_summary_present(self):
        proc = _FakeProc()
        state = self._make_state(proc=proc)
        state.start_recording("night")
        output_dir = (self.recordings_base / "night" / "20260707_010000").resolve()
        (output_dir / "summary.json").write_text(
            json.dumps({"stop_reason": "duration_complete", "reconnect_attempts": 3}),
            encoding="utf-8",
        )
        proc.returncode = 0

        recording = state.ui_state()["recording"]
        self.assertEqual(recording["state"], "completed")
        self.assertFalse(recording["active"])
        self.assertTrue(recording["summary_available"])

        # A finished recording must not block a new one.
        _payload, status = state.start_recording("session")
        self.assertEqual(int(status), 200)

    def test_start_recording_releases_ble_before_spawn(self):
        LoopSafeFakeAmusedSource.instances = []
        with patch("muse_tmr.sources.amused_source.AmusedSource", LoopSafeFakeAmusedSource):
            state = self._make_state()
            connected = state.connect()
            self.assertEqual(connected["connection_state"], "connected")

            payload, status = state.start_recording("night")
            self.assertEqual(int(status), 200)

            source = LoopSafeFakeAmusedSource.instances[0]
            self.assertTrue(source.stop_requested)
            self.assertEqual(state.state()["connection_state"], "disconnected")
            self.assertEqual(len(self.launcher.calls), 1)


class TestLocalMuseAppReport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.reports = root / "reports"
        self.procs = []

        def launcher(command, log_path):
            proc = _FakeProc(pid=5000 + len(self.procs))
            self.procs.append((list(command), proc))
            return proc

        self.server = create_local_app_server(
            AppConfig(port=0, source="amused"),
            launcher=launcher,
            terminator=TerminatorSpy(),
            recordings_base=root / "recordings",
            reports_base=self.reports,
            now_fn=lambda: _FIXED_NOW,
        )
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.app_state.shutdown)
        self.state = self.server.app_state

    def finish_recording(self):
        payload, _ = self.state.start_recording("session")
        output_dir = Path(payload["output_dir"])
        (output_dir / "summary.json").write_text(json.dumps({"stop_reason": "duration_complete"}))
        self.procs[0][1].returncode = 0
        return output_dir

    def test_no_report_while_recording(self):
        self.state.start_recording("session")
        _payload, status = self.state.build_report()
        self.assertEqual(int(status), 409)

    def test_build_runs_the_script_and_reports_progress(self):
        output_dir = self.finish_recording()
        self.assertEqual(self.state.ui_state()["recording"]["report"]["state"], "none")

        payload, status = self.state.build_report()
        self.assertEqual(int(status), 200)
        command, report_proc = self.procs[1]
        report_file = (self.reports / "session" / f"{output_dir.name}.html").resolve()
        self.assertEqual(command[0], sys.executable)
        self.assertTrue(Path(command[1]).as_posix().endswith("scripts/generate_nightly_report.py"))
        self.assertEqual(command[2:], [str(output_dir.resolve()), "--output", str(report_file)])
        self.assertEqual(payload["report"]["state"], "running")

        # A second click while it runs does not start another builder.
        self.state.build_report()
        self.assertEqual(len(self.procs), 2)

        report_file.write_text("<html>report</html>")
        report_proc.returncode = 0
        ready = self.state.ui_state()["recording"]["report"]
        self.assertEqual(ready["state"], "ready")
        self.assertEqual(ready["url"], f"/reports/session/{output_dir.name}.html")

    def test_report_build_blocks_auto_update_restart(self):
        self.finish_recording()
        self.assertTrue(self.state.idle_for_update())
        self.state.build_report()
        self.assertFalse(self.state.idle_for_update())
        self.procs[1][1].returncode = 0
        self.assertTrue(self.state.idle_for_update())

    def test_concurrent_build_requests_start_one_builder(self):
        import threading as threads

        self.finish_recording()
        gate = threads.Event()
        original = self.state._launcher

        def slow_launcher(command, log_path):
            gate.wait(2)
            return original(command, log_path)

        self.state._launcher = slow_launcher
        results = []
        workers = [threads.Thread(target=lambda: results.append(self.state.build_report())) for _ in range(2)]
        for worker in workers:
            worker.start()
        time.sleep(0.2)
        gate.set()
        for worker in workers:
            worker.join(5)
        self.assertEqual(len(self.procs), 2)  # the recorder plus exactly one report builder
        self.assertEqual(sorted(int(status) for _payload, status in results), [200, 200])

    def test_failed_build_is_reported(self):
        self.finish_recording()
        self.state.build_report()
        self.procs[1][1].returncode = 1
        failed = self.state.ui_state()["recording"]["report"]
        self.assertEqual(failed["state"], "failed")
        self.assertTrue(failed["log_path"].endswith("report.log"))

    def test_reports_are_served_and_nothing_else(self):
        (self.reports / "session").mkdir(parents=True)
        (self.reports / "session" / "a.html").write_text("<html>hello</html>")
        (self.reports / "session" / "a.json").write_text("{}")
        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.shutdown)
        host, port = self.server.server_address
        base = f"http://{host}:{port}"
        with urllib.request.urlopen(f"{base}/reports/session/a.html", timeout=2) as response:
            self.assertIn("hello", response.read().decode())
        for bad in ("/reports/session/a.json", "/reports/../../../etc/passwd", "/reports/%2e%2e/%2e%2e/etc/hosts"):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(f"{base}{bad}", timeout=2)
            self.assertEqual(error.exception.code, 404, bad)


class TestLocalMuseAppMeditation(TestLocalMuseAppReport):
    def start(self, **overrides):
        body = {"conditions": ["focus", "open"], "blocks": 4, "block_minutes": 0.5, "settle_seconds": 30, "seed": 4}
        body.update(overrides)
        return self.state.start_meditation(body)

    def test_bad_plans_are_rejected(self):
        for overrides in ({"conditions": ["focus"]}, {"blocks": 1}, {"blocks": 6}, {"block_minutes": 0}, {"settle_seconds": -1}):
            _payload, status = self.start(**overrides)
            self.assertEqual(int(status), 400, overrides)
        self.assertEqual(self.procs, [])

    def test_start_writes_the_plan_and_records_long_enough(self):
        payload, status = self.start(with_polar=True)
        self.assertEqual(int(status), 200)
        command, _proc = self.procs[0]
        self.assertNotIn("--duration-hours", command)
        self.assertEqual(command[command.index("--duration-seconds") + 1], "210")  # 30 + 4 x 30 + 60 slack
        self.assertIn("--with-polar", command)
        self.assertIn("--duration-from-first-frame", command)
        self.assertEqual(command[command.index("--preset") + 1], "p21")  # EEG only, no 64 Hz line
        output_dir = Path(payload["output_dir"])
        plan = json.loads((output_dir / "blocks.json").read_text())
        self.assertEqual([block["start_s"] for block in plan["blocks"]], [30.0, 60.0, 90.0, 120.0])
        self.assertEqual(payload["meditation"]["plan"]["conditions"], ["focus", "open"])
        self.assertIsNone(payload["meditation"]["first_frame_elapsed_seconds"])
        (output_dir / "progress.json").write_text(json.dumps({"elapsed_seconds": 12.0, "first_frame_elapsed_seconds": 4.5}))
        self.assertEqual(self.state.ui_state()["recording"]["meditation"]["first_frame_elapsed_seconds"], 4.5)

    def test_ratings_are_saved_into_blocks_json(self):
        payload, _ = self.start()
        output_dir = Path(payload["output_dir"])
        _payload, status = self.state.save_meditation_rating(
            {"block_index": 1, "depth": "7", "sensory_fading": 4, "drowsiness": "2"}
        )
        self.assertEqual(int(status), 200)
        plan = json.loads((output_dir / "blocks.json").read_text())
        block = plan["blocks"][1]
        self.assertEqual((block["depth"], block["sensory_fading"], block["drowsiness"]), (7.0, 4.0, 2.0))
        self.assertIsNone(plan["blocks"][0]["depth"])
        self.assertIsNone(plan["blocks"][0]["drowsiness"])
        for bad in (
            {"block_index": 1, "depth": 11},
            {"block_index": 1, "drowsiness": -1},
            {"block_index": 9, "depth": 3},
            {"depth": 3},
        ):
            _payload, status = self.state.save_meditation_rating(bad)
            self.assertEqual(int(status), 400, bad)

    def test_ratings_need_a_meditation_recording(self):
        self.state.start_recording("session")
        _payload, status = self.state.save_meditation_rating({"block_index": 0, "depth": 3})
        self.assertEqual(int(status), 409)

    def test_analyze_runs_analyze_meditation_and_links_the_report(self):
        payload, _ = self.start()
        output_dir = Path(payload["output_dir"])
        _payload, status = self.state.analyze_meditation()
        self.assertEqual(int(status), 409)  # still recording
        (output_dir / "summary.json").write_text(json.dumps({"stop_reason": "duration_complete"}))
        self.procs[0][1].returncode = 0

        payload, status = self.state.analyze_meditation()
        self.assertEqual(int(status), 200)
        command, analysis = self.procs[1]
        report_dir = (self.reports / "meditation" / output_dir.name).resolve()
        self.assertEqual(command[1:4], ["-m", "muse_tmr.cli.main", "analyze-meditation"])
        self.assertEqual(command[-4:], ["--blocks", str((output_dir / "blocks.json").resolve()), "--output-dir", str(report_dir)])
        self.assertEqual(payload["meditation"]["analysis"]["state"], "running")
        self.assertFalse(self.state.idle_for_update())

        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.html").write_text("<html>ok</html>")
        analysis.returncode = 0
        ready = self.state.ui_state()["recording"]["meditation"]["analysis"]
        self.assertEqual(ready["state"], "ready")
        self.assertEqual(ready["url"], f"/reports/meditation/{output_dir.name}/report.html")

    def test_plain_recording_has_no_meditation_section(self):
        payload, _ = self.state.start_recording("session")
        self.assertIsNone(payload["meditation"])


class TestLocalMuseAppMeditationSeries(TestLocalMuseAppReport):
    FIRST = {"conditions": ["focus", "open"], "blocks": 4, "block_minutes": 0.5, "settle_seconds": 30, "seed": 4}

    def setUp(self):
        super().setUp()
        # One minute per call, so every session gets its own folder.
        minutes = iter(range(1000))
        self.state._now = lambda: _FIXED_NOW + dt.timedelta(minutes=next(minutes))
        self.series_path = self.state._recordings_base_resolved().parent / "protocol" / "meditation" / "series.json"

    def finish(self, payload, seconds, first_frame=5.0, downtime=0.0):
        output_dir = Path(payload["output_dir"])
        (output_dir / "progress.json").write_text(json.dumps({"first_frame_elapsed_seconds": first_frame}))
        (output_dir / "summary.json").write_text(
            json.dumps({"duration_seconds": (first_frame or 0.0) + seconds, "downtime_seconds": downtime})
        )
        self.procs[-1][1].returncode = 0
        return output_dir

    def test_first_session_starts_the_series_with_the_h10(self):
        payload, status = self.state.start_meditation_series(self.FIRST)
        self.assertEqual(int(status), 200)
        command, _proc = self.procs[0]
        self.assertIn("--with-polar", command)  # whatever the checkbox says
        self.assertEqual(command[command.index("--preset") + 1], "p21")
        series = json.loads(self.series_path.read_text())
        self.assertEqual(
            (series["conditions"], series["blocks"], series["block_minutes"], series["settle_seconds"]),
            (["focus", "open"], 4, 0.5, 30.0),
        )
        output_dir = Path(payload["output_dir"])
        self.assertEqual(json.loads((output_dir / "blocks.json").read_text())["series"], series["id"])
        # A rating rewrites blocks.json and must keep the tag.
        self.state.save_meditation_rating({"block_index": 0, "depth": 5})
        self.assertEqual(json.loads((output_dir / "blocks.json").read_text())["series"], series["id"])
        status = self.state.meditation_series()
        self.assertEqual((status["counted"], status["target"]), (0, 8))
        self.assertEqual([session["state"] for session in status["sessions"]], ["recording"])

    def test_later_sessions_keep_the_layout_and_take_a_new_settle(self):
        self.finish(self.state.start_meditation_series(self.FIRST)[0], 150)
        payload, status = self.state.start_meditation_series(
            {"conditions": ["breath", "body"], "blocks": 8, "block_minutes": 3, "settle_seconds": 90}
        )
        self.assertEqual(int(status), 200)
        plan = payload["meditation"]["plan"]
        self.assertEqual((plan["conditions"], len(plan["blocks"]), plan["settle_seconds"]), (["focus", "open"], 4, 90.0))
        self.assertEqual(plan["blocks"][0]["end_s"] - plan["blocks"][0]["start_s"], 30.0)
        self.assertEqual(json.loads(self.series_path.read_text())["settle_seconds"], 90.0)
        self.finish(payload, 210)
        # Without a settle the series keeps the last one.
        payload, _status = self.state.start_meditation_series({})
        self.assertEqual(payload["meditation"]["plan"]["settle_seconds"], 90.0)

    def test_a_session_counts_once_it_ran_past_its_last_block(self):
        # The plan's last block ends at 30 + 4 x 30 = 150 s from the first frame.
        self.finish(self.state.start_meditation_series(self.FIRST)[0], 150 + 60)  # ran to the end
        self.finish(self.state.start_meditation_series({})[0], 150 - 3)  # stopped right at "All blocks done"
        self.finish(self.state.start_meditation_series({})[0], 210, downtime=30)  # a short dropout
        self.finish(self.state.start_meditation_series({})[0], 100)  # stopped in block 3
        self.finish(self.state.start_meditation_series({})[0], 210, downtime=120)  # reconnecting for 2 of 2.5 min
        self.finish(self.state.start_meditation_series({})[0], 600, first_frame=None)  # the headband never sent a frame
        self.state.start_meditation_series({})
        self.procs[-1][1].returncode = 1  # the recorder died, no summary

        status = self.state.meditation_series()
        self.assertEqual(status["counted"], 3)
        sessions = status["sessions"]  # oldest first
        self.assertEqual(
            [session["state"] for session in sessions],
            ["counted", "counted", "counted", "short", "short", "no_data", "unfinished"],
        )
        self.assertEqual((sessions[3]["covered_seconds"], sessions[3]["needed_seconds"]), (100.0, 150.0))
        self.assertEqual(sessions[4]["covered_seconds"], 90.0)

    def test_other_meditations_do_not_count(self):
        self.finish(self.state.start_meditation_series(self.FIRST)[0], 210)
        self.finish(self.state.start_meditation(dict(self.FIRST, with_polar=True))[0], 210)
        status = self.state.meditation_series()
        self.assertEqual((status["counted"], len(status["sessions"])), (1, 1))

    def test_first_session_without_a_layout_takes_the_form_defaults(self):
        _payload, status = self.state.start_meditation_series({"conditions": ["focus", "open"]})
        self.assertEqual(int(status), 200)
        series = json.loads(self.series_path.read_text())
        self.assertEqual((series["blocks"], series["block_minutes"], series["settle_seconds"]), (4, 8.0, 60.0))

    def test_bad_first_settings_start_nothing(self):
        _payload, status = self.state.start_meditation_series(dict(self.FIRST, blocks=6))
        self.assertEqual(int(status), 400)
        self.assertEqual(self.procs, [])
        self.assertFalse(self.series_path.exists())
        self.assertIsNone(self.state.meditation_series()["series"])

    def test_series_endpoints(self):
        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.shutdown)
        host, port = self.server.server_address

        def call(path="", body=None):
            request = urllib.request.Request(
                f"http://{host}:{port}/api/meditation/series{path}",
                data=None if body is None else json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.loads(response.read())

        self.assertEqual(call(), {"counted": 0, "series": None, "sessions": [], "target": 8})
        # A start from the page needs the contact gate armed and good for its window.
        clock = [0.0]
        self.state._contact_gate.time_fn = lambda: clock[0]
        self.state._contact_provider = MockContactProvider.for_scenario("all_good")
        self.state._connection_state = "connected"
        self.state.arm_recording_gate()
        for second in range(1, 6):
            clock[0] = float(second)
            self.state.gate()
        started = call("/start", self.FIRST)
        self.assertEqual(started["meditation"]["plan"]["series"], call()["series"]["id"])
        self.assertEqual([session["state"] for session in call()["sessions"]], ["recording"])
        self.assertIsNone(call("/new", {})["series"])

    def test_new_series_sets_the_old_one_aside(self):
        old_dir = self.finish(self.state.start_meditation_series(self.FIRST)[0], 210)
        old_id = json.loads(self.series_path.read_text())["id"]

        status = self.state.new_meditation_series()
        self.assertEqual((status["series"], status["counted"], status["sessions"]), (None, 0, []))
        self.assertTrue((self.series_path.parent / f"series_{old_id}.json").is_file())
        self.assertEqual(json.loads((old_dir / "blocks.json").read_text())["series"], old_id)

        payload, _status = self.state.start_meditation_series(dict(self.FIRST, conditions=["breath", "body"]))
        series = json.loads(self.series_path.read_text())
        self.assertNotEqual(series["id"], old_id)
        self.assertEqual(payload["meditation"]["plan"]["conditions"], ["breath", "body"])
        self.assertEqual([session["name"] for session in self.state.meditation_series()["sessions"]], [Path(payload["output_dir"]).name])


class TestLocalMuseAppRecentRecordings(TestLocalMuseAppReport):
    def make_recording(self, kind, name, summary=None, blocks=False, polar=False, progress=None):
        folder = self.state._recordings_base_resolved() / kind / name
        folder.mkdir(parents=True)
        if summary is not None:
            (folder / "summary.json").write_text(json.dumps(summary))
        if progress is not None:
            (folder / "progress.json").write_text(json.dumps(progress))
        if blocks:
            (folder / "blocks.json").write_text("{}")
        if polar:
            (folder / "polar").mkdir()
        return folder

    def test_lists_recordings_newest_first_with_their_reports(self):
        self.make_recording("night", "20261007_230000", {"duration_seconds": 28800, "stop_reason": "duration_complete"})
        self.make_recording("session", "20261008_235854", {"duration_seconds": 791, "stop_reason": "user_stopped"}, blocks=True, polar=True)
        (self.reports / "night").mkdir(parents=True)
        (self.reports / "night" / "20261007_230000.html").write_text("<html></html>")

        recordings = self.state.list_recordings()["recordings"]
        self.assertEqual([item["name"] for item in recordings], ["20261008_235854", "20261007_230000"])
        newest, older = recordings
        self.assertEqual((newest["stop_reason"], newest["meditation"], newest["with_polar"]), ("user_stopped", True, True))
        self.assertEqual(newest["started_at"], "2026-10-08T23:58:54")
        self.assertEqual(newest["report"]["state"], "none")
        self.assertEqual(newest["meditation_report"]["state"], "none")
        self.assertEqual(older["report"]["url"], "/reports/night/20261007_230000.html")
        self.assertIsNone(older["meditation_report"])

    def test_lists_the_newest_first_and_says_how_many_there_are(self):
        for name in ("20261007_010000", "20261008_010000", "20261009_010000"):
            self.make_recording("night", name, {"duration_seconds": 60})

        listed = self.state.list_recordings(limit=2)
        self.assertEqual([item["name"] for item in listed["recordings"]], ["20261009_010000", "20261008_010000"])
        self.assertEqual(listed["total"], 3)

    def test_orders_by_the_start_in_the_name_wherever_it_is(self):
        self.make_recording("session", "20261010_010000", {})
        self.make_recording("session", "smoke_muse_h10_20261009_012712", {})
        unnamed = self.make_recording("session", "notes", {})
        os.utime(unnamed, (0, 0))  # changed long ago

        listed = self.state.list_recordings()["recordings"]
        self.assertEqual([item["name"] for item in listed], ["20261010_010000", "smoke_muse_h10_20261009_012712", "notes"])
        self.assertEqual(listed[1]["started_at"], "2026-10-09T01:27:12")
        self.assertIsNone(listed[2]["started_at"])

    def test_lists_the_eeg_numbers_from_the_last_heartbeat(self):
        decoder = {
            "eeg_effective_sample_rate_hz": 254.6,
            "lost_subpackets": {"EEG": 888, "ACCGYRO": 70},
            "repeated_subpackets": {"EEG": 43, "ACCGYRO": 22},
        }
        self.make_recording("night", "20261009_010000", {"duration_seconds": 1380}, progress={"source_diagnostics": {"decoder": decoder}})
        # Before the decoder counted subpackets, the heartbeat had only the rate.
        self.make_recording("night", "20261008_010000", {}, progress={"source_diagnostics": {"decoder": {"eeg_effective_sample_rate_hz": 253.8}}})
        self.make_recording("night", "20261007_010000", {})

        newest, older, oldest = self.state.list_recordings()["recordings"]
        self.assertEqual(newest["signal"], {"eeg_rate_hz": 254.6, "lost_subpackets": 958, "repeated_subpackets": 65})
        self.assertEqual(older["signal"], {"eeg_rate_hz": 253.8, "lost_subpackets": None, "repeated_subpackets": None})
        self.assertIsNone(oldest["signal"])

    def test_builds_for_an_older_recording_and_rejects_bad_input(self):
        folder = self.make_recording("session", "20261008_235854", {"stop_reason": "user_stopped"})
        entry, status = self.state.build_report_for("session", "20261008_235854")
        self.assertEqual(int(status), 200)
        self.assertEqual(entry["report"]["state"], "running")
        command, _proc = self.procs[0]
        self.assertEqual(command[2], str(folder.resolve()))
        self.assertFalse(self.state.idle_for_update())

        _entry, status = self.state.analyze_meditation_for("session", "20261008_235854")
        self.assertEqual(int(status), 409)  # no blocks.json
        for kind, name, expected in (
            ("bogus", "x", 400),
            ("session", "../night", 400),
            ("session", "..", 400),
            ("session", "20990101_000000", 404),
        ):
            _payload, status = self.state.build_report_for(kind, name)
            self.assertEqual(int(status), expected, (kind, name))

    @unittest.skipIf(ON_WINDOWS, POSIX_PROCESS_CONTROL)
    def test_detached_recorder_is_still_live_after_an_app_restart(self):
        import os
        import subprocess

        folder = self.make_recording("session", "20261009_010000")
        # A recorder this app instance does not know about, with the folder on its command line.
        ready = folder / "ready"
        recorder = subprocess.Popen(
            # Long like the real caffeinate + python recorder command, with the folder near the end.
            [sys.executable, "-c", "import pathlib, sys, time; pathlib.Path(sys.argv[-1]).touch(); time.sleep(30)",
             "--preset", "p1034", "--duration-seconds", "2400", "--no-data-timeout-seconds", "45",
             "--max-reconnect-attempts", "1000", "--output-dir", str(folder.resolve()), str(ready)]
        )
        self.addCleanup(recorder.wait)
        self.addCleanup(recorder.kill)
        deadline = time.time() + 10
        while not ready.exists() and time.time() < deadline:  # until exec has replaced the forked command line
            time.sleep(0.02)
        (folder / "launch.json").write_text(json.dumps({"pid": recorder.pid}))
        _payload, status = self.state.build_report_for("session", folder.name)
        self.assertEqual(int(status), 409)
        self.assertTrue(self.state.list_recordings()["recordings"][0]["live"])

        # Same PID number but not this recording (PID reuse) and no heartbeat: finished.
        (folder / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
        self.assertFalse(self.state.list_recordings()["recordings"][0]["live"])

    def test_cli_recording_is_live_while_its_heartbeat_moves(self):
        import os

        folder = self.make_recording("session", "20261009_020000")
        (folder / "progress.json").write_text("{}")
        self.assertTrue(self.state.list_recordings()["recordings"][0]["live"])
        old = time.time() - 600
        os.utime(folder / "progress.json", (old, old))
        self.assertFalse(self.state.list_recordings()["recordings"][0]["live"])
        (folder / "summary.json").write_text(json.dumps({"stop_reason": "duration_complete"}))
        self.assertFalse(self.state.list_recordings()["recordings"][0]["live"])

    def test_live_recording_cannot_be_reported_yet(self):
        payload, _ = self.state.start_recording("session")
        name = Path(payload["output_dir"]).name
        _payload, status = self.state.build_report_for("session", name)
        self.assertEqual(int(status), 409)
        entry = self.state.list_recordings()["recordings"][0]
        self.assertTrue(entry["live"])
        self.assertIsNone(entry["report"])


class TestLocalMuseAppCalibration(TestLocalMuseAppReport):
    def test_start_records_with_the_h10_and_launches_the_voice_guide(self):
        from muse_tmr.protocol.calibration import RECORD_SECONDS

        payload, status = self.state.start_calibration()
        self.assertEqual(int(status), 200)
        recorder, _proc = self.procs[0]
        self.assertIn("--with-polar", recorder)
        # No optics: on p1034 they put a 64 Hz line into the EEG.
        self.assertEqual(recorder[recorder.index("--preset") + 1], "p21")
        self.assertEqual(payload["preset"], "p21")
        self.assertIn("--duration-from-first-frame", recorder)
        self.assertEqual(recorder[recorder.index("--duration-seconds") + 1], f"{RECORD_SECONDS:g}")
        output_dir = Path(payload["output_dir"])
        guide, _guide_proc = self.procs[1]
        self.assertEqual(
            guide,
            [CAFFEINATE, "-i", sys.executable, "-m", "muse_tmr.cli.main", "calibration-guide",
             str(output_dir.resolve()), "--recorder-pid", "5000"],
        )
        self.assertTrue(payload["calibration"]["guide_running"])
        self.assertIsNone(payload["calibration"]["step"])
        self.assertIsNone(payload["meditation"])
        self.assertFalse(self.state.idle_for_update())

        (output_dir / "calibration" / "state.json").write_text(
            json.dumps({"state": "running", "label": "Jaw", "step": 3, "steps": 12, "start_s": 180, "end_s": 300})
        )
        self.assertEqual(self.state.ui_state()["recording"]["calibration"]["step"]["label"], "Jaw")
        self.assertTrue(json.loads((output_dir / "launch.json").read_text())["calibration"])

    def test_guide_speaks_the_language_picked_in_the_form(self):
        _payload, status = self.state.start_calibration({"language": "ru"})
        self.assertEqual(int(status), 200)
        guide, _proc = self.procs[1]
        self.assertEqual(guide[-2:], ["--language", "ru"])

    def test_unknown_language_starts_nothing(self):
        payload, status = self.state.start_calibration({"language": "de"})
        self.assertEqual(int(status), 400)
        self.assertIn("en, ru", payload["error"])
        self.assertEqual(self.procs, [])

    def test_guide_that_cannot_start_stops_the_recording(self):
        original = self.state._launcher

        def launcher(command, log_path):
            if "calibration-guide" in command:
                raise OSError("no python")
            return original(command, log_path)

        self.state._launcher = launcher
        _payload, status = self.state.start_calibration()
        self.assertEqual(int(status), 500)
        self.assertEqual(self.state._terminator.signals, [(5000, signal.SIGINT)])

    def test_report_for_a_finished_run(self):
        payload, _ = self.state.start_calibration()
        output_dir = Path(payload["output_dir"])
        _payload, status = self.state.build_calibration_report()
        self.assertEqual(int(status), 409)  # still recording
        (output_dir / "calibration" / "cues.jsonl").write_text('{"kind": "segment", "segment": "settle", "elapsed_s": 0.0}\n')
        (output_dir / "summary.json").write_text(json.dumps({"stop_reason": "duration_complete"}))
        for _command, proc in self.procs:
            proc.returncode = 0

        payload, status = self.state.build_calibration_report()
        self.assertEqual(int(status), 200)
        command, report_proc = self.procs[2]
        report_dir = (self.reports / "calibration" / output_dir.name).resolve()
        self.assertEqual(command[1:4], ["-m", "muse_tmr.cli.main", "calibration-report"])
        self.assertEqual(command[4:], [str(output_dir.resolve()), "--output-dir", str(report_dir)])
        self.assertEqual(payload["calibration"]["report"]["state"], "running")

        (report_dir / "report.html").write_text("<html>ok</html>")
        report_proc.returncode = 0
        entry = self.state.list_recordings()["recordings"][0]
        self.assertTrue(entry["calibration"])
        self.assertEqual(entry["calibration_report"]["url"], f"/reports/calibration/{output_dir.name}/report.html")

    def test_report_needs_a_calibration_run(self):
        self.finish_recording()
        name = self.state.list_recordings()["recordings"][0]["name"]
        _payload, status = self.state.calibration_report_for("session", name)
        self.assertEqual(int(status), 409)
        self.assertFalse(self.state.list_recordings()["recordings"][0]["calibration"])

    def test_protocol_steps_for_the_form(self):
        protocol = self.state.calibration_protocol()
        self.assertEqual(protocol["protocol_seconds"], 1380)
        self.assertEqual(protocol["steps"][2]["label"], "Jaw slightly tense, teeth lightly touching")


class TestLocalMuseAppRecordingEndpoint(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.launcher = LauncherSpy()
        self.server = create_local_app_server(
            AppConfig(port=0, source="mock"),
            launcher=self.launcher,
            recordings_base=Path(self._tmp.name),
            now_fn=lambda: _FIXED_NOW,
        )
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.app_state.shutdown()
        self.server.server_close()
        self._tmp.cleanup()

    def _post(self, path, body):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=2)

    def test_record_endpoint_conflicts_in_mock_mode(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self._post("/api/session/record", {"kind": "night"})
        self.assertEqual(raised.exception.code, 409)
        self.assertEqual(self.launcher.calls, [])


class TestLocalMuseAppRecordingGate(unittest.TestCase):
    """Every recording start from the page goes through the armed contact gate."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.launcher = LauncherSpy()
        self.server = create_local_app_server(
            AppConfig(port=0, source="amused"),
            launcher=self.launcher,
            recordings_base=Path(self._tmp.name),
            now_fn=lambda: _FIXED_NOW,
        )
        self.state = self.server.app_state
        self.now = 1000.0
        self.state._contact_gate.time_fn = lambda: self.now
        # Stand in for a connected headband whose contact the test sets.
        self.state._contact_provider = MockContactProvider.for_scenario("all_good")
        self.state._connection_state = "connected"
        self.contact("all_good")
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.app_state.shutdown()
        self.server.server_close()
        self._tmp.cleanup()

    def contact(self, scenario, index=0):
        self.state._last_contact_snapshot = builtin_contact_snapshots(scenario)[index]

    def post(self, path, body=None):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body or {}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def gate_at(self, seconds):
        self.now = 1000.0 + seconds
        with urllib.request.urlopen(f"{self.base_url}/api/muse/gate", timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))

    def watch(self, first, last):
        """Look at the gate every second, the way the page polls it."""
        for second in range(first, last + 1):
            gate = self.gate_at(second)
        return gate

    def record_at(self, seconds):
        self.now = 1000.0 + seconds
        return self.post("/api/session/record", {"kind": "night"})

    def test_a_start_without_arming_is_refused(self):
        status, payload = self.record_at(60)
        self.assertEqual(status, 409)
        self.assertFalse(payload["gate"]["armed"])
        self.assertIn("gate_disarmed", payload["gate"]["reason_codes"])
        self.assertEqual(self.launcher.calls, [])

    def test_good_contact_starts_after_the_stability_window(self):
        status, armed = self.post("/api/session/record/arm")
        self.assertEqual(status, 200)
        self.assertEqual(armed["state"], "ready_countdown")

        self.watch(1, 4)
        status, payload = self.record_at(4.9)
        self.assertEqual(status, 409)
        self.assertIn("stability_window_pending", payload["gate"]["reason_codes"])
        self.assertEqual(self.launcher.calls, [])

        status, payload = self.record_at(5.0)
        self.assertEqual(status, 200)
        self.assertEqual(payload["kind"], "night")
        self.assertEqual(len(self.launcher.calls), 1)
        # Starting hands the headband to the recorder, which disarms the gate.
        self.assertFalse(self.gate_at(6)["armed"])

    def test_a_poor_missing_or_stale_channel_blocks_the_start(self):
        for scenario, index in (("one_channel_poor", 0), ("all_missing", 0), ("stale_data", 1)):
            self.contact(scenario, index)
            self.post("/api/session/record/arm")
            status, payload = self.record_at(60)
            self.assertEqual(status, 409, scenario)
            self.assertEqual(payload["gate"]["state"], "armed_waiting_contact", scenario)
        self.assertEqual(self.launcher.calls, [])

    def test_a_contact_drop_restarts_the_countdown(self):
        self.post("/api/session/record/arm")
        self.watch(1, 2)
        self.contact("flapping_af7", 1)
        self.assertEqual(self.gate_at(3)["state"], "armed_waiting_contact")
        self.contact("flapping_af7", 2)
        self.assertEqual(self.gate_at(4)["state"], "ready_countdown")

        self.watch(5, 8)
        status, _payload = self.record_at(8.5)
        self.assertEqual(status, 409)
        status, _payload = self.record_at(9)
        self.assertEqual(status, 200)
        self.assertEqual(len(self.launcher.calls), 1)

    def test_a_pause_in_polling_does_not_count_as_good_contact(self):
        self.post("/api/session/record/arm")
        self.watch(1, 3)
        # The page stops polling (a background tab) while the contact drops and
        # comes back; the gate never sees the drop.
        status, payload = self.record_at(5.1)
        self.assertEqual(status, 409)
        self.assertEqual(payload["gate"]["stable_for_seconds"], 0.0)

        self.watch(6, 10)
        status, _payload = self.record_at(10.1)
        self.assertEqual(status, 200)
        self.assertEqual(len(self.launcher.calls), 1)

    def test_disconnect_and_cancel_drop_the_armed_start(self):
        self.post("/api/session/record/arm")
        self.assertEqual(self.watch(1, 5)["state"], "ready")
        self.post("/api/session/record/disarm")
        status, payload = self.record_at(6)
        self.assertEqual(status, 409)
        self.assertFalse(payload["gate"]["armed"])

        self.now = 1006.0
        self.post("/api/session/record/arm")
        self.assertEqual(self.watch(7, 11)["state"], "ready")
        self.state.disconnect()
        self.state._connection_state = "connected"
        status, payload = self.record_at(12)
        self.assertEqual(status, 409)
        self.assertFalse(payload["gate"]["armed"])
        self.assertEqual(self.launcher.calls, [])

    def test_meditation_series_and_calibration_starts_need_the_gate_too(self):
        meditation = {"conditions": ["focus", "open"], "blocks": 4, "block_minutes": 0.5, "settle_seconds": 30}
        for path, body in (
            ("/api/meditation/start", meditation),
            ("/api/meditation/series/start", meditation),
            ("/api/calibration/start", {"language": "en"}),
        ):
            status, payload = self.post(path, body)
            self.assertEqual(status, 409, path)
            self.assertIn("gate", payload)
        self.assertEqual(self.launcher.calls, [])

        self.post("/api/session/record/arm")
        self.watch(1, 5)
        status, payload = self.post("/api/meditation/start", meditation)
        self.assertEqual(status, 200)
        self.assertTrue(payload["meditation"]["plan"])
        self.assertEqual(len(self.launcher.calls), 1)


if __name__ == "__main__":
    unittest.main()
