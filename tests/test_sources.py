import asyncio
import datetime as dt
import subprocess
import threading
import unittest
from unittest.mock import patch

from muse_tmr.ble.muse_realtime_decoder import DecodedData
from muse_tmr.data.sample_types import EEGSample, MuseFrame
from muse_tmr.sources.amused_source import AmusedSource
from muse_tmr.sources.base_source import BaseMuseSource, MuseDeviceInfo, MuseSourceMetadata

# A recent time rather than one in the first day of 1970: on Windows,
# datetime.timestamp() of a naive time that early raises OSError.
DECODED_AT = dt.datetime.fromtimestamp(1_760_000_003.0)


class FakeMuseSource(BaseMuseSource):
    def __init__(self):
        self.connected = False
        self.stopped = False

    async def discover(self):
        return [MuseDeviceInfo(name="Muse Test", address="test-address", rssi=-40)]

    async def connect(self, device=None):
        self.connected = True
        return MuseSourceMetadata(
            source_name="fake",
            device_name="Muse Test",
            device_id="test-address",
            capabilities={"eeg": True},
        )

    async def stream(self):
        yield MuseFrame(
            timestamp=1.0,
            eeg=EEGSample(timestamp=1.0, channels_uv={"TP9": [0.1]}),
            source="fake",
            raw_packet=b"\x01",
        )

    async def stop(self):
        self.stopped = True


class FakeStreamClient:
    def __init__(self, **kwargs):
        from muse_tmr.ble.muse_realtime_decoder import MuseRealtimeDecoder

        self.decoder = MuseRealtimeDecoder()
        self.packet_callback = None

    def on_packet(self, callback):
        self.packet_callback = callback

    async def connect_and_stream(self, address, duration_seconds=30, preset="p1034"):
        raw = b"\x11\x22"
        if self.packet_callback:
            self.packet_callback(raw)
        decoded = DecodedData(
            timestamp=DECODED_AT,
            packet_type="EEG",
            eeg={"TP9": [0.25]},
            raw_bytes=raw,
        )
        for callback in self.decoder.callbacks["any"]:
            callback(decoded)
        await asyncio.sleep(0)
        return True


class TestSources(unittest.IsolatedAsyncioTestCase):
    async def test_fake_source_implements_contract(self):
        source = FakeMuseSource()

        devices = await source.discover()
        metadata = await source.connect(devices[0])
        frames = [frame async for frame in source.stream()]
        await source.stop()

        self.assertEqual(metadata.source_name, "fake")
        self.assertEqual(frames[0].modalities(), ("eeg",))
        self.assertTrue(source.stopped)

    async def test_amused_source_converts_callbacks_to_frames(self):
        source = AmusedSource(
            address="test-address",
            stream_client_factory=FakeStreamClient,
            duration_seconds=1,
            verbose=False,
        )

        metadata = await source.connect()
        frames = [frame async for frame in source.stream()]
        await source.stop()

        self.assertEqual(metadata.source_name, "amused")
        self.assertEqual(source.packet_count, 1)
        self.assertEqual(source.frame_count, 1)
        self.assertEqual(frames[0].eeg.channels_uv["TP9"], (0.25,))
        self.assertEqual(frames[0].raw_packet, b"\x11\x22")
        diagnostics = source.diagnostics()
        self.assertEqual(diagnostics["packet_count"], 1)
        self.assertEqual(diagnostics["frame_count"], 1)
        self.assertIn("decoder", diagnostics)

    async def test_amused_discover_reports_corebluetooth_child_crash(self):
        completed = subprocess.CompletedProcess(
            args=["python", "-c", "..."],
            returncode=137,
            stdout="",
            stderr="Fatal Python error: Aborted\n",
        )
        source = AmusedSource(verbose=False)

        with patch("muse_tmr.sources.amused_source.sys.platform", "darwin"), patch(
            "muse_tmr.sources.amused_source.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(RuntimeError, "Muse BLE discovery crashed"):
                await source.discover()

    async def test_amused_discover_filters_devices_by_name(self):
        devices = [
            MuseDeviceInfo(name="Muse S Athena", address="aa", rssi=-40),
            MuseDeviceInfo(name="Other Headset", address="bb", rssi=-50),
        ]
        source = AmusedSource(verbose=False)

        async def fake_discover():
            return devices

        with patch(
            "muse_tmr.sources.amused_source._discover_muse_devices",
            side_effect=fake_discover,
        ):
            found = await source.discover()

        self.assertEqual([device.address for device in found], ["aa"])

    async def test_amused_connect_uses_first_discovered_device_without_address(self):
        device = MuseDeviceInfo(name="Muse S Athena", address="aa", rssi=-40)
        source = AmusedSource(
            stream_client_factory=FakeStreamClient,
            verbose=False,
        )

        async def fake_discover():
            return [device]

        with patch(
            "muse_tmr.sources.amused_source._discover_muse_devices",
            side_effect=fake_discover,
        ):
            metadata = await source.connect()

        self.assertEqual(source.address, "aa")
        self.assertEqual(metadata.device_name, "Muse S Athena")

    async def test_amused_connect_fails_when_no_devices_found(self):
        source = AmusedSource(
            stream_client_factory=FakeStreamClient,
            verbose=False,
        )

        async def fake_discover():
            return []

        with patch(
            "muse_tmr.sources.amused_source._discover_muse_devices",
            side_effect=fake_discover,
        ):
            with self.assertRaisesRegex(RuntimeError, "No Muse devices found"):
                await source.connect()

    async def test_amused_stream_raises_when_client_stream_fails(self):
        class FailingStreamClient(FakeStreamClient):
            async def connect_and_stream(self, address, duration_seconds=30, preset="p1034"):
                await asyncio.sleep(0)
                return False

        source = AmusedSource(
            address="test-address",
            stream_client_factory=FailingStreamClient,
            verbose=False,
        )
        await source.connect()

        with self.assertRaisesRegex(RuntimeError, "amused stream failed"):
            async for _ in source.stream():
                pass

        self.assertEqual(source.disconnect_reason, "stream_failed")

    async def test_amused_full_frame_queue_records_disconnect_reason(self):
        source = AmusedSource(
            address="test-address",
            stream_client_factory=FakeStreamClient,
            queue_size=1,
            verbose=False,
        )
        await source.connect()
        source._ensure_queue()
        decoded = DecodedData(
            timestamp=DECODED_AT,
            packet_type="EEG",
            eeg={"TP9": [0.25]},
            raw_bytes=b"\x11",
        )

        source._handle_decoded(decoded)
        source._handle_decoded(decoded)

        self.assertEqual(source.frame_count, 2)
        self.assertEqual(source.disconnect_reason, "frame_queue_full")

    async def test_amused_discover_subprocess_parses_device_payload(self):
        completed = subprocess.CompletedProcess(
            args=["python", "-c", "..."],
            returncode=0,
            stdout='[{"name": "Muse S Athena", "address": "aa", "rssi": -42}]',
            stderr="",
        )
        source = AmusedSource(verbose=False)

        with patch("muse_tmr.sources.amused_source.sys.platform", "darwin"), patch(
            "muse_tmr.sources.amused_source.subprocess.run",
            return_value=completed,
        ):
            devices = await source.discover()

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].address, "aa")
        self.assertEqual(devices[0].rssi, -42)

    async def test_amused_discover_subprocess_timeout_reports_permissions_hint(self):
        source = AmusedSource(verbose=False)

        with patch("muse_tmr.sources.amused_source.sys.platform", "darwin"), patch(
            "muse_tmr.sources.amused_source.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="python", timeout=15.0),
        ):
            with self.assertRaisesRegex(RuntimeError, "discovery timed out"):
                await source.discover()

    async def test_amused_discover_subprocess_invalid_payload_raises(self):
        completed = subprocess.CompletedProcess(
            args=["python", "-c", "..."],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
        source = AmusedSource(verbose=False)

        with patch("muse_tmr.sources.amused_source.sys.platform", "darwin"), patch(
            "muse_tmr.sources.amused_source.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid device data"):
                await source.discover()

    async def test_amused_stop_cancels_stream_task_from_another_loop(self):
        source = AmusedSource(verbose=False)
        ready = threading.Event()
        stopped = threading.Event()
        task_holder = {}

        async def sleeper():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                stopped.set()
                raise

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task_holder["loop"] = loop
            task_holder["task"] = loop.create_task(sleeper())
            ready.set()
            loop.run_forever()
            loop.close()

        thread = threading.Thread(target=run_loop)
        thread.start()
        ready.wait(timeout=2)
        source._stream_loop = task_holder["loop"]
        source._stream_task = task_holder["task"]

        await source.stop()

        self.assertTrue(source._stop_requested)
        self.assertTrue(stopped.wait(timeout=2))
        task_holder["loop"].call_soon_threadsafe(task_holder["loop"].stop)
        thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
