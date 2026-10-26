import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from muse_tmr.data.recorder import OvernightRecorder, RecordingConfig
from muse_tmr.data.sample_types import EEGSample, MuseFrame
from muse_tmr.data.watchdog import RecordingWatchdog
from muse_tmr.sources.base_source import BaseMuseSource, MuseDeviceInfo, MuseSourceMetadata

COARSE_CLOCK = sys.platform == "win32" and sys.version_info < (3, 13)
COARSE_CLOCK_REASON = (
    "on Windows before Python 3.13 time.monotonic() ticks every 15.6 ms and asyncio runs a timer up to one "
    "tick early, so after the last wait the deadline can read as not reached: the recorder takes the timeout "
    "for a stall and reconnects, and the 0.6 s recording takes 2 s"
)


class RecordingFakeSource(BaseMuseSource):
    def __init__(self):
        self.connect_count = 0
        self.stop_count = 0

    async def discover(self):
        return [MuseDeviceInfo(name="Muse Fake", address="fake")]

    async def connect(self, device=None):
        self.connect_count += 1
        return MuseSourceMetadata(
            source_name="fake",
            device_name="Muse Fake",
            device_id="fake",
            capabilities={"eeg": True, "raw_packets": True},
        )

    async def stream(self):
        yield MuseFrame(
            timestamp=1.0,
            eeg=EEGSample(timestamp=1.0, channels_uv={"TP9": [0.1]}),
            source="fake",
            raw_packet=b"\x01\x02",
        )

    async def stop(self):
        self.stop_count += 1


class DropoutFakeSource(RecordingFakeSource):
    async def stream(self):
        if self.connect_count == 1:
            await asyncio.sleep(0.05)
            return
        yield MuseFrame(
            timestamp=2.0,
            eeg=EEGSample(timestamp=2.0, channels_uv={"TP9": [0.2]}),
            source="fake",
            raw_packet=b"\x03\x04",
        )


class ErrorThenRecoverySource(RecordingFakeSource):
    async def stream(self):
        if self.connect_count == 1:
            raise RuntimeError("simulated disconnect")
        yield MuseFrame(
            timestamp=3.0,
            eeg=EEGSample(timestamp=3.0, channels_uv={"TP9": [0.3]}),
            source="fake",
            raw_packet=b"\x05\x06",
        )


class ReconnectAttemptFailsThenRecoversSource(RecordingFakeSource):
    """Simulates a device that is genuinely gone: the *reconnect attempt's*
    own connect() call fails (not just the stream), matching the crash seen
    in production where an unhandled exception from a reconnect's connect()
    took down the whole recording."""

    async def connect(self, device=None):
        self.connect_count += 1
        if self.connect_count == 2:
            raise RuntimeError("simulated reconnect failure")
        return MuseSourceMetadata(
            source_name="fake",
            device_name="Muse Fake",
            device_id="fake",
            capabilities={"eeg": True, "raw_packets": True},
        )

    async def stream(self):
        if self.connect_count < 3:
            await asyncio.sleep(0.05)
            return
        yield MuseFrame(
            timestamp=4.0,
            eeg=EEGSample(timestamp=4.0, channels_uv={"TP9": [0.4]}),
            source="fake",
            raw_packet=b"\x07\x08",
        )


class DiagnosticsFakeSource(RecordingFakeSource):
    async def stream(self):
        yield MuseFrame(
            timestamp=5.0,
            eeg=EEGSample(
                timestamp=5.0,
                channels_uv={
                    channel: [float(i % 7) for i in range(8)]
                    for channel in ("TP9", "AF7", "AF8", "TP10")
                },
            ),
            source="fake",
            raw_packet=b"\x09",
        )

    def diagnostics(self):
        return {"last_packet_age_seconds": 0.2, "decoder": {"eeg_rolling_sample_rate_hz": 128.0}}


class BrokenDiagnosticsFakeSource(DiagnosticsFakeSource):
    def diagnostics(self):
        raise RuntimeError("decoder gone")


class EndlessFakeSource(RecordingFakeSource):
    async def stream(self):
        timestamp = 10.0
        while True:
            yield MuseFrame(
                timestamp=timestamp,
                eeg=EEGSample(timestamp=timestamp, channels_uv={"TP9": [0.1]}),
                source="fake",
                raw_packet=b"\x0a",
            )
            timestamp += 0.01
            await asyncio.sleep(0.01)


class SlowConnectFakeSource(RecordingFakeSource):
    async def connect(self, device=None):
        # Like amused discovery without an explicit address.
        await asyncio.sleep(60)


class TestOvernightRecorder(unittest.IsolatedAsyncioTestCase):
    async def test_record_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = RecordingFakeSource()
            recorder = OvernightRecorder(
                RecordingConfig(
                    output_dir=Path(tmp),
                    # The source ends after its one frame. A 10 ms recording could end
                    # before that frame where the clock ticks every 15.6 ms (Windows
                    # before Python 3.13).
                    duration_seconds=1.0,
                    allow_short=True,
                )
            )

            summary = await recorder.record(source)

            self.assertEqual(summary.frame_count, 1)
            self.assertEqual(summary.raw_packet_count, 1)
            self.assertTrue(Path(summary.raw_path).exists())
            self.assertTrue(Path(summary.metadata_path).exists())
            self.assertTrue(Path(summary.events_path).exists())
            self.assertTrue(Path(summary.summary_path).exists())

            payload = json.loads(Path(summary.summary_path).read_text())
            self.assertEqual(payload["modality_counts"]["eeg"], 1)

            progress_path = Path(tmp) / "progress.json"
            self.assertTrue(progress_path.exists())
            progress = json.loads(progress_path.read_text())
            self.assertIn("frame_count", progress)
            self.assertIn("elapsed_seconds", progress)
            self.assertIn("battery_percent", progress)
            self.assertIsNotNone(progress["first_frame_elapsed_seconds"])

    async def test_no_data_timeout_reconnects_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = DropoutFakeSource()
            watchdog = RecordingWatchdog(
                no_data_timeout_seconds=0.01,
                modality_timeout_seconds=1.0,
                backoff_base_seconds=0.0,
            )
            recorder = OvernightRecorder(
                RecordingConfig(
                    output_dir=Path(tmp),
                    duration_seconds=0.08,
                    no_data_timeout_seconds=0.01,
                    max_reconnect_attempts=2,
                    allow_short=True,
                ),
                watchdog=watchdog,
            )

            summary = await recorder.record(source)

            self.assertGreaterEqual(summary.reconnect_attempts, 1)
            self.assertEqual(summary.frame_count, 1)
            events = Path(summary.events_path).read_text()
            self.assertIn("no_data_timeout", events)
            self.assertIn("reconnect_scheduled", events)

    async def test_stream_error_reconnects_and_logs_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = ErrorThenRecoverySource()
            watchdog = RecordingWatchdog(
                no_data_timeout_seconds=0.01,
                modality_timeout_seconds=1.0,
                backoff_base_seconds=0.0,
            )
            recorder = OvernightRecorder(
                RecordingConfig(
                    output_dir=Path(tmp),
                    duration_seconds=0.05,
                    no_data_timeout_seconds=0.01,
                    max_reconnect_attempts=2,
                    allow_short=True,
                ),
                watchdog=watchdog,
            )

            summary = await recorder.record(source)

            self.assertGreaterEqual(summary.reconnect_attempts, 1)
            self.assertEqual(summary.frame_count, 1)
            events = Path(summary.events_path).read_text()
            self.assertIn("stream_error", events)
            self.assertIn("simulated disconnect", events)

    async def test_reconnect_attempt_failure_does_not_crash_and_keeps_retrying(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = ReconnectAttemptFailsThenRecoversSource()
            watchdog = RecordingWatchdog(
                no_data_timeout_seconds=0.01,
                modality_timeout_seconds=1.0,
                backoff_base_seconds=0.0,
            )
            recorder = OvernightRecorder(
                RecordingConfig(
                    output_dir=Path(tmp),
                    duration_seconds=0.1,
                    no_data_timeout_seconds=0.01,
                    max_reconnect_attempts=3,
                    allow_short=True,
                ),
                watchdog=watchdog,
            )

            # Before the fix, source.connect() raising during a reconnect
            # attempt propagated uncaught and this await would raise instead
            # of returning a summary.
            summary = await recorder.record(source)

            self.assertEqual(summary.frame_count, 1)
            self.assertGreaterEqual(summary.reconnect_attempts, 2)
            events = Path(summary.events_path).read_text()
            self.assertIn("reconnect_failed", events)
            self.assertIn("simulated reconnect failure", events)

    async def test_progress_publishes_contact_and_source_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = OvernightRecorder(
                RecordingConfig(output_dir=Path(tmp), duration_seconds=0.01, allow_short=True)
            )

            await recorder.record(DiagnosticsFakeSource())

            progress = json.loads((Path(tmp) / "progress.json").read_text())
            contact = progress["contact"]
            self.assertEqual(contact["connection_state"], "connected")
            self.assertFalse(contact["stale"])
            self.assertEqual(
                sorted(contact["channels"]), ["AF7", "AF8", "TP10", "TP9"]
            )
            self.assertEqual(contact["channels"]["TP9"]["sample_count"], 8)
            self.assertEqual(
                progress["source_diagnostics"]["decoder"]["eeg_rolling_sample_rate_hz"], 128.0
            )

    async def test_failing_source_diagnostics_do_not_break_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = OvernightRecorder(
                # The source ends after its one frame, see test_record_writes_expected_outputs.
                RecordingConfig(output_dir=Path(tmp), duration_seconds=1.0, allow_short=True)
            )

            summary = await recorder.record(BrokenDiagnosticsFakeSource())

            self.assertEqual(summary.frame_count, 1)
            progress = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertIsNone(progress["source_diagnostics"])
            self.assertIn("TP9", progress["contact"]["channels"])

    async def test_cancel_finishes_as_user_stop_with_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = EndlessFakeSource()
            recorder = OvernightRecorder(
                RecordingConfig(output_dir=Path(tmp), duration_seconds=60, allow_short=True)
            )
            task = asyncio.create_task(recorder.record(source))
            await asyncio.sleep(0.1)

            # What asyncio.run does on SIGINT from the app's Stop button.
            task.cancel()
            summary = await task

            self.assertEqual(summary.stop_reason, "user_stopped")
            self.assertGreater(summary.frame_count, 0)
            self.assertEqual(source.stop_count, 1)
            payload = json.loads(Path(summary.summary_path).read_text())
            self.assertEqual(payload["stop_reason"], "user_stopped")
            events = Path(summary.events_path).read_text()
            self.assertIn("recording_stopped", events)

    async def test_cancel_while_connecting_still_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = SlowConnectFakeSource()
            recorder = OvernightRecorder(
                RecordingConfig(output_dir=Path(tmp), duration_seconds=60, allow_short=True)
            )
            task = asyncio.create_task(recorder.record(source))
            await asyncio.sleep(0.05)

            task.cancel()
            summary = await task

            self.assertEqual(summary.stop_reason, "user_stopped")
            self.assertEqual(summary.frame_count, 0)
            self.assertEqual(source.stop_count, 1)
            self.assertTrue(Path(summary.summary_path).exists())
            events = Path(summary.events_path).read_text()
            self.assertIn("recording_stopped", events)

    async def test_stop_is_not_lost_when_the_cancel_gets_swallowed(self):
        # What asyncio.wait_for can do on Python 3.11 when a frame and the
        # cancel arrive together: the cancel never reaches the recorder.
        class SwallowingSource(EndlessFakeSource):
            async def stream(self):
                timestamp = 10.0
                while True:
                    yield MuseFrame(
                        timestamp=timestamp,
                        eeg=EEGSample(timestamp=timestamp, channels_uv={"TP9": [0.1]}),
                        source="fake",
                    )
                    timestamp += 0.01
                    try:
                        await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        pass

        with tempfile.TemporaryDirectory() as tmp:
            recorder = OvernightRecorder(
                RecordingConfig(output_dir=Path(tmp), duration_seconds=30, allow_short=True)
            )
            task = asyncio.create_task(recorder.record(SwallowingSource()))
            await asyncio.sleep(0.1)
            task.cancel()
            summary = await asyncio.wait_for(task, timeout=5)
            self.assertEqual(summary.stop_reason, "user_stopped")

    @unittest.skipIf(COARSE_CLOCK, COARSE_CLOCK_REASON)
    async def test_duration_can_count_from_the_first_frame(self):
        class SlowConnect(EndlessFakeSource):
            async def connect(self, device=None):
                await asyncio.sleep(0.4)  # a slow BLE connect
                return await super().connect(device)

        for from_first_frame in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                recorder = OvernightRecorder(
                    RecordingConfig(
                        output_dir=Path(tmp),
                        duration_seconds=0.6,
                        allow_short=True,
                        duration_from_first_frame=from_first_frame,
                    )
                )
                started = asyncio.get_running_loop().time()
                summary = await recorder.record(SlowConnect())
                wall = asyncio.get_running_loop().time() - started
                self.assertEqual(summary.stop_reason, "duration_complete")
                if from_first_frame:
                    self.assertGreater(wall, 0.95)  # 0.4 s connect + a full 0.6 s of data
                else:
                    self.assertLess(wall, 0.85)

    def test_duration_requires_overnight_window_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                RecordingConfig(output_dir=Path(tmp), duration_seconds=10).validate()


if __name__ == "__main__":
    unittest.main()
