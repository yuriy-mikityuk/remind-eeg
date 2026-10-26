import asyncio
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from muse_tmr.data.polar_recorder import PolarRecorder, PolarRecordingConfig, decode_polar_session
from muse_tmr.data.polar_session import (
    ECG_NOISE_LEAD_SECONDS,
    ECG_NOISE_TAIL_SECONDS,
    NO_CONTACT_LEAD_SECONDS,
    NO_CONTACT_TAIL_SECONDS,
    fit_clock_mapping,
    load_polar_session,
    no_contact_spans,
)
from muse_tmr.data.recorder import CompanionProcess, OvernightRecorder, RecordingConfig
from muse_tmr.features.cardio_resp_features import extract_cardio_resp_features
from muse_tmr.sources.polar_h10 import (
    HEART_RATE_MEASUREMENT,
    PMD_CONTROL_POINT,
    PMD_DATA,
    PmdStreamSettings,
    start_command,
)

from muse_tmr.data.sample_types import EEGSample, MuseFrame
from tests.polar_synthetic import acc_frame_delta, ecg_frame, hr_measurement, write_raw_session
from tests.test_recorder import RecordingFakeSource

POSIX_STOP_SIGNALS = (
    "the recorder stops its companion with SIGINT, then SIGTERM, and these tests start children with "
    "preexec_fn; Windows can send neither signal to a child and has no preexec_fn"
)


class ClockMappingTest(unittest.TestCase):
    def test_offset_drift_and_positive_delays_recovered_within_10_ms(self):
        rng = np.random.default_rng(11)
        true_host = np.sort(rng.uniform(0, 8 * 3600, 20000))
        drift = 50e-6
        sensor = 599_616_000.0 + true_host * (1 + drift)
        received = 5000.0 + true_host + rng.exponential(0.03, true_host.size)
        mapping = fit_clock_mapping(sensor, received)
        error_ms = np.abs(mapping.to_host(sensor) - (5000.0 + true_host)) * 1000.0
        self.assertLess(error_ms.max(), 10.0)
        self.assertAlmostEqual(mapping.drift * 1e6, -50.0, delta=1.0)
        self.assertGreater(mapping.delay_p95_ms, mapping.delay_median_ms)


class LoadPolarSessionTest(unittest.TestCase):
    def test_decode_align_and_window_features(self):
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            truth = write_raw_session(tmp, 240, rng, breaths_per_min=6.0)
            decoded = decode_polar_session(Path(tmp))
            self.assertEqual(decoded["errors"], {})
            self.assertGreater(decoded["counts"]["acc_samples"], 0)

            session = load_polar_session(Path(tmp))
            beats = truth["beats_wall"]
            peaks = session.r_peaks["time"].to_numpy()
            errors = np.array([peaks[np.argmin(np.abs(peaks - beat))] - beat for beat in beats[1:-1]])
            self.assertLess(np.abs(errors).max() * 1000.0, 10.0)
            aligned = session.rr["aligned_to"] == "ecg_r_peak"
            self.assertGreater(aligned.mean(), 0.95)
            self.assertLess(session.alignment["rr"]["rr_agreement_ms"], 2.0)
            self.assertEqual(set(session.acc.columns), {"time", "x", "y", "z"})

            features = extract_cardio_resp_features(session, truth["wall0"] + 10, truth["wall0"] + 230)
            self.assertAlmostEqual(features["resp_rate_bpm"], 6.0, delta=0.5)
            self.assertAlmostEqual(features["edr_rate_bpm"], 6.0, delta=0.5)
            self.assertEqual(features["hf_band_valid"], 0.0)
            self.assertTrue(math.isfinite(features["rmssd_ms"]))

    def test_sensor_clock_reset_mid_session_gets_its_own_fit(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as tmp:
            truth = write_raw_session(tmp, 240, rng, reset_at_s=120.0)
            session = load_polar_session(Path(tmp))
            self.assertEqual(len(session.alignment["clock_segments"]), 2)
            times = session.ecg["time"].to_numpy()
            self.assertTrue(np.all(np.diff(times) > 0))
            peaks = session.r_peaks["time"].to_numpy()
            beats = [beat for beat in truth["beats_wall"][1:-1] if abs(beat - truth["wall0"] - 117.5) > 4.0]
            errors = np.array([peaks[np.argmin(np.abs(peaks - beat))] - beat for beat in beats])
            self.assertLess(np.abs(errors).max() * 1000.0, 10.0)

    def test_beats_around_lost_contact_are_dropped(self):
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            truth = write_raw_session(tmp, 240, rng, no_contact=(100.0, 140.0))
            session = load_polar_session(Path(tmp))
            # The flag and the noisy ECG each mark the span, each with its margins.
            low = truth["wall0"] + 100.0 - max(NO_CONTACT_LEAD_SECONDS, ECG_NOISE_LEAD_SECONDS)
            high = truth["wall0"] + 140.0 + max(NO_CONTACT_TAIL_SECONDS, ECG_NOISE_TAIL_SECONDS)
            for frame in (session.rr, session.r_peaks):
                times = frame["time"].to_numpy()
                self.assertFalse(np.any((times > low + 1.5) & (times < high - 1.5)))
                self.assertTrue(np.any(times < low) and np.any(times > high))
            self.assertEqual(len(session.quality["no_contact_spans"]), 1)
            self.assertAlmostEqual(session.quality["no_contact_seconds"], high - low, delta=1.5)
            self.assertGreater(session.quality["no_contact_dropped_rr"], 40)
            self.assertLess(session.rr["rr_ms"].max(), 1300.0)
            self.assertGreater((session.rr["aligned_to"] == "ecg_r_peak").mean(), 0.95)
            # The rail-to-rail noise must not set the detector's polarity for the clean ECG.
            peaks = session.r_peaks["time"].to_numpy()
            beats = [beat for beat in truth["beats_wall"][1:-1] if beat < low - 1.0 or beat > high + 1.0]
            errors = np.array([peaks[np.argmin(np.abs(peaks - beat))] - beat for beat in beats])
            self.assertLess(np.abs(errors).max() * 1000.0, 10.0)

            # A window across the hole uses its longest part with contact, not a splice.
            features = extract_cardio_resp_features(session, truth["wall0"] + 80, truth["wall0"] + 240)
            self.assertAlmostEqual(features["window_seconds"], 240 - (high - truth["wall0"]), delta=1.0)
            self.assertAlmostEqual(features["no_contact_seconds"], high - low, delta=1.0)
            self.assertTrue(math.isfinite(features["rmssd_ms"]))
            # 70% of this one has contact, but its longest stretch is under half of it.
            split = extract_cardio_resp_features(session, truth["wall0"] + 20, truth["wall0"] + 230)
            self.assertEqual(split["window_seconds"], 0.0)
            self.assertTrue(math.isnan(split["rmssd_ms"]))
            inside = extract_cardio_resp_features(session, low + 2, high - 2)
            self.assertEqual(inside["window_seconds"], 0.0)
            self.assertTrue(math.isnan(inside["rmssd_ms"]) and math.isnan(inside["edr_rate_bpm"]))
            # Mostly off the skin: a 35 s fragment does not stand in for the window.
            mostly_off = extract_cardio_resp_features(session, low - 35, high + 5)
            self.assertEqual(mostly_off["window_seconds"], 0.0)
            self.assertAlmostEqual(mostly_off["no_contact_seconds"], high - low, delta=0.5)
            self.assertTrue(math.isnan(mostly_off["resp_rate_bpm"]) and math.isnan(mostly_off["mean_hr_bpm"]))

    def test_noisy_ecg_is_dropped_even_while_contact_is_reported(self):
        # The battery pull: the H10 lost power before it reported anything.
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as tmp:
            truth = write_raw_session(tmp, 240, rng, no_contact=(100.0, 140.0), report_contact=False)
            session = load_polar_session(Path(tmp))
            self.assertTrue(all(contact for contact in session.hr["contact"] if contact is not None))
            low, high = truth["wall0"] + 100.0, truth["wall0"] + 140.0
            for frame in (session.rr, session.r_peaks):
                times = frame["time"].to_numpy()
                self.assertFalse(np.any((times > low) & (times < high)))
            self.assertEqual(session.quality["ecg_noise_spans"], 1)
            self.assertLess(session.rr["rr_ms"].max(), 1300.0)

    def test_two_power_ups_seconds_apart_get_their_own_clocks(self):
        # Battery back in, then the cover closed: the clock restarts twice, and
        # the second time the offset moves only ~20 s (25 s on the real run).
        rng = np.random.default_rng(13)
        with tempfile.TemporaryDirectory() as tmp:
            truth = write_raw_session(tmp, 240, rng, reset_at_s=(120.0, 140.0))
            session = load_polar_session(Path(tmp))
            segments = session.alignment["clock_segments"]
            self.assertEqual(len(segments), 3)
            self.assertTrue(all(abs(segment["drift_ppm"]) < 1000.0 for segment in segments))
            peaks = session.r_peaks["time"].to_numpy()
            beats = [beat for beat in truth["beats_wall"][1:-1] if beat - truth["wall0"] > 145.0]
            errors = np.array([peaks[np.argmin(np.abs(peaks - beat))] - beat for beat in beats])
            self.assertLess(np.abs(errors).max() * 1000.0, 10.0)

    def test_clock_fit_refuses_an_impossible_drift(self):
        sensor = np.arange(0.0, 60.0, 0.5)
        host = sensor + np.where(sensor < 30.0, 0.0, 25.0)  # two clock runs in one segment
        mapping = fit_clock_mapping(sensor, host)
        self.assertEqual(mapping.drift, 0.0)

    def test_unknown_contact_does_not_end_a_loss(self):
        hr = pd.DataFrame({"time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "contact": [True, False, None, None, True, None]})
        self.assertEqual(no_contact_spans(hr, lead_s=0.5, tail_s=0.5), [(0.5, 4.5)])
        open_end = pd.DataFrame({"time": [0.0, 1.0, 2.0], "contact": [True, False, None]})
        self.assertEqual(no_contact_spans(hr=open_end, lead_s=0.5, tail_s=0.5, end_of_data=9.0), [(0.5, 9.5)])

    def test_contact_never_regained_drops_everything_after(self):
        rng = np.random.default_rng(10)
        with tempfile.TemporaryDirectory() as tmp:
            truth = write_raw_session(tmp, 240, rng, no_contact=(180.0, 999.0))
            session = load_polar_session(Path(tmp))
            low = truth["wall0"] + 180.0 - NO_CONTACT_LEAD_SECONDS
            self.assertGreater(session.no_contact[0][1], truth["wall0"] + 240.0)
            for frame in (session.rr, session.r_peaks):
                self.assertFalse(np.any(frame["time"].to_numpy() > low + 1.5))

    def test_without_ecg_rr_keeps_receive_time_estimate(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            write_raw_session(tmp, 90, rng, with_ecg=False)
            session = load_polar_session(Path(tmp))
            self.assertTrue((session.rr["aligned_to"] == "host_receive").all())
            self.assertIn("note", session.alignment["rr"])


class MockPolarClient:
    """Streams HR/ECG/ACC notifications; can drop the link after a number of frames."""

    def __init__(self, frames_before_disconnect=(None,)):
        self.plan = list(frames_before_disconnect)
        self.disconnected = asyncio.Event()
        self.stream_settings = {}
        self.connects = 0
        self.stop_calls = 0
        self._task = None
        self._sensor_ns = 700_000_000_000_000_000

    async def connect(self, on_payload):
        limit = self.plan[min(self.connects, len(self.plan) - 1)]
        self.connects += 1
        self.disconnected = asyncio.Event()
        factor = bytes([0x05, 0x01, 0x00, 0x00, 0x80, 0x3F])  # 1.0f
        on_payload("tx", PMD_CONTROL_POINT, start_command(0, {0: 130, 1: 14}))
        on_payload("rx", PMD_CONTROL_POINT, bytes([0xF0, 0x02, 0x00, 0x00, 0x00]) + factor)
        on_payload("tx", PMD_CONTROL_POINT, start_command(2, {0: 50, 1: 16, 2: 2}))
        on_payload("rx", PMD_CONTROL_POINT, bytes([0xF0, 0x02, 0x02, 0x00, 0x00]) + factor)
        self.stream_settings = {
            "ecg": PmdStreamSettings(130, 14, 1, 1.0),
            "acc": PmdStreamSettings(50, 16, 3, 1.0, 2),
        }
        self._task = asyncio.ensure_future(self._stream(on_payload, limit))
        return {"device_name": "Polar H10 TEST", "model": "H10", "firmware": "x"}

    async def _stream(self, on_payload, limit):
        sent = 0
        while limit is None or sent < limit:
            self._sensor_ns += int(73 * 1e9 / 130)
            on_payload("rx", PMD_DATA, ecg_frame(self._sensor_ns, [100, 200, -50] * 24 + [7]))
            on_payload("rx", PMD_DATA, acc_frame_delta(self._sensor_ns, [(1, 2, 1000 + i) for i in range(36)]))
            on_payload("rx", HEART_RATE_MEASUREMENT, hr_measurement(60, (1000.0,), contact=True))
            sent += 1
            await asyncio.sleep(0.01)
        self.disconnected.set()

    async def stop(self):
        self.stop_calls += 1
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


class PolarRecorderTest(unittest.IsolatedAsyncioTestCase):
    def recorder(self, tmp, seconds):
        return PolarRecorder(
            PolarRecordingConfig(output_dir=Path(tmp), duration_seconds=seconds, backoff_base_seconds=0.01)
        )

    async def test_disconnect_mid_session_reconnects_and_files_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = MockPolarClient(frames_before_disconnect=(5, None))
            # The five frames take about 50 ms, but a Windows runner once spent all of a
            # 0.6 s recording on them and its new files, so the drop came after the end.
            summary = await self.recorder(tmp, 2.0).record(client)

            self.assertEqual(summary["stop_reason"], "duration_complete")
            self.assertEqual(summary["reconnects"], 1)
            self.assertEqual(client.connects, 2)
            self.assertGreaterEqual(client.stop_calls, 2)
            polar = Path(tmp) / "polar"
            for name in ("raw_notifications.jsonl", "events.jsonl", "clock_anchors.jsonl", "metadata.json",
                         "summary.json", "hr_rr.jsonl", "ecg.jsonl", "acc.jsonl"):
                self.assertTrue((polar / name).exists(), name)
            events = [json.loads(line)["event"] for line in (polar / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events.count("connected"), 2)
            self.assertIn("disconnected", events)
            self.assertEqual(events[-1], "recording_stopped")
            anchors = (polar / "clock_anchors.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["label"] for line in (anchors[0], anchors[-1])], ["start", "stop"])
            metadata = json.loads((polar / "metadata.json").read_text())
            self.assertEqual(metadata["streams"]["acc"]["sample_rate"], 50)
            self.assertNotIn("address", json.dumps(metadata))
            decode = summary["decode"]
            self.assertEqual(decode["errors"], {})
            self.assertEqual(decode["counts"]["ecg_frames"], decode["counts"]["acc_frames"])
            self.assertEqual(decode["counts"]["ecg_samples"], 73 * decode["counts"]["ecg_frames"])

            # decode-polar rebuilds the same files from the raw log alone.
            before = (polar / "acc.jsonl").read_text()
            (polar / "acc.jsonl").unlink()
            decode_polar_session(Path(tmp))
            self.assertEqual((polar / "acc.jsonl").read_text(), before)

    async def test_cancel_is_a_clean_user_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = MockPolarClient()
            task = asyncio.create_task(self.recorder(tmp, 60).record(client))
            await asyncio.sleep(0.2)
            task.cancel()
            summary = await task
            self.assertEqual(summary["stop_reason"], "user_stopped")
            self.assertGreaterEqual(client.stop_calls, 1)  # PMD streams always stopped

    async def test_failed_connects_end_at_attempt_limit(self):
        class Unreachable(MockPolarClient):
            async def connect(self, on_payload):
                raise RuntimeError("Polar H10 not found")

        with tempfile.TemporaryDirectory() as tmp:
            recorder = PolarRecorder(
                PolarRecordingConfig(
                    output_dir=Path(tmp), duration_seconds=5, max_reconnect_attempts=2, backoff_base_seconds=0.01
                )
            )
            summary = await recorder.record(Unreachable())
            self.assertEqual(summary["stop_reason"], "max_reconnect_attempts")
            self.assertEqual(summary["failed_connects"], 3)


class SlowFakeSource(RecordingFakeSource):
    async def stream(self):
        timestamp = 1.0
        while True:
            yield MuseFrame(timestamp=timestamp, eeg=EEGSample(timestamp=timestamp, channels_uv={"TP9": [0.1]}), source="fake")
            timestamp += 0.05
            await asyncio.sleep(0.05)


class ControlPointFragmentTest(unittest.IsolatedAsyncioTestCase):
    async def test_split_settings_response_and_stale_response_are_handled(self):
        from muse_tmr.sources.polar_h10 import (
            MEASUREMENT_ACC,
            PMD_CONTROL_POINT,
            ControlPointAssembler,
            PolarH10Client,
        )

        settings_first = bytes([0xF0, 0x01, 0x02, 0x00, 0x01]) + bytes([0x00, 0x04, 25, 0, 50, 0, 100, 0, 200, 0])
        settings_rest = bytes([0xF0, 0x01, 0x02, 0x00, 0x00]) + bytes([0x01, 0x01, 16, 0, 0x02, 0x03, 2, 0, 4, 0, 8, 0])
        assembler = ControlPointAssembler()
        self.assertIsNone(assembler.feed(settings_first))
        whole = assembler.feed(settings_rest)
        self.assertEqual(len(whole.parameters), 22)

        client = PolarH10Client()
        callback = client._notification(PMD_CONTROL_POINT)
        written = []

        class FakeBleak:
            is_connected = True

            async def write_gatt_char(self, uuid, data, response=True):
                written.append(bytes(data))
                if data[0] == 0x01:
                    callback(None, bytes([0xF0, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x82, 0x00]))  # stale ECG answer
                    callback(None, settings_first)
                    callback(None, settings_rest)
                elif data[0] == 0x02:
                    callback(None, bytes([0xF0, 0x02, 0x02, 0x00, 0x00, 0x05, 0x01, 0x00, 0x00, 0x80, 0x3F]))

        client._client = FakeBleak()
        stream = await client._start_stream(MEASUREMENT_ACC, {0: 50, 2: 4})
        self.assertEqual((stream.sample_rate, stream.resolution, stream.range, stream.factor), (50, 16, 4, 1.0))
        self.assertEqual(written[-1], bytes([0x02, 0x02, 0x00, 0x01, 50, 0, 0x01, 0x01, 16, 0, 0x02, 0x01, 4, 0]))


@unittest.skipIf(sys.platform == "win32", POSIX_STOP_SIGNALS)
class StopSignalTest(unittest.TestCase):
    """Reproduces the first live smoke: started via `( ... & )`, SIGINT is inherited as ignored."""

    def run_child(self, stop_signal):
        import signal
        import subprocess
        import time

        script = (
            "import asyncio, pathlib, sys\n"
            "from muse_tmr.cli.main import _cancel_on_stop_signals\n"
            "async def main():\n"
            "    _cancel_on_stop_signals()\n"
            "    pathlib.Path(sys.argv[1]).write_text('ready')\n"
            "    try:\n"
            "        await asyncio.sleep(60)\n"
            "    except asyncio.CancelledError:\n"
            "        pathlib.Path(sys.argv[1]).write_text('cancelled')\n"
            "asyncio.run(main())\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "state"
            import os

            import muse_tmr

            source_root = str(Path(muse_tmr.__file__).resolve().parents[1])
            child = subprocess.Popen(
                [sys.executable, "-c", script, str(marker)],
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
                env={**os.environ, "PYTHONPATH": source_root + os.pathsep + os.environ.get("PYTHONPATH", "")},
            )
            deadline = time.monotonic() + 20
            while not (marker.exists() and marker.read_text() == "ready") and time.monotonic() < deadline:
                time.sleep(0.05)
            child.send_signal(stop_signal)
            code = child.wait(timeout=20)
            return code, marker.read_text()

    def test_sigint_cancels_even_when_inherited_as_ignored(self):
        import signal

        self.assertEqual(self.run_child(signal.SIGINT), (0, "cancelled"))

    def test_sigterm_cancels_too(self):
        import signal

        self.assertEqual(self.run_child(signal.SIGTERM), (0, "cancelled"))

    def test_companion_escalates_to_sigterm_when_sigint_is_ignored(self):
        script = (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGTERM, lambda *a: sys.exit(5))\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "child.log"
            companion = CompanionProcess("polar", [sys.executable, "-c", script], log, stop_timeout_seconds=0.5)
            companion.start()
            import time

            deadline = time.monotonic() + 20
            while "ready" not in (log.read_text() if log.exists() else "") and time.monotonic() < deadline:
                time.sleep(0.05)
            companion.request_stop()
            self.assertEqual(companion.wait(), 5)


class CompanionTest(unittest.IsolatedAsyncioTestCase):
    async def test_crashing_companion_never_affects_the_muse_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            crashing = CompanionProcess(
                "polar", [sys.executable, "-c", "import sys; sys.exit(3)"], Path(tmp) / "polar" / "record-polar.log"
            )
            missing = CompanionProcess("broken", ["/nonexistent/binary"], Path(tmp) / "broken.log")
            recorder = OvernightRecorder(
                RecordingConfig(output_dir=Path(tmp), duration_seconds=2.5, allow_short=True),
                companions=[crashing, missing],
            )
            summary = await recorder.record(SlowFakeSource())

            self.assertEqual(summary.stop_reason, "duration_complete")
            events = [json.loads(line) for line in Path(summary.events_path).read_text().splitlines()]
            names = [(item["event"], item["details"].get("companion")) for item in events]
            self.assertIn(("companion_started", "polar"), names)
            self.assertIn(("companion_start_failed", "broken"), names)
            exited = [item for item in events if item["event"] == "companion_exited"]
            self.assertEqual(exited[0]["details"]["returncode"], 3)

    @unittest.skipIf(sys.platform == "win32", POSIX_STOP_SIGNALS)
    async def test_companion_is_stopped_even_when_muse_cleanup_raises(self):
        class BrokenStop(SlowFakeSource):
            async def stop(self):
                raise RuntimeError("BLE stack gone")

        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "got_sigint"
            script = (
                "import signal, sys, time, pathlib\n"
                f"signal.signal(signal.SIGINT, lambda *a: (pathlib.Path({str(marker)!r}).touch(), sys.exit(0)))\n"
                "time.sleep(60)\n"
            )
            sleeper = CompanionProcess("polar", [sys.executable, "-c", script], Path(tmp) / "polar.log")
            recorder = OvernightRecorder(
                RecordingConfig(output_dir=Path(tmp), duration_seconds=1.0, allow_short=True), companions=[sleeper]
            )
            with self.assertRaises(RuntimeError):
                await recorder.record(BrokenStop())
            sleeper.process.wait(timeout=10)
            self.assertTrue(marker.exists())

    @unittest.skipIf(sys.platform == "win32", POSIX_STOP_SIGNALS)
    async def test_long_running_companion_gets_sigint_after_the_muse_summary(self):
        script = "import signal, sys, time\nsignal.signal(signal.SIGINT, lambda *a: sys.exit(0))\ntime.sleep(60)\n"
        with tempfile.TemporaryDirectory() as tmp:
            sleeper = CompanionProcess("polar", [sys.executable, "-c", script], Path(tmp) / "polar.log")
            recorder = OvernightRecorder(
                RecordingConfig(output_dir=Path(tmp), duration_seconds=1.5, allow_short=True), companions=[sleeper]
            )
            summary = await recorder.record(SlowFakeSource())
            self.assertEqual(summary.stop_reason, "duration_complete")
            self.assertTrue(Path(summary.summary_path).exists())
            events = [json.loads(line) for line in Path(summary.events_path).read_text().splitlines()]
            stopped = [item for item in events if item["event"] == "companion_stopped"]
            self.assertEqual(stopped[0]["details"]["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
