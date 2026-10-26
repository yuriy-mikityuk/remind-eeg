import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

import muse_athena_protocol as proto
from muse_raw_stream import MuseRawStream
from muse_tmr.data.replay import ReplayConfig, ReplaySession, resolve_raw_path
from muse_tmr.testing import generate_night
from muse_tmr.testing.synthetic import START_EPOCH_SECONDS

# One minute of the short_allow_short scenario (p1034: EEG, IMU, optics and battery).
MINUTES = 1 / 60

COARSE_TIMERS = sys.platform == "win32" and sys.version_info < (3, 13)
COARSE_TIMERS_REASON = (
    "on Windows before Python 3.13 asyncio's sleeps of microseconds take milliseconds: "
    "the 1000x replay of 12 s took 3.3 s in CI, where the test allows 1 s"
)


def build_tag_packet(first_tag, first_data):
    header = bytearray(14)
    header[9] = first_tag
    return bytes(header) + first_data


def packet_offsets(raw_path):
    stream = MuseRawStream(str(raw_path))
    stream.open_read()
    try:
        return [(packet.timestamp - stream.session_start).total_seconds() for packet in stream.read_packets()]
    finally:
        stream.close()


class TestReplaySession(unittest.IsolatedAsyncioTestCase):
    async def test_replay_yields_muse_frames_from_recording_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            recording_dir = Path(tmp)
            night = generate_night(output_dir=recording_dir, scenario="short_allow_short", hours=MINUTES)
            session = ReplaySession(ReplayConfig(input_path=recording_dir, speed=0.0))

            metadata = await session.connect()
            frames = [frame async for frame in session.stream()]

            self.assertEqual(metadata.source_name, "replay")
            self.assertEqual(metadata.device_name, "Synthetic Muse S Athena")
            self.assertEqual(metadata.metadata["original_source"], "synthetic")
            self.assertEqual(len(frames), night.generation["ble"]["frames"])
            seen = set().union(*(frame.modalities() for frame in frames))
            self.assertTrue({"eeg", "imu", "ppg", "battery"} <= seen)
            self.assertTrue(all(frame.raw_packet for frame in frames))

    async def test_replay_filters_relative_time_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            recording_dir = Path(tmp)
            night = generate_night(output_dir=recording_dir, scenario="short_allow_short", hours=MINUTES)
            session = ReplaySession(
                ReplayConfig(
                    input_path=recording_dir,
                    speed=0.0,
                    start_seconds=5,
                    end_seconds=20,
                )
            )

            frames = [frame async for frame in session.stream()]
            offsets = [offset for offset in packet_offsets(night.raw_path) if 5 <= offset <= 20]

            self.assertEqual(len(frames), len(offsets))
            self.assertGreater(len(frames), 400)
            self.assertTrue(all(5 <= frame.timestamp - START_EPOCH_SECONDS <= 20 for frame in frames))

    async def test_replay_can_feed_downstream_collectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            recording_dir = Path(tmp)
            night = generate_night(output_dir=recording_dir, scenario="short_allow_short", hours=MINUTES)
            session = ReplaySession(ReplayConfig(input_path=recording_dir, speed=0.0))

            modality_counts = {}
            async for frame in session.stream():
                for modality in frame.modalities():
                    modality_counts[modality] = modality_counts.get(modality, 0) + 1

            expected = night.generation["ble"]["frames_with"]
            self.assertEqual({key: modality_counts[key] for key in expected}, expected)

    async def test_replay_drops_a_repeated_notification_and_marks_a_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw_amused.bin"
            stream = MuseRawStream(str(raw_path))
            stream.open_write()
            first = build_tag_packet(proto.TAG_EEG_4CH, bytes(28))
            after_gap = bytearray(build_tag_packet(proto.TAG_EEG_4CH, bytes([1]) + bytes(27)))
            after_gap[10] = 2  # counter 1 never arrived
            for packet in (first, first, bytes(after_gap)):
                stream.write_packet(packet, stream.session_start)
            stream.close()

            frames = [frame async for frame in ReplaySession(ReplayConfig(input_path=raw_path)).stream()]

            self.assertEqual([len(frame.eeg.channels_uv["AF7"]) if frame.eeg else 0 for frame in frames], [4, 0, 4])
            self.assertIsNotNone(frames[1].raw_packet)
            self.assertEqual(frames[2].eeg.gaps, ((0, 4),))

    @unittest.skipIf(COARSE_TIMERS, COARSE_TIMERS_REASON)
    async def test_replay_speed_sleeps_between_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            recording_dir = Path(tmp)
            night = generate_night(output_dir=recording_dir, scenario="short_allow_short", hours=MINUTES)
            session = ReplaySession(
                ReplayConfig(
                    input_path=recording_dir,
                    speed=1000.0,
                    end_seconds=12,
                )
            )

            started = asyncio.get_running_loop().time()
            frames = [frame async for frame in session.stream()]
            elapsed = asyncio.get_running_loop().time() - started

            offsets = [offset for offset in packet_offsets(night.raw_path) if offset <= 12]
            self.assertEqual(len(frames), len(offsets))
            # About 12 s of packet spacing at 1000x is 12 ms of sleeping.
            self.assertGreaterEqual(elapsed, (offsets[-1] - offsets[0]) / 1000.0)
            self.assertLess(elapsed, 1.0)


class TestReplayHelpers(unittest.TestCase):
    def test_resolve_raw_path_accepts_directory_or_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            recording_dir = Path(tmp)
            raw_path = generate_night(output_dir=recording_dir, scenario="short_allow_short", hours=MINUTES).raw_path

            self.assertEqual(resolve_raw_path(recording_dir), raw_path.resolve())
            self.assertEqual(resolve_raw_path(raw_path), raw_path.resolve())

    def test_config_rejects_invalid_time_range(self):
        with self.assertRaises(ValueError):
            ReplayConfig(
                input_path=Path("raw_amused.bin"),
                start_seconds=30,
                end_seconds=10,
            ).validate()


if __name__ == "__main__":
    unittest.main()
