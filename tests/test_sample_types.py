import unittest
import datetime as dt

from muse_realtime_decoder import DecodedData
from muse_tmr.data.sample_types import (
    EEGSample,
    IMUSample,
    MuseFrame,
    frame_from_decoded,
)


class TestSampleTypes(unittest.TestCase):
    def test_frame_reports_present_modalities(self):
        eeg = EEGSample(timestamp=1.0, channels_uv={"TP9": 0.1})
        frame = MuseFrame(timestamp=1.0, eeg=eeg, source="test")

        self.assertEqual(frame.modalities(), ("eeg",))
        self.assertEqual(frame.eeg.channels_uv["TP9"], (0.1,))

    def test_frame_serializes_missing_modalities_and_raw_packet(self):
        frame = MuseFrame(timestamp=1.0, source="test", raw_packet=b"\x01\x02")

        restored = MuseFrame.from_json(frame.to_json())

        self.assertEqual(restored.timestamp, 1.0)
        self.assertEqual(restored.modalities(), ())
        self.assertEqual(restored.raw_packet, b"\x01\x02")

    def test_decoded_data_conversion_preserves_units_and_source(self):
        timestamp = dt.datetime.fromtimestamp(2.0)
        decoded = DecodedData(
            timestamp=timestamp,
            packet_type="MULTI",
            eeg={"TP9": [1.0, 2.0]},
            ppg={"ir_1": [10.0]},
            imu={"accel": [[0.1, 0.2, 0.3]], "gyro": [[1.0, 2.0, 3.0]]},
            heart_rate=60.0,
            battery=90,
            raw_bytes=b"\x11\x22",
        )

        frame = frame_from_decoded(decoded, source="amused")

        self.assertEqual(frame.timestamp, 2.0)
        self.assertEqual(frame.source, "amused")
        self.assertEqual(frame.eeg.channels_uv["TP9"], (1.0, 2.0))
        self.assertEqual(frame.imu.accelerometer_g[0]["x"], 0.1)
        self.assertEqual(frame.heart_rate.bpm, 60.0)
        self.assertEqual(frame.battery.percent, 90.0)
        self.assertEqual(frame.raw_packet, b"\x11\x22")

    def test_gaps_reach_the_frame_and_survive_json(self):
        decoded = DecodedData(
            timestamp=dt.datetime.fromtimestamp(3.0),
            packet_type="MULTI",
            eeg={"TP9": [1.0] * 8},
            ppg={"LO_NIR": [10.0] * 2},
            imu={"accel": [[0.0, 0.0, 1.0]] * 6, "gyro": [[0.0, 0.0, 0.0]] * 6},
            gaps={"eeg": [(4, 8)], "ppg": [(0, 2)], "imu": [(3, 3)]},
        )

        frame = frame_from_decoded(decoded, source="amused")
        restored = MuseFrame.from_json(frame.to_json())

        self.assertEqual(frame.eeg.gaps, ((4, 8),))
        self.assertEqual(frame.ppg.gaps, ((0, 2),))
        self.assertEqual(frame.imu.gaps, ((3, 3),))
        self.assertEqual(len(frame.imu.accelerometer_g), 6)
        self.assertEqual(restored.eeg.gaps, ((4, 8),))
        self.assertEqual(restored.ppg.gaps, ((0, 2),))
        self.assertEqual(restored.imu.gaps, ((3, 3),))

    def test_no_gaps_key_without_gaps(self):
        eeg = EEGSample(timestamp=1.0, channels_uv={"TP9": [0.1]})
        imu = IMUSample(timestamp=1.0, accelerometer_g=[[0.0, 0.0, 1.0]])

        self.assertNotIn("gaps", eeg.to_dict())
        self.assertEqual(EEGSample.from_dict(eeg.to_dict()).gaps, ())
        self.assertNotIn("gaps", imu.to_dict())
        self.assertEqual(IMUSample.from_dict(imu.to_dict()).gaps, ())


if __name__ == "__main__":
    unittest.main()
