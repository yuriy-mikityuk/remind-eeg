"""
Tests for Muse Real-time Decoder
"""

import unittest
import datetime
import struct
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from muse_tmr.ble.muse_raw_stream import MuseRawStream
from muse_tmr.ble.muse_realtime_decoder import MuseRealtimeDecoder, DecodedData
from muse_tmr.ble import muse_athena_protocol as proto

from muse_tmr.data.sample_types import frame_from_decoded

CAPTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'muse_20250824_172756.bin')


def build_tag_packet(first_tag, first_data, extra_subpackets=None, counter=0):
    """Build a synthetic TAG-based packet; each TAG's subpackets count up from `counter`."""
    counters = {}

    def next_counter(tag):
        value = counters.get(tag, counter)
        counters[tag] = value + 1
        return value % 256

    header = bytearray(14)
    header[9] = first_tag
    header[10] = next_counter(first_tag)
    packet = bytes(header) + first_data
    if extra_subpackets:
        for tag, data in extra_subpackets:
            sub_header = bytearray(4)
            sub_header[0] = next_counter(tag)
            packet += bytes([tag]) + bytes(sub_header) + data
    return packet


class TestRealtimeDecoder(unittest.TestCase):
    """Test real-time packet decoding"""

    def setUp(self):
        """Set up test fixtures"""
        self.decoder = MuseRealtimeDecoder()

    def test_eeg_packet_decoding(self):
        """Test EEG packet decoding with TAG-based format"""
        eeg_packet = build_tag_packet(proto.TAG_EEG_4CH, bytes(28))

        decoded = self.decoder.decode(eeg_packet)

        self.assertIsNotNone(decoded.eeg)
        # Should have 4 channels
        self.assertEqual(len(decoded.eeg), 4)
        # Should use proper channel names
        for name in proto.EEG_CHANNELS_4:
            self.assertIn(name, decoded.eeg)
        # Each channel should have 4 samples (4ch mode)
        self.assertEqual(len(decoded.eeg['TP9']), 4)

    def test_eeg_packet_preserves_athena_channel_order(self):
        """Synthetic 4ch EEG columns should map to TP9, AF7, AF8, TP10."""
        rows = [
            [proto.EEG_MID_COUNT + 10, proto.EEG_MID_COUNT + 20,
             proto.EEG_MID_COUNT + 30, proto.EEG_MID_COUNT + 40],
            [proto.EEG_MID_COUNT + 11, proto.EEG_MID_COUNT + 21,
             proto.EEG_MID_COUNT + 31, proto.EEG_MID_COUNT + 41],
            [proto.EEG_MID_COUNT + 12, proto.EEG_MID_COUNT + 22,
             proto.EEG_MID_COUNT + 32, proto.EEG_MID_COUNT + 42],
            [proto.EEG_MID_COUNT + 13, proto.EEG_MID_COUNT + 23,
             proto.EEG_MID_COUNT + 33, proto.EEG_MID_COUNT + 43],
        ]
        eeg_packet = build_tag_packet(
            proto.TAG_EEG_4CH,
            _pack_14bit_lsb([value for row in rows for value in row]),
        )

        decoded = self.decoder.decode(eeg_packet)

        self.assertAlmostEqual(decoded.eeg['TP9'][0], _uv(proto.EEG_MID_COUNT + 10), places=5)
        self.assertAlmostEqual(decoded.eeg['AF7'][0], _uv(proto.EEG_MID_COUNT + 20), places=5)
        self.assertAlmostEqual(decoded.eeg['AF8'][0], _uv(proto.EEG_MID_COUNT + 30), places=5)
        self.assertAlmostEqual(decoded.eeg['TP10'][0], _uv(proto.EEG_MID_COUNT + 40), places=5)
        self.assertAlmostEqual(decoded.eeg['AF7'][3], _uv(proto.EEG_MID_COUNT + 23), places=5)

    def test_imu_packet_decoding(self):
        """Test IMU packet decoding with TAG-based format"""
        imu_packet = build_tag_packet(proto.TAG_ACCGYRO, bytes(36))

        decoded = self.decoder.decode(imu_packet)

        self.assertIsNotNone(decoded.imu)
        self.assertIn('accel', decoded.imu)
        self.assertIn('gyro', decoded.imu)
        # 3 samples x 3 accel channels
        self.assertEqual(len(decoded.imu['accel']), 3)
        self.assertEqual(len(decoded.imu['gyro']), 3)

    def test_optics_packet_decoding(self):
        """Test optics/PPG packet decoding with TAG-based format"""
        optics_packet = build_tag_packet(proto.TAG_OPTICS_8CH, bytes(40))

        decoded = self.decoder.decode(optics_packet)

        self.assertIsNotNone(decoded.ppg)
        # Should have 8 optics channels
        self.assertEqual(len(decoded.ppg), 8)
        for name in proto.OPTICS_CHANNELS_8:
            self.assertIn(name, decoded.ppg)

    def test_multi_subpacket(self):
        """Test packet with multiple subpacket types"""
        packet = build_tag_packet(
            proto.TAG_EEG_4CH, bytes(28),
            extra_subpackets=[
                (proto.TAG_ACCGYRO, bytes(36)),
                (proto.TAG_OPTICS_8CH, bytes(40)),
            ]
        )

        decoded = self.decoder.decode(packet)

        self.assertIsNotNone(decoded.eeg)
        self.assertIsNotNone(decoded.imu)
        self.assertIsNotNone(decoded.ppg)
        self.assertEqual(decoded.packet_type, 'MULTI')

        stats = self.decoder.get_stats()
        self.assertEqual(stats['tag_counts']['0x11'], 1)
        self.assertEqual(stats['tag_counts']['0x47'], 1)
        self.assertEqual(stats['tag_counts']['0x35'], 1)
        self.assertEqual(stats['tag_type_counts']['EEG'], 1)
        self.assertEqual(stats['tag_type_counts']['ACCGYRO'], 1)
        self.assertEqual(stats['tag_type_counts']['OPTICS'], 1)
        self.assertEqual(stats['unknown_tag_counts'], {})

    def test_several_eeg_subpackets_append_channel_samples(self):
        """Several EEG subpackets in one notification should not overwrite."""
        packet = build_tag_packet(
            proto.TAG_EEG_4CH, bytes(28),
            extra_subpackets=[
                (proto.TAG_EEG_4CH, bytes(28)),
            ]
        )

        decoded = self.decoder.decode(packet)
        stats = self.decoder.get_stats()

        self.assertIsNotNone(decoded.eeg)
        for name in proto.EEG_CHANNELS_4:
            self.assertEqual(len(decoded.eeg[name]), 8)
        self.assertEqual(stats['eeg_subpackets'], 2)
        self.assertEqual(stats['eeg_sample_rows'], 8)
        self.assertEqual(stats['eeg_values'], 32)
        self.assertEqual(stats['eeg_samples'], 32)

    def test_several_ppg_subpackets_append_channel_samples(self):
        """Several PPG subpackets in one notification should not overwrite."""
        packet = build_tag_packet(
            proto.TAG_OPTICS_8CH, bytes(40),
            extra_subpackets=[
                (proto.TAG_OPTICS_8CH, bytes(40)),
            ]
        )

        decoded = self.decoder.decode(packet)
        stats = self.decoder.get_stats()

        self.assertIsNotNone(decoded.ppg)
        for name in proto.OPTICS_CHANNELS_8:
            self.assertEqual(len(decoded.ppg[name]), 4)
        self.assertEqual(stats['ppg_subpackets'], 2)
        self.assertEqual(stats['ppg_sample_rows'], 4)
        self.assertEqual(stats['ppg_values'], 32)
        self.assertEqual(stats['ppg_samples'], 4)

    def test_two_imu_subpackets_append_rows_in_order(self):
        """Every ACCGYRO subpacket of a notification reaches the rows, not only the last."""
        decoded = self.decoder.decode(_notification(
            (proto.TAG_ACCGYRO, 0, _imu_data(1)),
            (proto.TAG_EEG_4CH, 0, bytes(28)),
            (proto.TAG_ACCGYRO, 1, _imu_data(2)),
        ))

        self.assertEqual(_imu_levels(decoded), [1, 1, 1, 2, 2, 2])
        self.assertEqual(len(decoded.imu['gyro']), 6)
        self.assertEqual(decoded.gaps, {})
        self.assertEqual(self.decoder.get_stats()['imu_samples'], 6)

    def test_three_imu_subpackets_append_rows_in_order(self):
        decoded = self.decoder.decode(_notification(
            *((proto.TAG_ACCGYRO, counter, _imu_data(counter + 1)) for counter in range(3))
        ))

        self.assertEqual(_imu_levels(decoded), [1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.assertAlmostEqual(decoded.imu['gyro'][8][2], 3 * proto.GYRO_SCALE, places=5)
        self.assertEqual(decoded.packet_type, 'IMU')
        self.assertEqual(self.decoder.get_stats()['imu_samples'], 9)

    def test_callback_system(self):
        """Test callback registration and triggering"""
        eeg_called = False
        packet_data = None

        def on_eeg(data: DecodedData):
            nonlocal eeg_called, packet_data
            eeg_called = True
            packet_data = data

        self.decoder.register_callback('eeg', on_eeg)

        eeg_packet = build_tag_packet(proto.TAG_EEG_4CH, bytes(28))
        self.decoder.decode(eeg_packet)

        self.assertTrue(eeg_called)
        self.assertIsNotNone(packet_data)
        self.assertIsNotNone(packet_data.eeg)

    def test_statistics_tracking(self):
        """Test statistics tracking"""
        self.decoder.reset_stats()

        eeg_packet = build_tag_packet(proto.TAG_EEG_4CH, bytes(28))
        imu_packet = build_tag_packet(proto.TAG_ACCGYRO, bytes(36))

        self.decoder.decode(eeg_packet)
        self.decoder.decode(imu_packet)

        stats = self.decoder.get_stats()

        self.assertEqual(stats['packets_decoded'], 2)
        self.assertGreater(stats['eeg_samples'], 0)
        self.assertEqual(stats['eeg_subpackets'], 1)
        self.assertEqual(stats['eeg_sample_rows'], 4)
        self.assertEqual(stats['eeg_values'], 16)
        self.assertGreater(stats['imu_samples'], 0)
        self.assertEqual(stats['decode_errors'], 0)

    def test_effective_eeg_sample_rate_uses_sample_rows(self):
        """Sample-rate accounting is based on per-channel EEG sample rows."""
        self.decoder.reset_stats()
        t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)

        for counter, seconds in enumerate((0, 1)):
            self.decoder.decode(
                build_tag_packet(proto.TAG_EEG_4CH, bytes(28), counter=counter),
                timestamp=t0 + datetime.timedelta(seconds=seconds),
            )

        stats = self.decoder.get_stats()
        self.assertEqual(stats['eeg_sample_rows'], 8)
        self.assertAlmostEqual(stats['eeg_effective_sample_rate_hz'], 8.0)

    def test_rolling_eeg_sample_rate_uses_recent_modality_window(self):
        """Rolling rate should avoid old startup samples."""
        self.decoder.reset_stats()
        t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)

        for counter, seconds in enumerate((0, 20, 21, 22)):
            self.decoder.decode(
                build_tag_packet(proto.TAG_EEG_4CH, bytes(28), counter=counter),
                timestamp=t0 + datetime.timedelta(seconds=seconds),
            )

        stats = self.decoder.get_stats()
        self.assertEqual(stats['eeg_sample_rows'], 16)
        self.assertAlmostEqual(stats['eeg_effective_sample_rate_hz'], 16 / 22)
        self.assertAlmostEqual(stats['eeg_rolling_sample_rate_hz'], 8 / 2)
        self.assertAlmostEqual(stats['eeg_rolling_subpackets_per_second'], 2 / 2)

    def test_unknown_and_short_payload_diagnostics(self):
        """Malformed packets should be visible in decoder diagnostics."""
        unknown = bytearray(14)
        unknown[9] = 0x99

        self.decoder.decode(bytes(unknown) + bytes(28))
        self.decoder.decode(b"")

        stats = self.decoder.get_stats()
        self.assertEqual(stats['unknown_tag_counts'], {'0x99': 1})
        self.assertEqual(stats['short_notifications'], 1)
        self.assertEqual(stats['truncated_notifications'], 1)

    def test_battery_telemetry_updates_decoder_diagnostics(self):
        """Battery telemetry should decode without polluting unknown TAG stats."""
        battery_raw = int(91.5 * 256).to_bytes(2, byteorder="little") + bytes(198)
        packet = bytearray(build_tag_packet(proto.TAG_BATTERY_1, battery_raw))
        packet[0] = len(packet)

        decoded = self.decoder.decode(bytes(packet))

        self.assertEqual(decoded.packet_type, 'BATTERY')
        self.assertAlmostEqual(decoded.battery, 91.5)
        stats = self.decoder.get_stats()
        self.assertEqual(stats['tag_counts'], {'0x88': 1})
        self.assertEqual(stats['tag_type_counts'], {'BATTERY': 1})
        self.assertEqual(stats['unknown_tag_counts'], {})
        self.assertEqual(stats['battery_packets'], 1)
        self.assertEqual(stats['battery_payload_bytes'], 200)
        self.assertAlmostEqual(stats['battery_last_percent'], 91.5)

    def test_drl_ref_telemetry_updates_decoder_diagnostics(self):
        """DRL/REF telemetry should be known but not emitted as contact data."""
        packet = build_tag_packet(proto.TAG_DRL_REF, bytes(24))

        decoded = self.decoder.decode(packet)

        self.assertEqual(decoded.packet_type, 'SENSOR')
        self.assertIsNone(decoded.eeg)
        stats = self.decoder.get_stats()
        self.assertEqual(stats['tag_counts'], {'0x53': 1})
        self.assertEqual(stats['tag_type_counts'], {'DRLREF': 1})
        self.assertEqual(stats['unknown_tag_counts'], {})
        self.assertEqual(stats['drl_ref_packets'], 1)
        self.assertEqual(stats['drl_ref_payload_bytes'], 24)

    def test_error_handling(self):
        """Test error handling for malformed packets"""
        # Empty packet
        decoded = self.decoder.decode(b'')
        self.assertEqual(decoded.packet_type, 'EMPTY')

        # Very short packet
        decoded = self.decoder.decode(b'\x00')
        self.assertIsNotNone(decoded)

        # Malformed packet shouldn't crash
        decoded = self.decoder.decode(b'\xFF\xFF\xFF')
        self.assertIsNotNone(decoded)

        stats = self.decoder.get_stats()
        self.assertGreaterEqual(stats['packets_decoded'], 3)


class TestSubpacketCounters(unittest.TestCase):
    """Repeated and lost subpackets, from the counter in each subpacket header."""

    def setUp(self):
        self.decoder = MuseRealtimeDecoder()

    def test_repeated_notification_adds_no_samples(self):
        packet = _notification((proto.TAG_EEG_4CH, 7, _eeg_data(1)))

        first = self.decoder.decode(packet)
        copy = self.decoder.decode(packet)

        self.assertEqual(len(first.eeg['AF7']), 4)
        self.assertIsNone(copy.eeg)
        self.assertEqual(copy.raw_bytes, packet)
        stats = self.decoder.get_stats()
        self.assertEqual(stats['eeg_sample_rows'], 4)
        self.assertEqual(stats['repeated_subpackets'], {'EEG': 1})
        self.assertEqual(stats['lost_subpackets'], {})

    def test_copy_of_an_earlier_notification_is_dropped(self):
        packets = [_notification((proto.TAG_EEG_4CH, counter, _eeg_data(counter))) for counter in range(4)]

        decoded = [self.decoder.decode(packet) for packet in packets[:3] + [packets[1], packets[3]]]

        self.assertIsNone(decoded[3].eeg)
        self.assertEqual(_levels(decoded[4]), [3] * 4)
        self.assertEqual(decoded[4].gaps, {})
        stats = self.decoder.get_stats()
        self.assertEqual(stats['repeated_subpackets'], {'EEG': 1})
        self.assertEqual(stats['lost_subpackets'], {})

    def test_counter_gap_is_marked_not_filled(self):
        self.decoder.decode(_notification((proto.TAG_EEG_4CH, 10, _eeg_data(1))))
        decoded = self.decoder.decode(_notification((proto.TAG_EEG_4CH, 12, _eeg_data(2))))

        self.assertEqual(_levels(decoded), [2] * 4)
        self.assertEqual(decoded.gaps, {'eeg': [(0, 4)]})
        self.assertEqual(self.decoder.get_stats()['lost_subpackets'], {'EEG': 1})

    def test_gap_inside_one_notification_is_marked_at_its_row(self):
        decoded = self.decoder.decode(_notification(
            (proto.TAG_EEG_4CH, 5, _eeg_data(1)),
            (proto.TAG_EEG_4CH, 8, _eeg_data(2)),
        ))

        self.assertEqual(len(decoded.eeg['AF7']), 8)
        self.assertEqual(decoded.gaps, {'eeg': [(4, 8)]})
        self.assertEqual(self.decoder.get_stats()['lost_subpackets'], {'EEG': 2})

    def test_counter_wraps_at_256(self):
        gaps = [
            self.decoder.decode(_notification((proto.TAG_EEG_4CH, counter, _eeg_data(counter)))).gaps
            for counter in (254, 255, 0, 2)
        ]

        self.assertEqual(gaps, [{}, {}, {}, {'eeg': [(0, 4)]}])
        self.assertEqual(self.decoder.get_stats()['lost_subpackets'], {'EEG': 1})

    def test_same_counter_with_new_data_is_kept(self):
        for counter, level in ((8, 1), (9, 2), (8, 3), (9, 4)):
            decoded = self.decoder.decode(_notification((proto.TAG_EEG_4CH, counter, _eeg_data(level))))
            self.assertEqual(_levels(decoded), [level] * 4)
            self.assertEqual(decoded.gaps, {})

        stats = self.decoder.get_stats()
        self.assertEqual(stats['eeg_sample_rows'], 16)
        self.assertEqual(stats['repeated_subpackets'], {})
        self.assertEqual(stats['lost_subpackets'], {})

    def test_optics_and_imu_counters(self):
        optics = (proto.TAG_OPTICS_8CH, 0, bytes(40))
        imu = (proto.TAG_ACCGYRO, 4, bytes(36))
        self.decoder.decode(_notification(optics, imu))

        decoded = self.decoder.decode(_notification(optics, imu, (proto.TAG_OPTICS_8CH, 2, bytes([1]) + bytes(39))))

        self.assertEqual(len(decoded.ppg['LO_NIR']), 2)
        self.assertEqual(decoded.gaps, {'ppg': [(0, 2)]})
        self.assertIsNone(decoded.imu)
        stats = self.decoder.get_stats()
        self.assertEqual(stats['repeated_subpackets'], {'OPTICS': 1, 'ACCGYRO': 1})
        self.assertEqual(stats['lost_subpackets'], {'OPTICS': 1})

    def test_lost_imu_subpacket_is_marked_as_a_gap(self):
        self.decoder.decode(_notification((proto.TAG_ACCGYRO, 20, _imu_data(1))))
        after_loss = self.decoder.decode(_notification((proto.TAG_ACCGYRO, 22, _imu_data(3))))
        inside = self.decoder.decode(_notification(
            (proto.TAG_ACCGYRO, 23, _imu_data(4)),
            (proto.TAG_ACCGYRO, 26, _imu_data(7)),
        ))

        self.assertEqual(_imu_levels(after_loss), [3, 3, 3])
        self.assertEqual(after_loss.gaps, {'imu': [(0, 3)]})
        self.assertEqual(_imu_levels(inside), [4, 4, 4, 7, 7, 7])
        self.assertEqual(inside.gaps, {'imu': [(3, 6)]})
        stats = self.decoder.get_stats()
        self.assertEqual(stats['lost_subpackets'], {'ACCGYRO': 3})
        self.assertEqual(stats['imu_samples'], 12)


class TestCommittedCapture(unittest.TestCase):
    """tests/test_data/muse_20250824_172756.bin: 120 notifications, 34 of them with two or three ACCGYRO subpackets."""

    def test_frames_get_every_imu_row_in_order(self):
        stream = MuseRawStream(CAPTURE)
        stream.open_read()
        decoder = MuseRealtimeDecoder()
        frame_rows, subpacket_rows, per_notification = [], [], []
        try:
            for packet in stream.read_packets():
                frame = frame_from_decoded(decoder.decode(packet.data, packet.timestamp))
                subpackets = proto.parse_payload(packet.data)["ACCGYRO"]
                per_notification.append(len(subpackets))
                for subpacket in subpackets:
                    subpacket_rows.extend(subpacket["data"][:, 0:3].tolist())
                if frame.imu:
                    frame_rows.extend([row["x"], row["y"], row["z"]] for row in frame.imu.accelerometer_g)
                    self.assertEqual(frame.imu.gaps, ())
        finally:
            stream.close()

        stats = decoder.get_stats()
        self.assertEqual((per_notification.count(2), per_notification.count(3)), (32, 2))
        self.assertEqual(stats['lost_subpackets'], {})
        self.assertEqual(stats['repeated_subpackets'], {})
        self.assertEqual(len(frame_rows), stats['imu_samples'])
        self.assertEqual(stats['imu_samples'], 357)
        self.assertEqual(frame_rows, subpacket_rows)


class TestDecodedData(unittest.TestCase):
    """Test DecodedData dataclass"""

    def test_decoded_data_creation(self):
        """Test creating DecodedData objects"""
        data = DecodedData(
            timestamp=datetime.datetime.now(),
            packet_type='TEST',
            eeg={'TP9': [1, 2, 3]},
            ppg={'LO_NIR': [100, 200]},
            imu={'accel': [[0, 0, 1]], 'gyro': [[0, 0, 0]]},
            heart_rate=72.5,
            battery=85,
            raw_bytes=b'\x00\x01\x02'
        )

        self.assertEqual(data.packet_type, 'TEST')
        self.assertEqual(len(data.eeg['TP9']), 3)
        self.assertEqual(data.heart_rate, 72.5)
        self.assertEqual(data.battery, 85)


def _pack_14bit_lsb(values):
    data = bytearray(28)
    for value_idx, value in enumerate(values):
        for bit_idx in range(14):
            if int(value) & (1 << bit_idx):
                output_bit = value_idx * 14 + bit_idx
                data[output_bit // 8] |= 1 << (output_bit % 8)
    return bytes(data)


def _uv(raw_count):
    return (float(raw_count) - proto.EEG_MID_COUNT) * proto.EEG_SCALE_UV_PER_COUNT


def _eeg_data(level):
    """4ch EEG data bytes with every sample at midscale + level."""
    return _pack_14bit_lsb([proto.EEG_MID_COUNT + level] * 16)


def _levels(decoded):
    """The AF7 samples of a decoded notification as levels above midscale."""
    return [round(value / proto.EEG_SCALE_UV_PER_COUNT) for value in decoded.eeg['AF7']]


def _imu_data(level):
    """ACCGYRO data bytes with every value of its 3 samples x 6 axes at `level` counts."""
    return struct.pack('<18h', *([level] * 18))


def _imu_levels(decoded):
    """The accelerometer x of every IMU row of a decoded notification, in counts."""
    return [round(row[0] / proto.ACC_SCALE) for row in decoded.imu['accel']]


def _notification(*subpackets):
    """One notification from (tag, counter, data) subpackets."""
    (first_tag, first_counter, first_data), *rest = subpackets
    header = bytearray(14)
    header[9] = first_tag
    header[10] = first_counter
    packet = bytes(header) + first_data
    for tag, counter, data in rest:
        packet += bytes([tag, counter, 0, 0, 0]) + data
    return packet


if __name__ == '__main__':
    unittest.main()
