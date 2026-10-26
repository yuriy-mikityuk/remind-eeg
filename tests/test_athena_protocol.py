"""
Tests for Muse Athena Protocol module
Validates correct bit-level decoding of EEG, IMU, and optics data.
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from muse_tmr.ble import muse_athena_protocol as proto


def build_tag_packet(first_tag, first_data, extra_subpackets=None):
    """Build a synthetic TAG-based packet for testing.

    Args:
        first_tag: TAG byte for the first subpacket (goes in header byte 9).
        first_data: Data bytes for the first subpacket.
        extra_subpackets: List of (tag, data) tuples for additional subpackets.

    Returns:
        Complete packet bytes with 14-byte header.
    """
    header = bytearray(14)
    header[9] = first_tag
    packet = bytes(header) + first_data

    if extra_subpackets:
        for tag, data in extra_subpackets:
            # Each extra subpacket: [TAG(1)] [header(4)] [data(N)]
            sub_header = bytearray(4)  # 4 bytes of subpacket header
            packet += bytes([tag]) + bytes(sub_header) + data

    return packet


class TestEEGDecoding(unittest.TestCase):
    """Test 14-bit LSB-first EEG decoding"""

    def test_all_zero_counts_center_to_negative_midscale(self):
        """All-zero ADC counts should decode below physiological zero."""
        data = bytes(28)
        result = proto.decode_eeg(data, n_channels=4)
        self.assertEqual(result.shape, (4, 4))
        expected = -proto.EEG_MID_COUNT * proto.EEG_SCALE_UV_PER_COUNT
        np.testing.assert_allclose(result, np.full((4, 4), expected), rtol=1e-6)

    def test_known_14bit_values(self):
        """Test decoding a known 14-bit LSB-first packed value"""
        # Pack a single 14-bit value = 1 (0b00000000000001) at position 0
        # LSB-first: bit0=1, bits1-13=0
        # In the first byte: bit0=1, rest 0 → byte = 0x01
        # Remaining bytes all zero
        data = bytearray(28)
        data[0] = 0x01  # bit0 of first value = 1

        result = proto.decode_eeg(data, n_channels=4)
        # First value is centered around the unsigned ADC midscale.
        expected = (1.0 - proto.EEG_MID_COUNT) * proto.EEG_SCALE_UV_PER_COUNT
        self.assertAlmostEqual(result[0, 0], expected, places=5)

    def test_max_14bit_value(self):
        """Test maximum 14-bit value (16383 = 0x3FFF)"""
        # All 14 bits set for first value
        # LSB-first: bits 0-7 all set → byte0 = 0xFF
        # bits 8-13 all set → byte1 bits 0-5 set = 0x3F
        data = bytearray(28)
        data[0] = 0xFF
        data[1] = 0x3F  # bits 8-13 of first value

        result = proto.decode_eeg(data, n_channels=4)
        expected = (16383.0 - proto.EEG_MID_COUNT) * proto.EEG_SCALE_UV_PER_COUNT
        self.assertAlmostEqual(result[0, 0], expected, places=3)

    def test_decode_eeg_raw_counts(self):
        """Raw-count decoder exposes unsigned wire values for diagnostics."""
        data = bytearray(28)
        data[0] = 0x01

        result = proto.decode_eeg_raw_counts(data, n_channels=4)

        self.assertEqual(result.dtype, np.uint16)
        self.assertEqual(result.shape, (4, 4))
        self.assertEqual(result[0, 0], 1)

    def test_midscale_count_centers_to_zero_uv(self):
        """ADC midscale should be physiological zero in public EEG output."""
        raw = np.array([[0, proto.EEG_MID_COUNT, proto.EEG_MAX_COUNT]], dtype=np.uint16)

        result = proto.eeg_counts_to_uv_centered(raw)

        self.assertAlmostEqual(
            result[0, 0],
            -proto.EEG_MID_COUNT * proto.EEG_SCALE_UV_PER_COUNT,
            places=5,
        )
        self.assertAlmostEqual(result[0, 1], 0.0, places=5)
        self.assertAlmostEqual(
            result[0, 2],
            (proto.EEG_MAX_COUNT - proto.EEG_MID_COUNT) * proto.EEG_SCALE_UV_PER_COUNT,
            places=5,
        )

    def test_decode_eeg_uncentered_uv_keeps_raw_scale(self):
        """Callers can still request raw scaled ADC counts when needed."""
        data = bytearray(28)
        data[0] = 0x01

        result = proto.decode_eeg(data, n_channels=4, centered=False)

        self.assertAlmostEqual(result[0, 0], proto.EEG_SCALE_UV_PER_COUNT, places=5)

    def test_4ch_shape(self):
        """4-channel mode produces (4, 4) array"""
        result = proto.decode_eeg(bytes(28), n_channels=4)
        self.assertEqual(result.shape, (4, 4))

    def test_8ch_shape(self):
        """8-channel mode produces (2, 8) array"""
        result = proto.decode_eeg(bytes(28), n_channels=8)
        self.assertEqual(result.shape, (2, 8))

    def test_scale_factor(self):
        """EEG scale factor should convert to microvolts"""
        # The unsigned 14-bit ADC range represents a 1450 uV peak-to-peak span.
        self.assertAlmostEqual(proto.EEG_SCALE * 16383, 1450.0, places=0)


class TestAccGyroDecoding(unittest.TestCase):
    """Test 16-bit little-endian IMU decoding"""

    def test_all_zeros(self):
        """All-zero data should decode to all zeros"""
        data = bytes(36)
        result = proto.decode_accgyro(data)
        self.assertEqual(result.shape, (3, 6))
        np.testing.assert_array_equal(result, np.zeros((3, 6)))

    def test_known_values(self):
        """Test decoding known 16-bit LE values"""
        # Create data with known int16 LE values
        # First sample, first channel (accel_x) = 100
        values = np.array([[100, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]], dtype=np.int16)
        data = values.tobytes()  # numpy int16 is native, but on most systems LE

        result = proto.decode_accgyro(data)
        # First accel value should be 100 * ACC_SCALE
        self.assertAlmostEqual(result[0, 0], 100.0 * proto.ACC_SCALE, places=6)

    def test_scale_separation(self):
        """Accel and gyro should use different scale factors"""
        values = np.array([[1000, 1000, 1000, 1000, 1000, 1000],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]], dtype='<i2')
        data = values.tobytes()

        result = proto.decode_accgyro(data)
        accel_val = result[0, 0]
        gyro_val = result[0, 3]

        # They should be different due to different scales
        self.assertNotAlmostEqual(accel_val, gyro_val, places=3)


class TestOpticsDecoding(unittest.TestCase):
    """Test 20-bit LSB-first optics decoding"""

    def test_all_zeros(self):
        """All-zero data should decode to all zeros"""
        result = proto.decode_optics(bytes(40), n_channels=8)
        self.assertEqual(result.shape, (2, 8))
        np.testing.assert_array_equal(result, np.zeros((2, 8)))

    def test_4ch_shape(self):
        """4-channel mode produces (3, 4) from 30 bytes"""
        result = proto.decode_optics(bytes(30), n_channels=4)
        self.assertEqual(result.shape, (3, 4))

    def test_8ch_shape(self):
        """8-channel mode produces (2, 8) from 40 bytes"""
        result = proto.decode_optics(bytes(40), n_channels=8)
        self.assertEqual(result.shape, (2, 8))

    def test_16ch_shape(self):
        """16-channel mode produces (1, 16) from 40 bytes"""
        result = proto.decode_optics(bytes(40), n_channels=16)
        self.assertEqual(result.shape, (1, 16))

    def test_known_20bit_value(self):
        """Test decoding a known 20-bit LSB-first value"""
        # Pack value = 1 at position 0
        data = bytearray(40)
        data[0] = 0x01  # bit0 of first value = 1

        result = proto.decode_optics(data, n_channels=8)
        expected = 1.0 * proto.OPTICS_SCALE
        self.assertAlmostEqual(result[0, 0], expected, places=8)


class TestPayloadParsing(unittest.TestCase):
    """Test full payload parsing with TAG-based subpackets"""

    def test_eeg_only_packet(self):
        """Parse packet with only EEG subpacket"""
        packet = build_tag_packet(proto.TAG_EEG_4CH, bytes(28))
        result = proto.parse_payload(packet)

        self.assertEqual(len(result["EEG"]), 1)
        self.assertEqual(len(result["ACCGYRO"]), 0)
        self.assertEqual(len(result["OPTICS"]), 0)

        eeg = result["EEG"][0]
        self.assertEqual(eeg["type"], "EEG")
        self.assertEqual(eeg["tag"], proto.TAG_EEG_4CH)
        self.assertEqual(eeg["n_channels"], 4)
        self.assertEqual(eeg["n_samples"], 4)
        self.assertEqual(eeg["data"].shape, (4, 4))

    def test_multi_subpacket(self):
        """Parse packet with EEG + ACCGYRO + OPTICS"""
        packet = build_tag_packet(
            proto.TAG_EEG_4CH, bytes(28),
            extra_subpackets=[
                (proto.TAG_ACCGYRO, bytes(36)),
                (proto.TAG_OPTICS_8CH, bytes(40)),
            ]
        )
        result = proto.parse_payload(packet)

        self.assertEqual(len(result["EEG"]), 1)
        self.assertEqual(len(result["ACCGYRO"]), 1)
        self.assertEqual(len(result["OPTICS"]), 1)

    def test_subpacket_counter_and_data_bytes(self):
        """The first byte of each subpacket header is that TAG's counter"""
        eeg_data = bytes(range(28))
        imu_data = bytes(range(36))
        packet = bytearray(build_tag_packet(
            proto.TAG_EEG_4CH, eeg_data,
            extra_subpackets=[(proto.TAG_ACCGYRO, imu_data)],
        ))
        packet[10] = 0x2A  # first subpacket header: bytes 10-13
        packet[14 + 28 + 1] = 0x07  # right after the next TAG
        result = proto.parse_payload(bytes(packet))

        self.assertEqual(result["EEG"][0]["counter"], 0x2A)
        self.assertEqual(result["EEG"][0]["data_bytes"], eeg_data)
        self.assertEqual(result["ACCGYRO"][0]["counter"], 0x07)
        self.assertEqual(result["ACCGYRO"][0]["data_bytes"], imu_data)

    def test_inspect_payload_reports_tags_and_types(self):
        """Payload diagnostics should count TAGs without decoding values."""
        packet = build_tag_packet(
            proto.TAG_EEG_4CH, bytes(28),
            extra_subpackets=[
                (proto.TAG_ACCGYRO, bytes(36)),
                (proto.TAG_OPTICS_8CH, bytes(40)),
            ]
        )

        result = proto.inspect_payload(packet)

        self.assertEqual(result["tags"], [proto.TAG_EEG_4CH, proto.TAG_ACCGYRO, proto.TAG_OPTICS_8CH])
        self.assertEqual(result["decoded_tag_types"], ["EEG", "ACCGYRO", "OPTICS"])
        self.assertEqual(result["unknown_tags"], [])
        self.assertFalse(result["truncated"])

    def test_inspect_payload_reports_unknown_and_truncated(self):
        """Payload diagnostics should explain malformed notifications."""
        unknown = bytearray(14)
        unknown[9] = 0x99

        unknown_result = proto.inspect_payload(bytes(unknown) + bytes(28))
        truncated_result = proto.inspect_payload(build_tag_packet(proto.TAG_EEG_4CH, bytes(4)))

        self.assertEqual(unknown_result["unknown_tags"], [0x99])
        self.assertFalse(unknown_result["truncated"])
        self.assertTrue(truncated_result["truncated"])

    def test_battery_new_firmware_tag_consumes_variable_payload(self):
        """TAG 0x88 telemetry should consume to packet boundary."""
        battery_raw = int(88.5 * 256).to_bytes(2, byteorder="little") + bytes(198)
        packet = bytearray(build_tag_packet(proto.TAG_BATTERY_1, battery_raw))
        packet[0] = len(packet)

        inspection = proto.inspect_payload(bytes(packet))
        parsed = proto.parse_payload(bytes(packet))

        self.assertEqual(inspection["tags"], [proto.TAG_BATTERY_1])
        self.assertEqual(inspection["decoded_tag_types"], ["BATTERY"])
        self.assertEqual(inspection["unknown_tags"], [])
        self.assertFalse(inspection["truncated"])
        self.assertEqual(inspection["bytes_consumed"], len(packet))
        self.assertEqual(len(parsed["BATTERY"]), 1)
        telemetry = parsed["BATTERY"][0]["data"]
        self.assertEqual(telemetry["payload_size"], 200)
        self.assertAlmostEqual(telemetry["battery_percent"], 88.5)

    def test_battery_new_firmware_subpacket_consumes_to_boundary(self):
        """A variable telemetry subpacket should not create trailing fake TAGs."""
        battery_raw = int(93.0 * 256).to_bytes(2, byteorder="little") + bytes(198)
        packet = bytearray(
            build_tag_packet(
                proto.TAG_EEG_4CH,
                bytes(28),
                extra_subpackets=[(proto.TAG_BATTERY_1, battery_raw)],
            )
        )
        packet[0] = len(packet)

        inspection = proto.inspect_payload(bytes(packet))
        parsed = proto.parse_payload(bytes(packet))

        self.assertEqual(inspection["tags"], [proto.TAG_EEG_4CH, proto.TAG_BATTERY_1])
        self.assertEqual(inspection["decoded_tag_types"], ["EEG", "BATTERY"])
        self.assertEqual(inspection["unknown_tags"], [])
        self.assertFalse(inspection["truncated"])
        self.assertEqual(len(parsed["EEG"]), 1)
        self.assertEqual(len(parsed["BATTERY"]), 1)
        self.assertAlmostEqual(parsed["BATTERY"][0]["data"]["battery_percent"], 93.0)

    def test_battery_old_firmware_tag_allows_following_subpacket(self):
        """TAG 0x98 keeps the fixed 20-byte telemetry payload length."""
        battery_raw = int(77.25 * 256).to_bytes(2, byteorder="little") + bytes(18)
        packet = build_tag_packet(
            proto.TAG_BATTERY_2,
            battery_raw,
            extra_subpackets=[(proto.TAG_EEG_4CH, bytes(28))],
        )

        inspection = proto.inspect_payload(packet)
        parsed = proto.parse_payload(packet)

        self.assertEqual(inspection["tags"], [proto.TAG_BATTERY_2, proto.TAG_EEG_4CH])
        self.assertEqual(inspection["decoded_tag_types"], ["BATTERY", "EEG"])
        self.assertEqual(inspection["unknown_tags"], [])
        self.assertFalse(inspection["truncated"])
        self.assertAlmostEqual(parsed["BATTERY"][0]["data"]["battery_percent"], 77.25)
        self.assertEqual(len(parsed["EEG"]), 1)

    def test_drl_ref_tag_is_known_telemetry(self):
        """TAG 0x53 should be counted as known DRL/REF telemetry."""
        packet = build_tag_packet(proto.TAG_DRL_REF, bytes(24))

        inspection = proto.inspect_payload(packet)
        parsed = proto.parse_payload(packet)

        self.assertEqual(inspection["tags"], [proto.TAG_DRL_REF])
        self.assertEqual(inspection["decoded_tag_types"], ["DRLREF"])
        self.assertEqual(inspection["unknown_tags"], [])
        self.assertFalse(inspection["truncated"])
        self.assertEqual(len(parsed["DRLREF"]), 1)
        self.assertEqual(parsed["DRLREF"][0]["data"]["payload_size"], 24)

    def test_short_payload_returns_empty(self):
        """Payload shorter than header returns empty result"""
        result = proto.parse_payload(bytes(10))
        self.assertEqual(len(result["EEG"]), 0)

    def test_unknown_first_tag(self):
        """Unknown first TAG returns empty result"""
        header = bytearray(14)
        header[9] = 0x99  # Unknown TAG
        result = proto.parse_payload(bytes(header) + bytes(28))
        self.assertEqual(len(result["EEG"]), 0)


class TestCommandEncoding(unittest.TestCase):
    """Test command encoding"""

    def test_encode_v6(self):
        """v6 command should encode correctly"""
        result = proto.encode_cmd("v6")
        # Length = len("v6\n") + 1 = 4
        self.assertEqual(result[0], 4)
        self.assertEqual(result[1:3], b"v6")
        self.assertEqual(result[3], ord("\n"))

    def test_encode_dc001(self):
        """dc001 command should encode correctly"""
        result = proto.encode_cmd("dc001")
        self.assertEqual(result[0], 7)  # len("dc001\n") + 1
        self.assertEqual(result[1:6], b"dc001")

    def test_preencoded_commands_match(self):
        """Pre-encoded COMMANDS dict should match encode_cmd output"""
        for name, expected in proto.COMMANDS.items():
            with self.subTest(cmd=name):
                self.assertEqual(proto.encode_cmd(name), expected)


class TestInitSequence(unittest.TestCase):
    """Test init sequence generation"""

    def test_default_sequence_length(self):
        """Default sequence should have correct number of steps"""
        seq = proto.get_init_sequence()
        self.assertEqual(len(seq), 12)

    def test_sequence_has_double_dc001(self):
        """Sequence must send dc001 twice"""
        seq = proto.get_init_sequence()
        dc001_count = sum(1 for desc, cmd, _ in seq if cmd == proto.COMMANDS["dc001"])
        self.assertEqual(dc001_count, 2)

    def test_sequence_has_halt_between(self):
        """There must be a halt between the two dc001 commands"""
        seq = proto.get_init_sequence()
        dc001_indices = [i for i, (_, cmd, _) in enumerate(seq) if cmd == proto.COMMANDS["dc001"]]
        self.assertEqual(len(dc001_indices), 2)

        # Check there's a halt between them
        halt_indices = [i for i, (_, cmd, _) in enumerate(seq) if cmd == proto.COMMANDS["h"]]
        between_halts = [i for i in halt_indices if dc001_indices[0] < i < dc001_indices[1]]
        self.assertGreater(len(between_halts), 0)

    def test_custom_preset(self):
        """Custom preset should appear in the sequence"""
        seq = proto.get_init_sequence("p1035")
        preset_cmds = [cmd for _, cmd, _ in seq if cmd == proto.COMMANDS["p1035"]]
        self.assertGreater(len(preset_cmds), 0)


class TestSensorConfig(unittest.TestCase):
    """Test sensor configuration constants"""

    def test_all_tags_have_config(self):
        """All defined TAGs should have sensor config"""
        tags = [proto.TAG_EEG_4CH, proto.TAG_EEG_8CH, proto.TAG_ACCGYRO,
                proto.TAG_OPTICS_4CH, proto.TAG_OPTICS_8CH, proto.TAG_OPTICS_16CH,
                proto.TAG_DRL_REF, proto.TAG_BATTERY_1, proto.TAG_BATTERY_2]
        for tag in tags:
            self.assertIn(tag, proto.SENSOR_CONFIG)

    def test_eeg_data_lengths(self):
        """EEG data length should be 28 bytes for both modes"""
        self.assertEqual(proto.SENSOR_CONFIG[proto.TAG_EEG_4CH][3], 28)
        self.assertEqual(proto.SENSOR_CONFIG[proto.TAG_EEG_8CH][3], 28)

    def test_accgyro_data_length(self):
        """ACCGYRO data length should be 36 bytes"""
        self.assertEqual(proto.SENSOR_CONFIG[proto.TAG_ACCGYRO][3], 36)


if __name__ == '__main__':
    unittest.main()
