"""Vectorised inverses of the Athena TAG decoders in ``muse_tmr.ble.muse_athena_protocol``.

``muse_tmr.testing.synthetic`` builds its BLE notifications with these. Each encoder
takes values in the decoder's output units and returns the bytes the decoder reads back
within one count of the input: ``decode_eeg_raw_counts`` (14-bit, LSB first),
``decode_accgyro`` (16-bit little endian), ``decode_optics`` (20-bit, LSB first) and
``decode_battery`` for ``TAG_BATTERY_2``. A notification is laid out the way
``parse_payload`` reads it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from muse_tmr.ble import muse_athena_protocol as proto

# Byte 0 of a notification holds its length; the longest in the committed capture
# (tests/test_data) is 244 bytes.
MAX_NOTIFICATION_BYTES = 244
# Bytes 0-8: the length, a running notification count in byte 1, then zeros. The
# decoders read only byte 0.
NOTIFICATION_PREFIX_BYTES = 9
# Every subpacket starts with its TAG, its counter and three bytes the decoders skip.
# For the first subpacket these are bytes 9-13 of the notification.
SUBPACKET_HEADER_BYTES = 5

BATTERY_DATA_BYTES = proto.SENSOR_CONFIG[proto.TAG_BATTERY_2][3]
OPTICS_TAGS = {4: proto.TAG_OPTICS_4CH, 8: proto.TAG_OPTICS_8CH, 16: proto.TAG_OPTICS_16CH}


def pack_lsb_first(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack rows of unsigned ``bits``-wide values LSB first, as ``_unpack_bits_lsb`` reads.

    ``values`` is (n, count); returns (n, count * bits / 8) uint8. Values are packed in
    groups that end on a byte boundary (4 for 14 bits, 2 for 20 bits).
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must be (subpackets, values per subpacket)")
    rows, count = values.shape
    group = next(size for size in range(1, 9) if (size * bits) % 8 == 0)
    if group * bits > 64 or count % group:
        raise ValueError(f"cannot pack {count} values of {bits} bits on byte boundaries")
    if values.size and (values.min() < 0 or values.max() >= (1 << bits)):
        raise ValueError(f"values must fit in {bits} unsigned bits")
    grouped = values.astype(np.uint64).reshape(rows, count // group, group)
    shifts = np.arange(group, dtype=np.uint64) * np.uint64(bits)
    words = np.bitwise_or.reduce(grouped << shifts, axis=2).astype("<u8")
    group_bytes = group * bits // 8
    packed = words.view(np.uint8).reshape(rows, count // group, 8)[:, :, :group_bytes]
    return np.ascontiguousarray(packed).reshape(rows, (count // group) * group_bytes)


def eeg_uv_to_counts(values_uv: np.ndarray) -> np.ndarray:
    """Centered microvolts to unsigned 14-bit counts; values past the rails clip there."""
    counts = np.rint(np.asarray(values_uv, dtype=float) / proto.EEG_SCALE_UV_PER_COUNT)
    counts += proto.EEG_MID_COUNT
    return np.clip(counts, 0, proto.EEG_MAX_COUNT).astype(np.uint16)


def encode_eeg_counts(counts: np.ndarray) -> np.ndarray:
    """(n, samples, channels) 14-bit counts to (n, 28) bytes, for 4 or 8 channels."""
    counts = np.asarray(counts)
    if counts.ndim != 3 or counts.shape[1:] not in ((4, 4), (2, 8)):
        raise ValueError("EEG subpackets are (n, 4, 4) or (n, 2, 8)")
    return pack_lsb_first(counts.reshape(counts.shape[0], 16), proto.EEG_BITS)


def encode_eeg_uv(values_uv: np.ndarray) -> np.ndarray:
    """(n, samples, channels) centered microvolts to (n, 28) bytes."""
    return encode_eeg_counts(eeg_uv_to_counts(values_uv))


def encode_accgyro(accel_g: np.ndarray, gyro_dps: np.ndarray) -> np.ndarray:
    """(n, 3, 3) acceleration in g and (n, 3, 3) rotation in deg/s to (n, 36) bytes."""
    accel = np.asarray(accel_g, dtype=float)
    gyro = np.asarray(gyro_dps, dtype=float)
    if accel.shape != gyro.shape or accel.ndim != 3 or accel.shape[1:] != (3, 3):
        raise ValueError("IMU subpackets are (n, 3 samples, 3 axes) for each sensor")
    raw = np.concatenate(
        [np.rint(accel / proto.ACC_SCALE), np.rint(gyro / proto.GYRO_SCALE)], axis=2
    )
    raw = np.clip(raw, -32768, 32767).astype("<i2")
    return np.ascontiguousarray(raw).view(np.uint8).reshape(raw.shape[0], 36)


def encode_optics(values: np.ndarray) -> np.ndarray:
    """(n, samples, channels) normalised optics values to (n, 30) or (n, 40) bytes.

    The shape picks the mode as ``decode_optics`` reads it: (n, 3, 4), (n, 2, 8) or
    (n, 1, 16).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[1:] not in ((3, 4), (2, 8), (1, 16)):
        raise ValueError("optics subpackets are (n, 3, 4), (n, 2, 8) or (n, 1, 16)")
    raw = np.clip(np.rint(values / proto.OPTICS_SCALE), 0, (1 << 20) - 1)
    return pack_lsb_first(raw.astype(np.uint32).reshape(values.shape[0], -1), 20)


def encode_battery(percent: np.ndarray) -> np.ndarray:
    """(n,) battery percent to (n, 20) ``TAG_BATTERY_2`` bytes: percent * 256 in bytes 0-1."""
    raw = np.clip(np.rint(np.asarray(percent, dtype=float) * 256.0), 0, 0xFFFF).astype("<u2")
    data = np.zeros((raw.shape[0], BATTERY_DATA_BYTES), dtype=np.uint8)
    data[:, :2] = raw.reshape(-1, 1).view(np.uint8)
    return data


def subpacket_records(tag: int, counters: np.ndarray, data: np.ndarray) -> np.ndarray:
    """TAG, counter (mod 256) and three zero bytes in front of each row of ``data``."""
    data = np.asarray(data, dtype=np.uint8)
    expected = proto.SENSOR_CONFIG[tag][3]
    if data.ndim != 2 or data.shape[1] != expected:
        raise ValueError(f"TAG 0x{tag:02x} carries {expected} data bytes")
    records = np.zeros((data.shape[0], SUBPACKET_HEADER_BYTES + expected), dtype=np.uint8)
    records[:, 0] = tag
    records[:, 1] = np.asarray(counters, dtype=np.int64) % proto.COUNTER_MODULO
    records[:, SUBPACKET_HEADER_BYTES:] = data
    return records


def assemble_notification(records: Sequence[bytes], index: int = 0) -> bytes:
    """One notification from subpacket records in order, as ``parse_payload`` reads it.

    The first record's TAG lands on byte 9 and its counter on byte 10, so its data starts
    at byte 14; the others follow with their own five-byte headers. Byte 0 is the length.
    """
    if not records:
        raise ValueError("a notification carries at least one subpacket")
    if records[0][0] not in proto.SENSOR_CONFIG:
        raise ValueError("the first subpacket needs a known TAG")
    body = b"".join(records)
    length = NOTIFICATION_PREFIX_BYTES + len(body)
    if length > MAX_NOTIFICATION_BYTES:
        raise ValueError(f"notification of {length} bytes is longer than {MAX_NOTIFICATION_BYTES}")
    return bytes((length, index % 256, 0, 0, 0, 0, 0, 0, 0)) + body
