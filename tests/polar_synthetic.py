"""Synthetic Polar H10 signals and byte frames for tests (no device data)."""

import math
import struct

import numpy as np

ECG_FS = 130.0
ACC_FS = 50.0
POLAR_EPOCH_OFFSET_S = 946684800


def beat_times(seconds, rng, mean_rr_s=1.0, breathing_hz=0.25, rsa_s=0.04, jitter_s=0.01):
    times, t = [], 1.0
    while t < seconds - 1.0:
        times.append(t)
        rr = mean_rr_s + rsa_s * math.sin(2 * math.pi * breathing_hz * t) + rng.normal(0, jitter_s)
        t += rr
    return np.asarray(times)


def ecg_signal(beats_s, seconds, rng, breathing_hz=0.25, amplitude_mod=0.1, noise_uv=15.0):
    t = np.arange(int(seconds * ECG_FS)) / ECG_FS
    ecg = 100.0 * np.sin(2 * math.pi * 0.2 * t) + rng.normal(0, noise_uv, t.size)
    for beat in beats_s:
        scale = 1.0 + amplitude_mod * math.sin(2 * math.pi * breathing_hz * beat)
        for offset, height, width in ((-0.025, -150.0, 0.008), (0.0, 1200.0, 0.009), (0.025, -300.0, 0.009), (0.25, 300.0, 0.045)):
            ecg += scale * height * np.exp(-0.5 * ((t - beat - offset) / width) ** 2)
    return t, ecg


def chest_acc(seconds, rng, breaths_per_min, amplitude_mg=6.0, noise_mg=1.5):
    """breaths_per_min may be a number or a function of time (seconds)."""
    t = np.arange(int(seconds * ACC_FS)) / ACC_FS
    if callable(breaths_per_min):
        rate_hz = np.array([breaths_per_min(value) for value in t]) / 60.0
        phase = 2 * math.pi * np.cumsum(rate_hz) / ACC_FS
    else:
        phase = 2 * math.pi * breaths_per_min / 60.0 * t
    breathing = amplitude_mg * (np.sin(phase) + 0.3 * np.sin(2 * phase + 0.5))
    direction = np.array([0.3, 0.2, 0.93])
    gravity = np.array([120.0, -60.0, 990.0])
    xyz = gravity + np.outer(breathing, direction) + rng.normal(0, noise_mg, (t.size, 3))
    return t, xyz


def hr_measurement(hr, rr_ms=(), uint16=False, contact=None, energy=None):
    flags = 0x01 if uint16 else 0x00
    if contact is not None:
        flags |= 0x04 | (0x02 if contact else 0x00)
    if energy is not None:
        flags |= 0x08
    if rr_ms:
        flags |= 0x10
    out = bytes([flags]) + (struct.pack("<H", hr) if uint16 else bytes([hr]))
    if energy is not None:
        out += struct.pack("<H", energy)
    for rr in rr_ms:
        out += struct.pack("<H", int(round(rr * 1024 / 1000)))
    return out


def pmd_header(measurement_type, timestamp_ns, frame_type, compressed):
    return bytes([measurement_type]) + int(timestamp_ns).to_bytes(8, "little") + bytes([frame_type | (0x80 if compressed else 0)])


def ecg_frame(timestamp_ns, samples_uv):
    body = b"".join(int(round(v)).to_bytes(3, "little", signed=True) for v in samples_uv)
    return pmd_header(0, timestamp_ns, 0, False) + body


def acc_frame_uncompressed(timestamp_ns, samples, frame_type=1):
    width = frame_type + 1
    body = b"".join(int(v).to_bytes(width, "little", signed=True) for sample in samples for v in sample)
    return pmd_header(2, timestamp_ns, frame_type, False) + body


def acc_frame_delta(timestamp_ns, samples, resolution=16, block=8):
    """Encode per the Polar delta format: reference sample, then [width][count] bit blocks."""
    width_bytes = math.ceil(resolution / 8)
    channels = len(samples[0])
    body = b"".join(int(v).to_bytes(width_bytes, "little", signed=True) for v in samples[0])
    previous = samples[0]
    rest = samples[1:]
    for start in range(0, len(rest), block):
        chunk = rest[start : start + block]
        deltas = []
        for sample in chunk:
            deltas.append([sample[c] - previous[c] for c in range(channels)])
            previous = sample
        bit_width = max(2, max(abs(d) for row in deltas for d in row).bit_length() + 1)
        bits, position = 0, 0
        for row in deltas:
            for d in row:
                bits |= (d & ((1 << bit_width) - 1)) << position
                position += bit_width
        body += bytes([bit_width, len(chunk)]) + bits.to_bytes(math.ceil(position / 8), "little")
    return pmd_header(2, timestamp_ns, 1, True) + body


def write_raw_session(
    session_dir, seconds, rng, *, breaths_per_min=12.0, drift_ppm=50.0, with_ecg=True, reset_at_s=None, gap_s=5.0,
    wall0=1_790_000_000.0, acc_breaths_per_min=None, no_contact=None, report_contact=True, beats_s=None,
):
    """Write polar/raw_notifications.jsonl (+ clock anchors) for a synthetic session.

    True time is host wall-clock. The sensor clock runs from the Polar default
    2019 epoch with a drift; every notification arrives after a random positive
    BLE delay. During no_contact=(start_s, end_s) the HR notifications say the
    electrodes are off the skin and carry random RR, and the ECG swings rail to
    rail with negative spikes; with report_contact=False the notifications keep
    saying contact, as when the battery comes out before the H10 notices.
    reset_at_s may be one time or several: the sensor clock restarts at each,
    after gap_s without link. beats_s gives the R waves, seconds from wall0, for
    a session that shares its beats with another recording; by default they are
    drawn around 60 bpm. Returns the ground truth.
    """
    import base64
    import json
    from pathlib import Path

    from muse_tmr.sources.polar_h10 import PMD_CONTROL_POINT, PMD_DATA, HEART_RATE_MEASUREMENT, start_command

    polar_dir = Path(session_dir) / "polar"
    polar_dir.mkdir(parents=True, exist_ok=True)
    mono0 = 5_000.0
    sensor0_ns = 599_616_000_000_000_000  # H10 default time after a reset
    drift = drift_ppm * 1e-6
    breathing_hz = breaths_per_min / 60.0

    if beats_s is None:
        beats = beat_times(seconds, rng, breathing_hz=breathing_hz)
    else:
        beats = np.asarray(beats_s, dtype=float)
        beats = beats[(beats >= 1.0) & (beats < seconds - 1.0)]
    t_ecg, ecg = ecg_signal(beats, seconds, rng, breathing_hz=breathing_hz)
    if no_contact is not None:
        off = np.flatnonzero((t_ecg >= no_contact[0]) & (t_ecg < no_contact[1]))
        ecg[off] = np.clip(rng.normal(0.0, 9000.0, off.size), -19630.0, 19630.0)
        ecg[off[::30]] = -19630.0
    t_acc, xyz = chest_acc(seconds, rng, acc_breaths_per_min or breaths_per_min)

    resets = () if reset_at_s is None else tuple(np.atleast_1d(reset_at_s).tolist())

    def sensor_ns(true_s):
        # After a power-down the H10 clock restarts from its default time.
        since = [reset for reset in resets if true_s >= reset]
        if since:
            true_s = true_s - max(since)
        return int(sensor0_ns + true_s * (1.0 + drift) * 1e9)

    def link_down(true_s):
        return any(reset - gap_s <= true_s < reset for reset in resets)

    records = []

    def add(true_s, direction, uuid, payload, char):
        if link_down(true_s):
            return
        delay = 0.0 if direction == "tx" else 0.004 + rng.exponential(0.03)
        arrival = true_s + delay
        records.append(
            {"dir": direction, "char": char, "uuid": uuid, "wall": wall0 + arrival, "mono": mono0 + arrival,
             "b64": base64.b64encode(payload).decode("ascii")}
        )

    factor = (0x05, 0x01) + tuple(struct.pack("<f", 1.0))
    if with_ecg:
        add(0.0, "tx", PMD_CONTROL_POINT, start_command(0, {0: 130, 1: 14}), "pmd_cp")
        add(0.0, "rx", PMD_CONTROL_POINT, bytes([0xF0, 0x02, 0x00, 0x00, 0x00]) + bytes(factor), "pmd_cp")
    add(0.0, "tx", PMD_CONTROL_POINT, start_command(2, {0: 50, 1: 16, 2: 2}), "pmd_cp")
    add(0.0, "rx", PMD_CONTROL_POINT, bytes([0xF0, 0x02, 0x02, 0x00, 0x00]) + bytes(factor), "pmd_cp")

    if with_ecg:
        for start in range(0, t_ecg.size - 73, 73):
            last = start + 72
            add(t_ecg[last], "rx", PMD_DATA, ecg_frame(sensor_ns(t_ecg[last]), ecg[start : last + 1]), "pmd_data")
    for start in range(0, t_acc.size - 36, 36):
        last = start + 35
        add(t_acc[last], "rx", PMD_DATA, acc_frame_delta(sensor_ns(t_acc[last]), [tuple(int(round(v)) for v in row) for row in xyz[start : last + 1]]), "pmd_data")

    rr_true = np.diff(beats) * 1000.0
    second = 1.0
    index = 1
    while second < seconds:
        batch = []
        while index < beats.size and beats[index] <= second:
            batch.append(rr_true[index - 1])
            index += 1
        off = no_contact is not None and no_contact[0] <= second < no_contact[1]
        if off:
            batch = [float(rng.uniform(300.0, 2000.0))]
        add(second, "rx", HEART_RATE_MEASUREMENT, hr_measurement(60, batch, contact=not off or not report_contact), "hr")
        second += 1.0

    records.sort(key=lambda record: record["mono"])
    with (polar_dir / "raw_notifications.jsonl").open("w") as handle:
        for seq, record in enumerate(records):
            handle.write(json.dumps({"seq": seq, **record}) + "\n")
    with (polar_dir / "clock_anchors.jsonl").open("w") as handle:
        for t in np.arange(0.0, seconds + 1, 60.0):
            handle.write(json.dumps({"label": "periodic", "wall": wall0 + t, "mono": mono0 + t}) + "\n")
    return {"wall0": wall0, "beats_wall": wall0 + beats, "rr_ms": rr_true, "breaths_per_min": breaths_per_min}
