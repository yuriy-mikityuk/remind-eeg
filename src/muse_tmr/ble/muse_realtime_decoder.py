"""
Muse Real-time Decoder
On-the-fly decoding of Muse S Athena BLE packets with minimal latency

Provides instant access to sensor values without intermediate storage.
Uses TAG-based subpacket parsing per the Athena protocol.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Deque, Tuple
from dataclasses import dataclass, field
import datetime
import logging
from collections import deque

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from muse_tmr.ble import muse_athena_protocol as proto

logger = logging.getLogger(__name__)

RATE_WINDOW_SECONDS = 10.0


@dataclass
class DecodedData:
    """Container for decoded sensor data"""
    timestamp: datetime.datetime
    packet_type: str
    eeg: Optional[Dict[str, List[float]]] = None
    ppg: Optional[Dict[str, List[float]]] = None
    imu: Optional[Dict[str, List[float]]] = None
    heart_rate: Optional[float] = None
    battery: Optional[float] = None
    raw_bytes: bytes = b''
    # 'eeg'/'ppg'/'imu' -> (index, missing): samples lost right before row `index`
    gaps: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)


class MuseRealtimeDecoder:
    """
    Real-time packet decoder for Muse S Athena data streams

    Uses TAG-based subpacket parsing with correct bit-unpacking:
    - EEG: 14-bit LSB-first
    - Optics/PPG: 20-bit LSB-first
    - IMU: 16-bit little-endian

    Features:
    - Callback-based processing
    - Minimal memory footprint
    - Stream statistics
    """

    def __init__(self):
        """Initialize decoder with default settings"""
        # Callbacks for different data types
        self.callbacks: Dict[str, List[Callable]] = {
            'eeg': [],
            'ppg': [],
            'imu': [],
            'heart_rate': [],
            'any': []  # Called for any packet
        }

        # Statistics
        self.stats = self._initial_stats()
        self._rate_events: Dict[str, Deque[Tuple[datetime.datetime, int]]] = {
            'notifications': deque(),
            'eeg_subpackets': deque(),
            'eeg_sample_rows': deque(),
            'ppg_subpackets': deque(),
            'ppg_sample_rows': deque(),
            'imu_sample_rows': deque(),
        }

        # Buffers for derived metrics
        self.ppg_buffer = []
        self.last_heart_rate = None

        self.sequence = proto.SubpacketSequence()

    def register_callback(self, data_type: str, callback: Callable[[DecodedData], None]):
        """
        Register a callback for specific data type

        Args:
            data_type: 'eeg', 'ppg', 'imu', 'heart_rate', or 'any'
            callback: Function to call with decoded data

        Example:
            decoder.register_callback('eeg', lambda data: print(f"EEG: {data.eeg}"))
        """
        if data_type in self.callbacks:
            self.callbacks[data_type].append(callback)

    @staticmethod
    def _initial_stats() -> Dict[str, Any]:
        return {
            'packets_decoded': 0,
            'eeg_samples': 0,
            'eeg_subpackets': 0,
            'eeg_sample_rows': 0,
            'eeg_values': 0,
            'eeg_first_sample_time': None,
            'eeg_last_sample_time': None,
            'ppg_samples': 0,
            'ppg_subpackets': 0,
            'ppg_sample_rows': 0,
            'ppg_values': 0,
            'ppg_first_sample_time': None,
            'ppg_last_sample_time': None,
            'imu_samples': 0,
            'imu_first_sample_time': None,
            'imu_last_sample_time': None,
            'battery_packets': 0,
            'battery_payload_bytes': 0,
            'battery_last_percent': None,
            'drl_ref_packets': 0,
            'drl_ref_payload_bytes': 0,
            'tag_counts': {},
            'tag_type_counts': {},
            'unknown_tag_counts': {},
            'repeated_subpackets': {},
            'lost_subpackets': {},
            'truncated_notifications': 0,
            'short_notifications': 0,
            'decode_errors': 0,
            'first_packet_time': None,
            'last_packet_time': None,
        }

    def decode(self, data: bytes, timestamp: Optional[datetime.datetime] = None) -> DecodedData:
        """
        Decode a raw BLE packet in real-time

        Args:
            data: Raw packet bytes from SENSOR_UUID notification
            timestamp: Packet timestamp

        Returns:
            DecodedData object with parsed values
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()

        self.stats['packets_decoded'] += 1
        if self.stats['first_packet_time'] is None:
            self.stats['first_packet_time'] = timestamp
        self.stats['last_packet_time'] = timestamp
        self._record_rate_event('notifications', timestamp, 1)

        if not data:
            self._record_payload_inspection(proto.inspect_payload(data))
            return DecodedData(timestamp=timestamp, packet_type='EMPTY', raw_bytes=data)

        decoded = DecodedData(
            timestamp=timestamp,
            packet_type='SENSOR',
            raw_bytes=data
        )

        try:
            self._record_payload_inspection(proto.inspect_payload(data))
            parsed = proto.parse_payload(data)
            self._drop_repeats(parsed)
            self._populate_decoded(parsed, decoded)
        except Exception as e:
            self.stats['decode_errors'] += 1
            logger.debug("Decode error: %s", e)

        # Trigger callbacks
        self._trigger_callbacks(decoded)

        return decoded

    def _drop_repeats(self, parsed: Dict[str, list]):
        """Drop repeated subpackets and note how many were lost before each kept one."""
        for sensor_type, subpackets in parsed.items():
            kept = []
            for subpacket in subpackets:
                lost = self.sequence.observe(
                    subpacket["tag"], subpacket["counter"], subpacket["data_bytes"]
                )
                if lost is None:
                    self._count_subpackets('repeated_subpackets', sensor_type, 1)
                    continue
                if lost:
                    self._count_subpackets('lost_subpackets', sensor_type, lost)
                subpacket["lost_before"] = lost
                kept.append(subpacket)
            subpackets[:] = kept

    def _count_subpackets(self, key: str, sensor_type: str, count: int):
        counts = self.stats[key]
        counts[sensor_type] = counts.get(sensor_type, 0) + count

    @staticmethod
    def _mark_gap(decoded: DecodedData, modality: str, index: int, subpacket: dict):
        missing = subpacket.get("lost_before", 0) * subpacket["n_samples"]
        if missing:
            decoded.gaps.setdefault(modality, []).append((index, missing))

    def _populate_decoded(self, parsed: Dict[str, list], decoded: DecodedData):
        """Populate DecodedData from parsed subpackets."""

        # EEG
        if parsed["EEG"]:
            decoded.eeg = {}
            for subpacket in parsed["EEG"]:
                arr = subpacket["data"]  # shape (n_samples, n_channels)
                n_channels = subpacket["n_channels"]
                sample_rows = arr.shape[0]
                values = sample_rows * n_channels
                self.stats['eeg_subpackets'] += 1
                self.stats['eeg_sample_rows'] += sample_rows
                self.stats['eeg_values'] += values
                self.stats['eeg_samples'] += values
                self._record_modality_time('eeg', decoded.timestamp)
                self._record_rate_event('eeg_subpackets', decoded.timestamp, 1)
                self._record_rate_event('eeg_sample_rows', decoded.timestamp, sample_rows)
                if n_channels == 4:
                    names = proto.EEG_CHANNELS_4
                else:
                    names = proto.EEG_CHANNELS_8
                self._mark_gap(decoded, 'eeg', len(decoded.eeg.get(names[0], ())), subpacket)
                for ch_idx in range(n_channels):
                    ch_name = names[ch_idx] if ch_idx < len(names) else f"ch{ch_idx}"
                    decoded.eeg.setdefault(ch_name, []).extend(arr[:, ch_idx].tolist())
            decoded.packet_type = 'EEG'

        # ACCGYRO (IMU)
        if parsed["ACCGYRO"]:
            decoded.imu = {'accel': [], 'gyro': []}
            for subpacket in parsed["ACCGYRO"]:
                arr = subpacket["data"]  # shape (3, 6)
                self._mark_gap(decoded, 'imu', len(decoded.imu['accel']), subpacket)
                decoded.imu['accel'].extend(arr[:, 0:3].tolist())
                decoded.imu['gyro'].extend(arr[:, 3:6].tolist())
                self.stats['imu_samples'] += arr.shape[0]
                self._record_modality_time('imu', decoded.timestamp)
                self._record_rate_event('imu_sample_rows', decoded.timestamp, arr.shape[0])
            if not parsed["EEG"]:
                decoded.packet_type = 'IMU'

        # Optics (PPG)
        if parsed["OPTICS"]:
            decoded.ppg = {}
            for subpacket in parsed["OPTICS"]:
                arr = subpacket["data"]  # shape (n_samples, n_channels)
                n_channels = subpacket["n_channels"]
                sample_rows = arr.shape[0]
                self.stats['ppg_subpackets'] += 1
                self.stats['ppg_sample_rows'] += sample_rows
                self.stats['ppg_values'] += sample_rows * n_channels
                self._record_modality_time('ppg', decoded.timestamp)
                self._record_rate_event('ppg_subpackets', decoded.timestamp, 1)
                self._record_rate_event('ppg_sample_rows', decoded.timestamp, sample_rows)
                if n_channels == 8:
                    names = proto.OPTICS_CHANNELS_8
                else:
                    names = [f"opt{i}" for i in range(n_channels)]
                self._mark_gap(decoded, 'ppg', len(decoded.ppg.get(names[0], ())), subpacket)
                for ch_idx in range(n_channels):
                    ch_name = names[ch_idx] if ch_idx < len(names) else f"opt{ch_idx}"
                    decoded.ppg.setdefault(ch_name, []).extend(arr[:, ch_idx].tolist())
                self.stats['ppg_samples'] += sample_rows

                # Update heart rate buffer using IR channel (index 0)
                ir_samples = arr[:, 0].tolist()
                self.ppg_buffer.extend(ir_samples)
                if len(self.ppg_buffer) > 128:  # 2 seconds at 64Hz
                    self._calculate_heart_rate(decoded)
                    if len(self.ppg_buffer) > 320:  # Keep max 5 seconds
                        self.ppg_buffer = self.ppg_buffer[-320:]

            if decoded.packet_type == 'SENSOR':
                decoded.packet_type = 'OPTICS'

        # Battery
        if parsed["BATTERY"]:
            for subpacket in parsed["BATTERY"]:
                data = subpacket["data"]
                self.stats['battery_packets'] += 1
                self.stats['battery_payload_bytes'] += int(data.get("payload_size", 0))
                percent = data.get("battery_percent")
                if percent is not None:
                    decoded.battery = float(percent)
                    self.stats['battery_last_percent'] = decoded.battery
            decoded.packet_type = 'BATTERY'

        # DRL/REF telemetry
        if parsed["DRLREF"]:
            for subpacket in parsed["DRLREF"]:
                data = subpacket["data"]
                self.stats['drl_ref_packets'] += 1
                self.stats['drl_ref_payload_bytes'] += int(data.get("payload_size", 0))

        # Combined packet type
        if parsed["EEG"] and (parsed["OPTICS"] or parsed["ACCGYRO"]):
            decoded.packet_type = 'MULTI'

    def _calculate_heart_rate(self, decoded: DecodedData):
        """Calculate heart rate from PPG buffer"""
        if len(self.ppg_buffer) < 128:  # Need at least 2 seconds
            return

        try:
            signal = np.array(
                self.ppg_buffer[-640:] if len(self.ppg_buffer) > 640
                else self.ppg_buffer
            )

            # Detrend
            signal = signal - np.mean(signal)

            if not SCIPY_AVAILABLE:
                return
            peaks, _ = find_peaks(signal, distance=40, prominence=np.std(signal) * 0.3)

            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / 64.0  # 64 Hz sampling
                heart_rate = 60.0 / np.mean(peak_intervals)

                if 40 < heart_rate < 200:  # Physiological range
                    decoded.heart_rate = heart_rate
                    self.last_heart_rate = heart_rate
                    logger.debug("Calculated HR: %.1f BPM", heart_rate)
        except Exception:
            pass

    def _trigger_callbacks(self, decoded: DecodedData):
        """Trigger registered callbacks"""
        if decoded.eeg and self.callbacks['eeg']:
            for callback in self.callbacks['eeg']:
                callback(decoded)

        if decoded.ppg and self.callbacks['ppg']:
            for callback in self.callbacks['ppg']:
                callback(decoded)

        if decoded.imu and self.callbacks['imu']:
            for callback in self.callbacks['imu']:
                callback(decoded)

        if decoded.heart_rate and self.callbacks['heart_rate']:
            for callback in self.callbacks['heart_rate']:
                callback(decoded)

        for callback in self.callbacks['any']:
            callback(decoded)

    def get_stats(self) -> Dict[str, Any]:
        """Get decoder statistics"""
        elapsed_seconds = self._stats_elapsed_seconds()
        eeg_elapsed_seconds = self._modality_elapsed_seconds('eeg')
        ppg_elapsed_seconds = self._modality_elapsed_seconds('ppg')
        imu_elapsed_seconds = self._modality_elapsed_seconds('imu')
        return {
            'packets_decoded': self.stats['packets_decoded'],
            'notifications_per_second': self._effective_rate(
                self.stats['packets_decoded'],
                elapsed_seconds,
            ),
            'rolling_notifications_per_second': self._rolling_rate('notifications'),
            'eeg_samples': self.stats['eeg_samples'],
            'eeg_subpackets': self.stats['eeg_subpackets'],
            'eeg_sample_rows': self.stats['eeg_sample_rows'],
            'eeg_values': self.stats['eeg_values'],
            'eeg_effective_sample_rate_hz': self._effective_rate(
                self.stats['eeg_sample_rows'],
                eeg_elapsed_seconds,
            ),
            'eeg_rolling_sample_rate_hz': self._rolling_rate('eeg_sample_rows'),
            'eeg_subpackets_per_second': self._effective_rate(
                self.stats['eeg_subpackets'],
                eeg_elapsed_seconds,
            ),
            'eeg_rolling_subpackets_per_second': self._rolling_rate('eeg_subpackets'),
            'ppg_samples': self.stats['ppg_samples'],
            'ppg_subpackets': self.stats['ppg_subpackets'],
            'ppg_sample_rows': self.stats['ppg_sample_rows'],
            'ppg_values': self.stats['ppg_values'],
            'ppg_effective_sample_rate_hz': self._effective_rate(
                self.stats['ppg_sample_rows'],
                ppg_elapsed_seconds,
            ),
            'ppg_rolling_sample_rate_hz': self._rolling_rate('ppg_sample_rows'),
            'ppg_subpackets_per_second': self._effective_rate(
                self.stats['ppg_subpackets'],
                ppg_elapsed_seconds,
            ),
            'ppg_rolling_subpackets_per_second': self._rolling_rate('ppg_subpackets'),
            'imu_samples': self.stats['imu_samples'],
            'imu_effective_sample_rate_hz': self._effective_rate(
                self.stats['imu_samples'],
                imu_elapsed_seconds,
            ),
            'imu_rolling_sample_rate_hz': self._rolling_rate('imu_sample_rows'),
            'battery_packets': self.stats['battery_packets'],
            'battery_payload_bytes': self.stats['battery_payload_bytes'],
            'battery_last_percent': self.stats['battery_last_percent'],
            'drl_ref_packets': self.stats['drl_ref_packets'],
            'drl_ref_payload_bytes': self.stats['drl_ref_payload_bytes'],
            'tag_counts': dict(self.stats['tag_counts']),
            'tag_type_counts': dict(self.stats['tag_type_counts']),
            'unknown_tag_counts': dict(self.stats['unknown_tag_counts']),
            'repeated_subpackets': dict(self.stats['repeated_subpackets']),
            'lost_subpackets': dict(self.stats['lost_subpackets']),
            'truncated_notifications': self.stats['truncated_notifications'],
            'short_notifications': self.stats['short_notifications'],
            'rate_window_seconds': RATE_WINDOW_SECONDS,
            'decode_errors': self.stats['decode_errors'],
            'error_rate': self.stats['decode_errors'] / max(1, self.stats['packets_decoded']),
            'last_heart_rate': self.last_heart_rate,
            'first_packet': self.stats['first_packet_time'],
            'last_packet': self.stats['last_packet_time']
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = self._initial_stats()
        for events in self._rate_events.values():
            events.clear()

    def _stats_elapsed_seconds(self) -> Optional[float]:
        first = self.stats['first_packet_time']
        last = self.stats['last_packet_time']
        if first is None or last is None:
            return None
        try:
            elapsed = (last - first).total_seconds()
        except AttributeError:
            return None
        return elapsed if elapsed > 0 else None

    @staticmethod
    def _effective_rate(samples: int, elapsed_seconds: Optional[float]) -> Optional[float]:
        if elapsed_seconds is None:
            return None
        return samples / elapsed_seconds

    def _record_payload_inspection(self, inspection: Dict[str, object]) -> None:
        for tag in inspection.get('tags', ()):
            tag_key = _tag_key(int(tag))
            self.stats['tag_counts'][tag_key] = self.stats['tag_counts'].get(tag_key, 0) + 1
        for tag_type in inspection.get('decoded_tag_types', ()):
            tag_type = str(tag_type)
            self.stats['tag_type_counts'][tag_type] = self.stats['tag_type_counts'].get(tag_type, 0) + 1
        for tag in inspection.get('unknown_tags', ()):
            tag_key = _tag_key(int(tag))
            self.stats['unknown_tag_counts'][tag_key] = (
                self.stats['unknown_tag_counts'].get(tag_key, 0) + 1
            )
        if inspection.get('truncated'):
            self.stats['truncated_notifications'] += 1
        if inspection.get('short_payload'):
            self.stats['short_notifications'] += 1

    def _record_modality_time(self, modality: str, timestamp: datetime.datetime) -> None:
        first_key = f'{modality}_first_sample_time'
        last_key = f'{modality}_last_sample_time'
        if self.stats[first_key] is None:
            self.stats[first_key] = timestamp
        self.stats[last_key] = timestamp

    def _modality_elapsed_seconds(self, modality: str) -> Optional[float]:
        first = self.stats[f'{modality}_first_sample_time']
        last = self.stats[f'{modality}_last_sample_time']
        if first is None or last is None:
            return None
        try:
            elapsed = (last - first).total_seconds()
        except AttributeError:
            return None
        return elapsed if elapsed > 0 else None

    def _record_rate_event(self, metric: str, timestamp: datetime.datetime, value: int) -> None:
        events = self._rate_events[metric]
        events.append((timestamp, int(value)))
        cutoff = timestamp - datetime.timedelta(seconds=RATE_WINDOW_SECONDS)
        while events and events[0][0] < cutoff:
            events.popleft()

    def _rolling_rate(self, metric: str) -> Optional[float]:
        events = self._rate_events[metric]
        if len(events) < 2:
            return None
        elapsed = (events[-1][0] - events[0][0]).total_seconds()
        if elapsed <= 0:
            return None
        return sum(value for _, value in list(events)[1:]) / elapsed


def _tag_key(tag: int) -> str:
    return f"0x{tag:02x}"


# Example real-time processing
def example_realtime_processing():
    """Example of real-time packet processing"""

    print("Real-time Decoder Example")
    print("=" * 60)

    decoder = MuseRealtimeDecoder()

    def on_eeg(data: DecodedData):
        first_channel = next(iter(data.eeg.keys()))
        print(f"EEG: {len(data.eeg)} channels, {first_channel}: {data.eeg[first_channel][0]:.1f} uV")

    def on_heart_rate(data: DecodedData):
        print(f"Heart Rate: {data.heart_rate:.0f} BPM")

    def on_imu(data: DecodedData):
        print(f"IMU: Accel={data.imu['accel']}, Gyro={data.imu['gyro']}")

    decoder.register_callback('eeg', on_eeg)
    decoder.register_callback('heart_rate', on_heart_rate)
    decoder.register_callback('imu', on_imu)

    # Build a synthetic TAG-based test packet:
    # 14-byte header (byte 9 = TAG_EEG_4CH = 0x11) + 28 bytes EEG data
    header = bytearray(14)
    header[9] = proto.TAG_EEG_4CH
    eeg_data = bytes(28)  # zeros = all channels at negative ADC midscale
    test_packet = bytes(header) + eeg_data

    decoded = decoder.decode(test_packet)
    print(f"Decoded: {decoded.packet_type}")

    stats = decoder.get_stats()
    print(f"\nStatistics:")
    print(f"  Packets: {stats['packets_decoded']}")
    print(f"  EEG samples: {stats['eeg_samples']}")
    print(f"  Error rate: {stats['error_rate']:.1%}")


if __name__ == "__main__":
    example_realtime_processing()
