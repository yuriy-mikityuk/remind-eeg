"""Continuous sample arrays from a stream of MuseFrames (#230).

A frame carries a few samples per channel and one timestamp, the host time its BLE
notification arrived. ``reconstruct_continuous()`` lays the samples of each stream end
to end in one float32 array per channel at the stream's nominal rate (EEG 256 Hz, IMU
52 Hz, optics 64 Hz) and keeps what the arrays lost:

- A frame more than ``gap_threshold_seconds`` after the one before starts after a
  dropout. ``zero_fill`` inserts ``round(gap * rate) - samples of the frame before``
  zeros, so the array index stays on host time across it; ``discontinuous`` inserts
  nothing and starts a new segment.
- ``EEGSample.gaps`` and its IMU and optics twins name samples the device sent that
  never arrived (#182). With ``fill_lost_samples`` (the default) they become zeros at
  their place, so the index follows the device's sample counter; without it the rows
  stay joined, as ``scripts/validate_rem_detection.py`` has always read them. With it a
  dropout also takes its size from the counter when the frame after it reports a skip
  at its start that agrees with host time to within a quarter second: host time is a
  sample or two off after each dropout of a synthetic night, the counter exact.

Each stream's sample 0 is its first frame's timestamp, the convention of the REM check's
``_collect()``: the samples of a frame are taken to start at its timestamp, about one
notification (12 EEG samples, 47 ms) after they were acquired. ``TimeMap`` maps host
time to the stream's own sample clock (index over nominal rate) through the frames'
timestamps, so an event logged at host time lands at the sample that arrived then, also
on a headband whose clock runs faster than nominal (257.2 Hz on the one session
measured, #160). ``measured_rate_hz`` is samples over host seconds, per night.

Nothing here imports ``muse_tmr.audio``, ``muse_tmr.protocol`` or ``muse_tmr.models``.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from muse_tmr.data.sample_types import MuseFrame

GAP_POLICIES = ("zero_fill", "discontinuous")
STREAMS = ("eeg", "accel", "gyro", "optics")
DROPOUT = "dropout"
LOST_SAMPLES = "lost_samples"
ACCEL_CHANNELS = ("ACC_X", "ACC_Y", "ACC_Z")
GYRO_CHANNELS = ("GYRO_X", "GYRO_Y", "GYRO_Z")
_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class ContinuousConfig:
    """How frames become arrays. ``channels`` and ``reference_channel`` are for EEG.

    ``channels`` None takes the channels of the first EEG frame in their order. A frame
    counts only when it has samples on ``reference_channel`` (None: the first channel),
    and that channel's sample count is the frame's width: another channel with fewer
    samples is padded with zeros, one with more is cut. ``streams`` names what to build
    besides EEG: ``accel``, ``gyro`` and ``optics``.
    """

    gap_policy: str = "zero_fill"
    gap_threshold_seconds: float = 0.5
    nominal_rate_hz: float = 256.0
    channels: Optional[Tuple[str, ...]] = None
    reference_channel: Optional[str] = None
    fill_lost_samples: bool = True
    streams: Tuple[str, ...] = ("eeg",)
    imu_rate_hz: float = 52.0
    optics_rate_hz: float = 64.0

    def validate(self) -> None:
        if self.gap_policy not in GAP_POLICIES:
            raise ValueError(f"gap_policy must be one of: {', '.join(GAP_POLICIES)}")
        if self.gap_threshold_seconds <= 0:
            raise ValueError("gap_threshold_seconds must be positive")
        for name in ("nominal_rate_hz", "imu_rate_hz", "optics_rate_hz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        unknown = set(self.streams) - set(STREAMS)
        if unknown:
            raise ValueError(f"unknown streams {sorted(unknown)}; known: {', '.join(STREAMS)}")
        if self.channels is not None and not self.channels:
            raise ValueError("channels must name at least one channel")
        if (
            self.channels is not None
            and self.reference_channel is not None
            and self.reference_channel not in self.channels
        ):
            raise ValueError("reference_channel must be one of channels")

    def rate_of(self, stream: str) -> float:
        if stream == "eeg":
            return float(self.nominal_rate_hz)
        if stream in ("accel", "gyro"):
            return float(self.imu_rate_hz)
        return float(self.optics_rate_hz)

    def to_dict(self) -> Dict[str, object]:
        return {
            "gap_policy": self.gap_policy,
            "gap_threshold_seconds": self.gap_threshold_seconds,
            "nominal_rate_hz": self.nominal_rate_hz,
            "fill_lost_samples": self.fill_lost_samples,
            "imu_rate_hz": self.imu_rate_hz,
            "optics_rate_hz": self.optics_rate_hz,
        }


@dataclass(frozen=True)
class GapMarker:
    """Samples a stream lost.

    ``kind`` is ``dropout`` (a frame came more than the threshold after the one before)
    or ``lost_samples`` (``gaps`` of a frame). ``index`` is where they are in the
    arrays: the first inserted zero, or the join when nothing was inserted. ``samples``
    is how many are missing, by host time for a dropout and by the device's counter
    otherwise; ``filled`` says whether zeros stand in for them. ``time`` is the host
    time of the frame after the loss and ``seconds`` the host time it covers.
    """

    kind: str
    index: int
    samples: int
    seconds: float
    time: float
    filled: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "index": self.index,
            "samples": self.samples,
            "seconds": self.seconds,
            "time": self.time,
            "filled": self.filled,
        }


@dataclass(frozen=True)
class ContinuousSegment:
    """Samples ``start_index`` to ``stop_index`` (exclusive) without a dropout between
    them; ``start_time`` is the host time of their first frame."""

    start_index: int
    stop_index: int
    start_time: float

    @property
    def samples(self) -> int:
        return self.stop_index - self.start_index


@dataclass(frozen=True)
class TimeMap:
    """Host time (Unix seconds) to a stream's sample clock and back.

    ``anchor_times`` are frame timestamps, strictly increasing, and ``anchor_indices``
    the array index of each frame's first sample. Between anchors the map is linear;
    before the first and after the last it runs at ``rate_hz``.
    """

    anchor_times: np.ndarray
    anchor_indices: np.ndarray
    rate_hz: float

    def index(self, unix_time):
        """The fractional array index at ``unix_time`` (a number or an array)."""
        times = np.asarray(unix_time, dtype=float)
        if self.anchor_times.size == 0:
            raise ValueError("the stream has no frames to map time with")
        first_t, last_t = self.anchor_times[0], self.anchor_times[-1]
        first_i, last_i = self.anchor_indices[0], self.anchor_indices[-1]
        result = np.interp(times, self.anchor_times, self.anchor_indices)
        result = np.where(times < first_t, first_i + (times - first_t) * self.rate_hz, result)
        result = np.where(times > last_t, last_i + (times - last_t) * self.rate_hz, result)
        return float(result) if result.ndim == 0 else result

    def seconds(self, unix_time):
        """Seconds on the sample clock (index over rate) at ``unix_time``."""
        index = self.index(unix_time)
        return index / self.rate_hz

    def unix_time(self, seconds):
        """The host time at ``seconds`` on the sample clock; the inverse of ``seconds()``."""
        indices = np.asarray(seconds, dtype=float) * self.rate_hz
        if self.anchor_times.size == 0:
            raise ValueError("the stream has no frames to map time with")
        first_t, last_t = self.anchor_times[0], self.anchor_times[-1]
        first_i, last_i = self.anchor_indices[0], self.anchor_indices[-1]
        result = np.interp(indices, self.anchor_indices, self.anchor_times)
        result = np.where(indices < first_i, first_t + (indices - first_i) / self.rate_hz, result)
        result = np.where(indices > last_i, last_t + (indices - last_i) / self.rate_hz, result)
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class ContinuousStream:
    """One stream's arrays (float32, equal length) and what they lost."""

    name: str
    channels: Mapping[str, np.ndarray]
    nominal_rate_hz: float
    t0: float
    t_last: float
    gap_policy: str
    segments: Tuple[ContinuousSegment, ...]
    gap_markers: Tuple[GapMarker, ...]
    time_map: TimeMap
    frame_count: int
    received_samples: int
    sample_count: int
    unfilled_dropout_samples: int = 0

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return tuple(self.channels)

    @property
    def n_samples(self) -> int:
        """Samples per channel, counted also when the arrays were not kept."""
        return self.sample_count

    @property
    def measured_rate_hz(self) -> float:
        """Samples per host second from the first frame to the last.

        It counts the zeros of a dropout as samples, and under ``discontinuous`` the
        samples a dropout would have taken, so both policies give the same number. With
        ``fill_lost_samples`` off the lost samples are not counted: then it is the
        ``sfreq`` the REM check has always reported.
        """
        span = self.t_last - self.t0
        if span <= 0:
            return float("nan")
        return (self.n_samples + self.unfilled_dropout_samples) / span

    def dropouts(self) -> Tuple[GapMarker, ...]:
        return tuple(marker for marker in self.gap_markers if marker.kind == DROPOUT)

    def lost_sample_markers(self) -> Tuple[GapMarker, ...]:
        return tuple(marker for marker in self.gap_markers if marker.kind == LOST_SAMPLES)

    def summary(self) -> Dict[str, object]:
        dropouts = self.dropouts()
        lost = self.lost_sample_markers()
        return {
            "stream": self.name,
            "channels": list(self.channel_names),
            "nominal_rate_hz": self.nominal_rate_hz,
            "measured_rate_hz": self.measured_rate_hz,
            "samples": self.n_samples,
            "received_samples": self.received_samples,
            "frames": self.frame_count,
            "segments": len(self.segments),
            "dropouts": len(dropouts),
            "dropout_samples": sum(marker.samples for marker in dropouts),
            "dropout_seconds": sum(marker.seconds for marker in dropouts),
            "lost_sample_places": len(lost),
            "lost_samples": sum(marker.samples for marker in lost),
            "gap_policy": self.gap_policy,
        }


@dataclass(frozen=True)
class ContinuousRecording:
    """The streams of one recording on a common start.

    ``t0`` is the timestamp of its first frame of any kind, where the first epoch of
    ``EpochBuilder`` starts and the night logs' ``timestamp_seconds`` count from. A
    stream starts ``offset_samples(name)`` samples after it. ``seconds()`` is recording
    time on the EEG sample clock, the time base of an EDF export.
    """

    streams: Mapping[str, ContinuousStream]
    config: ContinuousConfig
    t0: float
    frame_count: int

    @property
    def eeg(self) -> ContinuousStream:
        if "eeg" not in self.streams:
            raise ValueError("the recording has no EEG")
        return self.streams["eeg"]

    @property
    def channels(self) -> Mapping[str, np.ndarray]:
        return self.eeg.channels

    @property
    def measured_rate_hz(self) -> float:
        return self.eeg.measured_rate_hz

    @property
    def segments(self) -> Tuple[ContinuousSegment, ...]:
        return self.eeg.segments

    @property
    def gap_markers(self) -> Tuple[GapMarker, ...]:
        return self.eeg.gap_markers

    def offset_samples(self, name: str) -> int:
        """Samples between ``t0`` and the first sample of stream ``name``."""
        stream = self.streams[name]
        return int(round((stream.t0 - self.t0) * stream.nominal_rate_hz))

    def eeg_offset_seconds(self) -> float:
        return self.offset_samples("eeg") / self.eeg.nominal_rate_hz

    def seconds(self, unix_time):
        """Recording time of host time ``unix_time``: on the EEG sample clock from ``t0``."""
        return self.eeg_offset_seconds() + self.eeg.time_map.seconds(unix_time)

    def unix_time(self, seconds):
        """The host time at recording time ``seconds``; the inverse of ``seconds()``."""
        return self.eeg.time_map.unix_time(np.asarray(seconds, dtype=float) - self.eeg_offset_seconds())

    def summary(self) -> Dict[str, object]:
        return {
            "t0": self.t0,
            "frames": self.frame_count,
            "config": self.config.to_dict(),
            "streams": {name: stream.summary() for name, stream in self.streams.items()},
        }


class _StreamBuilder:
    def __init__(
        self,
        name: str,
        rate_hz: float,
        config: ContinuousConfig,
        channels: Optional[Tuple[str, ...]] = None,
        reference_channel: Optional[str] = None,
        keep_samples: bool = True,
    ) -> None:
        self.name = name
        self.rate_hz = rate_hz
        self.config = config
        self.channels = channels
        self.reference_channel = reference_channel
        self.keep_samples = keep_samples
        self.buffers: Dict[str, array] = {}
        self.n = 0
        self.t0: Optional[float] = None
        self.t_last: Optional[float] = None
        self.prev_ts: Optional[float] = None
        self.prev_width = 0
        self.segment_start = 0
        self.segment_time = 0.0
        self.segments: List[ContinuousSegment] = []
        self.markers: List[GapMarker] = []
        self.anchor_times = array("d")
        self.anchor_indices = array("d")
        self.frames = 0
        self.received = 0
        self.unfilled_dropout = 0
        self.started = False

    def _start(self, per_channel: Mapping[str, Sequence[float]]) -> None:
        if self.channels is None:
            self.channels = tuple(str(name) for name in per_channel)
        if self.reference_channel is None:
            self.reference_channel = self.channels[0]
        if self.keep_samples:
            self.buffers = {name: array("f") for name in self.channels}
        self.started = True

    def _zeros(self, count: int) -> None:
        if count <= 0:
            return
        if self.keep_samples:
            zeros = [0.0] * count
            for buffer in self.buffers.values():
                buffer.extend(zeros)
        self.n += count

    def add(self, ts: float, per_channel: Mapping[str, Sequence[float]], gaps: Sequence[Tuple[int, int]]) -> None:
        if not per_channel:
            return
        if not self.started:
            self._start(per_channel)
        reference = per_channel.get(self.reference_channel)
        if not reference:
            return
        width = len(reference)
        config = self.config
        after_dropout = False
        if self.t0 is None:
            self.t0 = ts
            self.segment_start = 0
            self.segment_time = ts
        elif ts - self.prev_ts > config.gap_threshold_seconds:
            # Insert the missing samples for the dropout span so the timeline stays real.
            missing = int(round((ts - self.prev_ts) * self.rate_hz)) - int(self.prev_width)
            if config.fill_lost_samples:
                missing = _counted_dropout(missing, gaps, self.rate_hz)
            seconds = (ts - self.prev_ts) - self.prev_width / self.rate_hz
            if config.gap_policy == "zero_fill":
                if missing > 0:
                    self.markers.append(GapMarker(DROPOUT, self.n, missing, seconds, ts, True))
                    self._zeros(missing)
            else:
                self.markers.append(GapMarker(DROPOUT, self.n, max(missing, 0), seconds, ts, False))
                self.unfilled_dropout += max(missing, 0)
                self.segments.append(ContinuousSegment(self.segment_start, self.n, self.segment_time))
                self.segment_start = self.n
                self.segment_time = ts
            after_dropout = True
        if not self.anchor_times or ts > self.anchor_times[-1]:
            self.anchor_times.append(ts)
            self.anchor_indices.append(float(self.n))

        inserts = self._inserts(gaps, width, after_dropout, ts)
        slots = width + sum(missing for _, missing in inserts)
        if self.keep_samples:
            for name in self.channels:
                samples = per_channel.get(name) or []
                if len(samples) == width:
                    values = list(samples)
                else:  # keep channels length-aligned even if one lags on a frame
                    values = list(samples)[:width] + [0.0] * max(0, width - len(samples))
                if inserts:
                    values = _with_zeros(values, inserts)
                self.buffers[name].extend(values)
        self.n += slots
        self.prev_ts = ts
        self.prev_width = slots
        self.t_last = ts
        self.frames += 1
        self.received += width

    def _inserts(self, gaps, width: int, after_dropout: bool, ts: float) -> List[Tuple[int, int]]:
        inserts = []
        for index, missing in gaps or ():
            index, missing = int(index), int(missing)
            if missing <= 0:
                continue
            if after_dropout and index == 0:
                continue  # the counter skipped inside the dropout, which its fill covers
            index = min(max(index, 0), width)
            position = self.n + index + sum(m for _, m in inserts)
            self.markers.append(
                GapMarker(LOST_SAMPLES, position, missing, missing / self.rate_hz, ts, self.config.fill_lost_samples)
            )
            if self.config.fill_lost_samples:
                inserts.append((index, missing))
        return inserts

    def build(self) -> Optional[ContinuousStream]:
        if self.t0 is None or self.t_last is None:
            return None
        segments = list(self.segments)
        segments.append(ContinuousSegment(self.segment_start, self.n, self.segment_time))
        channels = (
            {name: np.frombuffer(buffer, dtype=np.float32).copy() for name, buffer in self.buffers.items()}
            if self.keep_samples
            else {}
        )
        return ContinuousStream(
            name=self.name,
            channels=channels,
            nominal_rate_hz=self.rate_hz,
            t0=float(self.t0),
            t_last=float(self.t_last),
            gap_policy=self.config.gap_policy,
            segments=tuple(segments),
            gap_markers=tuple(self.markers),
            time_map=TimeMap(
                np.frombuffer(self.anchor_times, dtype=float).copy(),
                np.frombuffer(self.anchor_indices, dtype=float).copy(),
                self.rate_hz,
            ),
            frame_count=self.frames,
            received_samples=self.received,
            sample_count=self.n,
            unfilled_dropout_samples=self.unfilled_dropout,
        )


def _counted_dropout(host_missing: int, gaps: Sequence[Tuple[int, int]], rate_hz: float) -> int:
    """The samples a dropout lost: the device counter's count when the frame after it
    reports one at its start and it agrees with host time to within a quarter second,
    else host time's. Host time misses by a sample or two; a counter that wrapped while
    the stream was away (the decoder reports a step of up to half its range) misses by
    whole wraps, hundreds of samples, which the check catches."""
    counted = sum(int(missing) for index, missing in gaps or () if int(index) == 0 and int(missing) > 0)
    if counted and abs(counted - host_missing) <= 0.25 * rate_hz:
        return counted
    return host_missing


def _with_zeros(values: List[float], inserts: Sequence[Tuple[int, int]]) -> List[float]:
    """``values`` with ``missing`` zeros before each ``index`` (an index into ``values``)."""
    out: List[float] = []
    previous = 0
    for index, missing in sorted(inserts):
        out.extend(values[previous:index])
        out.extend([0.0] * missing)
        previous = index
    out.extend(values[previous:])
    return out


class ContinuousBuilder:
    """Feed frames with ``add()`` in time order, then ``result()``.

    ``keep_samples=False`` keeps only the timing (anchors, gaps, counts), for a caller
    that needs ``seconds()`` of a night without its arrays.
    """

    def __init__(self, config: Optional[ContinuousConfig] = None, *, keep_samples: bool = True) -> None:
        self.config = config or ContinuousConfig()
        self.config.validate()
        streams = ("eeg",) + tuple(name for name in self.config.streams if name != "eeg")
        channels = {"eeg": self.config.channels, "accel": ACCEL_CHANNELS, "gyro": GYRO_CHANNELS, "optics": None}
        self._builders = {
            name: _StreamBuilder(
                name,
                self.config.rate_of(name),
                self.config,
                channels=channels[name],
                reference_channel=self.config.reference_channel if name == "eeg" else None,
                keep_samples=keep_samples,
            )
            for name in streams
        }
        self.first_frame_time: Optional[float] = None
        self.frame_count = 0

    def add(self, frame: MuseFrame) -> None:
        self.frame_count += 1
        if self.first_frame_time is None:
            self.first_frame_time = float(frame.timestamp)
        ts = float(frame.timestamp)
        if frame.eeg is not None:
            self._builders["eeg"].add(ts, frame.eeg.channels_uv or {}, frame.eeg.gaps)
        imu = frame.imu
        if imu is not None:
            if "accel" in self._builders and imu.accelerometer_g:
                self._builders["accel"].add(ts, _axis_columns(imu.accelerometer_g, ACCEL_CHANNELS), imu.gaps)
            if "gyro" in self._builders and imu.gyroscope_dps:
                self._builders["gyro"].add(ts, _axis_columns(imu.gyroscope_dps, GYRO_CHANNELS), imu.gaps)
        if "optics" in self._builders and frame.ppg is not None and frame.ppg.channels:
            self._builders["optics"].add(ts, frame.ppg.channels, frame.ppg.gaps)

    def result(self) -> ContinuousRecording:
        streams = {}
        for name, builder in self._builders.items():
            stream = builder.build()
            if stream is not None:
                streams[name] = stream
        if "eeg" not in streams:
            raise ValueError("no usable EEG samples in the frames")
        return ContinuousRecording(
            streams=streams,
            config=self.config,
            t0=float(self.first_frame_time if self.first_frame_time is not None else streams["eeg"].t0),
            frame_count=self.frame_count,
        )


def _axis_columns(rows: Sequence[Mapping[str, float]], names: Tuple[str, ...]) -> Dict[str, List[float]]:
    return {name: [float(row.get(axis, 0.0)) for row in rows] for name, axis in zip(names, _AXES)}


def reconstruct_continuous(
    frames: Iterable[MuseFrame],
    config: Optional[ContinuousConfig] = None,
) -> ContinuousRecording:
    """The arrays of ``frames`` (see the module docstring); ValueError without EEG."""
    builder = ContinuousBuilder(config)
    for frame in frames:
        builder.add(frame)
    return builder.result()


async def reconstruct_recording(
    recording_input: Path,
    config: Optional[ContinuousConfig] = None,
    *,
    keep_samples: bool = True,
) -> ContinuousRecording:
    """``reconstruct_continuous()`` of a replayed recording directory, frame by frame."""
    from muse_tmr.data.replay import ReplayConfig, ReplaySession

    builder = ContinuousBuilder(config, keep_samples=keep_samples)
    session = ReplaySession(ReplayConfig(input_path=Path(recording_input), speed=0.0))
    await session.connect()
    try:
        async for frame in session.stream():
            builder.add(frame)
    finally:
        await session.stop()
    return builder.result()
