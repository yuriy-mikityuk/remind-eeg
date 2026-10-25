"""Seeded synthetic nights in the Athena TAG format. Synthetic input, not sleep.

``script_night()`` scripts a 30 s hypnogram with arousals. ``generate_night()`` turns it
into a recording directory laid out like the one ``muse-tmr record`` leaves:
``raw_amused.bin`` holds BLE notifications the decoders read back, next to
``metadata.json``, ``events.jsonl`` and ``summary.json``, and the ground truth
``hypnogram.csv``, ``arousals.csv``, ``rem_events.csv`` and ``generation.json`` (seed,
config, the counts the decoder should report and the sha256 of every other file but
``summary.json``). The same seed and config give the same bytes.

The stage signals are simple textbook shapes. The device behaviour follows the
repository's own measurements (``docs/numbers.md``): ``p21`` battery drain 10.60 %/h
and ``p1034`` 23.34 %/h, 29.03 notifications per second, one to three ACCGYRO
subpackets in a notification as in the committed capture, 1.0 % lost EEG and 0.4 % lost
optics subpackets, 43 repeats in 23 min, the optics line at about 64 Hz. The morphology
is an assumption, not a measurement: nothing measured on these nights is evidence about
sleep, and no detector weight or threshold may be fitted to them.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import struct
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import muse_athena_protocol as proto
from muse_raw_stream import MuseRawStream

from muse_tmr import __version__
from muse_tmr.data.recorder import RecordingSummary
from muse_tmr.presets import preset_provides_optics
from muse_tmr.testing import tag_encoder as enc

SYNTHETIC_NOTE = "synthetic input, not sleep"
GENERATION_SCHEMA_VERSION = 1
STAGES = ("W", "N1", "N2", "N3", "R")
SUPPORTED_PRESETS = ("p21", "p1034")
EPOCH_SECONDS = 30.0
EEG_RATE_HZ = 256.0
IMU_RATE_HZ = 52.0
OPTICS_RATE_HZ = 64.0
EEG_CHANNELS = tuple(proto.EEG_CHANNELS_4)
OPTICS_CHANNELS = tuple(proto.OPTICS_CHANNELS_8)
# 2026-01-14 23:00:00 UTC. A fixed start keeps the raw file's header reproducible. Mid
# January is far from daylight-saving changes, which the naive local datetimes of
# MuseRawStream would turn into a jump in the replayed timestamps.
START_EPOCH_SECONDS = 1_768_431_600.0
# A counter that steps 128 or more reads as a restarted stream, not as a loss.
_MAX_LOST_RUN = proto.COUNTER_MODULO // 2 - 1

# (1/f background, delta, theta, alpha, EMG) µV RMS per stage.
DEFAULT_STAGE_EEG_RMS_UV = {
    "W": (10.0, 2.0, 3.0, 12.0, 4.0),
    "N1": (12.0, 4.0, 6.0, 4.0, 2.0),
    "N2": (14.0, 10.0, 4.0, 2.0, 1.5),
    "N3": (16.0, 30.0, 3.0, 1.5, 1.0),
    "R": (10.0, 3.0, 8.0, 2.0, 0.5),
}
DEFAULT_HEART_RATE_BPM = {"W": 66.0, "N1": 62.0, "N2": 58.0, "N3": 56.0, "R": 63.0}
DEFAULT_DRAIN_PERCENT_PER_HOUR = {"p21": 10.60, "p1034": 23.34}

# Per-channel weight of each shared component: delta, theta, alpha, EMG, spindles.
_CHANNEL_WEIGHTS = {
    "TP9": (0.7, 0.9, 1.0, 1.0, 0.6),
    "AF7": (1.0, 1.0, 0.5, 0.7, 0.8),
    "AF8": (1.0, 1.0, 0.5, 0.7, 0.8),
    "TP10": (0.7, 0.9, 1.0, 1.0, 0.6),
}
# Head orientations (gravity in the headband's frame, g): on the back, left, right.
_POSTURES = ((0.05, 0.10, 0.99), (-0.90, 0.10, 0.42), (0.90, 0.10, 0.42))
# Optics DC level (normalised units) and pulsatile share per channel, OPTICS_CHANNELS
# order: the outer pair carries less pulse than the inner pair.
_OPTICS_DC = (9.0, 9.4, 7.5, 7.8, 12.0, 12.4, 10.5, 10.8)
_OPTICS_PULSE_SHARE = (0.003, 0.003, 0.003, 0.003, 0.012, 0.012, 0.012, 0.012)

# Independent random streams, so turning one component on or off leaves the others
# as they were.
_RNG_SCRIPT, _RNG_EEG, _RNG_EVENTS, _RNG_LINE, _RNG_IMU, _RNG_OPTICS, _RNG_BLE, _RNG_MOVES, _RNG_READOUT = range(9)


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream]))


# --------------------------------------------------------------------------- #
# Signal shapes, shared with the test fixtures
# --------------------------------------------------------------------------- #
def pink_noise(
    n: int,
    chi: float,
    rng: np.random.Generator,
    *,
    fs: float = EEG_RATE_HZ,
    low_hz: float = 0.0,
) -> np.ndarray:
    """Unit-RMS noise with a 1/f**chi power spectrum, nothing below ``low_hz``."""
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
    spectrum[1:] *= freqs[1:] ** (-chi / 2.0)
    spectrum[0] = 0.0
    if low_hz > 0:
        spectrum[freqs < low_hz] = 0.0
    signal = np.fft.irfft(spectrum, n)
    return signal / np.std(signal)


def band_limited_noise(
    n: int,
    low_hz: float,
    high_hz: float,
    rng: np.random.Generator,
    *,
    fs: float = EEG_RATE_HZ,
) -> np.ndarray:
    """Unit-RMS noise with a flat spectrum between ``low_hz`` and ``high_hz``."""
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0.0
    signal = np.fft.irfft(spectrum, n)
    std = np.std(signal)
    return signal / std if std > 0 else signal


def deflection(duration_s: float, amplitude_uv: float, fs: float = EEG_RATE_HZ) -> np.ndarray:
    """One biphasic eye-movement deflection: a single sine cycle, first lobe positive."""
    t = np.arange(int(round(duration_s * fs))) / fs
    return amplitude_uv * np.sin(2 * np.pi * t / duration_s)


def blink_shape(duration_s: float, amplitude_uv: float, fs: float = EEG_RATE_HZ) -> np.ndarray:
    """One monophasic blink: a Hann bump."""
    return amplitude_uv * np.hanning(int(round(duration_s * fs)))


# --------------------------------------------------------------------------- #
# Hypnogram
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Arousal:
    """A brief arousal inside a scored epoch; ``movement`` adds an IMU burst."""

    onset_seconds: float
    duration_seconds: float
    stage: str
    movement: bool
    position_change: bool


@dataclass(frozen=True)
class ScriptedNight:
    seed: int
    stages: Tuple[str, ...]
    arousals: Tuple[Arousal, ...]
    epoch_seconds: float = EPOCH_SECONDS

    @property
    def duration_seconds(self) -> float:
        return len(self.stages) * self.epoch_seconds

    def stage_counts(self) -> Dict[str, int]:
        return {stage: self.stages.count(stage) for stage in STAGES}

    def rem_runs(self) -> Tuple[Tuple[int, int], ...]:
        """(first, stop) epoch indexes of each run of consecutive R epochs."""
        runs = []
        start = None
        for index, stage in enumerate(self.stages + ("",)):
            if stage == "R" and start is None:
                start = index
            elif stage != "R" and start is not None:
                runs.append((start, index))
                start = None
        return tuple(runs)


def script_night(
    seed: int,
    hours: float = 8.0,
    rem_periods: Optional[int] = None,
    wake_bouts: Optional[int] = None,
    *,
    rem_interruptions: int = 0,
    arousals_per_hour: float = 8.0,
    movement_share: float = 0.4,
    stages: Optional[Sequence[str]] = None,
) -> ScriptedNight:
    """A seeded 30 s hypnogram (W, N1, N2, N3, R) with arousals.

    After 5 to 10 min awake and a few minutes of N1 the night runs in cycles, each
    N2, N3, N2 and a REM period, with N3 shrinking and REM growing from cycle to cycle.
    ``rem_periods`` defaults to the number of sleep cycles of 90 to 110 min the night
    holds, rounded, with no cycle under 70 min and at most 5 (4 or 5 for 8 h, 1 for 2 h,
    none for 1 h); ``wake_bouts`` to a brief awakening after every REM period but the
    last. ``rem_interruptions`` breaks each REM period with that many awakenings of
    30 to 60 s. Arousals of 3 to 15 s come at ``arousals_per_hour`` of sleep, a
    ``movement_share`` of them with movement; every awakening starts with one. Pass
    ``stages`` (as many epochs as ``hours`` holds) to keep a fixed hypnogram and script
    only the arousals.
    """
    rng = _rng(seed, _RNG_SCRIPT)
    total = int(round(hours * 3600 / EPOCH_SECONDS))
    if total <= 0:
        raise ValueError("hours must cover at least one 30 s epoch")
    if stages is not None:
        scripted = [str(stage) for stage in stages]
        unknown = sorted(set(scripted) - set(STAGES))
        if unknown:
            raise ValueError(f"unknown stages: {unknown}")
        if len(scripted) != total:
            raise ValueError(f"{len(scripted)} stages for {total} epochs of {hours} h")
        wake_starts: List[int] = []
    else:
        scripted, wake_starts = _script_stages(rng, total, rem_periods, wake_bouts)
        if rem_interruptions:
            wake_starts += _interrupt_rem(scripted, rem_interruptions, rng)
    arousals = _script_arousals(
        scripted, sorted(wake_starts), rng, arousals_per_hour, movement_share
    )
    return ScriptedNight(seed=int(seed), stages=tuple(scripted), arousals=arousals)


def _script_stages(rng, total, rem_periods, wake_bouts):
    stages = ["W"] * int(rng.integers(10, 21)) + ["N1"] * int(rng.integers(4, 9))
    available = total - len(stages)
    cycle_epochs = int(rng.integers(180, 221))
    if rem_periods is None:
        rem_periods = min(5, max(0, int(round(available / cycle_epochs))))
        # A cycle squeezed below 70 min would bring REM unusually early.
        while rem_periods and available / rem_periods < 140:
            rem_periods -= 1
    if wake_bouts is None:
        wake_bouts = max(0, rem_periods - 1)
    wake_starts: List[int] = []
    if available <= 0:
        return stages[:total], wake_starts
    if rem_periods <= 0:
        while len(stages) < total:
            stages += ["N2"] * int(rng.integers(20, 41)) + ["N3"] * int(rng.integers(10, 31))
        return stages[:total], wake_starts

    weights = rng.uniform(0.9, 1.1, rem_periods)
    lengths = np.floor(available * weights / weights.sum()).astype(int)
    lengths[-1] += available - int(lengths.sum())
    rem_share = (0.12, 0.20, 0.25, 0.30, 0.33)
    n3_share = (0.30, 0.22, 0.10, 0.04, 0.0)
    for cycle, length in enumerate(lengths):
        cycle_stages: List[str] = []
        if cycle > 0:
            cycle_stages += ["N1"] * int(rng.integers(2, 5))
        rem = max(6, int(round(length * rem_share[min(cycle, 4)] * rng.uniform(0.85, 1.15))))
        n3 = int(round(length * n3_share[min(cycle, 4)] * rng.uniform(0.8, 1.2)))
        n2 = max(4, int(length) - rem - n3 - len(cycle_stages))
        first_n2 = int(round(n2 * 0.55))
        cycle_stages += ["N2"] * first_n2 + ["N3"] * n3 + ["N2"] * (n2 - first_n2) + ["R"] * rem
        if cycle < rem_periods - 1 and cycle < wake_bouts:
            wake_starts.append(len(stages) + len(cycle_stages))
            cycle_stages += ["W"] * int(rng.integers(1, 5))
        stages += cycle_stages
    # Bouts beyond the REM periods go into N2.
    n2_epochs = [index for index, stage in enumerate(stages) if stage == "N2"]
    for _ in range(max(0, wake_bouts - (rem_periods - 1))):
        if not n2_epochs:
            break
        start = int(rng.choice(n2_epochs))
        length = int(rng.integers(1, 5))
        stages[start:start + length] = ["W"] * len(stages[start:start + length])
        wake_starts.append(start)
    return stages[:total], [start for start in wake_starts if start < total]


def _interrupt_rem(stages: List[str], interruptions: int, rng) -> List[int]:
    starts = []
    for first, stop in ScriptedNight(0, tuple(stages), ()).rem_runs():
        length = stop - first
        if length < 4:
            continue
        cuts = np.linspace(first, stop, interruptions + 2)[1:-1]
        for cut in cuts:
            start = int(round(cut + rng.uniform(-1.0, 1.0)))
            width = int(rng.integers(1, 3))
            if first < start and start + width < stop:
                stages[start:start + width] = ["W"] * width
                starts.append(start)
    return starts


def _script_arousals(stages, wake_starts, rng, arousals_per_hour, movement_share):
    arousals: List[Arousal] = []
    taken: List[Tuple[float, float]] = []
    for start in wake_starts:
        onset = start * EPOCH_SECONDS + float(rng.uniform(0.0, 5.0))
        duration = float(rng.uniform(5.0, 10.0))
        arousals.append(Arousal(onset, duration, "W", True, True))
        taken.append((onset, onset + duration))

    sleep_epochs = [index for index, stage in enumerate(stages) if stage != "W"]
    sleep_hours = len(sleep_epochs) * EPOCH_SECONDS / 3600
    count = int(rng.poisson(arousals_per_hour * sleep_hours)) if sleep_epochs else 0
    night_end = len(stages) * EPOCH_SECONDS
    for _ in range(count):
        epoch = int(rng.choice(sleep_epochs))
        duration = float(rng.uniform(3.0, 15.0))
        onset = epoch * EPOCH_SECONDS + float(rng.uniform(0.0, EPOCH_SECONDS))
        movement = bool(rng.random() < movement_share)
        position_change = bool(movement and rng.random() < 0.15)
        if onset + duration > night_end - 1.0:
            continue
        if any(onset < end + 20.0 and other < onset + duration + 20.0 for other, end in taken):
            continue
        arousals.append(Arousal(onset, duration, stages[epoch], movement, position_change))
        taken.append((onset, onset + duration))
    return tuple(sorted(arousals, key=lambda arousal: arousal.onset_seconds))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntheticNightConfig:
    """Everything a synthetic night depends on; the seed drives every random draw.

    Hypnogram: see ``script_night()``. ``stages`` replaces the scripted hypnogram.

    EEG (µV, centered, as the decoder returns it): per stage the RMS of a 1/f
    background (10 to 20 µV, from 0.3 Hz up), of delta (0.5 to 2 Hz, frontal), theta
    (4 to 8 Hz), alpha (8 to 12 Hz, temporal) and EMG (20 to 100 Hz), with 1 s ramps
    between epochs. Arousals bring wake alpha and EMG; N2 has spindles (12 to 14 Hz).
    R epochs carry eye-movement deflections of opposite polarity on AF7 and AF8
    (``deflection_uv``, ``deflection_seconds``, the shape of
    ``tests/rem_event_synthetic.py``), W epochs in-phase blinks. Optics presets add a
    line at ``line_hz`` on AF7 and 0.6 of it on AF8 (``line_uv``: None draws 20 to
    40 µV). ``bad_contact_channel`` floats that channel between the two fractions of
    the night in ``bad_contact_span``.

    IMU: gravity for one of three head positions, sensor noise, and a burst
    (``movement_scale``) at every movement, which can change position. Optics
    (``p1034`` only): a pulse at the stage heart rate in ``heart_rate_bpm``, faster for
    a while after arousals. Battery: ``battery_start_percent`` falling by
    ``battery_drain_percent_per_hour`` (None: 10.60 for ``p21``, 23.34 for ``p1034``);
    at 0 % the stream ends.

    BLE: notifications leave ``notifications_per_second`` times a second, each with the
    subpackets made since the one before in the order they were made, split at 244
    bytes. The IMU is read out one, two or three subpackets at a time, as often as
    ``imu_bundle_shares`` says, and a readout goes into one notification. Subpackets are
    lost at ``*_loss_fraction``. Whole notifications come twice, 1 to 3 notifications
    later, as often as makes ``eeg_repeats_per_hour`` EEG subpackets arrive twice.
    ``ble_gaps_per_hour`` gaps of ``ble_gap_seconds`` lose everything in them.
    """

    seed: int = 0
    hours: float = 8.0
    preset: str = "p21"
    scenario: str = "custom"
    rem_periods: Optional[int] = None
    wake_bouts: Optional[int] = None
    rem_interruptions: int = 0
    arousals_per_hour: float = 8.0
    movement_share: float = 0.4
    stages: Optional[Tuple[str, ...]] = None
    stage_eeg_rms_uv: Mapping[str, Tuple[float, float, float, float, float]] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_EEG_RMS_UV)
    )
    arousal_alpha_uv: float = 12.0
    arousal_emg_uv: float = 8.0
    spindles_per_minute: float = 4.0
    spindle_uv: Tuple[float, float] = (10.0, 25.0)
    rem_events_per_minute: Tuple[float, float] = (4.0, 12.0)
    deflection_uv: Tuple[float, float] = (20.0, 60.0)
    deflection_seconds: Tuple[float, float] = (0.3, 1.5)
    blinks_per_minute: float = 6.0
    blink_uv: Tuple[float, float] = (100.0, 200.0)
    line_hz: float = 63.93
    line_uv: Optional[float] = None
    bad_contact_channel: Optional[str] = None
    bad_contact_span: Tuple[float, float] = (0.0, 0.0)
    movement_scale: float = 1.0
    heart_rate_bpm: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_HEART_RATE_BPM))
    battery_start_percent: float = 100.0
    battery_drain_percent_per_hour: Optional[float] = None
    notifications_per_second: float = 29.03
    imu_bundle_shares: Tuple[float, ...] = (49 / 83, 32 / 83, 2 / 83)
    eeg_loss_fraction: float = 888 / 90011
    optics_loss_fraction: float = 169 / 44950
    imu_loss_fraction: float = 0.0
    eeg_repeats_per_hour: float = 43 / (1400 / 3600)
    ble_gaps_per_hour: float = 0.25
    ble_gap_seconds: Tuple[float, float] = (0.5, 1.5)

    def validate(self) -> None:
        if self.preset not in SUPPORTED_PRESETS:
            raise ValueError(f"preset must be one of {SUPPORTED_PRESETS}")
        if self.hours <= 0:
            raise ValueError("hours must be positive")
        if set(self.stage_eeg_rms_uv) != set(STAGES) or set(self.heart_rate_bpm) != set(STAGES):
            raise ValueError(f"stage tables need exactly the stages {STAGES}")
        if self.bad_contact_channel is not None and self.bad_contact_channel not in EEG_CHANNELS:
            raise ValueError(f"bad_contact_channel must be one of {EEG_CHANNELS}")
        if not 0.0 <= self.bad_contact_span[0] <= self.bad_contact_span[1] <= 1.0:
            raise ValueError("bad_contact_span must be two fractions of the night in order")
        for name in ("eeg_loss_fraction", "optics_loss_fraction", "imu_loss_fraction"):
            if not 0.0 <= getattr(self, name) < 0.5:
                raise ValueError(f"{name} must be in [0, 0.5)")
        if self.notifications_per_second <= 0:
            raise ValueError("notifications_per_second must be positive")
        if not self.imu_bundle_shares or min(self.imu_bundle_shares) < 0 or sum(self.imu_bundle_shares) <= 0:
            raise ValueError("imu_bundle_shares needs a share for 1, 2, ... subpackets, none negative")
        if not 0.0 < self.ble_gap_seconds[0] <= self.ble_gap_seconds[1] < 1.8:
            raise ValueError("BLE gaps must be under 1.8 s, or the EEG counter would wrap")
        if self.battery_start_percent <= 0 or self.drain_percent_per_hour <= 0:
            raise ValueError("battery start and drain must be positive")

    @property
    def has_optics(self) -> bool:
        return preset_provides_optics(self.preset)

    @property
    def drain_percent_per_hour(self) -> float:
        if self.battery_drain_percent_per_hour is not None:
            return float(self.battery_drain_percent_per_hour)
        return DEFAULT_DRAIN_PERCENT_PER_HOUR[self.preset]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["stage_eeg_rms_uv"] = {
            key: list(value) for key, value in sorted(self.stage_eeg_rms_uv.items())
        }
        payload["heart_rate_bpm"] = dict(sorted(self.heart_rate_bpm.items()))
        return payload


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RemEvent:
    onset_seconds: float
    duration_seconds: float
    amplitude_uv: float
    af8_ratio: float
    sign: int


@dataclass(frozen=True)
class SyntheticNight:
    """A generated recording directory and what went into it."""

    recording_dir: Path
    config: SyntheticNightConfig
    night: ScriptedNight
    generation: Mapping[str, object]
    generation_seconds: float

    @property
    def raw_path(self) -> Path:
        return self.recording_dir / "raw_amused.bin"

    @property
    def sha256(self) -> str:
        return str(self.generation["files"]["raw_amused.bin"]["sha256"])  # type: ignore[index]


def generate_night(
    config: Optional[SyntheticNightConfig] = None,
    output_dir: Optional[Path] = None,
    *,
    scenario: Optional[str] = None,
    seed: Optional[int] = None,
    hours: Optional[float] = None,
) -> SyntheticNight:
    """Write a synthetic recording into ``output_dir``.

    Give a ``config``, or a ``scenario`` name from ``muse_tmr.testing.scenarios``;
    ``seed`` and ``hours`` replace the config's.
    """
    if output_dir is None:
        raise ValueError("generate_night needs an output_dir")
    if scenario is not None:
        if config is not None:
            raise ValueError("give a config or a scenario, not both")
        from muse_tmr.testing.scenarios import scenario_config

        config = scenario_config(scenario, seed=0 if seed is None else seed, hours=hours)
    else:
        config = config or SyntheticNightConfig()
        overrides = {key: value for key, value in (("seed", seed), ("hours", hours)) if value is not None}
        if overrides:
            config = replace(config, **overrides)
    config.validate()

    started = time.perf_counter()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    night = script_night(
        config.seed,
        config.hours,
        config.rem_periods,
        config.wake_bouts,
        rem_interruptions=config.rem_interruptions,
        arousals_per_hour=config.arousals_per_hour,
        movement_share=config.movement_share,
        stages=config.stages,
    )
    death_seconds = config.battery_start_percent / config.drain_percent_per_hour * 3600.0
    duration = min(night.duration_seconds, death_seconds)
    stop_reason = "source_ended" if death_seconds < night.duration_seconds else "duration_complete"
    arousals = tuple(a for a in night.arousals if a.onset_seconds + a.duration_seconds <= duration)
    movements = _movement_windows(arousals, _rng(config.seed, _RNG_MOVES))

    eeg_counts, eeg_events = _eeg_counts(config, night, arousals, movements, duration)
    accel, gyro = _imu_signal(config, arousals, movements, duration)
    optics = (
        _optics_signal(config, night, arousals, movements, duration) if config.has_optics else None
    )
    streams = _streams(config, eeg_counts, accel, gyro, optics, duration)
    del eeg_counts, accel, gyro, optics

    raw_path = output_dir / "raw_amused.bin"
    ble = _write_raw(raw_path, streams, duration, config)
    del streams

    files = _write_ground_truth(output_dir, night, arousals, eeg_events["rem_events"], duration)
    files.update(_write_recording_files(output_dir, config, duration, stop_reason, ble))
    generation = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "synthetic": True,
        "note": SYNTHETIC_NOTE,
        "generator": __name__,
        "muse_tmr_version": __version__,
        "scenario": config.scenario,
        "seed": config.seed,
        "preset": config.preset,
        "config": config.to_dict(),
        "start_utc": _iso(0.0),
        "duration_seconds": duration,
        "stop_reason": stop_reason,
        "hypnogram": {
            "epochs": int(math.ceil(duration / EPOCH_SECONDS - 1e-9)),
            "stage_counts": _stage_counts(night, duration),
            "rem_periods": [
                [first * EPOCH_SECONDS, min(stop * EPOCH_SECONDS, duration)]
                for first, stop in night.rem_runs()
                if first * EPOCH_SECONDS < duration
            ],
        },
        "events": {
            "arousals": len(arousals),
            "movements": sum(1 for arousal in arousals if arousal.movement),
            "position_changes": sum(1 for arousal in arousals if arousal.position_change),
            "rem_events": len(eeg_events["rem_events"]),
            "blinks": eeg_events["blinks"],
            "spindles": eeg_events["spindles"],
        },
        "line_uv": eeg_events["line_uv"],
        "battery": {
            "start_percent": config.battery_start_percent,
            "drain_percent_per_hour": config.drain_percent_per_hour,
            "end_percent": max(
                0.0, config.battery_start_percent - config.drain_percent_per_hour * duration / 3600
            ),
            "empty_at_seconds": death_seconds if stop_reason == "source_ended" else None,
        },
        "ble": ble,
        "files": files,
    }
    _write_json(output_dir / "generation.json", generation)
    return SyntheticNight(
        recording_dir=output_dir,
        config=config,
        night=night,
        generation=generation,
        generation_seconds=time.perf_counter() - started,
    )


def _stage_counts(night: ScriptedNight, duration: float) -> Dict[str, int]:
    epochs = night.stages[: int(math.ceil(duration / EPOCH_SECONDS - 1e-9))]
    return {stage: epochs.count(stage) for stage in STAGES}


# --------------------------------------------------------------------------- #
# EEG
# --------------------------------------------------------------------------- #
def _stage_envelope(values: np.ndarray, n: int, fs: float, ramp_seconds: float) -> np.ndarray:
    """Per-epoch values held over each epoch, ramped linearly across epoch boundaries."""
    per_epoch = int(round(EPOCH_SECONDS * fs))
    envelope = np.repeat(np.asarray(values, dtype=float), per_epoch)
    if envelope.size < n:
        envelope = np.concatenate([envelope, np.full(n - envelope.size, float(values[-1]))])
    envelope = envelope[:n]
    ramp = int(round(ramp_seconds * fs))
    for epoch in np.flatnonzero(np.diff(values) != 0) + 1:
        boundary = int(epoch) * per_epoch
        low, high = max(0, boundary - ramp), min(n, boundary + ramp)
        if low < high:
            envelope[low:high] = np.interp(
                np.arange(low, high), [boundary - ramp, boundary + ramp], [values[epoch - 1], values[epoch]]
            )
    return envelope


def _event_envelope(
    spans: Sequence[Tuple[float, float]], n: int, fs: float, ramp_seconds: float
) -> np.ndarray:
    """1 inside each (onset, duration) span, ramping up and down at its edges, else 0."""
    envelope = np.zeros(n)
    ramp = max(1, int(round(ramp_seconds * fs)))
    for onset, duration in spans:
        first, stop = int(onset * fs), min(n, int((onset + duration) * fs))
        if stop - first < 2:
            continue
        shape = np.ones(stop - first)
        edge = min(ramp, shape.size // 2)
        shape[:edge] = np.linspace(0.0, 1.0, edge, endpoint=False)
        shape[shape.size - edge:] = np.linspace(1.0, 0.0, edge)
        envelope[first:stop] = np.maximum(envelope[first:stop], shape)
    return envelope


def _movement_windows(arousals: Sequence[Arousal], rng) -> List[Tuple[float, float]]:
    """(onset, seconds) of the movement inside each moving arousal."""
    return [
        (arousal.onset_seconds, min(arousal.duration_seconds, float(rng.uniform(1.0, 4.0))))
        for arousal in arousals
        if arousal.movement
    ]


def _eeg_counts(config: SyntheticNightConfig, night: ScriptedNight, arousals, movements, duration: float):
    """(n, 4) EEG counts at 256 Hz and the events placed in them."""
    n = int(duration * EEG_RATE_HZ) // 4 * 4
    table = np.array([config.stage_eeg_rms_uv[stage] for stage in night.stages], dtype=float)
    noise = _rng(config.seed, _RNG_EEG)
    events_rng = _rng(config.seed, _RNG_EVENTS)
    fs = EEG_RATE_HZ

    # The night-long components are kept in float32: a whole night is 7.4 M samples, and
    # float32 is far finer than one count (0.089 µV).
    awake = _event_envelope([(a.onset_seconds, a.duration_seconds) for a in arousals], n, fs, 0.5)
    background = _stage_envelope(table[:, 0], n, fs, 1.0).astype(np.float32)
    emg_env = (
        _stage_envelope(table[:, 4], n, fs, 1.0) * (1.0 - awake) + config.arousal_emg_uv * awake
    ).astype(np.float32)
    shared = (0.94 * background * pink_noise(n, 1.0, noise, low_hz=0.3)).astype(np.float32)
    delta = (
        _stage_envelope(table[:, 1], n, fs, 1.0)
        * (1.0 - 0.7 * awake)
        * band_limited_noise(n, 0.5, 2.0, noise)
    ).astype(np.float32)
    theta = (
        _stage_envelope(table[:, 2], n, fs, 1.0) * band_limited_noise(n, 4.0, 8.0, noise)
    ).astype(np.float32)
    alpha = (
        (_stage_envelope(table[:, 3], n, fs, 1.0) * (1.0 - awake) + config.arousal_alpha_uv * awake)
        * band_limited_noise(n, 8.0, 12.0, noise)
    ).astype(np.float32)
    del awake

    spindles, spindle_count = _spindles(config, night, n, events_rng)
    rem_events = _rem_events(config, night, duration, events_rng)
    blinks = _blinks(config, night, duration, events_rng)
    artifacts = _movement_artifacts(movements, config, n, events_rng)

    line_rng = _rng(config.seed, _RNG_LINE)
    if config.line_uv is not None:
        line_uv = float(config.line_uv) if config.has_optics else 0.0
    else:
        line_uv = float(line_rng.uniform(20.0, 40.0)) if config.has_optics else 0.0
    line_phase = float(line_rng.uniform(0.0, 2 * np.pi))

    counts = np.empty((n, len(EEG_CHANNELS)), dtype=np.uint16)
    for column, channel in enumerate(EEG_CHANNELS):
        w_delta, w_theta, w_alpha, w_emg, w_spindle = _CHANNEL_WEIGHTS[channel]
        values = shared + 0.34 * background * pink_noise(n, 1.0, noise, low_hz=0.3)
        values += w_delta * delta + w_theta * theta + w_alpha * alpha
        values += w_emg * emg_env * band_limited_noise(n, 20.0, 100.0, noise)
        values += w_spindle * spindles
        values += (1.0 if channel in ("AF7", "AF8") else 0.8) * artifacts
        if channel in ("AF7", "AF8"):
            _add_eye_events(values, channel, rem_events, blinks)
            if line_uv:
                share, shift = (1.0, 0.0) if channel == "AF7" else (0.6, 1.0)
                phase = 2 * np.pi * config.line_hz / fs * np.arange(n) + line_phase + shift
                values += share * line_uv * np.sin(phase)
                del phase
        if channel == config.bad_contact_channel:
            _float_channel(values, config.bad_contact_span, events_rng)
        counts[:, column] = enc.eeg_uv_to_counts(values)
    return counts, {
        "rem_events": rem_events,
        "blinks": len(blinks),
        "spindles": spindle_count,
        "line_uv": line_uv,
    }


def _spindles(config, night, n, rng) -> Tuple[np.ndarray, int]:
    signal = np.zeros(n, dtype=np.float32)
    rate = {"N2": config.spindles_per_minute, "N3": config.spindles_per_minute / 4}
    fs = EEG_RATE_HZ
    count = 0
    for epoch, stage in enumerate(night.stages):
        if stage not in rate:
            continue
        for _ in range(int(rng.poisson(rate[stage] * EPOCH_SECONDS / 60))):
            seconds = float(rng.uniform(0.5, 1.5))
            onset = epoch * EPOCH_SECONDS + float(rng.uniform(0.0, EPOCH_SECONDS - seconds))
            first = int(onset * fs)
            samples = int(seconds * fs)
            if first + samples > n:
                continue
            tt = np.arange(samples) / fs
            burst = float(rng.uniform(*config.spindle_uv)) * np.hanning(samples)
            frequency = float(rng.uniform(12.0, 14.0))
            burst *= np.sin(2 * np.pi * frequency * tt + float(rng.uniform(0, 2 * np.pi)))
            signal[first:first + samples] += burst
            count += 1
    return signal, count


def _rem_events(config, night: ScriptedNight, duration: float, rng) -> Tuple[RemEvent, ...]:
    events: List[RemEvent] = []
    for first, stop in night.rem_runs():
        density = float(rng.uniform(*config.rem_events_per_minute))
        for epoch in range(first, stop):
            start = epoch * EPOCH_SECONDS
            count = int(rng.poisson(density * EPOCH_SECONDS / 60))
            onsets = np.sort(rng.uniform(start + 0.25, start + EPOCH_SECONDS - 1.75, count))
            free_from = start
            for onset in onsets:
                seconds = float(rng.uniform(*config.deflection_seconds))
                amplitude = float(rng.uniform(*config.deflection_uv))
                ratio = float(rng.uniform(0.8, 1.0))
                sign = 1 if rng.random() < 0.5 else -1
                if onset < free_from or onset + seconds > min(start + EPOCH_SECONDS - 0.25, duration):
                    continue
                events.append(RemEvent(float(onset), seconds, amplitude, ratio, sign))
                free_from = onset + seconds + 0.3
    return tuple(events)


def _blinks(config, night: ScriptedNight, duration: float, rng):
    """(onset, amplitude, larger side, ratio of the other side) of each blink in W epochs."""
    blinks = []
    for epoch, stage in enumerate(night.stages):
        if stage != "W":
            continue
        start = epoch * EPOCH_SECONDS
        count = int(rng.poisson(config.blinks_per_minute * EPOCH_SECONDS / 60))
        for onset in np.sort(rng.uniform(start, start + EPOCH_SECONDS - 0.5, count)):
            amplitude = float(rng.uniform(*config.blink_uv))
            larger = "AF7" if rng.random() < 0.5 else "AF8"
            ratio = float(rng.uniform(0.7, 0.9))
            if onset + 0.35 <= duration:
                blinks.append((float(onset), amplitude, larger, ratio))
    return tuple(blinks)


def _add_eye_events(values: np.ndarray, channel: str, rem_events, blinks) -> None:
    fs = EEG_RATE_HZ
    for event in rem_events:
        shape = event.sign * deflection(event.duration_seconds, event.amplitude_uv)
        if channel == "AF8":
            shape = -event.af8_ratio * shape
        first = int(round(event.onset_seconds * fs))
        stop = min(values.size, first + shape.size)
        values[first:stop] += shape[: stop - first]
    for onset, amplitude, larger, ratio in blinks:
        shape = blink_shape(0.35, amplitude) * (1.0 if channel == larger else ratio)
        first = int(round(onset * fs))
        stop = min(values.size, first + shape.size)
        values[first:stop] += shape[: stop - first]


def _movement_artifacts(windows, config, n, rng) -> np.ndarray:
    signal = np.zeros(n, dtype=np.float32)
    fs = EEG_RATE_HZ
    for onset, seconds in windows:
        first = int(onset * fs)
        samples = min(n - first, int(seconds * fs))
        if samples < 16:
            continue
        burst = band_limited_noise(samples, 0.5, 4.0, rng) * np.hanning(samples)
        signal[first:first + samples] += float(rng.uniform(40.0, 120.0)) * config.movement_scale * burst
    return signal


def _float_channel(values: np.ndarray, span: Tuple[float, float], rng) -> None:
    """A lifted electrode: a large slow drift around an offset, saturating at the rails."""
    first, stop = int(span[0] * values.size), int(span[1] * values.size)
    if stop - first < 2:
        return
    drift = 350.0 * pink_noise(stop - first, 2.0, rng, low_hz=0.05)
    offset = float(rng.uniform(-300.0, 300.0))
    values[first:stop] = offset + drift + 20.0 * rng.standard_normal(stop - first)


# --------------------------------------------------------------------------- #
# IMU and optics
# --------------------------------------------------------------------------- #
def _imu_signal(config: SyntheticNightConfig, arousals, movements, duration: float):
    """(n, 3) acceleration in g and (n, 3) rotation in deg/s at 52 Hz."""
    rng = _rng(config.seed, _RNG_IMU)
    fs = IMU_RATE_HZ
    n = int(duration * fs) // 3 * 3
    postures = [np.asarray(p) / np.linalg.norm(p) for p in _POSTURES]
    gravity = np.empty((n, 3))
    current, cursor = 0, 0
    for arousal in arousals:
        if not arousal.position_change:
            continue
        first = min(n, int(arousal.onset_seconds * fs))
        stop = min(n, int((arousal.onset_seconds + min(arousal.duration_seconds, 4.0)) * fs))
        gravity[cursor:first] = postures[current]
        target = int(rng.choice([index for index in range(len(postures)) if index != current]))
        if stop > first:
            blend = np.linspace(0.0, 1.0, stop - first)[:, None]
            blend = blend * blend * (3.0 - 2.0 * blend)
            turning = (1.0 - blend) * postures[current] + blend * postures[target]
            gravity[first:stop] = turning / np.linalg.norm(turning, axis=1, keepdims=True)
        current, cursor = target, max(cursor, stop)
    gravity[cursor:] = postures[current]
    accel = gravity + 0.002 * rng.standard_normal((n, 3))
    gyro = 0.25 * rng.standard_normal((n, 3))
    for onset, seconds in movements:
        first = int(onset * fs)
        samples = min(n - first, int(seconds * fs))
        if samples < 8:
            continue
        burst = np.stack([band_limited_noise(samples, 0.5, 5.0, rng, fs=fs) for _ in range(6)], axis=1)
        burst *= config.movement_scale * np.hanning(samples)[:, None]
        accel[first:first + samples] += float(rng.uniform(0.1, 0.4)) * burst[:, :3]
        gyro[first:first + samples] += float(rng.uniform(30.0, 120.0)) * burst[:, 3:]
    return accel, gyro


def _optics_signal(config: SyntheticNightConfig, night: ScriptedNight, arousals, movements, duration):
    """(n, 8) normalised optics values at 64 Hz: a pulse at the scripted heart rate."""
    rng = _rng(config.seed, _RNG_OPTICS)
    fs = OPTICS_RATE_HZ
    n = int(duration * fs) // 2 * 2
    t = np.arange(n) / fs
    hr = _stage_envelope(np.array([config.heart_rate_bpm[s] for s in night.stages]), n, fs, 15.0)
    hr += 1.5 * band_limited_noise(n, 0.002, 0.02, rng, fs=fs)
    hr += 1.5 * np.sin(2 * np.pi * 0.25 * t + float(rng.uniform(0, 2 * np.pi)))
    for arousal in arousals:
        first = int(arousal.onset_seconds * fs)
        boost = float(rng.uniform(8.0, 18.0)) * (1.0 if arousal.movement else 0.5)
        shape = np.concatenate([
            np.linspace(0.0, 1.0, int(2 * fs), endpoint=False),
            np.ones(int(arousal.duration_seconds * fs)),
            np.exp(-np.arange(int(40 * fs)) / (10 * fs)),
        ])
        stop = min(n, first + shape.size)
        hr[first:stop] += boost * shape[: stop - first]
    phase = np.cumsum(hr / 60.0 / fs) % 1.0
    pulse = np.exp(-(((phase - 0.12) / 0.05) ** 2)) + 0.45 * np.exp(-(((phase - 0.38) / 0.09) ** 2))
    breathing = np.sin(2 * np.pi * 0.25 * t + float(rng.uniform(0, 2 * np.pi)))
    moving = np.zeros(n)
    for onset, seconds in movements:
        first = int(onset * fs)
        samples = min(n - first, int(seconds * fs))
        if samples >= 8:
            burst = band_limited_noise(samples, 0.3, 4.0, rng, fs=fs)
            moving[first:first + samples] += np.hanning(samples) * burst
    values = np.empty((n, len(OPTICS_CHANNELS)))
    for column, (dc, share) in enumerate(zip(_OPTICS_DC, _OPTICS_PULSE_SHARE)):
        values[:, column] = dc * (1.0 + 0.002 * breathing - share * pulse + 0.03 * moving)
        values[:, column] += 0.002 * rng.standard_normal(n)
    return values


# --------------------------------------------------------------------------- #
# BLE notifications
# --------------------------------------------------------------------------- #
# Decoder sensor types and the MuseFrame modality each one fills.
_MODALITIES = {"EEG": "eeg", "ACCGYRO": "imu", "OPTICS": "ppg", "BATTERY": "battery"}


@dataclass
class _Stream:
    name: str
    records: np.ndarray
    times: np.ndarray
    rows: int
    loss_fraction: float


def _chunked(encode, *arrays, chunk: int = 131_072) -> np.ndarray:
    total = arrays[0].shape[0]
    parts = [encode(*(array[start:start + chunk] for array in arrays)) for start in range(0, total, chunk)]
    return np.concatenate(parts) if parts else encode(*(array[:0] for array in arrays))


def _streams(config, eeg_counts, accel, gyro, optics, duration) -> List[_Stream]:
    """Every subpacket the device makes, with the time its last sample is ready."""
    streams = []
    count = eeg_counts.shape[0] // 4
    data = _chunked(enc.encode_eeg_counts, eeg_counts[: count * 4].reshape(count, 4, 4))
    streams.append(_Stream(
        "EEG",
        enc.subpacket_records(proto.TAG_EEG_4CH, np.arange(count), data),
        (np.arange(count) + 1) * 4 / EEG_RATE_HZ,
        4,
        config.eeg_loss_fraction,
    ))
    count = accel.shape[0] // 3
    data = _chunked(enc.encode_accgyro, accel.reshape(count, 3, 3), gyro.reshape(count, 3, 3))
    streams.append(_Stream(
        "ACCGYRO",
        enc.subpacket_records(proto.TAG_ACCGYRO, np.arange(count), data),
        _readout_times(
            (np.arange(count) + 1) * 3 / IMU_RATE_HZ,
            config.imu_bundle_shares,
            _rng(config.seed, _RNG_READOUT),
        ),
        3,
        config.imu_loss_fraction,
    ))
    if optics is not None:
        count = optics.shape[0] // 2
        data = _chunked(enc.encode_optics, optics.reshape(count, 2, len(OPTICS_CHANNELS)))
        streams.append(_Stream(
            "OPTICS",
            enc.subpacket_records(proto.TAG_OPTICS_8CH, np.arange(count), data),
            (np.arange(count) + 1) * 2 / OPTICS_RATE_HZ,
            2,
            config.optics_loss_fraction,
        ))
    times = np.arange(1, int(duration) + 1, dtype=float)
    percent = np.maximum(config.battery_start_percent - config.drain_percent_per_hour * times / 3600.0, 0.0)
    streams.append(_Stream(
        "BATTERY",
        enc.subpacket_records(proto.TAG_BATTERY_2, np.arange(times.size), enc.encode_battery(percent)),
        times,
        1,
        0.0,
    ))
    return streams


def _readout_times(times: np.ndarray, shares: Sequence[float], rng) -> np.ndarray:
    """``times`` moved to the last subpacket of each readout; readouts of 1, 2, ...
    consecutive subpackets come as often as ``shares`` says."""
    if times.size == 0:
        return times
    probabilities = np.asarray(shares, dtype=float) / float(sum(shares))
    sizes = rng.choice(np.arange(1, probabilities.size + 1), size=times.size, p=probabilities)
    ends = np.minimum(np.cumsum(sizes) - 1, times.size - 1)
    ends = ends[: int(np.searchsorted(ends, times.size - 1)) + 1]
    return times[ends[np.searchsorted(ends, np.arange(times.size))]]


def _ble_gaps(config, duration: float, rng) -> List[Tuple[float, float]]:
    if duration < 180.0:
        return []
    gaps: List[Tuple[float, float]] = []
    for _ in range(int(rng.poisson(config.ble_gaps_per_hour * duration / 3600.0))):
        seconds = float(rng.uniform(*config.ble_gap_seconds))
        start = float(rng.uniform(60.0, duration - 60.0 - seconds))
        if all(start > end + 5.0 or start + seconds < other - 5.0 for other, end in gaps):
            gaps.append((start, start + seconds))
    return sorted(gaps)


class _FixedStartRawStream(MuseRawStream):
    """A MuseRawStream whose header holds a fixed start, so the file is reproducible."""

    def __init__(self, filepath: Path, start_epoch_seconds: float) -> None:
        super().__init__(str(filepath))
        self._start_ms = int(round(start_epoch_seconds * 1000))

    def open_write(self):
        super().open_write()
        # open_write() stamps the wall clock into the header; overwrite it in place.
        self.file_handle.seek(0)
        self.file_handle.write(struct.pack("<4sBQ16s", b"MUSB", 2, self._start_ms, b"\x00" * 16))
        self.session_start = dt.datetime.fromtimestamp(self._start_ms / 1000)

    def write_at(self, data: bytes, milliseconds: int) -> None:
        # Half a millisecond on top, so the int() in write_packet lands on the intended value.
        self.write_packet(data, self.session_start + dt.timedelta(microseconds=milliseconds * 1000 + 500))


@dataclass
class _Copy:
    due: int
    original: int
    payload: bytes
    counts: Tuple[int, ...]
    between: List[int]


def _write_raw(raw_path: Path, streams: List[_Stream], duration: float, config) -> Dict[str, object]:
    """Pack the subpackets into notifications and write them; returns the BLE counts."""
    rng = _rng(config.seed, _RNG_BLE)
    interval = 1.0 / config.notifications_per_second
    count = int(duration / interval)
    ticks = np.arange(1, count + 1) * interval + rng.uniform(0.0, 0.004, count)
    ticks = ticks[ticks <= duration]
    tick_ms = np.floor(ticks * 1000.0).astype(np.int64)
    gaps = _ble_gaps(config, duration, rng)
    tick_lost = np.zeros(ticks.size, dtype=bool)
    for start, end in gaps:
        tick_lost |= (ticks >= start) & (ticks < end)

    subpackets: Dict[str, Dict[str, int]] = {}
    parts = []
    for order, stream in enumerate(streams):
        made = stream.times <= ticks[-1]
        stream.records, stream.times = stream.records[made], stream.times[made]
        total = stream.times.size
        tick_of = np.searchsorted(ticks, stream.times, side="left")
        lost = rng.random(total) < stream.loss_fraction
        if total:
            lost[[0, -1]] = False
        in_gap = tick_lost[tick_of] & ~lost
        delivered = ~(lost | in_gap)
        _check_lost_runs(stream.name, ~delivered)
        subpackets[stream.name] = {
            "produced": int(total),
            "delivered": int(delivered.sum()),
            "lost": int(lost.sum()),
            "lost_in_gaps": int(in_gap.sum()),
            "repeated": 0,
            "rows": int(delivered.sum()) * stream.rows,
        }
        index = np.flatnonzero(delivered)
        parts.append((tick_of[index], stream.times[index], np.full(index.size, order), index))
    tick_all = np.concatenate([part[0] for part in parts])
    time_all = np.concatenate([part[1] for part in parts])
    stream_all = np.concatenate([part[2] for part in parts])
    index_all = np.concatenate([part[3] for part in parts])
    sequence = np.lexsort((index_all, stream_all, time_all, tick_all))
    tick_all, stream_all, index_all = tick_all[sequence], stream_all[sequence], index_all[sequence]
    del time_all, sequence

    names = [stream.name for stream in streams]
    widths = [stream.records.shape[1] for stream in streams]
    blobs = []
    for stream in streams:
        blobs.append(stream.records.tobytes())
        stream.records = stream.records[:0]
    # A copied notification repeats the EEG subpackets in it, so the chance per
    # notification is the repeated share of all EEG subpackets.
    repeat_probability = config.eeg_repeats_per_hour / 3600.0 / (EEG_RATE_HZ / 4)
    frames_with = [0] * len(streams)
    unique = copies = 0
    pending: List[_Copy] = []
    raw = _FixedStartRawStream(raw_path, START_EPOCH_SECONDS)
    raw.open_write()
    try:
        for position, (payload, milliseconds, counts) in enumerate(
            _notifications(tick_all, stream_all, index_all, tick_ms, blobs, widths)
        ):
            raw.write_at(payload, milliseconds)
            unique += 1
            for order, value in enumerate(counts):
                frames_with[order] += value > 0
            if rng.random() < repeat_probability:
                offset = int(rng.integers(1, 4))
                pending.append(_Copy(position + offset - 1, position, payload, counts, [0] * len(streams)))
            due = []
            waiting = []
            for copy in pending:
                if copy.original < position:
                    copy.between = [a + b for a, b in zip(copy.between, counts)]
                (due if copy.due <= position else waiting).append(copy)
            pending = waiting
            for copy in due:
                # The decoder recognises a repeat among the last REPEAT_WINDOW subpackets
                # of each TAG; a copy that came later would read as new data.
                if all(
                    original + between <= proto.REPEAT_WINDOW
                    for original, between in zip(copy.counts, copy.between)
                    if original
                ):
                    raw.write_at(copy.payload, milliseconds)
                    copies += 1
                    for order, value in enumerate(copy.counts):
                        subpackets[names[order]]["repeated"] += value
    finally:
        raw.close()

    return {
        "notifications_per_second": config.notifications_per_second,
        "notifications": unique,
        "repeated_notifications": copies,
        "frames": unique + copies,
        "gaps": [[round(start, 3), round(end, 3)] for start, end in gaps],
        "subpackets": subpackets,
        "frames_with": {
            _MODALITIES[name]: frames_with[order]
            for order, name in enumerate(names)
            if frames_with[order]
        },
    }


def _check_lost_runs(name: str, missing: np.ndarray) -> None:
    if not missing.any():
        return
    edges = np.diff(np.concatenate([[0], missing.astype(np.int8), [0]]))
    longest = int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())
    if longest > _MAX_LOST_RUN:
        raise ValueError(f"{longest} {name} subpackets lost in a row; the counter would wrap")


def _notifications(tick_all, stream_all, index_all, tick_ms, blobs, widths) -> Iterator[tuple]:
    """(payload, milliseconds, subpackets per stream) of each notification in order: each
    tick's subpackets, split where 244 bytes are full."""
    count = 0
    records: List[bytes] = []
    size = enc.NOTIFICATION_PREFIX_BYTES
    counts = [0] * len(blobs)
    tick = -1
    extra = 0
    chunk = 200_000
    for start in range(0, tick_all.size, chunk):
        for current, order, index in zip(
            tick_all[start:start + chunk].tolist(),
            stream_all[start:start + chunk].tolist(),
            index_all[start:start + chunk].tolist(),
        ):
            width = widths[order]
            if current != tick or size + width > enc.MAX_NOTIFICATION_BYTES:
                if records:
                    yield enc.assemble_notification(records, count), int(tick_ms[tick]) + extra, tuple(counts)
                    count += 1
                extra = extra + 1 if current == tick else 0
                tick = current
                records = []
                size = enc.NOTIFICATION_PREFIX_BYTES
                counts = [0] * len(blobs)
            records.append(blobs[order][index * width:(index + 1) * width])
            size += width
            counts[order] += 1
    if records:
        yield enc.assemble_notification(records, count), int(tick_ms[tick]) + extra, tuple(counts)


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def _iso(offset_seconds: float) -> str:
    return dt.datetime.fromtimestamp(START_EPOCH_SECONDS + offset_seconds, tz=dt.timezone.utc).isoformat()


def _file_entry(path: Path) -> Dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_ground_truth(output_dir: Path, night: ScriptedNight, arousals, rem_events, duration: float):
    lines = ["epoch,start_s,end_s,stage"]
    for epoch, stage in enumerate(night.stages):
        start = epoch * EPOCH_SECONDS
        if start >= duration:
            break
        lines.append(f"{epoch},{start:.3f},{min(start + EPOCH_SECONDS, duration):.3f},{stage}")
    _write_text(output_dir / "hypnogram.csv", "\n".join(lines) + "\n")

    lines = ["onset_s,duration_s,stage,movement,position_change"]
    lines += [
        f"{a.onset_seconds:.3f},{a.duration_seconds:.3f},{a.stage},{int(a.movement)},{int(a.position_change)}"
        for a in arousals
    ]
    _write_text(output_dir / "arousals.csv", "\n".join(lines) + "\n")

    lines = ["onset_s,duration_s,epoch,amplitude_uv,af8_ratio,sign"]
    lines += [
        f"{e.onset_seconds:.4f},{e.duration_seconds:.4f},{int(e.onset_seconds // EPOCH_SECONDS)},"
        f"{e.amplitude_uv:.2f},{e.af8_ratio:.3f},{e.sign}"
        for e in rem_events
    ]
    _write_text(output_dir / "rem_events.csv", "\n".join(lines) + "\n")
    return {
        name: _file_entry(output_dir / name)
        for name in ("raw_amused.bin", "hypnogram.csv", "arousals.csv", "rem_events.csv")
    }


def _write_recording_files(output_dir: Path, config: SyntheticNightConfig, duration, stop_reason, ble):
    metadata = {
        "started_at": _iso(0.0),
        "synthetic": True,
        "note": SYNTHETIC_NOTE,
        "scenario": config.scenario,
        "seed": config.seed,
        "source": {
            "source_name": "synthetic",
            "device_name": "Synthetic Muse S Athena",
            "device_id": f"synthetic-{config.scenario}-{config.seed}",
            "capabilities": {
                "eeg": True,
                "imu": True,
                "ppg": config.has_optics,
                "heart_rate": config.has_optics,
                "battery": True,
                "raw_packets": True,
            },
            "metadata": {"preset": config.preset, "strategy": "synthetic", "generator": __name__},
        },
        "config": {"duration_seconds": config.hours * 3600.0, "source_name": "synthetic"},
    }
    _write_json(output_dir / "metadata.json", metadata)
    started = {"source": "synthetic", "note": SYNTHETIC_NOTE}
    events = [
        {"event": "recording_started", "timestamp": 0.0, "details": started},
        {"event": "recording_stopped", "timestamp": duration, "details": {"reason": stop_reason}},
    ]
    _write_text(output_dir / "events.jsonl", "".join(json.dumps(e, sort_keys=True) + "\n" for e in events))

    directory = output_dir.resolve()
    summary = RecordingSummary(
        output_dir=str(directory),
        raw_path=str(directory / "raw_amused.bin"),
        decoded_frames_path="",
        metadata_path=str(directory / "metadata.json"),
        events_path=str(directory / "events.jsonl"),
        summary_path=str(directory / "summary.json"),
        started_at=_iso(0.0),
        ended_at=_iso(duration),
        duration_seconds=duration,
        frame_count=int(ble["frames"]),
        raw_packet_count=int(ble["frames"]),
        decoded_frame_count=0,
        modality_counts=dict(ble["frames_with"]),
        reconnect_attempts=0,
        downtime_seconds=0.0,
        stop_reason=stop_reason,
    ).to_dict()
    summary.update({"synthetic": True, "note": SYNTHETIC_NOTE})
    _write_json(output_dir / "summary.json", summary)
    return {name: _file_entry(output_dir / name) for name in ("metadata.json", "events.jsonl")}
