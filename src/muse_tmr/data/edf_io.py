"""EDF+ export and import (#230), through edfio, the one library of the ``[edf]`` extra.

``edfio`` is pure Python on top of NumPy, so ``pip install "remind-eeg[edf]"`` needs no
compiler on any platform, and it is the writer MNE's own EDF export uses; pyedflib wraps
a C library and needs a wheel for each platform and Python. Neither writes EDF+D. edfio
reads EDF, EDF+C, BDF and BDF+C and writes EDF+C and BDF+C: a dropout is written
zero-filled and annotated ``BAD_ACQ_SKIP``, and ``read_edf()`` refuses an EDF+D file
rather than join its records as if they were contiguous.

``write_edf_plus()`` writes one EDF+C file of a ``ContinuousRecording``:

- 1 s data records at the nominal rates: EEG 256 Hz, accelerometer and gyroscope 52 Hz,
  optics 64 Hz, all starting at the recording's first frame, in UTC; the header's
  recording field holds the EEG rate measured on the night and the gap policy.
- EEG in uV with digital -8192 to 8191 over physical -725.044 to 724.9557 uV, the 14-bit
  counts of the headband exactly (1450/16383 uV per count; the 1450 uV full scale is
  vendor-stated and unverified). The accelerometer (g) and gyroscope (deg/s) keep their
  16-bit counts too. Optics are 20-bit, more than an EDF sample holds, so they and any
  signal not on its count grid (an imported one) are quantised to 16 bits over their
  own range; ``EdfExportResult.signals`` gives each signal's resolution.
- A TAL track with the night's annotations (``muse_tmr.data.night_events``), its zero
  fills as ``BAD_ACQ_SKIP`` and, at 0 s, the notes of a synthetic or imported night.
  Onsets are on the EEG sample clock, as the EEG samples are. A synthetic night also
  says so in the header's patient field.

``read_edf()`` turns an EDF back into MuseFrames through a channel map, so a third-party
file such as a BOAS headband recording (OpenNeuro ds005555) can be replayed; units are
converted to uV, g and deg/s where the header names them. ``read_edf_hypnogram()``
returns the stage annotations of an EDF+ file.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from muse_tmr.ble import muse_athena_protocol as proto
from muse_tmr.data.continuous import (
    ACCEL_CHANNELS,
    GYRO_CHANNELS,
    ContinuousConfig,
    ContinuousRecording,
    reconstruct_recording,
)
from muse_tmr.data.night_events import (
    SYNTHETIC_NOTE,
    NightAnnotation,
    StageInterval,
    collect_night_annotations,
    ensure_exportable,
    load_metadata,
    notes_for,
    stage_label,
)
from muse_tmr.data.sample_types import EEGSample, IMUSample, MuseFrame, PPGSample

EDF_EXTRA_HINT = 'EDF needs the edfio package: pip install "remind-eeg[edf]"'
RECORD_SECONDS = 1.0
EQUIPMENT_CODE = "Muse_S_Athena"
GAP_TEXT = "BAD_ACQ_SKIP"
EEG_COUNT_UV = proto.EEG_SCALE_UV_PER_COUNT
EEG_DIGITAL_RANGE = (-proto.EEG_MID_COUNT, proto.EEG_MAX_COUNT - proto.EEG_MID_COUNT)
INT16_RANGE = (-32768, 32767)
EEG_TRANSDUCER = "Muse S Athena dry electrode; 1450 uV full scale is vendor-stated, unverified"
EXPORT_STREAMS = ("eeg", "accel", "gyro", "optics")
EEG_CHANNEL_NAMES = tuple(proto.EEG_CHANNELS_8)
OPTICS_CHANNEL_NAMES = tuple(proto.OPTICS_CHANNELS_8)
# A synthetic night's EDF says so in its patient field. EDF+ separates subfields with
# spaces and allows more after the four it defines, so the note goes in word by word and
# the header reads "X X X X synthetic input, not sleep".
SYNTHETIC_SUBFIELDS = tuple(SYNTHETIC_NOTE.split())
_STREAM_UNITS = {"eeg": "uV", "accel": "g", "gyro": "deg/s", "optics": ""}
_STREAM_STEPS = {"eeg": EEG_COUNT_UV, "accel": proto.ACC_SCALE, "gyro": abs(proto.GYRO_SCALE)}
_EEG_UNITS = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "microvolt": 1.0, "microvolts": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}
_ACCEL_UNITS = {"g": 1.0, "mg": 1e-3, "m/s^2": 1 / 9.80665, "m/s2": 1 / 9.80665, "m/s²": 1 / 9.80665}
_GYRO_UNITS = {"deg/s": 1.0, "dps": 1.0, "°/s": 1.0, "degree/s": 1.0, "rad/s": 180.0 / math.pi}
_TAL_UNSAFE = re.compile(r"[\x00-\x1f\x7f]")
# The decoder names optics channels opt0, opt1, ... outside the 8-channel mode.
_DECODER_OPTICS_NAME = re.compile(r"^opt\d+$")

# The headband files of the BOAS dataset (OpenNeuro ds005555,
# sub-*_task-Sleep_acq-headband_eeg.edf) as its README and sidecars describe them: HB_1
# and HB_2 are forehead EEG near AF7 and AF8 (uV, EEGReference "M1 (left mastoid)"),
# HB_IMU_1 to 3 the accelerometer (g), 4 to 6 the gyroscope (deg/s) and HB_PULSE a pulse
# channel (nA), all at 256 Hz. Some nights have no IMU or pulse channels.
BOAS_HEADBAND_CHANNEL_MAP = {
    "HB_1": "eeg:AF7",
    "HB_2": "eeg:AF8",
    "HB_IMU_1": "accel:x",
    "HB_IMU_2": "accel:y",
    "HB_IMU_3": "accel:z",
    "HB_IMU_4": "gyro:x",
    "HB_IMU_5": "gyro:y",
    "HB_IMU_6": "gyro:z",
    "HB_PULSE": "ppg:HB_PULSE",
}
CHANNEL_MAPS = {"boas-headband": BOAS_HEADBAND_CHANNEL_MAP}


def _edfio():
    try:
        import edfio
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(EDF_EXTRA_HINT) from exc
    return edfio


def edf_available() -> bool:
    try:
        import edfio  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EdfSignalInfo:
    label: str
    stream: str
    unit: str
    rate_hz: float
    samples: int
    lossless: bool
    resolution: float
    physical_range: Tuple[float, float]
    digital_range: Tuple[int, int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "stream": self.stream,
            "unit": self.unit,
            "rate_hz": self.rate_hz,
            "samples": self.samples,
            "lossless": self.lossless,
            "resolution": self.resolution,
            "physical_range": list(self.physical_range),
            "digital_range": list(self.digital_range),
        }


@dataclass(frozen=True)
class EdfExportResult:
    path: str
    bytes: int
    records: int
    record_seconds: float
    duration_seconds: float
    start_utc: str
    signals: Tuple[EdfSignalInfo, ...]
    annotations: int
    gap_annotations: int
    measured_rate_hz: float
    gap_policy: str
    recording_summary: Mapping[str, object]
    notes: Tuple[str, ...] = ()
    synthetic: bool = False
    annotation_onsets: Tuple[Tuple[float, float, str], ...] = field(default=(), repr=False)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "path": self.path,
            "bytes": self.bytes,
            "records": self.records,
            "record_seconds": self.record_seconds,
            "duration_seconds": self.duration_seconds,
            "start_utc": self.start_utc,
            "signals": [signal.to_dict() for signal in self.signals],
            "annotations": self.annotations,
            "gap_annotations": self.gap_annotations,
            "measured_rate_hz": self.measured_rate_hz,
            "gap_policy": self.gap_policy,
            "recording": dict(self.recording_summary),
            "notes": list(self.notes),
        }
        if self.synthetic:
            payload = {"synthetic": True, "note": SYNTHETIC_NOTE, **payload}
        return payload


@dataclass(frozen=True)
class _Encoded:
    digital: np.ndarray
    physical_range: Tuple[float, float]
    digital_range: Tuple[int, int]
    lossless: bool
    resolution: float


def fit_header_number(value: float, mode: str = "nearest") -> float:
    """``value`` as the nearest number (or the next one down, ``floor``, or up, ``ceil``)
    that an 8-character EDF header field holds."""
    if value == 0 or not math.isfinite(value):
        if not math.isfinite(value):
            raise ValueError(f"not a finite number: {value}")
        return 0.0
    for decimals in range(7, -1, -1):
        factor = 10.0 ** decimals
        if mode == "floor":
            candidate = math.floor(value * factor) / factor
        elif mode == "ceil":
            candidate = math.ceil(value * factor) / factor
        else:
            candidate = round(value, decimals)
        text = str(int(candidate)) if float(candidate).is_integer() else str(candidate)
        if len(text) <= 8 and "e" not in text.lower():
            return float(text)
    raise ValueError(f"{value} does not fit an 8-character EDF header field")


def _encode(values: np.ndarray, step: Optional[float], digital_range: Tuple[int, int]) -> _Encoded:
    values = np.asarray(values, dtype=np.float64)
    if step is not None and values.size:
        counts = np.rint(values / step)
        within = counts.min() >= digital_range[0] and counts.max() <= digital_range[1]
        if within and float(np.max(np.abs(values / step - counts))) <= 0.01:
            low = fit_header_number(digital_range[0] * step)
            high = fit_header_number(digital_range[1] * step)
            return _Encoded(
                counts.astype(np.int16),
                (low, high),
                digital_range,
                True,
                (high - low) / (digital_range[1] - digital_range[0]),
            )
    finite = values[np.isfinite(values)]
    low_value = float(finite.min()) if finite.size else 0.0
    high_value = float(finite.max()) if finite.size else 1.0
    if high_value <= low_value:
        high_value = low_value + 1.0
    low = fit_header_number(low_value, "floor")
    high = fit_header_number(high_value, "ceil")
    d_low, d_high = INT16_RANGE
    scaled = (np.nan_to_num(values, nan=low) - low) / (high - low) * (d_high - d_low) + d_low
    digital = np.clip(np.rint(scaled), d_low, d_high).astype(np.int16)
    return _Encoded(digital, (low, high), INT16_RANGE, False, (high - low) / (d_high - d_low))


def _tal_text(text: str) -> str:
    cleaned = _TAL_UNSAFE.sub(" ", str(text)).strip()
    return cleaned or "annotation"


def _start_parts(t0: float) -> Tuple[dt.date, dt.time, dt.datetime]:
    moment = dt.datetime.fromtimestamp(t0, tz=dt.timezone.utc)
    return moment.date(), moment.time().replace(tzinfo=None), moment


def _recording_subfields(measured_rate_hz: float, gap_policy: str) -> Tuple[str, ...]:
    rate = f"{measured_rate_hz:.4f}" if math.isfinite(measured_rate_hz) else "unknown"
    return (f"measured_rate={rate}Hz", f"gaps={gap_policy}")


def write_edf_plus(
    recording: ContinuousRecording,
    output_path: Path,
    *,
    annotations: Sequence[NightAnnotation] = (),
    notes: Sequence[str] = (),
    synthetic: bool = False,
    streams: Sequence[str] = EXPORT_STREAMS,
) -> EdfExportResult:
    """Write ``recording`` as one EDF+C file (see the module docstring).

    ``annotations`` are placed by their host time on the EEG sample clock and clamped to
    the file's span. ``notes`` go in as annotations at 0 s. ValueError for a
    ``discontinuous`` recording: edfio writes EDF+C only.
    """
    edfio = _edfio()
    if recording.config.gap_policy != "zero_fill":
        raise ValueError("EDF+D is not written (edfio writes EDF+C only): export with gap_policy zero_fill")
    eeg = recording.eeg
    rate = eeg.nominal_rate_hz
    samples_per_record = rate * RECORD_SECONDS
    if not float(samples_per_record).is_integer():
        raise ValueError(f"{rate} Hz does not fill {RECORD_SECONDS} s records with whole samples")
    eeg_offset = recording.offset_samples("eeg")
    # Enough records for every stream: the IMU or optics can run a little past the EEG.
    records = max(
        (
            int(math.ceil((recording.offset_samples(name) + stream.n_samples) / (stream.nominal_rate_hz * RECORD_SECONDS)))
            for name, stream in recording.streams.items()
            if name in streams and stream.channels
        ),
        default=1,
    )
    records = max(1, records)
    duration = records * RECORD_SECONDS

    signals = []
    infos = []
    for name in streams:
        stream = recording.streams.get(name)
        if stream is None or not stream.channels:
            continue
        stream_rate = stream.nominal_rate_hz
        per_record = stream_rate * RECORD_SECONDS
        if not float(per_record).is_integer():
            raise ValueError(f"stream {name} at {stream_rate} Hz does not fill {RECORD_SECONDS} s records")
        total = int(records * per_record)
        offset = recording.offset_samples(name)
        for label, values in stream.channels.items():
            placed = np.zeros(total, dtype=np.float64)
            source_start = max(0, -offset)
            target_start = max(0, offset)
            count = max(0, min(len(values) - source_start, total - target_start))
            if count:
                placed[target_start:target_start + count] = values[source_start:source_start + count]
            step = _STREAM_STEPS.get(name)
            encoded = _encode(placed, step, EEG_DIGITAL_RANGE if name == "eeg" else INT16_RANGE)
            signal = edfio.EdfSignal.from_digital(
                encoded.digital,
                stream_rate,
                label=str(label)[:16],
                transducer_type=EEG_TRANSDUCER if name == "eeg" else f"Muse S Athena {name}",
                physical_dimension=_STREAM_UNITS[name],
                physical_range=encoded.physical_range,
                digital_range=encoded.digital_range,
                prefiltering="unknown (headband firmware)",
            )
            signals.append(signal)
            infos.append(
                EdfSignalInfo(
                    label=str(label)[:16],
                    stream=name,
                    unit=_STREAM_UNITS[name],
                    rate_hz=stream_rate,
                    samples=int(len(values)),
                    lossless=encoded.lossless,
                    resolution=encoded.resolution,
                    physical_range=encoded.physical_range,
                    digital_range=encoded.digital_range,
                )
            )

    edf_annotations, timed = _edf_annotations(recording, annotations, notes, duration, eeg_offset)
    gap_count = sum(1 for _, _, text in timed if text == GAP_TEXT)
    start_date, start_time, start_moment = _start_parts(recording.t0)
    patient = edfio.Patient(additional=SYNTHETIC_SUBFIELDS) if synthetic else edfio.Patient()
    recording_field = _recording_field(edfio, start_date, eeg.measured_rate_hz, recording.config.gap_policy)
    edf = edfio.Edf(
        signals,
        patient=patient,
        recording=recording_field,
        starttime=start_time,
        data_record_duration=RECORD_SECONDS,
        annotations=edf_annotations,
    )
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edf.write(output_path)
    return EdfExportResult(
        path=str(output_path),
        bytes=output_path.stat().st_size,
        records=records,
        record_seconds=RECORD_SECONDS,
        duration_seconds=duration,
        start_utc=start_moment.isoformat(),
        signals=tuple(infos),
        annotations=len(edf_annotations),
        gap_annotations=gap_count,
        measured_rate_hz=eeg.measured_rate_hz,
        gap_policy=recording.config.gap_policy,
        recording_summary=recording.summary(),
        notes=tuple(notes),
        synthetic=synthetic,
        annotation_onsets=tuple(timed),
    )


def _recording_field(edfio, start_date: dt.date, measured_rate_hz: float, gap_policy: str):
    subfields = list(_recording_subfields(measured_rate_hz, gap_policy))
    while True:
        try:
            return edfio.Recording(startdate=start_date, equipment_code=EQUIPMENT_CODE, additional=tuple(subfields))
        except ValueError:
            if not subfields:
                raise
            subfields.pop()


@dataclass(frozen=True)
class PlacedAnnotation:
    """An annotation on an export's time base: seconds from its start on the EEG sample
    clock, clamped to the export's span."""

    onset_s: float
    duration_s: float
    text: str
    kind: str


def place_annotations(
    recording: ContinuousRecording,
    annotations: Sequence[NightAnnotation],
    *,
    notes: Sequence[str] = (),
    duration: Optional[float] = None,
) -> List[PlacedAnnotation]:
    """The notes (at 0 s), the EEG's zero fills (``BAD_ACQ_SKIP``) and ``annotations``
    on the export's time base, sorted. A stretch that starts before 0 s is cut at 0, one
    wholly outside the export is left out, and onsets are rounded to the microsecond.
    ``duration`` is the export's span; default the EEG's."""
    eeg = recording.eeg
    rate = eeg.nominal_rate_hz
    eeg_offset = recording.offset_samples("eeg")
    data_end = (eeg_offset + eeg.n_samples) / rate
    span = data_end if duration is None else float(duration)
    placed: List[PlacedAnnotation] = [PlacedAnnotation(0.0, 0.0, _tal_text(note), "note") for note in notes]
    for marker in eeg.dropouts():
        if marker.filled and marker.samples > 0:
            placed.append(PlacedAnnotation((eeg_offset + marker.index) / rate, marker.samples / rate, GAP_TEXT, GAP_TEXT))
    if eeg_offset > 0:
        placed.append(PlacedAnnotation(0.0, eeg_offset / rate, GAP_TEXT, GAP_TEXT))
    if data_end < span - 1e-9:
        placed.append(PlacedAnnotation(data_end, span - data_end, GAP_TEXT, GAP_TEXT))
    if annotations:
        onsets = np.atleast_1d(recording.seconds(np.array([item.time_unix for item in annotations], dtype=float)))
        for item, onset in zip(annotations, onsets):
            onset = float(onset)
            length = max(float(item.duration_s), 0.0)
            if onset < 0:
                if length > 0 and onset + length <= 0:
                    continue
                length = max(0.0, length + onset) if length > 0 else 0.0
                onset = 0.0
            if onset > span:
                if length > 0:
                    continue
                onset = span
            length = min(length, max(0.0, span - onset))
            placed.append(PlacedAnnotation(onset, length, _tal_text(item.text), item.kind))
    placed = [
        PlacedAnnotation(round(item.onset_s, 6), round(item.duration_s, 6), item.text, item.kind) for item in placed
    ]
    placed.sort(key=lambda item: (item.onset_s, item.text))
    return placed


def _edf_annotations(
    recording: ContinuousRecording,
    annotations: Sequence[NightAnnotation],
    notes: Sequence[str],
    duration: float,
    eeg_offset: int,
):
    """The edfio annotations and their (onset, duration, text) in file seconds."""
    edfio = _edfio()
    placed = place_annotations(recording, annotations, notes=notes, duration=duration)
    timed = [(item.onset_s, item.duration_s, item.text) for item in placed]
    edf_annotations = [
        edfio.EdfAnnotation(onset, length if length > 0 else None, text) for onset, length, text in timed
    ]
    return edf_annotations, timed


def export_config(gap_policy: str = "zero_fill", gap_threshold_seconds: float = 0.5) -> ContinuousConfig:
    """The reconstruction an export uses: every stream, lost samples zero-filled."""
    return ContinuousConfig(
        gap_policy=gap_policy,
        gap_threshold_seconds=gap_threshold_seconds,
        streams=EXPORT_STREAMS,
    )


def export_edf(
    recording_dir: Path,
    output_path: Path,
    *,
    config: Optional[ContinuousConfig] = None,
    hypnograms: Sequence[Path] = (),
    night_schedule: Optional[Mapping[str, object]] = None,
) -> EdfExportResult:
    """Replay ``recording_dir`` and write it with its night's annotations as EDF+C.

    SealedNightError, before anything is read, for a night whose arm came from a night
    schedule that is still sealed (``night_events.blinding_refusal()``).
    """
    recording_dir = Path(recording_dir).expanduser()
    ensure_exportable(recording_dir, night_schedule)
    recording, annotations, metadata = load_export_inputs(
        recording_dir, config=config, hypnograms=hypnograms, night_schedule=night_schedule
    )
    return write_edf_plus(
        recording,
        output_path,
        annotations=annotations,
        notes=notes_for(metadata),
        synthetic=metadata.get("synthetic") is True,
    )


def load_export_inputs(
    recording_dir: Path,
    *,
    config: Optional[ContinuousConfig] = None,
    hypnograms: Sequence[Path] = (),
    night_schedule: Optional[Mapping[str, object]] = None,
):
    """The reconstruction, the annotations and the metadata an export writes."""
    recording = asyncio.run(reconstruct_recording(recording_dir, config or export_config()))
    annotations = collect_night_annotations(
        recording_dir,
        t0=recording.t0,
        hypnograms=hypnograms,
        to_unix=recording.unix_time,
        night_schedule=night_schedule,
    )
    return recording, annotations, load_metadata(recording_dir)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EdfChannel:
    label: str
    target: str
    name: str
    rate_hz: float
    unit: str
    scale: float


def _read(path: Path):
    edfio = _edfio()
    path = Path(path).expanduser()
    if path.suffix.lower() == ".bdf":
        edf = edfio.read_bdf(path)
    else:
        # Read at once: a lazily read file stays memory-mapped, and Windows cannot delete
        # or replace a mapped file.
        edf = edfio.read_edf(path, lazy_load_data=False)
    reserved = edf.reserved.strip().upper()
    if reserved.startswith(("EDF+D", "BDF+D")):
        raise ValueError(
            f"{path.name} is EDF+D (discontinuous); edfio would read its records as if contiguous, so it is not read"
        )
    return edf


def _parse_target(target: str) -> Tuple[str, str]:
    text = str(target).strip()
    if text.lower() == "skip":
        return "skip", ""
    kind, _, name = text.partition(":")
    if not name:
        return "eeg", kind
    kind = kind.strip().lower()
    if kind not in ("eeg", "accel", "gyro", "ppg", "skip"):
        raise ValueError(f"channel target must be eeg:, accel:, gyro:, ppg: or skip, got {target!r}")
    return kind, name.strip()


def default_channel_map(labels: Iterable[str]) -> Dict[str, str]:
    """The targets of the labels ``write_edf_plus()`` writes: EEG names, ACC_*, GYRO_*
    and optics names; other labels are left out."""
    mapping: Dict[str, str] = {}
    for label in labels:
        if label in EEG_CHANNEL_NAMES:
            mapping[label] = f"eeg:{label}"
        elif label in ACCEL_CHANNELS:
            mapping[label] = f"accel:{label[-1].lower()}"
        elif label in GYRO_CHANNELS:
            mapping[label] = f"gyro:{label[-1].lower()}"
        elif label in OPTICS_CHANNEL_NAMES or _DECODER_OPTICS_NAME.match(label):
            mapping[label] = f"ppg:{label}"
    return mapping


def resolve_channel_map(spec: Optional[object]) -> Optional[Dict[str, str]]:
    """A channel map from a mapping, a named map (``boas-headband``) or
    ``LABEL=TARGET,LABEL=TARGET`` text, where a target without a kind is EEG and
    ``skip`` leaves the signal out."""
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        return {str(key): str(value) for key, value in spec.items()}
    text = str(spec).strip()
    if text in CHANNEL_MAPS:
        return dict(CHANNEL_MAPS[text])
    mapping = {}
    for item in text.split(","):
        if not item.strip():
            continue
        label, separator, target = (part.strip() for part in item.partition("="))
        if not separator or not label or not target:
            raise ValueError(f"expected LABEL=TARGET in the channel map, got {item.strip()!r}")
        _parse_target(target)
        mapping[label] = target
    if not mapping:
        raise ValueError("the channel map is empty")
    return mapping


def _unit_scale(kind: str, unit: str, label: str, eeg_unit_scale: Optional[float]) -> float:
    key = unit.strip().lower().replace(" ", "")
    if kind == "eeg":
        if eeg_unit_scale is not None:
            return float(eeg_unit_scale)
        if key in _EEG_UNITS:
            return _EEG_UNITS[key]
        raise ValueError(f"EEG channel {label} is in {unit!r}, not a known voltage unit: give its scale to uV")
    if kind == "accel":
        return _ACCEL_UNITS.get(key, 1.0)
    if kind == "gyro":
        return _GYRO_UNITS.get(key, 1.0)
    return 1.0


def edf_channels(
    path: Path,
    channel_map: Optional[Mapping[str, str]] = None,
    *,
    eeg_unit_scale: Optional[float] = None,
) -> Tuple[EdfChannel, ...]:
    """The signals ``read_edf()`` takes from ``path`` and where they go."""
    return _channels_of(_read(path), channel_map, eeg_unit_scale)


def _channels_of(edf, channel_map: Optional[Mapping[str, str]], eeg_unit_scale: Optional[float]) -> Tuple[EdfChannel, ...]:
    labels = [signal.label for signal in edf.signals]
    # A label the file lacks is passed over: BOAS headband files without IMU or pulse
    # read through the same map as the ones with them.
    mapping = dict(channel_map) if channel_map is not None else default_channel_map(labels)
    channels = []
    for signal in edf.signals:
        target = mapping.get(signal.label)
        if target is None:
            continue
        kind, name = _parse_target(target)
        if kind == "skip":
            continue
        scale = _unit_scale(kind, signal.physical_dimension, signal.label, eeg_unit_scale)
        channels.append(EdfChannel(signal.label, kind, name, float(signal.sampling_frequency), signal.physical_dimension, scale))
    return tuple(channels)


def edf_start_unix(edf) -> float:
    """The start of an EDF as Unix seconds, its header read as UTC. An anonymised EDF+
    start date (``Startdate X``) counts as 1 January 1985, the date the EDF+ spec asks
    such a file to carry in its old header field."""
    try:
        start = edf.startdatetime
    except Exception:  # edfio.AnonymizedDateError, or an unreadable date
        start = dt.datetime.combine(dt.date(1985, 1, 1), edf.starttime)
    return start.replace(tzinfo=dt.timezone.utc).timestamp()


def read_edf(
    path: Path,
    *,
    channel_map: Optional[Mapping[str, str]] = None,
    frame_seconds: float = 0.25,
    skip_gaps: bool = True,
    eeg_unit_scale: Optional[float] = None,
    rates: Optional[Mapping[str, float]] = None,
    source: str = "edf",
) -> Iterator[MuseFrame]:
    """MuseFrames of ``frame_seconds`` from an EDF or BDF file.

    ``channel_map`` maps a signal label to ``eeg:NAME``, ``accel:x|y|z``,
    ``gyro:x|y|z``, ``ppg:NAME`` or ``skip``; without it the labels ``write_edf_plus()``
    writes are taken (``default_channel_map()``). A frame's timestamp is the time of its
    first sample, as ``continuous`` reads frames. With ``skip_gaps`` the stretches a
    ``BAD_ACQ_SKIP`` annotation covers are left out, frames end and start at their edges,
    and the frame after one says how many samples it skipped in its ``gaps``, as the
    decoder says it of lost subpackets: so a dropout an export filled is a dropout again,
    in the same place and of the same size. ``rates`` resamples a kind (``eeg``,
    ``accel``, ``gyro``, ``ppg``) whose rate differs to the one given, with
    ``scipy.signal.resample_poly``; ``resampled_rates()`` says which would be. ValueError
    for EDF+D, for a file with no channel mapped to EEG, for channels of one kind at
    different rates and for an EEG unit that is not a voltage (give ``eeg_unit_scale``,
    the factor to uV).
    """
    if frame_seconds <= 0:
        raise ValueError("frame_seconds must be positive")
    edf = _read(path)
    channels = _channels_of(edf, channel_map, eeg_unit_scale)
    if not any(channel.target == "eeg" for channel in channels):
        raise ValueError(f"{Path(path).name} has no channel mapped to EEG")
    by_label = {signal.label: signal for signal in edf.signals}
    kind_rates = _kind_rates(channels)
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for kind, rate in kind_rates.items():
        values = {
            channel.name: np.asarray(by_label[channel.label].data, dtype=np.float64) * channel.scale
            for channel in channels
            if channel.target == kind
        }
        target = float(rates[kind]) if rates and kind in rates else rate
        if target != rate:
            values = {name: _resample(array, rate, target) for name, array in values.items()}
            kind_rates[kind] = target
        data[kind] = values
    start = edf_start_unix(edf)
    previous_end: Optional[float] = None
    for span_start, span_end in _data_spans(float(edf.duration), _gap_spans(edf) if skip_gaps else []):
        skipped = None
        if previous_end is not None:
            # On each kind's own sample grid, so the counts add up to the file's.
            skipped = {
                kind: int(round(span_start * rate)) - int(round(previous_end * rate))
                for kind, rate in kind_rates.items()
            }
        begin = span_start
        while begin < span_end - 1e-9:
            end = min(span_end, begin + frame_seconds)
            frame = _frame(start + begin, begin, end, kind_rates, data, source, skipped)
            if frame is not None:
                yield frame
                skipped = None
            begin = end
        previous_end = span_end


def _kind_rates(channels: Sequence[EdfChannel]) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for kind in dict.fromkeys(channel.target for channel in channels):
        kind_rates = {channel.rate_hz for channel in channels if channel.target == kind}
        if len(kind_rates) != 1:
            raise ValueError(f"the {kind} channels have different rates: {sorted(kind_rates)}")
        rates[kind] = kind_rates.pop()
    return rates


def resampled_rates(
    path: Path,
    rates: Mapping[str, float],
    *,
    channel_map: Optional[Mapping[str, str]] = None,
    eeg_unit_scale: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:
    """``{kind: (file rate, rate)}`` of each kind ``read_edf(rates=rates)`` resamples."""
    kind_rates = _kind_rates(_channels_of(_read(path), channel_map, eeg_unit_scale))
    return {
        kind: (rate, float(rates[kind]))
        for kind, rate in kind_rates.items()
        if kind in rates and float(rates[kind]) != rate
    }


def _resample(values: np.ndarray, rate: float, target: float) -> np.ndarray:
    from fractions import Fraction

    from scipy.signal import resample_poly

    ratio = Fraction(target / rate).limit_denominator(1000)
    if ratio.numerator <= 0:
        raise ValueError(f"cannot resample {rate} Hz to {target} Hz")
    return np.asarray(resample_poly(values, ratio.numerator, ratio.denominator), dtype=np.float64)


def _gap_spans(edf) -> List[Tuple[float, float]]:
    spans = []
    for annotation in edf.annotations:
        if str(annotation.text).startswith(GAP_TEXT) and annotation.duration:
            spans.append((float(annotation.onset), float(annotation.onset) + float(annotation.duration)))
    return spans


def _data_spans(duration: float, gaps: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """``[0, duration)`` less the ``gaps``, as (start, end) in order."""
    spans = []
    position = 0.0
    for gap_start, gap_end in sorted(gaps):
        gap_start, gap_end = max(0.0, gap_start), min(duration, gap_end)
        if gap_start > position + 1e-9:
            spans.append((position, gap_start))
        position = max(position, gap_end)
    if duration > position + 1e-9:
        spans.append((position, duration))
    return spans


def _slice(values: Mapping[str, np.ndarray], rate: float, begin: float, end: float) -> Dict[str, Tuple[float, ...]]:
    first = int(round(begin * rate))
    stop = int(round(end * rate))
    return {name: tuple(array[first:stop].tolist()) for name, array in values.items()}


def _frame(timestamp, begin, end, rates, data, source, skipped=None) -> Optional[MuseFrame]:
    def gaps(kind):
        missing = (skipped or {}).get(kind, 0)
        return ((0, missing),) if missing > 0 else ()

    eeg = imu = ppg = None
    if "eeg" in data:
        channels = _slice(data["eeg"], rates["eeg"], begin, end)
        if any(channels.values()):
            eeg = EEGSample(timestamp, channels, source=source, gaps=gaps("eeg"))
    accel_rows = gyro_rows = None
    if "accel" in data:
        accel = _slice(data["accel"], rates["accel"], begin, end)
        accel_rows = _rows(accel)
    if "gyro" in data:
        gyro = _slice(data["gyro"], rates["gyro"], begin, end)
        gyro_rows = _rows(gyro)
    if accel_rows or gyro_rows:
        imu = IMUSample(
            timestamp,
            accelerometer_g=accel_rows or None,
            gyroscope_dps=gyro_rows or None,
            source=source,
            gaps=gaps("accel" if "accel" in data else "gyro"),
        )
    if "ppg" in data:
        channels = _slice(data["ppg"], rates["ppg"], begin, end)
        if any(channels.values()):
            ppg = PPGSample(timestamp, channels, source=source, gaps=gaps("ppg"))
    if eeg is None and imu is None and ppg is None:
        return None
    return MuseFrame(timestamp=timestamp, eeg=eeg, imu=imu, ppg=ppg, source=source)


def _rows(axes: Mapping[str, Tuple[float, ...]]) -> Tuple[Dict[str, float], ...]:
    count = max((len(values) for values in axes.values()), default=0)
    return tuple(
        {axis: (axes[axis][row] if axis in axes and row < len(axes[axis]) else 0.0) for axis in ("x", "y", "z")}
        for row in range(count)
    )


def read_edf_annotations(path: Path) -> Tuple[Tuple[float, Optional[float], str], ...]:
    """(onset, duration, text) of every EDF+ annotation, onsets from the file start."""
    edf = _read(path)
    return tuple((float(item.onset), item.duration, str(item.text)) for item in edf.annotations)


def is_stage_text(text: str) -> bool:
    """Whether an annotation text is a sleep stage: an EDF+ ``Sleep stage`` text or a
    bare stage name (W, N1, R, ...)."""
    cleaned = str(text).strip()
    return cleaned.lower().startswith("sleep stage") or (
        stage_label(cleaned) != "unknown" and not cleaned.lstrip("-").isdigit()
    )


def read_edf_hypnogram(path: Path) -> Tuple[StageInterval, ...]:
    """The stage annotations of an EDF+ file as intervals; a stage without a duration
    lasts 30 s."""
    intervals = []
    for onset, duration, text in read_edf_annotations(path):
        if not is_stage_text(text):
            continue
        length = float(duration) if duration else 30.0
        intervals.append(StageInterval(onset, length, text.strip(), stage_label(text)))
    return tuple(sorted(intervals, key=lambda interval: interval.onset_s))


def edf_header(path: Path) -> Dict[str, object]:
    """What an import records about an EDF: start, duration, signals and header notes.
    The patient and recording fields are not copied: a third-party file may name a
    person or a device there."""
    edf = _read(path)
    notes = [str(item.text) for item in edf.annotations if float(item.onset) == 0.0]
    return {
        "start_utc": dt.datetime.fromtimestamp(edf_start_unix(edf), tz=dt.timezone.utc).isoformat(),
        "duration_seconds": float(edf.duration),
        "reserved": edf.reserved,
        "signals": [
            {"label": s.label, "rate_hz": float(s.sampling_frequency), "unit": s.physical_dimension}
            for s in edf.signals
        ],
        "synthetic": SYNTHETIC_NOTE in edf.local_patient_identification or SYNTHETIC_NOTE in notes,
    }
