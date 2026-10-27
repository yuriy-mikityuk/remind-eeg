"""Nights recorded by other apps, as REMind recording directories (#230).

``import_recording()`` reads a file in one of ``IMPORT_FORMATS`` and writes
``decoded_frames.jsonl``, ``metadata.json`` and ``summary.json``, which ``replay``, the
REM report, ``annotate-template`` and ``simulate-replay-cues`` read like any recording.
An import has no raw packets (``raw_packet_count`` 0, ``capabilities.raw_packets``
false), so ``validate-pilot1-recording`` fails it, and nothing that plays sound runs on
it: report and simulation only, as the banner every report of it carries says.

EEG values become centred microvolts as ``value * eeg_scale + eeg_offset``. The scale
comes from ``IMPORT_SCALE_TABLE``, which names where each row comes from, or from
``--eeg-scale``; Mind Monitor's units are unverified, so its imports need one. An EDF's
values are converted from the unit its header names, or with ``--eeg-scale`` in its
place, and every stream is resampled to the rate the pipeline reads (EEG 256 Hz, IMU
52 Hz, optics 64 Hz) when it has another. What each format keeps and loses:
``docs/import.md``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from muse_tmr import __version__
from muse_tmr.ble import muse_athena_protocol as proto
from muse_tmr.data.continuous import ContinuousConfig
from muse_tmr.data.night_events import SYNTHETIC_NOTE
from muse_tmr.data.replay import DECODED_FRAMES_FILENAME
from muse_tmr.data.sample_types import EEGSample, MuseFrame

IMPORT_SCHEMA_VERSION = 1
IMPORT_FORMATS = ("decoded-frames", "edf", "muselsl-csv", "openmuse-csv", "brainflow-csv", "mindmonitor-csv")
IMPORT_NOTE = "report and simulation only"
EEG_CHANNELS = tuple(proto.EEG_CHANNELS_4)
# The rates the epoch builder and the feature extractors read; an EDF stream at another
# rate is resampled to them.
_RATES = ContinuousConfig()
PIPELINE_RATES = {
    "eeg": _RATES.nominal_rate_hz,
    "accel": _RATES.imu_rate_hz,
    "gyro": _RATES.imu_rate_hz,
    "ppg": _RATES.optics_rate_hz,
}
# Centring the 14-bit counts: count 8192 is 8192 * 1450/16383 uV.
MIDSCALE_UV = proto.EEG_MID_COUNT * proto.EEG_SCALE_UV_PER_COUNT
# BrainFlow before 5.23.0 scaled Athena counts by 1650/4095, the 12-bit factor.
BRAINFLOW_OLD_SCALE = 1650.0 / 4095.0
BRAINFLOW_FIXED_VERSION = (5, 23, 0)
SAMPLES_PER_FRAME = 12
# Unix seconds before 2000-01-01 are read as a clock that counts from the stream start.
_UNIX_FLOOR = 946684800.0


@dataclass(frozen=True)
class ImportScale:
    """How one format's EEG values become centred microvolts, and the evidence."""

    fmt: str
    applies_to: str
    eeg_scale: Optional[float]
    eeg_offset_uv: float
    provenance: str
    status: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "format": self.fmt,
            "applies_to": self.applies_to,
            "eeg_scale": self.eeg_scale,
            "eeg_offset_uv": self.eeg_offset_uv,
            "provenance": self.provenance,
            "status": self.status,
        }


IMPORT_SCALE_TABLE: Tuple[ImportScale, ...] = (
    ImportScale(
        "decoded-frames",
        "decoded_frames.jsonl of muse-tmr record",
        1.0,
        0.0,
        "written by this repository's decoder in centred uV (muse_tmr.ble.muse_athena_protocol)",
        "same decoder",
    ),
    ImportScale(
        "edf",
        "EDF or BDF with EEG in uV, mV or V",
        1.0,
        0.0,
        "physical values of the file, converted from the unit its header names",
        "header units",
    ),
    ImportScale(
        "muselsl-csv",
        "muse-lsl record CSV of a Muse S Athena (Athena support since muse-lsl PR #228, merged 2026-06-30)",
        1.0,
        0.0,
        "muselsl/athena.py decodes (raw - 8192) * 1450/16383, the centred uV of this repository",
        "source code read 2026-10-27",
    ),
    ImportScale(
        "openmuse-csv",
        "the EEG DataFrame of OpenMuse.decode_rawdata(), saved as CSV",
        1.0,
        -MIDSCALE_UV,
        "OpenMuse/decode.py scales raw * 1450/16383 without subtracting 8192, a pedestal of 725.04 uV",
        "source code read 2026-10-27",
    ),
    ImportScale(
        "brainflow-csv",
        "BrainFlow 5.23.0 or later, MUSE_S_ATHENA_BOARD default preset, DataFilter.write_file",
        1.0,
        0.0,
        "BrainFlow PR #855 (scale 1450/16383) and PR #856 (centred on 8192), merged 2026-08-18, first in 5.23.0",
        "source code read 2026-10-27",
    ),
    ImportScale(
        "brainflow-csv",
        "BrainFlow before 5.23.0, MUSE_S_ATHENA_BOARD default preset, DataFilter.write_file",
        proto.EEG_SCALE_UV_PER_COUNT / BRAINFLOW_OLD_SCALE,
        -MIDSCALE_UV,
        "before PR #855 BrainFlow scaled Athena counts by 1650/4095 uV, 4.55 times 1450/16383, and before "
        "PR #856 it did not centre them on 8192",
        "source code read 2026-10-27",
    ),
    ImportScale(
        "mindmonitor-csv",
        "Mind Monitor CSV export (RAW_TP9 ... RAW_TP10)",
        None,
        0.0,
        "no primary document of Mind Monitor's EEG units for the Athena was found",
        "unverified",
    ),
)


@dataclass(frozen=True)
class ImportResult:
    output_dir: str
    fmt: str
    frame_count: int
    eeg_samples: int
    modality_counts: Mapping[str, int]
    start_utc: str
    duration_seconds: float
    banner: str
    synthetic: bool = False

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "output_dir": self.output_dir,
            "format": self.fmt,
            "frame_count": self.frame_count,
            "eeg_samples": self.eeg_samples,
            "modality_counts": dict(self.modality_counts),
            "start_utc": self.start_utc,
            "duration_seconds": self.duration_seconds,
            "banner": self.banner,
        }
        if self.synthetic:
            payload = {"synthetic": True, "note": SYNTHETIC_NOTE, **payload}
        return payload


def scale_banner(fmt: str, eeg_scale: float) -> str:
    return f"imported: {fmt}, scale {eeg_scale:g}; {IMPORT_NOTE}"


def resolve_scale(
    fmt: str,
    *,
    eeg_scale: Optional[float] = None,
    eeg_offset: Optional[float] = None,
    source_version: Optional[str] = None,
) -> Tuple[ImportScale, str]:
    """The scale row an import uses and who declared it (``table`` or ``user``).

    ``--eeg-scale`` replaces the table: then the offset is ``--eeg-offset`` or 0.
    ``brainflow-csv`` needs ``source_version``, the BrainFlow version that wrote the
    file, or a scale, since the two table rows differ 4.55 times. ValueError for a
    format without a verified row and no scale (``mindmonitor-csv``).
    """
    if fmt not in IMPORT_FORMATS:
        raise ValueError(f"format must be one of: {', '.join(IMPORT_FORMATS)}")
    if eeg_scale is not None:
        if not math.isfinite(eeg_scale) or eeg_scale == 0:
            raise ValueError("--eeg-scale must be a finite number other than 0")
        offset = float(eeg_offset or 0.0)
        return ImportScale(fmt, "declared with --eeg-scale", float(eeg_scale), offset, "given by the user", "user-declared"), "user"
    rows = [row for row in IMPORT_SCALE_TABLE if row.fmt == fmt]
    if fmt == "brainflow-csv":
        if source_version is None:
            raise ValueError(
                "brainflow-csv needs --source-version, the BrainFlow version that wrote the file "
                "(before 5.23.0 its EEG is 4.55 times too large and uncentred), or --eeg-scale"
            )
        rows = [rows[0] if _version(source_version) >= BRAINFLOW_FIXED_VERSION else rows[1]]
    row = rows[0]
    if row.eeg_scale is None:
        raise ValueError(
            f"{fmt} needs --eeg-scale (and --eeg-offset if its values are not centred): {row.provenance}"
        )
    if eeg_offset is not None:
        row = ImportScale(row.fmt, row.applies_to, row.eeg_scale, float(eeg_offset), row.provenance, row.status)
    return row, "table"


def _version(text: str) -> Tuple[int, ...]:
    parts = []
    for piece in str(text).strip().lstrip("v").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


# --------------------------------------------------------------------------- #
# Readers: each yields frames with the format's own EEG values
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Clock:
    kind: str
    description: str
    offset: float = 0.0


def _resolve_clock(first: float, last: float, path: Path, start_time: Optional[float], unix_kind: str) -> _Clock:
    if first >= _UNIX_FLOOR:
        return _Clock("unix", unix_kind)
    if start_time is not None:
        return _Clock("relative", "relative to the stream start, anchored at --start-time", start_time - first)
    anchor = path.stat().st_mtime - max(0.0, last - first)
    return _Clock(
        "relative",
        "relative to the stream start, anchored at the file's modification time minus its length (approximate)",
        anchor - first,
    )


def _frames_from_columns(
    timestamps: np.ndarray,
    columns: Mapping[str, np.ndarray],
    source: str,
) -> Iterator[MuseFrame]:
    """Frames of up to SAMPLES_PER_FRAME rows; a frame never spans a jump in time."""
    count = len(timestamps)
    start = 0
    while start < count:
        stop = min(count, start + SAMPLES_PER_FRAME)
        if stop - start > 1:
            steps = np.diff(timestamps[start:stop])
            jumps = np.nonzero(steps > 0.1)[0]
            if jumps.size:
                stop = start + int(jumps[0]) + 1
        timestamp = float(timestamps[start])
        channels = {name: values[start:stop].tolist() for name, values in columns.items()}
        yield MuseFrame(timestamp=timestamp, eeg=EEGSample(timestamp, channels, source=source), source=source)
        start = stop


def _read_csv(path: Path, **kwargs):
    import pandas as pd

    # One pass over the whole file: Mind Monitor's Elements column mixes text and blanks,
    # which chunked type guessing warns about.
    return pd.read_csv(path, low_memory=False, **kwargs)


def _eeg_columns(frame, names: Sequence[str], aliases: Mapping[str, str]) -> Dict[str, np.ndarray]:
    columns = {}
    for column in names:
        values = frame[column].to_numpy(dtype=float)
        columns[aliases.get(column, column)] = values
    return columns


def _read_muselsl(path: Path, start_time: Optional[float]):
    table = _read_csv(path)
    if "timestamps" not in table.columns:
        raise ValueError(f"{path.name} is not a muse-lsl CSV: no timestamps column")
    present = [name for name in EEG_CHANNELS if name in table.columns]
    if not present:
        raise ValueError(f"{path.name} has none of the EEG columns {', '.join(EEG_CHANNELS)}")
    table = table.dropna(subset=present + ["timestamps"])
    times = table["timestamps"].to_numpy(dtype=float)
    clock = _resolve_clock(times[0], times[-1], path, start_time, "muse-lsl timestamps read as Unix seconds")
    return times + clock.offset, _eeg_columns(table, present, {}), clock


def _read_openmuse(path: Path, start_time: Optional[float]):
    table = _read_csv(path)
    if "time" not in table.columns:
        raise ValueError(f"{path.name} is not an OpenMuse EEG CSV: no time column")
    aliases = {f"EEG_{name}": name for name in EEG_CHANNELS}
    present = [column for column in aliases if column in table.columns]
    if not present:
        raise ValueError(f"{path.name} has none of the columns {', '.join(aliases)}")
    table = table.dropna(subset=present + ["time"])
    times = table["time"].to_numpy(dtype=float)
    clock = _resolve_clock(times[0], times[-1], path, start_time, "OpenMuse time read as Unix seconds")
    return times + clock.offset, _eeg_columns(table, present, aliases), clock


# BrainFlow MUSE_S_ATHENA_BOARD, default preset: 11 rows, EEG TP9, AF7, AF8, TP10 in rows
# 1 to 4 and the Unix timestamp in row 9 (src/board_controller/brainflow_boards.cpp).
BRAINFLOW_EEG_ROWS = {"TP9": 1, "AF7": 2, "AF8": 3, "TP10": 4}
BRAINFLOW_TIMESTAMP_ROW = 9
BRAINFLOW_ROWS = 11


def _read_brainflow(path: Path, start_time: Optional[float]):
    table = _read_csv(path, sep="\t", header=None)
    if table.shape[1] != BRAINFLOW_ROWS:
        raise ValueError(
            f"{path.name} has {table.shape[1]} columns; a BrainFlow MUSE_S_ATHENA_BOARD default-preset file "
            f"written by DataFilter.write_file has {BRAINFLOW_ROWS}"
        )
    times = table[BRAINFLOW_TIMESTAMP_ROW].to_numpy(dtype=float)
    columns = {name: table[row].to_numpy(dtype=float) for name, row in BRAINFLOW_EEG_ROWS.items()}
    clock = _resolve_clock(times[0], times[-1], path, start_time, "BrainFlow timestamp channel, Unix seconds")
    return times + clock.offset, columns, clock


MINDMONITOR_EEG_COLUMNS = {f"RAW_{name}": name for name in EEG_CHANNELS}


def _read_mindmonitor(path: Path, start_time: Optional[float]):
    table = _read_csv(path)
    if "TimeStamp" not in table.columns:
        raise ValueError(f"{path.name} is not a Mind Monitor CSV: no TimeStamp column")
    present = [column for column in MINDMONITOR_EEG_COLUMNS if column in table.columns]
    if not present:
        raise ValueError(f"{path.name} has none of the columns {', '.join(MINDMONITOR_EEG_COLUMNS)}")
    table = table.dropna(subset=present)
    stamps = [dt.datetime.fromisoformat(str(value).strip()) for value in table["TimeStamp"]]
    times = np.array(
        [stamp.timestamp() if stamp.tzinfo else stamp.astimezone().timestamp() for stamp in stamps],
        dtype=float,
    )
    clock = _Clock("local", "Mind Monitor TimeStamp, local time without a zone, read in this machine's zone")
    if start_time is not None:
        clock = _Clock("relative", "Mind Monitor TimeStamp, anchored at --start-time", start_time - times[0])
    return times + clock.offset, _eeg_columns(table, present, MINDMONITOR_EEG_COLUMNS), clock


_CSV_READERS = {
    "muselsl-csv": _read_muselsl,
    "openmuse-csv": _read_openmuse,
    "brainflow-csv": _read_brainflow,
    "mindmonitor-csv": _read_mindmonitor,
}


def _scaled(frame: MuseFrame, scale: float, offset: float) -> MuseFrame:
    if frame.eeg is None or (scale == 1.0 and offset == 0.0):
        return frame
    channels = {
        name: tuple(value * scale + offset for value in values) for name, values in frame.eeg.channels_uv.items()
    }
    eeg = EEGSample(frame.eeg.timestamp, channels, source=frame.eeg.source, gaps=frame.eeg.gaps)
    return MuseFrame(
        timestamp=frame.timestamp,
        eeg=eeg,
        imu=frame.imu,
        ppg=frame.ppg,
        heart_rate=frame.heart_rate,
        battery=frame.battery,
        source=frame.source,
    )


def _source_frames(
    path: Path,
    fmt: str,
    scale: ImportScale,
    *,
    declared_by: str,
    start_time: Optional[float],
    channel_map: Optional[Mapping[str, str]],
    source: str,
) -> Tuple[Iterator[MuseFrame], Dict[str, object]]:
    """The frames of ``path`` in centred uV, and what the import records."""
    factor = float(scale.eeg_scale)  # type: ignore[arg-type]
    offset = float(scale.eeg_offset_uv)
    if fmt == "decoded-frames":
        frames = (_scaled(frame, factor, offset) for frame in _decoded_frames(path, source))
        return frames, {
            "clock": {"kind": "unix", "description": "frame timestamps of the recorder, Unix seconds"},
            "synthetic": _says_synthetic(path.parent / "metadata.json"),
        }
    if fmt == "edf":
        from muse_tmr.data.edf_io import edf_header, read_edf, resampled_rates

        # A declared scale stands in for the header's unit; the table's row (1.0) keeps it.
        unit_scale = factor if declared_by == "user" else None
        header = edf_header(path)
        resampled = resampled_rates(path, PIPELINE_RATES, channel_map=channel_map, eeg_unit_scale=unit_scale)
        details = {
            "clock": {"kind": "unix", "description": "EDF start, read as UTC, plus the sample index"},
            "edf": {key: header[key] for key in ("start_utc", "duration_seconds", "signals")},
            "channel_map": dict(channel_map) if channel_map else None,
            "eeg_unit": "--eeg-scale in place of the header's unit" if unit_scale is not None else "header",
            "resampled": {
                kind: {"from_hz": rates[0], "to_hz": rates[1]} for kind, rates in sorted(resampled.items())
            },
            "synthetic": bool(header.get("synthetic")),
        }
        frames = (
            _scaled(frame, 1.0, offset)
            for frame in read_edf(
                path, channel_map=channel_map, eeg_unit_scale=unit_scale, rates=PIPELINE_RATES, source=source
            )
        )
        return frames, details
    times, columns, clock = _CSV_READERS[fmt](path, start_time)
    if times.size == 0:
        raise ValueError(f"{path.name} holds no EEG rows")
    order = np.argsort(times, kind="stable")
    times = times[order]
    columns = {name: values[order] * factor + offset for name, values in columns.items()}
    details = {"clock": {"kind": clock.kind, "description": clock.description}, "synthetic": False}
    if clock.offset:
        details["clock"]["offset_seconds"] = clock.offset  # type: ignore[index]
    return _frames_from_columns(times, columns, source), details


def _decoded_frames(path: Path, source: str) -> Iterator[MuseFrame]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                frame = MuseFrame.from_json(line)
                yield MuseFrame(
                    timestamp=frame.timestamp,
                    eeg=frame.eeg,
                    imu=frame.imu,
                    ppg=frame.ppg,
                    heart_rate=frame.heart_rate,
                    battery=frame.battery,
                    source=source,
                )


def _says_synthetic(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("synthetic") is True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(seconds: float) -> str:
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).isoformat()


def default_import_name(path: Path) -> str:
    """The directory an import goes into by default under ``imported/``: the input's
    stem, or for a recorder's ``decoded_frames.jsonl``, a name every recording uses, the
    name of the recording directory it is in."""
    path = Path(path)
    if path.name == DECODED_FRAMES_FILENAME and path.parent.name:
        return path.parent.name
    return path.stem


def _check_output_dir(output_dir: Path, input_sha256: str) -> None:
    """An import writes into a new or empty directory, or over an earlier import of the
    same file (as with another scale), never over another file's."""
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"{output_dir} is not a directory")
    names = [path.name for path in output_dir.iterdir() if not path.name.startswith(".")]
    if not names:
        return
    metadata = output_dir / "metadata.json"
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
    except ValueError:
        payload = {}
    block = payload.get("import") if isinstance(payload, dict) else None
    if not isinstance(block, dict) or "raw_amused.bin" in names:
        raise ValueError(f"{output_dir} holds files that are not an earlier import; import into a new directory")
    if block.get("input_sha256") not in (None, input_sha256):
        raise ValueError(
            f"{output_dir} holds the import of another file ({block.get('input_name')}); "
            "import into a new directory, or remove that one first"
        )


def import_recording(
    path: Path,
    fmt: str,
    out_dir: Path,
    eeg_scale: Optional[float] = None,
    eeg_offset: Optional[float] = None,
    *,
    source_version: Optional[str] = None,
    start_time: Optional[float] = None,
    channel_map: Optional[Mapping[str, str]] = None,
) -> ImportResult:
    """Import ``path`` (format ``fmt``) into the recording directory ``out_dir``.

    ``start_time`` (Unix seconds) anchors a clock that counts from the stream start
    (muse-lsl's LSL clock, OpenMuse's ``time``); without it the file's modification time
    less its length does, recorded as approximate. ``channel_map`` is for ``edf``
    (``edf_io.read_edf()``). ValueError when the scale cannot be resolved
    (``resolve_scale()``), for an unreadable file and for an output directory that holds
    anything but an earlier import of the same file.
    """
    path = Path(path).expanduser()
    out_dir = Path(out_dir).expanduser()
    if not path.is_file():
        raise ValueError(f"import input not found: {path}")
    scale, declared_by = resolve_scale(fmt, eeg_scale=eeg_scale, eeg_offset=eeg_offset, source_version=source_version)
    if channel_map is not None and fmt != "edf":
        raise ValueError("a channel map is for --format edf only")
    input_sha256 = _sha256(path)
    _check_output_dir(out_dir, input_sha256)
    source = f"import:{fmt}"
    frames, details = _source_frames(
        path,
        fmt,
        scale,
        declared_by=declared_by,
        start_time=start_time,
        channel_map=channel_map,
        source=source,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / DECODED_FRAMES_FILENAME
    tmp_path = frames_path.with_suffix(".jsonl.tmp")
    frame_count = 0
    eeg_samples = 0
    modality_counts: Dict[str, int] = {}
    first = last = None
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for frame in frames:
                handle.write(frame.to_json(include_raw=False) + "\n")
                frame_count += 1
                for modality in frame.modalities():
                    modality_counts[modality] = modality_counts.get(modality, 0) + 1
                if frame.eeg is not None and frame.eeg.channels_uv:
                    eeg_samples += len(next(iter(frame.eeg.channels_uv.values())))
                first = frame.timestamp if first is None else first
                last = frame.timestamp
        if frame_count == 0 or first is None or last is None:
            raise ValueError(f"{path.name} gave no frames")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, frames_path)

    synthetic = bool(details.pop("synthetic", False))
    banner = scale_banner(fmt, float(scale.eeg_scale))  # type: ignore[arg-type]
    import_block = {
        "schema_version": IMPORT_SCHEMA_VERSION,
        "format": fmt,
        "eeg_scale": scale.eeg_scale,
        "eeg_offset_uv": scale.eeg_offset_uv,
        "scale_declared_by": declared_by,
        "scale_applies_to": scale.applies_to,
        "scale_provenance": scale.provenance,
        "scale_status": scale.status,
        "source_version": source_version,
        "input_name": path.name,
        "input_bytes": path.stat().st_size,
        "input_sha256": input_sha256,
        "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "muse_tmr_version": __version__,
        "note": IMPORT_NOTE,
        "banner": banner,
        **details,
    }
    capabilities = {
        "eeg": modality_counts.get("eeg", 0) > 0,
        "imu": modality_counts.get("imu", 0) > 0,
        "ppg": modality_counts.get("ppg", 0) > 0,
        "heart_rate": modality_counts.get("heart_rate", 0) > 0,
        "battery": modality_counts.get("battery", 0) > 0,
        "raw_packets": False,
    }
    metadata: Dict[str, object] = {
        "started_at": _iso(first),
        "source": {
            "source_name": source,
            "device_name": f"imported {fmt}",
            "device_id": f"import-{fmt}",
            "capabilities": capabilities,
            "metadata": {"format": fmt},
        },
        "config": {"duration_seconds": last - first, "source_name": source},
        "import": import_block,
    }
    if synthetic:
        metadata = {"synthetic": True, "note": SYNTHETIC_NOTE, **metadata}
    _write_json(out_dir / "metadata.json", metadata)
    summary: Dict[str, object] = {
        "output_dir": str(out_dir.resolve()),
        "raw_path": "",
        "decoded_frames_path": str(frames_path.resolve()),
        "metadata_path": str((out_dir / "metadata.json").resolve()),
        "events_path": "",
        "summary_path": str((out_dir / "summary.json").resolve()),
        "started_at": _iso(first),
        "ended_at": _iso(last),
        "duration_seconds": last - first,
        "frame_count": frame_count,
        "raw_packet_count": 0,
        "decoded_frame_count": frame_count,
        "modality_counts": modality_counts,
        "reconnect_attempts": 0,
        "downtime_seconds": 0.0,
        "stop_reason": "imported",
        "import": {"format": fmt, "banner": banner},
    }
    if synthetic:
        summary = {"synthetic": True, "note": SYNTHETIC_NOTE, **summary}
    _write_json(out_dir / "summary.json", summary)
    return ImportResult(
        output_dir=str(out_dir),
        fmt=fmt,
        frame_count=frame_count,
        eeg_samples=eeg_samples,
        modality_counts=modality_counts,
        start_utc=_iso(first),
        duration_seconds=last - first,
        banner=banner,
        synthetic=synthetic,
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# What other commands ask of an imported directory
# --------------------------------------------------------------------------- #
def import_block(recording_input: Path) -> Optional[Mapping[str, object]]:
    """The ``import`` block of a recording directory's (or its file's) metadata, or None."""
    path = Path(recording_input).expanduser()
    directory = path if path.is_dir() else path.parent
    metadata = directory / "metadata.json"
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    block = payload.get("import") if isinstance(payload, dict) else None
    return block if isinstance(block, Mapping) else None


def import_banner(recording_input: Path) -> Optional[str]:
    """``imported: <format>, scale <x>; report and simulation only`` for an import."""
    block = import_block(recording_input)
    if block is None:
        return None
    scale = block.get("eeg_scale")
    if isinstance(scale, (int, float)) and not isinstance(scale, bool) and math.isfinite(scale):
        return scale_banner(str(block.get("format", "unknown")), float(scale))
    return f"imported: {block.get('format', 'unknown')}, scale undeclared; {IMPORT_NOTE}"


def require_declared_scale(recording_input: Path, command: str) -> None:
    """ValueError for an import without a declared EEG scale; nothing for other inputs."""
    block = import_block(recording_input)
    if block is None:
        return
    scale = block.get("eeg_scale")
    if isinstance(scale, (int, float)) and not isinstance(scale, bool) and math.isfinite(scale) and scale != 0:
        return
    raise ValueError(
        f"{command}: {Path(recording_input)} was imported from {block.get('format', 'an unknown format')} "
        "without a declared EEG scale; import it again with --eeg-scale"
    )


def scale_table() -> List[Dict[str, object]]:
    return [row.to_dict() for row in IMPORT_SCALE_TABLE]
