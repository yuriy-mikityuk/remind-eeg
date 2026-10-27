"""A night as an MNE ``RawArray`` (#230). ``mne`` is imported only when called.

``to_raw()`` takes a recording directory or a ``ContinuousRecording``: the EEG in volts
as ``eeg`` channels, and the accelerometer, gyroscope and optics as ``misc`` channels
interpolated linearly onto the EEG's 256 Hz grid, since one Raw has one rate. The
annotations are the ones an EDF export writes (``edf_io.place_annotations()``): the
night's events, its zero fills as ``BAD_ACQ_SKIP``, which MNE leaves out of epochs and
filters by default, and the notes of a synthetic or imported night. Under the
``discontinuous`` gap policy nothing is filled; each join gets ``BAD boundary`` and
``EDGE boundary`` as ``mne.concatenate_raws`` writes them, and the ``misc`` channels
follow the EEG through host time, since each stream lost its own samples at a join.
``meas_date`` is the recording's first frame, in UTC.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from muse_tmr.data.continuous import ContinuousConfig, ContinuousRecording, reconstruct_recording
from muse_tmr.data.edf_io import EXPORT_STREAMS, export_config, place_annotations
from muse_tmr.data.night_events import NightAnnotation, collect_night_annotations, ensure_exportable, load_metadata, notes_for

MNE_HINT = "to_raw needs MNE-Python: pip install mne"
UV_TO_V = 1e-6


def _mne():
    try:
        import mne
    except ImportError as exc:
        raise ImportError(MNE_HINT) from exc
    return mne


def to_raw(
    source: Union[Path, str, ContinuousRecording],
    *,
    annotations: Optional[Sequence[NightAnnotation]] = None,
    notes: Sequence[str] = (),
    config: Optional[ContinuousConfig] = None,
    streams: Sequence[str] = EXPORT_STREAMS,
    hypnograms: Sequence[Path] = (),
    night_schedule=None,
):
    """An ``mne.io.RawArray`` of a recording directory or reconstruction.

    For a directory the night's annotations and notes are collected as for
    ``export-edf``, and a night from a sealed schedule raises SealedNightError.
    """
    mne = _mne()
    if isinstance(source, ContinuousRecording):
        recording = source
        annotations = tuple(annotations or ())
    else:
        recording_dir = Path(source).expanduser()
        ensure_exportable(recording_dir, night_schedule)
        recording = asyncio.run(reconstruct_recording(recording_dir, config or export_config()))
        if annotations is None:
            annotations = collect_night_annotations(
                recording_dir,
                t0=recording.t0,
                hypnograms=hypnograms,
                to_unix=recording.unix_time,
                night_schedule=night_schedule,
            )
        notes = tuple(notes) or notes_for(load_metadata(recording_dir))

    eeg = recording.eeg
    rate = eeg.nominal_rate_hz
    offset = recording.offset_samples("eeg")
    total = offset + eeg.n_samples
    names, types, rows = [], [], []
    for name, values in eeg.channels.items():
        row = np.zeros(total)
        row[offset:] = np.asarray(values, dtype=float) * UV_TO_V
        names.append(name)
        types.append("eeg")
        rows.append(row)
    grid = np.arange(total) / rate
    discontinuous = recording.config.gap_policy == "discontinuous"
    for stream_name in streams:
        if stream_name == "eeg" or stream_name not in recording.streams:
            continue
        stream = recording.streams[stream_name]
        if not stream.channels:
            continue
        if discontinuous:
            # Where each EEG sample's host time falls on this stream's own samples.
            positions = stream.time_map.index(recording.unix_time(grid))
        else:
            # As the EDF places it: at its nominal rate from its own first sample.
            positions = grid * stream.nominal_rate_hz - recording.offset_samples(stream_name)
        indices = np.arange(stream.n_samples, dtype=float)
        for name, values in stream.channels.items():
            names.append(name)
            types.append("misc")
            rows.append(np.interp(positions, indices, np.asarray(values, dtype=float), left=0.0, right=0.0))
    info = mne.create_info(names, sfreq=rate, ch_types=types)
    raw = mne.io.RawArray(np.vstack(rows), info, verbose="error")
    raw.set_meas_date(dt.datetime.fromtimestamp(recording.t0, tz=dt.timezone.utc))

    placed = place_annotations(recording, annotations, notes=notes, duration=total / rate)
    onsets = [item.onset_s for item in placed]
    durations = [item.duration_s for item in placed]
    descriptions = [item.text for item in placed]
    if recording.config.gap_policy == "discontinuous":
        for segment in eeg.segments[1:]:
            join = (offset + segment.start_index) / rate
            for text in ("BAD boundary", "EDGE boundary"):
                onsets.append(join)
                durations.append(0.0)
                descriptions.append(text)
    if onsets:
        raw.set_annotations(
            mne.Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=raw.info["meas_date"])
        )
    return raw
