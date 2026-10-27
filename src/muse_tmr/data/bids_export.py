"""A night as an EEG-BIDS dataset (#230).

``export_bids()`` writes one session under ``<root>/sub-<id>/ses-<date>/eeg/``:

- ``_task-sleep_eeg.edf``, the EDF+C of ``edf_io.write_edf_plus()``;
- ``_channels.tsv`` with each signal's type (EEG, ACCEL, GYRO, MISC for optics), unit
  and rate;
- ``_eeg.json`` with the required fields, ``EEGReference`` "Fpz (vendor-stated,
  unverified)" and the 1450 uV full scale marked vendor-stated and unverified; for a
  night imported from an EDF, whose device the import does not record, the reference,
  manufacturer, model, cap and placement are "n/a" and there is no full scale;
- ``_events.tsv`` with the cue, gate, guard, audio and recording events and the zero
  fills, and ``_acq-scoring_events.tsv`` with the stages, the second events file sleep
  datasets that pass the validator use, since BIDS has no sleep scoring file yet;
- at the root ``dataset_description.json``, ``participants.tsv`` with the pseudonymous
  subject ID given, and a README.

HED tags are not written. A synthetic night's dataset says "synthetic input, not sleep"
in its name, README, ``_eeg.json`` and EDF, and a root holding one kind of night refuses
the other kind. A night from a sealed night schedule is refused before anything is
written (``night_events.blinding_refusal()``).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from muse_tmr import __version__
from muse_tmr.data.continuous import ContinuousConfig, ContinuousRecording
from muse_tmr.data.edf_io import (
    GAP_TEXT,
    EdfExportResult,
    load_export_inputs,
    place_annotations,
    write_edf_plus,
)
from muse_tmr.data.night_events import (
    SYNTHETIC_NOTE,
    NightAnnotation,
    ensure_exportable,
    notes_for,
    recorded_with_muse,
    stage_label,
)

BIDS_VERSION = "1.10.0"
EEG_REFERENCE = "Fpz (vendor-stated, unverified)"
FULL_SCALE_NOTE = (
    "14-bit counts of 1450/16383 uV, centred on count 8192 (-725.04 to 724.96 uV); "
    "the 1450 uV full scale is vendor-stated and unverified"
)
CODE_URL = "https://github.com/yuriy-mikityuk/remind-eeg"
_LABEL = re.compile(r"^[A-Za-z0-9]+$")
_CHANNEL_TYPES = {"eeg": "EEG", "accel": "ACCEL", "gyro": "GYRO", "optics": "MISC"}
_CHANNEL_UNITS = {"eeg": "uV", "accel": "g", "gyro": "deg/s", "optics": "n/a"}
_DESCRIPTIONS = {
    "eeg": "Muse S Athena dry electrode, reference near Fpz (vendor-stated, unverified)",
    "accel": "headband accelerometer",
    "gyro": "headband gyroscope",
    "optics": "headband optics, raw normalised counts",
}
# A night imported from an EDF: its values are what the import made of the file's, and
# nothing is known of the device.
_IMPORTED_EDF_DESCRIPTIONS = {
    "eeg": "EEG imported from an EDF; device and reference not recorded",
    "accel": "accelerometer imported from an EDF",
    "gyro": "gyroscope imported from an EDF",
    "optics": "pulse or optics channel imported from an EDF, in that file's unit",
}
_MUSE_DEVICE = {
    "EEGReference": EEG_REFERENCE,
    "Manufacturer": "InteraXon",
    "ManufacturersModelName": "Muse S Athena",
    "CapManufacturer": "InteraXon",
    "CapManufacturersModelName": "Muse S Athena headband",
    "EEGPlacementScheme": "10-20 positions TP9, AF7, AF8 and TP10 (vendor-stated)",
}
_UNKNOWN_DEVICE = {key: "n/a" for key in _MUSE_DEVICE}
_EVENT_TYPES = {
    "recording": "A recorder or pilot runner event from events.jsonl.",
    "scheduler_play": "A cue the REM-gated scheduler played (scheduler_events.jsonl).",
    "scheduler_skip": "A run of scheduler skips with the same reasons.",
    "scheduler_pause": "A scheduler pause.",
    "scheduler_stop": "A scheduler stop.",
    "rem_gate_open": "A gate-open run of the REM gate (rem_gate_decisions.jsonl).",
    "arousal_guard": "An arousal guard decision other than allow, runs merged.",
    "audio": "A cue the audio player played, blocked or stopped (audio_playback.jsonl).",
    "awakening": "An awakening marker (awakening_events.jsonl).",
    "note": "A note about the whole recording.",
    GAP_TEXT: "Samples that never arrived, zero-filled in the EDF.",
}


@dataclass(frozen=True)
class BidsExportResult:
    root: str
    subject: str
    session: str
    files: Tuple[str, ...]
    edf: EdfExportResult
    events: int
    stages: int
    synthetic: bool

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "root": self.root,
            "subject": self.subject,
            "session": self.session,
            "files": list(self.files),
            "edf": self.edf.to_dict(),
            "events": self.events,
            "stages": self.stages,
        }
        if self.synthetic:
            payload = {"synthetic": True, "note": SYNTHETIC_NOTE, **payload}
        return payload


def _check_label(value: str, name: str) -> str:
    value = str(value).strip()
    if not _LABEL.match(value):
        raise ValueError(f"the BIDS {name} label must be letters and digits only, got {value!r}")
    return value


def _tsv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(_cell(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cell(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value).replace("\t", " ").replace("\n", " ")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_bids(
    recording_dir: Path,
    bids_root: Path,
    *,
    subject: str = "01",
    session: Optional[str] = None,
    task: str = "sleep",
    hypnograms: Sequence[Path] = (),
    night_schedule: Optional[Mapping[str, object]] = None,
    power_line_frequency: Optional[float] = None,
    config: Optional[ContinuousConfig] = None,
) -> BidsExportResult:
    """Write ``recording_dir`` as one BIDS session (see the module docstring).

    ``session`` defaults to the night's start date in UTC (``YYYYMMDD``). A session that
    already holds files is refused, and so is a root whose dataset is synthetic when this
    night is not, or the other way round.
    """
    recording_dir = Path(recording_dir).expanduser()
    bids_root = Path(bids_root).expanduser()
    subject = _check_label(subject, "subject")
    task = _check_label(task, "task")
    ensure_exportable(recording_dir, night_schedule)
    recording, annotations, metadata = load_export_inputs(
        recording_dir, config=config, hypnograms=hypnograms, night_schedule=night_schedule
    )
    synthetic = metadata.get("synthetic") is True
    start = dt.datetime.fromtimestamp(recording.t0, tz=dt.timezone.utc)
    session = _check_label(session or start.strftime("%Y%m%d"), "session")
    _check_root(bids_root, synthetic)
    eeg_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "eeg"
    if eeg_dir.exists() and any(eeg_dir.iterdir()):
        raise ValueError(f"{eeg_dir} already holds files; give another --session or --subject")
    eeg_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{subject}_ses-{session}_task-{task}"
    notes = notes_for(metadata)
    muse = recorded_with_muse(metadata)

    edf_result = write_edf_plus(
        recording,
        eeg_dir / f"{stem}_eeg.edf",
        annotations=annotations,
        notes=notes,
        synthetic=synthetic,
        muse_headband=muse,
    )
    files: List[Path] = [eeg_dir / f"{stem}_eeg.edf"]
    files.append(_write_channels(eeg_dir / f"{stem}_channels.tsv", edf_result, muse))
    files.append(
        _write_sidecar(eeg_dir / f"{stem}_eeg.json", recording, edf_result, metadata, task, power_line_frequency, notes)
    )
    placed = place_annotations(recording, annotations, notes=notes, duration=edf_result.duration_seconds)
    events = [item for item in placed if item.kind != "stage"]
    stages = [item for item in placed if item.kind == "stage"]
    files += _write_events(eeg_dir / f"{stem}_events.tsv", events)
    files += _write_scoring(eeg_dir / f"{stem}_acq-scoring_events.tsv", stages)
    files += _write_root(bids_root, subject, synthetic, metadata)
    return BidsExportResult(
        root=str(bids_root),
        subject=subject,
        session=session,
        # BIDS paths, with forward slashes on every system.
        files=tuple(path.relative_to(bids_root).as_posix() for path in files),
        edf=edf_result,
        events=len(events),
        stages=len(stages),
        synthetic=synthetic,
    )


def _check_root(root: Path, synthetic: bool) -> None:
    description = root / "dataset_description.json"
    if not description.exists():
        return
    try:
        payload = json.loads(description.read_text(encoding="utf-8"))
    except ValueError:
        raise ValueError(f"{description} is not valid JSON") from None
    root_synthetic = SYNTHETIC_NOTE in str(payload.get("Name", ""))
    if root_synthetic != synthetic:
        kind = "synthetic" if root_synthetic else "recorded"
        raise ValueError(f"{root} holds a {kind} dataset; a synthetic night and a recorded one never share a root")


def _write_channels(path: Path, edf_result: EdfExportResult, muse: bool) -> Path:
    descriptions = _DESCRIPTIONS if muse else _IMPORTED_EDF_DESCRIPTIONS
    rows = []
    for signal in edf_result.signals:
        rows.append(
            (
                signal.label,
                _CHANNEL_TYPES[signal.stream],
                _CHANNEL_UNITS[signal.stream],
                signal.rate_hz,
                descriptions[signal.stream],
                "good",
            )
        )
    _tsv(path, ("name", "type", "units", "sampling_frequency", "description", "status"), rows)
    return path


def _write_sidecar(
    path: Path,
    recording: ContinuousRecording,
    edf_result: EdfExportResult,
    metadata: Mapping[str, object],
    task: str,
    power_line_frequency: Optional[float],
    notes: Sequence[str],
) -> Path:
    types = [_CHANNEL_TYPES[signal.stream] for signal in edf_result.signals]
    synthetic = metadata.get("synthetic") is True
    imported = metadata.get("import") if isinstance(metadata.get("import"), Mapping) else None
    description = "One night with a Muse S Athena headband, recorded at home."
    if synthetic:
        description = f"A generated night from muse_tmr.testing: {SYNTHETIC_NOTE}."
    elif imported:
        description = f"A night imported into REMind: {imported.get('banner')}."
    eeg = recording.eeg
    muse = recorded_with_muse(metadata)
    device = _MUSE_DEVICE if muse else _UNKNOWN_DEVICE
    payload: Dict[str, object] = {
        "TaskName": task,
        "TaskDescription": description,
        "EEGReference": device["EEGReference"],
        "SamplingFrequency": eeg.nominal_rate_hz,
        "PowerLineFrequency": power_line_frequency if power_line_frequency is not None else "n/a",
        "SoftwareFilters": "n/a",
        "HardwareFilters": "n/a",
        "Manufacturer": device["Manufacturer"],
        "ManufacturersModelName": device["ManufacturersModelName"],
        "CapManufacturer": device["CapManufacturer"],
        "CapManufacturersModelName": device["CapManufacturersModelName"],
        "SoftwareVersions": f"REMind {__version__}",
        "EEGChannelCount": types.count("EEG"),
        "ECGChannelCount": 0,
        "EMGChannelCount": 0,
        "EOGChannelCount": 0,
        "MISCChannelCount": types.count("MISC"),
        "TriggerChannelCount": 0,
        "RecordingDuration": edf_result.duration_seconds,
        "RecordingType": "continuous",
        "EEGPlacementScheme": device["EEGPlacementScheme"],
        "EEGGround": "n/a",
        # REMind's own keys; BIDS allows keys it does not define in a sidecar. The full
        # scale is the headband's, unknown for a night imported from an EDF.
        **({"EEGFullScale": FULL_SCALE_NOTE} if muse else {}),
        "MeasuredSamplingFrequency": eeg.measured_rate_hz if math.isfinite(eeg.measured_rate_hz) else None,
        "GapPolicy": (
            f"{recording.config.gap_policy}: samples that never arrived are zeros in the EDF; a frame more than "
            f"{recording.config.gap_threshold_seconds:g} s late starts a dropout, annotated {GAP_TEXT}"
        ),
    }
    if notes:
        payload["Notes"] = list(notes)
    _write_json(path, payload)
    return path


def _write_events(path: Path, events) -> List[Path]:
    rows = [(item.onset_s, item.duration_s, item.kind, item.text) for item in events]
    _tsv(path, ("onset", "duration", "trial_type", "description"), rows)
    sidecar = path.with_suffix(".json")
    kinds = sorted({item.kind for item in events})
    payload: Dict[str, object] = {
        "onset": {"Description": "Seconds from the start of the EDF, on the EEG sample clock.", "Units": "s"},
        "duration": {"Description": "Seconds; 0 for an event without a length.", "Units": "s"},
        "trial_type": {
            "Description": "What the event is.",
            "Levels": {kind: _EVENT_TYPES.get(kind, kind) for kind in kinds},
        },
        "description": {"Description": "The event as the EDF+ annotation of the EDF writes it."},
    }
    if "audio" in kinds:
        # REMind's player handled the cues; each audio event names its backend, so a
        # dry-run night's events say that nothing was heard.
        payload["StimulusPresentation"] = {"SoftwareName": "REMind", "SoftwareVersion": __version__, "Code": CODE_URL}
    _write_json(sidecar, payload)
    return [path, sidecar]


def _write_scoring(path: Path, stages) -> List[Path]:
    rows = [(item.onset_s, item.duration_s, item.text, stage_label(item.text)) for item in stages]
    _tsv(path, ("onset", "duration", "stage", "label"), rows)
    sidecar = path.with_suffix(".json")
    _write_json(
        sidecar,
        {
            "onset": {"Description": "Seconds from the start of the EDF, on the EEG sample clock.", "Units": "s"},
            "duration": {"Description": "Seconds the stage lasts.", "Units": "s"},
            "stage": {
                "Description": (
                    "EDF+ sleep stage text. Lossy: the labels map W to wake, N to nrem (covering N1, N2 "
                    "and N3), R to probable_rem and ? to unknown; no stage N1, N2 or N3 is written."
                ),
                "Levels": {
                    "Sleep stage W": "wake",
                    "Sleep stage N": "NREM, stage not given",
                    "Sleep stage R": "probable REM",
                    "Sleep stage ?": "unknown",
                },
            },
            "label": {"Description": "The REM annotation label of muse_tmr.annotations."},
        },
    )
    return [path, sidecar]


def _write_root(root: Path, subject: str, synthetic: bool, metadata: Mapping[str, object]) -> List[Path]:
    written: List[Path] = []
    description = root / "dataset_description.json"
    if not description.exists():
        name = "REMind nights"
        if synthetic:
            name = f"REMind synthetic nights ({SYNTHETIC_NOTE})"
        _write_json(
            description,
            {
                "Name": name,
                "BIDSVersion": BIDS_VERSION,
                "DatasetType": "raw",
                "GeneratedBy": [
                    {
                        "Name": "REMind",
                        "Version": __version__,
                        "Description": "muse-tmr export-bids",
                        "CodeURL": CODE_URL,
                    }
                ],
            },
        )
        written.append(description)
    readme = root / "README.md"
    if not readme.exists():
        lines = [
            "# REMind nights",
            "",
            "Nights exported by `muse-tmr export-bids`. Each session has the EDF+, its channels,",
            "the night's cue, gate and guard events and, where there is one, a hypnogram in the",
            "`acq-scoring` events file.",
            "",
            "A session recorded with a Muse S Athena headband says so in its `_eeg.json`, with the",
            "1450 uV full scale of the EEG and its reference near Fpz, both vendor-stated and",
            "unverified. A night imported from an EDF has n/a there: the import does not record",
            "its device. Subject IDs are pseudonyms chosen at export.",
        ]
        if synthetic:
            lines[0] = "# REMind synthetic nights"
            lines.insert(2, "Synthetic input, not sleep: these nights were generated by `muse_tmr.testing`.")
            lines.insert(3, "")
        readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(readme)
    participants = root / "participants.tsv"
    existing = set()
    if participants.exists():
        for line in participants.read_text(encoding="utf-8").splitlines()[1:]:
            if line.strip():
                existing.add(line.split("\t")[0])
    participant = f"sub-{subject}"
    if participant not in existing:
        if not participants.exists():
            participants.write_text("participant_id\n", encoding="utf-8")
        with participants.open("a", encoding="utf-8") as handle:
            handle.write(participant + "\n")
        written.append(participants)
    sidecar = root / "participants.json"
    if not sidecar.exists():
        _write_json(sidecar, {"participant_id": {"Description": "A pseudonymous ID chosen at export."}})
        written.append(sidecar)
    return written
