"""The BOAS dataset (OpenNeuro ds005555) read as REMind nights (#226).

BOAS, the Bitbrain Open Access Sleep dataset, holds 128 nights of 100 participants,
recorded at once with a Bitbrain Ikon headband and a Micromed PSG. ``participants.tsv``
lists each night (``sub-<n>``) with ``pid``, the participant, so that a person's nights
can be kept together. Version 1.1.3 is doi:10.18112/openneuro.ds005555.v1.1.3, CC0.
Per night, under ``sub-<n>/eeg/``:

- ``sub-<n>_task-Sleep_acq-headband_eeg.edf``: ``HB_1`` and ``HB_2``, forehead EEG near
  AF7 and AF8 in uV against the left mastoid, and on most nights ``HB_IMU_1`` to ``6``
  and ``HB_PULSE``, all at 256 Hz (``edf_io.BOAS_HEADBAND_CHANNEL_MAP``).
- ``sub-<n>_task-Sleep_acq-psg_events.tsv``: one row per 30 s epoch with ``stage_hum``,
  the consensus of three AASM scorers on the PSG (a fourth decided when all three
  differed), and ``stage_ai``, Bitbrain's network on the PSG.
- ``sub-<n>_task-Sleep_acq-headband_events.tsv``: the same epochs with the ``stage_ai`` of
  that network on the headband.

Both events files of a night list the same onsets, from 0 s in steps of 30 s, and the
headband and PSG sidecars give the same ``RecordingDuration``: the dataset puts both on
one clock. ``load_boas_hypnogram()`` refuses a night whose two files differ or whose
epochs do not follow one another. Stages are 0 W, 1 to 3 N1 to N3, 4 REM, 8 a PSG
disconnection (human-scored only) and -2 artefact or missing data (AI-scored only); the
REM annotation label of each comes from ``night_events.stage_label()``.

``BoasHeadbandSession`` replays the headband EDF through ``edf_io.read_edf()`` as
MuseFrames with ``source="boas"``: EEG as AF7 and AF8 (there is no TP9 or TP10), the IMU
resampled to 52 Hz and ``HB_PULSE`` to 64 Hz as optics. On the nights checked for #226 the
IMU is not in the units its header names (an accelerometer magnitude of 10 to 70 "g" over
a night, gyroscope offsets of tens to hundreds of deg/s), so ``DETECTOR_MODALITIES``, what
the REM check gives the detector, is EEG alone (``docs/validation_external.md``).

Nothing here imports muse_tmr.audio, muse_tmr.protocol or muse_tmr.models.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from muse_tmr.data.edf_io import BOAS_HEADBAND_CHANNEL_MAP, read_edf
from muse_tmr.data.import_recording import PIPELINE_RATES
from muse_tmr.data.night_events import SYNTHETIC_NOTE, stage_label
from muse_tmr.data.sample_types import MuseFrame
from muse_tmr.sources.base_source import BaseMuseSource, MuseDeviceInfo, MuseSourceMetadata

BOAS_DATASET_ID = "ds005555"
BOAS_DATASET_NAME = "The Bitbrain Open Access Sleep (BOAS) dataset"
BOAS_DATASET_VERSION = "1.1.3"
BOAS_DOI = "10.18112/openneuro.ds005555.v1.1.3"
BOAS_URL = "https://github.com/OpenNeuroDatasets/ds005555"
BOAS_SOURCE = "boas"
BOAS_EPOCH_SECONDS = 30.0
BOAS_MODALITIES = ("eeg", "imu", "ppg")
# What the REM check feeds the detector: the headband IMU is not in the units it names.
DETECTOR_MODALITIES = ("eeg",)
BOAS_STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R", 8: "disconnection", -2: "artefact"}
HEADBAND_EEG_LABELS = ("HB_1", "HB_2")
HEADBAND_IMU_LABELS = tuple(f"HB_IMU_{index}" for index in range(1, 7))
HEADBAND_PULSE_LABEL = "HB_PULSE"

_SUBJECT = re.compile(r"^sub-(\d+)$")
_VERSION_IN_DOI = re.compile(r"\.v(\d+(?:\.\d+)*)$")
_MISSING = ("", "n/a", "na", "nan")
_TOLERANCE_SECONDS = 1e-6


@dataclass(frozen=True)
class BoasNight:
    """One night of the dataset: its recording id, participant and files."""

    recording_id: str
    pid: Optional[str]
    night_dir: Path

    @property
    def root(self) -> Path:
        return self.night_dir.parent

    @property
    def number(self) -> int:
        return _subject_number(self.recording_id)

    def _file(self, suffix: str) -> Path:
        return self.night_dir / "eeg" / f"{self.recording_id}_task-Sleep_{suffix}"

    @property
    def headband_edf(self) -> Path:
        return self._file("acq-headband_eeg.edf")

    @property
    def headband_events(self) -> Path:
        return self._file("acq-headband_events.tsv")

    @property
    def headband_channels(self) -> Path:
        return self._file("acq-headband_channels.tsv")

    @property
    def psg_events(self) -> Path:
        return self._file("acq-psg_events.tsv")

    def missing_files(self) -> Tuple[str, ...]:
        """The files REMind reads for this night that are not on disk."""
        paths = (self.headband_edf, self.headband_channels, self.headband_events, self.psg_events)
        return tuple(path.name for path in paths if not path.is_file())

    def headband_channel_names(self) -> Tuple[str, ...]:
        """The channels the headband sidecar lists; empty when it is missing."""
        if not self.headband_channels.is_file():
            return ()
        return tuple(str(row.get("name", "")).strip() for row in _read_tsv(self.headband_channels))

    @property
    def has_imu(self) -> bool:
        return set(HEADBAND_IMU_LABELS) <= set(self.headband_channel_names())

    @property
    def has_pulse(self) -> bool:
        return HEADBAND_PULSE_LABEL in self.headband_channel_names()


@dataclass(frozen=True)
class BoasEpochLabel:
    """One scored 30 s epoch: onset from the EDF start, the consensus stage and its label.

    ``reference_stage_ai`` is the ``stage_ai`` of the headband events file, Bitbrain's
    network on the same headband EEG REMind reads; ``psg_stage_ai`` that network on the
    PSG. A missing value is None.
    """

    epoch_index: int
    onset_s: float
    duration_s: float
    stage_hum: Optional[int]
    label: str
    reference_stage_ai: Optional[int]
    psg_stage_ai: Optional[int]

    @property
    def end_s(self) -> float:
        return self.onset_s + self.duration_s

    @property
    def stage_name(self) -> str:
        return stage_name(self.stage_hum)

    def to_dict(self) -> Dict[str, object]:
        return {
            "epoch": self.epoch_index,
            "onset_s": self.onset_s,
            "duration_s": self.duration_s,
            "stage_hum": self.stage_hum,
            "label": self.label,
            "reference_stage_ai": self.reference_stage_ai,
            "psg_stage_ai": self.psg_stage_ai,
        }


def stage_name(code: Optional[int]) -> str:
    """W, N1, N2, N3 or R for a BOAS stage code; ``disconnection``, ``artefact`` or ``n/a``
    for the rest."""
    if code is None:
        return "n/a"
    return BOAS_STAGE_NAMES.get(int(code), str(code))


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
def dataset_description(root: Path) -> Dict[str, object]:
    """``dataset_description.json`` of the dataset at ``root``; empty when it is missing."""
    path = Path(root).expanduser() / "dataset_description.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def dataset_version(root: Path) -> Optional[str]:
    """The version in the dataset's DOI (``1.1.3`` for doi:10.18112/openneuro.ds005555.v1.1.3)."""
    doi = str(dataset_description(root).get("DatasetDOI") or "")
    match = _VERSION_IN_DOI.search(doi.strip())
    return match.group(1) if match else None


def dataset_doi(root: Path) -> Optional[str]:
    doi = str(dataset_description(root).get("DatasetDOI") or "").strip()
    return doi[len("doi:"):] if doi.lower().startswith("doi:") else (doi or None)


def is_synthetic_dataset(root: Path) -> bool:
    """Whether the dataset at ``root`` says it is synthetic, as the test layouts do."""
    description = dataset_description(root)
    return description.get("synthetic") is True or description.get("note") == SYNTHETIC_NOTE


def read_participants(root: Path) -> Dict[str, Optional[str]]:
    """``participant_id`` to ``pid`` from ``participants.tsv``, in its order."""
    path = Path(root).expanduser() / "participants.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"participants.tsv not found under {root}")
    participants: Dict[str, Optional[str]] = {}
    for row in _read_tsv(path):
        recording_id = str(row.get("participant_id", "")).strip()
        if not recording_id:
            continue
        if not _SUBJECT.match(recording_id):
            raise ValueError(f"participants.tsv: {recording_id!r} is not a sub-<n> recording id")
        if recording_id in participants:
            raise ValueError(f"participants.tsv lists {recording_id} twice")
        participants[recording_id] = _text_or_none(row.get("pid"))
    return participants


def iter_boas_nights(root: Path) -> Iterator[BoasNight]:
    """Every night of the dataset at ``root`` in recording order (sub-1, sub-2, ...).

    The nights are the rows of ``participants.tsv``, each with its ``pid``, whether its
    files are on disk yet or not (``BoasNight.missing_files()``). A ``sub-<n>`` directory
    that ``participants.tsv`` does not list is a ValueError, since its participant, and
    so its fold, would be unknown.
    """
    root = Path(root).expanduser()
    participants = read_participants(root)
    on_disk = sorted(
        (path.name for path in root.iterdir() if path.is_dir() and _SUBJECT.match(path.name)),
        key=_subject_number,
    )
    unlisted = [name for name in on_disk if name not in participants]
    if unlisted:
        raise ValueError(f"participants.tsv does not list {', '.join(unlisted)}")
    for recording_id in sorted(participants, key=_subject_number):
        yield BoasNight(recording_id=recording_id, pid=participants[recording_id], night_dir=root / recording_id)


def boas_night(path: Path) -> BoasNight:
    """The night of a ``sub-<n>`` directory, its ``eeg`` directory or its headband EDF,
    with the ``pid`` that the ``participants.tsv`` beside it gives (None without one)."""
    night_dir = _night_dir(Path(path))
    if night_dir is None:
        raise ValueError(f"{path} is not a BOAS night (sub-<n>/eeg/sub-<n>_task-Sleep_acq-headband_eeg.edf)")
    pid = None
    if (night_dir.parent / "participants.tsv").is_file():
        pid = read_participants(night_dir.parent).get(night_dir.name)
    return BoasNight(recording_id=night_dir.name, pid=pid, night_dir=night_dir)


def is_boas_night(path: Path) -> bool:
    """Whether ``path`` is a BOAS night whose headband EDF is on disk."""
    night_dir = _night_dir(Path(path))
    if night_dir is None:
        return False
    return BoasNight(night_dir.name, None, night_dir).headband_edf.is_file()


def _night_dir(path: Path) -> Optional[Path]:
    path = path.expanduser()
    if path.is_file() and path.name.endswith("_task-Sleep_acq-headband_eeg.edf"):
        path = path.parent
    if path.name == "eeg" and _SUBJECT.match(path.parent.name):
        path = path.parent
    if _SUBJECT.match(path.name) and (path / "eeg").is_dir():
        return path.resolve()
    return None


# --------------------------------------------------------------------------- #
# The scored epochs
# --------------------------------------------------------------------------- #
def load_boas_hypnogram(night: Union[BoasNight, Path]) -> Tuple[BoasEpochLabel, ...]:
    """The consensus hypnogram of a night, one ``BoasEpochLabel`` per 30 s epoch.

    ``stage_hum`` comes from the PSG events file, the only one that has it, and maps to
    the REM annotation labels: 0 ``wake``, 1 to 3 ``nrem``, 4 ``probable_rem`` and 8, -2
    or a missing value ``unknown``. ValueError when the headband events file lists other
    onsets or another number of epochs, or when the epochs do not start at 0 s and follow
    one another without a gap, 30 s each.
    """
    night = night if isinstance(night, BoasNight) else boas_night(Path(night))
    psg = _read_tsv(night.psg_events)
    if psg and "stage_hum" not in psg[0]:
        raise ValueError(f"{night.psg_events.name} has no stage_hum column")
    headband = _read_tsv(night.headband_events) if night.headband_events.is_file() else None
    if headband is not None and len(headband) != len(psg):
        raise ValueError(
            f"{night.recording_id}: the headband events list {len(headband)} epochs and the PSG events {len(psg)}"
        )
    labels: List[BoasEpochLabel] = []
    expected_onset = 0.0
    for index, row in enumerate(psg):
        onset = _float(row.get("onset"), night.psg_events, index)
        duration = _float(row.get("duration"), night.psg_events, index)
        if abs(onset - expected_onset) > _TOLERANCE_SECONDS:
            raise ValueError(
                f"{night.recording_id}: PSG epoch {index} starts at {onset} s, not {expected_onset} s; "
                "its epochs must start at 0 s and follow one another"
            )
        # The README says every epoch is 30 s, and the detector's epoch k is paired with
        # the one at 30 k s.
        if abs(duration - BOAS_EPOCH_SECONDS) > _TOLERANCE_SECONDS:
            raise ValueError(f"{night.recording_id}: PSG epoch {index} lasts {duration} s, not {BOAS_EPOCH_SECONDS:g} s")
        reference_ai = None
        if headband is not None:
            other = headband[index]
            other_onset = _float(other.get("onset"), night.headband_events, index)
            other_duration = _float(other.get("duration"), night.headband_events, index)
            if abs(other_onset - onset) > _TOLERANCE_SECONDS or abs(other_duration - duration) > _TOLERANCE_SECONDS:
                raise ValueError(
                    f"{night.recording_id}: epoch {index} is at {onset} s for {duration} s in the PSG events "
                    f"and at {other_onset} s for {other_duration} s in the headband events"
                )
            reference_ai = _stage_code(other.get("stage_ai"))
        stage_hum = _stage_code(row.get("stage_hum"))
        labels.append(
            BoasEpochLabel(
                epoch_index=index,
                onset_s=onset,
                duration_s=duration,
                stage_hum=stage_hum,
                label="unknown" if stage_hum is None else stage_label(stage_hum),
                reference_stage_ai=reference_ai,
                psg_stage_ai=_stage_code(row.get("stage_ai")),
            )
        )
        expected_onset = onset + duration
    return tuple(labels)


def stage_counts(labels: Sequence[BoasEpochLabel]) -> Dict[str, int]:
    """Epochs per consensus stage name."""
    counts: Dict[str, int] = {}
    for row in labels:
        counts[row.stage_name] = counts.get(row.stage_name, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# The headband as a source
# --------------------------------------------------------------------------- #
def boas_channel_map(modalities: Sequence[str] = BOAS_MODALITIES) -> Dict[str, str]:
    """``edf_io.BOAS_HEADBAND_CHANNEL_MAP`` for ``modalities`` (``eeg``, ``imu``, ``ppg``)."""
    unknown = sorted(set(modalities) - set(BOAS_MODALITIES))
    if unknown:
        raise ValueError(f"unknown modalities {unknown}; choose from {', '.join(BOAS_MODALITIES)}")
    if "eeg" not in modalities:
        raise ValueError("a BOAS night is read with its EEG")
    kinds = {"eeg": ("eeg",), "imu": ("accel", "gyro"), "ppg": ("ppg",)}
    wanted = {kind for modality in modalities for kind in kinds[modality]}
    return {label: target for label, target in BOAS_HEADBAND_CHANNEL_MAP.items() if target.split(":")[0] in wanted}


class BoasHeadbandSession(BaseMuseSource):
    """A BOAS headband EDF replayed as MuseFrames with ``source="boas"``.

    ``modalities`` picks what the frames carry: ``eeg`` (AF7 and AF8, always), ``imu``
    (accelerometer and gyroscope at 52 Hz) and ``ppg`` (``HB_PULSE`` at 64 Hz). A night
    without IMU or pulse channels gives frames without them. Frames are ``frame_seconds``
    long and start at the EDF start, read as UTC; the IMU and pulse are resampled for the
    whole file at once (``edf_io.read_edf``), so near a frame they read a little of what
    follows it, as any filter over a recorded file does. Reading needs the ``[edf]``
    extra.
    """

    def __init__(
        self,
        night: Union[BoasNight, Path],
        *,
        modalities: Sequence[str] = BOAS_MODALITIES,
        frame_seconds: float = 0.25,
    ) -> None:
        self.night = night if isinstance(night, BoasNight) else boas_night(Path(night))
        self.modalities = tuple(modalities)
        self.channel_map = boas_channel_map(self.modalities)
        self.frame_seconds = float(frame_seconds)
        self.metadata: Optional[MuseSourceMetadata] = None
        self._stop_requested = False

    async def discover(self) -> Sequence[MuseDeviceInfo]:
        return [
            MuseDeviceInfo(
                name=f"BOAS {self.night.recording_id}",
                address=str(self.night.headband_edf),
                rssi=0,
                metadata={"recording_id": self.night.recording_id, "dataset": BOAS_DATASET_ID},
            )
        ]

    async def connect(self, device: Optional[MuseDeviceInfo] = None) -> MuseSourceMetadata:
        self._stop_requested = False
        if not self.night.headband_edf.is_file():
            raise FileNotFoundError(f"BOAS headband EDF not found: {self.night.headband_edf}")
        self.metadata = MuseSourceMetadata(
            source_name=BOAS_SOURCE,
            device_name="Bitbrain Ikon headband (BOAS)",
            device_id=self.night.recording_id,
            capabilities={
                "eeg": True,
                "imu": "imu" in self.modalities and self.night.has_imu,
                "ppg": "ppg" in self.modalities and self.night.has_pulse,
                "heart_rate": False,
                "battery": False,
                "raw_packets": False,
            },
            metadata={
                "recording_id": self.night.recording_id,
                "pid": self.night.pid or "",
                "edf_path": str(self.night.headband_edf),
                "modalities": ",".join(self.modalities),
            },
        )
        return self.metadata

    async def stream(self) -> AsyncIterator[MuseFrame]:
        if self.metadata is None:
            await self.connect()
        frames = read_edf(
            self.night.headband_edf,
            channel_map=self.channel_map,
            frame_seconds=self.frame_seconds,
            rates=PIPELINE_RATES,
            source=BOAS_SOURCE,
        )
        for frame in frames:
            if self._stop_requested:
                break
            yield frame

    async def stop(self) -> None:
        self._stop_requested = True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _subject_number(recording_id: str) -> int:
    match = _SUBJECT.match(str(recording_id))
    return int(match.group(1)) if match else 0


def _text_or_none(value: object) -> Optional[str]:
    text = "" if value is None else str(value).strip()
    return None if text.lower() in _MISSING else text


def _stage_code(value: object) -> Optional[int]:
    text = _text_or_none(value)
    if text is None:
        return None
    number = float(text)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"not a stage code: {value!r}")
    return int(number)


def _float(value: object, path: Path, index: int) -> float:
    text = _text_or_none(value)
    if text is None:
        raise ValueError(f"{Path(path).name}: row {index} has no onset or duration")
    return float(text)


def subject_number(recording_id: str) -> int:
    """The number of a ``sub-<n>`` recording id, for sorting."""
    return _subject_number(recording_id)
