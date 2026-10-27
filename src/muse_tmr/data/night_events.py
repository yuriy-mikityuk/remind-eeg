"""A night's logs as timed annotations, for EDF+, BIDS and MNE exports (#230).

``collect_night_annotations()`` reads what a recording directory holds and returns one
``NightAnnotation`` per event: host time, duration, text, kind and the file it came from.

- ``events.jsonl``: recorder events (``recording_started``, reconnects, dropouts,
  ``recording_stopped``) and the Pilot 4/5 events, ``emergency_stop_triggered`` among
  them. Only the event's name and a few safe details (reason, attempt, modality) are
  kept: no error text, path or device name.
- ``scheduler_events.jsonl``: ``play``, ``skip``, ``pause`` and ``stop`` with their
  reason codes. A run of skips with the same reasons is one annotation, from the first
  to the next scheduler event after the last.
- ``rem_gate_decisions.jsonl``: one ``rem_gate_open`` annotation per gate-open run, read
  the way ``muse_tmr.validation.gate_log.gate_open_runs()`` reads it (importing that
  package would pull in the audio, protocol and model packages). A night from before
  that log gets the runs from the scheduler's ``rem_gate_open`` and
  ``rem_gate_closed`` reason codes, which find an opening only at the next play.
- ``arousal_guard_events.jsonl``: every decision but ``allow``, runs merged like skips.
- ``audio_playback.jsonl``: each played, blocked or stopped cue, without the device name.
- ``awakening_events.jsonl``: each marker with ``attributed_to_cue``, without its notes.
- ``hypnogram.csv`` of a synthetic night and any hypnogram or annotation file given.

Stages go through the four REM annotation labels and out as standard EDF+ texts, a
lossy map that is recorded in ``docs/formats.md``: ``wake`` is ``Sleep stage W``,
``nrem`` is ``Sleep stage N``, ``probable_rem`` is ``Sleep stage R`` and ``unknown`` is
``Sleep stage ?``. No export writes N1, N2 or N3, even for a hypnogram that has them.

The scheduler, gate, guard and awakening logs count ``timestamp_seconds`` from the
night's first epoch start, ``first_epoch_start`` in the Pilot 4/5 summary. A runner logs a
scheduler or guard row at the start of its epoch (a TLR cue at its planned time) and acts
on it when the epoch is complete, so the row goes in at that epoch's decision, where the
cue played and where the gate-open runs count from: the decision of the first epoch of
the gate log starting at or after it, as ``gate_log.emitting_record()`` finds it, or
without a gate log ``DEFAULT_EPOCH_SECONDS`` after it.

A night whose arm came from a night schedule that is still sealed cannot be exported:
its logs name the arm (``blinding_refusal()``).
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
import json
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# The note muse_tmr.testing writes into everything a synthetic night holds. It is kept
# here so that muse_tmr.data does not import muse_tmr.testing.
SYNTHETIC_NOTE = "synthetic input, not sleep"
# muse_tmr.annotations.REM_LABELS; that package imports muse_tmr.models.
REM_LABELS = ("wake", "nrem", "probable_rem", "unknown")
LABEL_TO_EDF_STAGE = {
    "wake": "Sleep stage W",
    "nrem": "Sleep stage N",
    "probable_rem": "Sleep stage R",
    "unknown": "Sleep stage ?",
}
EDF_STAGE_TEXTS = tuple(LABEL_TO_EDF_STAGE.values())
STAGE_EPOCH_SECONDS = 30.0
PILOT_SUMMARY_FILENAMES = ("pilot5_summary.json", "pilot4_summary.json")
GATE_LOG_FILENAME = "rem_gate_decisions.jsonl"
# EpochConfig's epoch length, which the protocol fidelity report also falls back on: a
# night without a gate log has its scheduler and guard rows decided this long after
# the epoch start they carry.
DEFAULT_EPOCH_SECONDS = 30.0
# Epoch starts are sums of floats; this much apart they are the same moment (gate_log).
_TIME_TOLERANCE_SECONDS = 1e-6
# Merged runs of these kinds: one annotation per run of equal texts.
_MERGED_KINDS = ("scheduler_skip", "arousal_guard")
# Recorder and pilot event details that may go into a text; nothing free-form.
_SAFE_DETAILS = ("reason", "attempt", "modality", "backoff_seconds", "companion", "returncode", "pilot_id")

_STAGE_ALIASES = {
    "w": "wake",
    "wake": "wake",
    "awake": "wake",
    "sleep stage w": "wake",
    "n": "nrem",
    "n1": "nrem",
    "n2": "nrem",
    "n3": "nrem",
    "n4": "nrem",
    "nrem": "nrem",
    "s1": "nrem",
    "s2": "nrem",
    "s3": "nrem",
    "s4": "nrem",
    "sleep stage n": "nrem",
    "sleep stage n1": "nrem",
    "sleep stage n2": "nrem",
    "sleep stage n3": "nrem",
    "sleep stage 1": "nrem",
    "sleep stage 2": "nrem",
    "sleep stage 3": "nrem",
    "sleep stage 4": "nrem",
    "r": "probable_rem",
    "rem": "probable_rem",
    "sleep stage r": "probable_rem",
    "probable_rem": "probable_rem",
    "unknown": "unknown",
    "?": "unknown",
    "sleep stage ?": "unknown",
}
# Integer stages as YASA and the BOAS dataset (OpenNeuro ds005555) write them.
_INTEGER_STAGES = {0: "wake", 1: "nrem", 2: "nrem", 3: "nrem", 4: "probable_rem"}
# Columns that can hold a stage, in the order they are tried.
STAGE_COLUMNS = ("stage", "stage_hum", "sleep_stage", "label", "stage_ai", "trial_type", "value", "description")


class SealedNightError(ValueError):
    """The night's arm came from a night schedule that has not been unsealed."""


@dataclass(frozen=True)
class NightAnnotation:
    """One timed event of a night.

    ``time_unix`` is host time; ``onset_s`` is seconds from the ``t0`` the collector was
    given. An export puts it on its own time base from ``time_unix``.
    """

    time_unix: float
    duration_s: float
    text: str
    kind: str
    source: str
    onset_s: float = 0.0

    def as_tuple(self) -> Tuple[float, float, str]:
        return (self.onset_s, self.duration_s, self.text)

    def to_dict(self) -> Dict[str, object]:
        return {
            "onset_s": self.onset_s,
            "duration_s": self.duration_s,
            "text": self.text,
            "kind": self.kind,
            "source": self.source,
            "time_unix": self.time_unix,
        }


@dataclass(frozen=True)
class StageInterval:
    """A scored stretch of a hypnogram: ``onset_s`` from the recording start (or, with
    ``absolute``, host time), its length, the stage as written and its REM label."""

    onset_s: float
    duration_s: float
    stage: str
    label: str
    absolute: bool = False

    @property
    def end_s(self) -> float:
        return self.onset_s + self.duration_s


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_label(stage: object) -> str:
    """The REM annotation label of a stage as a stager or a hypnogram writes it.

    W and Wake are ``wake``; N, N1 to N3, S1 to S4 and the EDF+ texts ``Sleep stage N``,
    ``N1`` to ``N3`` and ``1`` to ``4`` are ``nrem``; R, REM and ``Sleep stage R`` are
    ``probable_rem``. A bare integer is read as YASA and BOAS write stages: 0 W, 1 to 3
    NREM, 4 REM. Anything else, artefact and movement codes among it, is ``unknown``.
    """
    if stage is None:
        return "unknown"
    if isinstance(stage, float) and math.isnan(stage):
        return "unknown"
    text = str(stage).strip()
    try:
        number = float(text)
    except ValueError:
        return _STAGE_ALIASES.get(text.lower(), "unknown")
    if number.is_integer():
        return _INTEGER_STAGES.get(int(number), "unknown")
    return "unknown"


def edf_stage_text(label: str) -> str:
    """The EDF+ standard text of a REM annotation label (the recorded lossy map)."""
    if label not in LABEL_TO_EDF_STAGE:
        raise ValueError(f"label must be one of: {', '.join(REM_LABELS)}")
    return LABEL_TO_EDF_STAGE[label]


def label_at(intervals: Sequence[StageInterval], seconds: float) -> Optional[str]:
    """The label of the interval holding ``seconds``; None outside every interval.
    ``intervals`` must be sorted by onset and not overlap."""
    starts = [interval.onset_s for interval in intervals]
    position = bisect.bisect_right(starts, seconds) - 1
    if position >= 0 and seconds < intervals[position].end_s:
        return intervals[position].label
    return None


def read_hypnogram(path: Path, *, column: Optional[str] = None) -> Tuple[StageInterval, ...]:
    """The stage intervals of a hypnogram file, sorted by onset.

    - ``.edf``: EDF+ annotations whose text is a stage (``muse_tmr.data.edf_io``).
    - ``.tsv``: a BIDS events file with ``onset``, ``duration`` and a stage column (the
      first of ``STAGE_COLUMNS`` present, or ``column``), as the ``acq-scoring`` file of
      ``export-bids`` and the BOAS events files are.
    - ``.csv``: ``start_s``/``end_s`` (a synthetic night's ``hypnogram.csv``),
      ``onset``/``duration``, or ``epoch`` with 30 s epochs, and a stage column; or the
      rows of ``annotate-template`` (``start_time``, ``end_time``, ``label``), which are
      host times and come back ``absolute``.
    - ``.json``: the rows of ``annotate-template``.

    Onsets are seconds from the recording start, the EDF start for an export.
    """
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix in (".edf", ".bdf"):
        from muse_tmr.data.edf_io import read_edf_hypnogram

        intervals = read_edf_hypnogram(path)
    elif suffix == ".tsv":
        intervals = _table_intervals(_read_table(path, "\t"), column, path)
    elif suffix == ".csv":
        intervals = _table_intervals(_read_table(path, ","), column, path)
    elif suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path} holds no annotation rows")
        intervals = _table_intervals([{str(k): v for k, v in row.items()} for row in rows], column, path)
    else:
        raise ValueError(f"hypnogram must be .edf, .tsv, .csv or .json: {path}")
    return tuple(sorted(intervals, key=lambda interval: interval.onset_s))


def _read_table(path: Path, delimiter: str) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def _table_intervals(rows: Sequence[Mapping[str, object]], column: Optional[str], path: Path) -> List[StageInterval]:
    if not rows:
        return []
    names = set(rows[0])
    stage_column = column or next((name for name in STAGE_COLUMNS if name in names), None)
    if stage_column is None or stage_column not in names:
        raise ValueError(f"{path} has no stage column; give one of {', '.join(names)}")
    intervals = []
    for row in rows:
        stage = row.get(stage_column)
        if stage in (None, "") or str(stage).strip().lower() == "n/a":
            continue
        absolute = False
        if "start_time" in row and "end_time" in row and "onset" not in row:
            onset = _float(row["start_time"])
            duration = _float(row["end_time"]) - onset
            absolute = True
        elif "start_s" in row:
            onset = _float(row["start_s"])
            duration = _float(row["end_s"]) - onset if row.get("end_s") not in (None, "") else STAGE_EPOCH_SECONDS
        elif "onset" in row:
            onset = _float(row["onset"])
            duration = _float(row.get("duration")) if row.get("duration") not in (None, "", "n/a") else STAGE_EPOCH_SECONDS
        elif "epoch" in row:
            onset = _float(row["epoch"]) * STAGE_EPOCH_SECONDS
            duration = STAGE_EPOCH_SECONDS
        else:
            raise ValueError(f"{path} has no onset, start_s, start_time or epoch column")
        label = str(stage).strip() if stage_column == "label" and str(stage).strip() in REM_LABELS else stage_label(stage)
        intervals.append(StageInterval(onset, duration, str(stage).strip(), label, absolute))
    return intervals


def _float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"not a number: {value!r}") from None


# --------------------------------------------------------------------------- #
# Blinding
# --------------------------------------------------------------------------- #
def load_metadata(recording_dir: Path) -> Dict[str, object]:
    path = Path(recording_dir) / "metadata.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def sealed_night_arm(metadata: Mapping[str, object]) -> Optional[Mapping[str, object]]:
    """The ``night_arm`` block of a night whose arm came from a night schedule that was
    still sealed when it ran; None for any other night."""
    arm = metadata.get("night_arm")
    if not isinstance(arm, Mapping) or not arm.get("sealed_schedule"):
        return None
    if arm.get("schedule_unsealed_at_utc"):
        return None
    return arm


def blinding_refusal(metadata: Mapping[str, object], schedule: Optional[Mapping[str, object]] = None) -> Optional[str]:
    """Why a night cannot be exported yet, or None.

    A night from a sealed schedule names its arm in its logs (``sham_arm``,
    ``no_cue_arm``, ``virtual_play``), so its export waits for ``unseal-night-schedule``.
    ``schedule`` is the schedule's JSON (``NightSchedule.to_dict()``), already checked
    against its sealed companion; it must be the night's and unsealed.
    """
    arm = sealed_night_arm(metadata)
    if arm is None:
        return None
    schedule_id = arm.get("schedule_id") or "its night schedule"
    night = f"night {arm.get('night_id')}" if arm.get("night_id") else "this night"
    unseal = (
        f"{night} ran with an arm from the sealed night schedule {schedule_id}, and its logs name the arm. "
        f"Run `muse-tmr unseal-night-schedule <schedule>` after the series' last delayed retest, "
        "then export with --night-schedule <schedule>"
    )
    if schedule is None:
        return unseal
    if schedule.get("schedule_id") != arm.get("schedule_id"):
        return f"--night-schedule is {schedule.get('schedule_id')}, but {night} ran from {schedule_id}"
    if schedule.get("commitment_sha256") != arm.get("schedule_commitment_sha256"):
        return f"night schedule {schedule_id} does not match the commitment {night} ran with"
    if not schedule.get("unsealed_at_utc"):
        return f"night schedule {schedule_id} is still sealed. " + unseal
    return None


def ensure_exportable(recording_dir: Path, schedule: Optional[Mapping[str, object]] = None) -> None:
    """SealedNightError when ``blinding_refusal()`` refuses the night."""
    refusal = blinding_refusal(load_metadata(Path(recording_dir)), schedule)
    if refusal is not None:
        raise SealedNightError(refusal)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def collect_night_annotations(
    recording_dir: Path,
    *,
    t0: Optional[float] = None,
    night_start: Optional[float] = None,
    hypnograms: Sequence[Path] = (),
    to_unix: Optional[Callable[[float], float]] = None,
    include_hypnogram: bool = True,
    night_schedule: Optional[Mapping[str, object]] = None,
) -> Tuple[NightAnnotation, ...]:
    """The night's annotations (see the module docstring), sorted by time.

    ``t0`` (Unix seconds) is where ``onset_s`` counts from: the export's start. Without
    it: the Pilot 4/5 summary's ``first_epoch_start``, else ``metadata.json``'s
    ``started_at``. ``night_start`` is what the night logs' ``timestamp_seconds`` count
    from; default ``first_epoch_start``, else ``t0``. ``to_unix`` turns the onset of a
    hypnogram given in recording seconds into host time; an export passes
    ``ContinuousRecording.unix_time``, which counts on the EEG sample clock from the
    first frame, and without it the onset is taken as host seconds from ``t0``.
    ``hypnograms`` are extra stage files (``read_hypnogram()``).

    A synthetic night's ``hypnogram.csv`` counts on the generator's sample clock from
    its start, and the night's first frame holds the generator's first EEG sample, so
    it goes through ``to_unix`` too; without ``to_unix`` it counts from ``started_at``.
    Another recording's ``hypnogram.csv`` is not read: its time base is unknown, so it
    goes in through ``hypnograms``. SealedNightError for a night from a sealed schedule
    (``blinding_refusal()``).
    """
    recording_dir = Path(recording_dir).expanduser()
    metadata = load_metadata(recording_dir)
    refusal = blinding_refusal(metadata, night_schedule)
    if refusal is not None:
        raise SealedNightError(refusal)
    started_at = _iso_seconds(metadata.get("started_at"))
    first_epoch_start = _first_epoch_start(recording_dir)
    if t0 is None:
        t0 = first_epoch_start if first_epoch_start is not None else started_at
    if t0 is None:
        raise ValueError(f"{recording_dir} gives no start time: pass t0")
    if night_start is None:
        night_start = first_epoch_start if first_epoch_start is not None else t0
    synthetic_to_unix = to_unix
    if synthetic_to_unix is None and started_at is not None:
        synthetic_to_unix = lambda seconds: started_at + seconds  # noqa: E731
    if to_unix is None:
        to_unix = lambda seconds: t0 + seconds  # noqa: E731

    annotations: List[NightAnnotation] = []
    annotations += _recording_events(recording_dir / "events.jsonl", started_at, night_start)
    scheduler_rows = _read_jsonl(recording_dir / "scheduler_events.jsonl")
    gate_rows = _read_jsonl(recording_dir / GATE_LOG_FILENAME)
    decided = _decision_time(gate_rows)
    annotations += _scheduler_events(scheduler_rows, night_start, decided)
    if gate_rows:
        annotations += _gate_runs(gate_rows, night_start)
    else:
        annotations += _gate_runs_from_scheduler(scheduler_rows, night_start, decided)
    annotations += _guard_events(_read_jsonl(recording_dir / "arousal_guard_events.jsonl"), night_start, decided)
    annotations += _audio_events(_read_jsonl(recording_dir / "audio_playback.jsonl"))
    annotations += _awakenings(_read_jsonl(recording_dir / "awakening_events.jsonl"), night_start)
    synthetic_hypnogram = recording_dir / "hypnogram.csv"
    if (
        include_hypnogram
        and metadata.get("synthetic") is True
        and synthetic_hypnogram.exists()
        and synthetic_to_unix is not None
    ):
        annotations += _stage_annotations(read_hypnogram(synthetic_hypnogram), synthetic_to_unix, "hypnogram.csv")
    for path in hypnograms:
        path = Path(path).expanduser()
        annotations += _stage_annotations(read_hypnogram(path), to_unix, path.name)
    ordered = sorted(annotations, key=lambda item: (item.time_unix, item.kind, item.text))
    return tuple(replace(item, onset_s=item.time_unix - t0) for item in ordered)


def notes_for(metadata: Mapping[str, object]) -> Tuple[str, ...]:
    """The notes an export of this recording carries at its start: the synthetic note,
    and the banner of an imported night."""
    notes = []
    if metadata.get("synthetic") is True:
        notes.append(SYNTHETIC_NOTE)
    block = metadata.get("import")
    if isinstance(block, Mapping) and block.get("banner"):
        notes.append(str(block["banner"]))
    return tuple(notes)


def recorded_with_muse(metadata: Mapping[str, object]) -> bool:
    """Whether the night comes from a Muse S Athena, so that an export may name the
    headband, its reference and its full scale: every night REMind records or generates,
    and the imports of the apps that read one. Not a night imported from an EDF, whose
    device the import does not record (a BOAS headband night is a Bitbrain one)."""
    block = metadata.get("import")
    return not (isinstance(block, Mapping) and block.get("format") == "edf")


def _first_epoch_start(recording_dir: Path) -> Optional[float]:
    for name in PILOT_SUMMARY_FILENAMES:
        path = recording_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = payload.get("first_epoch_start") if isinstance(payload, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _iso_seconds(value: object) -> Optional[float]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        moment = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.timestamp()


def _seconds(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _codes(row: Mapping[str, object]) -> Tuple[str, ...]:
    return tuple(str(code) for code in row.get("reason_codes", ()) or ())


def _recording_events(path: Path, started_at: Optional[float], night_start: float) -> List[NightAnnotation]:
    rows = _read_jsonl(path)
    reference = next(
        (_seconds(row.get("timestamp")) for row in rows if row.get("event") == "recording_started"),
        None,
    )
    annotations = []
    for row in rows:
        name = str(row.get("event", "")).strip()
        if not name or name == "audio_playback_result":  # audio_playback.jsonl has each play
            continue
        when = _iso_seconds(row.get("timestamp_utc"))
        if when is None and _seconds(row.get("timestamp_seconds")) is not None:
            when = night_start + float(row["timestamp_seconds"])  # type: ignore[arg-type]
        if when is None and started_at is not None and reference is not None and _seconds(row.get("timestamp")) is not None:
            when = started_at + float(row["timestamp"]) - reference  # type: ignore[arg-type]
        if when is None:
            continue
        details = row.get("details") if isinstance(row.get("details"), Mapping) else {}
        merged = {**{key: row[key] for key in row if key in _SAFE_DETAILS}, **dict(details)}  # type: ignore[arg-type]
        parts = [name] + [f"{key}={merged[key]}" for key in _SAFE_DETAILS if key in merged and merged[key] not in (None, "")]
        annotations.append(NightAnnotation(when, 0.0, " ".join(parts), "recording", path.name))
    return annotations


def _span_ends(times: Sequence[float]) -> List[float]:
    """For each of the sorted ``times``, the next later one; the last ones get one
    median step more."""
    steps = [b - a for a, b in zip(times, times[1:]) if b > a]
    step = statistics.median(steps) if steps else 0.0
    ends = [0.0] * len(times)
    later: Optional[float] = None
    for position in range(len(times) - 1, -1, -1):
        value = times[position]
        if position + 1 < len(times) and times[position + 1] > value:
            later = times[position + 1]
        ends[position] = later if later is not None and later > value else value + step
    return ends


def _decision_time(gate_rows: Sequence[Mapping[str, object]]) -> Callable[[float], float]:
    """When the epoch that logged a scheduler or guard row at ``seconds`` was decided:
    the decision of the first epoch of the gate log starting at or after ``seconds``, and
    without a gate log, or past its end, ``DEFAULT_EPOCH_SECONDS`` after ``seconds``."""
    epochs = sorted(
        (float(row["timestamp_seconds"]), float(row.get("duration_seconds", 0.0)))  # type: ignore[arg-type]
        for row in gate_rows
        if _seconds(row.get("timestamp_seconds")) is not None
    )
    starts = [start for start, _ in epochs]

    def decided(seconds: float) -> float:
        position = bisect.bisect_left(starts, seconds - _TIME_TOLERANCE_SECONDS)
        if position < len(epochs):
            start, duration = epochs[position]
            return start + duration
        return seconds + DEFAULT_EPOCH_SECONDS

    return decided


def _timed_rows(
    rows: Sequence[Mapping[str, object]],
    night_start: float,
    decided: Callable[[float], float],
) -> List[Tuple[float, Mapping[str, object]]]:
    """(host time of its epoch's decision, row) of the rows with a ``timestamp_seconds``,
    in time order."""
    timed = [
        (night_start + decided(float(row["timestamp_seconds"])), row)  # type: ignore[arg-type]
        for row in rows
        if _seconds(row.get("timestamp_seconds")) is not None
    ]
    return sorted(timed, key=lambda item: item[0])


def _merge_runs(items: Sequence[Tuple[float, float, str, str]], source: str) -> List[NightAnnotation]:
    """(start, end, text, kind) in order; equal neighbours of a merged kind become one."""
    annotations: List[NightAnnotation] = []
    run_start = run_end = 0.0
    run_text = run_kind = None
    count = 0

    def flush() -> None:
        if run_text is None:
            return
        text = run_text if count == 1 else f"{run_text} n={count}"
        annotations.append(NightAnnotation(run_start, max(0.0, run_end - run_start), text, str(run_kind), source))

    for start, end, text, kind in items:
        if kind in _MERGED_KINDS and kind == run_kind and text == run_text:
            run_end = end
            count += 1
            continue
        flush()
        run_start, run_end, run_text, run_kind, count = start, end, text, kind, 1
        if kind not in _MERGED_KINDS:
            # A point event: never extended by a neighbour.
            run_end = start
    flush()
    return annotations


def _scheduler_events(
    rows: Sequence[Mapping[str, object]],
    night_start: float,
    decided: Callable[[float], float],
) -> List[NightAnnotation]:
    timed = _timed_rows(rows, night_start, decided)
    ends = _span_ends([start for start, _ in timed])
    items = []
    for (start, row), end in zip(timed, ends):
        event_type = str(row.get("event_type", ""))
        if event_type not in ("play", "skip", "pause", "stop"):
            continue
        parts = [f"scheduler {event_type}"]
        for key, name in (("cue_id", "cue"), ("protocol", "protocol"), ("puzzle_id", "puzzle")):
            if row.get(key):
                parts.append(f"{name}={row[key]}")
        codes = _codes(row)
        if codes:
            parts.append("reasons=" + ",".join(codes))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        if metadata.get("virtual_play"):  # type: ignore[union-attr]
            parts.append("virtual_play")
        items.append((start, end, " ".join(parts), f"scheduler_{event_type}"))
    return _merge_runs(items, "scheduler_events.jsonl")


def _gate_runs(rows: Sequence[Mapping[str, object]], night_start: float) -> List[NightAnnotation]:
    records = sorted(
        (
            (float(row["timestamp_seconds"]), int(row.get("index", 0)), float(row.get("duration_seconds", 0.0)), bool(row.get("gate_open")))
            for row in rows
            if _seconds(row.get("timestamp_seconds")) is not None
        ),
        key=lambda record: (record[0], record[1]),
    )
    annotations = []
    run_start: Optional[float] = None
    last_open: Optional[float] = None
    for timestamp, _index, duration, gate_open in records:
        decision = timestamp + duration
        if gate_open:
            if run_start is None:
                run_start = decision
            last_open = decision
            continue
        if run_start is not None:
            annotations.append(_gate_annotation(night_start, run_start, decision, GATE_LOG_FILENAME))
            run_start = None
    if run_start is not None and last_open is not None:
        annotations.append(_gate_annotation(night_start, run_start, last_open, GATE_LOG_FILENAME))
    return annotations


def _gate_runs_from_scheduler(
    rows: Sequence[Mapping[str, object]],
    night_start: float,
    decided: Callable[[float], float],
) -> List[NightAnnotation]:
    annotations = []
    run_start: Optional[float] = None
    last_open: Optional[float] = None
    for row in rows:
        seconds = _seconds(row.get("timestamp_seconds"))
        if seconds is None:
            continue
        seconds = decided(seconds)
        codes = _codes(row)
        if "rem_gate_open" in codes:
            if run_start is None:
                run_start = seconds
            last_open = seconds
        elif "rem_gate_closed" in codes and run_start is not None:
            annotations.append(_gate_annotation(night_start, run_start, seconds, "scheduler_events.jsonl"))
            run_start = None
    if run_start is not None and last_open is not None:
        annotations.append(_gate_annotation(night_start, run_start, last_open, "scheduler_events.jsonl"))
    return annotations


def _gate_annotation(night_start: float, start: float, end: float, source: str) -> NightAnnotation:
    return NightAnnotation(night_start + start, max(0.0, end - start), "rem_gate_open", "rem_gate_open", source)


def _guard_events(
    rows: Sequence[Mapping[str, object]],
    night_start: float,
    decided: Callable[[float], float],
) -> List[NightAnnotation]:
    timed = _timed_rows(rows, night_start, decided)
    ends = _span_ends([start for start, _ in timed])
    items = []
    for (start, row), end in zip(timed, ends):
        action = str(row.get("action", ""))
        if action in ("", "allow"):
            items.append((start, end, "", "allow"))
            continue
        parts = [f"arousal_guard {action}"]
        codes = _codes(row)
        if codes:
            parts.append("reasons=" + ",".join(codes))
        if action == "lower_volume" and _seconds(row.get("volume_multiplier")) is not None:
            parts.append(f"volume_multiplier={float(row['volume_multiplier']):g}")  # type: ignore[arg-type]
        if action == "pause" and _seconds(row.get("pause_seconds")):
            parts.append(f"pause_seconds={float(row['pause_seconds']):g}")  # type: ignore[arg-type]
        items.append((start, end, " ".join(parts), "arousal_guard"))
    return [item for item in _merge_runs(items, "arousal_guard_events.jsonl") if item.kind != "allow"]


def _audio_events(rows: Sequence[Mapping[str, object]]) -> List[NightAnnotation]:
    annotations = []
    for row in rows:
        started = _iso_seconds(row.get("started_at"))
        ended = _iso_seconds(row.get("ended_at"))
        when = started if started is not None else ended
        if when is None:
            continue
        status = str(row.get("status", "unknown"))
        cue_id = str(row.get("cue_id", ""))
        if cue_id == "emergency-stop":
            text = "audio emergency_stop"
        else:
            parts = [f"audio {status}", f"cue={cue_id}"]
            if _seconds(row.get("effective_volume")) is not None and status == "played":
                parts.append(f"volume={float(row['effective_volume']):.3f}")  # type: ignore[arg-type]
            if row.get("backend_name"):
                # A dry-run or mock "played" made no sound; the backend says which it was.
                parts.append(f"backend={row['backend_name']}")
            if row.get("arm") and row.get("arm") != "cue":
                parts.append(f"arm={row['arm']}")
            text = " ".join(parts)
        duration = max(0.0, ended - started) if started is not None and ended is not None else 0.0
        annotations.append(NightAnnotation(when, duration, text, "audio", "audio_playback.jsonl"))
    return annotations


def _awakenings(rows: Sequence[Mapping[str, object]], night_start: float) -> List[NightAnnotation]:
    annotations = []
    for row in rows:
        seconds = _seconds(row.get("timestamp_seconds"))
        when = night_start + seconds if seconds is not None else _iso_seconds(row.get("timestamp_utc"))
        if when is None:
            continue
        name = str(row.get("event_type") or "awakening")
        text = name if not row.get("attributed_to_cue") else f"{name} attributed_to_cue={row['attributed_to_cue']}"
        annotations.append(NightAnnotation(when, 0.0, text, "awakening", "awakening_events.jsonl"))
    return annotations


def _stage_annotations(
    intervals: Iterable[StageInterval],
    to_unix: Callable[[float], float],
    source: str,
) -> List[NightAnnotation]:
    annotations = []
    for interval in intervals:
        when = interval.onset_s if interval.absolute else float(to_unix(interval.onset_s))
        annotations.append(
            NightAnnotation(when, interval.duration_s, edf_stage_text(interval.label), "stage", source)
        )
    return annotations
