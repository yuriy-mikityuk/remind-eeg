"""How a cueing night's timing compares with its protocol preset (#231).

It reads what the night logged, ``scheduler_events.jsonl``, ``audio_playback.jsonl``,
``rem_gate_decisions.jsonl``, ``awakening_events.jsonl`` and ``metadata.json``, or a
Pilot 3 report's plan and epochs, and compares them with the preset the night ran:
the intervals between cues, the cues per REM period, the stable REM before the first
cue, whether the pause rules held, the volume of the cues in a period, the cue length
and the TLR block against its plan. A scheduler play the player blocked (after an
emergency stop) or skipped made no sound, so it is left out of every check and only
counted; each play is paired with its line of the audio log by ``audio_for_plays()``,
as the cue arousal scorer pairs them.

A REM period here is a gate-open run as ``muse_tmr.validation.gate_log`` defines it
(``load_gate_open_runs()``), the same as for the cue arousal scorer; a cue is placed at
the decision of the epoch that played it (``emitting_record()``). It is retrospective,
decides and plays nothing, and imports nothing from ``muse_tmr.audio``.

Parameter fidelity is not result fidelity: every report carries the
``parameter_fidelity_is_not_result_fidelity`` block that says why.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from muse_tmr.protocol import (
    ProtocolPreset,
    get_protocol_preset,
    load_tmr_scheduler_events,
    protocol_block_of,
    resolve_protocol_preset,
)
from muse_tmr.validation.gate_log import (
    REM_GATE_DECISIONS_FILENAME,
    TIME_TOLERANCE_SECONDS,
    GateDecisionRecord,
    GateOpenRun,
    emitting_record,
    load_gate_open_runs,
    load_rem_gate_decisions,
    run_at,
)
from muse_tmr.validation.pilot4 import audio_for_plays, load_awakening_events

PROTOCOL_FIDELITY_SCHEMA_VERSION = 1
REPORT_FILENAME = "protocol_fidelity.json"
MARKDOWN_FILENAME = "protocol_fidelity.md"
SYNTHETIC_NOTE = "synthetic input, not sleep"
SUMMARY_FILENAMES = ("pilot4_summary.json", "pilot5_summary.json")
STREAMS = ("tlr", "puzzle")
# A cue decided at the end of a 30 s epoch, when no gate log says how long its epoch was.
_DEFAULT_EPOCH_SECONDS = 30.0
# Two cue lengths this close are the same.
_DURATION_TOLERANCE_SECONDS = 0.01

# Always in the report: what matching a paper's parameters does not show.
PARAMETER_FIDELITY_IS_NOT_RESULT_FIDELITY = {
    "statement": (
        "Parameter fidelity is not result fidelity: this report says how the night's cue timing compares "
        "with a paper's parameters, not that anything the paper measured happened."
    ),
    "no_psg_verified_rem": (
        "A REM period here is a gate-open run of REMind's detector on forehead EEG; no PSG scored it, and "
        "the detector has only been checked against an automated stager."
    ),
    "no_eog": "No EOG: no eye signal, lucid or not, is detected or verified.",
    "gain_not_db_spl": (
        "Volumes are a software gain from 0 to 1, not dB SPL, so a paper's sound levels cannot be compared "
        "with them."
    ),
    "first_cue_latency": (
        "With 30 s epochs and the gate's 60 s stable window the first cue comes 60 to 85 s after a REM onset "
        "(a synthetic test, #167): later than the Nijmegen arm of Konkoly et al. 2021, which cued after the "
        "first 30 s epoch of REM, and than Jafarzadeh Esfahani, Picard-Deland et al. 2024, who cued about "
        "20 s after the first rapid eye movement."
    ),
}

# Stop rules that the gate itself keeps: cueing ends when the gate-open run does.
_REM_STOPS = {"nrem", "rem_gate_closed", "rem_unstable", "not_slow_wave_sleep"}

# The arousal guard's pause reasons and the preset pause trigger each stands for.
GUARD_TRIGGERS = {
    "motion_arousal_proxy": "movement",
    "motion_not_still": "movement",
    "cue_related_arousal": "movement",
    "alpha_arousal_proxy": "alpha",
    "sudden_heart_rate_change": "heart_rate",
    "heart_rate_out_of_range": "heart_rate",
    "multi_channel_eeg_artifact": "poor_signal",
    "repeated_artifact_quality": "poor_signal",
    "repeated_arousal_guard_pause": "arousal",
}


@dataclass(frozen=True)
class ProtocolFidelityReport:
    recording: str
    preset: Mapping[str, object]
    night_protocol: Optional[Mapping[str, object]]
    settings: Mapping[str, object]
    inputs: Mapping[str, Optional[str]]
    summary: Mapping[str, object]
    rem_periods: Tuple[Mapping[str, object], ...]
    checks: Mapping[str, Mapping[str, object]]
    deviations: Tuple[Mapping[str, object], ...]
    limitations: Tuple[str, ...]
    synthetic: bool = False
    generated_at_utc: str = ""
    schema_version: int = PROTOCOL_FIDELITY_SCHEMA_VERSION
    parameter_fidelity_is_not_result_fidelity: Mapping[str, str] = field(
        default_factory=lambda: dict(PARAMETER_FIDELITY_IS_NOT_RESULT_FIDELITY)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "rem_periods", tuple(self.rem_periods))
        object.__setattr__(self, "deviations", tuple(self.deviations))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())
        # The block is not optional: a report without it would read as more than it is.
        object.__setattr__(
            self,
            "parameter_fidelity_is_not_result_fidelity",
            {**PARAMETER_FIDELITY_IS_NOT_RESULT_FIDELITY, **dict(self.parameter_fidelity_is_not_result_fidelity)},
        )

    def statuses(self) -> Dict[str, object]:
        return {name: check.get("status") for name, check in self.checks.items()}

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "schema_version": self.schema_version,
            "recording": self.recording,
            "generated_at_utc": self.generated_at_utc,
            "synthetic": self.synthetic,
            "parameter_fidelity_is_not_result_fidelity": dict(self.parameter_fidelity_is_not_result_fidelity),
            "preset": dict(self.preset),
            "night_protocol": dict(self.night_protocol) if self.night_protocol else None,
            "settings": dict(self.settings),
            "inputs": dict(self.inputs),
            "summary": dict(self.summary),
            "rem_periods": [dict(period) for period in self.rem_periods],
            "checks": {name: dict(check) for name, check in self.checks.items()},
            "deviations": [dict(item) for item in self.deviations],
            "limitations": list(self.limitations),
        }
        if self.synthetic:
            payload["note"] = SYNTHETIC_NOTE
        return payload

    def save(self, output_path: Path) -> Path:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output_path

    def save_markdown(self, output_path: Path) -> Path:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def save_outputs(self, output_dir: Path) -> Dict[str, Path]:
        output_dir = Path(output_dir).expanduser()
        return {
            "report": self.save(output_dir / REPORT_FILENAME),
            "markdown": self.save_markdown(output_dir / MARKDOWN_FILENAME),
        }

    def to_markdown(self) -> str:
        preset = self.preset
        lines = [f"# Protocol fidelity: {self.recording}", ""]
        if self.synthetic:
            lines.extend([f"**{SYNTHETIC_NOTE.capitalize()}.**", ""])
        lines.extend(["## Parameter fidelity is not result fidelity", ""])
        lines.extend(f"- {text}" for text in self.parameter_fidelity_is_not_result_fidelity.values())
        lines.extend([
            "",
            "## Preset",
            "",
            f"- {preset.get('name')} v{preset.get('version')}: {preset.get('title')}",
            f"- content hash `{preset.get('content_hash')}`, compared because: {preset.get('source')}",
        ])
        for citation in preset.get("citations", ()) or ():
            arm = f", {citation.get('arm')}" if citation.get("arm") else ""
            lines.append(
                f"- {citation.get('authors')} {citation.get('year')}{arm} ({citation.get('source_check')}): "
                f"{citation.get('url')}"
            )
        summary = self.summary
        plays = summary.get("plays", {}) or {}
        not_played = summary.get("plays_blocked_or_skipped", {}) or {}
        lines.extend([
            "",
            "## Night",
            "",
            f"- REM periods (gate-open runs): {summary.get('rem_periods')}, with cues: {summary.get('rem_periods_with_cues')}",
            f"- plays: TLR {plays.get('tlr', 0)}, puzzle {plays.get('puzzle', 0)}; outside REM periods: "
            f"{summary.get('plays_outside_rem_periods')}",
            f"- blocked or skipped by the player, left out of the checks: TLR {not_played.get('tlr', 0)}, "
            f"puzzle {not_played.get('puzzle', 0)}",
            "",
            "## Checks",
            "",
            "| Check | Status | Observed | Preset |",
            "| --- | --- | --- | --- |",
        ])
        for name, check in self.checks.items():
            lines.append(
                f"| {name} | {check.get('status')} | {_cell(check.get('observed'))} | {_cell(check.get('published'))} |"
            )
        if self.deviations:
            lines.extend(["", "## Where REMind differs from the paper", ""])
            lines.extend(
                f"- {item.get('parameter')}: published {item.get('published')}; REMind {item.get('remind')}"
                for item in self.deviations
            )
        lines.extend(["", "## Limitations", ""])
        if self.limitations:
            lines.extend(f"- `{code}`" for code in self.limitations)
        else:
            lines.append("- none")
        return "\n".join(lines) + "\n"


def build_protocol_fidelity(
    recording_dir: Path,
    *,
    pilot3_report: Optional[Path] = None,
    preset_name: Optional[str] = None,
) -> ProtocolFidelityReport:
    """The fidelity report of a Pilot 4/5 night in ``recording_dir``, or with
    ``pilot3_report`` of that report's plan over the recording it replayed.

    ``preset_name`` compares with another preset than the one the night recorded; a
    night without a protocol block is compared with ``remind_default_v1``.
    """
    recording_dir = Path(recording_dir).expanduser()
    night = _load_night(recording_dir, pilot3_report)
    limitations: List[str] = []
    preset, source = _preset_for(night.protocol, preset_name, limitations)
    settings = dict((night.protocol or {}).get("settings") or preset.runner_settings().to_dict())
    if night.synthetic:
        limitations.insert(0, "synthetic_input")
    if not night.records:
        limitations.append("no_gate_log")
    if not night.events_found:
        limitations.append("no_scheduler_events")
    if night.pilot3:
        limitations.append("pilot3_plan_not_played")
    elif not night.audio_found:
        limitations.append("no_audio_log")

    placed, not_played = _place_plays(night)
    pause_times = _guard_pause_times(night)
    if sum(not_played.values()):
        limitations.append("plays_blocked_or_skipped")
    if night.audio_mismatch:
        limitations.append("audio_log_mismatch")
    periods = _rem_periods(night, placed)
    stride = _stride(night.records)
    checks = {
        "inter_cue_intervals": _interval_check(preset, placed, night.runs),
        "cues_per_rem_period": _count_check(preset, placed, night.runs, pause_times),
        "stable_rem_before_first_cue": _stable_check(preset, placed, night.runs, stride),
        "pause_rules": _pause_check(preset, night, placed, limitations),
        "volume_ramp": _ramp_check(preset, placed, night),
        "cue_duration": _duration_check(preset, placed),
        "tlr_block": _tlr_check(preset, placed, night, pause_times),
    }
    plays_outside = sum(1 for play in placed if play.run is None)
    summary = {
        "rem_periods": len(night.runs),
        "rem_periods_with_cues": sum(1 for period in periods if sum(period["plays"].values()) > 0),
        "plays": {stream: sum(1 for play in placed if play.stream == stream) for stream in STREAMS},
        "plays_blocked_or_skipped": not_played,
        "plays_outside_rem_periods": plays_outside,
        "uncued_puzzle_play_count": night.uncued_puzzle_play_count,
        "stride_seconds": stride,
    }
    return ProtocolFidelityReport(
        recording=recording_dir.name,
        preset={
            "name": preset.name,
            "version": preset.version,
            "title": preset.title,
            "content_hash": preset.content_hash(),
            "runnable": preset.runnable,
            "source": source,
            "citations": [
                {
                    "authors": citation.authors,
                    "year": citation.year,
                    "arm": citation.arm,
                    "source_check": citation.source_check,
                    "url": citation.url,
                }
                for citation in preset.citations
            ],
        },
        night_protocol=night.protocol,
        settings=settings,
        inputs=night.sources,
        summary=summary,
        rem_periods=tuple(periods),
        checks=checks,
        deviations=tuple(
            {"parameter": item.parameter, "published": item.published, "remind": item.remind}
            for item in preset.deviations
        ),
        limitations=tuple(dict.fromkeys(limitations)),
        synthetic=night.synthetic,
    )


def load_protocol_fidelity_report(input_path: Path) -> Dict[str, object]:
    return json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Night:
    events: Tuple[Mapping[str, object], ...]
    events_found: bool
    records: Tuple[GateDecisionRecord, ...]
    runs: Tuple[GateOpenRun, ...]
    # The audio log line of each play event, in order (None: no log, or after a mismatch).
    play_audio: Tuple[Optional[Mapping[str, object]], ...]
    audio_found: bool
    audio_mismatch: bool
    awakening_offsets: Optional[Tuple[float, ...]]
    awakenings_unplaced: int
    protocol: Optional[Dict[str, object]]
    tlr_plan: Optional[Mapping[str, object]]
    uncued_puzzle_play_count: Optional[int]
    pilot3: bool
    synthetic: bool
    sources: Dict[str, Optional[str]]


@dataclass(frozen=True)
class _Play:
    stream: str
    event: Mapping[str, object]
    decision_seconds: float
    record: Optional[GateDecisionRecord]
    run: Optional[GateOpenRun]
    volume: Optional[float]
    duration_seconds: Optional[float]


def _load_night(recording_dir: Path, pilot3_report: Optional[Path]) -> _Night:
    sources: Dict[str, Optional[str]] = {}
    metadata = _read_json(recording_dir / "metadata.json")
    synthetic = bool(metadata.get("synthetic"))
    if pilot3_report is not None:
        report_path = Path(pilot3_report).expanduser()
        report = _read_json(report_path)
        sources["pilot3_report"] = report_path.name
        events = tuple(dict(event) for event in report.get("scheduler_events", ()) or ())
        records = load_rem_gate_decisions(report)
        runs = load_gate_open_runs(report)
        metrics = dict(report.get("metrics", {}) or {})
        return _Night(
            events=events,
            events_found="scheduler_events" in report,
            records=records,
            runs=runs,
            play_audio=tuple(None for event in events if event.get("event_type") == "play"),
            audio_found=False,
            audio_mismatch=False,
            awakening_offsets=None,
            awakenings_unplaced=0,
            protocol=protocol_block_of(report),
            tlr_plan=None,
            uncued_puzzle_play_count=_optional_int(metrics.get("uncued_puzzle_play_count")),
            pilot3=True,
            synthetic=synthetic or bool(report.get("synthetic")),
            sources=sources,
        )

    events_path = recording_dir / "scheduler_events.jsonl"
    events = tuple(event.to_dict() for event in load_tmr_scheduler_events(events_path))
    sources["scheduler_events"] = events_path.name if events_path.exists() else None
    records = load_rem_gate_decisions(recording_dir)
    runs = load_gate_open_runs(recording_dir)
    sources["gate_log"] = REM_GATE_DECISIONS_FILENAME if records else None
    audio_path = recording_dir / "audio_playback.jsonl"
    play_audio, audio_found, audio_mismatch = audio_for_plays(
        [event for event in events if event.get("event_type") == "play"], audio_path
    )
    sources["audio_log"] = audio_path.name if audio_found else None
    summary = {}
    for name in SUMMARY_FILENAMES:
        summary = _read_json(recording_dir / name)
        if summary:
            sources["summary"] = name
            break
    sources["metadata"] = "metadata.json" if metadata else None
    first_epoch_start = _optional_float(summary.get("first_epoch_start"))
    awakening_path = recording_dir / "awakening_events.jsonl"
    offsets, unplaced = None, 0
    if awakening_path.exists():
        offsets, unplaced = _awakening_offsets(load_awakening_events(awakening_path), first_epoch_start)
        sources["awakening_events"] = awakening_path.name
    return _Night(
        events=events,
        events_found=events_path.exists(),
        records=records,
        runs=runs,
        play_audio=play_audio,
        audio_found=audio_found,
        audio_mismatch=audio_mismatch,
        awakening_offsets=offsets,
        awakenings_unplaced=unplaced,
        protocol=protocol_block_of(metadata) or protocol_block_of(summary),
        tlr_plan=metadata.get("tlr_block_plan") if isinstance(metadata.get("tlr_block_plan"), Mapping) else None,
        uncued_puzzle_play_count=_optional_int(summary.get("uncued_puzzle_play_count")),
        pilot3=False,
        synthetic=synthetic,
        sources=sources,
    )


def _preset_for(
    protocol: Optional[Mapping[str, object]],
    preset_name: Optional[str],
    limitations: List[str],
) -> Tuple[ProtocolPreset, str]:
    if preset_name:
        preset = get_protocol_preset(preset_name)
        if protocol and protocol.get("name") != preset.name:
            limitations.append("compared_with_another_preset")
        return preset, "asked for with --protocol-preset"
    if not protocol:
        limitations.append("no_protocol_block")
        return resolve_protocol_preset(None), "the night recorded no preset; the default"
    try:
        preset = get_protocol_preset(str(protocol.get("name")))
    except KeyError:
        limitations.append("unknown_preset_in_night")
        return resolve_protocol_preset(None), "the night's preset is not in this registry; the default"
    if protocol.get("content_hash") != preset.content_hash():
        limitations.append("preset_changed_since_night")
    if protocol.get("overrides"):
        limitations.append("night_overrode_preset_settings")
    return preset, "the night's preset"


def _awakening_offsets(events, first_epoch_start: Optional[float]) -> Tuple[Tuple[float, ...], int]:
    """Awakening markers in seconds from the first epoch's start, placed as the cue
    arousal scorer places them (``AwakeningEvent.night_offset_seconds()``): markers
    without ``timestamp_seconds`` by their UTC time when the summary gives the start."""
    offsets = []
    unplaced = 0
    for event in events:
        offset = event.night_offset_seconds(first_epoch_start)
        if offset is None:
            unplaced += 1
            continue
        offsets.append(offset)
    return tuple(sorted(offsets)), unplaced


# --------------------------------------------------------------------------- #
# Placing the plays
# --------------------------------------------------------------------------- #
def _place_plays(night: _Night) -> Tuple[List[_Play], Dict[str, int]]:
    """The plays the player played, each at the decision of the epoch that played it, and
    by stream how many it blocked or skipped.

    A play with no audio line (a Pilot 3 plan, a night without its audio log, or after a
    mismatch) counts as played; its volume then comes from the scheduler event.
    """
    records = sorted(night.records, key=lambda record: (record.timestamp_seconds, record.index))
    starts = [record.timestamp_seconds for record in records]
    plays = []
    not_played = {stream: 0 for stream in STREAMS}
    play_events = [event for event in night.events if event.get("event_type") == "play"]
    for event, audio in zip(play_events, night.play_audio):
        metadata = dict(event.get("metadata", {}) or {})
        stream = str(event.get("protocol") or "")
        if metadata.get("synthetic_cue") or stream not in STREAMS:
            continue
        if audio is not None and audio.get("status") != "played":
            not_played[stream] += 1
            continue
        scheduled = float(event["timestamp_seconds"])
        record = emitting_record(records, scheduled, starts=starts)
        if metadata.get("decision_seconds") is not None:
            decision = float(metadata["decision_seconds"])
        elif record is not None:
            decision = record.decision_seconds
        else:
            decision = scheduled + _DEFAULT_EPOCH_SECONDS
        volume = None if audio is None else _optional_float(audio.get("effective_volume"))
        if volume is None:
            volume = _optional_float(metadata.get("effective_volume"))
        plays.append(
            _Play(
                stream=stream,
                event=event,
                decision_seconds=decision,
                record=record,
                run=run_at(night.runs, decision),
                volume=volume,
                duration_seconds=_optional_float(metadata.get("duration_seconds")),
            )
        )
    return sorted(plays, key=lambda play: play.decision_seconds), not_played


def _rem_periods(night: _Night, plays: Sequence[_Play]) -> List[Dict[str, object]]:
    periods = []
    for run in night.runs:
        in_run = [play for play in plays if play.run is not None and play.run.index == run.index]
        first = in_run[0] if in_run else None
        periods.append(
            {
                **run.to_dict(),
                "plays": {stream: sum(1 for play in in_run if play.stream == stream) for stream in STREAMS},
                "first_cue_seconds_after_gate_open": (
                    None if first is None else _round(first.decision_seconds - run.start_seconds)
                ),
                "stable_rem_seconds_at_first_cue": (
                    None if first is None or first.record is None else first.record.stable_seconds
                ),
            }
        )
    return periods


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _stream_plays(plays: Sequence[_Play], stream: str) -> List[_Play]:
    return [play for play in plays if stream == "any" or play.stream == stream]


def _by_run(plays: Sequence[_Play]) -> Dict[int, List[_Play]]:
    grouped: Dict[int, List[_Play]] = {}
    for play in plays:
        if play.run is not None:
            grouped.setdefault(play.run.index, []).append(play)
    return grouped


def _guard_pause_times(night: _Night) -> List[float]:
    """When the arousal guard paused the scheduler, each at the decision of its epoch."""
    records = sorted(night.records, key=lambda record: (record.timestamp_seconds, record.index))
    starts = [record.timestamp_seconds for record in records]
    times = []
    for event in night.events:
        if event.get("event_type") == "pause" and "arousal_guard_pause" in (event.get("reason_codes") or ()):
            record = emitting_record(records, float(event["timestamp_seconds"]), starts=starts)
            if record is not None:
                times.append(record.decision_seconds)
    return sorted(times)


def _by_block(plays: Sequence[_Play], pause_times: Sequence[float]) -> List[Tuple[int, int, List[_Play]]]:
    """The plays of each REM period by the scheduler's REM block, as (period, block,
    plays) in time order, blocks without plays left out.

    A guard pause ends the scheduler's block; the next cue of the same REM period starts
    a new one, with a new TLR block and up to ``max_puzzle_cues_per_block`` puzzle cues.
    A play's block is the number of guard pauses in its period before it.
    """
    grouped: Dict[Tuple[int, int], List[_Play]] = {}
    for play in plays:
        if play.run is None:
            continue
        block = sum(
            1
            for at in pause_times
            if play.run.start_seconds - TIME_TOLERANCE_SECONDS <= at < play.decision_seconds
        )
        grouped.setdefault((play.run.index, block), []).append(play)
    return [(period, block, items) for (period, block), items in sorted(grouped.items())]


def _interval_check(preset: ProtocolPreset, plays: Sequence[_Play], runs: Sequence[GateOpenRun]) -> Dict[str, object]:
    timings = [timing for timing in preset.timing if timing.interval_seconds is not None]
    if not timings:
        return {"status": "not_published", "observed": None, "published": None}
    streams = {}
    statuses = []
    for timing in timings:
        low, high = timing.interval_seconds
        intervals = []
        for in_run in _by_run(_stream_plays(plays, timing.stream)).values():
            intervals.extend(
                later.decision_seconds - earlier.decision_seconds for earlier, later in zip(in_run, in_run[1:])
            )
        shorter = sum(1 for value in intervals if value < low - TIME_TOLERANCE_SECONDS)
        longer = sum(1 for value in intervals if high is not None and value > high + TIME_TOLERANCE_SECONDS)
        if not intervals:
            status = "no_intervals"
        elif shorter or longer:
            status = "differs"
        else:
            status = "as_published"
        statuses.append(status)
        streams[timing.stream] = {
            "status": status,
            "published_seconds": [low, high],
            "interval_count": len(intervals),
            "within": len(intervals) - shorter - longer,
            "shorter": shorter,
            "longer": longer,
            "back_to_back": sum(1 for value in intervals if value <= TIME_TOLERANCE_SECONDS),
            "min_seconds": _round(min(intervals)) if intervals else None,
            "median_seconds": _round(statistics.median(intervals)) if intervals else None,
            "max_seconds": _round(max(intervals)) if intervals else None,
            "note": timing.note,
        }
    return {
        "status": _combined(statuses),
        "observed": {
            stream: f"{row['within']} of {row['interval_count']} within" for stream, row in streams.items()
        },
        "published": {stream: row["published_seconds"] for stream, row in streams.items()},
        "streams": streams,
    }


def _count_check(
    preset: ProtocolPreset,
    plays: Sequence[_Play],
    runs: Sequence[GateOpenRun],
    pause_times: Sequence[float],
) -> Dict[str, object]:
    streams = {}
    statuses = []
    for timing in preset.timing:
        stream_plays = _stream_plays(plays, timing.stream)
        counts = [
            len([play for play in stream_plays if play.run is not None and play.run.index == run.index])
            for run in runs
        ]
        blocks = _by_block(stream_plays, pause_times)
        per_block = [[len(items) for period, _, items in blocks if period == run.index] for run in runs]
        block_counts = [len(items) for _, _, items in blocks]
        maximum = timing.max_cues_per_rem_period
        with_cues = [count for count in counts if count > 0]
        if maximum is None:
            status = "not_published"
        elif not with_cues:
            status = "no_cues"
        elif any(count > maximum for count in block_counts):
            status = "differs"
        else:
            status = "as_published"
        statuses.append(status)
        streams[timing.stream] = {
            "status": status,
            "published_max": maximum,
            "per_period": counts,
            "per_block": per_block,
            "periods_with_cues": len(with_cues),
            "blocks_over_max": 0 if maximum is None else sum(1 for count in block_counts if count > maximum),
            "mean_in_periods_with_cues": _round(statistics.mean(with_cues)) if with_cues else None,
            "note": timing.note,
        }
    if not streams:
        return {"status": "not_published", "observed": None, "published": None}
    return {
        "status": _combined(statuses),
        "observed": {stream: row["per_period"] for stream, row in streams.items()},
        "published": {stream: row["published_max"] for stream, row in streams.items()},
        "streams": streams,
        "note": (
            "the published limit is compared per REM block: a guard pause ends the scheduler's block and the "
            "next cue of the same REM period starts a new one; per_period counts the whole REM period"
        ),
    }


def _stable_check(
    preset: ProtocolPreset,
    plays: Sequence[_Play],
    runs: Sequence[GateOpenRun],
    stride: float,
) -> Dict[str, object]:
    start = preset.pauses.start_after_rem_seconds
    resume = preset.pauses.resume_after_rem_seconds
    rows = []
    for position, run in enumerate(runs):
        in_run = [play for play in plays if play.run is not None and play.run.index == run.index]
        if not in_run:
            continue
        published = start if position == 0 or resume is None else resume
        observed = in_run[0].record.stable_seconds if in_run[0].record is not None else None
        rows.append({"rem_period": run.index, "published_seconds": published, "observed_seconds": observed})
    if start is None and resume is None:
        status = "not_published"
    elif not rows:
        status = "no_cues"
    else:
        statuses = []
        for row in rows:
            published, observed = row["published_seconds"], row["observed_seconds"]
            if published is None or observed is None:
                statuses.append("not_available")
            elif observed < published - TIME_TOLERANCE_SECONDS:
                statuses.append("earlier_than_published")
            elif observed >= published + stride - TIME_TOLERANCE_SECONDS:
                statuses.append("later_than_published")
            else:
                statuses.append("as_published")
            row["status"] = statuses[-1]
        status = _worst(statuses, ("earlier_than_published", "later_than_published", "not_available", "as_published"))
    observed = [row["observed_seconds"] for row in rows]
    return {
        "status": status,
        "observed": observed or None,
        "published": {"start_after_rem_seconds": start, "resume_after_rem_seconds": resume},
        "periods": rows,
        "stride_seconds": stride,
        "note": "stable REM-like time by the gate's own count at the deciding epoch, not time since a scored REM onset",
    }


def _pause_check(
    preset: ProtocolPreset,
    night: _Night,
    plays: Sequence[_Play],
    limitations: List[str],
) -> Dict[str, object]:
    rules = preset.pauses
    records = sorted(night.records, key=lambda record: (record.timestamp_seconds, record.index))
    starts = [record.timestamp_seconds for record in records]
    play_times = sorted(play.decision_seconds for play in plays)
    guard_rows = []
    not_checked = 0
    for event in night.events:
        if event.get("event_type") != "pause" or "arousal_guard_pause" not in (event.get("reason_codes") or ()):
            continue
        reasons = tuple(event.get("reason_codes") or ())
        triggers = sorted({GUARD_TRIGGERS[code] for code in reasons if code in GUARD_TRIGGERS}) or ["arousal"]
        published = [value for value in (rules.pause_for(trigger) for trigger in triggers) if value is not None]
        record = emitting_record(records, float(event["timestamp_seconds"]), starts=starts)
        at = record.decision_seconds if record is not None else float(event["timestamp_seconds"]) + _DEFAULT_EPOCH_SECONDS
        next_play = next((time for time in play_times if time > at + TIME_TOLERANCE_SECONDS), None)
        if not published:
            not_checked += 1
            continue
        required = max(published)
        gap = None if next_play is None else next_play - at
        guard_rows.append(
            {
                "at_seconds": _round(at),
                "triggers": triggers,
                "published_pause_seconds": required,
                "seconds_to_next_cue": None if gap is None else _round(gap),
                "honoured": gap is None or gap >= required - TIME_TOLERANCE_SECONDS,
            }
        )
    awakening_rows = []
    if rules.pause_after_awakening_seconds is not None and night.awakening_offsets:
        for offset in night.awakening_offsets:
            next_play = next((time for time in play_times if time > offset + TIME_TOLERANCE_SECONDS), None)
            gap = None if next_play is None else next_play - offset
            awakening_rows.append(
                {
                    "at_seconds": _round(offset),
                    "published_pause_seconds": rules.pause_after_awakening_seconds,
                    "seconds_to_next_cue": None if gap is None else _round(gap),
                    "honoured": gap is None or gap >= rules.pause_after_awakening_seconds - TIME_TOLERANCE_SECONDS,
                }
            )
    if night.awakenings_unplaced:
        limitations.append("awakening_markers_unplaced")
    outside = sum(1 for play in plays if play.run is None)
    after_arousal = _cues_after_guard_pause_in_period(plays, _guard_pause_times(night))
    stops_on_arousal = "arousal" in rules.stops_on
    violations = (
        sum(1 for row in guard_rows if not row["honoured"])
        + sum(1 for row in awakening_rows if not row["honoured"])
        + outside
        + (after_arousal if stops_on_arousal else 0)
    )
    has_rules = bool(
        rules.pause_seconds
        or rules.pause_until_clear
        or rules.pause_after_awakening_seconds is not None
        or rules.stops_on
    )
    # A pause no cue came after shows nothing either way.
    checked = sum(1 for row in guard_rows + awakening_rows if row["seconds_to_next_cue"] is not None)
    # A stop on leaving REM holds by the gate: no cue outside a gate-open run.
    stop_checked = bool(plays) and bool(set(rules.stops_on) & _REM_STOPS)
    timed = bool(rules.pause_seconds) or rules.pause_after_awakening_seconds is not None
    if not has_rules:
        status = "not_published"
    elif not plays:
        status = "no_cues"
    elif violations:
        status = "differs"
    elif checked or stop_checked:
        status = "as_published"
    elif not timed and rules.pause_until_clear:
        status = "not_checkable"
    else:
        status = "nothing_to_check"
    return {
        "status": status,
        "observed": {
            "guard_pauses_checked": len(guard_rows),
            "guard_pauses_honoured": sum(1 for row in guard_rows if row["honoured"]),
            "awakenings_checked": len(awakening_rows),
            "awakenings_honoured": sum(1 for row in awakening_rows if row["honoured"]),
            "cues_outside_rem_periods": outside,
            "cues_after_a_guard_pause_in_the_same_period": after_arousal,
        },
        "published": {
            "pause_seconds": dict(rules.pause_seconds),
            "pause_until_clear": list(rules.pause_until_clear),
            "pause_after_awakening_seconds": rules.pause_after_awakening_seconds,
            "stops_on": list(rules.stops_on),
        },
        "guard_pauses": guard_rows,
        "guard_pauses_without_a_published_duration": not_checked,
        "awakenings": awakening_rows,
        "awakening_markers": None if night.awakening_offsets is None else len(night.awakening_offsets),
        "untimed_rules_not_checked": list(rules.pause_until_clear),
        "note": rules.note,
    }


def _cues_after_guard_pause_in_period(plays: Sequence[_Play], pause_times: Sequence[float]) -> int:
    count = 0
    for play in plays:
        if play.run is None:
            continue
        if any(play.run.start_seconds - TIME_TOLERANCE_SECONDS <= at < play.decision_seconds for at in pause_times):
            count += 1
    return count


def _ramp_check(preset: ProtocolPreset, plays: Sequence[_Play], night: _Night) -> Dict[str, object]:
    ramp = preset.ramp
    published = {
        "start": ramp.start,
        "step_percent_per_cue": ramp.step_percent_per_cue,
        "step_db_per_cycle": ramp.step_db_per_cycle,
        "on_arousal": ramp.on_arousal,
        "cap": ramp.cap,
        "level_db_spl": list(ramp.level_db_spl) if ramp.level_db_spl else None,
        "background_noise": ramp.background_noise,
    }
    rows = []
    for index, in_run in sorted(_by_run(plays).items()):
        volumes = [play.volume for play in in_run if play.volume is not None]
        if not volumes:
            continue
        steps = [later - earlier for earlier, later in zip(volumes, volumes[1:])]
        rows.append(
            {
                "rem_period": index,
                "first_gain": volumes[0],
                "last_gain": volumes[-1],
                "largest_step": _round(max(steps, key=abs), 6) if steps else 0.0,
                "constant": all(abs(step) <= 1e-12 for step in steps),
            }
        )
    # Background noise alone says nothing about how loud the cues were.
    described = bool(ramp.ramps or ramp.start or ramp.on_arousal or ramp.cap or ramp.level_db_spl)
    if not rows:
        status = "not_available"
    elif ramp.ramps:
        status = "not_implemented" if all(row["constant"] for row in rows) else "differs"
    elif not described:
        status = "not_published"
    else:
        status = "as_published" if all(row["constant"] for row in rows) else "differs"
    return {
        "status": status,
        "observed": {"periods": len(rows), "constant_gain": all(row["constant"] for row in rows) if rows else None},
        "published": published,
        "periods": rows,
        "note": "gain from 0 to 1, not dB SPL" + ("; a Pilot 3 plan plays nothing" if night.pilot3 else ""),
    }


def _duration_check(preset: ProtocolPreset, plays: Sequence[_Play]) -> Dict[str, object]:
    streams = {}
    statuses = []
    for stream in STREAMS:
        spec = preset.cue_spec(stream) or preset.cue_spec("any")
        durations = sorted({play.duration_seconds for play in plays if play.stream == stream and play.duration_seconds is not None})
        published = None if spec is None else spec.duration_seconds
        if not durations:
            status = "no_cues"
        elif published is None:
            status = "not_published"
        elif all(abs(value - published) <= _DURATION_TOLERANCE_SECONDS for value in durations):
            status = "as_published"
        else:
            status = "differs"
        statuses.append(status)
        streams[stream] = {
            "status": status,
            "published_seconds": published,
            "observed_seconds": durations,
            "described": None if spec is None else spec.description,
        }
    return {
        "status": _combined(statuses),
        "observed": {stream: row["observed_seconds"] for stream, row in streams.items()},
        "published": {stream: row["published_seconds"] for stream, row in streams.items()},
        "streams": streams,
    }


def _tlr_check(
    preset: ProtocolPreset,
    plays: Sequence[_Play],
    night: _Night,
    pause_times: Sequence[float],
) -> Dict[str, object]:
    tlr = [play for play in plays if play.stream == "tlr"]
    plan = night.tlr_plan
    if not tlr and plan is None:
        return {"status": "not_used", "observed": None, "published": None}
    config = dict((plan or {}).get("config", {}) or {})
    repetitions = _optional_int(config.get("repetitions"))
    interval = _optional_float(config.get("interval_seconds"))
    rows = []
    # One row per TLR block: after a guard pause the scheduler plays the block again from
    # its first cue in the same REM period.
    for period, block, in_block in _by_block(tlr, pause_times):
        logged = [float(play.event["timestamp_seconds"]) for play in in_block]
        played = [play.decision_seconds for play in in_block]
        rows.append(
            {
                "rem_period": period,
                "block": block,
                "cues": len(in_block),
                # False when a guard pause, the end of the period or of the night came
                # before the plan's last cue.
                "complete": None if repetitions is None else len(in_block) == repetitions,
                "planned_intervals_seconds": [_round(b - a) for a, b in zip(logged, logged[1:])],
                "played_intervals_seconds": [_round(b - a) for a, b in zip(played, played[1:])],
                "back_to_back": sum(1 for a, b in zip(played, played[1:]) if b - a <= TIME_TOLERANCE_SECONDS),
            }
        )
    if plan is None:
        status = "no_plan"
    elif not rows:
        status = "no_cues"
    else:
        complete = all(row["complete"] is not False for row in rows)
        on_time = interval is None or all(
            abs(value - interval) <= TIME_TOLERANCE_SECONDS for row in rows for value in row["played_intervals_seconds"]
        )
        status = "as_planned" if complete and on_time else "differs"
    timing = preset.timing_for("tlr")
    return {
        "status": status,
        "observed": {"periods": len({row["rem_period"] for row in rows}), "blocks": len(rows), "cues": len(tlr)},
        "published": None
        if timing is None
        else {"interval_seconds": list(timing.interval_seconds) if timing.interval_seconds else None,
              "max_cues_per_rem_period": timing.max_cues_per_rem_period},
        "plan": {"repetitions": repetitions, "interval_seconds": interval} if plan is not None else None,
        "blocks": rows,
        "note": "planned: the scheduler's logged times; played: the decisions of the epochs that played them",
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _combined(statuses: Iterable[str]) -> str:
    """One status for the streams of a check: any difference first, then a match, then
    why there was nothing to compare."""
    statuses = list(statuses)
    if not statuses:
        return "not_published"
    for status in ("differs", "as_published", "not_published", "not_available", "no_intervals", "no_cues"):
        if status in statuses:
            return status
    return statuses[0]


def _worst(statuses: Sequence[str], order: Sequence[str]) -> str:
    for status in order:
        if status in statuses:
            return status
    return statuses[0]


def _stride(records: Sequence[GateDecisionRecord]) -> float:
    starts = sorted(record.timestamp_seconds for record in records)
    steps = [later - earlier for earlier, later in zip(starts, starts[1:]) if later > earlier]
    return _round(statistics.median(steps)) if steps else _DEFAULT_EPOCH_SECONDS


def _cell(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, Mapping):
        shown = [(key, item) for key, item in value.items() if item not in (None, "", [], {})]
        return "; ".join(f"{key} {_cell(item)}" for key, item in shown) or "n/a"
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(item) for item in value) or "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("|", "/")


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_int(value: object) -> Optional[int]:
    return None if value is None else int(value)


def _read_json(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
