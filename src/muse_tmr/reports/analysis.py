"""Cued-vs-uncued analysis reports.

Schema 3 (#228) reads several retests of a night, one per ``retest_wave`` (the morning
after, ``day7``, ``day10``), and gives the metrics and the cued-minus-uncued deltas of
each in ``wave_metrics``; ``rows``, ``condition_metrics`` and ``effect_summary`` stay
those of the morning retest, or of the earliest wave given without one. Rater codes
(``reports.incorporation_coding``) decide a puzzle's incorporation when the night has
any, the sleeper's own links when it has none; ``incorporation_solved`` crosses
incorporation with solving per wave. A sham night's plays keep the puzzle they stand in
for, but they played the sham sound: they count as ``sham_play_count``, never as plays
of the cued puzzle's sound. Files of schema 1 and 2 still load.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from muse_tmr.protocol import NightPuzzleSession, PuzzleCueAssignment, TmrSchedulerEvent
from muse_tmr.protocol.tmr_scheduler import SHAM_ARM_REASON
from muse_tmr.reports.dream_report import DreamReport
from muse_tmr.reports.incorporation_coding import (
    ANY_CATEGORY,
    CODING_CATEGORIES,
    IncorporationCodingSheet,
    rater_incorporation,
)
from muse_tmr.reports.morning_retest import RETEST_WAVES, MorningRetest

if TYPE_CHECKING:
    from muse_tmr.reports.cue_recall import CueRecallTest

# 2 adds cue_recall, the rows' cue_recalled and metadata.protocol (#231); 3 adds
# wave_metrics, incorporation, incorporation_solved, the rows' wave_results,
# self_linked, rater_codes and sham_play_count, cue_timing's arm and its sham and
# virtual puzzle plays, and the limitations missing_delayed_retest, sham_arm and
# no_cue_arm (#228).
ANALYSIS_SCHEMA_VERSION = 3
ANALYSIS_CONDITIONS = ("cued", "uncued")
# The waves after the morning one; a night without either has missing_delayed_retest.
DELAYED_RETEST_WAVES = tuple(wave for wave in RETEST_WAVES if wave != "morning")
# Where a puzzle's incorporation comes from: rater codes, the sleeper's own links, or nothing.
INCORPORATION_SOURCES = ("raters", "self_report", "none")

RetestInput = Union[MorningRetest, Sequence[MorningRetest]]


@dataclass(frozen=True)
class PuzzleAnalysisRow:
    puzzle_id: str
    cue_condition: str
    solved: bool
    duration_seconds: float
    confidence: float
    cue_id: str = ""
    incorporated: bool = False
    incorporation_confidence: Optional[float] = None
    cue_play_count: int = 0
    first_cue_time_seconds: Optional[float] = None
    last_cue_time_seconds: Optional[float] = None
    cue_time_seconds: Tuple[float, ...] = ()
    # Whether the morning sound check placed this puzzle's sound; None without one.
    cue_recalled: Optional[bool] = None
    # A sham night's plays that stood in for this puzzle's cue: the sham sound played.
    sham_play_count: int = 0
    # Whether the sleeper linked the puzzle in the dream report. ``incorporated`` follows
    # the raters instead when the night has rater codes.
    self_linked: bool = False
    # rater_count and, per category and for any category, the share of raters who coded
    # it; empty without rater codes.
    rater_codes: Mapping[str, object] = field(default_factory=dict)
    # Each retest wave's solved, duration_seconds and confidence.
    wave_results: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "puzzle_id", _required_str(self.puzzle_id, "puzzle_id"))
        if self.cue_condition not in ANALYSIS_CONDITIONS:
            raise ValueError(f"cue_condition must be one of: {', '.join(ANALYSIS_CONDITIONS)}")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        _validate_confidence(self.confidence, "confidence")
        if self.incorporation_confidence is not None:
            _validate_confidence(self.incorporation_confidence, "incorporation_confidence")
        if self.cue_play_count < 0:
            raise ValueError("cue_play_count must be non-negative")
        if self.sham_play_count < 0:
            raise ValueError("sham_play_count must be non-negative")
        cue_times = tuple(float(value) for value in self.cue_time_seconds)
        if any(value < 0 for value in cue_times):
            raise ValueError("cue_time_seconds must be non-negative")
        object.__setattr__(self, "cue_id", str(self.cue_id).strip())
        object.__setattr__(self, "cue_time_seconds", cue_times)
        object.__setattr__(self, "rater_codes", dict(self.rater_codes))
        object.__setattr__(
            self, "wave_results", {str(wave): dict(result) for wave, result in self.wave_results.items()}
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "puzzle_id": self.puzzle_id,
            "cue_id": self.cue_id,
            "cue_condition": self.cue_condition,
            "solved": self.solved,
            "duration_seconds": self.duration_seconds,
            "confidence": self.confidence,
            "incorporated": self.incorporated,
            "incorporation_confidence": self.incorporation_confidence,
            "self_linked": self.self_linked,
            "rater_codes": dict(self.rater_codes),
            "cue_play_count": self.cue_play_count,
            "sham_play_count": self.sham_play_count,
            "first_cue_time_seconds": self.first_cue_time_seconds,
            "last_cue_time_seconds": self.last_cue_time_seconds,
            "cue_time_seconds": list(self.cue_time_seconds),
            "cue_recalled": self.cue_recalled,
            "wave_results": {wave: dict(result) for wave, result in self.wave_results.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PuzzleAnalysisRow":
        incorporated = _boolish(payload.get("incorporated", False))
        return cls(
            puzzle_id=str(payload["puzzle_id"]),
            cue_id=str(payload.get("cue_id", "")),
            cue_condition=str(payload["cue_condition"]),
            solved=_boolish(payload["solved"]),
            duration_seconds=float(payload["duration_seconds"]),
            confidence=float(payload["confidence"]),
            incorporated=incorporated,
            incorporation_confidence=_optional_float(payload.get("incorporation_confidence")),
            cue_play_count=int(payload.get("cue_play_count", 0)),
            first_cue_time_seconds=_optional_float(payload.get("first_cue_time_seconds")),
            last_cue_time_seconds=_optional_float(payload.get("last_cue_time_seconds")),
            cue_time_seconds=tuple(float(value) for value in payload.get("cue_time_seconds", ())),
            cue_recalled=(
                None if payload.get("cue_recalled") is None else _boolish(payload.get("cue_recalled"))
            ),
            sham_play_count=int(payload.get("sham_play_count", 0)),
            # Before schema 3 the sleeper's links were the only source of incorporated.
            self_linked=_boolish(payload.get("self_linked", incorporated)),
            rater_codes=dict(payload.get("rater_codes", {}) or {}),
            wave_results=dict(payload.get("wave_results", {}) or {}),
        )


@dataclass(frozen=True)
class ConditionMetrics:
    cue_condition: str
    puzzle_count: int
    solved_count: int
    solve_rate: Optional[float]
    incorporated_count: int
    incorporation_rate: Optional[float]
    mean_duration_seconds: Optional[float]
    mean_confidence: Optional[float]
    cue_play_count: int
    puzzles_with_cues: int
    first_cue_time_seconds: Optional[float] = None
    last_cue_time_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.cue_condition not in ANALYSIS_CONDITIONS:
            raise ValueError(f"cue_condition must be one of: {', '.join(ANALYSIS_CONDITIONS)}")
        if self.puzzle_count < 0:
            raise ValueError("puzzle_count must be non-negative")
        if not 0 <= self.solved_count <= self.puzzle_count:
            raise ValueError("solved_count must fit inside puzzle_count")
        if not 0 <= self.incorporated_count <= self.puzzle_count:
            raise ValueError("incorporated_count must fit inside puzzle_count")
        if self.cue_play_count < 0 or self.puzzles_with_cues < 0:
            raise ValueError("cue counts must be non-negative")

    def to_dict(self) -> Dict[str, object]:
        return {
            "cue_condition": self.cue_condition,
            "puzzle_count": self.puzzle_count,
            "solved_count": self.solved_count,
            "solve_rate": self.solve_rate,
            "incorporated_count": self.incorporated_count,
            "incorporation_rate": self.incorporation_rate,
            "mean_duration_seconds": self.mean_duration_seconds,
            "mean_confidence": self.mean_confidence,
            "cue_play_count": self.cue_play_count,
            "puzzles_with_cues": self.puzzles_with_cues,
            "first_cue_time_seconds": self.first_cue_time_seconds,
            "last_cue_time_seconds": self.last_cue_time_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConditionMetrics":
        return cls(
            cue_condition=str(payload["cue_condition"]),
            puzzle_count=int(payload["puzzle_count"]),
            solved_count=int(payload["solved_count"]),
            solve_rate=_optional_float(payload.get("solve_rate")),
            incorporated_count=int(payload["incorporated_count"]),
            incorporation_rate=_optional_float(payload.get("incorporation_rate")),
            mean_duration_seconds=_optional_float(payload.get("mean_duration_seconds")),
            mean_confidence=_optional_float(payload.get("mean_confidence")),
            cue_play_count=int(payload.get("cue_play_count", 0)),
            puzzles_with_cues=int(payload.get("puzzles_with_cues", 0)),
            first_cue_time_seconds=_optional_float(payload.get("first_cue_time_seconds")),
            last_cue_time_seconds=_optional_float(payload.get("last_cue_time_seconds")),
        )


@dataclass(frozen=True)
class AnalysisLimitation:
    code: str
    severity: str
    message: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _required_str(self.code, "code"))
        object.__setattr__(self, "severity", _required_str(self.severity, "severity"))
        object.__setattr__(self, "message", _required_str(self.message, "message"))

    def to_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AnalysisLimitation":
        return cls(
            code=str(payload["code"]),
            severity=str(payload["severity"]),
            message=str(payload["message"]),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class CuedUncuedAnalysisReport:
    session_id: str
    rows: Tuple[PuzzleAnalysisRow, ...]
    condition_metrics: Tuple[ConditionMetrics, ...]
    effect_summary: Mapping[str, object]
    cue_timing: Mapping[str, object]
    limitations: Tuple[AnalysisLimitation, ...]
    analysis_id: str = ""
    generated_at_utc: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    # The morning sound check by condition, or {"available": False}.
    cue_recall: Mapping[str, object] = field(default_factory=dict)
    # Per retest wave: its retest_id, reported_at_utc, condition_metrics and effect_summary.
    wave_metrics: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    # Where incorporation came from (source, rater_ids, split and uncoded puzzles).
    incorporation: Mapping[str, object] = field(default_factory=dict)
    # Incorporated or not against solved or not, per wave, or {"available": False}.
    incorporation_solved: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _required_str(self.session_id, "session_id"))
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "condition_metrics", tuple(self.condition_metrics))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if not self.analysis_id:
            object.__setattr__(self, "analysis_id", f"{self.session_id}-cued-uncued-analysis")
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())
        puzzle_ids = [row.puzzle_id for row in self.rows]
        if len(set(puzzle_ids)) != len(puzzle_ids):
            raise ValueError("analysis rows cannot contain duplicate puzzle IDs")

    @property
    def limitation_codes(self) -> Tuple[str, ...]:
        return tuple(item.code for item in self.limitations)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "session_id": self.session_id,
            "generated_at_utc": self.generated_at_utc,
            "row_count": len(self.rows),
            "rows": [row.to_dict() for row in self.rows],
            "condition_metrics": {
                metrics.cue_condition: metrics.to_dict()
                for metrics in self.condition_metrics
            },
            "effect_summary": dict(self.effect_summary),
            "wave_metrics": {wave: dict(block) for wave, block in self.wave_metrics.items()},
            "cue_timing": dict(self.cue_timing),
            "cue_recall": dict(self.cue_recall),
            "incorporation": dict(self.incorporation),
            "incorporation_solved": dict(self.incorporation_solved),
            "limitations": [limitation.to_dict() for limitation in self.limitations],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        metrics = {item.cue_condition: item for item in self.condition_metrics}
        cued = metrics.get("cued")
        uncued = metrics.get("uncued")
        solve_delta = self.effect_summary.get("solve_rate_difference_cued_minus_uncued")
        incorporation_delta = self.effect_summary.get(
            "incorporation_rate_difference_cued_minus_uncued"
        )
        lines = [
            f"# Cued vs Uncued Analysis: {self.session_id}",
            "",
            "## Summary",
            f"- Cued solve rate: {_format_rate(cued.solve_rate if cued else None)}",
            f"- Uncued solve rate: {_format_rate(uncued.solve_rate if uncued else None)}",
            f"- Solve-rate delta: {_format_rate(solve_delta)}",
            f"- Incorporation-rate delta: {_format_rate(incorporation_delta)}",
            f"- Incorporation from: {_incorporation_line(self.incorporation)}",
            f"- Puzzle cue plays: {self.cue_timing.get('puzzle_play_event_count', 0)}",
        ]
        if self.cue_timing.get("sham_puzzle_play_count"):
            lines.append(
                f"- Sham plays in place of puzzle cues: {self.cue_timing.get('sham_puzzle_play_count')} "
                "(the sham sound, not counted as cue plays)"
            )
        lines.append(f"- Sound check (manipulation check): {_recall_line(self.cue_recall)}")
        if len(self.wave_metrics) > 1:
            lines += ["", "## Retest Waves", "", "| Wave | Cued solve rate | Uncued solve rate | Delta |",
                      "| --- | ---: | ---: | ---: |"]
            for wave, block in self.wave_metrics.items():
                by_condition = dict(block.get("condition_metrics", {}) or {})
                effect = dict(block.get("effect_summary", {}) or {})
                lines.append(
                    f"| {wave} | {_format_rate((by_condition.get('cued') or {}).get('solve_rate'))} | "
                    f"{_format_rate((by_condition.get('uncued') or {}).get('solve_rate'))} | "
                    f"{_format_rate(effect.get('solve_rate_difference_cued_minus_uncued'))} |"
                )
        if self.incorporation_solved.get("available"):
            lines += ["", "## Incorporation And Solving", "",
                      "| Wave | Incorporated: solved | Incorporated: unsolved | Not: solved | Not: unsolved |",
                      "| --- | ---: | ---: | ---: | ---: |"]
            for wave, block in dict(self.incorporation_solved.get("waves", {}) or {}).items():
                yes = dict(block.get("incorporated", {}) or {})
                no = dict(block.get("not_incorporated", {}) or {})
                lines.append(
                    f"| {wave} | {yes.get('solved', 0)} | {yes.get('unsolved', 0)} | "
                    f"{no.get('solved', 0)} | {no.get('unsolved', 0)} |"
                )
        lines += ["", "## Limitations"]
        if self.limitations:
            lines.extend(f"- `{item.code}`: {item.message}" for item in self.limitations)
        else:
            lines.append("- No limitations recorded.")
        lines.extend(["", "## Puzzle Rows"])
        lines.append("| Puzzle | Condition | Solved | Incorporated | Cue plays | First cue |")
        lines.append("| --- | --- | --- | --- | ---: | ---: |")
        for row in self.rows:
            lines.append(
                "| "
                f"{row.puzzle_id} | {row.cue_condition} | {_yes_no(row.solved)} | "
                f"{_yes_no(row.incorporated)} | {row.cue_play_count} | "
                f"{_format_seconds(row.first_cue_time_seconds)} |"
            )
        return "\n".join(lines) + "\n"

    def save(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    def save_markdown(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CuedUncuedAnalysisReport":
        raw_metrics = payload.get("condition_metrics", {})
        if isinstance(raw_metrics, Mapping):
            metrics_payload = raw_metrics.values()
        else:
            metrics_payload = raw_metrics
        return cls(
            schema_version=int(payload.get("schema_version", ANALYSIS_SCHEMA_VERSION)),
            analysis_id=str(payload.get("analysis_id", "")),
            session_id=str(payload["session_id"]),
            generated_at_utc=str(payload.get("generated_at_utc", "")),
            rows=tuple(PuzzleAnalysisRow.from_dict(item) for item in payload.get("rows", ())),
            condition_metrics=tuple(ConditionMetrics.from_dict(item) for item in metrics_payload),
            effect_summary=dict(payload.get("effect_summary", {}) or {}),
            cue_timing=dict(payload.get("cue_timing", {}) or {}),
            limitations=tuple(
                AnalysisLimitation.from_dict(item)
                for item in payload.get("limitations", ())
            ),
            metadata=dict(payload.get("metadata", {}) or {}),
            cue_recall=dict(payload.get("cue_recall", {}) or {}),
            wave_metrics=dict(payload.get("wave_metrics", {}) or {}),
            incorporation=dict(payload.get("incorporation", {}) or {}),
            incorporation_solved=dict(payload.get("incorporation_solved", {}) or {}),
        )

    @classmethod
    def load(cls, input_path: Path) -> "CuedUncuedAnalysisReport":
        return cls.from_dict(json.loads(input_path.expanduser().read_text(encoding="utf-8")))


def build_cued_uncued_analysis(
    session: NightPuzzleSession,
    assignment: PuzzleCueAssignment,
    retest: RetestInput,
    *,
    dream_report: Optional[DreamReport] = None,
    scheduler_events: Iterable[TmrSchedulerEvent] = (),
    cue_recall: Optional["CueRecallTest"] = None,
    incorporation_codings: Iterable[IncorporationCodingSheet] = (),
    protocol: Optional[Mapping[str, object]] = None,
    protocol_source: Optional[str] = None,
    analysis_id: str = "",
    generated_at_utc: str = "",
    min_group_size: int = 5,
) -> CuedUncuedAnalysisReport:
    """The descriptive cued-vs-uncued report of one night.

    ``retest`` is one ``MorningRetest`` or several, at most one per ``retest_wave``; the
    rows and the top-level metrics are those of the morning wave, or of the earliest wave
    given without one, and ``wave_metrics`` has every wave given. ``incorporation_codings``
    are rater coding sheets of this night, preferred over the dream report's own links.
    ``cue_recall`` is the morning sound check (``reports.cue_recall``); without it, or
    when it was finished after the retest, the limitations say so. ``protocol`` is the
    night's protocol preset block, copied into ``metadata.protocol`` with where it came
    from (``protocol_source``).
    """
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")
    retests = retests_by_wave(retest)
    assignment.validate_against_session(session)
    for item in retests.values():
        item.validate_against_session(session)
    if dream_report is not None:
        dream_report.validate_against_session(session)
    if cue_recall is not None:
        cue_recall.validate_against_session(session)
    primary_wave = primary_retest_wave(retests)
    primary = retests[primary_wave]

    incorporations, incorporation_summary = incorporation_by_puzzle(
        session, dream_report=dream_report, incorporation_codings=incorporation_codings
    )
    events = tuple(scheduler_events)
    arm = _night_arm(events)
    cue_times_by_puzzle, sham_times_by_puzzle, unknown_event_puzzle_ids = _cue_times_by_puzzle(session, events)
    recalled = (
        {trial.puzzle_id: trial.correct for trial in cue_recall.trials} if cue_recall is not None else {}
    )
    results_by_wave = {
        wave: {result.puzzle_id: result for result in item.results} for wave, item in retests.items()
    }

    rows = tuple(
        _build_row(
            puzzle_id,
            assignment=assignment,
            results_by_wave=results_by_wave,
            primary_wave=primary_wave,
            incorporation=incorporations[puzzle_id],
            cue_times=cue_times_by_puzzle.get(puzzle_id, ()),
            sham_times=sham_times_by_puzzle.get(puzzle_id, ()),
            cue_recalled=recalled.get(puzzle_id),
        )
        for puzzle_id in session.puzzle_ids
    )
    metrics = tuple(_condition_metrics(condition, rows) for condition in ANALYSIS_CONDITIONS)
    wave_metrics = {wave: _wave_block(wave, item, rows) for wave, item in retests.items()}
    recall_summary = _cue_recall_summary(cue_recall, assignment, primary)
    limitations = _limitations(
        rows,
        metrics,
        dream_report=dream_report,
        scheduler_events=events,
        unknown_event_puzzle_ids=unknown_event_puzzle_ids,
        min_group_size=min_group_size,
        recall_summary=recall_summary,
        arm=arm,
        waves=tuple(retests),
    )
    metadata: Dict[str, object] = {
        "assignment_seed": assignment.seed,
        "dream_report_available": dream_report is not None,
        "scheduler_events_available": bool(events),
        "cue_recall_available": cue_recall is not None,
        "incorporation_codings_available": incorporation_summary["source"] == "raters",
        "retest_waves": list(retests),
        "primary_retest_wave": primary_wave,
        "min_group_size": min_group_size,
    }
    if protocol:
        metadata["protocol"] = dict(protocol)
        metadata["protocol_source"] = protocol_source
    report = CuedUncuedAnalysisReport(
        analysis_id=analysis_id,
        session_id=session.session_id,
        generated_at_utc=generated_at_utc,
        rows=rows,
        condition_metrics=metrics,
        effect_summary=_effect_summary(metrics),
        cue_timing=_cue_timing_summary(rows, events, arm),
        cue_recall=recall_summary,
        wave_metrics=wave_metrics,
        incorporation=incorporation_summary,
        incorporation_solved=_incorporation_solved(rows, tuple(retests), incorporation_summary["source"]),
        limitations=limitations,
        metadata=metadata,
    )
    return report


def load_cued_uncued_analysis(input_path: Path) -> CuedUncuedAnalysisReport:
    return CuedUncuedAnalysisReport.load(input_path)


def retests_by_wave(retest: RetestInput) -> Dict[str, MorningRetest]:
    """The retests of a night by wave, in ``RETEST_WAVES`` order; two of one wave raise."""
    items = (retest,) if isinstance(retest, MorningRetest) else tuple(retest)
    if not items:
        raise ValueError("the analysis needs at least one retest")
    by_wave: Dict[str, MorningRetest] = {}
    for item in items:
        if item.retest_wave in by_wave:
            raise ValueError(f"two retests of the {item.retest_wave} wave; give one per wave")
        by_wave[item.retest_wave] = item
    sessions = sorted({item.session_id for item in items})
    if len(sessions) > 1:
        raise ValueError(f"the retests belong to different sessions: {', '.join(sessions)}")
    return {wave: by_wave[wave] for wave in RETEST_WAVES if wave in by_wave}


def primary_retest_wave(retests: Mapping[str, MorningRetest]) -> str:
    """The wave the rows and top-level metrics show: morning, else the earliest given."""
    return "morning" if "morning" in retests else next(iter(retests))


def incorporation_by_puzzle(
    session: NightPuzzleSession,
    *,
    dream_report: Optional[DreamReport] = None,
    incorporation_codings: Iterable[IncorporationCodingSheet] = (),
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """Each session puzzle's incorporation, and where it came from.

    With rater coding sheets of this night, a puzzle is incorporated when more than half
    of the raters who coded it put it in some category (``rater_incorporation()``);
    without them, when the sleeper linked it in the dream report; with neither, the
    source is ``none`` and no puzzle is incorporated. A sheet of another night raises.
    """
    sheets = tuple(sheet.validate_against_session(session) for sheet in incorporation_codings)
    links = (
        {link.puzzle_id: link for link in dream_report.puzzle_incorporations if link.incorporated}
        if dream_report is not None
        else {}
    )
    raters = rater_incorporation(sheets, session) if sheets else {}
    source = "raters" if sheets else ("self_report" if dream_report is not None else "none")
    per_puzzle: Dict[str, Dict[str, object]] = {}
    for puzzle_id in session.puzzle_ids:
        link = links.get(puzzle_id)
        codes = raters.get(puzzle_id, {})
        rater_codes = (
            {key: codes[key] for key in ("rater_count",) + CODING_CATEGORIES + (ANY_CATEGORY,)} if codes else {}
        )
        per_puzzle[puzzle_id] = {
            "incorporated": bool(codes.get("incorporated")) if source == "raters" else link is not None,
            "self_linked": link is not None,
            "confidence": link.confidence if link is not None else None,
            "rater_codes": rater_codes,
        }
    summary: Dict[str, object] = {
        "source": source,
        "rater_ids": sorted(sheet.rater_id for sheet in sheets),
        "rater_count": len(sheets),
        "incorporated_count": sum(1 for item in per_puzzle.values() if item["incorporated"]),
        "self_linked_count": sum(1 for item in per_puzzle.values() if item["self_linked"]),
        "split_puzzle_ids": [puzzle_id for puzzle_id, codes in raters.items() if codes.get("split")],
        "uncoded_puzzle_ids": [puzzle_id for puzzle_id, codes in raters.items() if not codes.get("rater_count")],
    }
    return per_puzzle, summary


def _build_row(
    puzzle_id: str,
    *,
    assignment: PuzzleCueAssignment,
    results_by_wave: Mapping[str, Mapping[str, object]],
    primary_wave: str,
    incorporation: Mapping[str, object],
    cue_times: Tuple[float, ...],
    sham_times: Tuple[float, ...],
    cue_recalled: Optional[bool] = None,
) -> PuzzleAnalysisRow:
    retest_result = results_by_wave[primary_wave][puzzle_id]
    wave_results = {
        wave: {
            "solved": result.solved,
            "duration_seconds": result.duration_seconds,
            "confidence": result.confidence,
        }
        for wave, results in results_by_wave.items()
        for result in (results[puzzle_id],)
    }
    condition = "cued" if assignment.is_cued(puzzle_id) else "uncued"
    return PuzzleAnalysisRow(
        puzzle_id=puzzle_id,
        cue_id=retest_result.cue_id,
        cue_condition=condition,
        solved=retest_result.solved,
        duration_seconds=retest_result.duration_seconds,
        confidence=retest_result.confidence,
        incorporated=bool(incorporation["incorporated"]),
        incorporation_confidence=incorporation["confidence"],  # type: ignore[arg-type]
        cue_play_count=len(cue_times),
        first_cue_time_seconds=cue_times[0] if cue_times else None,
        last_cue_time_seconds=cue_times[-1] if cue_times else None,
        cue_time_seconds=cue_times,
        cue_recalled=cue_recalled,
        sham_play_count=len(sham_times),
        self_linked=bool(incorporation["self_linked"]),
        rater_codes=dict(incorporation["rater_codes"]),  # type: ignore[arg-type]
        wave_results=wave_results,
    )


def _wave_block(wave: str, retest: MorningRetest, rows: Tuple[PuzzleAnalysisRow, ...]) -> Dict[str, object]:
    """One wave's metrics and cued-minus-uncued deltas, from the rows with that wave's results."""
    wave_rows = tuple(
        replace(
            row,
            solved=bool(row.wave_results[wave]["solved"]),
            duration_seconds=float(row.wave_results[wave]["duration_seconds"]),  # type: ignore[arg-type]
            confidence=float(row.wave_results[wave]["confidence"]),  # type: ignore[arg-type]
        )
        for row in rows
    )
    metrics = tuple(_condition_metrics(condition, wave_rows) for condition in ANALYSIS_CONDITIONS)
    return {
        "retest_id": retest.retest_id,
        "reported_at_utc": retest.reported_at_utc,
        "condition_metrics": {item.cue_condition: item.to_dict() for item in metrics},
        "effect_summary": _effect_summary(metrics),
    }


def _incorporation_solved(
    rows: Tuple[PuzzleAnalysisRow, ...],
    waves: Tuple[str, ...],
    source: str,
) -> Dict[str, object]:
    """Incorporated or not against solved or not, per wave, overall and by condition."""
    if source == "none":
        return {"available": False, "source": source}

    def cell(subset: Sequence[PuzzleAnalysisRow], wave: str) -> Dict[str, object]:
        solved = sum(1 for row in subset if row.wave_results[wave]["solved"])
        return {
            "puzzles": len(subset),
            "solved": solved,
            "unsolved": len(subset) - solved,
            "solve_rate": _rate(solved, len(subset)),
        }

    by_wave: Dict[str, object] = {}
    for wave in waves:
        block: Dict[str, object] = {
            "incorporated": cell([row for row in rows if row.incorporated], wave),
            "not_incorporated": cell([row for row in rows if not row.incorporated], wave),
        }
        block["by_condition"] = {
            condition: {
                "incorporated": cell(
                    [row for row in rows if row.cue_condition == condition and row.incorporated], wave
                ),
                "not_incorporated": cell(
                    [row for row in rows if row.cue_condition == condition and not row.incorporated], wave
                ),
            }
            for condition in ANALYSIS_CONDITIONS
        }
        by_wave[wave] = block
    return {"available": True, "source": source, "waves": by_wave, "interpretation": "descriptive_only"}


def _cue_recall_summary(
    cue_recall: Optional["CueRecallTest"],
    assignment: PuzzleCueAssignment,
    retest: MorningRetest,
) -> Dict[str, object]:
    """The sound check by condition: correct trials over trials, a sound not placed counting
    as wrong; and whether it was finished before the retest was saved."""
    if cue_recall is None:
        return {"available": False}
    before = _before(cue_recall.completed_at_utc, retest.reported_at_utc)
    by_condition = {}
    for condition in ANALYSIS_CONDITIONS:
        trials = [
            trial
            for trial in cue_recall.trials
            if (assignment.is_cued(trial.puzzle_id) if condition == "cued" else assignment.is_uncued(trial.puzzle_id))
        ]
        correct = sum(1 for trial in trials if trial.correct)
        by_condition[condition] = {
            "trial_count": len(trials),
            "correct_count": correct,
            "recall_rate": _rate(correct, len(trials)),
        }
    return {
        "available": True,
        "trial_count": cue_recall.trial_count,
        "answered_count": cue_recall.answered_count,
        "correct_count": cue_recall.correct_count,
        "recall_rate": cue_recall.recall_rate,
        "by_condition": by_condition,
        "recall_rate_difference_cued_minus_uncued": _difference(
            by_condition["cued"]["recall_rate"], by_condition["uncued"]["recall_rate"]
        ),
        "completed_at_utc": cue_recall.completed_at_utc,
        "retest_reported_at_utc": retest.reported_at_utc,
        "before_retest": before,
        "interpretation": "manipulation_check",
    }


def _before(first: str, second: str) -> Optional[bool]:
    try:
        return dt.datetime.fromisoformat(first) <= dt.datetime.fromisoformat(second)
    except (TypeError, ValueError):
        return None


def _condition_metrics(condition: str, rows: Tuple[PuzzleAnalysisRow, ...]) -> ConditionMetrics:
    condition_rows = tuple(row for row in rows if row.cue_condition == condition)
    cue_times = tuple(
        timestamp
        for row in condition_rows
        for timestamp in row.cue_time_seconds
    )
    return ConditionMetrics(
        cue_condition=condition,
        puzzle_count=len(condition_rows),
        solved_count=sum(1 for row in condition_rows if row.solved),
        solve_rate=_rate(sum(1 for row in condition_rows if row.solved), len(condition_rows)),
        incorporated_count=sum(1 for row in condition_rows if row.incorporated),
        incorporation_rate=_rate(
            sum(1 for row in condition_rows if row.incorporated),
            len(condition_rows),
        ),
        mean_duration_seconds=_mean(row.duration_seconds for row in condition_rows),
        mean_confidence=_mean(row.confidence for row in condition_rows),
        cue_play_count=sum(row.cue_play_count for row in condition_rows),
        puzzles_with_cues=sum(1 for row in condition_rows if row.cue_play_count > 0),
        first_cue_time_seconds=min(cue_times) if cue_times else None,
        last_cue_time_seconds=max(cue_times) if cue_times else None,
    )


def _effect_summary(metrics: Tuple[ConditionMetrics, ...]) -> Dict[str, object]:
    by_condition = {item.cue_condition: item for item in metrics}
    cued = by_condition["cued"]
    uncued = by_condition["uncued"]
    return {
        "solve_rate_difference_cued_minus_uncued": _difference(cued.solve_rate, uncued.solve_rate),
        "incorporation_rate_difference_cued_minus_uncued": _difference(
            cued.incorporation_rate,
            uncued.incorporation_rate,
        ),
        "mean_duration_difference_seconds_cued_minus_uncued": _difference(
            cued.mean_duration_seconds,
            uncued.mean_duration_seconds,
        ),
        "mean_confidence_difference_cued_minus_uncued": _difference(
            cued.mean_confidence,
            uncued.mean_confidence,
        ),
        "interpretation": "descriptive_only",
    }


def _cue_timing_summary(
    rows: Tuple[PuzzleAnalysisRow, ...],
    events: Tuple[TmrSchedulerEvent, ...],
    arm: Optional[str] = None,
) -> Dict[str, object]:
    puzzle_play_rows = tuple(row for row in rows if row.cue_play_count > 0)
    all_times = tuple(timestamp for row in rows for timestamp in row.cue_time_seconds)
    return {
        "scheduler_event_count": len(events),
        "puzzle_play_event_count": sum(row.cue_play_count for row in rows),
        "puzzles_with_cues": len(puzzle_play_rows),
        "first_puzzle_cue_seconds": min(all_times) if all_times else None,
        "last_puzzle_cue_seconds": max(all_times) if all_times else None,
        "cued_puzzle_play_count": sum(
            row.cue_play_count for row in rows if row.cue_condition == "cued"
        ),
        # Any play that names an uncued puzzle, the sham sound's included.
        "uncued_puzzle_play_count": sum(
            row.cue_play_count + row.sham_play_count for row in rows if row.cue_condition == "uncued"
        ),
        # The night's arm as its events have it; None without events.
        "arm": arm,
        "sham_puzzle_play_count": sum(row.sham_play_count for row in rows),
        "virtual_puzzle_play_count": sum(
            1 for event in events if event.virtual_play and event.protocol == "puzzle"
        ),
    }


def _limitations(
    rows: Tuple[PuzzleAnalysisRow, ...],
    metrics: Tuple[ConditionMetrics, ...],
    *,
    dream_report: Optional[DreamReport],
    scheduler_events: Tuple[TmrSchedulerEvent, ...],
    unknown_event_puzzle_ids: Tuple[str, ...],
    min_group_size: int,
    recall_summary: Optional[Mapping[str, object]] = None,
    arm: Optional[str] = None,
    waves: Tuple[str, ...] = ("morning",),
) -> Tuple[AnalysisLimitation, ...]:
    limitations = [
        AnalysisLimitation(
            code="descriptive_only",
            severity="info",
            message=(
                "Report is descriptive and does not estimate statistical significance "
                "or causal effect."
            ),
        )
    ]
    small_groups = {
        item.cue_condition: item.puzzle_count
        for item in metrics
        if item.puzzle_count < min_group_size
    }
    if small_groups:
        limitations.append(
            AnalysisLimitation(
                code="small_n",
                severity="warning",
                message=(
                    "At least one condition has fewer puzzles than the configured "
                    "minimum group size; treat deltas as exploratory."
                ),
                metadata={"group_sizes": small_groups, "min_group_size": min_group_size},
            )
        )
    if dream_report is None:
        limitations.append(
            AnalysisLimitation(
                code="missing_dream_report",
                severity="warning",
                message="No dream report was supplied, so incorporation effects are incomplete.",
            )
        )
    if not scheduler_events:
        limitations.append(
            AnalysisLimitation(
                code="missing_scheduler_events",
                severity="warning",
                message="No scheduler event log was supplied, so cue timing cannot be audited.",
            )
        )
    missing_cued_timing = tuple(
        row.puzzle_id
        for row in rows
        if row.cue_condition == "cued" and row.cue_play_count == 0
    )
    # A sham or no-cue night plays no cued puzzle's sound by design.
    if scheduler_events and missing_cued_timing and arm not in ("sham", "no_cue"):
        limitations.append(
            AnalysisLimitation(
                code="missing_cued_timing",
                severity="warning",
                message="Some cued puzzles have no puzzle play event in the scheduler log.",
                metadata={"puzzle_ids": missing_cued_timing},
            )
        )
    uncued_plays = tuple(
        row.puzzle_id
        for row in rows
        if row.cue_condition == "uncued" and (row.cue_play_count > 0 or row.sham_play_count > 0)
    )
    if uncued_plays:
        limitations.append(
            AnalysisLimitation(
                code="uncued_cue_play_observed",
                severity="error",
                message="Scheduler log includes puzzle play events for uncued controls.",
                metadata={"puzzle_ids": uncued_plays},
            )
        )
    if unknown_event_puzzle_ids:
        limitations.append(
            AnalysisLimitation(
                code="unknown_scheduler_puzzle_ids",
                severity="warning",
                message="Scheduler log includes puzzle events outside this session.",
                metadata={"puzzle_ids": unknown_event_puzzle_ids},
            )
        )
    if arm == "sham":
        limitations.append(
            AnalysisLimitation(
                code="sham_arm",
                severity="info",
                message=(
                    "A sham night: every play was the untrained sham sound at a cued puzzle's moment. The plays "
                    "count as sham_play_count, not as plays of the cued puzzles' sounds."
                ),
                metadata={"sham_puzzle_play_count": sum(row.sham_play_count for row in rows)},
            )
        )
    elif arm == "no_cue":
        limitations.append(
            AnalysisLimitation(
                code="no_cue_arm",
                severity="info",
                message=(
                    "A no-cue night: nothing played. Its virtual_play skips mark where a cue night would have "
                    "played."
                ),
            )
        )
    missing_waves = tuple(wave for wave in DELAYED_RETEST_WAVES if wave not in waves)
    if missing_waves:
        limitations.append(
            AnalysisLimitation(
                code="missing_delayed_retest",
                severity="warning",
                message=(
                    "No retest a week or ten days later was supplied for some waves; cueing effects have shown "
                    "up only at a delayed retest in some studies. A missing wave is not a zero."
                ),
                metadata={"waves_given": list(waves), "missing_waves": list(missing_waves)},
            )
        )
    recall_summary = recall_summary or {}
    if not recall_summary.get("available"):
        limitations.append(
            AnalysisLimitation(
                code="cue_recall_missing",
                severity="warning",
                message=(
                    "No morning sound check was supplied, so it is unknown whether each sound still "
                    "pointed to its puzzle."
                ),
            )
        )
    elif recall_summary.get("before_retest") is False:
        limitations.append(
            AnalysisLimitation(
                code="cue_recall_after_retest",
                severity="warning",
                message=(
                    "The sound check was finished after the retest was saved, so the retest may have "
                    "reminded the sleeper of the pairings."
                ),
                metadata={
                    "completed_at_utc": recall_summary.get("completed_at_utc"),
                    "retest_reported_at_utc": recall_summary.get("retest_reported_at_utc"),
                },
            )
        )
    return tuple(limitations)


def _night_arm(events: Tuple[TmrSchedulerEvent, ...]) -> Optional[str]:
    """The arm a night's scheduler events carry; None without events. One night's log
    has one arm, so a log with two raises."""
    arms = sorted({event.arm for event in events})
    if not arms:
        return None
    if len(arms) > 1:
        raise ValueError(f"the scheduler events hold more than one arm ({', '.join(arms)}); give one night's log")
    return arms[0]


def _is_sham_play(event: TmrSchedulerEvent) -> bool:
    return event.arm == "sham" or SHAM_ARM_REASON in event.reason_codes


def _cue_times_by_puzzle(
    session: NightPuzzleSession,
    events: Tuple[TmrSchedulerEvent, ...],
) -> Tuple[Dict[str, Tuple[float, ...]], Dict[str, Tuple[float, ...]], Tuple[str, ...]]:
    """The times of each session puzzle's plays, its sham plays apart, and the puzzle IDs
    of plays outside the session."""
    session_ids = set(session.puzzle_ids)
    times: Dict[str, List[float]] = {puzzle_id: [] for puzzle_id in session.puzzle_ids}
    sham: Dict[str, List[float]] = {puzzle_id: [] for puzzle_id in session.puzzle_ids}
    unknown = []
    for event in events:
        if event.event_type != "play" or event.protocol != "puzzle" or not event.puzzle_id:
            continue
        if event.puzzle_id not in session_ids:
            unknown.append(event.puzzle_id)
            continue
        (sham if _is_sham_play(event) else times)[event.puzzle_id].append(event.timestamp_seconds)
    return (
        {puzzle_id: tuple(sorted(values)) for puzzle_id, values in times.items()},
        {puzzle_id: tuple(sorted(values)) for puzzle_id, values in sham.items()},
        tuple(sorted(set(unknown))),
    )


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _mean(values: Iterable[float]) -> Optional[float]:
    items = tuple(values)
    if not items:
        return None
    return sum(items) / len(items)


def _difference(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def _required_str(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _validate_confidence(value: float, name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"expected yes/no boolean value, got {value!r}")


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _format_rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1%}"


def _recall_line(summary: Mapping[str, object]) -> str:
    if not summary.get("available"):
        return "not supplied"
    by_condition = summary.get("by_condition", {}) or {}
    cued = (by_condition.get("cued") or {}).get("recall_rate")
    uncued = (by_condition.get("uncued") or {}).get("recall_rate")
    return f"{_format_rate(summary.get('recall_rate'))} placed (cued {_format_rate(cued)}, uncued {_format_rate(uncued)})"


def _incorporation_line(summary: Mapping[str, object]) -> str:
    source = summary.get("source")
    if source == "raters":
        count = int(summary.get("rater_count", 0) or 0)
        return f"rater codes ({count} rater{'s' if count != 1 else ''})"
    if source == "self_report":
        return "the dream report's own links"
    return "no dream report and no rater codes"


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
