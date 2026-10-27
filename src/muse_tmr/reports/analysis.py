"""Cued-vs-uncued analysis reports."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple

from muse_tmr.protocol import NightPuzzleSession, PuzzleCueAssignment, TmrSchedulerEvent
from muse_tmr.reports.dream_report import DreamReport
from muse_tmr.reports.morning_retest import MorningRetest

if TYPE_CHECKING:
    from muse_tmr.reports.cue_recall import CueRecallTest

# 2 adds cue_recall, the rows' cue_recalled and metadata.protocol (#231).
ANALYSIS_SCHEMA_VERSION = 2
ANALYSIS_CONDITIONS = ("cued", "uncued")


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
        cue_times = tuple(float(value) for value in self.cue_time_seconds)
        if any(value < 0 for value in cue_times):
            raise ValueError("cue_time_seconds must be non-negative")
        object.__setattr__(self, "cue_id", str(self.cue_id).strip())
        object.__setattr__(self, "cue_time_seconds", cue_times)

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
            "cue_play_count": self.cue_play_count,
            "first_cue_time_seconds": self.first_cue_time_seconds,
            "last_cue_time_seconds": self.last_cue_time_seconds,
            "cue_time_seconds": list(self.cue_time_seconds),
            "cue_recalled": self.cue_recalled,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PuzzleAnalysisRow":
        return cls(
            puzzle_id=str(payload["puzzle_id"]),
            cue_id=str(payload.get("cue_id", "")),
            cue_condition=str(payload["cue_condition"]),
            solved=_boolish(payload["solved"]),
            duration_seconds=float(payload["duration_seconds"]),
            confidence=float(payload["confidence"]),
            incorporated=_boolish(payload.get("incorporated", False)),
            incorporation_confidence=_optional_float(payload.get("incorporation_confidence")),
            cue_play_count=int(payload.get("cue_play_count", 0)),
            first_cue_time_seconds=_optional_float(payload.get("first_cue_time_seconds")),
            last_cue_time_seconds=_optional_float(payload.get("last_cue_time_seconds")),
            cue_time_seconds=tuple(float(value) for value in payload.get("cue_time_seconds", ())),
            cue_recalled=(
                None if payload.get("cue_recalled") is None else _boolish(payload.get("cue_recalled"))
            ),
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
            "cue_timing": dict(self.cue_timing),
            "cue_recall": dict(self.cue_recall),
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
            f"- Puzzle cue plays: {self.cue_timing.get('puzzle_play_event_count', 0)}",
            f"- Sound check (manipulation check): {_recall_line(self.cue_recall)}",
            "",
            "## Limitations",
        ]
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
        )

    @classmethod
    def load(cls, input_path: Path) -> "CuedUncuedAnalysisReport":
        return cls.from_dict(json.loads(input_path.expanduser().read_text(encoding="utf-8")))


def build_cued_uncued_analysis(
    session: NightPuzzleSession,
    assignment: PuzzleCueAssignment,
    retest: MorningRetest,
    *,
    dream_report: Optional[DreamReport] = None,
    scheduler_events: Iterable[TmrSchedulerEvent] = (),
    cue_recall: Optional["CueRecallTest"] = None,
    protocol: Optional[Mapping[str, object]] = None,
    protocol_source: Optional[str] = None,
    analysis_id: str = "",
    generated_at_utc: str = "",
    min_group_size: int = 5,
) -> CuedUncuedAnalysisReport:
    """The descriptive cued-vs-uncued report of one night.

    ``cue_recall`` is the morning sound check (``reports.cue_recall``); without it, or
    when it was finished after the retest, the limitations say so. ``protocol`` is the
    night's protocol preset block, copied into ``metadata.protocol`` with where it came
    from (``protocol_source``).
    """
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")
    assignment.validate_against_session(session)
    retest.validate_against_session(session)
    if dream_report is not None:
        dream_report.validate_against_session(session)
    if cue_recall is not None:
        cue_recall.validate_against_session(session)
    if retest.session_id != session.session_id:
        raise ValueError("retest session_id does not match night puzzle session")

    retest_by_id = {result.puzzle_id: result for result in retest.results}
    incorporations = _incorporation_map(dream_report)
    events = tuple(scheduler_events)
    cue_times_by_puzzle, unknown_event_puzzle_ids = _cue_times_by_puzzle(session, events)
    recalled = (
        {trial.puzzle_id: trial.correct for trial in cue_recall.trials} if cue_recall is not None else {}
    )

    rows = tuple(
        _build_row(
            puzzle_id,
            assignment=assignment,
            retest_by_id=retest_by_id,
            incorporations=incorporations,
            cue_times_by_puzzle=cue_times_by_puzzle,
            cue_recalled=recalled.get(puzzle_id),
        )
        for puzzle_id in session.puzzle_ids
    )
    metrics = tuple(_condition_metrics(condition, rows) for condition in ANALYSIS_CONDITIONS)
    recall_summary = _cue_recall_summary(cue_recall, assignment, retest)
    limitations = _limitations(
        rows,
        metrics,
        dream_report=dream_report,
        scheduler_events=events,
        unknown_event_puzzle_ids=unknown_event_puzzle_ids,
        min_group_size=min_group_size,
        recall_summary=recall_summary,
    )
    metadata: Dict[str, object] = {
        "assignment_seed": assignment.seed,
        "dream_report_available": dream_report is not None,
        "scheduler_events_available": bool(events),
        "cue_recall_available": cue_recall is not None,
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
        cue_timing=_cue_timing_summary(rows, events),
        cue_recall=recall_summary,
        limitations=limitations,
        metadata=metadata,
    )
    return report


def load_cued_uncued_analysis(input_path: Path) -> CuedUncuedAnalysisReport:
    return CuedUncuedAnalysisReport.load(input_path)


def _build_row(
    puzzle_id: str,
    *,
    assignment: PuzzleCueAssignment,
    retest_by_id: Mapping[str, object],
    incorporations: Mapping[str, object],
    cue_times_by_puzzle: Mapping[str, Tuple[float, ...]],
    cue_recalled: Optional[bool] = None,
) -> PuzzleAnalysisRow:
    retest_result = retest_by_id[puzzle_id]
    incorporation = incorporations.get(puzzle_id)
    cue_times = cue_times_by_puzzle.get(puzzle_id, ())
    condition = "cued" if assignment.is_cued(puzzle_id) else "uncued"
    return PuzzleAnalysisRow(
        puzzle_id=puzzle_id,
        cue_id=retest_result.cue_id,
        cue_condition=condition,
        solved=retest_result.solved,
        duration_seconds=retest_result.duration_seconds,
        confidence=retest_result.confidence,
        incorporated=incorporation is not None,
        incorporation_confidence=(
            incorporation.confidence if incorporation is not None else None
        ),
        cue_play_count=len(cue_times),
        first_cue_time_seconds=cue_times[0] if cue_times else None,
        last_cue_time_seconds=cue_times[-1] if cue_times else None,
        cue_time_seconds=cue_times,
        cue_recalled=cue_recalled,
    )


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
        "uncued_puzzle_play_count": sum(
            row.cue_play_count for row in rows if row.cue_condition == "uncued"
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
    if scheduler_events and missing_cued_timing:
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
        if row.cue_condition == "uncued" and row.cue_play_count > 0
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


def _incorporation_map(
    dream_report: Optional[DreamReport],
) -> Dict[str, object]:
    if dream_report is None:
        return {}
    return {
        link.puzzle_id: link
        for link in dream_report.puzzle_incorporations
        if link.incorporated
    }


def _cue_times_by_puzzle(
    session: NightPuzzleSession,
    events: Tuple[TmrSchedulerEvent, ...],
) -> Tuple[Dict[str, Tuple[float, ...]], Tuple[str, ...]]:
    session_ids = set(session.puzzle_ids)
    times: Dict[str, list] = {puzzle_id: [] for puzzle_id in session.puzzle_ids}
    unknown = []
    for event in events:
        if event.event_type != "play" or event.protocol != "puzzle" or not event.puzzle_id:
            continue
        if event.puzzle_id not in session_ids:
            unknown.append(event.puzzle_id)
            continue
        times[event.puzzle_id].append(event.timestamp_seconds)
    return (
        {puzzle_id: tuple(sorted(values)) for puzzle_id, values in times.items()},
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


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
