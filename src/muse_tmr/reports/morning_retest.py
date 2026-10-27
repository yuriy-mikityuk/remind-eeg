"""Morning puzzle retest data capture.

Schema 2 (#228) adds ``retest_wave``: ``morning``, the retest the morning after the
night, or ``day7`` and ``day10``, the same puzzles a week or ten days later. A schema 1
file loads as it was, as a ``morning`` retest.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from muse_tmr.protocol import NightPuzzleSession, PuzzleCatalog, PuzzleCueAssignment

MORNING_RETEST_SCHEMA_VERSION = 2
RETEST_CUE_CONDITIONS = ("cued", "uncued", "unknown")
RETEST_WAVES = ("morning", "day7", "day10")


@dataclass(frozen=True)
class MorningRetestResult:
    puzzle_id: str
    response: str
    solved: bool
    duration_seconds: float
    confidence: float
    cue_condition: str = "unknown"
    cue_id: str = ""
    blind_index: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "puzzle_id", _required_str(self.puzzle_id, "puzzle_id"))
        object.__setattr__(self, "response", str(self.response))
        object.__setattr__(self, "cue_id", str(self.cue_id).strip())
        object.__setattr__(self, "notes", str(self.notes))
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        _validate_confidence(self.confidence, "confidence")
        if self.cue_condition not in RETEST_CUE_CONDITIONS:
            raise ValueError(f"cue_condition must be one of: {', '.join(RETEST_CUE_CONDITIONS)}")
        if self.blind_index < 0:
            raise ValueError("blind_index must be non-negative")

    def to_dict(self) -> Dict[str, object]:
        return {
            "puzzle_id": self.puzzle_id,
            "cue_id": self.cue_id,
            "cue_condition": self.cue_condition,
            "blind_index": self.blind_index,
            "response": self.response,
            "solved": self.solved,
            "duration_seconds": self.duration_seconds,
            "confidence": self.confidence,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MorningRetestResult":
        return cls(
            puzzle_id=str(payload["puzzle_id"]),
            cue_id=str(payload.get("cue_id", "")),
            cue_condition=str(payload.get("cue_condition", "unknown")),
            blind_index=int(payload.get("blind_index", 0)),
            response=str(payload.get("response", "")),
            solved=_boolish(payload["solved"]),
            duration_seconds=float(payload["duration_seconds"]),
            confidence=float(payload["confidence"]),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class MorningRetest:
    session_id: str
    results: Tuple[MorningRetestResult, ...]
    retest_id: str = ""
    reported_at_utc: str = ""
    notes: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = MORNING_RETEST_SCHEMA_VERSION
    # Which retest of the night's puzzles this is: morning, day7 or day10.
    retest_wave: str = "morning"

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _required_str(self.session_id, "session_id"))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "notes", str(self.notes))
        wave = str(self.retest_wave).strip().lower()
        if wave not in RETEST_WAVES:
            raise ValueError(f"retest_wave must be one of: {', '.join(RETEST_WAVES)}")
        object.__setattr__(self, "retest_wave", wave)
        if not self.retest_id:
            object.__setattr__(self, "retest_id", f"{self.session_id}-{wave}-retest")
        if not self.reported_at_utc:
            object.__setattr__(self, "reported_at_utc", _utc_now())
        puzzle_ids = [result.puzzle_id for result in self.results]
        if len(set(puzzle_ids)) != len(puzzle_ids):
            raise ValueError("morning retest cannot contain duplicate puzzle IDs")

    @property
    def puzzle_ids(self) -> Tuple[str, ...]:
        return tuple(result.puzzle_id for result in self.results)

    @property
    def solved_count(self) -> int:
        return sum(1 for result in self.results if result.solved)

    @property
    def unsolved_count(self) -> int:
        return len(self.results) - self.solved_count

    @property
    def mean_duration_seconds(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.duration_seconds for result in self.results) / len(self.results)

    def validate_against_session(
        self,
        session: NightPuzzleSession,
        *,
        require_complete: bool = True,
    ) -> "MorningRetest":
        if self.session_id != session.session_id:
            raise ValueError("morning retest session_id does not match night puzzle session")
        session_ids = set(session.puzzle_ids)
        result_ids = set(self.puzzle_ids)
        extra = tuple(sorted(result_ids - session_ids))
        missing = tuple(sorted(session_ids - result_ids))
        if extra:
            raise ValueError(f"morning retest includes unknown session puzzles: {extra}")
        if require_complete and missing:
            raise ValueError(f"morning retest is missing session puzzles: {missing}")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "retest_id": self.retest_id,
            "session_id": self.session_id,
            "retest_wave": self.retest_wave,
            "reported_at_utc": self.reported_at_utc,
            "result_count": len(self.results),
            "solved_count": self.solved_count,
            "unsolved_count": self.unsolved_count,
            "mean_duration_seconds": self.mean_duration_seconds,
            "results": [result.to_dict() for result in self.results],
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    def save(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MorningRetest":
        return cls(
            schema_version=int(payload.get("schema_version", MORNING_RETEST_SCHEMA_VERSION)),
            retest_id=str(payload.get("retest_id", "")),
            session_id=str(payload["session_id"]),
            reported_at_utc=str(payload.get("reported_at_utc", "")),
            results=tuple(
                MorningRetestResult.from_dict(item)
                for item in payload.get("results", ())
            ),
            notes=str(payload.get("notes", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
            retest_wave=str(payload.get("retest_wave") or "morning"),
        )

    @classmethod
    def load(cls, input_path: Path) -> "MorningRetest":
        return cls.from_dict(json.loads(input_path.expanduser().read_text(encoding="utf-8")))


def build_morning_retest(
    session: NightPuzzleSession,
    results: Iterable[MorningRetestResult],
    *,
    catalog: Optional[PuzzleCatalog] = None,
    assignment: Optional[PuzzleCueAssignment] = None,
    retest_id: str = "",
    reported_at_utc: str = "",
    notes: str = "",
    require_complete: bool = True,
    retest_wave: str = "morning",
) -> MorningRetest:
    if assignment is not None:
        assignment.validate_against_session(session)

    session_order = {puzzle_id: index + 1 for index, puzzle_id in enumerate(session.puzzle_ids)}
    enriched = tuple(
        _enrich_result(result, session_order, catalog=catalog, assignment=assignment)
        for result in results
    )
    retest = MorningRetest(
        retest_id=retest_id,
        session_id=session.session_id,
        reported_at_utc=reported_at_utc,
        results=tuple(sorted(enriched, key=lambda result: result.blind_index)),
        notes=notes,
        retest_wave=retest_wave,
        metadata={
            "session_puzzle_count": len(session.puzzle_ids),
            "assignment_available": assignment is not None,
            "catalog_available": catalog is not None,
            "blind_condition_note": "cue_condition is stored for analysis; do not reveal during retest",
        },
    )
    return retest.validate_against_session(session, require_complete=require_complete)


def load_morning_retest(input_path: Path) -> MorningRetest:
    return MorningRetest.load(input_path)


def _enrich_result(
    result: MorningRetestResult,
    session_order: Mapping[str, int],
    *,
    catalog: Optional[PuzzleCatalog],
    assignment: Optional[PuzzleCueAssignment],
) -> MorningRetestResult:
    if result.puzzle_id not in session_order:
        return result
    cue_id = result.cue_id
    if catalog is not None and not cue_id:
        cue_id = catalog.get_puzzle(result.puzzle_id).cue_id
    cue_condition = result.cue_condition
    if assignment is not None and cue_condition == "unknown":
        if assignment.is_cued(result.puzzle_id):
            cue_condition = "cued"
        elif assignment.is_uncued(result.puzzle_id):
            cue_condition = "uncued"
    return MorningRetestResult(
        puzzle_id=result.puzzle_id,
        cue_id=cue_id,
        cue_condition=cue_condition,
        blind_index=session_order[result.puzzle_id],
        response=result.response,
        solved=result.solved,
        duration_seconds=result.duration_seconds,
        confidence=result.confidence,
        notes=result.notes,
    )


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


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
