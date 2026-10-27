"""Cued-vs-uncued assignment helpers."""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generic, Mapping, Optional, Sequence, Tuple, TypeVar

from muse_tmr.protocol.puzzle_protocol import NightPuzzleSession, PuzzleCatalog

CUE_RANDOMIZATION_SCHEMA_VERSION = 1

T = TypeVar("T")


@dataclass(frozen=True)
class CueAssignment(Generic[T]):
    cued: Tuple[T, ...]
    uncued: Tuple[T, ...]
    seed: int


def split_cued_uncued(
    items: Sequence[T],
    seed: int,
    cued_count: Optional[int] = None,
) -> CueAssignment:
    if cued_count is None:
        cued_count = len(items) // 2
    if cued_count < 0 or cued_count > len(items):
        raise ValueError("cued_count must fit inside items")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return CueAssignment(
        cued=tuple(shuffled[:cued_count]),
        uncued=tuple(shuffled[cued_count:]),
        seed=seed,
    )


@dataclass(frozen=True)
class PuzzleCueAssignment:
    session_id: str
    cued_puzzle_ids: Tuple[str, ...]
    uncued_puzzle_ids: Tuple[str, ...]
    seed: int
    generated_at_utc: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = CUE_RANDOMIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _required_str(self.session_id, "session_id"))
        object.__setattr__(self, "cued_puzzle_ids", _normalized_ids(self.cued_puzzle_ids, "cued_puzzle_ids"))
        object.__setattr__(self, "uncued_puzzle_ids", _normalized_ids(self.uncued_puzzle_ids, "uncued_puzzle_ids"))
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())
        all_ids = self.cued_puzzle_ids + self.uncued_puzzle_ids
        if not all_ids:
            raise ValueError("assignment must include at least one puzzle")
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("cued and uncued puzzle IDs must be unique and non-overlapping")

    @property
    def scheduled_puzzle_ids(self) -> Tuple[str, ...]:
        """Puzzle IDs eligible for cue scheduling; uncued tasks are excluded."""

        return self.cued_puzzle_ids

    @property
    def all_puzzle_ids(self) -> Tuple[str, ...]:
        return self.cued_puzzle_ids + self.uncued_puzzle_ids

    def is_cued(self, puzzle_id: str) -> bool:
        return puzzle_id in self.cued_puzzle_ids

    def is_uncued(self, puzzle_id: str) -> bool:
        return puzzle_id in self.uncued_puzzle_ids

    def ensure_schedulable(self, puzzle_id: str) -> None:
        if self.is_cued(puzzle_id):
            return
        if self.is_uncued(puzzle_id):
            raise ValueError(f"uncued puzzle must not be scheduled: {puzzle_id}")
        raise KeyError(f"puzzle is not assigned in this session: {puzzle_id}")

    def scheduled_cue_ids(self, catalog: PuzzleCatalog) -> Tuple[str, ...]:
        cue_ids = []
        for puzzle_id in self.scheduled_puzzle_ids:
            self.ensure_schedulable(puzzle_id)
            cue_ids.append(catalog.get_puzzle(puzzle_id).cue_id)
        return tuple(cue_ids)

    def validate_against_session(self, session: NightPuzzleSession) -> "PuzzleCueAssignment":
        expected = set(session.puzzle_ids)
        assigned = set(self.all_puzzle_ids)
        if self.session_id != session.session_id:
            raise ValueError("assignment session_id does not match night puzzle session")
        if assigned != expected:
            missing = tuple(sorted(expected - assigned))
            extra = tuple(sorted(assigned - expected))
            raise ValueError(
                f"assignment must cover exactly the session puzzles: missing={missing} extra={extra}"
            )
        return self

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "generated_at_utc": self.generated_at_utc,
            "seed": self.seed,
            "cued_puzzle_ids": list(self.cued_puzzle_ids),
            "uncued_puzzle_ids": list(self.uncued_puzzle_ids),
            "scheduled_puzzle_ids": list(self.scheduled_puzzle_ids),
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
    def from_dict(cls, payload: Mapping[str, object]) -> "PuzzleCueAssignment":
        return cls(
            schema_version=int(payload.get("schema_version", CUE_RANDOMIZATION_SCHEMA_VERSION)),
            session_id=str(payload["session_id"]),
            generated_at_utc=str(payload.get("generated_at_utc", "")),
            seed=int(payload["seed"]),
            cued_puzzle_ids=tuple(str(item) for item in payload.get("cued_puzzle_ids", ())),
            uncued_puzzle_ids=tuple(str(item) for item in payload.get("uncued_puzzle_ids", ())),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def load(cls, input_path: Path) -> "PuzzleCueAssignment":
        return cls.from_dict(json.loads(input_path.expanduser().read_text(encoding="utf-8")))


def assign_cued_uncued_puzzles(
    session: NightPuzzleSession,
    *,
    seed: int,
    cued_count: Optional[int] = None,
    protocol: Optional[Mapping[str, object]] = None,
) -> PuzzleCueAssignment:
    """Split the session; ``protocol`` is the preset block recorded in the metadata."""
    assignment = split_cued_uncued(session.puzzle_ids, seed=seed, cued_count=cued_count)
    metadata: Dict[str, object] = {
        "session_puzzle_count": len(session.puzzle_ids),
        "requested_cued_count": cued_count,
    }
    if protocol:
        metadata["protocol"] = dict(protocol)
    puzzle_assignment = PuzzleCueAssignment(
        session_id=session.session_id,
        cued_puzzle_ids=tuple(str(item) for item in assignment.cued),
        uncued_puzzle_ids=tuple(str(item) for item in assignment.uncued),
        seed=seed,
        metadata=metadata,
    )
    return puzzle_assignment.validate_against_session(session)


def load_puzzle_cue_assignment(input_path: Path) -> PuzzleCueAssignment:
    return PuzzleCueAssignment.load(input_path)


def _normalized_ids(values: Sequence[str], name: str) -> Tuple[str, ...]:
    normalized = tuple(_required_str(value, name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate IDs")
    return normalized


def _required_str(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not contain empty IDs")
    return text


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
