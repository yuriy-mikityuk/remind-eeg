"""Morning dream report data capture.

Schema 2 (#228) adds ``woke_from_sound``, ``cue_perception`` and
``eye_signal_attempt_times``. ``cues_heard`` stays, for readers of schema 1; in schema 2
it is ``cue_perception`` in ``in_dream``, ``awake`` or ``unsure``. A schema 1 file loads
as it was, its ``cue_perception`` read from ``cues_heard`` (yes: ``unsure``, no:
``not_perceived``) and ``woke_from_sound`` ``unknown``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from muse_tmr.protocol import NightPuzzleSession, PuzzleCatalog

DREAM_REPORT_SCHEMA_VERSION = 2
WOKE_FROM_SOUND_ANSWERS = ("yes", "no", "unknown")
# Did the sleeper hear a sound in the night, and where: not at all, inside a dream, while
# awake, heard but unsure which, or no answer.
CUE_PERCEPTIONS = ("not_perceived", "in_dream", "awake", "unsure", "unknown")
# The perceptions that mean a sound was heard, which cues_heard says for schema 1.
HEARD_PERCEPTIONS = ("in_dream", "awake", "unsure")
# A clock time (04:12, 04:12:30) or an ISO 8601 time with a date.
_CLOCK_TIME = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?$")


@dataclass(frozen=True)
class DreamPuzzleIncorporation:
    puzzle_id: str
    dream_content: str
    incorporated: bool = True
    cue_id: str = ""
    confidence: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "puzzle_id", _required_str(self.puzzle_id, "puzzle_id"))
        object.__setattr__(self, "dream_content", str(self.dream_content).strip())
        object.__setattr__(self, "cue_id", str(self.cue_id).strip())
        object.__setattr__(self, "notes", str(self.notes))
        _validate_confidence(self.confidence, "confidence")
        if self.incorporated and not self.dream_content:
            raise ValueError("incorporated puzzle links require dream_content")

    def to_dict(self) -> Dict[str, object]:
        return {
            "puzzle_id": self.puzzle_id,
            "cue_id": self.cue_id,
            "incorporated": self.incorporated,
            "dream_content": self.dream_content,
            "confidence": self.confidence,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DreamPuzzleIncorporation":
        return cls(
            puzzle_id=str(payload["puzzle_id"]),
            cue_id=str(payload.get("cue_id", "")),
            incorporated=_boolish(payload.get("incorporated", True)),
            dream_content=str(payload.get("dream_content", "")),
            confidence=float(payload.get("confidence", 1.0)),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class DreamReport:
    session_id: str
    lucid: bool
    cues_heard: bool
    confidence: float
    dream_text: str
    puzzle_incorporations: Tuple[DreamPuzzleIncorporation, ...] = ()
    report_id: str = ""
    reported_at_utc: str = ""
    notes: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = DREAM_REPORT_SCHEMA_VERSION
    # Did a sound wake the sleeper: yes, no or unknown.
    woke_from_sound: str = "unknown"
    # One of CUE_PERCEPTIONS; None reads it from cues_heard (True: unsure, False:
    # not_perceived), as for a schema 1 file.
    cue_perception: Optional[str] = None
    # When the sleeper thinks they made the eye signal (LRLR), as clock times or ISO
    # times. A marking only: nothing detects or checks an eye signal.
    eye_signal_attempt_times: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _required_str(self.session_id, "session_id"))
        object.__setattr__(self, "dream_text", str(self.dream_text))
        object.__setattr__(self, "notes", str(self.notes))
        object.__setattr__(self, "puzzle_incorporations", tuple(self.puzzle_incorporations))
        object.__setattr__(self, "woke_from_sound", _choice(self.woke_from_sound, WOKE_FROM_SOUND_ANSWERS, "woke_from_sound"))
        if self.cue_perception is None:
            object.__setattr__(self, "cue_perception", perception_from_cues_heard(self.cues_heard))
        perception = _choice(self.cue_perception, CUE_PERCEPTIONS, "cue_perception")
        object.__setattr__(self, "cue_perception", perception)
        if perception != "unknown" and bool(self.cues_heard) != (perception in HEARD_PERCEPTIONS):
            raise ValueError(f"cues_heard={self.cues_heard} contradicts cue_perception={perception}")
        object.__setattr__(
            self,
            "eye_signal_attempt_times",
            tuple(_attempt_time(item) for item in self.eye_signal_attempt_times),
        )
        _validate_confidence(self.confidence, "confidence")
        if not self.report_id:
            object.__setattr__(self, "report_id", f"{self.session_id}-dream-report")
        if not self.reported_at_utc:
            object.__setattr__(self, "reported_at_utc", _utc_now())
        puzzle_ids = [link.puzzle_id for link in self.puzzle_incorporations]
        if len(set(puzzle_ids)) != len(puzzle_ids):
            raise ValueError("puzzle_incorporations cannot contain duplicate puzzle IDs")

    @property
    def incorporated_puzzle_ids(self) -> Tuple[str, ...]:
        return tuple(
            link.puzzle_id
            for link in self.puzzle_incorporations
            if link.incorporated
        )

    @property
    def puzzle_incorporation_count(self) -> int:
        return len(self.incorporated_puzzle_ids)

    def validate_against_session(self, session: NightPuzzleSession) -> "DreamReport":
        if self.session_id != session.session_id:
            raise ValueError("dream report session_id does not match night puzzle session")
        session_puzzle_ids = set(session.puzzle_ids)
        unknown = tuple(
            sorted(
                link.puzzle_id
                for link in self.puzzle_incorporations
                if link.puzzle_id not in session_puzzle_ids
            )
        )
        if unknown:
            raise ValueError(f"dream report links unknown session puzzles: {unknown}")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "session_id": self.session_id,
            "reported_at_utc": self.reported_at_utc,
            "lucid": self.lucid,
            "cues_heard": self.cues_heard,
            "confidence": self.confidence,
            "dream_text": self.dream_text,
            "puzzle_incorporation_count": self.puzzle_incorporation_count,
            "puzzle_incorporations": [
                link.to_dict() for link in self.puzzle_incorporations
            ],
            "notes": self.notes,
            "metadata": dict(self.metadata),
            "woke_from_sound": self.woke_from_sound,
            "cue_perception": self.cue_perception,
            "eye_signal_attempt_times": list(self.eye_signal_attempt_times),
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
    def from_dict(cls, payload: Mapping[str, object]) -> "DreamReport":
        return cls(
            schema_version=int(payload.get("schema_version", DREAM_REPORT_SCHEMA_VERSION)),
            report_id=str(payload.get("report_id", "")),
            session_id=str(payload["session_id"]),
            reported_at_utc=str(payload.get("reported_at_utc", "")),
            lucid=_boolish(payload["lucid"]),
            cues_heard=_boolish(payload["cues_heard"]),
            confidence=float(payload["confidence"]),
            dream_text=str(payload.get("dream_text", "")),
            puzzle_incorporations=tuple(
                DreamPuzzleIncorporation.from_dict(item)
                for item in payload.get("puzzle_incorporations", ())
            ),
            notes=str(payload.get("notes", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
            woke_from_sound=str(payload.get("woke_from_sound") or "unknown"),
            cue_perception=_optional_text(payload.get("cue_perception")),
            eye_signal_attempt_times=tuple(str(item) for item in payload.get("eye_signal_attempt_times", ()) or ()),
        )

    @classmethod
    def load(cls, input_path: Path) -> "DreamReport":
        return cls.from_dict(json.loads(input_path.expanduser().read_text(encoding="utf-8")))


def build_dream_report(
    session: NightPuzzleSession,
    *,
    lucid: bool,
    confidence: float,
    dream_text: str,
    cues_heard: Optional[bool] = None,
    puzzle_incorporation_text: Optional[Mapping[str, str]] = None,
    catalog: Optional[PuzzleCatalog] = None,
    report_id: str = "",
    reported_at_utc: str = "",
    notes: str = "",
    woke_from_sound: str = "unknown",
    cue_perception: Optional[str] = None,
    eye_signal_attempt_times: Iterable[str] = (),
) -> DreamReport:
    """A schema 2 report. Give ``cue_perception``, ``cues_heard`` or both; the one left
    out is read from the other, and two that contradict each other raise."""
    if cues_heard is None and cue_perception is None:
        raise ValueError("give cue_perception or cues_heard")
    if cues_heard is None:
        cues_heard = _choice(cue_perception, CUE_PERCEPTIONS, "cue_perception") in HEARD_PERCEPTIONS
    links = []
    for puzzle_id, text in (puzzle_incorporation_text or {}).items():
        cue_id = catalog.get_puzzle(puzzle_id).cue_id if catalog is not None else ""
        links.append(
            DreamPuzzleIncorporation(
                puzzle_id=puzzle_id,
                cue_id=cue_id,
                dream_content=text,
                incorporated=True,
            )
        )
    report = DreamReport(
        report_id=report_id,
        session_id=session.session_id,
        reported_at_utc=reported_at_utc,
        lucid=lucid,
        cues_heard=cues_heard,
        confidence=confidence,
        dream_text=dream_text,
        puzzle_incorporations=tuple(links),
        notes=notes,
        metadata={
            "session_puzzle_count": len(session.puzzle_ids),
            "linked_puzzle_count": len(links),
        },
        woke_from_sound=woke_from_sound,
        cue_perception=cue_perception,
        eye_signal_attempt_times=tuple(eye_signal_attempt_times),
    )
    return report.validate_against_session(session)


def perception_from_cues_heard(cues_heard: bool) -> str:
    """What a schema 1 yes/no answer says in schema 2 terms: heard, unsure where; or not."""
    return "unsure" if cues_heard else "not_perceived"


def parse_eye_signal_attempt_times(text: str) -> Tuple[str, ...]:
    """Times from a text separated by commas or semicolons, each checked as ``DreamReport`` does."""
    return tuple(_attempt_time(item) for item in re.split(r"[,;]", str(text or "")) if item.strip())


def load_dream_report(input_path: Path) -> DreamReport:
    return DreamReport.load(input_path)


def _required_str(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _choice(value: object, choices: Tuple[str, ...], name: str) -> str:
    text = str(value).strip().lower()
    if text not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(choices)}")
    return text


def _optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _attempt_time(value: object) -> str:
    """A clock time (HH:MM or HH:MM:SS) or an ISO 8601 time with a date, as given."""
    text = str(value).strip()
    if _CLOCK_TIME.match(text):
        return text
    if "T" in text or " " in text:
        try:
            dt.datetime.fromisoformat(text)
            return text
        except ValueError:
            pass
    raise ValueError(f"eye-signal attempt time {value!r} is not HH:MM, HH:MM:SS or an ISO date and time")


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
