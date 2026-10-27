"""Blind coding of dream incorporation by raters (#228).

The sleeper's own puzzle links in a dream report are self-report. Here a rater codes
the dream text against the night's puzzles instead, in three categories, as the two
blind raters of Konkoly et al. 2026 did (docs/references.md):

- ``sound_cue``: the report has a sound that fits this puzzle's sound;
- ``direct_conceptual``: the report has the puzzle itself, its objects, words or task;
- ``indirect_conceptual``: the report has something clearly related to the puzzle
  without the puzzle itself.

``export_blind_coding_packet()`` gives the rater what coding needs and nothing more:
the dream text and the night's puzzles (ID and prompt) in ``blind_index`` order, the
night's session order, which the cued/uncued split does not touch. No cue ID, no cue
condition, no assignment, no arm, nothing the sleeper said about sounds and none of
their own links; it takes no assignment and no schedule, so it cannot hold them. A rater
who codes ``sound_cue`` needs the puzzles' sounds, which the packet leaves out with
everything else about cues: hand them over with the puzzle materials.

A rater's codes are an ``IncorporationCodingSheet`` of ``IncorporationCoding`` rows, one
per puzzle. ``coding_agreement()`` compares two raters: percent agreement and Cohen's
kappa per category, in pure Python. ``rater_incorporation()`` is what the analysis reads:
with several raters, a puzzle counts as incorporated when more than half of those who
coded it put it in some category (with two raters, both).

At n = 1 the sleeper may code their own dreams. The packet keeps them blind to which
puzzles were cued, not to the dream: they wrote it.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from muse_tmr.protocol import NightPuzzleSession, PuzzleCatalog
from muse_tmr.reports.dream_report import DreamReport

BLIND_CODING_PACKET_SCHEMA_VERSION = 1
INCORPORATION_CODING_SCHEMA_VERSION = 1
PACKET_KIND = "blind_coding_packet"
CODING_KIND = "incorporation_coding"
CODING_CATEGORIES = ("sound_cue", "direct_conceptual", "indirect_conceptual")
# The analysis's "incorporated": some category, by more than half of the raters.
ANY_CATEGORY = "any"
CATEGORY_DEFINITIONS = {
    "sound_cue": "The report has a sound that fits this puzzle's sound, or says that sound was heard.",
    "direct_conceptual": "The report has this puzzle itself: its objects, its words or its task.",
    "indirect_conceptual": "The report has something clearly related to this puzzle, without the puzzle itself.",
}
PACKET_INSTRUCTIONS = (
    "Read the dream report, then code every puzzle below: for each category, does the report hold it for this "
    "puzzle? Code every puzzle, with none when no category fits. The packet says nothing about which sounds "
    "played that night, and it should stay that way: do not look at the night's logs before you are done."
)
NONE_CODE = "none"


@dataclass(frozen=True)
class BlindCodingPuzzle:
    blind_index: int
    puzzle_id: str
    prompt: str

    def __post_init__(self) -> None:
        if self.blind_index <= 0:
            raise ValueError("blind_index must be positive")
        object.__setattr__(self, "puzzle_id", _required_str(self.puzzle_id, "puzzle_id"))
        object.__setattr__(self, "prompt", str(self.prompt).strip())

    def to_dict(self) -> Dict[str, object]:
        return {"blind_index": self.blind_index, "puzzle_id": self.puzzle_id, "prompt": self.prompt}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BlindCodingPuzzle":
        return cls(
            blind_index=int(payload["blind_index"]),
            puzzle_id=str(payload["puzzle_id"]),
            prompt=str(payload.get("prompt", "")),
        )


@dataclass(frozen=True)
class BlindCodingPacket:
    """One night's dream text and puzzles for a rater, in ``blind_index`` order."""

    session_id: str
    dream_text: str
    puzzles: Tuple[BlindCodingPuzzle, ...]
    packet_id: str = ""
    generated_at_utc: str = ""
    schema_version: int = BLIND_CODING_PACKET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _required_str(self.session_id, "session_id"))
        object.__setattr__(self, "dream_text", str(self.dream_text))
        object.__setattr__(self, "puzzles", tuple(sorted(self.puzzles, key=lambda puzzle: puzzle.blind_index)))
        if not self.packet_id:
            object.__setattr__(self, "packet_id", f"{self.session_id}-blind-coding")
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())
        if not self.puzzles:
            raise ValueError("a blind coding packet needs at least one puzzle")
        puzzle_ids = [puzzle.puzzle_id for puzzle in self.puzzles]
        if len(set(puzzle_ids)) != len(puzzle_ids):
            raise ValueError("a blind coding packet cannot hold a puzzle twice")
        if [puzzle.blind_index for puzzle in self.puzzles] != list(range(1, len(self.puzzles) + 1)):
            raise ValueError("blind_index must run 1, 2, 3, ...")

    @property
    def puzzle_ids(self) -> Tuple[str, ...]:
        return tuple(puzzle.puzzle_id for puzzle in self.puzzles)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": PACKET_KIND,
            "schema_version": self.schema_version,
            "packet_id": self.packet_id,
            "session_id": self.session_id,
            "generated_at_utc": self.generated_at_utc,
            "instructions": PACKET_INSTRUCTIONS,
            "categories": dict(CATEGORY_DEFINITIONS),
            "dream_text": self.dream_text,
            "puzzles": [puzzle.to_dict() for puzzle in self.puzzles],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Blind coding packet: {self.session_id}",
            "",
            PACKET_INSTRUCTIONS,
            "",
            "## Categories",
            "",
        ]
        lines += [f"- `{name}`: {text}" for name, text in CATEGORY_DEFINITIONS.items()]
        lines += ["", "## Dream report", "", self.dream_text.strip() or "(empty)", "", "## Puzzles", ""]
        lines += [f"{puzzle.blind_index}. `{puzzle.puzzle_id}`: {puzzle.prompt}" for puzzle in self.puzzles]
        lines += [
            "",
            "Record the codes with `muse-tmr record-incorporation-coding <packet.json> --rater-id <you> "
            "--code <puzzle_id>=<categories or none> ...`, one `--code` per puzzle.",
        ]
        return "\n".join(lines) + "\n"

    def save(self, output_path: Path) -> Path:
        return _write_json(output_path, self.to_dict())

    def save_markdown(self, output_path: Path) -> Path:
        return _write_text(output_path, self.to_markdown())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BlindCodingPacket":
        if payload.get("kind") not in (None, PACKET_KIND):
            raise ValueError(f"not a blind coding packet: kind={payload.get('kind')!r}")
        return cls(
            schema_version=int(payload.get("schema_version", BLIND_CODING_PACKET_SCHEMA_VERSION)),
            packet_id=str(payload.get("packet_id", "")),
            session_id=str(payload["session_id"]),
            generated_at_utc=str(payload.get("generated_at_utc", "")),
            dream_text=str(payload.get("dream_text", "")),
            puzzles=tuple(BlindCodingPuzzle.from_dict(item) for item in payload.get("puzzles", ()) or ()),
        )

    @classmethod
    def load(cls, input_path: Path) -> "BlindCodingPacket":
        return cls.from_dict(_read_json(input_path))


@dataclass(frozen=True)
class IncorporationCoding:
    """One rater's codes for one puzzle."""

    rater_id: str
    puzzle_id: str
    sound_cue: bool = False
    direct_conceptual: bool = False
    indirect_conceptual: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rater_id", _required_str(self.rater_id, "rater_id"))
        object.__setattr__(self, "puzzle_id", _required_str(self.puzzle_id, "puzzle_id"))
        for name in CODING_CATEGORIES:
            object.__setattr__(self, name, _boolish(getattr(self, name)))
        object.__setattr__(self, "notes", str(self.notes))

    @property
    def categories(self) -> Tuple[str, ...]:
        return tuple(name for name in CODING_CATEGORIES if getattr(self, name))

    @property
    def incorporated(self) -> bool:
        return bool(self.categories)

    def code(self, category: str) -> bool:
        if category == ANY_CATEGORY:
            return self.incorporated
        if category not in CODING_CATEGORIES:
            raise KeyError(f"unknown coding category {category!r}")
        return bool(getattr(self, category))

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"rater_id": self.rater_id, "puzzle_id": self.puzzle_id}
        payload.update({name: bool(getattr(self, name)) for name in CODING_CATEGORIES})
        payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IncorporationCoding":
        return cls(
            rater_id=str(payload["rater_id"]),
            puzzle_id=str(payload["puzzle_id"]),
            notes=str(payload.get("notes", "")),
            **{name: _boolish(payload.get(name, False)) for name in CODING_CATEGORIES},
        )


@dataclass(frozen=True)
class IncorporationCodingSheet:
    """A rater's codes for one night's packet."""

    session_id: str
    rater_id: str
    codings: Tuple[IncorporationCoding, ...]
    packet_id: str = ""
    coded_at_utc: str = ""
    notes: str = ""
    schema_version: int = INCORPORATION_CODING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _required_str(self.session_id, "session_id"))
        object.__setattr__(self, "rater_id", _required_str(self.rater_id, "rater_id"))
        object.__setattr__(self, "codings", tuple(self.codings))
        object.__setattr__(self, "notes", str(self.notes))
        if not self.coded_at_utc:
            object.__setattr__(self, "coded_at_utc", _utc_now())
        others = sorted({coding.rater_id for coding in self.codings} - {self.rater_id})
        if others:
            raise ValueError(f"a coding sheet of {self.rater_id} holds codes of {', '.join(others)}")
        puzzle_ids = [coding.puzzle_id for coding in self.codings]
        if len(set(puzzle_ids)) != len(puzzle_ids):
            raise ValueError("a coding sheet codes each puzzle once")

    @property
    def puzzle_ids(self) -> Tuple[str, ...]:
        return tuple(coding.puzzle_id for coding in self.codings)

    def coding_for(self, puzzle_id: str) -> Optional[IncorporationCoding]:
        return next((coding for coding in self.codings if coding.puzzle_id == puzzle_id), None)

    def validate_against_session(self, session: NightPuzzleSession) -> "IncorporationCodingSheet":
        if self.session_id != session.session_id:
            raise ValueError("coding sheet session_id does not match night puzzle session")
        unknown = sorted(set(self.puzzle_ids) - set(session.puzzle_ids))
        if unknown:
            raise ValueError(f"coding sheet codes puzzles outside the session: {tuple(unknown)}")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": CODING_KIND,
            "schema_version": self.schema_version,
            "packet_id": self.packet_id,
            "session_id": self.session_id,
            "rater_id": self.rater_id,
            "coded_at_utc": self.coded_at_utc,
            "coding_count": len(self.codings),
            "incorporated_count": sum(1 for coding in self.codings if coding.incorporated),
            "codings": [coding.to_dict() for coding in self.codings],
            "notes": self.notes,
        }

    def save(self, output_path: Path) -> Path:
        return _write_json(output_path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IncorporationCodingSheet":
        if payload.get("kind") not in (None, CODING_KIND):
            raise ValueError(f"not an incorporation coding sheet: kind={payload.get('kind')!r}")
        return cls(
            schema_version=int(payload.get("schema_version", INCORPORATION_CODING_SCHEMA_VERSION)),
            packet_id=str(payload.get("packet_id", "")),
            session_id=str(payload["session_id"]),
            rater_id=str(payload["rater_id"]),
            coded_at_utc=str(payload.get("coded_at_utc", "")),
            codings=tuple(IncorporationCoding.from_dict(item) for item in payload.get("codings", ()) or ()),
            notes=str(payload.get("notes", "")),
        )

    @classmethod
    def load(cls, input_path: Path) -> "IncorporationCodingSheet":
        return cls.from_dict(_read_json(input_path))


@dataclass(frozen=True)
class CodingAgreement:
    """Two raters against each other over the puzzles both coded."""

    rater_ids: Tuple[str, str]
    item_count: int
    categories: Mapping[str, Mapping[str, object]]
    session_ids: Tuple[str, ...] = ()
    unmatched: Tuple[Mapping[str, str], ...] = ()
    generated_at_utc: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "unmatched", tuple(dict(item) for item in self.unmatched))
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": "incorporation_coding_agreement",
            "rater_ids": list(self.rater_ids),
            "session_ids": list(self.session_ids),
            "item_count": self.item_count,
            "categories": {name: dict(stats) for name, stats in self.categories.items()},
            "unmatched": [dict(item) for item in self.unmatched],
            "generated_at_utc": self.generated_at_utc,
            "kappa_definition": (
                "Cohen's kappa of two raters' yes/no codes of the same puzzles, per category; 'any' is "
                "whether a rater put the puzzle in some category"
            ),
        }

    def save(self, output_path: Path) -> Path:
        return _write_json(output_path, self.to_dict())


def export_blind_coding_packet(
    session: NightPuzzleSession,
    dream_report: DreamReport,
    catalog: PuzzleCatalog,
    *,
    packet_id: str = "",
    generated_at_utc: str = "",
) -> BlindCodingPacket:
    """The night's dream text and puzzles in ``blind_index`` order (the session's order,
    1 to N, as the retest numbers them). Nothing else of the dream report goes in."""
    dream_report.validate_against_session(session)
    puzzles = tuple(
        BlindCodingPuzzle(blind_index=index, puzzle_id=puzzle_id, prompt=catalog.get_puzzle(puzzle_id).prompt)
        for index, puzzle_id in enumerate(session.puzzle_ids, start=1)
    )
    return BlindCodingPacket(
        session_id=session.session_id,
        dream_text=dream_report.dream_text,
        puzzles=puzzles,
        packet_id=packet_id,
        generated_at_utc=generated_at_utc,
    )


def build_incorporation_coding(
    packet: BlindCodingPacket,
    rater_id: str,
    codes: Mapping[str, Iterable[str]],
    *,
    notes: Optional[Mapping[str, str]] = None,
    sheet_notes: str = "",
    coded_at_utc: str = "",
) -> IncorporationCodingSheet:
    """A rater's sheet for ``packet``: ``codes`` maps every packet puzzle to its
    categories, an empty list (or ``none``) for none. A puzzle left out is an error, not
    a no: agreement needs every puzzle coded."""
    notes = dict(notes or {})
    missing = [puzzle_id for puzzle_id in packet.puzzle_ids if puzzle_id not in codes]
    extra = sorted(set(codes) - set(packet.puzzle_ids))
    if missing or extra:
        raise ValueError(f"code every puzzle of the packet once: missing={tuple(missing)} extra={tuple(extra)}")
    extra_notes = sorted(set(notes) - set(packet.puzzle_ids))
    if extra_notes:
        raise ValueError(f"notes for puzzles outside the packet: {tuple(extra_notes)}")
    codings = []
    for puzzle_id in packet.puzzle_ids:
        categories = _categories(codes[puzzle_id])
        codings.append(
            IncorporationCoding(
                rater_id=rater_id,
                puzzle_id=puzzle_id,
                notes=notes.get(puzzle_id, ""),
                **{name: name in categories for name in CODING_CATEGORIES},
            )
        )
    return IncorporationCodingSheet(
        session_id=packet.session_id,
        rater_id=rater_id,
        codings=tuple(codings),
        packet_id=packet.packet_id,
        coded_at_utc=coded_at_utc,
        notes=sheet_notes,
    )


def parse_coding_categories(text: str) -> Tuple[str, ...]:
    """``direct_conceptual,sound_cue`` or ``none`` as a tuple of categories."""
    return _categories(part for part in str(text).split(",") if part.strip())


def coding_agreement(sheets: Iterable[IncorporationCodingSheet]) -> CodingAgreement:
    """Percent agreement and Cohen's kappa per category, and for any category, between
    the two raters of ``sheets``, over every puzzle of every night both coded. A puzzle
    only one rater coded is listed as unmatched and left out."""
    sheets = tuple(sheets)
    raters = sorted({sheet.rater_id for sheet in sheets})
    if len(raters) != 2:
        raise ValueError(f"agreement needs the sheets of exactly two raters, got {len(raters)}: {', '.join(raters)}")
    codes: Dict[str, Dict[Tuple[str, str], IncorporationCoding]] = {rater: {} for rater in raters}
    for sheet in sheets:
        for coding in sheet.codings:
            key = (sheet.session_id, coding.puzzle_id)
            if key in codes[sheet.rater_id]:
                raise ValueError(f"{sheet.rater_id} coded {key[1]} of {key[0]} twice")
            codes[sheet.rater_id][key] = coding
    first, second = raters
    shared = sorted(set(codes[first]) & set(codes[second]))
    unmatched = [
        {"rater_id": rater, "session_id": key[0], "puzzle_id": key[1]}
        for rater in raters
        for key in sorted(set(codes[rater]) - set(shared))
    ]
    categories: Dict[str, Dict[str, object]] = {}
    for category in CODING_CATEGORIES + (ANY_CATEGORY,):
        a = [codes[first][key].code(category) for key in shared]
        b = [codes[second][key].code(category) for key in shared]
        kappa = cohens_kappa(a, b)
        agreements = sum(1 for left, right in zip(a, b) if left == right)
        categories[category] = {
            "items": len(shared),
            "agreements": agreements,
            "percent_agreement": agreements / len(shared) if shared else None,
            "kappa": kappa,
            "kappa_note": _kappa_note(a, b, kappa),
            "yes_counts": {first: sum(a), second: sum(b)},
        }
    return CodingAgreement(
        rater_ids=(first, second),
        item_count=len(shared),
        categories=categories,
        session_ids=tuple(sorted({key[0] for key in shared})),
        unmatched=tuple(unmatched),
    )


def cohens_kappa(first: Sequence[bool], second: Sequence[bool]) -> Optional[float]:
    """Cohen's kappa of two raters' yes/no codes of the same items: observed agreement
    against the agreement their own yes rates give by chance. None without items, or when
    chance agreement is 1 (both raters gave one and the same answer to every item)."""
    if len(first) != len(second):
        raise ValueError("kappa needs two codes per item")
    count = len(first)
    if count == 0:
        return None
    observed = sum(1 for left, right in zip(first, second) if bool(left) == bool(right)) / count
    first_yes = sum(1 for value in first if value) / count
    second_yes = sum(1 for value in second if value) / count
    chance = first_yes * second_yes + (1 - first_yes) * (1 - second_yes)
    if chance >= 1.0 - 1e-12:
        return None
    return (observed - chance) / (1 - chance)


def rater_incorporation(
    sheets: Iterable[IncorporationCodingSheet],
    session: NightPuzzleSession,
) -> Dict[str, Dict[str, object]]:
    """For each session puzzle: how many raters coded it, the share of them per category
    and for any category, and ``incorporated``: more than half of them put it in some
    category. A puzzle no rater coded has ``rater_count`` 0 and is not incorporated."""
    sheets = tuple(sheet.validate_against_session(session) for sheet in sheets if sheet.session_id == session.session_id)
    raters = [sheet.rater_id for sheet in sheets]
    if len(set(raters)) != len(raters):
        raise ValueError(f"more than one coding sheet of the same rater for {session.session_id}")
    result: Dict[str, Dict[str, object]] = {}
    for puzzle_id in session.puzzle_ids:
        coded = [coding for coding in (sheet.coding_for(puzzle_id) for sheet in sheets) if coding is not None]
        count = len(coded)
        shares = {
            category: (sum(1 for coding in coded if coding.code(category)) / count if count else None)
            for category in CODING_CATEGORIES + (ANY_CATEGORY,)
        }
        result[puzzle_id] = {
            "rater_count": count,
            **shares,
            "incorporated": bool(count) and float(shares[ANY_CATEGORY] or 0.0) > 0.5,
            "split": bool(count) and 0.0 < float(shares[ANY_CATEGORY] or 0.0) < 1.0,
        }
    return result


def load_blind_coding_packet(input_path: Path) -> BlindCodingPacket:
    return BlindCodingPacket.load(input_path)


def load_incorporation_coding(input_path: Path) -> IncorporationCodingSheet:
    return IncorporationCodingSheet.load(input_path)


def _categories(values: Iterable[str]) -> Tuple[str, ...]:
    names: List[str] = []
    for value in values:
        name = str(value).strip().lower()
        if not name:
            continue
        if name == NONE_CODE:
            continue
        if name not in CODING_CATEGORIES:
            raise ValueError(f"unknown coding category {value!r}: use {', '.join(CODING_CATEGORIES)} or {NONE_CODE}")
        if name not in names:
            names.append(name)
    return tuple(names)


def _kappa_note(first: Sequence[bool], second: Sequence[bool], kappa: Optional[float]) -> Optional[str]:
    if not first:
        return "no puzzle coded by both raters"
    if kappa is None:
        return "undefined: both raters gave the same answer to every puzzle, so chance agreement is 1"
    return None


def _required_str(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"expected yes/no boolean value, got {value!r}")


def _read_json(input_path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{input_path} holds no JSON object")
    return payload


def _write_json(output_path: Path, payload: Mapping[str, object]) -> Path:
    return _write_text(output_path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _write_text(output_path: Path, text: str) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
