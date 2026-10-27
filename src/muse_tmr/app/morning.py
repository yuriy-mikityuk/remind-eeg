"""The morning after a puzzle night, from the app: the dream report, the sound check and
the blind retest.

The forms save through the same builders as `record-dream-report`, `record-cue-recall`
and `record-puzzle-retest`, into the files the Pilot 5 runbook uses
(`data/reports/<night>_dream_report.json`, `<night>_cue_recall.json` and
`<night>_retest.json`). A night is a puzzle session file in `data/protocol`. Its cue
assignment is the one with the same session ID and its catalog is one that holds all of
its puzzles, so neither depends on file names; the sound check plays the night's cues
from a library in `data/cues` or `data/protocol` that holds every one of them, at the
newest volume calibration in `data/calibration`.

Nothing this module hands to the page says which puzzles were cued: the sound check's
trials are numbered 1 to N for every puzzle in an order the session alone decides, and
a saved check comes back as the answers only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar

from muse_tmr.audio import (
    AudioBackend,
    AudioCuePlayer,
    CueLibrary,
    VolumeCalibration,
    load_cue_library,
    load_volume_calibrations,
)
from muse_tmr.protocol import (
    NightPuzzleSession,
    PuzzleCatalog,
    PuzzleCueAssignment,
    load_night_puzzle_session,
    load_puzzle_catalog,
    load_puzzle_cue_assignment,
)
from muse_tmr.reports import (
    MorningRetestResult,
    build_dream_report,
    build_morning_retest,
    load_dream_report,
    load_morning_retest,
)
from muse_tmr.reports.cue_recall import (
    CueRecallTrialPlan,
    CueRecallUnavailable,
    build_cue_recall_test,
    cue_recall_playback_config,
    load_cue_recall_test,
    play_cue_recall_trial,
    plan_cue_recall,
)

# night-001_puzzles.json (the runbooks) and pilot5_smoke_session.json are both night "night-001"/"pilot5_smoke".
SESSION_SUFFIXES = ("_puzzles", "_session")
SHARED_CATALOG = "puzzle_catalog.json"
# Where the sound check looks for the night's cues, in this order, under data/.
CUE_LIBRARY_DIRS = ("cues", "protocol")
CALIBRATION_DIR = "calibration"

T = TypeVar("T")


class MorningFileExists(Exception):
    """The report is already saved and the request did not ask to replace it."""


class UnknownNight(LookupError):
    """No puzzle session in data/protocol has that night name."""


@dataclass(frozen=True)
class PuzzleNight:
    name: str
    session_path: Path
    session: NightPuzzleSession
    catalog_path: Optional[Path]
    catalog: Optional[PuzzleCatalog]
    assignment: Optional[PuzzleCueAssignment]

    def dream_report_path(self, reports_dir: Path) -> Path:
        return reports_dir / f"{self.name}_dream_report.json"

    def retest_path(self, reports_dir: Path) -> Path:
        return reports_dir / f"{self.name}_retest.json"

    def cue_recall_path(self, reports_dir: Path) -> Path:
        return reports_dir / f"{self.name}_cue_recall.json"

    def cue_recall_audio_log(self, reports_dir: Path) -> Path:
        return reports_dir / f"{self.name}_cue_recall_audio.jsonl"


def find_nights(protocol_dir: Path) -> List[PuzzleNight]:
    """Every puzzle session in protocol_dir, the newest file first."""
    if not protocol_dir.is_dir():
        return []
    sessions: List[Tuple[Path, NightPuzzleSession]] = []
    assignments: List[PuzzleCueAssignment] = []
    catalogs: List[Tuple[Path, PuzzleCatalog]] = []
    for path in protocol_dir.glob("*.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        # The catalog loader takes any JSON object, so a catalog needs puzzles of its own.
        if "puzzle_ids" in payload:
            session = _load(load_night_puzzle_session, path)
            if session is not None:
                sessions.append((path, session))
        elif "cued_puzzle_ids" in payload:
            assignment = _load(load_puzzle_cue_assignment, path)
            if assignment is not None:
                assignments.append(assignment)
        elif isinstance(payload.get("puzzles"), list) and payload["puzzles"]:
            catalog = _load(load_puzzle_catalog, path)
            if catalog is not None:
                catalogs.append((path, catalog))

    nights = []
    for path, session in sorted(sessions, key=lambda item: (item[0].stat().st_mtime, item[0].name), reverse=True):
        name = _night_name(path)
        catalog_path, catalog = _catalog_for(session, name, catalogs)
        assignment = next((item for item in assignments if item.session_id == session.session_id), None)
        nights.append(PuzzleNight(name, path, session, catalog_path, catalog, assignment))
    return nights


def nights_payload(protocol_dir: Path, reports_dir: Path) -> Dict[str, Any]:
    return {"nights": [_summary(night, reports_dir) for night in find_nights(protocol_dir)]}


def night_payload(protocol_dir: Path, reports_dir: Path, name: str) -> Dict[str, Any]:
    """The night's puzzles in session order with their prompts, and what is saved so far."""
    night = _night(protocol_dir, name)
    payload = _summary(night, reports_dir)
    payload["puzzles"] = [
        {
            "puzzle_id": puzzle_id,
            "position": index + 1,
            "prompt": night.catalog.get_puzzle(puzzle_id).prompt if night.catalog is not None else None,
        }
        for index, puzzle_id in enumerate(night.session.puzzle_ids)
    ]
    payload["dream_report"] = _saved_dream_report(night.dream_report_path(reports_dir))
    payload["retest"] = _saved_retest(night.retest_path(reports_dir))
    payload["cue_recall"] = _cue_recall_payload(night, protocol_dir, reports_dir)
    return payload


def play_cue_recall(
    protocol_dir: Path,
    reports_dir: Path,
    body: Mapping[str, Any],
    *,
    backend: AudioBackend,
) -> Dict[str, Any]:
    """Play one sound of the night's sound check through ``backend``.

    The answer says only that sound ``trial`` was played and how; never its cue or
    puzzle. Each play is logged to ``<night>_cue_recall_audio.jsonl``.
    """
    night = _night(protocol_dir, str(body.get("night") or ""))
    trial = int(body["trial"])
    reason = _cue_recall_unavailable(night, protocol_dir)
    if reason is not None:
        raise CueRecallUnavailable(reason)
    plan = next((item for item in _cue_recall_plans(night, protocol_dir) if item.trial == trial), None)
    if plan is None:
        raise ValueError(f"the sound check has no sound {trial}")
    player = AudioCuePlayer(
        cue_recall_playback_config(
            _calibration(protocol_dir),
            log_path=night.cue_recall_audio_log(reports_dir),
        ),
        backend=backend,
    )
    result = play_cue_recall_trial(player, plan)
    return {"trial": trial, "status": result["status"], "backend": result["backend_name"]}


def save_cue_recall(protocol_dir: Path, reports_dir: Path, body: Mapping[str, Any]) -> Dict[str, Any]:
    """Save the sound check. ``answers`` maps each sound number to a puzzle ID, or to
    nothing when the sleeper did not know."""
    night = _night(protocol_dir, str(body.get("night") or ""))
    if night.catalog is None:
        raise ValueError(f"no puzzle catalog in data/protocol holds the puzzles of {night.name}")
    if night.assignment is None:
        raise ValueError(f"no cue assignment in data/protocol for session {night.session.session_id}")
    path = night.cue_recall_path(reports_dir)
    _refuse_overwrite(path, body)
    answers = body.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise ValueError("answers must map every sound to a puzzle or to nothing")
    play_counts = body.get("play_counts") or {}
    if not isinstance(play_counts, dict):
        raise ValueError("play_counts must map sounds to counts")
    test = build_cue_recall_test(
        night.session,
        night.catalog,
        night.assignment,
        {int(trial): (str(answer) if answer else None) for trial, answer in answers.items()},
        play_counts={int(trial): int(count) for trial, count in play_counts.items()},
        started_at_utc=str(body.get("started_at_utc") or ""),
        notes=str(body.get("notes") or ""),
        metadata={"saved_by": "app", "audio_log": night.cue_recall_audio_log(reports_dir).name},
    )
    test.save(path)
    return {"saved": path.name, "reported_at_utc": test.completed_at_utc}


def save_dream_report(protocol_dir: Path, reports_dir: Path, body: Mapping[str, Any]) -> Dict[str, Any]:
    night = _night(protocol_dir, str(body.get("night") or ""))
    path = night.dream_report_path(reports_dir)
    _refuse_overwrite(path, body)
    links = body.get("puzzle_links") or {}
    if not isinstance(links, dict):
        raise ValueError("puzzle_links must map puzzle IDs to text")
    report = build_dream_report(
        night.session,
        catalog=night.catalog,
        lucid=_yes_no(body, "lucid"),
        cues_heard=_yes_no(body, "cues_heard"),
        confidence=_fraction(body, "confidence"),
        dream_text=str(body.get("dream_text") or ""),
        puzzle_incorporation_text={
            str(puzzle_id): str(text).strip() for puzzle_id, text in links.items() if str(text or "").strip()
        },
        notes=str(body.get("notes") or ""),
    )
    report.save(path)
    return {"saved": path.name, "reported_at_utc": report.reported_at_utc}


def save_retest(protocol_dir: Path, reports_dir: Path, body: Mapping[str, Any]) -> Dict[str, Any]:
    night = _night(protocol_dir, str(body.get("night") or ""))
    if night.catalog is None:
        raise ValueError(f"no puzzle catalog in data/protocol holds the puzzles of {night.name}")
    if night.assignment is None:
        raise ValueError(f"no cue assignment in data/protocol for session {night.session.session_id}")
    path = night.retest_path(reports_dir)
    _refuse_overwrite(path, body)
    entries = body.get("results")
    if not isinstance(entries, list) or not entries:
        raise ValueError("results must list every puzzle of the night")
    results = tuple(
        MorningRetestResult(
            puzzle_id=str(entry["puzzle_id"]),
            response=str(entry.get("response") or ""),
            solved=bool(entry.get("solved")),
            duration_seconds=float(entry["duration_seconds"]),
            confidence=_fraction(entry, "confidence"),
            notes=str(entry.get("notes") or ""),
        )
        for entry in entries
    )
    retest = build_morning_retest(
        night.session,
        results,
        catalog=night.catalog,
        assignment=night.assignment,
        notes=str(body.get("notes") or ""),
    )
    retest.save(path)
    return {"saved": path.name, "reported_at_utc": retest.reported_at_utc}


def _summary(night: PuzzleNight, reports_dir: Path) -> Dict[str, Any]:
    return {
        "name": night.name,
        "session_id": night.session.session_id,
        "session_file": night.session_path.name,
        "puzzle_count": len(night.session.puzzle_ids),
        "catalog_file": night.catalog_path.name if night.catalog_path is not None else None,
        "assignment_found": night.assignment is not None,
        "dream_report_saved": night.dream_report_path(reports_dir).is_file(),
        "cue_recall_saved": night.cue_recall_path(reports_dir).is_file(),
        "retest_saved": night.retest_path(reports_dir).is_file(),
    }


def _cue_recall_payload(night: PuzzleNight, protocol_dir: Path, reports_dir: Path) -> Dict[str, Any]:
    """The sound check as the page sees it: sounds 1 to N, one per puzzle, and whether
    they can play. No cue, puzzle or condition is tied to a number here."""
    reason = _cue_recall_unavailable(night, protocol_dir)
    return {
        "trials": [{"trial": index} for index in range(1, len(night.session.puzzle_ids) + 1)],
        "available": reason is None,
        "unavailable_reason": reason,
        "saved": _saved_cue_recall(night.cue_recall_path(reports_dir)),
    }


def _cue_recall_unavailable(night: PuzzleNight, protocol_dir: Path) -> Optional[str]:
    if night.catalog is None:
        return "no puzzle catalog in data/protocol holds this night's puzzles"
    if night.assignment is None:
        return "this night's cue assignment is missing"
    try:
        _cue_recall_plans(night, protocol_dir)
    except CueRecallUnavailable as exc:
        return str(exc)
    if _calibration(protocol_dir) is None:
        return "no volume calibration in data/calibration; run calibrate-volume first"
    return None


def _cue_recall_plans(night: PuzzleNight, protocol_dir: Path) -> Tuple[CueRecallTrialPlan, ...]:
    library = _cue_library_for(night, protocol_dir)
    if library is None:
        raise CueRecallUnavailable("no cue library in data/cues or data/protocol holds the night's sounds")
    return plan_cue_recall(night.session, night.catalog, library)


def _cue_library_for(night: PuzzleNight, protocol_dir: Path) -> Optional[CueLibrary]:
    """The first library, data/cues before data/protocol, that has a cue for every puzzle
    of the night; else the one that has the most, so the plan can say how many lack one."""
    if night.catalog is None:
        return None
    wanted = {night.catalog.get_puzzle(puzzle_id).cue_id for puzzle_id in night.session.puzzle_ids}
    best: Tuple[int, Optional[CueLibrary]] = (0, None)
    for directory in CUE_LIBRARY_DIRS:
        folder = protocol_dir.parent / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            payload = _read_json(path)
            if not isinstance(payload, dict) or not isinstance(payload.get("cues"), list):
                continue
            library = _load(load_cue_library, path)
            if library is None:
                continue
            held = len(wanted & {cue.cue_id for cue in library.cues})
            if held == len(wanted):
                return library
            if held > best[0]:
                best = (held, library)
    return best[1]


def _calibration(protocol_dir: Path) -> Optional[VolumeCalibration]:
    """The latest record of the newest calibration file in data/calibration."""
    folder = protocol_dir.parent / CALIBRATION_DIR
    if not folder.is_dir():
        return None
    for path in sorted(folder.glob("*.json"), key=lambda item: (item.stat().st_mtime, item.name), reverse=True):
        store = _load(load_volume_calibrations, path)
        if store is not None and store.calibrations:
            return store.latest()
    return None


def _saved_cue_recall(path: Path) -> Optional[Dict[str, Any]]:
    """A saved sound check for the form: the answers by sound number, nothing else."""
    if not path.is_file():
        return None
    test = load_cue_recall_test(path)
    return {
        "reported_at_utc": test.completed_at_utc,
        "notes": test.notes,
        "answers": {str(trial.trial): trial.response_puzzle_id for trial in test.trials},
    }


def _saved_dream_report(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    report = load_dream_report(path)
    return {
        "reported_at_utc": report.reported_at_utc,
        "lucid": report.lucid,
        "cues_heard": report.cues_heard,
        "confidence": report.confidence,
        "dream_text": report.dream_text,
        "notes": report.notes,
        "puzzle_links": {link.puzzle_id: link.dream_content for link in report.puzzle_incorporations},
    }


def _saved_retest(path: Path) -> Optional[Dict[str, Any]]:
    """A saved retest for the form, without the cue conditions the file keeps for the analysis."""
    if not path.is_file():
        return None
    retest = load_morning_retest(path)
    return {
        "reported_at_utc": retest.reported_at_utc,
        "notes": retest.notes,
        "results": {
            result.puzzle_id: {
                "response": result.response,
                "solved": result.solved,
                "duration_seconds": result.duration_seconds,
                "confidence": result.confidence,
                "notes": result.notes,
            }
            for result in retest.results
        },
    }


def _night(protocol_dir: Path, name: str) -> PuzzleNight:
    for night in find_nights(protocol_dir):
        if night.name == name:
            return night
    raise UnknownNight(f"no puzzle night named {name!r} in data/protocol")


def _night_name(session_path: Path) -> str:
    stem = session_path.stem
    for suffix in SESSION_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)]
    return stem


def _catalog_for(
    session: NightPuzzleSession, name: str, catalogs: List[Tuple[Path, PuzzleCatalog]]
) -> Tuple[Optional[Path], Optional[PuzzleCatalog]]:
    """The catalog named after the night, then the shared one, then any that holds every puzzle."""
    ranked = sorted(
        catalogs,
        key=lambda item: (item[0].stem != f"{name}_catalog", item[0].name != SHARED_CATALOG, item[0].name),
    )
    for path, catalog in ranked:
        known = catalog.puzzle_by_id()
        if all(puzzle_id in known for puzzle_id in session.puzzle_ids):
            return path, catalog
    return None, None


def _refuse_overwrite(path: Path, body: Mapping[str, Any]) -> None:
    if path.is_file() and not body.get("replace"):
        raise MorningFileExists(f"{path.name} is already saved")


def _yes_no(body: Mapping[str, Any], key: str) -> bool:
    value = body.get(key)
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("yes", "no"):
        return str(value).lower() == "yes"
    raise ValueError(f"{key} must be yes or no")


def _fraction(body: Mapping[str, Any], key: str) -> float:
    value = body.get(key)
    if value in (None, ""):
        raise ValueError(f"{key} is required")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return number


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load(loader: Callable[[Path], T], path: Path) -> Optional[T]:
    try:
        return loader(path)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
