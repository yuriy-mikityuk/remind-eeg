"""The morning sound check: which puzzle does this sound belong to? (#231)

Sanders et al. 2019 and Konkoly et al. 2026 played each puzzle's sound in the morning,
before the retest, and asked which puzzle it went with. The answers show whether the
pairings the cues are meant to bring back still held; they are a manipulation check,
not an outcome.

Every puzzle of the night gets one trial, cued and uncued alike, in an order shuffled
with a seed made from the session ID alone (``cue_recall_order()``). So the order, the
number of sounds and their labels ("Sound 1", "Sound 2", ...) are the same whichever
puzzles were cued, and nothing shown during the test says which. The saved file keeps
``cue_condition`` per trial, from the assignment, for the analysis only.

Sounds play through the ``AudioCuePlayer`` the caller built: its backend resolved as the
CLI resolves it (``resolve_audio_backend()``), its volume capped by the calibrated
comfortable volume and the hard cap (``cue_recall_playback_config()``). This module
creates no backend.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from muse_tmr.audio import (
    AudioCuePlayer,
    AudioPlaybackConfig,
    CueLibrary,
    CueMetadata,
    TestCue,
    VolumeCalibration,
    audio_config_with_calibration,
)
from muse_tmr.protocol import NightPuzzleSession, PuzzleCatalog, PuzzleCueAssignment

CUE_RECALL_SCHEMA_VERSION = 1
CUE_RECALL_CONDITIONS = ("cued", "uncued", "unknown")
# Mixed into the order seed so it shares nothing with the assignment's seed.
_ORDER_SALT = "remind-cue-recall"
# What a played trial keeps of the player's result.
_PLAY_FIELDS = (
    "status",
    "backend_name",
    "requested_volume",
    "capped_volume",
    "effective_volume",
    "max_volume",
    "reason_codes",
    "started_at",
)


class CueRecallUnavailable(ValueError):
    """The night's sounds cannot all be played. The message names no puzzle."""


@dataclass(frozen=True)
class CueRecallTrialPlan:
    """One sound of the check: the trial number shown and the puzzle it belongs to."""

    trial: int
    puzzle_id: str
    cue: CueMetadata


@dataclass(frozen=True)
class CueRecallTrial:
    """One sound and the puzzle the sleeper said it belongs to (None: did not know)."""

    trial: int
    puzzle_id: str
    cue_id: str
    response_puzzle_id: Optional[str]
    cue_condition: str = "unknown"
    play_count: int = 0
    plays: Tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.trial <= 0:
            raise ValueError("trial must be positive")
        if self.cue_condition not in CUE_RECALL_CONDITIONS:
            raise ValueError(f"cue_condition must be one of: {', '.join(CUE_RECALL_CONDITIONS)}")
        if self.play_count < 0:
            raise ValueError("play_count must be non-negative")
        response = None if self.response_puzzle_id in (None, "") else str(self.response_puzzle_id)
        object.__setattr__(self, "response_puzzle_id", response)
        object.__setattr__(self, "plays", tuple(dict(play) for play in self.plays))

    @property
    def answered(self) -> bool:
        return self.response_puzzle_id is not None

    @property
    def correct(self) -> bool:
        return self.response_puzzle_id == self.puzzle_id

    def to_dict(self) -> Dict[str, object]:
        return {
            "trial": self.trial,
            "puzzle_id": self.puzzle_id,
            "cue_id": self.cue_id,
            "response_puzzle_id": self.response_puzzle_id,
            "correct": self.correct,
            "cue_condition": self.cue_condition,
            "play_count": self.play_count,
            "plays": [dict(play) for play in self.plays],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CueRecallTrial":
        return cls(
            trial=int(payload["trial"]),
            puzzle_id=str(payload["puzzle_id"]),
            cue_id=str(payload.get("cue_id", "")),
            response_puzzle_id=payload.get("response_puzzle_id"),  # type: ignore[arg-type]
            cue_condition=str(payload.get("cue_condition", "unknown")),
            play_count=int(payload.get("play_count", 0)),
            plays=tuple(payload.get("plays", ()) or ()),
        )


@dataclass(frozen=True)
class CueRecallTest:
    """A night's sound check, one trial per puzzle in the order the sounds played."""

    session_id: str
    trials: Tuple[CueRecallTrial, ...]
    order_seed: int
    test_id: str = ""
    started_at_utc: str = ""
    completed_at_utc: str = ""
    notes: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = CUE_RECALL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("session_id must not be empty")
        object.__setattr__(self, "trials", tuple(sorted(self.trials, key=lambda trial: trial.trial)))
        if not self.test_id:
            object.__setattr__(self, "test_id", f"{self.session_id}-cue-recall")
        if not self.completed_at_utc:
            object.__setattr__(self, "completed_at_utc", _utc_now())
        if not self.started_at_utc:
            object.__setattr__(self, "started_at_utc", self.completed_at_utc)
        numbers = [trial.trial for trial in self.trials]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("cue recall trials must be numbered 1 to N")
        puzzles = [trial.puzzle_id for trial in self.trials]
        if len(set(puzzles)) != len(puzzles):
            raise ValueError("each puzzle gets one cue recall trial")

    @property
    def trial_count(self) -> int:
        return len(self.trials)

    @property
    def answered_count(self) -> int:
        return sum(1 for trial in self.trials if trial.answered)

    @property
    def correct_count(self) -> int:
        return sum(1 for trial in self.trials if trial.correct)

    @property
    def recall_rate(self) -> Optional[float]:
        """Correct trials over all trials; a sound the sleeper could not place counts as wrong."""
        return _rate(self.correct_count, self.trial_count)

    def by_condition(self) -> Dict[str, Dict[str, object]]:
        rows = {}
        for condition in ("cued", "uncued"):
            trials = [trial for trial in self.trials if trial.cue_condition == condition]
            correct = sum(1 for trial in trials if trial.correct)
            rows[condition] = {
                "trial_count": len(trials),
                "correct_count": correct,
                "recall_rate": _rate(correct, len(trials)),
            }
        return rows

    def validate_against_session(self, session: NightPuzzleSession) -> "CueRecallTest":
        if self.session_id != session.session_id:
            raise ValueError("cue recall session_id does not match night puzzle session")
        if {trial.puzzle_id for trial in self.trials} != set(session.puzzle_ids):
            raise ValueError("cue recall must have one trial for every puzzle of the session")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "test_id": self.test_id,
            "session_id": self.session_id,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "order_seed": self.order_seed,
            "trial_count": self.trial_count,
            "answered_count": self.answered_count,
            "correct_count": self.correct_count,
            "recall_rate": self.recall_rate,
            "by_condition": self.by_condition(),
            "trials": [trial.to_dict() for trial in self.trials],
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    def save(self, output_path: Path) -> Path:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output_path

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CueRecallTest":
        return cls(
            schema_version=int(payload.get("schema_version", CUE_RECALL_SCHEMA_VERSION)),
            test_id=str(payload.get("test_id", "")),
            session_id=str(payload["session_id"]),
            started_at_utc=str(payload.get("started_at_utc", "")),
            completed_at_utc=str(payload.get("completed_at_utc", "")),
            order_seed=int(payload.get("order_seed", 0)),
            trials=tuple(CueRecallTrial.from_dict(item) for item in payload.get("trials", ()) or ()),
            notes=str(payload.get("notes", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def load(cls, input_path: Path) -> "CueRecallTest":
        return cls.from_dict(json.loads(Path(input_path).expanduser().read_text(encoding="utf-8")))


def load_cue_recall_test(input_path: Path) -> CueRecallTest:
    return CueRecallTest.load(input_path)


def cue_recall_order_seed(session_id: str) -> int:
    """The shuffle seed of a night's sounds, from its session ID only."""
    digest = hashlib.sha256(f"{_ORDER_SALT}:{session_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def cue_recall_order(session: NightPuzzleSession) -> Tuple[str, ...]:
    """The night's puzzles in the order their sounds play. Nothing but the session decides it."""
    order = list(session.puzzle_ids)
    random.Random(cue_recall_order_seed(session.session_id)).shuffle(order)
    return tuple(order)


def plan_cue_recall(
    session: NightPuzzleSession,
    catalog: PuzzleCatalog,
    cue_library: CueLibrary,
) -> Tuple[CueRecallTrialPlan, ...]:
    """One trial per puzzle of the night, numbered in ``cue_recall_order()``.

    Every puzzle needs a generated-tone cue in ``cue_library``, the cues the player can
    play. Otherwise ``CueRecallUnavailable`` says how many lack one, never which: a
    library that only holds the cued puzzles' sounds would give the split away.
    """
    order = cue_recall_order(session)
    plans = []
    unplayable = 0
    for index, puzzle_id in enumerate(order, start=1):
        cue_id = catalog.get_puzzle(puzzle_id).cue_id
        try:
            cue = cue_library.by_id(cue_id)
        except KeyError:
            unplayable += 1
            continue
        if cue.cue_type != "generated_tone" or cue.frequency_hz is None or cue.duration_seconds <= 0:
            unplayable += 1
            continue
        plans.append(CueRecallTrialPlan(trial=index, puzzle_id=puzzle_id, cue=cue))
    if unplayable:
        raise CueRecallUnavailable(
            f"{unplayable} of the night's {len(order)} puzzles have no generated-tone cue in the cue library, "
            "so the sound check cannot play every sound"
        )
    return tuple(plans)


def cue_recall_playback_config(
    calibration: VolumeCalibration,
    *,
    hard_max_volume: float = 0.20,
    default_volume: float = 0.02,
    log_path: Optional[Path] = None,
) -> AudioPlaybackConfig:
    """The player settings of the sound check: as at night, the cue's volume hint capped
    by the calibrated comfortable volume and the hard cap."""
    return audio_config_with_calibration(
        AudioPlaybackConfig(
            max_volume=hard_max_volume,
            default_volume=default_volume,
            device_name=calibration.device_name,
            log_path=log_path,
        ),
        calibration,
    )


def play_cue_recall_trial(player: AudioCuePlayer, plan: CueRecallTrialPlan) -> Dict[str, object]:
    """Play one trial's sound and return what the trial keeps of it.

    The player computes the volume with ``AudioPlaybackConfig.cue_volume()``: the cue's
    hint, or the default volume without one, under the player's cap.
    """
    cue = plan.cue
    result = player.play_test_cue(
        TestCue(cue_id=cue.cue_id, frequency_hz=float(cue.frequency_hz), duration_seconds=cue.duration_seconds),
        volume=cue.volume_hint,
    )
    payload = result.to_dict()
    return {key: payload.get(key) for key in _PLAY_FIELDS}


def build_cue_recall_test(
    session: NightPuzzleSession,
    catalog: PuzzleCatalog,
    assignment: PuzzleCueAssignment,
    answers: Mapping[int, Optional[str]],
    *,
    plays: Optional[Mapping[int, Sequence[Mapping[str, object]]]] = None,
    play_counts: Optional[Mapping[int, int]] = None,
    started_at_utc: str = "",
    completed_at_utc: str = "",
    notes: str = "",
    metadata: Optional[Mapping[str, object]] = None,
) -> CueRecallTest:
    """The sound check of ``session`` from ``answers``, trial number to puzzle ID or None.

    Every trial needs an answer; None is "did not know". ``cue_condition`` comes from
    ``assignment`` and is only written to the file.
    """
    assignment.validate_against_session(session)
    order = cue_recall_order(session)
    answers = {int(trial): answer for trial, answer in answers.items()}
    expected = set(range(1, len(order) + 1))
    if set(answers) != expected:
        raise ValueError(f"the sound check needs an answer for every sound, 1 to {len(order)}")
    known = set(session.puzzle_ids)
    plays = {int(trial): tuple(items) for trial, items in (plays or {}).items()}
    play_counts = {int(trial): int(count) for trial, count in (play_counts or {}).items()}
    trials = []
    for index, puzzle_id in enumerate(order, start=1):
        response = answers[index]
        response = None if response in (None, "") else str(response)
        if response is not None and response not in known:
            raise ValueError(f"sound {index}: {response!r} is not a puzzle of this night")
        trial_plays = plays.get(index, ())
        trials.append(
            CueRecallTrial(
                trial=index,
                puzzle_id=puzzle_id,
                cue_id=catalog.get_puzzle(puzzle_id).cue_id,
                response_puzzle_id=response,
                cue_condition=_condition(assignment, puzzle_id),
                play_count=max(play_counts.get(index, 0), len(trial_plays)),
                plays=trial_plays,
            )
        )
    test = CueRecallTest(
        session_id=session.session_id,
        trials=tuple(trials),
        order_seed=cue_recall_order_seed(session.session_id),
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        notes=notes,
        metadata={
            "blind_condition_note": "cue_condition is stored for analysis; it was not shown during the check",
            **dict(metadata or {}),
        },
    )
    return test.validate_against_session(session)


def _condition(assignment: PuzzleCueAssignment, puzzle_id: str) -> str:
    if assignment.is_cued(puzzle_id):
        return "cued"
    if assignment.is_uncued(puzzle_id):
        return "uncued"
    return "unknown"


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator <= 0 else numerator / denominator


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
