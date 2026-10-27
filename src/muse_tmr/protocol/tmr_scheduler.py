"""REM-gated TMR cue scheduler contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from muse_tmr.audio import (
    AudioPlaybackConfig,
    CueLibrary,
    VolumeCalibration,
    calibrated_max_volume,
)
from muse_tmr.models import RemGateDecision
from muse_tmr.protocol.arousal_guard import ArousalGuardDecision
from muse_tmr.protocol.night_schedule import CUEING_ARMS
from muse_tmr.protocol.puzzle_protocol import PuzzleCatalog
from muse_tmr.protocol.randomization import PuzzleCueAssignment
from muse_tmr.protocol.tlr_protocol import TlrBlockPlan

# The reason a sham night's play carries, the reason of a no-cue night's skip in place of
# a play, and the metadata flag that marks that skip (#228).
SHAM_ARM_REASON = "sham_arm"
NO_CUE_ARM_REASON = "no_cue_arm"
VIRTUAL_PLAY = "virtual_play"


@dataclass(frozen=True)
class CueDecision:
    should_play: bool
    reason_codes: Tuple[str, ...] = ()
    max_volume: Optional[float] = None
    calibration_device_name: Optional[str] = None


def arousal_guard_decision(reason_codes: Iterable[str]) -> CueDecision:
    """Convert safety guard reason codes into a cue playback decision."""

    unique_reasons = tuple(dict.fromkeys(reason_codes))
    return CueDecision(
        should_play=not unique_reasons,
        reason_codes=unique_reasons,
    )


def calibrated_cue_decision(
    reason_codes: Iterable[str],
    *,
    calibration: Optional[VolumeCalibration],
    fallback_max_volume: float = 0.20,
) -> CueDecision:
    """Convert scheduler guards into a cue decision with calibrated volume cap."""

    unique_reasons = list(dict.fromkeys(reason_codes))
    if calibration is None:
        if "volume_calibration_missing" not in unique_reasons:
            unique_reasons.append("volume_calibration_missing")
        return CueDecision(
            should_play=False,
            reason_codes=tuple(unique_reasons),
            max_volume=fallback_max_volume,
        )

    max_volume = calibrated_max_volume(calibration, hard_cap=fallback_max_volume)
    return CueDecision(
        should_play=not unique_reasons,
        reason_codes=tuple(unique_reasons),
        max_volume=max_volume,
        calibration_device_name=calibration.device_name,
    )


@dataclass(frozen=True)
class TmrSchedulerConfig:
    puzzle_cue_interval_seconds: float = 30.0
    cooldown_seconds: float = 120.0
    max_puzzle_cues_per_block: int = 4
    enable_tlr_block: bool = True
    # The night's arm (muse_tmr.protocol.night_schedule): cue plays as planned, sham plays
    # sham_cue_id in place of every puzzle and TLR cue, no_cue plays nothing.
    arm: str = "cue"
    sham_cue_id: Optional[str] = None

    def validate(self) -> None:
        if self.puzzle_cue_interval_seconds < 0:
            raise ValueError("puzzle_cue_interval_seconds must be non-negative")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if self.max_puzzle_cues_per_block <= 0:
            raise ValueError("max_puzzle_cues_per_block must be positive")
        if self.arm not in CUEING_ARMS:
            raise ValueError(f"arm must be one of: {', '.join(CUEING_ARMS)}")
        if self.arm == "sham" and not self.sham_cue_id:
            raise ValueError("the sham arm needs sham_cue_id")


@dataclass(frozen=True)
class TmrSchedulerEvent:
    event_type: str
    timestamp_seconds: float
    cue_id: Optional[str] = None
    protocol: Optional[str] = None
    puzzle_id: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    # The arm of the night that logged it. "cue", the default, is left out of the log,
    # so a cue night's events read as they did before arms existed.
    arm: str = "cue"

    def __post_init__(self) -> None:
        if self.event_type not in {"play", "skip", "pause", "stop"}:
            raise ValueError("event_type must be one of: play, skip, pause, stop")
        if self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        if self.arm not in CUEING_ARMS:
            raise ValueError(f"arm must be one of: {', '.join(CUEING_ARMS)}")
        object.__setattr__(self, "reason_codes", _unique(self.reason_codes))

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "event_type": self.event_type,
            "timestamp_seconds": self.timestamp_seconds,
            "reason_codes": list(self.reason_codes),
            "metadata": dict(self.metadata),
        }
        if self.cue_id is not None:
            payload["cue_id"] = self.cue_id
        if self.protocol is not None:
            payload["protocol"] = self.protocol
        if self.puzzle_id is not None:
            payload["puzzle_id"] = self.puzzle_id
        if self.arm != "cue":
            payload["arm"] = self.arm
        return payload

    @property
    def virtual_play(self) -> bool:
        """A no-cue night's skip where another arm would have played."""
        return self.event_type == "skip" and bool(self.metadata.get(VIRTUAL_PLAY))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TmrSchedulerEvent":
        return cls(
            event_type=str(payload["event_type"]),
            timestamp_seconds=float(payload["timestamp_seconds"]),
            cue_id=_optional_str(payload.get("cue_id")),
            protocol=_optional_str(payload.get("protocol")),
            puzzle_id=_optional_str(payload.get("puzzle_id")),
            reason_codes=tuple(str(item) for item in payload.get("reason_codes", ())),
            metadata=dict(payload.get("metadata", {}) or {}),
            arm=str(payload.get("arm") or "cue"),
        )


class TmrCueScheduler:
    """Deterministic scheduler for REM-gated TMR/TLR cue eligibility.

    This class logs scheduling decisions only. Audio playback remains behind
    `AudioCuePlayer`, and REM confidence remains behind `StableRemGate`.

    A play event keeps the cue's own `volume_hint`; on `lower_volume` it adds the
    guard's `volume_multiplier`, which the player applies after its cap. Given the
    player's `playback_config`, every play event also logs the volume that player
    plays it at (`AudioPlaybackConfig.cue_volume()`).

    `config.arm` decides only what a cue the gate lets through becomes, never when one
    comes: in `cue` it is played; in `sham` it names `config.sham_cue_id` instead, keeps
    the puzzle, the protocol and the metadata (length and volume included) of the cue it
    stands in for and adds `sham_arm`; in `no_cue` it is logged as a `skip` with
    `no_cue_arm` and `metadata.virtual_play`. So the three arms of one night log their
    cues at the same times. Every event carries the arm.
    """

    def __init__(
        self,
        *,
        assignment: PuzzleCueAssignment,
        catalog: PuzzleCatalog,
        cue_library: CueLibrary,
        config: Optional[TmrSchedulerConfig] = None,
        tlr_block_plan: Optional[TlrBlockPlan] = None,
        event_log_path: Optional[Path] = None,
        playback_config: Optional[AudioPlaybackConfig] = None,
    ) -> None:
        self.assignment = assignment
        self.catalog = catalog
        self.cue_library = cue_library
        self.config = config or TmrSchedulerConfig()
        self.config.validate()
        self.tlr_block_plan = tlr_block_plan
        self.event_log_path = event_log_path
        self.playback_config = playback_config
        self._scheduled_puzzles = _scheduled_puzzles(assignment, catalog, cue_library)
        if self.config.arm == "sham":
            validate_sham_cue(
                str(self.config.sham_cue_id),
                catalog=catalog,
                cue_library=cue_library,
                tlr_cue_ids=tlr_cue_ids_of(tlr_block_plan),
            )
        self._events = []
        self._active_block = False
        self._tlr_block_played = False
        self._tlr_block_start_seconds = None
        self._tlr_event_index = 0
        self._next_puzzle_index = 0
        self._puzzle_cues_in_block = 0
        self._next_puzzle_time_seconds = 0.0
        self._cooldown_until_seconds = 0.0
        self._stopped = False

    @property
    def events(self) -> Tuple[TmrSchedulerEvent, ...]:
        return tuple(self._events)

    @property
    def scheduled_puzzle_ids(self) -> Tuple[str, ...]:
        return tuple(puzzle_id for puzzle_id, _ in self._scheduled_puzzles)

    def update(
        self,
        gate_decision: RemGateDecision,
        *,
        timestamp_seconds: float,
        reason_codes: Iterable[str] = (),
        guard_decision: Optional[ArousalGuardDecision] = None,
    ) -> Tuple[TmrSchedulerEvent, ...]:
        if timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")

        if self._stopped:
            return (self._emit_skip(timestamp_seconds, ("scheduler_stopped",)),)

        guard_event = self._apply_guard_decision(guard_decision, timestamp_seconds)
        if guard_event is not None:
            return (guard_event,)

        if not gate_decision.gate_open:
            return (self._handle_closed_gate(gate_decision, timestamp_seconds),)

        volume_context = _guard_volume_context(guard_decision)
        guard_reasons = _unique(reason_codes)
        if guard_reasons:
            return (self._emit_skip(timestamp_seconds, guard_reasons),)

        if timestamp_seconds < self._cooldown_until_seconds:
            return (
                self._emit_skip(
                    timestamp_seconds,
                    ("scheduler_cooldown_active",),
                    metadata={"cooldown_until_seconds": self._cooldown_until_seconds},
                ),
            )

        if not self._active_block:
            self._start_rem_block(timestamp_seconds)

        tlr_events = self._maybe_emit_tlr_block(timestamp_seconds, volume_context)
        if tlr_events:
            return tlr_events

        if timestamp_seconds < self._next_puzzle_time_seconds:
            return (
                self._emit_skip(
                    timestamp_seconds,
                    ("cue_interval_active",),
                    metadata={"next_puzzle_time_seconds": self._next_puzzle_time_seconds},
                ),
            )

        if self._puzzle_cues_in_block >= self.config.max_puzzle_cues_per_block:
            self._cooldown_until_seconds = timestamp_seconds + self.config.cooldown_seconds
            return (
                self._emit_skip(
                    timestamp_seconds,
                    ("max_puzzle_cues_per_block_reached",),
                    metadata={"cooldown_until_seconds": self._cooldown_until_seconds},
                ),
            )

        if self._next_puzzle_index >= len(self._scheduled_puzzles):
            return (self._emit_skip(timestamp_seconds, ("no_cued_puzzles_remaining",)),)

        return (self._emit_next_puzzle_cue(timestamp_seconds, volume_context),)

    def stop(
        self,
        *,
        timestamp_seconds: float,
        reason_codes: Iterable[str] = ("scheduler_stop",),
    ) -> TmrSchedulerEvent:
        self._stopped = True
        self._active_block = False
        event = TmrSchedulerEvent(
            event_type="stop",
            timestamp_seconds=timestamp_seconds,
            reason_codes=_unique(reason_codes),
        )
        return self._record(event)

    def _apply_guard_decision(
        self,
        guard_decision: Optional[ArousalGuardDecision],
        timestamp_seconds: float,
    ) -> Optional[TmrSchedulerEvent]:
        if guard_decision is None or guard_decision.action in {"allow", "lower_volume"}:
            return None
        if guard_decision.should_stop:
            return self.stop(
                timestamp_seconds=timestamp_seconds,
                reason_codes=("arousal_guard_stop",) + guard_decision.reason_codes,
            )
        if guard_decision.should_pause:
            return self._pause_for_guard(guard_decision, timestamp_seconds)
        return None

    def _pause_for_guard(
        self,
        guard_decision: ArousalGuardDecision,
        timestamp_seconds: float,
    ) -> TmrSchedulerEvent:
        self._active_block = False
        self._tlr_block_played = False
        self._tlr_block_start_seconds = None
        self._tlr_event_index = 0
        self._puzzle_cues_in_block = 0
        pause_seconds = max(self.config.cooldown_seconds, guard_decision.pause_seconds)
        self._cooldown_until_seconds = timestamp_seconds + pause_seconds
        return self._record(
            TmrSchedulerEvent(
                event_type="pause",
                timestamp_seconds=timestamp_seconds,
                reason_codes=("arousal_guard_pause",) + guard_decision.reason_codes,
                metadata={
                    "cooldown_until_seconds": self._cooldown_until_seconds,
                    "pause_seconds": pause_seconds,
                    "arousal_guard": guard_decision.to_dict(),
                },
            )
        )

    def _handle_closed_gate(
        self,
        gate_decision: RemGateDecision,
        timestamp_seconds: float,
    ) -> TmrSchedulerEvent:
        reasons = _unique(("rem_gate_closed",) + tuple(gate_decision.reason_codes))
        if self._active_block:
            self._active_block = False
            self._tlr_block_played = False
            self._tlr_block_start_seconds = None
            self._tlr_event_index = 0
            self._puzzle_cues_in_block = 0
            self._cooldown_until_seconds = timestamp_seconds + self.config.cooldown_seconds
            return self._record(
                TmrSchedulerEvent(
                    event_type="pause",
                    timestamp_seconds=timestamp_seconds,
                    reason_codes=reasons,
                    metadata={"cooldown_until_seconds": self._cooldown_until_seconds},
                )
            )
        return self._emit_skip(timestamp_seconds, reasons)

    def _start_rem_block(self, timestamp_seconds: float) -> None:
        self._active_block = True
        self._tlr_block_played = False
        self._tlr_block_start_seconds = timestamp_seconds
        self._tlr_event_index = 0
        self._puzzle_cues_in_block = 0
        self._next_puzzle_time_seconds = timestamp_seconds

    def _maybe_emit_tlr_block(
        self,
        timestamp_seconds: float,
        volume_context: Mapping[str, object],
    ) -> Tuple[TmrSchedulerEvent, ...]:
        if not self.config.enable_tlr_block or self.tlr_block_plan is None or self._tlr_block_played:
            return ()

        block_start = (
            self._tlr_block_start_seconds
            if self._tlr_block_start_seconds is not None
            else timestamp_seconds
        )
        self._tlr_block_start_seconds = block_start

        if not self.tlr_block_plan.events:
            self._complete_tlr_block(block_start)
            return ()

        due_events = []
        while self._tlr_event_index < len(self.tlr_block_plan.events):
            block_event = self.tlr_block_plan.events[self._tlr_event_index]
            scheduled_time = block_start + block_event.offset_seconds
            if timestamp_seconds < scheduled_time:
                break
            due_events.append(self._emit_tlr_event(block_event, scheduled_time, volume_context))
            self._tlr_event_index += 1

        if self._tlr_event_index >= len(self.tlr_block_plan.events):
            self._complete_tlr_block(block_start)

        if due_events:
            return tuple(due_events)

        next_event = self.tlr_block_plan.events[self._tlr_event_index]
        return (
            self._emit_skip(
                timestamp_seconds,
                ("tlr_interval_active",),
                metadata={
                    "next_tlr_time_seconds": block_start + next_event.offset_seconds,
                },
            ),
        )

    def _complete_tlr_block(self, block_start_seconds: float) -> None:
        self._tlr_block_played = True
        self._next_puzzle_time_seconds = max(
            self._next_puzzle_time_seconds,
            block_start_seconds + self.tlr_block_plan.puzzle_cue_start_offset_seconds,
        )

    def _emit_tlr_event(
        self,
        block_event,
        timestamp_seconds: float,
        volume_context: Mapping[str, object],
    ) -> TmrSchedulerEvent:
        cue = self.cue_library.by_id(block_event.cue_id)
        if cue.protocol != "tlr":
            raise ValueError(f"TLR block cue must use protocol='tlr': {block_event.cue_id}")
        metadata = _cue_metadata(
            cue_duration_seconds=block_event.duration_seconds,
            cue_volume_hint=cue.volume_hint,
            volume_context=volume_context,
            extra={
                "block_event_type": block_event.event_type,
                "tlr_event_index": self._tlr_event_index + 1,
            },
            playback_config=self.playback_config,
        )
        return self._emit_cue(
            TmrSchedulerEvent(
                event_type="play",
                timestamp_seconds=timestamp_seconds,
                cue_id=block_event.cue_id,
                protocol="tlr",
                reason_codes=("rem_gate_open", "tlr_block"),
                metadata=metadata,
            )
        )

    def _emit_next_puzzle_cue(
        self,
        timestamp_seconds: float,
        volume_context: Mapping[str, object],
    ) -> TmrSchedulerEvent:
        puzzle_id, cue_id = self._scheduled_puzzles[self._next_puzzle_index]
        self.assignment.ensure_schedulable(puzzle_id)
        cue = self.cue_library.by_id(cue_id)
        metadata = _cue_metadata(
            cue_duration_seconds=cue.duration_seconds,
            cue_volume_hint=cue.volume_hint,
            volume_context=volume_context,
            extra={"puzzle_cue_index": self._puzzle_cues_in_block + 1},
            playback_config=self.playback_config,
        )
        event = self._emit_cue(
            TmrSchedulerEvent(
                event_type="play",
                timestamp_seconds=timestamp_seconds,
                cue_id=cue_id,
                protocol="puzzle",
                puzzle_id=puzzle_id,
                reason_codes=("rem_gate_open", "cued_puzzle"),
                metadata=metadata,
            )
        )
        self._next_puzzle_index += 1
        self._puzzle_cues_in_block += 1
        self._next_puzzle_time_seconds = timestamp_seconds + self.config.puzzle_cue_interval_seconds
        return event

    def _emit_skip(
        self,
        timestamp_seconds: float,
        reason_codes: Iterable[str],
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> TmrSchedulerEvent:
        return self._record(
            TmrSchedulerEvent(
                event_type="skip",
                timestamp_seconds=timestamp_seconds,
                reason_codes=_unique(reason_codes),
                metadata=dict(metadata or {}),
            )
        )

    def _emit_cue(self, event: TmrSchedulerEvent) -> TmrSchedulerEvent:
        """Record a cue the gate let through as the night's arm has it. Only the cue's
        ID (sham) or the event type (no_cue) changes, and a reason is added."""
        if self.config.arm == "sham":
            event = replace(
                event,
                cue_id=self.config.sham_cue_id,
                reason_codes=event.reason_codes + (SHAM_ARM_REASON,),
            )
        elif self.config.arm == "no_cue":
            event = replace(
                event,
                event_type="skip",
                reason_codes=event.reason_codes + (NO_CUE_ARM_REASON,),
                metadata={**dict(event.metadata), VIRTUAL_PLAY: True},
            )
        return self._record(event)

    def _record(self, event: TmrSchedulerEvent) -> TmrSchedulerEvent:
        if event.arm != self.config.arm:
            event = replace(event, arm=self.config.arm)
        if event.event_type == "play" and event.arm == "no_cue":
            raise RuntimeError("a no_cue night never plays a cue")
        self._events.append(event)
        if self.event_log_path is not None:
            append_tmr_scheduler_events((event,), self.event_log_path)
        return event


def append_tmr_scheduler_events(
    events: Iterable[TmrSchedulerEvent],
    output_path: Path,
) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
    return output_path


def load_tmr_scheduler_events(input_path: Path) -> Tuple[TmrSchedulerEvent, ...]:
    input_path = input_path.expanduser()
    if not input_path.exists():
        return ()
    return tuple(
        TmrSchedulerEvent.from_dict(json.loads(line))
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def tlr_cue_ids_of(tlr_block_plan: Optional[TlrBlockPlan]) -> Tuple[str, ...]:
    """The TLR cue of a block plan and of each of its events."""
    if tlr_block_plan is None:
        return ()
    return tuple(dict.fromkeys((tlr_block_plan.cue_id,) + tuple(event.cue_id for event in tlr_block_plan.events)))


def ensure_sham_untrained(
    sham_cue_id: str,
    *,
    catalog: Optional[PuzzleCatalog] = None,
    cue_library: Optional[CueLibrary] = None,
    tlr_cue_ids: Iterable[str] = (),
) -> None:
    """Raise ValueError when the sham cue is a trained one.

    Trained are the cue of any puzzle in ``catalog``, the TLR cues in ``tlr_cue_ids`` (a
    block plan's, from ``tlr_cue_ids_of()``) and every ``tlr`` cue of ``cue_library``.
    With ``cue_library`` a sham cue that sounds like one of them, a generated tone of the
    same frequency or a sound from the same file, is refused too.
    """
    sham_cue_id = str(sham_cue_id or "").strip()
    if not sham_cue_id:
        raise ValueError("the sham cue_id must not be empty")
    trained: Dict[str, str] = {}
    for puzzle in catalog.puzzles if catalog is not None else ():
        if puzzle.cue_id:
            trained.setdefault(puzzle.cue_id, f"the cue of puzzle {puzzle.puzzle_id}")
    for cue_id in tlr_cue_ids:
        trained.setdefault(str(cue_id), "a TLR cue")
    library_cues = {cue.cue_id: cue for cue in cue_library.cues} if cue_library is not None else {}
    for cue in library_cues.values():
        if cue.protocol == "tlr":
            trained.setdefault(cue.cue_id, "a TLR cue")
    if sham_cue_id in trained:
        raise ValueError(
            f"sham cue {sham_cue_id} is {trained[sham_cue_id]}: a sham night needs a sound that was never trained"
        )
    sham = library_cues.get(sham_cue_id)
    if sham is None:
        return
    for cue_id, what in trained.items():
        other = library_cues.get(cue_id)
        if other is None:
            continue
        same_tone = (
            sham.cue_type == other.cue_type == "generated_tone"
            and sham.frequency_hz is not None
            and sham.frequency_hz == other.frequency_hz
        )
        same_file = sham.cue_type == other.cue_type == "sound" and bool(sham.path) and sham.path == other.path
        if same_tone or same_file:
            raise ValueError(
                f"sham cue {sham_cue_id} sounds like {cue_id}, {what}: a sham night needs a sound that was never trained"
            )


def validate_sham_cue(
    sham_cue_id: str,
    *,
    catalog: Optional[PuzzleCatalog],
    cue_library: CueLibrary,
    tlr_cue_ids: Iterable[str] = (),
) -> None:
    """The sham cue is in ``cue_library``, declared ``protocol="sham"``, and untrained."""
    ensure_sham_untrained(sham_cue_id, catalog=catalog, cue_library=cue_library, tlr_cue_ids=tlr_cue_ids)
    cue = cue_library.by_id(sham_cue_id)
    if cue.protocol != "sham":
        raise ValueError(f"sham cue {sham_cue_id} must use protocol='sham' in the cue library")


def _scheduled_puzzles(
    assignment: PuzzleCueAssignment,
    catalog: PuzzleCatalog,
    cue_library: CueLibrary,
) -> Tuple[Tuple[str, str], ...]:
    scheduled = []
    for puzzle_id in assignment.scheduled_puzzle_ids:
        assignment.ensure_schedulable(puzzle_id)
        cue_id = catalog.get_puzzle(puzzle_id).cue_id
        cue = cue_library.by_id(cue_id)
        if cue.protocol != "puzzle":
            raise ValueError(f"scheduled puzzle cue must use protocol='puzzle': {cue_id}")
        scheduled.append((puzzle_id, cue_id))
    return tuple(scheduled)


def _guard_volume_context(
    guard_decision: Optional[ArousalGuardDecision],
) -> Mapping[str, object]:
    if guard_decision is None or not guard_decision.should_lower_volume:
        return {}
    return {
        "arousal_guard_action": guard_decision.action,
        "arousal_guard_reason_codes": list(guard_decision.reason_codes),
        "volume_multiplier": guard_decision.volume_multiplier,
    }


def _cue_metadata(
    *,
    cue_duration_seconds: float,
    cue_volume_hint: Optional[float],
    volume_context: Mapping[str, object],
    extra: Mapping[str, object],
    playback_config: Optional[AudioPlaybackConfig],
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "duration_seconds": cue_duration_seconds,
        **dict(extra),
    }
    if cue_volume_hint is not None:
        metadata["volume_hint"] = cue_volume_hint
    # The hint stays as it is: scaling it here, before the player's cap, lost the
    # multiplier whenever the cap sat below the hint (#255).
    metadata.update(volume_context)
    if playback_config is not None:
        volume_multiplier = float(volume_context.get("volume_multiplier", 1.0))
        metadata.update(
            playback_config.cue_volume(
                cue_volume_hint,
                volume_multiplier=volume_multiplier,
            ).to_dict()
        )
    return metadata


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _unique(reason_codes: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(code for code in reason_codes if code))
