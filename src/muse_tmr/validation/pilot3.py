"""Pilot 3 replay-only cue plan simulation."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from muse_tmr.audio import CueLibrary
from muse_tmr.data.replay import ReplayConfig, ReplaySession
from muse_tmr.features import (
    EpochBuilder,
    EpochConfig,
    SleepEpoch,
    epoch_step_seconds,
    extract_eeg_features,
    extract_imu_features,
    extract_ppg_features,
    new_rem_event_tracker,
)
from muse_tmr.models import (
    HeuristicRemDetector,
    RemGateConfig,
    RemGateDecision,
    RemPrediction,
    StableRemGate,
)
from muse_tmr.protocol import PuzzleCatalog, PuzzleCueAssignment
from muse_tmr.protocol.arousal_guard import (
    ArousalGuard,
    ArousalGuardConfig,
    ArousalGuardDecision,
)
from muse_tmr.protocol.night_schedule import NightArmContext
from muse_tmr.protocol.puzzle_protocol import NightPuzzleSession
from muse_tmr.protocol.tmr_scheduler import (
    TmrCueScheduler,
    TmrSchedulerConfig,
    TmrSchedulerEvent,
    validate_sham_cue,
)

# 2 adds the night's arm, night_id and schedule_commitment_sha256, the metric
# virtual_play_count and, on a run with an arm, the criteria no_play_in_no_cue_arm and
# arm_matches_sealed_schedule (#228).
PILOT3_SCHEMA_VERSION = 2
# What an injected cue is called in its play event (--inject-cues-every-seconds, #227).
SYNTHETIC_CUE = "synthetic_cue"


@dataclass(frozen=True)
class Pilot3Criterion:
    name: str
    passed: bool
    observed: object
    target: str
    message: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "target": self.target,
            "message": self.message,
        }


@dataclass(frozen=True)
class Pilot3EpochResult:
    index: int
    timestamp_seconds: float
    duration_seconds: float
    prediction: RemPrediction
    gate_decision: RemGateDecision
    arousal_guard_decision: ArousalGuardDecision
    scheduler_events: Tuple[TmrSchedulerEvent, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "timestamp_seconds": self.timestamp_seconds,
            "duration_seconds": self.duration_seconds,
            "prediction": self.prediction.to_dict(),
            "gate_decision": self.gate_decision.to_dict(),
            "arousal_guard_decision": self.arousal_guard_decision.to_dict(),
            "scheduler_events": [event.to_dict() for event in self.scheduler_events],
        }


@dataclass(frozen=True)
class Pilot3ReplayCueSimulationReport:
    recording_input: str
    session_id: str
    criteria: Tuple[Pilot3Criterion, ...]
    epoch_results: Tuple[Pilot3EpochResult, ...]
    scheduler_events: Tuple[TmrSchedulerEvent, ...]
    metrics: Mapping[str, object] = field(default_factory=dict)
    audio_backend: str = "mock"
    audio_playback_executed: bool = False
    generated_at_utc: str = ""
    schema_version: int = PILOT3_SCHEMA_VERSION
    pilot_id: str = "m8_pilot3_replay_cue_simulation"
    # play events injected into gate-open epochs to check where the cue arousal scorer
    # puts its windows; never from the scheduler and never in scheduler_events or cue_plan.
    injected_cues: Tuple[TmrSchedulerEvent, ...] = ()
    # The protocol preset block of the run (ProtocolPreset.manifest()), or empty.
    protocol: Mapping[str, object] = field(default_factory=dict)
    # The night's arm, its ID and the sealed schedule's commitment; None without an arm.
    arm: Optional[str] = None
    night_id: Optional[str] = None
    schedule_commitment_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "epoch_results", tuple(self.epoch_results))
        object.__setattr__(self, "scheduler_events", tuple(self.scheduler_events))
        object.__setattr__(self, "injected_cues", tuple(self.injected_cues))
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())

    @property
    def passed(self) -> bool:
        return all(criterion.passed for criterion in self.criteria)

    @property
    def failed_criteria(self) -> Tuple[str, ...]:
        return tuple(criterion.name for criterion in self.criteria if not criterion.passed)

    @property
    def cue_plan(self) -> Tuple[TmrSchedulerEvent, ...]:
        return tuple(event for event in self.scheduler_events if event.event_type == "play")

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pilot_id": self.pilot_id,
            "generated_at_utc": self.generated_at_utc,
            "passed": self.passed,
            "failed_criteria": list(self.failed_criteria),
            "recording_input": self.recording_input,
            "session_id": self.session_id,
            "audio_backend": self.audio_backend,
            "audio_playback_executed": self.audio_playback_executed,
            "protocol": dict(self.protocol) if self.protocol else None,
            "arm": self.arm,
            "night_id": self.night_id,
            "schedule_commitment_sha256": self.schedule_commitment_sha256,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "metrics": dict(self.metrics),
            "cue_plan": [event.to_dict() for event in self.cue_plan],
            "scheduler_events": [event.to_dict() for event in self.scheduler_events],
            "injected_cues": [event.to_dict() for event in self.injected_cues],
            "epochs": [result.to_dict() for result in self.epoch_results],
        }

    def save(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path

    def save_scheduler_events(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(event.to_dict(), sort_keys=True) for event in self.scheduler_events]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return output_path


async def simulate_replay_cue_plan(
    input_path: Path,
    *,
    catalog: PuzzleCatalog,
    session: NightPuzzleSession,
    assignment: PuzzleCueAssignment,
    cue_library: CueLibrary,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
    epoch_config: Optional[EpochConfig] = None,
    gate_config: Optional[RemGateConfig] = None,
    scheduler_config: Optional[TmrSchedulerConfig] = None,
    arousal_guard_config: Optional[ArousalGuardConfig] = None,
    inject_cues_every_seconds: Optional[float] = None,
    protocol: Optional[Mapping[str, object]] = None,
    night_arm: Optional[NightArmContext] = None,
) -> Pilot3ReplayCueSimulationReport:
    epoch_config = epoch_config or EpochConfig()
    replay = ReplaySession(
        ReplayConfig(
            input_path=input_path,
            speed=0.0,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
    )
    await replay.connect()
    try:
        builder = EpochBuilder(epoch_config)
        epochs = [epoch async for epoch in builder.build(replay.stream())]
    finally:
        await replay.stop()

    return simulate_cue_plan_from_epochs(
        epochs,
        catalog=catalog,
        session=session,
        assignment=assignment,
        cue_library=cue_library,
        recording_input=str(replay.recording_dir),
        gate_config=gate_config,
        scheduler_config=scheduler_config,
        arousal_guard_config=arousal_guard_config,
        stride_seconds=epoch_config.stride_seconds,
        inject_cues_every_seconds=inject_cues_every_seconds,
        protocol=protocol,
        night_arm=night_arm,
    )


def simulate_cue_plan_from_epochs(
    epochs: Sequence[SleepEpoch],
    *,
    catalog: PuzzleCatalog,
    session: NightPuzzleSession,
    assignment: PuzzleCueAssignment,
    cue_library: CueLibrary,
    recording_input: str = "",
    detector: Optional[HeuristicRemDetector] = None,
    gate_config: Optional[RemGateConfig] = None,
    scheduler_config: Optional[TmrSchedulerConfig] = None,
    arousal_guard_config: Optional[ArousalGuardConfig] = None,
    stride_seconds: Optional[float] = None,
    inject_cues_every_seconds: Optional[float] = None,
    protocol: Optional[Mapping[str, object]] = None,
    night_arm: Optional[NightArmContext] = None,
) -> Pilot3ReplayCueSimulationReport:
    """Run the detector, gate, guard and scheduler over consecutive epochs of one session.

    The gate moves on by the real time between epoch starts (``epoch_step_seconds``), so
    its stable window and cooldown take as long at a 10 s stride as at 30 s.
    ``stride_seconds`` is the stride the epochs were built with and counts for the first
    epoch; None takes the gap between the first two epoch starts, or the epoch length
    when there is only one.

    ``inject_cues_every_seconds`` adds a ``synthetic_cue`` play event to a gate-open epoch
    whenever that many seconds have passed since the last one, decided like a scheduler
    play at the end of the epoch, so the cue arousal scorer's windows can be checked on a
    replayed night. They go into ``injected_cues`` only: the scheduler never sees them and
    the scheduler events stay as they are without them.

    ``protocol`` is the preset block (``ProtocolPreset.manifest()``) the report records;
    the configs passed here are what runs.

    ``night_arm`` runs the scheduler in that arm (``muse_tmr.protocol.night_schedule``);
    a ``scheduler_config`` whose arm is not ``cue`` does the same. The sham cue is checked
    whenever the run names one, whatever its arm, so a refusal says nothing about it.
    """
    assignment.validate_against_session(session)
    _validate_scheduled_cues(catalog, assignment, cue_library)
    scheduler_config = scheduler_config or TmrSchedulerConfig(enable_tlr_block=False)
    night_arm = arm_context_for(night_arm, scheduler_config)
    if night_arm is not None:
        if night_arm.sham_cue_id:
            validate_sham_cue(night_arm.sham_cue_id, catalog=catalog, cue_library=cue_library)
        scheduler_config = replace(scheduler_config, arm=night_arm.arm, sham_cue_id=night_arm.sham_cue_id)
    stride_seconds = _resolve_stride_seconds(epochs, stride_seconds)
    if inject_cues_every_seconds is not None and inject_cues_every_seconds <= 0:
        raise ValueError("inject_cues_every_seconds must be positive")

    detector = detector or HeuristicRemDetector()
    gate = StableRemGate(gate_config or RemGateConfig())
    arousal_guard = ArousalGuard(
        arousal_guard_config
        or ArousalGuardConfig(optics_capable=gate.config.optics_capable)
    )
    scheduler = TmrCueScheduler(
        assignment=assignment,
        catalog=catalog,
        cue_library=cue_library,
        config=scheduler_config,
    )

    first_start_time = epochs[0].start_time if epochs else 0.0
    epoch_results = []
    scheduler_events = []
    injected_cues = []
    last_injected_seconds: Optional[float] = None
    rem_tracker = new_rem_event_tracker()
    previous_start_time = None

    for epoch in epochs:
        timestamp_seconds = max(0.0, epoch.start_time - first_start_time)
        eeg = extract_eeg_features(epoch, rem_tracker=rem_tracker)
        imu = extract_imu_features(epoch)
        ppg = extract_ppg_features(epoch)
        prediction = detector.predict_features(eeg=eeg, imu=imu, ppg=ppg)
        gate_decision = gate.update(
            prediction,
            duration_seconds=epoch_step_seconds(
                epoch.start_time, previous_start_time, stride_seconds
            ),
        )
        previous_start_time = epoch.start_time
        guard_decision = arousal_guard.evaluate(
            timestamp_seconds=timestamp_seconds,
            eeg=eeg,
            imu=imu,
            ppg=ppg,
        )
        events = scheduler.update(
            gate_decision,
            timestamp_seconds=timestamp_seconds,
            guard_decision=guard_decision,
        )
        scheduler_events.extend(events)
        if inject_cues_every_seconds is not None and gate_decision.gate_open:
            decision_seconds = timestamp_seconds + epoch.duration_seconds
            if (
                last_injected_seconds is None
                or decision_seconds - last_injected_seconds >= inject_cues_every_seconds - 1e-6
            ):
                injected_cues.append(
                    _injected_cue(timestamp_seconds, decision_seconds, epoch.index, inject_cues_every_seconds)
                )
                last_injected_seconds = decision_seconds
        epoch_results.append(
            Pilot3EpochResult(
                index=epoch.index,
                timestamp_seconds=timestamp_seconds,
                duration_seconds=epoch.duration_seconds,
                prediction=prediction,
                gate_decision=gate_decision,
                arousal_guard_decision=guard_decision,
                scheduler_events=events,
            )
        )

    scheduler_events_tuple = tuple(scheduler_events)
    metrics = _build_metrics(
        epochs=epochs,
        epoch_results=tuple(epoch_results),
        scheduler_events=scheduler_events_tuple,
        assignment=assignment,
        stride_seconds=stride_seconds,
    )
    metrics["injected_cue_count"] = len(injected_cues)
    metrics["inject_cues_every_seconds"] = inject_cues_every_seconds
    criteria = _build_criteria(metrics, audio_playback_executed=False) + arm_criteria(
        night_arm, scheduler_plays=int(metrics.get("cue_plan_count", 0)), criterion=Pilot3Criterion
    )
    arm_fields = night_arm.summary_fields() if night_arm is not None else {}
    return Pilot3ReplayCueSimulationReport(
        recording_input=recording_input,
        session_id=session.session_id,
        criteria=criteria,
        epoch_results=tuple(epoch_results),
        scheduler_events=scheduler_events_tuple,
        metrics=metrics,
        audio_backend="mock",
        audio_playback_executed=False,
        injected_cues=tuple(injected_cues),
        protocol=dict(protocol or {}),
        arm=arm_fields.get("arm"),
        night_id=arm_fields.get("night_id"),
        schedule_commitment_sha256=arm_fields.get("schedule_commitment_sha256"),
    )


def arm_context_for(
    night_arm: Optional[NightArmContext],
    scheduler_config: TmrSchedulerConfig,
) -> Optional[NightArmContext]:
    """The night's arm: ``night_arm``, else one made from a scheduler config set to
    another arm than ``cue``, else None. Two different arms raise."""
    if night_arm is None:
        if scheduler_config.arm == "cue":
            return None
        return NightArmContext(arm=scheduler_config.arm, sham_cue_id=scheduler_config.sham_cue_id)
    if scheduler_config.arm not in ("cue", night_arm.arm):
        raise ValueError("the scheduler config and night_arm name two different arms")
    return night_arm


def arm_criteria(
    night_arm: Optional[NightArmContext],
    *,
    scheduler_plays: int,
    criterion,
    played_audio_rows: Optional[int] = None,
) -> Tuple[object, ...]:
    """The criteria a run with an arm adds, built with ``criterion`` (the pilot's
    criterion class): ``no_play_in_no_cue_arm`` for any arm, and with a sealed schedule
    ``arm_matches_sealed_schedule``. None, without an arm."""
    if night_arm is None:
        return ()
    observed: Dict[str, object] = {"scheduler_plays": scheduler_plays}
    if played_audio_rows is not None:
        observed["played_audio_rows"] = played_audio_rows
    quiet = scheduler_plays == 0 and not played_audio_rows
    criteria = [
        criterion(
            name="no_play_in_no_cue_arm",
            passed=night_arm.arm != "no_cue" or quiet,
            observed=observed,
            target="in the no_cue arm, zero scheduler play events and zero played audio rows",
        )
    ]
    if night_arm.from_schedule:
        check = night_arm.schedule_check()
        criteria.append(
            criterion(
                name="arm_matches_sealed_schedule",
                passed=bool(check.get("commitment_matches") and check.get("arm_matches")),
                observed=check,
                target="the arm the sealed night schedule gives this night, under an unchanged commitment",
            )
        )
    return tuple(criteria)


def _injected_cue(
    timestamp_seconds: float,
    decision_seconds: float,
    epoch_index: int,
    every_seconds: float,
) -> TmrSchedulerEvent:
    """A play event for the cue arousal scorer, tagged so it never reads as a real cue."""
    return TmrSchedulerEvent(
        event_type="play",
        timestamp_seconds=timestamp_seconds,
        cue_id=SYNTHETIC_CUE,
        protocol="synthetic",
        reason_codes=("rem_gate_open", SYNTHETIC_CUE),
        metadata={
            SYNTHETIC_CUE: True,
            "decision_seconds": decision_seconds,
            "epoch_index": epoch_index,
            "inject_cues_every_seconds": every_seconds,
            "audio_backend": "mock",
        },
    )


def _validate_scheduled_cues(
    catalog: PuzzleCatalog,
    assignment: PuzzleCueAssignment,
    cue_library: CueLibrary,
) -> None:
    for puzzle_id in assignment.scheduled_puzzle_ids:
        assignment.ensure_schedulable(puzzle_id)
        cue_library.by_id(catalog.get_puzzle(puzzle_id).cue_id)


def _resolve_stride_seconds(
    epochs: Sequence[SleepEpoch],
    stride_seconds: Optional[float],
) -> float:
    if stride_seconds is not None:
        if stride_seconds <= 0:
            raise ValueError("stride_seconds must be positive")
        return float(stride_seconds)
    if len(epochs) >= 2:
        gap = round(epochs[1].start_time - epochs[0].start_time, 6)
        if gap <= 0:
            raise ValueError("epoch start times must increase")
        return gap
    if epochs:
        return epochs[0].duration_seconds
    return EpochConfig().stride_seconds


def _decision_seconds(result: Pilot3EpochResult) -> float:
    """Seconds from the first epoch's start to the end of this one, when its decision
    exists; the scheduler events carry the epoch start instead."""
    return result.timestamp_seconds + result.duration_seconds


def _build_metrics(
    *,
    epochs: Sequence[SleepEpoch],
    epoch_results: Tuple[Pilot3EpochResult, ...],
    scheduler_events: Tuple[TmrSchedulerEvent, ...],
    assignment: PuzzleCueAssignment,
    stride_seconds: float,
) -> Dict[str, object]:
    event_type_counts: Dict[str, int] = {}
    play_by_puzzle: Dict[str, int] = {}
    virtual_play_count = 0
    for event in scheduler_events:
        event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        if event.event_type == "play" and event.puzzle_id is not None:
            play_by_puzzle[event.puzzle_id] = play_by_puzzle.get(event.puzzle_id, 0) + 1
        virtual_play_count += int(event.virtual_play)

    uncued_plays = {
        puzzle_id: count
        for puzzle_id, count in play_by_puzzle.items()
        if assignment.is_uncued(puzzle_id)
    }
    gate_open_count = sum(1 for result in epoch_results if result.gate_decision.gate_open)
    cue_plan = tuple(event for event in scheduler_events if event.event_type == "play")
    first_gate_open = next(
        (result for result in epoch_results if result.gate_decision.gate_open), None
    )
    first_play = next(
        (
            result
            for result in epoch_results
            if any(event.event_type == "play" for event in result.scheduler_events)
        ),
        None,
    )
    return {
        "epoch_count": len(epochs),
        "stride_seconds": stride_seconds,
        # The first epoch's start in the frames' clock (Unix seconds), what every
        # timestamp_seconds here counts from; None without epochs.
        "first_epoch_start": epochs[0].start_time if epochs else None,
        "first_gate_open_seconds": (
            _decision_seconds(first_gate_open) if first_gate_open is not None else None
        ),
        "first_play_seconds": _decision_seconds(first_play) if first_play is not None else None,
        "gate_open_count": gate_open_count,
        "event_type_counts": event_type_counts,
        "scheduler_event_count": len(scheduler_events),
        "cue_plan_count": len(cue_plan),
        # A no_cue night's skips where another arm would have played.
        "virtual_play_count": virtual_play_count,
        "play_by_puzzle": play_by_puzzle,
        "cued_puzzle_ids": list(assignment.cued_puzzle_ids),
        "uncued_puzzle_ids": list(assignment.uncued_puzzle_ids),
        "uncued_puzzle_play_count": sum(uncued_plays.values()),
        "uncued_puzzle_plays": uncued_plays,
    }


def _build_criteria(
    metrics: Mapping[str, object],
    *,
    audio_playback_executed: bool,
) -> Tuple[Pilot3Criterion, ...]:
    return (
        Pilot3Criterion(
            name="epochs_present",
            passed=int(metrics.get("epoch_count", 0)) > 0,
            observed=metrics.get("epoch_count", 0),
            target="> 0 replay epochs",
        ),
        Pilot3Criterion(
            name="audio_backend_mocked",
            passed=not audio_playback_executed,
            observed=audio_playback_executed,
            target="audio_playback_executed is false",
        ),
        Pilot3Criterion(
            name="no_uncued_puzzle_cues",
            passed=int(metrics.get("uncued_puzzle_play_count", 0)) == 0,
            observed=metrics.get("uncued_puzzle_plays", {}),
            target="scheduler play events only target cued puzzles",
        ),
        Pilot3Criterion(
            name="scheduler_events_generated",
            passed=int(metrics.get("scheduler_event_count", 0)) > 0,
            observed=metrics.get("scheduler_event_count", 0),
            target="inspectable scheduler event stream",
        ),
    )


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
