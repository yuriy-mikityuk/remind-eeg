"""Pilot 4 low-volume REM-gated cueing workflow."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from muse_tmr.ble.muse_raw_stream import MuseRawStream

from muse_tmr.audio import (
    AudioBackend,
    AudioCuePlayer,
    AudioPlaybackConfig,
    CueLibrary,
    CuePlaybackResult,
    TestCue,
    VolumeCalibration,
    audio_config_with_calibration,
    resolve_audio_backend,
)
from muse_tmr.data.replay import ReplaySession
from muse_tmr.data.sample_types import MuseFrame
from muse_tmr.features import (
    EpochBuilder,
    EpochConfig,
    epoch_step_seconds,
    extract_eeg_features,
    extract_imu_features,
    extract_ppg_features,
    new_rem_event_tracker,
)
from muse_tmr.models import HeuristicRemDetector, RemGateConfig, StableRemGate
from muse_tmr.protocol import PuzzleCatalog, PuzzleCueAssignment
from muse_tmr.protocol.arousal_guard import ArousalGuard, ArousalGuardConfig
from muse_tmr.protocol.night_schedule import PLAYING_ARMS, NightArmContext
from muse_tmr.protocol.puzzle_protocol import NightPuzzleSession
from muse_tmr.protocol.tmr_scheduler import (
    TmrCueScheduler,
    TmrSchedulerConfig,
    TmrSchedulerEvent,
    tlr_cue_ids_of,
    validate_sham_cue,
)
from muse_tmr.protocol.tlr_protocol import TlrBlockPlan
from muse_tmr.sources.base_source import BaseMuseSource, MuseSourceMetadata
from muse_tmr.validation.gate_log import (
    REM_GATE_DECISIONS_FILENAME,
    gate_decision_record,
    write_rem_gate_decisions,
)
from muse_tmr.validation.pilot3 import arm_context_for, arm_criteria

# 2 adds first_epoch_start, rem_gate_decisions_path and cue_arousal (#227); 3 adds
# protocol, the preset block of the night (#231); 4 adds arm, night_id,
# schedule_commitment_sha256, virtual_play_count and, on a night with an arm, the
# criteria no_play_in_no_cue_arm and arm_matches_sealed_schedule (#228).
PILOT4_SCHEMA_VERSION = 4
# What the sleeper says about an awakening: did a cue wake them?
AWAKENING_ATTRIBUTIONS = ("yes", "no", "unsure")


@dataclass(frozen=True)
class Pilot4Criterion:
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
class Pilot4CueingConfig:
    output_dir: Path
    duration_seconds: float
    source_name: str = "amused"
    allow_short: bool = False
    no_data_timeout_seconds: float = 30.0
    hard_max_volume: float = 0.20
    default_volume: float = 0.02
    fade_in_seconds: float = 0.25
    fade_out_seconds: float = 0.25
    audio_backend_name: str = "dry-run"
    allow_dry_run_fallback: bool = False
    emergency_stop_path: Optional[Path] = None
    epoch_config: EpochConfig = field(default_factory=EpochConfig)
    gate_config: RemGateConfig = field(default_factory=RemGateConfig)
    scheduler_config: TmrSchedulerConfig = field(
        default_factory=lambda: TmrSchedulerConfig(enable_tlr_block=False)
    )
    # None builds the guard default with gate_config.optics_capable.
    arousal_guard_config: Optional[ArousalGuardConfig] = None
    pilot_id: str = "m8_pilot4_low_volume_rem_gated_cueing"
    summary_filename: str = "pilot4_summary.json"
    require_tlr_block: bool = False
    # Score the night's cue-locked arousal proxies at its end (reports.cue_arousal).
    score_cue_arousal: bool = True
    # The protocol preset block (ProtocolPreset.manifest()) metadata.json and the
    # summary record; the configs above are what runs.
    protocol: Mapping[str, object] = field(default_factory=dict)
    # The night's arm (muse_tmr.protocol.night_schedule); None runs the cue arm as every
    # night did before arms, and records no arm.
    night_arm: Optional[NightArmContext] = None

    def validate(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not self.allow_short and not 7200 <= self.duration_seconds <= 28800:
            raise ValueError("pilot4 cueing nights must be between 2 and 8 hours")
        if self.no_data_timeout_seconds <= 0:
            raise ValueError("no_data_timeout_seconds must be positive")
        if not 0.0 <= self.hard_max_volume <= 1.0:
            raise ValueError("hard_max_volume must be between 0.0 and 1.0")
        if not 0.0 <= self.default_volume <= 1.0:
            raise ValueError("default_volume must be between 0.0 and 1.0")
        if not self.pilot_id.strip():
            raise ValueError("pilot_id must not be empty")
        if not self.summary_filename.strip():
            raise ValueError("summary_filename must not be empty")
        if Path(self.summary_filename).name != self.summary_filename:
            raise ValueError("summary_filename must be a file name, not a path")
        # Raises when night_arm and the scheduler config name two different arms.
        arm_context_for(self.night_arm, self.scheduler_config)


@dataclass(frozen=True)
class Pilot4CueingSummary:
    output_dir: str
    raw_path: str
    metadata_path: str
    recording_events_path: str
    scheduler_events_path: str
    arousal_guard_events_path: str
    audio_log_path: str
    awakening_events_path: str
    summary_path: str
    emergency_stop_path: str
    started_at: str
    ended_at: str
    duration_seconds: float
    stop_reason: str
    criteria: Tuple[Pilot4Criterion, ...]
    frame_count: int = 0
    raw_packet_count: int = 0
    epoch_count: int = 0
    modality_counts: Mapping[str, int] = field(default_factory=dict)
    scheduler_event_type_counts: Mapping[str, int] = field(default_factory=dict)
    audio_status_counts: Mapping[str, int] = field(default_factory=dict)
    cue_play_count: int = 0
    puzzle_cue_play_count: int = 0
    tlr_cue_play_count: int = 0
    uncued_puzzle_play_count: int = 0
    max_requested_volume: float = 0.0
    max_effective_volume: float = 0.0
    calibration_device_name: str = ""
    calibration_scheduler_max_volume: float = 0.0
    hard_max_volume: float = 0.20
    audio_backend_name: str = "unknown"
    requested_audio_backend_name: str = "unknown"
    resolved_audio_backend_name: str = "unknown"
    emergency_stop_triggered: bool = False
    tlr_block_required: bool = False
    stride_seconds: float = 30.0
    # Seconds from the first epoch's start to the end of the epoch whose decision first
    # opened the gate or first played a cue; None when that never happened.
    first_gate_open_seconds: Optional[float] = None
    first_play_seconds: Optional[float] = None
    # The first epoch's start in the frames' clock (Unix seconds): what the scheduler's
    # timestamp_seconds count from. None when no epoch was built.
    first_epoch_start: Optional[float] = None
    rem_gate_decisions_path: str = ""
    # The scorer's figures for this night (reports.cue_arousal), or its status.
    cue_arousal: Mapping[str, object] = field(default_factory=dict)
    # The protocol preset block of the night, as in metadata.json.
    protocol: Mapping[str, object] = field(default_factory=dict)
    # The night's arm, its ID and the sealed schedule's commitment; None without an arm.
    arm: Optional[str] = None
    night_id: Optional[str] = None
    schedule_commitment_sha256: Optional[str] = None
    # A no_cue night's skips where another arm would have played.
    virtual_play_count: int = 0
    generated_at_utc: str = ""
    schema_version: int = PILOT4_SCHEMA_VERSION
    pilot_id: str = "m8_pilot4_low_volume_rem_gated_cueing"

    def __post_init__(self) -> None:
        object.__setattr__(self, "criteria", tuple(self.criteria))
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())

    @property
    def passed(self) -> bool:
        return all(criterion.passed for criterion in self.criteria)

    @property
    def failed_criteria(self) -> Tuple[str, ...]:
        return tuple(criterion.name for criterion in self.criteria if not criterion.passed)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pilot_id": self.pilot_id,
            "generated_at_utc": self.generated_at_utc,
            "passed": self.passed,
            "failed_criteria": list(self.failed_criteria),
            "output_dir": self.output_dir,
            "raw_path": self.raw_path,
            "metadata_path": self.metadata_path,
            "recording_events_path": self.recording_events_path,
            "scheduler_events_path": self.scheduler_events_path,
            "arousal_guard_events_path": self.arousal_guard_events_path,
            "audio_log_path": self.audio_log_path,
            "awakening_events_path": self.awakening_events_path,
            "summary_path": self.summary_path,
            "emergency_stop_path": self.emergency_stop_path,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "stop_reason": self.stop_reason,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "frame_count": self.frame_count,
            "raw_packet_count": self.raw_packet_count,
            "epoch_count": self.epoch_count,
            "stride_seconds": self.stride_seconds,
            "first_gate_open_seconds": self.first_gate_open_seconds,
            "first_play_seconds": self.first_play_seconds,
            "first_epoch_start": self.first_epoch_start,
            "rem_gate_decisions_path": self.rem_gate_decisions_path,
            "cue_arousal": dict(self.cue_arousal),
            "protocol": dict(self.protocol) if self.protocol else None,
            "arm": self.arm,
            "night_id": self.night_id,
            "schedule_commitment_sha256": self.schedule_commitment_sha256,
            "modality_counts": dict(self.modality_counts),
            "scheduler_event_type_counts": dict(self.scheduler_event_type_counts),
            "audio_status_counts": dict(self.audio_status_counts),
            "cue_play_count": self.cue_play_count,
            "puzzle_cue_play_count": self.puzzle_cue_play_count,
            "tlr_cue_play_count": self.tlr_cue_play_count,
            "uncued_puzzle_play_count": self.uncued_puzzle_play_count,
            "virtual_play_count": self.virtual_play_count,
            "max_requested_volume": self.max_requested_volume,
            "max_effective_volume": self.max_effective_volume,
            "calibration_device_name": self.calibration_device_name,
            "calibration_scheduler_max_volume": self.calibration_scheduler_max_volume,
            "hard_max_volume": self.hard_max_volume,
            "audio_backend_name": self.audio_backend_name,
            "requested_audio_backend_name": self.requested_audio_backend_name,
            "resolved_audio_backend_name": self.resolved_audio_backend_name,
            "emergency_stop_triggered": self.emergency_stop_triggered,
            "tlr_block_required": self.tlr_block_required,
        }

    def save(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path


@dataclass
class _Pilot4State:
    frame_count: int = 0
    raw_packet_count: int = 0
    epoch_count: int = 0
    modality_counts: Dict[str, int] = field(default_factory=dict)
    playback_results: list = field(default_factory=list)
    stop_reason: str = "duration_complete"
    emergency_stop_triggered: bool = False
    first_epoch_start: Optional[float] = None
    previous_epoch_start: Optional[float] = None
    first_gate_open_seconds: Optional[float] = None
    first_play_seconds: Optional[float] = None


@dataclass(frozen=True)
class AwakeningEvent:
    """A manual awakening marker.

    ``timestamp_seconds`` (optional) places it on the night's time base, seconds from the
    first epoch's start like the scheduler events; without it the cue arousal scorer and
    the protocol fidelity report place it by ``timestamp_utc`` (``night_offset_seconds()``).
    ``attributed_to_cue`` (optional, ``yes``, ``no`` or ``unsure``) is whether the sleeper
    thinks a cue woke them. Rows written before both fields existed load as they are.
    """

    event_type: str = "awakening"
    timestamp_utc: str = ""
    notes: str = ""
    timestamp_seconds: Optional[float] = None
    attributed_to_cue: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            object.__setattr__(self, "timestamp_utc", _utc_now())
        if not self.event_type.strip():
            raise ValueError("event_type must not be empty")
        if self.timestamp_seconds is not None:
            object.__setattr__(self, "timestamp_seconds", float(self.timestamp_seconds))
            if self.timestamp_seconds < 0:
                raise ValueError("timestamp_seconds must be non-negative")
        if self.attributed_to_cue is not None:
            attribution = str(self.attributed_to_cue).strip().lower()
            if attribution not in AWAKENING_ATTRIBUTIONS:
                raise ValueError(
                    f"attributed_to_cue must be one of: {', '.join(AWAKENING_ATTRIBUTIONS)}"
                )
            object.__setattr__(self, "attributed_to_cue", attribution)

    def night_offset_seconds(self, first_epoch_start: Optional[float]) -> Optional[float]:
        """Where the marker falls on the night's time base: ``timestamp_seconds``, else
        ``timestamp_utc`` minus ``first_epoch_start`` (Unix seconds), a time without an
        offset read as UTC. None when neither places it."""
        if self.timestamp_seconds is not None:
            return self.timestamp_seconds
        if first_epoch_start is None:
            return None
        try:
            moment = dt.datetime.fromisoformat(self.timestamp_utc)
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=dt.timezone.utc)
        return moment.timestamp() - first_epoch_start

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "event_type": self.event_type,
            "timestamp_utc": self.timestamp_utc,
            "notes": self.notes,
        }
        if self.timestamp_seconds is not None:
            payload["timestamp_seconds"] = self.timestamp_seconds
        if self.attributed_to_cue is not None:
            payload["attributed_to_cue"] = self.attributed_to_cue
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AwakeningEvent":
        seconds = payload.get("timestamp_seconds")
        attribution = payload.get("attributed_to_cue")
        return cls(
            event_type=str(payload.get("event_type", "awakening")),
            timestamp_utc=str(payload.get("timestamp_utc", "")),
            notes=str(payload.get("notes", "")),
            timestamp_seconds=None if seconds is None else float(seconds),
            attributed_to_cue=None if attribution in (None, "") else str(attribution),
        )


async def run_pilot4_cueing_night(
    source: BaseMuseSource,
    *,
    config: Pilot4CueingConfig,
    catalog: PuzzleCatalog,
    session: NightPuzzleSession,
    assignment: PuzzleCueAssignment,
    cue_library: CueLibrary,
    calibration: VolumeCalibration,
    backend: Optional[AudioBackend] = None,
    tlr_block_plan: Optional[TlrBlockPlan] = None,
) -> Pilot4CueingSummary:
    config.validate()
    # Before connect() and before any file: a night that asked for sound must not
    # run silently. An injected backend (tests, custom players) is used as given.
    resolved_backend = backend
    if resolved_backend is None:
        resolved_backend = resolve_audio_backend(
            config.audio_backend_name,
            allow_dry_run_fallback=config.allow_dry_run_fallback,
        ).backend
    assignment.validate_against_session(session)
    night_arm = arm_context_for(config.night_arm, config.scheduler_config)
    # The sham cue is checked whenever the night names one, whatever its arm, so that a
    # refusal says nothing about the arm.
    _validate_pilot4_cues(
        catalog,
        assignment,
        cue_library,
        tlr_block_plan=tlr_block_plan,
        sham_cue_id=night_arm.sham_cue_id if night_arm is not None else None,
    )
    if config.require_tlr_block and tlr_block_plan is None:
        raise ValueError("Pilot run requires a TLR block plan")
    scheduler_config = config.scheduler_config
    if night_arm is not None:
        scheduler_config = replace(scheduler_config, arm=night_arm.arm, sham_cue_id=night_arm.sham_cue_id)

    output_dir = config.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_amused.bin"
    metadata_path = output_dir / "metadata.json"
    recording_events_path = output_dir / "events.jsonl"
    scheduler_events_path = output_dir / "scheduler_events.jsonl"
    arousal_guard_events_path = output_dir / "arousal_guard_events.jsonl"
    audio_log_path = output_dir / "audio_playback.jsonl"
    awakening_events_path = output_dir / "awakening_events.jsonl"
    rem_gate_decisions_path = output_dir / REM_GATE_DECISIONS_FILENAME
    summary_path = output_dir / config.summary_filename
    emergency_stop_path = (
        config.emergency_stop_path.expanduser()
        if config.emergency_stop_path is not None
        else output_dir / "EMERGENCY_STOP"
    )
    scheduler_events_path.touch(exist_ok=True)
    arousal_guard_events_path.touch(exist_ok=True)
    audio_log_path.touch(exist_ok=True)
    awakening_events_path.touch(exist_ok=True)
    rem_gate_decisions_path.touch(exist_ok=True)

    state = _Pilot4State()
    started_at_dt = dt.datetime.now(dt.timezone.utc)
    started_monotonic = time.monotonic()
    deadline = started_monotonic + config.duration_seconds
    metadata = await source.connect()
    _write_metadata(
        metadata_path,
        metadata,
        config,
        calibration,
        started_at_dt,
        resolved_backend.name,
        tlr_block_plan=tlr_block_plan,
        night_arm=night_arm,
    )
    _append_jsonl(
        recording_events_path,
        {
            "event": "pilot_started",
            "timestamp_utc": started_at_dt.isoformat(),
            "pilot_id": config.pilot_id,
            "source": metadata.source_name,
            "emergency_stop_path": str(emergency_stop_path),
        },
    )

    audio_config = audio_config_with_calibration(
        AudioPlaybackConfig(
            max_volume=config.hard_max_volume,
            default_volume=config.default_volume,
            fade_in_seconds=config.fade_in_seconds,
            fade_out_seconds=config.fade_out_seconds,
            device_name=calibration.device_name,
            log_path=audio_log_path,
        ),
        calibration,
    )
    player = AudioCuePlayer(
        audio_config,
        backend=resolved_backend,
    )
    detector = HeuristicRemDetector()
    gate = StableRemGate(config.gate_config)
    arousal_guard = ArousalGuard(
        config.arousal_guard_config
        or ArousalGuardConfig(optics_capable=config.gate_config.optics_capable),
        event_log_path=arousal_guard_events_path,
    )
    scheduler = TmrCueScheduler(
        assignment=assignment,
        catalog=catalog,
        cue_library=cue_library,
        config=scheduler_config,
        tlr_block_plan=tlr_block_plan,
        event_log_path=scheduler_events_path,
        playback_config=player.config,
    )
    raw_stream = MuseRawStream(str(raw_path))
    raw_stream.open_write()

    try:
        frame_stream = _recording_frame_stream(
            source=source,
            raw_stream=raw_stream,
            state=state,
            deadline=deadline,
            no_data_timeout_seconds=config.no_data_timeout_seconds,
        )
        builder = EpochBuilder(config.epoch_config)
        rem_tracker = new_rem_event_tracker()
        async for epoch in builder.build(frame_stream):
            state.epoch_count += 1
            if state.first_epoch_start is None:
                state.first_epoch_start = epoch.start_time
            _maybe_trigger_emergency_stop(player, state, emergency_stop_path, recording_events_path)
            timestamp_seconds = max(0.0, epoch.start_time - state.first_epoch_start)
            eeg = extract_eeg_features(epoch, rem_tracker=rem_tracker)
            imu = extract_imu_features(epoch)
            ppg = extract_ppg_features(epoch)
            prediction = detector.predict_features(eeg=eeg, imu=imu, ppg=ppg)
            # The gate keeps time in real seconds: the stride, not the epoch length.
            gate_decision = gate.update(
                prediction,
                duration_seconds=epoch_step_seconds(
                    epoch.start_time,
                    state.previous_epoch_start,
                    config.epoch_config.stride_seconds,
                ),
            )
            state.previous_epoch_start = epoch.start_time
            # The decision is dropped once the scheduler has it; the log keeps it for the
            # cue arousal scorer's REM periods and controls.
            write_rem_gate_decisions(
                (
                    gate_decision_record(
                        index=epoch.index,
                        timestamp_seconds=timestamp_seconds,
                        duration_seconds=epoch.duration_seconds,
                        decision=gate_decision,
                    ),
                ),
                rem_gate_decisions_path,
                append=True,
            )
            decision_seconds = epoch.end_time - state.first_epoch_start
            if gate_decision.gate_open and state.first_gate_open_seconds is None:
                state.first_gate_open_seconds = decision_seconds
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
            for event in events:
                if event.event_type == "play":
                    if state.first_play_seconds is None:
                        state.first_play_seconds = decision_seconds
                    # On a worker thread: afplay blocks until the tone ends, and the BLE
                    # client stamps each packet when the event loop takes it, so a tone
                    # played on the loop would stamp every packet of its length late and
                    # the cue arousal scorer would read them as lost.
                    result = await asyncio.to_thread(
                        _play_scheduler_event, event, cue_library, player, arm=scheduler_config.arm
                    )
                    state.playback_results.append(result)
                    _append_jsonl(
                        recording_events_path,
                        {
                            "event": "audio_playback_result",
                            "timestamp_seconds": event.timestamp_seconds,
                            "scheduler_event": event.to_dict(),
                            "playback_result": result.to_dict(),
                        },
                    )
                _maybe_trigger_emergency_stop(player, state, emergency_stop_path, recording_events_path)
    finally:
        raw_stream.close()
        await source.stop()

    ended_at_dt = dt.datetime.now(dt.timezone.utc)
    _append_jsonl(
        recording_events_path,
        {
            "event": "pilot_stopped",
            "timestamp_utc": ended_at_dt.isoformat(),
            "pilot_id": config.pilot_id,
            "reason": state.stop_reason,
        },
    )

    summary = _build_summary(
        config=config,
        output_dir=output_dir,
        raw_path=raw_path,
        metadata_path=metadata_path,
        recording_events_path=recording_events_path,
        scheduler_events_path=scheduler_events_path,
        arousal_guard_events_path=arousal_guard_events_path,
        audio_log_path=audio_log_path,
        awakening_events_path=awakening_events_path,
        rem_gate_decisions_path=rem_gate_decisions_path,
        summary_path=summary_path,
        emergency_stop_path=emergency_stop_path,
        started_at=started_at_dt,
        ended_at=ended_at_dt,
        state=state,
        scheduler_events=scheduler.events,
        calibration=calibration,
        player=player,
        assignment=assignment,
        night_arm=night_arm,
    )
    summary.save(summary_path)
    if config.score_cue_arousal:
        # A replayed source has its own clock in its raw file; this night's raw file
        # holds the same frames clamped to when it was opened.
        replay_input = source.raw_path if isinstance(source, ReplaySession) else raw_path
        block = await _score_own_night(
            output_dir,
            summary_path=summary_path,
            replay_input=replay_input,
            optics_capable=config.gate_config.optics_capable,
        )
        summary = replace(summary, cue_arousal=block)
        summary.save(summary_path)
    return summary


def append_awakening_event(output_path: Path, event: AwakeningEvent) -> Path:
    _append_jsonl(output_path, event.to_dict())
    return output_path.expanduser()


def load_awakening_events(input_path: Path) -> Tuple[AwakeningEvent, ...]:
    """The markers of an ``awakening_events.jsonl``; none when the file is missing."""
    input_path = Path(input_path).expanduser()
    if not input_path.exists():
        return ()
    return tuple(
        AwakeningEvent.from_dict(json.loads(line))
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def audio_for_plays(
    plays: Sequence[Mapping[str, object]],
    audio_path: Optional[Path],
) -> Tuple[Tuple[Optional[Mapping[str, object]], ...], bool, bool]:
    """Each scheduler play's line of ``audio_playback.jsonl``, in order; whether the log
    was found; whether the matching broke off.

    The player logs one line per play, in the order of the plays, and one more for an
    emergency stop. A play whose line names another cue ends the matching: it and the
    plays after it get none, and the mismatch is reported. The cue arousal scorer and
    the protocol fidelity report both read a night's plays this way.
    """
    if audio_path is None or not Path(audio_path).expanduser().exists():
        return tuple(None for _ in plays), False, False
    entries = [
        entry
        for entry in _read_jsonl(Path(audio_path))
        if not (entry.get("cue_id") == "emergency-stop" and entry.get("status") == "stopped")
    ]
    matched: List[Optional[Mapping[str, object]]] = []
    position = 0
    mismatch = False
    for play in plays:
        metadata = dict(play.get("metadata", {}) or {})
        if metadata.get("synthetic_cue"):
            matched.append(None)
            continue
        if mismatch or position >= len(entries) or entries[position].get("cue_id") != play.get("cue_id"):
            mismatch = True
            matched.append(None)
            continue
        matched.append(entries[position])
        position += 1
    real_plays = sum(1 for play in plays if not dict(play.get("metadata", {}) or {}).get("synthetic_cue"))
    if position != len(entries) and real_plays:
        mismatch = True
    return tuple(matched), True, mismatch


async def _score_own_night(
    output_dir: Path,
    *,
    summary_path: Path,
    replay_input: Path,
    optics_capable: bool,
) -> Dict[str, object]:
    """Score the night's cue-locked arousal proxies and return the summary block.

    The scorer reads the logs this night wrote and replays its recording, or the
    recording a replayed source came from. When it fails, the summary keeps the night
    and records why there are no figures.
    """
    try:
        # Imported here: the scorer imports muse_tmr.validation, whose __init__ imports
        # this module.
        from muse_tmr.reports.cue_arousal import score_cue_arousal

        report = await score_cue_arousal(
            output_dir,
            summary_path=summary_path,
            replay_input=replay_input,
            optics_capable=optics_capable,
        )
        return report.summary_block(report.save_outputs(output_dir))
    except Exception as exc:  # noqa: BLE001 - the night's summary must outlive the scorer
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


async def _recording_frame_stream(
    *,
    source: BaseMuseSource,
    raw_stream: MuseRawStream,
    state: _Pilot4State,
    deadline: float,
    no_data_timeout_seconds: float,
) -> AsyncIterator[MuseFrame]:
    stream = source.stream().__aiter__()
    while time.monotonic() < deadline:
        timeout = min(no_data_timeout_seconds, max(0.01, deadline - time.monotonic()))
        try:
            frame = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
        except asyncio.TimeoutError:
            state.stop_reason = (
                "duration_complete" if time.monotonic() >= deadline else "no_data_timeout"
            )
            break
        except StopAsyncIteration:
            state.stop_reason = "source_ended"
            break

        state.frame_count += 1
        for modality in frame.modalities():
            state.modality_counts[modality] = state.modality_counts.get(modality, 0) + 1
        if frame.raw_packet:
            packet_timestamp = dt.datetime.fromtimestamp(frame.timestamp)
            if packet_timestamp < raw_stream.session_start:
                packet_timestamp = raw_stream.session_start
            raw_stream.write_packet(frame.raw_packet, packet_timestamp)
            state.raw_packet_count += 1
        yield frame


def _play_scheduler_event(
    event: TmrSchedulerEvent,
    cue_library: CueLibrary,
    player: AudioCuePlayer,
    *,
    arm: str = "cue",
) -> CuePlaybackResult:
    """Play one scheduler play event of a night run in ``arm``.

    Only the cue and sham arms play, and only events their own scheduler logged; a
    no-cue night never gets here. A sham play is the sham cue's tone for as long and as
    loud as the cue it stands in for: the length and the hint come from the event.
    """
    if arm not in PLAYING_ARMS or event.arm != arm:
        raise ValueError("cues play in the cue and sham arms only, and only that night's own events")
    if event.cue_id is None:
        raise ValueError("scheduler play event is missing cue_id")
    cue = cue_library.by_id(event.cue_id)
    if cue.frequency_hz is None:
        raise ValueError(f"pilot4 cue must include frequency_hz: {cue.cue_id}")
    duration_seconds = cue.duration_seconds
    volume = cue.volume_hint
    if arm == "sham":
        duration_seconds = float(event.metadata.get("duration_seconds", duration_seconds))
        hint = event.metadata.get("volume_hint")
        volume = None if hint is None else float(hint)
    # The cue's hint, or the player's default without one. The player caps it and only
    # then applies a lower_volume multiplier, so every cap still holds and the cue plays
    # at that fraction of its normal volume.
    return player.play_test_cue(
        TestCue(
            cue_id=cue.cue_id,
            frequency_hz=cue.frequency_hz,
            duration_seconds=duration_seconds,
        ),
        volume=volume,
        volume_multiplier=float(event.metadata.get("volume_multiplier", 1.0)),
        arm=arm,
    )


def _maybe_trigger_emergency_stop(
    player: AudioCuePlayer,
    state: _Pilot4State,
    emergency_stop_path: Path,
    recording_events_path: Path,
) -> None:
    if state.emergency_stop_triggered or not emergency_stop_path.exists():
        return
    result = player.emergency_stop()
    state.emergency_stop_triggered = True
    _append_jsonl(
        recording_events_path,
        {
            "event": "emergency_stop_triggered",
            "timestamp_utc": _utc_now(),
            "emergency_stop_path": str(emergency_stop_path),
            "playback_result": result.to_dict(),
        },
    )


def _build_summary(
    *,
    config: Pilot4CueingConfig,
    output_dir: Path,
    raw_path: Path,
    metadata_path: Path,
    recording_events_path: Path,
    scheduler_events_path: Path,
    arousal_guard_events_path: Path,
    audio_log_path: Path,
    awakening_events_path: Path,
    rem_gate_decisions_path: Path,
    summary_path: Path,
    emergency_stop_path: Path,
    started_at: dt.datetime,
    ended_at: dt.datetime,
    state: _Pilot4State,
    scheduler_events: Tuple[TmrSchedulerEvent, ...],
    calibration: VolumeCalibration,
    player: AudioCuePlayer,
    assignment: PuzzleCueAssignment,
    night_arm: Optional[NightArmContext] = None,
) -> Pilot4CueingSummary:
    scheduler_event_type_counts = _event_type_counts(scheduler_events)
    audio_status_counts = _audio_status_counts(state.playback_results)
    cue_play_count = scheduler_event_type_counts.get("play", 0)
    puzzle_cue_play_count = sum(
        1 for event in scheduler_events if event.event_type == "play" and event.protocol == "puzzle"
    )
    tlr_cue_play_count = sum(
        1 for event in scheduler_events if event.event_type == "play" and event.protocol == "tlr"
    )
    uncued_puzzle_play_count = sum(
        1
        for event in scheduler_events
        if event.event_type == "play"
        and event.puzzle_id is not None
        and assignment.is_uncued(event.puzzle_id)
    )
    max_requested_volume = max(
        (result.requested_volume for result in state.playback_results),
        default=0.0,
    )
    max_effective_volume = max(
        (result.effective_volume for result in state.playback_results),
        default=0.0,
    )
    virtual_plays = tuple(event for event in scheduler_events if event.virtual_play)
    # A no_cue night's TLR block counts when its virtual plays show it would have played.
    tlr_block_count = tlr_cue_play_count
    if night_arm is not None and night_arm.arm == "no_cue":
        tlr_block_count = sum(1 for event in virtual_plays if event.protocol == "tlr")
    played_audio_rows = sum(
        1 for row in _read_jsonl(audio_log_path) if row.get("status") == "played"
    )
    criteria = _build_criteria(
        epoch_count=state.epoch_count,
        scheduler_events=scheduler_events,
        cue_play_count=cue_play_count,
        tlr_cue_play_count=tlr_cue_play_count,
        uncued_puzzle_play_count=uncued_puzzle_play_count,
        max_effective_volume=max_effective_volume,
        calibration=calibration,
        hard_max_volume=config.hard_max_volume,
        arousal_guard_events_path=arousal_guard_events_path,
        emergency_stop_path=emergency_stop_path,
        tlr_block_required=config.require_tlr_block,
        tlr_block_count=tlr_block_count,
    ) + arm_criteria(
        night_arm,
        scheduler_plays=cue_play_count,
        played_audio_rows=played_audio_rows,
        criterion=Pilot4Criterion,
    )
    arm_fields = night_arm.summary_fields() if night_arm is not None else {}
    return Pilot4CueingSummary(
        output_dir=str(output_dir),
        raw_path=str(raw_path),
        metadata_path=str(metadata_path),
        recording_events_path=str(recording_events_path),
        scheduler_events_path=str(scheduler_events_path),
        arousal_guard_events_path=str(arousal_guard_events_path),
        audio_log_path=str(audio_log_path),
        awakening_events_path=str(awakening_events_path),
        summary_path=str(summary_path),
        emergency_stop_path=str(emergency_stop_path),
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        duration_seconds=(ended_at - started_at).total_seconds(),
        stop_reason=state.stop_reason,
        criteria=criteria,
        frame_count=state.frame_count,
        raw_packet_count=state.raw_packet_count,
        epoch_count=state.epoch_count,
        modality_counts=state.modality_counts,
        scheduler_event_type_counts=scheduler_event_type_counts,
        audio_status_counts=audio_status_counts,
        cue_play_count=cue_play_count,
        puzzle_cue_play_count=puzzle_cue_play_count,
        tlr_cue_play_count=tlr_cue_play_count,
        uncued_puzzle_play_count=uncued_puzzle_play_count,
        max_requested_volume=max_requested_volume,
        max_effective_volume=max_effective_volume,
        calibration_device_name=calibration.device_name,
        calibration_scheduler_max_volume=calibration.scheduler_max_volume,
        hard_max_volume=config.hard_max_volume,
        audio_backend_name=player.backend.name,
        requested_audio_backend_name=config.audio_backend_name,
        resolved_audio_backend_name=player.backend.name,
        emergency_stop_triggered=state.emergency_stop_triggered,
        tlr_block_required=config.require_tlr_block,
        stride_seconds=config.epoch_config.stride_seconds,
        first_gate_open_seconds=state.first_gate_open_seconds,
        first_play_seconds=state.first_play_seconds,
        first_epoch_start=state.first_epoch_start,
        rem_gate_decisions_path=str(rem_gate_decisions_path),
        protocol=dict(config.protocol),
        arm=arm_fields.get("arm"),
        night_id=arm_fields.get("night_id"),
        schedule_commitment_sha256=arm_fields.get("schedule_commitment_sha256"),
        virtual_play_count=len(virtual_plays),
        pilot_id=config.pilot_id,
    )


def _build_criteria(
    *,
    epoch_count: int,
    scheduler_events: Tuple[TmrSchedulerEvent, ...],
    cue_play_count: int,
    tlr_cue_play_count: int,
    uncued_puzzle_play_count: int,
    max_effective_volume: float,
    calibration: VolumeCalibration,
    hard_max_volume: float,
    arousal_guard_events_path: Path,
    emergency_stop_path: Path,
    tlr_block_required: bool,
    tlr_block_count: Optional[int] = None,
) -> Tuple[Pilot4Criterion, ...]:
    scheduler_plays = tuple(event for event in scheduler_events if event.event_type == "play")
    stable_rem_plays = all("rem_gate_open" in event.reason_codes for event in scheduler_plays)
    criteria = [
        Pilot4Criterion(
            name="epochs_present",
            passed=epoch_count > 0,
            observed=epoch_count,
            target="> 0 live epochs",
        ),
        Pilot4Criterion(
            name="scheduler_events_generated",
            passed=bool(scheduler_events),
            observed=len(scheduler_events),
            target="inspectable scheduler event stream",
        ),
        Pilot4Criterion(
            name="cues_only_after_stable_rem",
            passed=stable_rem_plays,
            observed=cue_play_count,
            target="every scheduler play event includes rem_gate_open",
        ),
        Pilot4Criterion(
            name="no_uncued_puzzle_cues",
            passed=uncued_puzzle_play_count == 0,
            observed=uncued_puzzle_play_count,
            target="zero uncued puzzle play events",
        ),
        Pilot4Criterion(
            name="volume_within_calibration_cap",
            passed=max_effective_volume <= min(calibration.scheduler_max_volume, hard_max_volume),
            observed=max_effective_volume,
            target=f"<= min(calibration={calibration.scheduler_max_volume}, hard_cap={hard_max_volume})",
        ),
        Pilot4Criterion(
            name="arousal_guard_events_logged",
            passed=arousal_guard_events_path.exists(),
            observed=str(arousal_guard_events_path),
            target="arousal guard JSONL exists",
        ),
        Pilot4Criterion(
            name="emergency_stop_available",
            passed=bool(str(emergency_stop_path)),
            observed=str(emergency_stop_path),
            target="operator can create this file to block future playback",
        ),
    ]
    if tlr_block_required:
        block_count = tlr_cue_play_count if tlr_block_count is None else tlr_block_count
        criteria.append(
            Pilot4Criterion(
                name="tlr_block_played",
                passed=block_count > 0,
                observed=block_count,
                target="at least one scheduler play event with protocol=tlr (on a no_cue night, a virtual_play skip)",
            )
        )
    return tuple(criteria)


def _validate_pilot4_cues(
    catalog: PuzzleCatalog,
    assignment: PuzzleCueAssignment,
    cue_library: CueLibrary,
    *,
    tlr_block_plan: Optional[TlrBlockPlan] = None,
    sham_cue_id: Optional[str] = None,
) -> None:
    if sham_cue_id:
        validate_sham_cue(
            sham_cue_id,
            catalog=catalog,
            cue_library=cue_library,
            tlr_cue_ids=tlr_cue_ids_of(tlr_block_plan),
        )
        sham = cue_library.by_id(sham_cue_id)
        if sham.cue_type != "generated_tone" or sham.frequency_hz is None:
            raise ValueError(f"pilot4 plays a generated_tone sham cue with frequency_hz only: {sham_cue_id}")
    for puzzle_id in assignment.scheduled_puzzle_ids:
        cue = cue_library.by_id(catalog.get_puzzle(puzzle_id).cue_id)
        if cue.cue_type != "generated_tone":
            raise ValueError(f"pilot4 supports generated_tone cues only: {cue.cue_id}")
        if cue.frequency_hz is None:
            raise ValueError(f"pilot4 generated cue must include frequency_hz: {cue.cue_id}")
    if tlr_block_plan is None:
        return
    for block_event in tlr_block_plan.events:
        cue = cue_library.by_id(block_event.cue_id)
        if cue.protocol != "tlr":
            raise ValueError(f"TLR block cue must use protocol='tlr': {cue.cue_id}")
        if cue.cue_type != "generated_tone":
            raise ValueError(f"TLR block supports generated_tone cues only: {cue.cue_id}")
        if cue.frequency_hz is None:
            raise ValueError(f"TLR generated cue must include frequency_hz: {cue.cue_id}")


def _write_metadata(
    metadata_path: Path,
    metadata: MuseSourceMetadata,
    config: Pilot4CueingConfig,
    calibration: VolumeCalibration,
    started_at: dt.datetime,
    resolved_backend_name: str,
    *,
    tlr_block_plan: Optional[TlrBlockPlan] = None,
    night_arm: Optional[NightArmContext] = None,
) -> None:
    payload = {
        "started_at": started_at.isoformat(),
        "pilot_id": config.pilot_id,
        "protocol": dict(config.protocol) if config.protocol else None,
        "tlr_block_plan": tlr_block_plan.to_dict() if tlr_block_plan is not None else None,
        "source": {
            "source_name": metadata.source_name,
            "device_name": metadata.device_name,
            "device_id": metadata.device_id,
            "capabilities": dict(metadata.capabilities),
            "metadata": dict(metadata.metadata or {}),
        },
        "audio": {
            "backend_name": config.audio_backend_name,
            "requested_backend_name": config.audio_backend_name,
            "resolved_backend_name": resolved_backend_name,
            "hard_max_volume": config.hard_max_volume,
            "default_volume": config.default_volume,
            "calibration": calibration.to_dict(),
        },
    }
    if night_arm is not None:
        payload["night_arm"] = night_arm.to_dict()
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(output_path: Path, payload: Mapping[str, object]) -> None:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    path = Path(path).expanduser()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _event_type_counts(events: Iterable[TmrSchedulerEvent]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    return counts


def _audio_status_counts(results: Iterable[CuePlaybackResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
