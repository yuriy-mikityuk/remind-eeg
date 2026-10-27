"""Targeted lucidity reactivation cue protocol components."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from muse_tmr.audio import (
    AudioCuePlayer,
    CueLibrary,
    CueMetadata,
    CuePlaybackResult,
    TestCue,
)

# The schema of a TLR cue's metadata, and of plans and trainings written before #264.
TLR_PROTOCOL_SCHEMA_VERSION = 1
# 2 adds ``tone``, the TLR tone a plan was made with or a training played (#264).
TLR_BLOCK_PLAN_SCHEMA_VERSION = 2
TLR_TRAINING_SCHEMA_VERSION = 2
DEFAULT_TLR_CUE_ID = "tlr_soft_tone"
# Two tones this close in frequency and length are the same tone.
_TONE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TlrTone:
    """The TLR tone a block plan was made with, or a training played (#264).

    ``frequency_hz`` is None for a TLR cue that is not a generated tone. ``library_id``
    is the cue library the cue came from, "" when not known.
    """

    cue_id: str
    frequency_hz: Optional[float]
    duration_seconds: float
    library_id: str = ""

    @classmethod
    def of_cue(cls, cue: CueMetadata, *, library_id: str = "") -> "TlrTone":
        return cls(
            cue_id=cue.cue_id,
            frequency_hz=cue.frequency_hz,
            duration_seconds=cue.duration_seconds,
            library_id=library_id,
        )

    def differs_from(self, cue: CueMetadata) -> bool:
        """Whether ``cue`` sounds otherwise: another frequency or another length."""
        return not (
            _same(self.frequency_hz, cue.frequency_hz) and _same(self.duration_seconds, cue.duration_seconds)
        )

    def describe(self) -> str:
        return describe_tone(self.frequency_hz, self.duration_seconds)

    def to_dict(self) -> Dict[str, object]:
        return {
            "cue_id": self.cue_id,
            "frequency_hz": self.frequency_hz,
            "duration_seconds": self.duration_seconds,
            "library_id": self.library_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TlrTone":
        frequency = payload.get("frequency_hz")
        return cls(
            cue_id=str(payload["cue_id"]),
            frequency_hz=None if frequency is None else float(frequency),
            duration_seconds=float(payload["duration_seconds"]),
            library_id=str(payload.get("library_id", "") or ""),
        )


def describe_tone(frequency_hz: Optional[float], duration_seconds: float) -> str:
    """'600 Hz for 0.65 s', or 'a sound of 1 s' without a frequency."""
    if frequency_hz is None:
        return f"a sound of {duration_seconds:g} s"
    return f"{frequency_hz:g} Hz for {duration_seconds:g} s"


@dataclass(frozen=True)
class TlrCueConfig:
    cue_id: str = DEFAULT_TLR_CUE_ID
    frequency_hz: float = 396.0
    duration_seconds: float = 1.0
    volume_hint: float = 0.05
    description: str = "Default soft generated tone for TLR pre-sleep training and REM blocks."

    def __post_init__(self) -> None:
        if not self.cue_id.strip():
            raise ValueError("cue_id must not be empty")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not 0.0 <= self.volume_hint <= 1.0:
            raise ValueError("volume_hint must be between 0.0 and 1.0")


@dataclass(frozen=True)
class TlrTrainingConfig:
    repetitions: int = 3
    interval_seconds: float = 2.0
    volume: Optional[float] = None
    backend_name: str = "dry-run"

    def __post_init__(self) -> None:
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        if self.volume is not None and not 0.0 <= self.volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")
        if not self.backend_name.strip():
            raise ValueError("backend_name must not be empty")

    def to_dict(self) -> Dict[str, object]:
        return {
            "repetitions": self.repetitions,
            "interval_seconds": self.interval_seconds,
            "volume": self.volume,
            "backend_name": self.backend_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TlrTrainingConfig":
        volume = payload.get("volume")
        return cls(
            repetitions=int(payload.get("repetitions", 3)),
            interval_seconds=float(payload.get("interval_seconds", 2.0)),
            volume=None if volume is None else float(volume),
            backend_name=str(payload.get("backend_name", "dry-run")),
        )


@dataclass(frozen=True)
class TlrTrainingEvent:
    event_type: str
    cue_id: str
    repetition_index: int
    scheduled_offset_seconds: float
    playback_status: str
    requested_volume: float
    effective_volume: float
    reason_codes: Tuple[str, ...] = ()
    occurred_at_utc: str = ""

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type must not be empty")
        if not self.cue_id.strip():
            raise ValueError("cue_id must not be empty")
        if self.repetition_index <= 0:
            raise ValueError("repetition_index must be positive")
        if self.scheduled_offset_seconds < 0:
            raise ValueError("scheduled_offset_seconds must be non-negative")
        if not self.occurred_at_utc:
            object.__setattr__(self, "occurred_at_utc", _utc_now())
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_type": self.event_type,
            "cue_id": self.cue_id,
            "repetition_index": self.repetition_index,
            "scheduled_offset_seconds": self.scheduled_offset_seconds,
            "playback_status": self.playback_status,
            "requested_volume": self.requested_volume,
            "effective_volume": self.effective_volume,
            "reason_codes": list(self.reason_codes),
            "occurred_at_utc": self.occurred_at_utc,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TlrTrainingEvent":
        return cls(
            event_type=str(payload["event_type"]),
            cue_id=str(payload["cue_id"]),
            repetition_index=int(payload["repetition_index"]),
            scheduled_offset_seconds=float(payload["scheduled_offset_seconds"]),
            playback_status=str(payload["playback_status"]),
            requested_volume=float(payload["requested_volume"]),
            effective_volume=float(payload["effective_volume"]),
            reason_codes=tuple(str(item) for item in payload.get("reason_codes", ())),
            occurred_at_utc=str(payload.get("occurred_at_utc", "")),
        )


@dataclass(frozen=True)
class TlrTrainingSession:
    session_id: str
    cue_id: str
    config: TlrTrainingConfig
    events: Tuple[TlrTrainingEvent, ...]
    started_at_utc: str = ""
    completed_at_utc: str = ""
    schema_version: int = TLR_TRAINING_SCHEMA_VERSION
    # The tone the training played; None in a training saved before schema 2.
    tone: Optional[TlrTone] = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not self.cue_id.strip():
            raise ValueError("cue_id must not be empty")
        object.__setattr__(self, "events", tuple(self.events))
        if not self.started_at_utc:
            object.__setattr__(self, "started_at_utc", _utc_now())
        if not self.completed_at_utc:
            object.__setattr__(self, "completed_at_utc", self.started_at_utc)

    @property
    def event_count(self) -> int:
        return len(self.events)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "cue_id": self.cue_id,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "config": self.config.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }
        if self.tone is not None:
            payload["tone"] = self.tone.to_dict()
        return payload

    def save(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TlrTrainingSession":
        return cls(
            schema_version=int(payload.get("schema_version", TLR_PROTOCOL_SCHEMA_VERSION)),
            session_id=str(payload["session_id"]),
            cue_id=str(payload["cue_id"]),
            started_at_utc=str(payload.get("started_at_utc", "")),
            completed_at_utc=str(payload.get("completed_at_utc", "")),
            config=TlrTrainingConfig.from_dict(dict(payload.get("config", {}))),
            events=tuple(
                TlrTrainingEvent.from_dict(item)
                for item in payload.get("events", ())
            ),
            tone=_tone_from(payload.get("tone")),
        )

    @classmethod
    def load(cls, input_path: Path) -> "TlrTrainingSession":
        return cls.from_dict(json.loads(input_path.expanduser().read_text(encoding="utf-8")))


@dataclass(frozen=True)
class TlrBlockConfig:
    enabled: bool = True
    repetitions: int = 3
    interval_seconds: float = 8.0
    post_block_pause_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.repetitions < 0:
            raise ValueError("repetitions must be non-negative")
        if self.interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        if self.post_block_pause_seconds < 0:
            raise ValueError("post_block_pause_seconds must be non-negative")
        if self.enabled and self.repetitions == 0:
            raise ValueError("enabled TLR block requires at least one repetition")

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "repetitions": self.repetitions,
            "interval_seconds": self.interval_seconds,
            "post_block_pause_seconds": self.post_block_pause_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TlrBlockConfig":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            repetitions=int(payload.get("repetitions", 3)),
            interval_seconds=float(payload.get("interval_seconds", 8.0)),
            post_block_pause_seconds=float(payload.get("post_block_pause_seconds", 10.0)),
        )


@dataclass(frozen=True)
class TlrBlockEvent:
    event_type: str
    cue_id: str
    offset_seconds: float
    duration_seconds: float

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type must not be empty")
        if not self.cue_id.strip():
            raise ValueError("cue_id must not be empty")
        if self.offset_seconds < 0:
            raise ValueError("offset_seconds must be non-negative")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_type": self.event_type,
            "cue_id": self.cue_id,
            "offset_seconds": self.offset_seconds,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TlrBlockEvent":
        return cls(
            event_type=str(payload["event_type"]),
            cue_id=str(payload["cue_id"]),
            offset_seconds=float(payload["offset_seconds"]),
            duration_seconds=float(payload["duration_seconds"]),
        )


@dataclass(frozen=True)
class TlrBlockPlan:
    cue_id: str
    config: TlrBlockConfig
    events: Tuple[TlrBlockEvent, ...]
    total_duration_seconds: float
    schema_version: int = TLR_BLOCK_PLAN_SCHEMA_VERSION
    # The protocol preset block the plan was made with ("protocol") and the training it
    # was checked against ("training"), among others.
    metadata: Mapping[str, object] = field(default_factory=dict)
    # The tone the plan was made with; None in a plan saved before schema 2, whose
    # events keep only its length.
    tone: Optional[TlrTone] = None

    def __post_init__(self) -> None:
        if not self.cue_id.strip():
            raise ValueError("cue_id must not be empty")
        object.__setattr__(self, "events", tuple(self.events))
        if self.total_duration_seconds < 0:
            raise ValueError("total_duration_seconds must be non-negative")

    @property
    def puzzle_cue_start_offset_seconds(self) -> float:
        return self.total_duration_seconds

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "schema_version": self.schema_version,
            "cue_id": self.cue_id,
            "config": self.config.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "total_duration_seconds": self.total_duration_seconds,
            "puzzle_cue_start_offset_seconds": self.puzzle_cue_start_offset_seconds,
            "metadata": dict(self.metadata),
        }
        if self.tone is not None:
            payload["tone"] = self.tone.to_dict()
        return payload

    def save(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TlrBlockPlan":
        return cls(
            schema_version=int(payload.get("schema_version", TLR_PROTOCOL_SCHEMA_VERSION)),
            cue_id=str(payload["cue_id"]),
            config=TlrBlockConfig.from_dict(dict(payload.get("config", {}))),
            events=tuple(
                TlrBlockEvent.from_dict(item)
                for item in payload.get("events", ())
            ),
            total_duration_seconds=float(payload.get("total_duration_seconds", 0.0)),
            metadata=dict(payload.get("metadata", {}) or {}),
            tone=_tone_from(payload.get("tone")),
        )

    @classmethod
    def load(cls, input_path: Path) -> "TlrBlockPlan":
        return cls.from_dict(json.loads(input_path.expanduser().read_text(encoding="utf-8")))


def default_tlr_cue(
    config: Optional[TlrCueConfig] = None,
    *,
    protocol: Optional[Mapping[str, object]] = None,
) -> CueMetadata:
    """The generated TLR cue; ``protocol`` is the preset block it was made with."""
    config = config or TlrCueConfig()
    metadata: Dict[str, object] = {"schema_version": TLR_PROTOCOL_SCHEMA_VERSION}
    if protocol:
        metadata["protocol"] = dict(protocol)
    return CueMetadata(
        cue_id=config.cue_id,
        cue_type="generated_tone",
        protocol="tlr",
        duration_seconds=config.duration_seconds,
        frequency_hz=config.frequency_hz,
        volume_hint=config.volume_hint,
        tags=("tlr", "generated", "default"),
        description=config.description,
        metadata=metadata,
    )


def default_tlr_cue_library(
    config: Optional[TlrCueConfig] = None,
    *,
    protocol: Optional[Mapping[str, object]] = None,
) -> CueLibrary:
    cue = default_tlr_cue(config, protocol=protocol)
    return CueLibrary(library_id="tlr_default", cues=(cue,))


def train_tlr_cue(
    cue: CueMetadata,
    player: AudioCuePlayer,
    *,
    config: Optional[TlrTrainingConfig] = None,
    session_id: str = "tlr-training",
    event_log_path: Optional[Path] = None,
    library_id: str = "",
) -> TlrTrainingSession:
    """Play ``cue`` before sleep; the session records its tone and ``library_id``."""
    config = config or TlrTrainingConfig()
    _validate_training_cue(cue)
    # As in Pilot 4 and 5: the cue's hint, or the player's default (None) without one.
    volume = config.volume if config.volume is not None else cue.volume_hint
    test_cue = TestCue(
        cue_id=cue.cue_id,
        frequency_hz=float(cue.frequency_hz),
        duration_seconds=cue.duration_seconds,
    )

    started_at = _utc_now()
    events = []
    for index in range(1, config.repetitions + 1):
        result = player.play_test_cue(test_cue, volume=volume)
        event = _training_event_from_result(
            result,
            repetition_index=index,
            scheduled_offset_seconds=(index - 1) * config.interval_seconds,
        )
        events.append(event)
    completed_at = _utc_now()

    session = TlrTrainingSession(
        session_id=session_id,
        cue_id=cue.cue_id,
        config=config,
        events=tuple(events),
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        tone=TlrTone.of_cue(cue, library_id=library_id),
    )
    if event_log_path is not None:
        write_tlr_events(session.events, event_log_path)
    return session


def plan_tlr_block(
    cue: CueMetadata,
    *,
    config: Optional[TlrBlockConfig] = None,
    metadata: Optional[Mapping[str, object]] = None,
    library_id: str = "",
) -> TlrBlockPlan:
    """The REM TLR block for ``cue``; ``metadata`` (the preset block, say) goes into the
    plan, and so does the cue's tone, with ``library_id``, the library it came from."""
    config = config or TlrBlockConfig()
    metadata = dict(metadata or {})
    if cue.protocol != "tlr":
        raise ValueError("TLR block cue must use protocol='tlr'")
    tone = TlrTone.of_cue(cue, library_id=library_id)
    if not config.enabled:
        return TlrBlockPlan(
            cue_id=cue.cue_id,
            config=config,
            events=(),
            total_duration_seconds=0.0,
            metadata=metadata,
            tone=tone,
        )

    events = tuple(
        TlrBlockEvent(
            event_type="tlr_cue",
            cue_id=cue.cue_id,
            offset_seconds=index * config.interval_seconds,
            duration_seconds=cue.duration_seconds,
        )
        for index in range(config.repetitions)
    )
    last_event_offset = events[-1].offset_seconds if events else 0.0
    total_duration = last_event_offset + cue.duration_seconds + config.post_block_pause_seconds
    return TlrBlockPlan(
        cue_id=cue.cue_id,
        config=config,
        events=events,
        total_duration_seconds=total_duration,
        metadata=metadata,
        tone=tone,
    )


def tlr_tone_mismatch(plan: TlrBlockPlan, cue_library: CueLibrary) -> Optional[str]:
    """Why a night with ``cue_library`` would not play the TLR tone ``plan`` was made
    with, or None (#264).

    The night plays its own library's cue under each block event's cue ID, for that
    cue's length. A plan with a tone (schema 2) is checked on its frequency and length;
    an older plan only on the length its events keep. A cue ID the library lacks raises
    KeyError, as the night would.
    """
    preset = _preset_name(plan.metadata)
    for cue_id in dict.fromkeys(event.cue_id for event in plan.events):
        cue = cue_library.by_id(cue_id)
        night = describe_tone(cue.frequency_hz, cue.duration_seconds)
        tone = plan.tone if plan.tone is not None and plan.tone.cue_id == cue_id else None
        if tone is not None:
            if not tone.differs_from(cue):
                continue
            made_from = f" from cue library {tone.library_id}" if tone.library_id else ""
            return (
                f"the night's cue library has {cue_id} at {night}, but the TLR block plan was made with "
                f"{tone.describe()}{made_from}: the night would play another TLR tone than the one planned and "
                f"trained. Train, plan and run the night from one cue library: {_library_fix(preset)}, then run "
                "train-tlr-cue and plan-tlr-block on it again."
            )
        lengths = sorted({event.duration_seconds for event in plan.events if event.cue_id == cue_id})
        if all(_same(cue.duration_seconds, length) for length in lengths):
            continue
        planned = " and ".join(f"{length:g} s" for length in lengths)
        return (
            f"the night's cue library has {cue_id} at {night}, but the TLR block plan, which records no tone "
            f"(schema 1, made before #264), was made with a cue of {planned}. Plan the block again from the "
            "night's cue library, after training on that library."
        )
    return None


def training_tone_mismatch(training: TlrTrainingSession, cue: CueMetadata) -> Optional[str]:
    """Why ``cue`` is not the tone ``training`` played, or None (#264)."""
    if training.cue_id != cue.cue_id:
        return (
            f"the training played {training.cue_id}, not {cue.cue_id}: plan the block for the cue that was "
            f"trained (--cue-id {training.cue_id})"
        )
    if training.tone is None:
        return (
            f"the training summary {training.session_id} records no tone (made before #264): run "
            "train-tlr-cue again, or leave out --training"
        )
    if not training.tone.differs_from(cue):
        return None
    made_from = f" from cue library {training.tone.library_id}" if training.tone.library_id else ""
    return (
        f"the training played {cue.cue_id} at {training.tone.describe()}{made_from}, but this cue library has "
        f"it at {describe_tone(cue.frequency_hz, cue.duration_seconds)}. Train and plan from one cue library: "
        "the one the night will use."
    )


def load_tlr_training_session(input_path: Path) -> TlrTrainingSession:
    return TlrTrainingSession.load(input_path)


def load_tlr_block_plan(input_path: Path) -> TlrBlockPlan:
    return TlrBlockPlan.load(input_path)


def write_tlr_events(events: Tuple[TlrTrainingEvent, ...], output_path: Path) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
    return output_path


def _validate_training_cue(cue: CueMetadata) -> None:
    if cue.protocol != "tlr":
        raise ValueError("TLR training cue must use protocol='tlr'")
    if cue.cue_type != "generated_tone":
        raise ValueError("TLR pre-sleep training currently supports generated_tone cues")
    if cue.frequency_hz is None or cue.frequency_hz <= 0:
        raise ValueError("generated TLR cue requires frequency_hz")


def _training_event_from_result(
    result: CuePlaybackResult,
    *,
    repetition_index: int,
    scheduled_offset_seconds: float,
) -> TlrTrainingEvent:
    return TlrTrainingEvent(
        event_type="tlr_training_cue",
        cue_id=result.cue_id,
        repetition_index=repetition_index,
        scheduled_offset_seconds=scheduled_offset_seconds,
        playback_status=result.status,
        requested_volume=result.requested_volume,
        effective_volume=result.effective_volume,
        reason_codes=result.reason_codes,
        occurred_at_utc=result.started_at or _utc_now(),
    )


def _tone_from(payload: object) -> Optional[TlrTone]:
    return TlrTone.from_dict(payload) if isinstance(payload, Mapping) and payload else None


def _same(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=_TONE_TOLERANCE)


def _preset_name(metadata: Mapping[str, object]) -> Optional[str]:
    protocol = metadata.get("protocol") if isinstance(metadata, Mapping) else None
    name = protocol.get("name") if isinstance(protocol, Mapping) else None
    return str(name) if name else None


def _library_fix(preset: Optional[str]) -> str:
    if preset:
        return f"create-cue-library --protocol-preset {preset} makes one with that preset's TLR tone"
    return "create-cue-library makes one, with --protocol-preset NAME for a preset's TLR tone"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
