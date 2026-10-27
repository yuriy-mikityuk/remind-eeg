"""Protocol presets: a paper's cueing parameters, cited, hashed and loadable (#231).

A preset holds what a paper reports about when and how cues were played (``CueSpec``,
``CueTiming``, ``PauseRules``, ``IntensityRamp``), where it comes from (``Citation``),
where REMind cannot follow it (``PlatformDeviation``) and the settings REMind's runners
use for it (``RunnerSettings``). The published values are documentation and inputs of
the protocol fidelity report; only ``RunnerSettings`` reaches a runner, through the
builders below.

No preset sets a volume. The builders leave ``max_volume``, ``default_volume``,
``volume_hint``, the hard cap and the calibrated comfortable volume alone; dB SPL
figures and ramps are text and numbers to compare a night with, nothing plays them.
A preset never lowers the REM gate's stable window below the repository's 60 s either.

``content_hash()`` is the sha256 of the preset's canonical JSON, so any change to a
value or a text gives a new hash; ``manifest(overrides)`` is the block a night records,
with the explicit flags that overrode the preset and their ``effective_hash``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Dict, Mapping, Optional, Tuple

from muse_tmr.protocol.tlr_protocol import DEFAULT_TLR_CUE_ID, TlrBlockConfig, TlrCueConfig
from muse_tmr.protocol.tmr_scheduler import TmrSchedulerConfig

PROTOCOL_MANIFEST_SCHEMA_VERSION = 1
SOURCE_CHECKS = ("verified_full_text", "notes_only")
CUE_STREAMS = ("tlr", "puzzle", "any")
# The gate's stable window in the repository; a preset may ask for more, never for less.
REPOSITORY_MIN_STABLE_SECONDS = 60.0
# The TLR cue's description when a preset keeps the repository tone.
DEFAULT_TLR_DESCRIPTION = TlrCueConfig().description


@dataclass(frozen=True)
class Citation:
    """Where a preset's parameters come from.

    ``source_check`` is ``verified_full_text`` when the parameters were read in the
    paper's full text, ``notes_only`` when they come from the research notes
    (``docs/research/2026-10-uniqueness/notes/``) without the full text.
    """

    authors: str
    year: int
    title: str
    venue: str
    url: str
    source_check: str
    arm: str = ""
    doi: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.source_check not in SOURCE_CHECKS:
            raise ValueError(f"source_check must be one of: {', '.join(SOURCE_CHECKS)}")
        for name in ("authors", "title", "venue", "url"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"citation {name} must not be empty")


@dataclass(frozen=True)
class CueSpec:
    """A cue as the paper describes it. ``stream`` is the REMind cue stream it maps to."""

    stream: str
    description: str
    duration_seconds: Optional[float] = None
    frequencies_hz: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _check_stream(self.stream)
        object.__setattr__(self, "frequencies_hz", tuple(float(value) for value in self.frequencies_hz))


@dataclass(frozen=True)
class CueTiming:
    """How often a stream of cues came, as published; None where the paper is silent.

    ``interval_seconds`` is (shortest, longest) between two cues, the longest None when
    only a minimum is given. ``max_cues_per_rem_period`` is a stated limit, not a mean.
    """

    stream: str
    interval_seconds: Optional[Tuple[float, Optional[float]]] = None
    max_cues_per_rem_period: Optional[int] = None
    note: str = ""

    def __post_init__(self) -> None:
        _check_stream(self.stream)
        if self.interval_seconds is not None:
            low, high = self.interval_seconds
            low = float(low)
            high = None if high is None else float(high)
            if low < 0 or (high is not None and high < low):
                raise ValueError("interval_seconds must be (low, high) with 0 <= low <= high")
            object.__setattr__(self, "interval_seconds", (low, high))
        if self.max_cues_per_rem_period is not None and self.max_cues_per_rem_period < 0:
            raise ValueError("max_cues_per_rem_period must be non-negative")


@dataclass(frozen=True)
class PauseRules:
    """When cueing waits, as published.

    ``start_after_rem_seconds`` is the REM before the first cue of a REM period and
    ``resume_after_rem_seconds`` the REM before cueing starts again after a stage change.
    ``pause_seconds`` maps a trigger (movement, alpha, blink, poor_signal, heart_rate,
    arousal) to the pause after it; ``pause_until_clear`` lists triggers that held
    cueing until the sign was gone, without a duration.
    """

    start_after_rem_seconds: Optional[float] = None
    resume_after_rem_seconds: Optional[float] = None
    pause_seconds: Tuple[Tuple[str, float], ...] = ()
    pause_until_clear: Tuple[str, ...] = ()
    pause_after_awakening_seconds: Optional[float] = None
    stops_on: Tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        pauses = tuple((str(trigger), float(seconds)) for trigger, seconds in self.pause_seconds)
        if any(seconds < 0 for _, seconds in pauses):
            raise ValueError("pause_seconds must be non-negative")
        object.__setattr__(self, "pause_seconds", pauses)
        object.__setattr__(self, "pause_until_clear", tuple(self.pause_until_clear))
        object.__setattr__(self, "stops_on", tuple(self.stops_on))

    def pause_for(self, trigger: str) -> Optional[float]:
        """The published pause after ``trigger``, else the one after any arousal."""
        pauses = dict(self.pause_seconds)
        if trigger in pauses:
            return pauses[trigger]
        return pauses.get("arousal")


@dataclass(frozen=True)
class IntensityRamp:
    """How loud cues were and how that changed, as published.

    Documentation and an input of the fidelity report only: REMind plays every cue at
    the gain its player computes, and no builder reads these fields. ``rises`` says the
    paper raised the intensity during cueing, with or without a stated step.
    """

    start: str = ""
    rises: bool = False
    step_percent_per_cue: Optional[float] = None
    step_db_per_cycle: Optional[float] = None
    on_arousal: str = ""
    cap: str = ""
    level_db_spl: Optional[Tuple[float, float]] = None
    background_noise: str = ""

    def __post_init__(self) -> None:
        if self.level_db_spl is not None:
            low, high = self.level_db_spl
            object.__setattr__(self, "level_db_spl", (float(low), float(high)))

    @property
    def ramps(self) -> bool:
        return bool(self.rises or self.step_percent_per_cue or self.step_db_per_cycle)


@dataclass(frozen=True)
class PlatformDeviation:
    """Where REMind on a Muse headband does something else than the paper."""

    parameter: str
    published: str
    remind: str
    reason: str = ""


@dataclass(frozen=True)
class RunnerSettings:
    """The settings REMind's runners use for a preset. The defaults are the repository's."""

    gate_min_stable_seconds: float = 60.0
    puzzle_cue_interval_seconds: float = 30.0
    scheduler_cooldown_seconds: float = 120.0
    max_puzzle_cues_per_block: int = 4
    tlr_block_enabled: bool = True
    tlr_repetitions: int = 3
    tlr_interval_seconds: float = 8.0
    tlr_post_block_pause_seconds: float = 10.0
    tlr_tone_frequency_hz: float = 396.0
    tlr_tone_duration_seconds: float = 1.0
    puzzle_count: int = 4
    # None cues half of the night's puzzles, as split_cued_uncued() does.
    cued_puzzle_count: Optional[int] = None

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(self, item.name, _coerce(item.name, getattr(self, item.name)))
        if self.gate_min_stable_seconds <= 0:
            raise ValueError("gate_min_stable_seconds must be positive")
        if self.puzzle_cue_interval_seconds < 0 or self.scheduler_cooldown_seconds < 0:
            raise ValueError("scheduler intervals must be non-negative")
        if self.max_puzzle_cues_per_block <= 0:
            raise ValueError("max_puzzle_cues_per_block must be positive")
        if self.tlr_repetitions < 0 or self.tlr_interval_seconds < 0 or self.tlr_post_block_pause_seconds < 0:
            raise ValueError("TLR block settings must be non-negative")
        if self.tlr_tone_frequency_hz <= 0 or self.tlr_tone_duration_seconds <= 0:
            raise ValueError("the TLR tone needs a positive frequency and duration")
        if self.puzzle_count <= 0:
            raise ValueError("puzzle_count must be positive")
        # Checked against puzzle_count only in a preset: an override may cue more puzzles
        # than the preset's night has when the session was generated with more.
        if self.cued_puzzle_count is not None and self.cued_puzzle_count < 0:
            raise ValueError("cued_puzzle_count must be non-negative")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# What each RunnerSettings field is as a JSON value; overrides are coerced to it.
_SETTING_TYPES = {item.name: item.type for item in fields(RunnerSettings)}


@dataclass(frozen=True)
class ProtocolPreset:
    """One protocol: what a paper reports, and the settings REMind runs it with."""

    name: str
    version: int
    title: str
    runnable: bool
    design_note: str
    citations: Tuple[Citation, ...] = ()
    caveats: Tuple[str, ...] = ()
    cues: Tuple[CueSpec, ...] = ()
    timing: Tuple[CueTiming, ...] = ()
    pauses: PauseRules = field(default_factory=PauseRules)
    ramp: IntensityRamp = field(default_factory=IntensityRamp)
    training: str = ""
    morning: str = ""
    runner: RunnerSettings = field(default_factory=RunnerSettings)
    deviations: Tuple[PlatformDeviation, ...] = ()
    not_runnable_reason: str = ""

    def __post_init__(self) -> None:
        for name in ("citations", "caveats", "cues", "timing", "deviations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.name.strip() or self.name != self.name.strip().lower() or " " in self.name:
            raise ValueError("preset name must be a lowercase identifier")
        if self.version <= 0:
            raise ValueError("preset version must be positive")
        if not self.runnable and not self.not_runnable_reason.strip():
            raise ValueError("a preset that cannot run must say why")
        if self.runner.gate_min_stable_seconds < REPOSITORY_MIN_STABLE_SECONDS:
            raise ValueError(
                f"a preset may not lower the gate's stable window below {REPOSITORY_MIN_STABLE_SECONDS:g} s"
            )
        cued = self.runner.cued_puzzle_count
        if cued is not None and cued > self.runner.puzzle_count:
            raise ValueError("a preset's cued_puzzle_count must fit inside its puzzle_count")

    # --- identity ---------------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return _jsonable(asdict(self))

    def content_hash(self) -> str:
        """sha256 of the canonical JSON of ``to_dict()``."""
        return _sha256(self.to_dict())

    def manifest(
        self,
        overrides: Optional[Mapping[str, object]] = None,
        *,
        registry_version: Optional[int] = None,
    ) -> Dict[str, object]:
        """The block a night, a plan or an analysis records about its protocol.

        ``overrides`` are the explicit flags that replaced preset settings, by
        ``RunnerSettings`` field name. ``effective_hash`` covers the content hash and
        the overrides, so two runs of one preset with different flags differ there.
        """
        if registry_version is None:
            from muse_tmr.protocol.protocol_registry import PROTOCOL_REGISTRY_VERSION

            registry_version = PROTOCOL_REGISTRY_VERSION
        normalized = normalize_overrides(overrides)
        content_hash = self.content_hash()
        return {
            "schema_version": PROTOCOL_MANIFEST_SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "registry_version": int(registry_version),
            "content_hash": content_hash,
            "runnable": self.runnable,
            "overrides": normalized,
            "effective_hash": _sha256({"content_hash": content_hash, "overrides": normalized}),
            "settings": self.runner_settings(normalized).to_dict(),
        }

    # --- builders -----------------------------------------------------------------

    def runner_settings(self, overrides: Optional[Mapping[str, object]] = None) -> RunnerSettings:
        """The preset's runner settings with ``overrides`` applied (explicit flags win)."""
        normalized = normalize_overrides(overrides)
        return replace(self.runner, **normalized) if normalized else self.runner

    def scheduler_config(
        self,
        *,
        enable_tlr_block: bool,
        overrides: Optional[Mapping[str, object]] = None,
    ) -> TmrSchedulerConfig:
        settings = self.runner_settings(overrides)
        return TmrSchedulerConfig(
            puzzle_cue_interval_seconds=settings.puzzle_cue_interval_seconds,
            cooldown_seconds=settings.scheduler_cooldown_seconds,
            max_puzzle_cues_per_block=settings.max_puzzle_cues_per_block,
            enable_tlr_block=enable_tlr_block,
        )

    def tlr_block_config(self, overrides: Optional[Mapping[str, object]] = None) -> TlrBlockConfig:
        settings = self.runner_settings(overrides)
        return TlrBlockConfig(
            enabled=settings.tlr_block_enabled,
            repetitions=settings.tlr_repetitions,
            interval_seconds=settings.tlr_interval_seconds,
            post_block_pause_seconds=settings.tlr_post_block_pause_seconds,
        )

    def tlr_cue_config(
        self,
        *,
        cue_id: str = DEFAULT_TLR_CUE_ID,
        overrides: Optional[Mapping[str, object]] = None,
    ) -> TlrCueConfig:
        """The generated TLR tone; its volume hint stays the TlrCueConfig default."""
        settings = self.runner_settings(overrides)
        return TlrCueConfig(
            cue_id=cue_id,
            frequency_hz=settings.tlr_tone_frequency_hz,
            duration_seconds=settings.tlr_tone_duration_seconds,
            description=self._tlr_description(settings),
        )

    def gate_overrides(self, overrides: Optional[Mapping[str, object]] = None) -> Dict[str, float]:
        """The RemGateConfig fields this preset sets: only the stable window."""
        return {"min_stable_seconds": self.runner_settings(overrides).gate_min_stable_seconds}

    def puzzle_counts(self, overrides: Optional[Mapping[str, object]] = None) -> Tuple[int, Optional[int]]:
        """(puzzles per night, cued puzzles); None cued means half."""
        settings = self.runner_settings(overrides)
        return settings.puzzle_count, settings.cued_puzzle_count

    # --- lookups ------------------------------------------------------------------

    def cue_spec(self, stream: str) -> Optional[CueSpec]:
        return next((cue for cue in self.cues if cue.stream == stream), None)

    def timing_for(self, stream: str) -> Optional[CueTiming]:
        return next((item for item in self.timing if item.stream == stream), None)

    def _tlr_description(self, settings: RunnerSettings) -> str:
        default = RunnerSettings()
        if (settings.tlr_tone_frequency_hz, settings.tlr_tone_duration_seconds) == (
            default.tlr_tone_frequency_hz,
            default.tlr_tone_duration_seconds,
        ):
            return DEFAULT_TLR_DESCRIPTION
        return (
            f"Generated {settings.tlr_tone_frequency_hz:g} Hz tone of {settings.tlr_tone_duration_seconds:g} s "
            f"standing in for the TLR cue of protocol preset {self.name}."
        )


def protocol_block_of(payload: object) -> Optional[Dict[str, object]]:
    """The protocol block a record carries, or None.

    A night's ``metadata.json``, a Pilot 4/5 summary and a Pilot 3 report keep it as
    ``protocol``; an assignment, a session, a TLR plan and an analysis under
    ``metadata.protocol``.
    """
    if not isinstance(payload, Mapping):
        return None
    candidates = [payload.get("protocol")]
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.append(metadata.get("protocol"))
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("name") and candidate.get("content_hash"):
            return dict(candidate)
    return None


def normalize_overrides(overrides: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """``overrides`` by RunnerSettings field, each as that field's type, sorted by name."""
    if not overrides:
        return {}
    unknown = sorted(set(overrides) - set(_SETTING_TYPES))
    if unknown:
        raise ValueError(f"unknown protocol settings: {', '.join(unknown)}")
    return {name: _coerce(name, overrides[name]) for name in sorted(overrides)}


def canonical_json(payload: object) -> str:
    """The JSON a hash is taken of: sorted keys, no spaces, tuples as lists."""
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _coerce(name: str, value: object) -> object:
    kind = str(_SETTING_TYPES.get(name, ""))
    if value is None:
        if "Optional" in kind:
            return None
        raise ValueError(f"{name} must not be None")
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if "int" in kind:
        return int(value)
    if kind == "float":
        return float(value)
    return value


def _check_stream(stream: str) -> None:
    if stream not in CUE_STREAMS:
        raise ValueError(f"stream must be one of: {', '.join(CUE_STREAMS)}")
