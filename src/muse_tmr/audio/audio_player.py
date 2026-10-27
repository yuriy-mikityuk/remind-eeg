"""Low-volume audio playback interface for sleep cues."""

from __future__ import annotations

import datetime as dt
import json
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AudioPlaybackConfig:
    max_volume: float = 0.20
    default_volume: float = 0.05
    fade_in_seconds: float = 0.25
    fade_out_seconds: float = 0.25
    device_name: Optional[str] = None
    log_path: Optional[Path] = None

    def validate(self) -> None:
        if not 0.0 <= self.max_volume <= 1.0:
            raise ValueError("max_volume must be between 0.0 and 1.0")
        if not 0.0 <= self.default_volume <= 1.0:
            raise ValueError("default_volume must be between 0.0 and 1.0")
        if self.fade_in_seconds < 0:
            raise ValueError("fade_in_seconds must be non-negative")
        if self.fade_out_seconds < 0:
            raise ValueError("fade_out_seconds must be non-negative")

    def cue_volume(
        self,
        volume: Optional[float] = None,
        *,
        volume_multiplier: float = 1.0,
    ) -> CueVolume:
        """The volume a player with this config plays a cue at.

        ``volume`` (``default_volume`` when None) is capped at ``max_volume`` first and
        then scaled by ``volume_multiplier``, the arousal guard's ``lower_volume``
        factor, so a lowered cue plays at exactly that fraction of what it would have
        played at, and never above a cap.
        """
        requested_volume = self.default_volume if volume is None else volume
        _validate_volume(requested_volume)
        if not 0.0 <= volume_multiplier <= 1.0:
            raise ValueError("volume_multiplier must be between 0.0 and 1.0")
        capped_volume = min(requested_volume, self.max_volume)
        return CueVolume(
            requested_volume=requested_volume,
            capped_volume=capped_volume,
            volume_multiplier=volume_multiplier,
            effective_volume=capped_volume * volume_multiplier,
        )


@dataclass(frozen=True)
class CueVolume:
    """One cue's volume: asked for, capped at max_volume, the multiplier, what plays."""

    requested_volume: float
    capped_volume: float
    volume_multiplier: float
    effective_volume: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "requested_volume": self.requested_volume,
            "capped_volume": self.capped_volume,
            "volume_multiplier": self.volume_multiplier,
            "effective_volume": self.effective_volume,
        }


@dataclass(frozen=True)
class TestCue:
    cue_id: str = "test-cue"
    frequency_hz: float = 440.0
    duration_seconds: float = 1.0

    def validate(self) -> None:
        if not self.cue_id:
            raise ValueError("cue_id must not be empty")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")


@dataclass(frozen=True)
class AudioPlaybackRequest:
    cue_id: str
    frequency_hz: float
    duration_seconds: float
    requested_volume: float
    effective_volume: float
    max_volume: float
    fade_in_seconds: float
    fade_out_seconds: float
    device_name: Optional[str] = None
    # Applied after the cap: effective_volume = capped_volume * volume_multiplier.
    volume_multiplier: float = 1.0

    @property
    def capped_volume(self) -> float:
        return min(self.requested_volume, self.max_volume)

    @property
    def volume_capped(self) -> bool:
        return self.capped_volume < self.requested_volume

    def to_dict(self) -> Dict[str, object]:
        return {
            "cue_id": self.cue_id,
            "frequency_hz": self.frequency_hz,
            "duration_seconds": self.duration_seconds,
            "requested_volume": self.requested_volume,
            "capped_volume": self.capped_volume,
            "volume_multiplier": self.volume_multiplier,
            "effective_volume": self.effective_volume,
            "max_volume": self.max_volume,
            "volume_capped": self.volume_capped,
            "fade_in_seconds": self.fade_in_seconds,
            "fade_out_seconds": self.fade_out_seconds,
            "device_name": self.device_name,
        }


@dataclass(frozen=True)
class CuePlaybackResult:
    cue_id: str
    status: str
    backend_name: str
    requested_volume: float
    effective_volume: float
    max_volume: float
    volume_capped: bool
    fade_in_seconds: float
    fade_out_seconds: float
    device_name: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    volume_multiplier: float = 1.0

    @property
    def played(self) -> bool:
        return self.status == "played"

    @property
    def capped_volume(self) -> float:
        return min(self.requested_volume, self.max_volume)

    def to_dict(self) -> Dict[str, object]:
        return {
            "cue_id": self.cue_id,
            "status": self.status,
            "backend_name": self.backend_name,
            "requested_volume": self.requested_volume,
            "capped_volume": self.capped_volume,
            "volume_multiplier": self.volume_multiplier,
            "effective_volume": self.effective_volume,
            "max_volume": self.max_volume,
            "volume_capped": self.volume_capped,
            "fade_in_seconds": self.fade_in_seconds,
            "fade_out_seconds": self.fade_out_seconds,
            "device_name": self.device_name,
            "reason_codes": list(self.reason_codes),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class AudioBackend:
    name = "base"

    def play_test_cue(self, request: AudioPlaybackRequest) -> Tuple[str, ...]:
        raise NotImplementedError

    def stop(self) -> Tuple[str, ...]:
        return ()


class DryRunAudioBackend(AudioBackend):
    name = "dry-run"

    def play_test_cue(self, request: AudioPlaybackRequest) -> Tuple[str, ...]:
        reasons = ["dry_run"]
        if request.device_name:
            reasons.append("device_selected")
        return tuple(reasons)


class MockAudioBackend(AudioBackend):
    name = "mock"

    def __init__(self) -> None:
        self.requests: List[AudioPlaybackRequest] = []
        self.stop_calls = 0

    def play_test_cue(self, request: AudioPlaybackRequest) -> Tuple[str, ...]:
        self.requests.append(request)
        reasons = ["mock_playback"]
        if request.device_name:
            reasons.append("device_selected")
        return tuple(reasons)

    def stop(self) -> Tuple[str, ...]:
        self.stop_calls += 1
        return ("mock_stop",)


class MacOSAfplayBackend(AudioBackend):
    name = "afplay"

    def play_test_cue(self, request: AudioPlaybackRequest) -> Tuple[str, ...]:
        if shutil.which("afplay") is None:
            return ("afplay_unavailable",)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
            _write_test_tone(handle.name, request)
            subprocess.run(["afplay", handle.name], check=True)

        reasons = ["system_playback"]
        if request.device_name:
            reasons.append("device_selection_unsupported")
        return tuple(reasons)


class AudioCuePlayer:
    """Safe playback facade for low-volume sleep cues.

    This layer only plays an already-approved cue. REM gating, arousal guards, and
    cue scheduling remain separate protocol layers.
    """

    def __init__(
        self,
        config: Optional[AudioPlaybackConfig] = None,
        backend: Optional[AudioBackend] = None,
    ) -> None:
        self.config = config or AudioPlaybackConfig()
        self.config.validate()
        self.backend = backend or create_audio_backend("system")
        self._emergency_stop_active = False

    @property
    def emergency_stop_active(self) -> bool:
        return self._emergency_stop_active

    def play_test_cue(
        self,
        cue: Optional[TestCue] = None,
        *,
        volume: Optional[float] = None,
        volume_multiplier: float = 1.0,
    ) -> CuePlaybackResult:
        """Play ``cue`` at ``volume`` (``default_volume`` when None), capped at
        ``max_volume`` and then scaled by ``volume_multiplier``, as
        ``AudioPlaybackConfig.cue_volume()`` computes it."""
        cue = cue or TestCue()
        cue.validate()
        volume_plan = self.config.cue_volume(volume, volume_multiplier=volume_multiplier)
        request = AudioPlaybackRequest(
            cue_id=cue.cue_id,
            frequency_hz=cue.frequency_hz,
            duration_seconds=cue.duration_seconds,
            requested_volume=volume_plan.requested_volume,
            effective_volume=volume_plan.effective_volume,
            max_volume=self.config.max_volume,
            fade_in_seconds=self.config.fade_in_seconds,
            fade_out_seconds=self.config.fade_out_seconds,
            device_name=self.config.device_name,
            volume_multiplier=volume_plan.volume_multiplier,
        )

        if self._emergency_stop_active:
            result = _result_from_request(
                request,
                backend_name=self.backend.name,
                status="blocked",
                reason_codes=("emergency_stop_active",),
            )
            self._log_result(result)
            return result

        reasons = []
        if request.volume_capped:
            reasons.append("volume_capped")
        if request.volume_multiplier < 1.0:
            reasons.append("volume_lowered")

        started_at = _utc_now()
        backend_reasons = self.backend.play_test_cue(request)
        ended_at = _utc_now()
        reasons.extend(backend_reasons)
        status = "played"
        if "afplay_unavailable" in backend_reasons:
            status = "skipped"

        result = _result_from_request(
            request,
            backend_name=self.backend.name,
            status=status,
            reason_codes=_unique(reasons),
            started_at=started_at,
            ended_at=ended_at,
        )
        self._log_result(result)
        return result

    def emergency_stop(self) -> CuePlaybackResult:
        self._emergency_stop_active = True
        ended_at = _utc_now()
        reason_codes = _unique(("emergency_stop",) + self.backend.stop())
        result = CuePlaybackResult(
            cue_id="emergency-stop",
            status="stopped",
            backend_name=self.backend.name,
            requested_volume=0.0,
            effective_volume=0.0,
            max_volume=self.config.max_volume,
            volume_capped=False,
            fade_in_seconds=self.config.fade_in_seconds,
            fade_out_seconds=self.config.fade_out_seconds,
            device_name=self.config.device_name,
            reason_codes=reason_codes,
            ended_at=ended_at,
        )
        self._log_result(result)
        return result

    def clear_emergency_stop(self) -> None:
        self._emergency_stop_active = False

    def _log_result(self, result: CuePlaybackResult) -> None:
        if self.config.log_path is None:
            return
        log_path = self.config.log_path.expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


class AudioPlayer(AudioCuePlayer):
    """Backward-compatible alias for the M4 audio cue player."""

    def __init__(
        self,
        max_volume: float = 0.20,
        backend: Optional[AudioBackend] = None,
    ) -> None:
        super().__init__(AudioPlaybackConfig(max_volume=max_volume), backend=backend)


def create_audio_backend(name: str) -> AudioBackend:
    if name == "system":
        if shutil.which("afplay") is not None:
            return MacOSAfplayBackend()
        return DryRunAudioBackend()
    if name == "afplay":
        return MacOSAfplayBackend()
    if name == "dry-run":
        return DryRunAudioBackend()
    if name == "mock":
        return MockAudioBackend()
    raise ValueError("audio backend must be one of: system, afplay, dry-run, mock")


class AudioBackendFallbackError(ValueError):
    """A real audio backend was asked for, but every cue would be silent."""


@dataclass(frozen=True)
class BackendResolution:
    requested_name: str
    resolved_name: str
    backend: AudioBackend


def resolve_audio_backend(
    name: str, *, allow_dry_run_fallback: bool = False, option: str = "--backend"
) -> BackendResolution:
    """Build the backend for ``name`` without swapping real audio for silence.

    ``system`` and ``afplay`` need ``afplay`` on PATH; without it ``system`` would
    become dry-run and ``afplay`` would skip every cue. That raises
    ``AudioBackendFallbackError`` unless ``allow_dry_run_fallback`` is set, which
    returns the dry-run backend instead. Other names go to ``create_audio_backend()``.
    ``option`` is the flag the error names (``--cue-backend`` in the app).
    Finding ``afplay`` does not show that the intended speaker is connected or audible.
    """
    if name in ("system", "afplay"):
        if shutil.which("afplay") is not None:
            backend: AudioBackend = MacOSAfplayBackend()
        elif allow_dry_run_fallback:
            backend = DryRunAudioBackend()
        else:
            raise AudioBackendFallbackError(
                f"{option} {name} needs afplay, which is not on PATH, so no cue would make "
                "a sound. Run it on macOS with /usr/bin on PATH, or add --allow-dry-run "
                "to run without sound on purpose."
            )
    else:
        backend = create_audio_backend(name)
    return BackendResolution(requested_name=name, resolved_name=backend.name, backend=backend)


def _result_from_request(
    request: AudioPlaybackRequest,
    *,
    backend_name: str,
    status: str,
    reason_codes: Tuple[str, ...],
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
) -> CuePlaybackResult:
    return CuePlaybackResult(
        cue_id=request.cue_id,
        status=status,
        backend_name=backend_name,
        requested_volume=request.requested_volume,
        effective_volume=request.effective_volume,
        max_volume=request.max_volume,
        volume_capped=request.volume_capped,
        fade_in_seconds=request.fade_in_seconds,
        fade_out_seconds=request.fade_out_seconds,
        device_name=request.device_name,
        reason_codes=reason_codes,
        started_at=started_at,
        ended_at=ended_at,
        volume_multiplier=request.volume_multiplier,
    )


def _write_test_tone(path: str, request: AudioPlaybackRequest) -> None:
    sample_rate_hz = 44100
    frame_count = int(sample_rate_hz * request.duration_seconds)
    amplitude = int(32767 * request.effective_volume)
    fade_in_frames = int(sample_rate_hz * request.fade_in_seconds)
    fade_out_frames = int(sample_rate_hz * request.fade_out_seconds)

    with wave.open(path, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate_hz)
        frames = bytearray()
        for index in range(frame_count):
            envelope = _fade_envelope(index, frame_count, fade_in_frames, fade_out_frames)
            phase = 2.0 * math.pi * request.frequency_hz * index / sample_rate_hz
            sample = int(amplitude * envelope * math.sin(phase))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        audio.writeframes(bytes(frames))


def _fade_envelope(
    index: int,
    frame_count: int,
    fade_in_frames: int,
    fade_out_frames: int,
) -> float:
    envelope = 1.0
    if fade_in_frames > 0 and index < fade_in_frames:
        envelope = min(envelope, index / fade_in_frames)
    if fade_out_frames > 0:
        frames_from_end = frame_count - index - 1
        if frames_from_end < fade_out_frames:
            envelope = min(envelope, frames_from_end / fade_out_frames)
    return max(0.0, min(1.0, envelope))


def _validate_volume(volume: float) -> None:
    if not 0.0 <= volume <= 1.0:
        raise ValueError("volume must be between 0.0 and 1.0")


def _unique(reason_codes: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(code for code in reason_codes if code))


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
