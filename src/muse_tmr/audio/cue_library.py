"""Cue library metadata and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

CUE_LIBRARY_SCHEMA_VERSION = 1
CUE_TYPES = ("sound", "generated_tone", "silence")
# "sham" marks an untrained sound that a sham night plays in place of the trained cues (#228).
CUE_PROTOCOLS = ("puzzle", "tlr", "test", "generic", "sham")
# The starter library's untrained sound, the sham arm's default.
DEFAULT_SHAM_CUE_ID = "sham_soft_tone"


@dataclass(frozen=True)
class CueMetadata:
    cue_id: str
    cue_type: str
    duration_seconds: float
    tags: Tuple[str, ...] = ()
    protocol: str = "generic"
    path: Optional[str] = None
    frequency_hz: Optional[float] = None
    volume_hint: Optional[float] = None
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "cue_id": self.cue_id,
            "cue_type": self.cue_type,
            "duration_seconds": self.duration_seconds,
            "tags": list(self.tags),
            "protocol": self.protocol,
            "description": self.description,
            "metadata": dict(self.metadata),
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.frequency_hz is not None:
            payload["frequency_hz"] = self.frequency_hz
        if self.volume_hint is not None:
            payload["volume_hint"] = self.volume_hint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CueMetadata":
        return cls(
            cue_id=str(payload["cue_id"]),
            cue_type=str(payload["cue_type"]),
            duration_seconds=float(payload["duration_seconds"]),
            tags=_string_tuple(payload.get("tags", ())),
            protocol=str(payload.get("protocol", "generic")),
            path=_optional_string(payload.get("path")),
            frequency_hz=_optional_float(payload.get("frequency_hz")),
            volume_hint=_optional_float(payload.get("volume_hint")),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class CueLibraryValidationIssue:
    severity: str
    cue_id: Optional[str]
    reason_code: str
    message: str

    @property
    def blocking(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> Dict[str, object]:
        return {
            "severity": self.severity,
            "cue_id": self.cue_id,
            "reason_code": self.reason_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class CueLibraryValidationReport:
    issues: Tuple[CueLibraryValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not any(issue.blocking for issue in self.issues)

    @property
    def blocking_issues(self) -> Tuple[CueLibraryValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.blocking)

    def to_dict(self) -> Dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class CueLibrary:
    cues: Tuple[CueMetadata, ...]
    schema_version: int = CUE_LIBRARY_SCHEMA_VERSION
    library_id: str = "default"

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "library_id": self.library_id,
            "cues": [cue.to_dict() for cue in self.cues],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CueLibrary":
        return cls(
            schema_version=int(payload.get("schema_version", CUE_LIBRARY_SCHEMA_VERSION)),
            library_id=str(payload.get("library_id", "default")),
            cues=tuple(CueMetadata.from_dict(cue) for cue in payload.get("cues", ())),
        )

    def save(self, output_path: Path) -> Path:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, input_path: Path) -> "CueLibrary":
        return cls.from_dict(json.loads(input_path.expanduser().read_text(encoding="utf-8")))

    def by_id(self, cue_id: str) -> CueMetadata:
        for cue in self.cues:
            if cue.cue_id == cue_id:
                return cue
        raise KeyError(f"unknown cue_id: {cue_id}")

    def filter(
        self,
        *,
        protocol: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Tuple[CueMetadata, ...]:
        cues = self.cues
        if protocol is not None:
            cues = tuple(cue for cue in cues if cue.protocol == protocol)
        if tag is not None:
            cues = tuple(cue for cue in cues if tag in cue.tags)
        return tuple(cues)

    def validate(
        self,
        *,
        base_dir: Optional[Path] = None,
        check_files: bool = True,
    ) -> CueLibraryValidationReport:
        issues = []
        if self.schema_version != CUE_LIBRARY_SCHEMA_VERSION:
            issues.append(
                CueLibraryValidationIssue(
                    severity="error",
                    cue_id=None,
                    reason_code="unsupported_schema_version",
                    message=f"unsupported schema_version={self.schema_version}",
                )
            )

        seen = set()
        for cue in self.cues:
            issues.extend(_validate_cue_metadata(cue))
            if cue.cue_id in seen:
                issues.append(
                    CueLibraryValidationIssue(
                        severity="error",
                        cue_id=cue.cue_id,
                        reason_code="duplicate_cue_id",
                        message=f"duplicate cue_id: {cue.cue_id}",
                    )
                )
            seen.add(cue.cue_id)

            if check_files and cue.cue_type == "sound" and cue.path:
                cue_path = resolve_cue_path(cue, base_dir=base_dir)
                if not cue_path.exists():
                    issues.append(
                        CueLibraryValidationIssue(
                            severity="error",
                            cue_id=cue.cue_id,
                            reason_code="cue_file_missing",
                            message=f"cue file does not exist: {cue.path}",
                        )
                    )
                elif not cue_path.is_file():
                    issues.append(
                        CueLibraryValidationIssue(
                            severity="error",
                            cue_id=cue.cue_id,
                            reason_code="cue_path_not_file",
                            message=f"cue path is not a file: {cue.path}",
                        )
                    )
        return CueLibraryValidationReport(tuple(issues))


def load_cue_library(input_path: Path) -> CueLibrary:
    return CueLibrary.load(input_path)


def export_cue_library(library: CueLibrary, output_path: Path) -> Path:
    return library.save(output_path)


def default_cue_library() -> CueLibrary:
    return CueLibrary(
        library_id="starter",
        cues=(
            CueMetadata(
                cue_id="puzzle_soft_tone",
                cue_type="generated_tone",
                protocol="puzzle",
                duration_seconds=1.0,
                frequency_hz=528.0,
                volume_hint=0.05,
                tags=("puzzle", "generated", "soft"),
                description="Starter generated tone for puzzle cue smoke tests.",
            ),
            CueMetadata(
                cue_id="tlr_soft_tone",
                cue_type="generated_tone",
                protocol="tlr",
                duration_seconds=1.0,
                frequency_hz=396.0,
                volume_hint=0.05,
                tags=("tlr", "generated", "soft"),
                description="Starter generated tone for TLR cue smoke tests.",
            ),
            CueMetadata(
                cue_id="silence_control",
                cue_type="silence",
                protocol="generic",
                duration_seconds=1.0,
                tags=("control", "silence"),
                description="Silent control cue for scheduler and analysis tests.",
            ),
            CueMetadata(
                cue_id=DEFAULT_SHAM_CUE_ID,
                cue_type="generated_tone",
                protocol="sham",
                duration_seconds=1.0,
                frequency_hz=330.0,
                volume_hint=0.05,
                tags=("sham", "generated", "untrained"),
                description=(
                    "Starter untrained tone for sham nights: never trained before sleep, "
                    "played in place of the trained cues."
                ),
            ),
        ),
    )


def validate_cue_library_file(
    input_path: Path,
    *,
    check_files: bool = True,
) -> CueLibraryValidationReport:
    input_path = input_path.expanduser()
    library = CueLibrary.load(input_path)
    return library.validate(base_dir=input_path.parent, check_files=check_files)


def resolve_cue_path(cue: CueMetadata, *, base_dir: Optional[Path] = None) -> Path:
    if cue.path is None:
        raise ValueError("cue has no path")
    cue_path = Path(cue.path).expanduser()
    if cue_path.is_absolute():
        return cue_path
    return (base_dir or Path.cwd()).expanduser() / cue_path


def _validate_cue_metadata(cue: CueMetadata) -> Tuple[CueLibraryValidationIssue, ...]:
    issues = []
    if not cue.cue_id.strip():
        issues.append(_issue(cue, "empty_cue_id", "cue_id must not be empty"))
    if any(char.isspace() for char in cue.cue_id):
        issues.append(_issue(cue, "cue_id_contains_whitespace", "cue_id must not contain whitespace"))
    if cue.cue_type not in CUE_TYPES:
        issues.append(_issue(cue, "invalid_cue_type", f"cue_type must be one of: {', '.join(CUE_TYPES)}"))
    if cue.protocol not in CUE_PROTOCOLS:
        issues.append(_issue(cue, "invalid_protocol", f"protocol must be one of: {', '.join(CUE_PROTOCOLS)}"))
    if cue.duration_seconds <= 0:
        issues.append(_issue(cue, "invalid_duration", "duration_seconds must be positive"))
    if cue.volume_hint is not None and not 0.0 <= cue.volume_hint <= 1.0:
        issues.append(_issue(cue, "invalid_volume_hint", "volume_hint must be between 0.0 and 1.0"))
    if cue.cue_type == "sound":
        if not cue.path:
            issues.append(_issue(cue, "sound_path_missing", "sound cues must include a path"))
        if cue.frequency_hz is not None:
            issues.append(_issue(cue, "sound_frequency_unexpected", "sound cues must not include frequency_hz"))
    if cue.cue_type == "generated_tone":
        if cue.path:
            issues.append(_issue(cue, "generated_path_unexpected", "generated_tone cues must not include a path"))
        if cue.frequency_hz is None or cue.frequency_hz <= 0:
            issues.append(_issue(cue, "generated_frequency_missing", "generated_tone cues require frequency_hz"))
    if cue.cue_type == "silence":
        if cue.path:
            issues.append(_issue(cue, "silence_path_unexpected", "silence cues must not include a path"))
        if cue.frequency_hz is not None:
            issues.append(_issue(cue, "silence_frequency_unexpected", "silence cues must not include frequency_hz"))
    return tuple(issues)


def _issue(cue: CueMetadata, reason_code: str, message: str) -> CueLibraryValidationIssue:
    return CueLibraryValidationIssue(
        severity="error",
        cue_id=cue.cue_id,
        reason_code=reason_code,
        message=message,
    )


def _string_tuple(value: object) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item) for item in value)


def _optional_string(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)
