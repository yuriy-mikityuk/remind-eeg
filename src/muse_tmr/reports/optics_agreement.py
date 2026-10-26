"""Heart-rate agreement of the Athena optics with a Polar H10 ECG recorded with it (#234).

The Muse estimates heart rate from its forehead optics in four ways: the decoder's
``heart_rate`` stream (peaks on the first optics channel, LO_NIR on ``p1034``, over its
last 2 to 5 s), the PPG row's ``ppg_estimated_hr_bpm`` (peak intervals on the channel
with the largest variance), the same estimate on every optics channel
(``estimate_channel_heart_rates``) and the heart-rate series the sudden heart-rate
change rule reads (``heart_rate_series``). This report compares each with the heart rate
from the H10's ECG R-peaks outside its no-contact spans: per epoch for the first three,
per 10 s window for the series. It gives bias, SD, 95 % limits of agreement and MAE
(Bland-Altman), overall and by ``ppg_signal_quality``, ``ppg_confidence`` and the
series' quality. It then runs the sudden heart-rate change rule over the epochs as the
arousal guard does by default and scores its pauses against the ECG heart-rate change
between the same windows: false jumps, missed jumps, and what a quality gate on
``ppg_confidence`` or on the series would keep.

This is agreement with a chest ECG, a proxy for heart rate only: no PSG takes part, and
nothing here estimates blood oxygen or sleep stages. It reads recordings and features only:
it imports nothing from ``muse_tmr.audio``, ``muse_tmr.models`` or
``muse_tmr.protocol``, and Polar data never reaches ``MuseFrame``, the detector, the
gate, the guard, the scheduler or audio.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from muse_tmr.data.polar_recorder import POLAR_DIRNAME
from muse_tmr.data.polar_session import PolarSession, load_polar_session
from muse_tmr.data.replay import ReplayConfig, ReplaySession
from muse_tmr.features.cardio_resp_features import CardioRespConfig, correct_rr
from muse_tmr.features.epochs import EpochBuilder, EpochConfig, SleepEpoch
from muse_tmr.features.ppg_features import (
    PPG_SIGNAL_QUALITIES,
    ChannelHeartRate,
    HeartRatePoint,
    PPGFeatureConfig,
    SuddenHeartRateChange,
    estimate_channel_heart_rates,
    extract_ppg_features,
    heart_rate_change_across_epochs,
)
from muse_tmr.presets import preset_provides_optics
from muse_tmr.testing.synthetic import SYNTHETIC_NOTE

OPTICS_AGREEMENT_SCHEMA_VERSION = 1
JSON_FILENAME = "optics_agreement.json"
MARKDOWN_FILENAME = "optics_agreement.md"

# The strata of the agreement rows. The decoder and ppg_estimated_hr_bpm are split by
# the epoch's quality fields, each channel by its own estimate's, the series by its
# epoch's ppg_signal_quality and ppg_confidence and by each value's own quality.
STRATA = ("all", "ppg_signal_quality", "ppg_confidence", "series_quality", "periodicity")
# The quality fields a gate on the HR rule can key on, as the arousal guard reads them.
GATE_FIELDS = ("ppg_confidence", "hr_series_median_quality")

_NONE_LEVEL = "none"
_TIME_TOLERANCE_SECONDS = 1e-6


class OpticsAgreementInputError(ValueError):
    """The recording cannot be compared: no Polar data, no ECG or no optics."""


@dataclass(frozen=True)
class OpticsAgreementConfig:
    """How the optics are compared with the ECG.

    ``epoch_seconds`` is the comparison window of the epoch estimators; an epoch whose
    optics cover less than ``min_optics_coverage`` of it is not compared there. An ECG
    reference needs beat intervals spanning ``min_ecg_coverage`` of its window. The jump
    fields mirror ``ArousalGuardConfig``'s defaults (``sudden_hr_change_pause_bpm``,
    ``sudden_hr_change_margin_bpm``, ``sudden_hr_change_pause_count`` and
    ``sudden_hr_change_link_seconds``). The ECG change of an epoch is its largest one
    between the windows the rule pairs. A fired epoch is a false jump when neither it
    nor the epoch before has an ECG change of ``jump_bpm`` less ``jump_margin_bpm``. An
    ECG change of ``jump_bpm`` or more counts once, also when the next epoch sees it again
    in the same direction over overlapping windows, and is missed when none of its epochs
    and not the one after them fires. Below ``min_epochs`` compared epochs the report
    carries ``n_epochs_small``, below ``min_jump_events`` fired epochs and ECG jumps
    together ``few_jump_events``.
    """

    epoch_seconds: float = 30.0
    min_optics_coverage: float = 0.9
    min_ecg_coverage: float = 0.7
    confidence_bins: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    series_quality_bins: Tuple[float, ...] = (0.0, 0.5, 0.7, 0.8, 0.9, 1.0)
    gate_thresholds: Tuple[float, ...] = (0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    jump_bpm: float = 10.0
    jump_margin_bpm: float = 1.0
    jump_pause_count: int = 1
    jump_link_seconds: float = 20.0
    hr_rule_blind_skipped_share: float = 0.5
    min_epochs: int = 60
    min_jump_events: int = 10
    ppg: PPGFeatureConfig = field(default_factory=PPGFeatureConfig)
    cardio: CardioRespConfig = field(default_factory=CardioRespConfig)

    def validate(self) -> None:
        self.ppg.validate()
        if self.epoch_seconds < self.ppg.hr_series_window_seconds:
            raise ValueError(
                f"epoch_seconds must hold one heart-rate series window "
                f"({self.ppg.hr_series_window_seconds:g} s)"
            )
        if not 0.0 < self.min_ecg_coverage <= 1.0:
            raise ValueError("min_ecg_coverage must be in (0, 1]")
        if not 0.0 <= self.min_optics_coverage <= 1.0:
            raise ValueError("min_optics_coverage must be between 0 and 1")
        for name in ("confidence_bins", "series_quality_bins"):
            edges = getattr(self, name)
            if len(edges) < 2 or list(edges) != sorted(set(edges)) or edges[0] != 0.0 or edges[-1] != 1.0:
                raise ValueError(f"{name} must rise from 0 to 1")
        if any(not 0.0 < value <= 1.0 for value in self.gate_thresholds):
            raise ValueError("gate_thresholds must be in (0, 1]")
        if self.jump_bpm <= 0 or not 0.0 <= self.jump_margin_bpm < self.jump_bpm:
            raise ValueError("jump_bpm must be positive and jump_margin_bpm in [0, jump_bpm)")
        if self.jump_pause_count <= 0:
            raise ValueError("jump_pause_count must be positive")
        if self.jump_link_seconds < 0:
            raise ValueError("jump_link_seconds must be non-negative")
        if not 0.0 <= self.hr_rule_blind_skipped_share <= 1.0:
            raise ValueError("hr_rule_blind_skipped_share must be between 0 and 1")
        if self.min_epochs < 0 or self.min_jump_events < 0:
            raise ValueError("min_epochs and min_jump_events must be non-negative")

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        return _json_ready(payload)


@dataclass(frozen=True)
class AgreementRow:
    """Bland-Altman agreement of one estimator (and channel) in one stratum.

    ``n_reference`` windows had an ECG reference, ``n`` of them an estimate too; the
    differences are estimate minus ECG, in bpm.
    """

    estimator: str
    channel: str
    stratum: str
    level: str
    n_reference: int
    n: int
    bias_bpm: float
    sd_bpm: float
    loa_low_bpm: float
    loa_high_bpm: float
    mae_bpm: float

    @property
    def coverage(self) -> float:
        return self.n / self.n_reference if self.n_reference else math.nan

    def to_dict(self) -> Dict[str, object]:
        return {
            "estimator": self.estimator,
            "channel": self.channel,
            "stratum": self.stratum,
            "level": self.level,
            "n_reference": self.n_reference,
            "n": self.n,
            "coverage": _finite_or_none(self.coverage),
            "bias_bpm": _finite_or_none(self.bias_bpm),
            "sd_bpm": _finite_or_none(self.sd_bpm),
            "loa_low_bpm": _finite_or_none(self.loa_low_bpm),
            "loa_high_bpm": _finite_or_none(self.loa_high_bpm),
            "mae_bpm": _finite_or_none(self.mae_bpm),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AgreementRow":
        return cls(
            estimator=str(payload["estimator"]),
            channel=str(payload.get("channel", "")),
            stratum=str(payload["stratum"]),
            level=str(payload["level"]),
            n_reference=int(payload["n_reference"]),
            n=int(payload["n"]),
            bias_bpm=_float(payload.get("bias_bpm")),
            sd_bpm=_float(payload.get("sd_bpm")),
            loa_low_bpm=_float(payload.get("loa_low_bpm")),
            loa_high_bpm=_float(payload.get("loa_high_bpm")),
            mae_bpm=_float(payload.get("mae_bpm")),
        )


@dataclass(frozen=True)
class JumpRow:
    """The sudden heart-rate change rule against the ECG in one stratum of epochs.

    ``epochs`` had an ECG change to score against; ``fired`` of them paused,
    ``false_jumps`` of those without an ECG change of ``jump_bpm`` less the margin in
    their own epoch or the one before. ``ecg_jumps`` are the ECG changes of ``jump_bpm``
    or more, each counted once in the epoch it starts in, ``missed`` of them without a
    pause in their epochs or the one after.
    """

    stratum: str
    level: str
    epochs: int
    fired: int
    false_jumps: int
    ecg_jumps: int
    missed: int

    @property
    def false_jump_rate(self) -> float:
        return self.false_jumps / self.fired if self.fired else math.nan

    @property
    def miss_rate(self) -> float:
        return self.missed / self.ecg_jumps if self.ecg_jumps else math.nan

    def to_dict(self) -> Dict[str, object]:
        return {
            "stratum": self.stratum,
            "level": self.level,
            "epochs": self.epochs,
            "fired": self.fired,
            "false_jumps": self.false_jumps,
            "false_jump_rate": _finite_or_none(self.false_jump_rate),
            "ecg_jumps": self.ecg_jumps,
            "missed": self.missed,
            "miss_rate": _finite_or_none(self.miss_rate),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "JumpRow":
        return cls(
            stratum=str(payload["stratum"]),
            level=str(payload["level"]),
            epochs=int(payload["epochs"]),
            fired=int(payload["fired"]),
            false_jumps=int(payload["false_jumps"]),
            ecg_jumps=int(payload["ecg_jumps"]),
            missed=int(payload["missed"]),
        )


@dataclass(frozen=True)
class GateRow:
    """The rule with a quality gate: epochs whose ``field`` is below ``threshold`` (or
    missing) do not fire and do not feed the next epoch's comparison, as in the guard.
    ``field`` "none" is the rule as it runs today."""

    field: str
    threshold: float
    gated_epochs: int
    fired: int
    false_jumps: int
    ecg_jumps: int
    caught: int

    @property
    def false_jump_rate(self) -> float:
        return self.false_jumps / self.fired if self.fired else math.nan

    @property
    def miss_rate(self) -> float:
        return (self.ecg_jumps - self.caught) / self.ecg_jumps if self.ecg_jumps else math.nan

    def to_dict(self) -> Dict[str, object]:
        return {
            "field": self.field,
            "threshold": self.threshold,
            "gated_epochs": self.gated_epochs,
            "fired": self.fired,
            "false_jumps": self.false_jumps,
            "false_jump_rate": _finite_or_none(self.false_jump_rate),
            "ecg_jumps": self.ecg_jumps,
            "caught": self.caught,
            "miss_rate": _finite_or_none(self.miss_rate),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GateRow":
        return cls(
            field=str(payload["field"]),
            threshold=float(payload["threshold"]),
            gated_epochs=int(payload["gated_epochs"]),
            fired=int(payload["fired"]),
            false_jumps=int(payload["false_jumps"]),
            ecg_jumps=int(payload["ecg_jumps"]),
            caught=int(payload["caught"]),
        )


@dataclass(frozen=True)
class JumpEvent:
    """One change the rule counted, with the ECG change between the same two windows.
    Times are seconds from the first epoch's start."""

    epoch_index: int
    kind: str
    start_seconds: float
    end_seconds: float
    optics_delta_bpm: float
    ecg_delta_bpm: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "epoch_index": self.epoch_index,
            "kind": self.kind,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "optics_delta_bpm": self.optics_delta_bpm,
            "ecg_delta_bpm": _finite_or_none(self.ecg_delta_bpm),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "JumpEvent":
        return cls(
            epoch_index=int(payload["epoch_index"]),
            kind=str(payload["kind"]),
            start_seconds=float(payload["start_seconds"]),
            end_seconds=float(payload["end_seconds"]),
            optics_delta_bpm=float(payload["optics_delta_bpm"]),
            ecg_delta_bpm=_float(payload.get("ecg_delta_bpm")),
        )


@dataclass(frozen=True)
class AgreementLimitation:
    code: str
    message: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"code": self.code, "message": self.message, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AgreementLimitation":
        return cls(
            code=str(payload["code"]),
            message=str(payload["message"]),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class OpticsAgreementReport:
    recording: str
    synthetic: bool
    config: Mapping[str, object]
    counts: Mapping[str, object]
    agreement: Tuple[AgreementRow, ...]
    jumps: Tuple[JumpRow, ...]
    gates: Tuple[GateRow, ...]
    separation: Mapping[str, object]
    events: Tuple[JumpEvent, ...]
    limitations: Tuple[AgreementLimitation, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    generated_at_utc: str = ""
    schema_version: int = OPTICS_AGREEMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("agreement", "jumps", "gates", "events", "limitations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())

    @property
    def limitation_codes(self) -> Tuple[str, ...]:
        return tuple(item.code for item in self.limitations)

    def rows(
        self,
        estimator: str,
        *,
        channel: Optional[str] = None,
        stratum: str = "all",
    ) -> Tuple[AgreementRow, ...]:
        return tuple(
            row
            for row in self.agreement
            if row.estimator == estimator
            and row.stratum == stratum
            and (channel is None or row.channel == channel)
        )

    def overall(self, estimator: str, channel: str = "") -> Optional[AgreementRow]:
        rows = self.rows(estimator, channel=channel)
        return rows[0] if rows else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report": "optics_agreement",
            "recording": self.recording,
            "generated_at_utc": self.generated_at_utc,
            "synthetic": self.synthetic,
            "note": SYNTHETIC_NOTE if self.synthetic else "",
            "config": dict(self.config),
            "counts": dict(self.counts),
            "agreement": [row.to_dict() for row in self.agreement],
            "jumps": [row.to_dict() for row in self.jumps],
            "gates": [row.to_dict() for row in self.gates],
            "separation": dict(self.separation),
            "events": [event.to_dict() for event in self.events],
            "limitations": [item.to_dict() for item in self.limitations],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OpticsAgreementReport":
        return cls(
            schema_version=int(payload.get("schema_version", OPTICS_AGREEMENT_SCHEMA_VERSION)),
            recording=str(payload.get("recording", "")),
            generated_at_utc=str(payload.get("generated_at_utc", "")),
            synthetic=bool(payload.get("synthetic", False)),
            config=dict(payload.get("config", {}) or {}),
            counts=dict(payload.get("counts", {}) or {}),
            agreement=tuple(AgreementRow.from_dict(item) for item in payload.get("agreement", ())),
            jumps=tuple(JumpRow.from_dict(item) for item in payload.get("jumps", ())),
            gates=tuple(GateRow.from_dict(item) for item in payload.get("gates", ())),
            separation=dict(payload.get("separation", {}) or {}),
            events=tuple(JumpEvent.from_dict(item) for item in payload.get("events", ())),
            limitations=tuple(
                AgreementLimitation.from_dict(item) for item in payload.get("limitations", ())
            ),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def save(self, output_path: Path) -> Path:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return output_path

    def save_markdown(self, output_path: Path) -> Path:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def write(self, output_dir: Path) -> Dict[str, Path]:
        """Write ``optics_agreement.json`` and ``optics_agreement.md`` into ``output_dir``."""
        output_dir = Path(output_dir).expanduser()
        return {
            "json": self.save(output_dir / JSON_FILENAME),
            "markdown": self.save_markdown(output_dir / MARKDOWN_FILENAME),
        }

    @classmethod
    def load(cls, input_path: Path) -> "OpticsAgreementReport":
        return cls.from_dict(json.loads(Path(input_path).expanduser().read_text(encoding="utf-8")))

    def to_markdown(self) -> str:
        return _markdown(self)


def load_optics_agreement_report(input_path: Path) -> OpticsAgreementReport:
    return OpticsAgreementReport.load(input_path)


async def analyze_optics_agreement(
    recording_dir: Path,
    config: Optional[OpticsAgreementConfig] = None,
) -> OpticsAgreementReport:
    """Compare the optics heart rate of a recording with the Polar H10 ECG in its
    ``polar/`` folder, over ``ReplaySession`` and ``EpochBuilder``.

    Raises ``OpticsAgreementInputError`` when the recording has no ``polar/`` folder,
    the Polar session no ECG R-peaks, or the recording no optics. Loading the Polar
    session writes its decoded files into ``polar/`` when only the raw log is there, as
    ``decode-polar`` does.
    """
    config = config or OpticsAgreementConfig()
    config.validate()
    recording_dir = Path(recording_dir).expanduser()
    metadata = _recording_metadata(recording_dir)
    preset = _preset(metadata)
    if preset and not preset_provides_optics(preset):
        raise OpticsAgreementInputError(
            f"{recording_dir} was recorded on {preset}, which streams no optics; "
            "record with --preset p1034"
        )
    session = _polar_session(recording_dir)

    replay = ReplaySession(ReplayConfig(input_path=recording_dir, speed=0.0))
    await replay.connect()
    accumulator = _Accumulator(session, config)
    try:
        builder = EpochBuilder(
            EpochConfig(epoch_seconds=config.epoch_seconds, stride_seconds=config.epoch_seconds)
        )
        async for epoch in builder.build(replay.stream()):
            accumulator.add(epoch)
    finally:
        await replay.stop()
    return accumulator.report(
        recording=recording_dir.name,
        synthetic=metadata.get("synthetic") is True,
        preset=preset,
    )


def optics_agreement_from_epochs(
    epochs: Iterable[SleepEpoch],
    session: PolarSession,
    config: Optional[OpticsAgreementConfig] = None,
    *,
    recording: str = "",
    synthetic: bool = False,
) -> OpticsAgreementReport:
    """The report for consecutive epochs of one recording and its loaded Polar session.

    Raises ``OpticsAgreementInputError`` when the session has no ECG R-peaks or no epoch
    holds optics.
    """
    config = config or OpticsAgreementConfig()
    config.validate()
    accumulator = _Accumulator(session, config)
    for epoch in epochs:
        accumulator.add(epoch)
    return accumulator.report(recording=recording, synthetic=synthetic, preset="")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _recording_metadata(recording_dir: Path) -> Mapping[str, object]:
    path = recording_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _preset(metadata: Mapping[str, object]) -> str:
    source = metadata.get("source")
    if not isinstance(source, Mapping):
        return ""
    source_metadata = source.get("metadata")
    if not isinstance(source_metadata, Mapping):
        return ""
    return str(source_metadata.get("preset") or "")


def _polar_session(recording_dir: Path) -> PolarSession:
    polar_dir = recording_dir / POLAR_DIRNAME
    if not (polar_dir / "raw_notifications.jsonl").exists() and not (polar_dir / "ecg.jsonl").exists():
        raise OpticsAgreementInputError(
            f"{recording_dir} has no {POLAR_DIRNAME}/ folder with a Polar H10 recording; "
            "record with --with-polar (or the app's session kind with the H10)"
        )
    session = load_polar_session(recording_dir)
    _require_r_peaks(session)
    return session


def _require_r_peaks(session: PolarSession) -> None:
    if session.r_peaks is None or len(session.r_peaks) < 3:
        raise OpticsAgreementInputError(
            "the Polar session has no ECG R-peaks to compare with (recorded with --no-ecg, "
            "or the strap had no contact)"
        )


# --------------------------------------------------------------------------- #
# The ECG reference
# --------------------------------------------------------------------------- #
class _EcgReference:
    """Heart rate from consecutive ECG R-peaks, outside the no-contact spans.

    Intervals outside ``min_rr_ms``..``max_rr_ms`` or across a no-contact span are left
    out; each run of the others is cleaned with ``correct_rr`` (ectopic and missed beats).
    A window's heart rate is 60 over the mean corrected interval of the intervals inside
    it, when those span ``min_ecg_coverage`` of the window.
    """

    def __init__(self, session: PolarSession, config: OpticsAgreementConfig) -> None:
        self.config = config
        peaks = np.sort(np.asarray(session.r_peaks["time"], dtype=float))
        self.peak_count = int(peaks.size)
        self.starts = peaks[:-1]
        self.ends = peaks[1:]
        self.rr_ms = (self.ends - self.starts) * 1000.0
        cardio = config.cardio
        valid = (self.rr_ms >= cardio.min_rr_ms) & (self.rr_ms <= cardio.max_rr_ms)
        self.no_contact = [(float(low), float(high)) for low, high in (session.no_contact or ())]
        for low, high in self.no_contact:
            valid &= ~((self.starts < high) & (self.ends > low))
        self.valid = valid
        self.corrected = self.rr_ms.copy()
        flagged = np.zeros(self.rr_ms.size, dtype=bool)
        for first, stop in _runs(valid):
            corrected, run_flagged = correct_rr(self.rr_ms[first:stop], cardio)
            self.corrected[first:stop] = corrected
            flagged[first:stop] = run_flagged
        self.interval_count = int(valid.sum())
        self.corrected_count = int(flagged[valid].sum())

    def heart_rate(self, start: float, end: float) -> float:
        first = int(np.searchsorted(self.starts, start - _TIME_TOLERANCE_SECONDS, side="left"))
        stop = int(np.searchsorted(self.ends, end + _TIME_TOLERANCE_SECONDS, side="right"))
        if stop - first < 2:
            return math.nan
        index = np.arange(first, stop)[self.valid[first:stop]]
        if index.size < 2:
            return math.nan
        if float(np.sum(self.rr_ms[index])) / 1000.0 < self.config.min_ecg_coverage * (end - start):
            return math.nan
        return 60000.0 / float(np.mean(self.corrected[index]))

    def no_contact_seconds(self, start: float, end: float) -> float:
        return float(
            sum(max(0.0, min(high, end) - max(low, start)) for low, high in self.no_contact)
        )


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


# --------------------------------------------------------------------------- #
# Per-epoch records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _EpochRecord:
    index: int
    start_time: float
    end_time: float
    has_optics: bool
    comparable: bool
    ecg_bpm: float
    decoder_bpm: float
    ppg_bpm: float
    ppg_confidence: float
    ppg_signal_quality: str
    series_quality: float
    blind: bool
    channels: Tuple[ChannelHeartRate, ...]
    points: Tuple[HeartRatePoint, ...]
    point_ecg_bpm: Tuple[float, ...]
    used: Tuple[HeartRatePoint, ...]
    changes: Tuple[SuddenHeartRateChange, ...]
    centres: Tuple[float, ...]
    centre_ecg_bpm: Tuple[float, ...]
    ecg_change: Optional[SuddenHeartRateChange]

    @property
    def ecg_change_bpm(self) -> float:
        return self.ecg_change.delta_bpm if self.ecg_change is not None else math.nan


class _Accumulator:
    def __init__(self, session: PolarSession, config: OpticsAgreementConfig) -> None:
        _require_r_peaks(session)
        self.config = config
        self.session = session
        self.ecg = _EcgReference(session, config)
        self.records: List[_EpochRecord] = []

    def add(self, epoch: SleepEpoch) -> None:
        config = self.config
        row = extract_ppg_features(epoch, config.ppg)
        channels = estimate_channel_heart_rates(epoch, config.ppg)
        half = config.ppg.hr_series_window_seconds / 2.0
        point_ecg = tuple(
            self.ecg.heart_rate(point.time - half, point.time + half) for point in row.hr_series
        )
        centres = tuple(point.time for point in row.hr_series) or _window_centres(epoch, config)
        centre_ecg = (
            point_ecg
            if row.hr_series
            else tuple(self.ecg.heart_rate(centre - half, centre + half) for centre in centres)
        )
        previous = self.records[-1] if self.records else None
        decoder_bpm = row.mean_hr_bpm if row.hr_source == "heart_rate" else math.nan
        record = _EpochRecord(
            index=epoch.index,
            start_time=epoch.start_time,
            end_time=epoch.end_time,
            has_optics=row.ppg_channel_count > 0,
            comparable=row.ppg_channel_count > 0 and row.ppg_coverage >= config.min_optics_coverage,
            ecg_bpm=self.ecg.heart_rate(epoch.start_time, epoch.end_time),
            decoder_bpm=decoder_bpm,
            ppg_bpm=row.ppg_estimated_hr_bpm,
            ppg_confidence=row.ppg_confidence,
            ppg_signal_quality=row.ppg_signal_quality,
            series_quality=row.hr_series_median_quality,
            blind=bool(
                row.hr_series
                and row.hr_series_skipped_count / len(row.hr_series) > config.hr_rule_blind_skipped_share
            ),
            channels=channels,
            points=row.hr_series,
            point_ecg_bpm=point_ecg,
            used=row.hr_series_used,
            changes=row.sudden_hr_changes,
            centres=centres,
            centre_ecg_bpm=centre_ecg,
            ecg_change=_ecg_change(centres, centre_ecg, previous, config),
        )
        self.records.append(record)

    def report(self, *, recording: str, synthetic: bool, preset: str) -> OpticsAgreementReport:
        config = self.config
        records = self.records
        if not any(record.has_optics for record in records):
            raise OpticsAgreementInputError(
                "the recording holds no optics samples to compare with the ECG; record "
                "with --preset p1034"
            )
        start = records[0].start_time
        end = records[-1].end_time
        no_contact_seconds = self.ecg.no_contact_seconds(start, end)
        compared = [record for record in records if record.comparable and math.isfinite(record.ecg_bpm)]
        agreement = _agreement_rows(records, config)
        outcomes = _rule_outcomes(records, config, gate=None)
        jumps = _jump_rows(records, outcomes, config)
        gates = _gate_rows(records, config)
        separation = _separation(records, outcomes, config)
        events = _events(records, outcomes, start)
        counts = {
            "epochs": len(records),
            "optics_epochs": sum(1 for record in records if record.has_optics),
            "ecg_reference_epochs": sum(1 for record in records if math.isfinite(record.ecg_bpm)),
            "compared_epochs": len(compared),
            "series_values": sum(len(record.points) for record in records),
            "series_values_with_reference": sum(
                1 for record in records for value in record.point_ecg_bpm if math.isfinite(value)
            ),
            "blind_epochs": sum(1 for record in records if record.blind),
            "r_peaks": self.ecg.peak_count,
            "ecg_intervals": self.ecg.interval_count,
            "ecg_intervals_corrected": self.ecg.corrected_count,
            "duration_seconds": end - start,
            "no_contact_seconds": no_contact_seconds,
        }
        limitations = _limitations(
            synthetic=synthetic,
            no_contact_seconds=no_contact_seconds,
            compared_epochs=len(compared),
            jumps=jumps,
            config=config,
        )
        metadata = {"preset": preset, "polar": _polar_summary(self.session)}
        return OpticsAgreementReport(
            recording=recording,
            synthetic=synthetic,
            config=config.to_dict(),
            counts=counts,
            agreement=agreement,
            jumps=jumps,
            gates=gates,
            separation=separation,
            events=events,
            limitations=limitations,
            metadata=metadata,
        )


def _window_centres(epoch: SleepEpoch, config: OpticsAgreementConfig) -> Tuple[float, ...]:
    """Centres of series windows over an epoch that has no series of its own."""
    window = config.ppg.hr_series_window_seconds
    step = config.ppg.hr_series_step_seconds
    centres = []
    centre = epoch.start_time + window / 2.0
    while centre + window / 2.0 <= epoch.end_time + _TIME_TOLERANCE_SECONDS:
        centres.append(centre)
        centre += step
    return tuple(centres)


def _ecg_change(
    centres: Sequence[float],
    values: Sequence[float],
    previous: Optional[_EpochRecord],
    config: OpticsAgreementConfig,
) -> Optional[SuddenHeartRateChange]:
    """The largest ECG heart-rate change with the rule's pairing, and the two window
    centres it lies between: windows of this epoch ``hr_series_compare_seconds`` to
    ``hr_series_max_compare_seconds`` apart, and windows of the previous epoch to this
    one's at most ``jump_link_seconds`` later. None when no pair has a reference at both
    ends."""
    shortest = config.ppg.hr_series_compare_seconds - _TIME_TOLERANCE_SECONDS
    longest = config.ppg.hr_series_max_compare_seconds + _TIME_TOLERANCE_SECONDS
    largest: Optional[SuddenHeartRateChange] = None
    pairs = [
        (centres[i], values[i], centres[j], values[j])
        for j in range(len(centres))
        for i in range(j)
        if shortest <= centres[j] - centres[i] <= longest
    ]
    if previous is not None and config.jump_link_seconds > 0:
        pairs += [
            (before, before_value, after, after_value)
            for after, after_value in zip(centres, values)
            for before, before_value in zip(previous.centres, previous.centre_ecg_bpm)
            if 0 < after - before <= config.jump_link_seconds + _TIME_TOLERANCE_SECONDS
        ]
    for before, before_value, after, after_value in pairs:
        if not (math.isfinite(before_value) and math.isfinite(after_value)):
            continue
        delta = after_value - before_value
        if largest is None or abs(delta) > abs(largest.delta_bpm):
            largest = SuddenHeartRateChange(
                start_time=before,
                end_time=after,
                before_bpm=before_value,
                after_bpm=after_value,
                delta_bpm=delta,
            )
    return largest


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #
def _agreement_rows(records: Sequence[_EpochRecord], config: OpticsAgreementConfig) -> Tuple[AgreementRow, ...]:
    rows: List[AgreementRow] = []
    optics = [record for record in records if record.comparable and math.isfinite(record.ecg_bpm)]

    def epoch_levels(record: _EpochRecord) -> Dict[str, str]:
        return {
            "ppg_signal_quality": record.ppg_signal_quality,
            "ppg_confidence": _bin_label(record.ppg_confidence, config.confidence_bins),
            "series_quality": _bin_label(record.series_quality, config.series_quality_bins),
        }

    for estimator, value in (
        ("decoder_heart_rate", lambda record: record.decoder_bpm),
        ("ppg_estimated_hr_bpm", lambda record: record.ppg_bpm),
    ):
        pairs = [(value(record), record.ecg_bpm, epoch_levels(record)) for record in optics]
        rows += _stratified(estimator, "", pairs, config)

    names = sorted({channel.channel for record in optics for channel in record.channels})
    for name in names:
        pairs = []
        for record in optics:
            estimate = next((channel for channel in record.channels if channel.channel == name), None)
            if estimate is None:
                pairs.append((math.nan, record.ecg_bpm, {"ppg_signal_quality": "Missing"}))
                continue
            pairs.append(
                (
                    estimate.estimated_hr_bpm,
                    record.ecg_bpm,
                    {
                        "ppg_signal_quality": estimate.signal_quality,
                        "ppg_confidence": _bin_label(estimate.confidence, config.confidence_bins),
                        "periodicity": _bin_label(estimate.periodicity, config.series_quality_bins),
                    },
                )
            )
        rows += _stratified("channel", name, pairs, config)

    used_pairs, skipped_pairs = [], []
    for record in records:
        # A series value by its epoch's quality fields, which the guard's gates read, and
        # by its own quality.
        epoch = {
            "ppg_signal_quality": record.ppg_signal_quality,
            "ppg_confidence": _bin_label(record.ppg_confidence, config.confidence_bins),
        }
        for point, reference in zip(record.points, record.point_ecg_bpm):
            if not math.isfinite(reference):
                continue
            levels = {**epoch, "series_quality": _bin_label(point.quality, config.series_quality_bins)}
            used_pairs.append((point.bpm if point.used else math.nan, reference, levels))
            skipped_pairs.append((math.nan if point.used else point.bpm, reference, levels))
    rows += _stratified("hr_series", "", used_pairs, config)
    rows += _stratified("hr_series_skipped", "", skipped_pairs, config)
    return tuple(rows)


def _stratified(
    estimator: str,
    channel: str,
    pairs: Sequence[Tuple[float, float, Mapping[str, str]]],
    config: OpticsAgreementConfig,
) -> List[AgreementRow]:
    rows = [_agreement(estimator, channel, "all", "all", pairs)]
    for stratum in STRATA[1:]:
        levels = [levels.get(stratum) for _estimate, _reference, levels in pairs]
        if all(level is None for level in levels):
            continue
        for level in _ordered_levels({level for level in levels if level is not None}, stratum, config):
            subset = [pair for pair, item in zip(pairs, levels) if item == level]
            rows.append(_agreement(estimator, channel, stratum, level, subset))
    return rows


def _agreement(
    estimator: str,
    channel: str,
    stratum: str,
    level: str,
    pairs: Sequence[Tuple[float, float, Mapping[str, str]]],
) -> AgreementRow:
    differences = np.asarray(
        [estimate - reference for estimate, reference, _levels in pairs if math.isfinite(estimate)],
        dtype=float,
    )
    bias = float(np.mean(differences)) if differences.size else math.nan
    sd = float(np.std(differences, ddof=1)) if differences.size >= 2 else math.nan
    return AgreementRow(
        estimator=estimator,
        channel=channel,
        stratum=stratum,
        level=level,
        n_reference=len(pairs),
        n=int(differences.size),
        bias_bpm=bias,
        sd_bpm=sd,
        loa_low_bpm=bias - 1.96 * sd if math.isfinite(sd) else math.nan,
        loa_high_bpm=bias + 1.96 * sd if math.isfinite(sd) else math.nan,
        mae_bpm=float(np.mean(np.abs(differences))) if differences.size else math.nan,
    )


def _bin_label(value: float, edges: Sequence[float]) -> str:
    if not math.isfinite(value):
        return _NONE_LEVEL
    for low, high in zip(edges[:-1], edges[1:]):
        if value < high or high == edges[-1]:
            return f"{low:g}-{high:g}"
    return f"{edges[-2]:g}-{edges[-1]:g}"


def _ordered_levels(levels: Iterable[str], stratum: str, config: OpticsAgreementConfig) -> List[str]:
    if stratum == "ppg_signal_quality":
        order = list(PPG_SIGNAL_QUALITIES)
    else:
        edges = config.confidence_bins if stratum == "ppg_confidence" else config.series_quality_bins
        order = [f"{low:g}-{high:g}" for low, high in zip(edges[:-1], edges[1:])] + [_NONE_LEVEL]
    known = [level for level in order if level in levels]
    return known + sorted(set(levels) - set(order))


# --------------------------------------------------------------------------- #
# The sudden heart-rate change rule against the ECG
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Outcome:
    fired: bool
    gated: bool
    counted: Tuple[SuddenHeartRateChange, ...]
    across: Optional[SuddenHeartRateChange]


def _rule_outcomes(
    records: Sequence[_EpochRecord],
    config: OpticsAgreementConfig,
    gate: Optional[Tuple[str, float]],
) -> Tuple[_Outcome, ...]:
    """The rule over consecutive epochs as ArousalGuard runs it on an optics preset: the
    row's changes from ``jump_bpm`` less the margin, plus the largest change from the
    previous epoch's used values within ``jump_link_seconds`` unless a counted row change
    in the same direction overlaps it. A gated epoch does not fire and its series does
    not reach the next epoch."""
    smallest = config.jump_bpm - config.jump_margin_bpm
    previous_used: Tuple[HeartRatePoint, ...] = ()
    outcomes = []
    for record in records:
        if gate is not None and not _gate_value(record, gate[0]) >= gate[1]:
            outcomes.append(_Outcome(False, True, (), None))
            previous_used = ()
            continue
        counted = tuple(change for change in record.changes if abs(change.delta_bpm) >= smallest)
        across = None
        if config.jump_link_seconds > 0 and previous_used and record.used:
            change = heart_rate_change_across_epochs(previous_used, record.used, config.jump_link_seconds)
            if (
                change is not None
                and abs(change.delta_bpm) >= smallest
                and not any(_same_step(change, other) for other in counted)
            ):
                across = change
        steps = len(counted) + (1 if across is not None else 0)
        outcomes.append(_Outcome(steps >= config.jump_pause_count, False, counted, across))
        previous_used = record.used
    return tuple(outcomes)


def _gate_value(record: _EpochRecord, name: str) -> float:
    if name == "ppg_confidence":
        return record.ppg_confidence
    if name == "hr_series_median_quality":
        return record.series_quality
    raise ValueError(f"unknown gate field {name}")


def _same_step(first: SuddenHeartRateChange, second: SuddenHeartRateChange) -> bool:
    return (first.delta_bpm > 0) == (second.delta_bpm > 0) and max(
        first.start_time, second.start_time
    ) <= min(first.end_time, second.end_time)


def _scored(record: _EpochRecord) -> bool:
    return record.has_optics and math.isfinite(record.ecg_change_bpm)


@dataclass(frozen=True)
class _Score:
    """One scored epoch. A fire counts for an ECG change of ``jump_bpm`` less the margin
    in its own epoch or the one before, as a step at an epoch boundary pauses the epoch
    after it; any other fire is a false jump. An ECG change of ``jump_bpm`` or more is one
    ECG jump, scored on the epoch it starts in: a change in the next epoch in the same
    direction whose windows overlap its own (the rule's test for one step seen from both
    sides of a boundary) is the same change. The jump is caught when one of its epochs or
    the one after them fires (#251 scored the rule the same way)."""

    record: _EpochRecord
    outcome: _Outcome
    false_jump: bool
    ecg_jump: bool
    caught: bool


def _scores(
    records: Sequence[_EpochRecord],
    outcomes: Sequence[_Outcome],
    config: OpticsAgreementConfig,
) -> List[_Score]:
    smallest = config.jump_bpm - config.jump_margin_bpm
    jumps = [_scored(record) and abs(record.ecg_change_bpm) >= config.jump_bpm for record in records]
    # A window over a step averages both rates, so the windows of the next epoch, paired
    # with this one's, see the same step again.
    goes_on = [
        index > 0
        and jumps[index]
        and jumps[index - 1]
        and _same_step(records[index - 1].ecg_change, records[index].ecg_change)
        for index in range(len(records))
    ]
    scores = []
    for index, (record, outcome) in enumerate(zip(records, outcomes)):
        if not _scored(record):
            continue
        near = abs(record.ecg_change_bpm)
        if index > 0 and math.isfinite(records[index - 1].ecg_change_bpm):
            near = max(near, abs(records[index - 1].ecg_change_bpm))
        ecg_jump = jumps[index] and not goes_on[index]
        caught = False
        if ecg_jump:
            last = index
            while last + 1 < len(records) and goes_on[last + 1]:
                last += 1
            caught = any(item.fired for item in outcomes[index:last + 2])
        scores.append(
            _Score(
                record=record,
                outcome=outcome,
                false_jump=outcome.fired and near < smallest,
                ecg_jump=ecg_jump,
                caught=caught,
            )
        )
    return scores


def _jump_row(stratum: str, level: str, scores: Sequence[_Score]) -> JumpRow:
    return JumpRow(
        stratum=stratum,
        level=level,
        epochs=len(scores),
        fired=sum(1 for score in scores if score.outcome.fired),
        false_jumps=sum(1 for score in scores if score.false_jump),
        ecg_jumps=sum(1 for score in scores if score.ecg_jump),
        missed=sum(1 for score in scores if score.ecg_jump and not score.caught),
    )


def _jump_rows(
    records: Sequence[_EpochRecord],
    outcomes: Sequence[_Outcome],
    config: OpticsAgreementConfig,
) -> Tuple[JumpRow, ...]:
    scores = _scores(records, outcomes, config)
    rows = [_jump_row("all", "all", scores)]
    for stratum, level_of in (
        ("ppg_signal_quality", lambda record: record.ppg_signal_quality),
        ("ppg_confidence", lambda record: _bin_label(record.ppg_confidence, config.confidence_bins)),
        ("series_quality", lambda record: _bin_label(record.series_quality, config.series_quality_bins)),
        ("hr_rule_blind", lambda record: "blind" if record.blind else "not blind"),
    ):
        levels = [level_of(score.record) for score in scores]
        order = (
            ["blind", "not blind"]
            if stratum == "hr_rule_blind"
            else _ordered_levels(set(levels), stratum, config)
        )
        for level in order:
            items = [score for score, item_level in zip(scores, levels) if item_level == level]
            if items:
                rows.append(_jump_row(stratum, level, items))
    return tuple(rows)


def _gate_rows(records: Sequence[_EpochRecord], config: OpticsAgreementConfig) -> Tuple[GateRow, ...]:
    rows = [_gate_row(records, _rule_outcomes(records, config, None), "none", 0.0, config)]
    for name in GATE_FIELDS:
        for threshold in config.gate_thresholds:
            outcomes = _rule_outcomes(records, config, (name, threshold))
            rows.append(_gate_row(records, outcomes, name, threshold, config))
    return tuple(rows)


def _gate_row(
    records: Sequence[_EpochRecord],
    outcomes: Sequence[_Outcome],
    name: str,
    threshold: float,
    config: OpticsAgreementConfig,
) -> GateRow:
    scores = _scores(records, outcomes, config)
    return GateRow(
        field=name,
        threshold=threshold,
        gated_epochs=sum(1 for score in scores if score.outcome.gated),
        fired=sum(1 for score in scores if score.outcome.fired),
        false_jumps=sum(1 for score in scores if score.false_jump),
        ecg_jumps=sum(1 for score in scores if score.ecg_jump),
        caught=sum(1 for score in scores if score.caught),
    )


def _separation(
    records: Sequence[_EpochRecord],
    outcomes: Sequence[_Outcome],
    config: OpticsAgreementConfig,
) -> Dict[str, object]:
    """How well each gate field tells the rule's true fires from its false ones: the
    medians and the AUC, the chance that a true fire has the higher value."""
    fired = [score for score in _scores(records, outcomes, config) if score.outcome.fired]
    result: Dict[str, object] = {}
    for name in GATE_FIELDS:
        true_values = [_gate_value(score.record, name) for score in fired if not score.false_jump]
        false_values = [_gate_value(score.record, name) for score in fired if score.false_jump]
        result[name] = {
            "true_fires": len(true_values),
            "false_fires": len(false_values),
            "median_true": _finite_or_none(_median(true_values)),
            "median_false": _finite_or_none(_median(false_values)),
            "auc": _finite_or_none(_auc(true_values, false_values)),
        }
    return result


def _events(
    records: Sequence[_EpochRecord],
    outcomes: Sequence[_Outcome],
    origin: float,
) -> Tuple[JumpEvent, ...]:
    by_time = {}
    for record in records:
        for centre, value in zip(record.centres, record.centre_ecg_bpm):
            by_time[round(centre, 6)] = value
    events = []
    for record, outcome in zip(records, outcomes):
        changes = [("row", change) for change in outcome.counted]
        if outcome.across is not None:
            changes.append(("across", outcome.across))
        for kind, change in changes:
            before = by_time.get(round(change.start_time, 6), math.nan)
            after = by_time.get(round(change.end_time, 6), math.nan)
            events.append(
                JumpEvent(
                    epoch_index=record.index,
                    kind=kind,
                    start_seconds=change.start_time - origin,
                    end_seconds=change.end_time - origin,
                    optics_delta_bpm=change.delta_bpm,
                    ecg_delta_bpm=after - before,
                )
            )
    return tuple(events)


def _limitations(
    *,
    synthetic: bool,
    no_contact_seconds: float,
    compared_epochs: int,
    jumps: Sequence[JumpRow],
    config: OpticsAgreementConfig,
) -> Tuple[AgreementLimitation, ...]:
    items = []
    if synthetic:
        items.append(
            AgreementLimitation(
                "synthetic_input",
                f"{SYNTHETIC_NOTE}: the recording and the H10 session were generated from the "
                "same scripted beats, so the agreement checks the code, not the headband.",
            )
        )
    if no_contact_seconds > 0:
        items.append(
            AgreementLimitation(
                "no_contact_seconds",
                f"{no_contact_seconds:.0f} s of the recording had no H10 skin contact or ECG "
                "noise like it; no reference there.",
                {"seconds": no_contact_seconds},
            )
        )
    if compared_epochs < config.min_epochs:
        items.append(
            AgreementLimitation(
                "n_epochs_small",
                f"{compared_epochs} compared epochs, fewer than {config.min_epochs}: the limits "
                "of agreement are wide estimates.",
                {"compared_epochs": compared_epochs, "min_epochs": config.min_epochs},
            )
        )
    overall = jumps[0] if jumps else None
    events = (overall.fired + overall.ecg_jumps) if overall is not None else 0
    if events < config.min_jump_events:
        items.append(
            AgreementLimitation(
                "few_jump_events",
                f"{events} fired epochs and ECG jumps together, fewer than "
                f"{config.min_jump_events}: the false-jump and miss rates rest on few events.",
                {"events": events, "min_jump_events": config.min_jump_events},
            )
        )
    return tuple(items)


def _polar_summary(session: PolarSession) -> Dict[str, object]:
    """Alignment and contact figures of the H10 session; no device name or address."""
    quality = session.quality or {}
    alignment = session.alignment or {}
    mapping = alignment.get("mapping") or {}
    rr = alignment.get("rr") or {}
    return _json_ready(
        {
            "r_peaks": quality.get("r_peaks"),
            "no_contact_seconds": quality.get("no_contact_seconds"),
            "ecg_noise_spans": quality.get("ecg_noise_spans"),
            "clock_segments": len(alignment.get("clock_segments") or ()),
            "drift_ppm": mapping.get("drift_ppm"),
            "delay_median_ms": mapping.get("delay_median_ms"),
            "ntp_step_ms": alignment.get("ntp_step_ms"),
            "rr_matched": rr.get("matched"),
            "rr_beats": rr.get("beats"),
        }
    )


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
_ESTIMATOR_NAMES = {
    "decoder_heart_rate": "decoder `heart_rate`",
    "ppg_estimated_hr_bpm": "`ppg_estimated_hr_bpm`",
    "channel": "channel",
    "hr_series": "`heart_rate_series`, used",
    "hr_series_skipped": "`heart_rate_series`, skipped",
}
_STRATUM_TITLES = {
    "ppg_signal_quality": "By `ppg_signal_quality`",
    "ppg_confidence": "By `ppg_confidence`",
    "series_quality": "By the series' quality (epoch median, or the value's own)",
    "periodicity": "By channel periodicity",
}


def _markdown(report: OpticsAgreementReport) -> str:
    counts = report.counts
    config = report.config
    title = f"# Optics heart-rate agreement: {report.recording or 'recording'}"
    lines = [title, ""]
    if report.synthetic:
        lines += [f"**{SYNTHETIC_NOTE.capitalize()}.**", ""]
    lines += [
        "Heart rate from the Athena forehead optics against heart rate from Polar H10 ECG "
        "R-peaks recorded with it, per epoch and per window of the heart-rate series. This is "
        "agreement with a chest ECG, a proxy for heart rate; no PSG takes part.",
        "",
        f"- Epochs: {counts.get('epochs', 0)} of {_number(config.get('epoch_seconds'))} s, "
        f"{counts.get('optics_epochs', 0)} with optics, {counts.get('compared_epochs', 0)} "
        "compared with an ECG reference.",
        f"- ECG: {counts.get('r_peaks', 0)} R-peaks, {counts.get('ecg_intervals_corrected', 0)} "
        f"of {counts.get('ecg_intervals', 0)} beat intervals corrected; "
        f"{_number(counts.get('no_contact_seconds'), 0)} s without contact.",
        f"- Series: {counts.get('series_values', 0)} values, "
        f"{counts.get('series_values_with_reference', 0)} with an ECG reference; "
        f"{counts.get('blind_epochs', 0)} epochs skipped more than half of theirs.",
        "",
        "## Agreement",
        "",
        "Differences are optics minus ECG in bpm; the 95 % limits of agreement are the bias "
        "plus and minus 1.96 SD. Coverage is the share of the windows with an ECG reference "
        "that have an estimate.",
        "",
    ]
    lines += _agreement_table([row for row in report.agreement if row.stratum == "all"], with_level=False)
    for stratum in STRATA[1:]:
        rows = [row for row in report.agreement if row.stratum == stratum and row.n]
        if rows:
            lines += ["", f"### {_STRATUM_TITLES[stratum]}", ""]
            lines += _agreement_table(rows, with_level=True)

    jump_bpm = _number(config.get("jump_bpm"))
    smallest = _number(_float(config.get("jump_bpm")) - _float(config.get("jump_margin_bpm")))
    ppg = config.get("ppg") or {}
    lines += [
        "",
        "## Sudden heart-rate changes against the ECG",
        "",
        f"The rule as the arousal guard runs it by default: series changes of {smallest} bpm "
        f"or more between values {_number(ppg.get('hr_series_compare_seconds'))} to "
        f"{_number(ppg.get('hr_series_max_compare_seconds'))} s apart, and from the previous "
        f"epoch within {_number(config.get('jump_link_seconds'))} s. The ECG change of an epoch is its "
        "largest between the windows the rule pairs. A fired epoch is a false jump when "
        f"neither it nor the epoch before changed by {smallest} bpm or more on the ECG. An "
        f"ECG change of {jump_bpm} bpm or more counts once, also when the next epoch sees it "
        "again over overlapping windows, and is missed when none of its epochs and not the "
        "one after them fired. Epochs without an ECG change to compare are not scored.",
        "",
        "| Stratum | Level | Epochs | Fired | False jumps | ECG jumps | Missed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.jumps:
        lines.append(
            f"| {row.stratum} | {row.level} | {row.epochs} | {row.fired} | "
            f"{row.false_jumps}{_rate(row.false_jump_rate)} | {row.ecg_jumps} | "
            f"{row.missed}{_rate(row.miss_rate)} |"
        )
    lines += [
        "",
        "### Quality gates for the HR rule",
        "",
        "Epochs under the threshold neither fire nor feed the next epoch's comparison, as "
        "`min_ppg_confidence_for_hr_rule` and `min_hr_series_quality_for_hr_rule` make the "
        "guard do.",
        "",
        "| Gate | Threshold | Gated | Fired | False jumps | ECG jumps caught |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.gates:
        threshold = "" if row.field == "none" else _number(row.threshold, 1)
        lines.append(
            f"| {row.field} | {threshold} | {row.gated_epochs} | {row.fired} | "
            f"{row.false_jumps}{_rate(row.false_jump_rate)} | {row.caught} of {row.ecg_jumps} |"
        )
    lines += [
        "",
        "Separation of true and false fires (AUC: the chance that a true fire has the higher "
        "value):",
        "",
        "| Field | True fires | False fires | Median, true | Median, false | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in GATE_FIELDS:
        item = report.separation.get(name) or {}
        lines.append(
            f"| {name} | {item.get('true_fires', 0)} | {item.get('false_fires', 0)} | "
            f"{_number(item.get('median_true'), 2)} | {_number(item.get('median_false'), 2)} | "
            f"{_number(item.get('auc'), 2)} |"
        )
    if report.events:
        lines += [
            "",
            "### Counted changes",
            "",
            "| Epoch | Kind | From (s) | To (s) | Optics (bpm) | ECG (bpm) |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for event in report.events:
            lines.append(
                f"| {event.epoch_index} | {event.kind} | {event.start_seconds:.0f} | "
                f"{event.end_seconds:.0f} | {event.optics_delta_bpm:+.1f} | "
                f"{_signed(event.ecg_delta_bpm)} |"
            )
    lines += ["", "## Limitations", ""]
    if report.limitations:
        lines += [f"- `{item.code}`: {item.message}" for item in report.limitations]
    else:
        lines.append("- None recorded.")
    return "\n".join(lines) + "\n"


def _agreement_table(rows: Sequence[AgreementRow], *, with_level: bool) -> List[str]:
    header = "| Estimator | Channel |" + (" Level |" if with_level else "")
    header += " Windows | Coverage | Bias | SD | 95 % limits | MAE |"
    rule = "|---|---|" + ("---|" if with_level else "") + "---:|---:|---:|---:|---|---:|"
    lines = [header, rule]
    for row in rows:
        limits = (
            f"{row.loa_low_bpm:+.1f} to {row.loa_high_bpm:+.1f}"
            if math.isfinite(row.loa_low_bpm)
            else ""
        )
        level = f" {row.level} |" if with_level else ""
        lines.append(
            f"| {_ESTIMATOR_NAMES.get(row.estimator, row.estimator)} | {row.channel} |{level} "
            f"{row.n_reference} | {_percent(row.coverage)} | {_signed(row.bias_bpm)} | "
            f"{_number(row.sd_bpm, 1)} | {limits} | {_number(row.mae_bpm, 1)} |"
        )
    return lines


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _median(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def _auc(higher: Sequence[float], lower: Sequence[float]) -> float:
    first = [value for value in higher if math.isfinite(value)]
    second = [value for value in lower if math.isfinite(value)]
    if not first or not second:
        return math.nan
    wins = sum((a > b) + 0.5 * (a == b) for a in first for b in second)
    return float(wins / (len(first) * len(second)))


def _finite_or_none(value) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _float(value) -> float:
    return math.nan if value is None else float(value)


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _number(value, digits: Optional[int] = None) -> str:
    """``value`` with ``digits`` decimals, or as short as it goes without them; "" for
    a missing or non-finite value."""
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:g}" if digits is None else f"{value:.{digits}f}"


def _signed(value: float) -> str:
    return f"{value:+.1f}" if value is not None and math.isfinite(value) else ""


def _percent(value: float) -> str:
    return f"{100.0 * value:.0f} %" if math.isfinite(value) else ""


def _rate(value: float) -> str:
    return f" ({100.0 * value:.0f} %)" if math.isfinite(value) else ""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
