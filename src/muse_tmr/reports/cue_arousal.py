"""Cue-locked arousal proxies with paired quiet controls (#227).

For every scheduler ``play`` of a night, and for a quiet control time paired with it,
this scores what the recording shows right after that moment against the 20 s before
it: a rise of alpha (8 to 12 Hz), beta (15 to 30 Hz) or high-frequency EMG-band power
(55 to 95 Hz, the 64 Hz optics line bridged) in the EEG, a movement the IMU features call
an arousal proxy, and a heart-rate jump in the optics heart-rate series. A control is a
moment in the same gate state with no ``play`` within 60 s, scored with the same windows,
so a cue's rate can be read against the night's own background rate.

It is retrospective: it reads the night's logs (scheduler events, audio playback, the
gate log, awakening markers, the summary) and one ``ReplaySession`` pass over the raw
recording, and never runs during the night. It does not decide or play anything and
imports nothing from ``muse_tmr.audio``.

Every flag is an arousal proxy, not a scored arousal: forehead EEG has no occipital
alpha and no chin EMG, nothing here was compared with PSG-scored arousals, the rise
ratios and the heart-rate jump are UNFITTED starting values, and a cue's volume is a
software gain from 0 to 1, not a sound level. Nothing here measures a benefit.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import AsyncIterable, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import detrend

from muse_tmr.ble.muse_athena_protocol import COUNTER_MODULO, SENSOR_CONFIG
from muse_tmr.ble.muse_raw_stream import MuseRawStream
from muse_tmr.data.replay import ReplayConfig, ReplaySession, resolve_raw_path
from muse_tmr.data.sample_types import EEGSample, MuseFrame
from muse_tmr.features.complexity_features import bridge_lines
from muse_tmr.features.epochs import SleepEpoch
from muse_tmr.features.imu_features import IMUFeatureConfig, extract_imu_features
from muse_tmr.features.ppg_features import PPGFeatureConfig, heart_rate_series
from muse_tmr.reports.analysis import AnalysisLimitation
from muse_tmr.validation.gate_log import (
    REM_GATE_DECISIONS_FILENAME,
    GateDecisionRecord,
    GateOpenRun,
    emitting_record,
    gate_open_runs,
    load_rem_gate_decisions,
    run_at,
)
from muse_tmr.validation.pilot4 import AwakeningEvent, load_awakening_events

CUE_AROUSAL_SCHEMA_VERSION = 1
AROUSAL_FLAGS = ("alpha", "beta", "emg", "imu", "hr")
SUMMARY_FLAGS = AROUSAL_FLAGS + ("any",)
WINDOW_KINDS = ("cue", "control")
SYNTHETIC_NOTE = "synthetic input, not sleep"
# Thresholds set by hand that no night has been fitted to; reports say so until a pull
# request carries real-night numbers.
UNFITTED_THRESHOLDS = ("alpha_rise_ratio", "beta_rise_ratio", "emg_rise_ratio", "hr_jump_bpm")
SILENT_BACKENDS = ("dry-run", "mock")
EVENTS_FILENAME = "cue_arousal_events.jsonl"
REPORT_FILENAME = "cue_arousal_report.json"
MARKDOWN_FILENAME = "cue_arousal_report.md"
SUMMARY_FILENAMES = ("pilot4_summary.json", "pilot5_summary.json")
# A cue decided at the end of a 30 s epoch, when no gate log says how long its epoch was.
_DEFAULT_EPOCH_SECONDS = 30.0
_FRAME_LEAD_SECONDS = 1.0
_TIME_TOLERANCE_SECONDS = 1e-6
# The decoder reports a loss of up to this many subpackets of a TAG; a longer one reads
# as a restarted stream (muse_athena_protocol.SubpacketSequence).
_REPORTED_LOSS_SUBPACKETS = COUNTER_MODULO // 2 - 1
# Samples in one EEG subpacket, by the layout's channel count.
_EEG_SUBPACKET_SAMPLES = {
    channels: samples for kind, channels, samples, *_ in SENSOR_CONFIG.values() if kind == "EEG"
}


@dataclass(frozen=True)
class CueArousalConfig:
    """Windows and thresholds of the cue-locked arousal proxies.

    A window is ``baseline_seconds`` before the cue and ``post_window_seconds`` from it.
    The EEG is cut into ``segment_seconds`` Hann segments every ``segment_step_seconds``,
    each with the optics line (``line_hz``, searched within ``line_search_hz``) fitted
    and taken out, and band powers are averaged over the usable channels and over
    ``subwindow_seconds`` (an AASM arousal lasts at least 3 s), stepped by
    ``subwindow_step_seconds``. A band rises when the largest sub-window after the cue is
    at least its ratio times the median sub-window before it; the EMG band has the line
    bins bridged first (``complexity_features.bridge_lines``). The IMU flag is an
    arousal-proxy movement in ``imu``'s cue window (2 s before to 10 s after), with
    ``imu``'s thresholds. The HR flag is a used value of the heart-rate series (``ppg``)
    after the cue at least ``hr_jump_bpm`` above the median of those before it.

    UNFITTED: ``alpha_rise_ratio``, ``beta_rise_ratio``, ``emg_rise_ratio`` and
    ``hr_jump_bpm`` are starting values set by hand, not fitted on any night. No
    synthetic result may change them (#227).
    """

    post_window_seconds: float = 10.0
    baseline_seconds: float = 20.0
    control_exclusion_seconds: float = 60.0
    control_spacing_seconds: float = 30.0
    eeg_channels: Tuple[str, ...] = ("TP9", "AF7", "AF8", "TP10")
    eeg_sample_rate_hz: float = 256.0
    alpha_band_hz: Tuple[float, float] = (8.0, 12.0)
    beta_band_hz: Tuple[float, float] = (15.0, 30.0)
    emg_band_hz: Tuple[float, float] = (55.0, 95.0)
    line_hz: Tuple[float, ...] = (64.0,)
    line_search_hz: float = 0.5
    line_half_width_hz: float = 1.5
    segment_seconds: float = 1.0
    segment_step_seconds: float = 0.5
    subwindow_seconds: float = 3.0
    subwindow_step_seconds: float = 1.0
    alpha_rise_ratio: float = 2.0
    beta_rise_ratio: float = 2.0
    emg_rise_ratio: float = 2.0
    hr_jump_bpm: float = 10.0
    min_hr_values: int = 3
    min_window_coverage: float = 0.8
    max_segment_missing_fraction: float = 0.1
    clipping_uv: float = 500.0
    clipping_fraction: float = 0.05
    flat_std_uv: float = 1e-6
    # How late an EEG frame may come before it starts after a gap, where the subpacket
    # counter does not allow more (an unknown layout; see _gap_lateness()).
    clock_slack_seconds: float = 0.5
    awakening_window_seconds: float = 120.0
    rem_termination_seconds: float = 60.0
    min_cues_for_rates: int = 10
    volume_bin_edges: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
    interval_bin_edges_seconds: Tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)
    imu: IMUFeatureConfig = field(default_factory=IMUFeatureConfig)
    ppg: PPGFeatureConfig = field(default_factory=PPGFeatureConfig)

    def validate(self) -> None:
        for name in (
            "post_window_seconds",
            "baseline_seconds",
            "eeg_sample_rate_hz",
            "segment_seconds",
            "segment_step_seconds",
            "subwindow_seconds",
            "subwindow_step_seconds",
            "hr_jump_bpm",
            "clock_slack_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("alpha_rise_ratio", "beta_rise_ratio", "emg_rise_ratio"):
            if getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be above 1")
        if self.control_exclusion_seconds < 0 or self.control_spacing_seconds < 0:
            raise ValueError("control distances must be non-negative")
        if self.subwindow_seconds < self.segment_seconds:
            raise ValueError("subwindow_seconds must hold at least one segment")
        if self.subwindow_seconds > min(self.baseline_seconds, self.post_window_seconds):
            raise ValueError("subwindow_seconds must fit in the baseline and the post window")
        nyquist = self.eeg_sample_rate_hz / 2.0
        for name in ("alpha_band_hz", "beta_band_hz", "emg_band_hz"):
            low, high = getattr(self, name)
            if not 0 <= low < high <= nyquist:
                raise ValueError(f"{name} must be inside (0, nyquist)")
        if not 0.0 < self.min_window_coverage <= 1.0:
            raise ValueError("min_window_coverage must be inside (0, 1]")
        if not 0.0 <= self.max_segment_missing_fraction < 1.0:
            raise ValueError("max_segment_missing_fraction must be in [0, 1)")
        if self.min_hr_values < 1:
            raise ValueError("min_hr_values must be at least 1")
        if list(self.volume_bin_edges) != sorted(self.volume_bin_edges):
            raise ValueError("volume_bin_edges must increase")
        if list(self.interval_bin_edges_seconds) != sorted(self.interval_bin_edges_seconds):
            raise ValueError("interval_bin_edges_seconds must increase")
        self.imu.validate()
        self.ppg.validate()

    def to_dict(self) -> Dict[str, object]:
        payload = _json_safe(asdict(self))
        payload["unfitted_thresholds"] = list(UNFITTED_THRESHOLDS)
        return payload

    @property
    def span_before_seconds(self) -> float:
        """How long before a cue its frames are needed: the baseline and half a heart-rate
        series window, so the series has values centred in the whole baseline."""
        return max(
            self.baseline_seconds + self.ppg.hr_series_window_seconds / 2.0,
            self.imu.cue_pre_window_seconds,
        )

    @property
    def span_after_seconds(self) -> float:
        return max(
            self.post_window_seconds + self.ppg.hr_series_window_seconds / 2.0,
            self.imu.cue_post_window_seconds,
        )


@dataclass(frozen=True)
class CueArousalWindow:
    """The scores of one cue or control window, a line of ``cue_arousal_events.jsonl``.

    ``offset_seconds`` is the moment scored, from the first epoch's start (the
    scheduler's time base): for a cue the end of the epoch whose decision played it.
    ``time`` is the same moment in the recording's clock (Unix seconds). ``flags`` holds
    each arousal proxy, None where the window could not be scored, and ``any``.
    ``cue`` carries a cue's own logs and is empty for a control.
    """

    kind: str
    index: int
    time: float
    flags: Mapping[str, Optional[bool]]
    values: Mapping[str, object] = field(default_factory=dict)
    quality_flags: Tuple[str, ...] = ()
    offset_seconds: Optional[float] = None
    paired_index: Optional[int] = None
    gate_open: Optional[bool] = None
    rem_period_index: Optional[int] = None
    seconds_since_gate_open: Optional[float] = None
    rem_period_ends_in_seconds: Optional[float] = None
    awakening_in_seconds: Optional[float] = None
    cue: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in WINDOW_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(WINDOW_KINDS)}")
        object.__setattr__(self, "quality_flags", tuple(dict.fromkeys(self.quality_flags)))

    @property
    def played(self) -> bool:
        """A cue the player played as logged, or with no audio log; never a control."""
        return self.kind == "cue" and bool(self.cue.get("played", True))

    def to_dict(self) -> Dict[str, object]:
        return _json_safe(
            {
                "kind": self.kind,
                "index": self.index,
                "offset_seconds": self.offset_seconds,
                "time": self.time,
                "paired_index": self.paired_index,
                "gate_open": self.gate_open,
                "rem_period_index": self.rem_period_index,
                "seconds_since_gate_open": self.seconds_since_gate_open,
                "rem_period_ends_in_seconds": self.rem_period_ends_in_seconds,
                "awakening_in_seconds": self.awakening_in_seconds,
                "flags": dict(self.flags),
                "values": dict(self.values),
                "quality_flags": list(self.quality_flags),
                "cue": dict(self.cue),
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CueArousalWindow":
        return cls(
            kind=str(payload["kind"]),
            index=int(payload["index"]),
            time=float(payload["time"]),
            flags={str(key): value for key, value in dict(payload.get("flags", {}) or {}).items()},
            values=dict(payload.get("values", {}) or {}),
            quality_flags=tuple(str(item) for item in payload.get("quality_flags", ()) or ()),
            offset_seconds=_optional_float(payload.get("offset_seconds")),
            paired_index=_optional_int(payload.get("paired_index")),
            gate_open=payload.get("gate_open"),  # type: ignore[arg-type]
            rem_period_index=_optional_int(payload.get("rem_period_index")),
            seconds_since_gate_open=_optional_float(payload.get("seconds_since_gate_open")),
            rem_period_ends_in_seconds=_optional_float(payload.get("rem_period_ends_in_seconds")),
            awakening_in_seconds=_optional_float(payload.get("awakening_in_seconds")),
            cue=dict(payload.get("cue", {}) or {}),
        )


@dataclass(frozen=True)
class CueArousalReport:
    """Cue-locked arousal proxies of one night, descriptive and versioned."""

    recording: str
    windows: Tuple[CueArousalWindow, ...]
    outcomes: Mapping[str, object]
    dose_tables: Mapping[str, object]
    limitations: Tuple[AnalysisLimitation, ...]
    rem_periods: Tuple[Mapping[str, object], ...] = ()
    config: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    synthetic: bool = False
    generated_at_utc: str = ""
    schema_version: int = CUE_AROUSAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", tuple(self.windows))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "rem_periods", tuple(self.rem_periods))
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())

    @property
    def limitation_codes(self) -> Tuple[str, ...]:
        return tuple(item.code for item in self.limitations)

    @property
    def cues(self) -> Tuple[CueArousalWindow, ...]:
        return tuple(window for window in self.windows if window.kind == "cue")

    @property
    def controls(self) -> Tuple[CueArousalWindow, ...]:
        return tuple(window for window in self.windows if window.kind == "control")

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        if self.synthetic:
            payload.update({"synthetic": True, "note": SYNTHETIC_NOTE})
        payload.update(
            {
                "schema_version": self.schema_version,
                "report": "cue_arousal",
                "recording": self.recording,
                "generated_at_utc": self.generated_at_utc,
                "outcomes": dict(self.outcomes),
                "dose_tables": dict(self.dose_tables),
                "rem_periods": [dict(period) for period in self.rem_periods],
                "limitations": [limitation.to_dict() for limitation in self.limitations],
                "config": dict(self.config),
                "metadata": dict(self.metadata),
                "windows": [window.to_dict() for window in self.windows],
            }
        )
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CueArousalReport":
        return cls(
            recording=str(payload.get("recording", "")),
            windows=tuple(CueArousalWindow.from_dict(item) for item in payload.get("windows", ()) or ()),
            outcomes=dict(payload.get("outcomes", {}) or {}),
            dose_tables=dict(payload.get("dose_tables", {}) or {}),
            limitations=tuple(
                AnalysisLimitation.from_dict(item) for item in payload.get("limitations", ()) or ()
            ),
            rem_periods=tuple(dict(item) for item in payload.get("rem_periods", ()) or ()),
            config=dict(payload.get("config", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
            synthetic=payload.get("synthetic") is True,
            generated_at_utc=str(payload.get("generated_at_utc", "")),
            schema_version=int(payload.get("schema_version", CUE_AROUSAL_SCHEMA_VERSION)),
        )

    @classmethod
    def load(cls, input_path: Path) -> "CueArousalReport":
        return cls.from_dict(json.loads(Path(input_path).expanduser().read_text(encoding="utf-8")))

    def save(self, output_path: Path) -> Path:
        return _write_text(
            output_path,
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    def save_markdown(self, output_path: Path) -> Path:
        return _write_text(output_path, self.to_markdown())

    def save_events(self, output_path: Path) -> Path:
        lines = [
            json.dumps(window.to_dict(), sort_keys=True, allow_nan=False)
            for window in sorted(self.windows, key=lambda window: (window.time, window.kind))
        ]
        return _write_text(output_path, "".join(line + "\n" for line in lines))

    def save_outputs(self, output_dir: Path) -> Dict[str, Path]:
        """``cue_arousal_events.jsonl``, ``cue_arousal_report.json`` and ``.md`` in ``output_dir``."""
        output_dir = Path(output_dir).expanduser()
        return {
            "events_path": self.save_events(output_dir / EVENTS_FILENAME),
            "report_path": self.save(output_dir / REPORT_FILENAME),
            "markdown_path": self.save_markdown(output_dir / MARKDOWN_FILENAME),
        }

    def summary_block(self, paths: Optional[Mapping[str, Path]] = None) -> Dict[str, object]:
        """The figures a Pilot 4/5 summary embeds as ``cue_arousal``."""
        outcomes = self.outcomes
        block: Dict[str, object] = {"status": "scored", "schema_version": self.schema_version}
        for key in (
            "cue_count",
            "played_cue_count",
            "scored_cue_count",
            "control_count",
            "scored_control_count",
            "arousal_proxy_per_cue",
            "arousal_proxy_per_control",
            "arousal_proxy_difference",
            "movement_events_per_cue",
            "movement_events_per_control",
            "cue_attributed_awakening_rate",
            "rem_periods_with_disruption_fraction",
            "rem_termination_within_60s_of_cue_rate",
            "rem_termination_within_60s_of_control_rate",
        ):
            block[key] = outcomes.get(key)
        block["limitations"] = list(self.limitation_codes)
        if self.synthetic:
            block.update({"synthetic": True, "note": SYNTHETIC_NOTE})
        for name, path in dict(paths or {}).items():
            block[name] = str(path)
        return _json_safe(block)

    def to_markdown(self) -> str:
        outcomes = self.outcomes
        lines = [f"# Cue-locked arousal proxies: {self.recording}", ""]
        if self.synthetic:
            lines += [f"**{SYNTHETIC_NOTE.capitalize()}.** Nothing here is about sleep.", ""]
        lines += [
            "Arousal proxies, not scored arousals. The thresholds are UNFITTED and the volume "
            "is a software gain, not a sound level. Descriptive only.",
            "",
            "## Summary",
            f"- Cues: {outcomes.get('cue_count', 0)} logged, {outcomes.get('played_cue_count', 0)} played, "
            f"{outcomes.get('scored_cue_count', 0)} scored; controls: {outcomes.get('control_count', 0)}, "
            f"{outcomes.get('scored_control_count', 0)} scored",
            "",
            "| Flag | Per cue | Per control | Difference |",
            "| --- | ---: | ---: | ---: |",
        ]
        counts = dict(outcomes.get("flag_counts", {}) or {})
        differences = dict(outcomes.get("arousal_proxy_difference", {}) or {})
        for flag in SUMMARY_FLAGS:
            cue = dict(counts.get("cue", {}) or {}).get(flag, {})
            control = dict(counts.get("control", {}) or {}).get(flag, {})
            lines.append(
                f"| {flag} | {_count_text(cue)} | {_count_text(control)} | "
                f"{_points_text(differences.get(flag))} |"
            )
        lines += [
            "",
            f"- Movement events per cue: {_number_text(outcomes.get('movement_events_per_cue'))}; "
            f"per control: {_number_text(outcomes.get('movement_events_per_control'))}",
            f"- Cue-attributed awakening rate (within "
            f"{_number_text(self.config.get('awakening_window_seconds'), 0)} s): "
            f"{_rate_text(outcomes.get('cue_attributed_awakening_rate'))}",
            f"- REM periods (gate-open runs): {outcomes.get('rem_period_count', 'n/a')}, "
            f"with a cue: {outcomes.get('cued_rem_period_count', 'n/a')}, with any disruption: "
            f"{_rate_text(outcomes.get('rem_periods_with_disruption_fraction'))}",
            f"- REM period ended within {_number_text(self.config.get('rem_termination_seconds'), 0)} s: "
            f"after cues {_rate_text(outcomes.get('rem_termination_within_60s_of_cue_rate'))}, "
            f"after controls {_rate_text(outcomes.get('rem_termination_within_60s_of_control_rate'))}",
            "",
            "## Dose tables",
        ]
        titles = (
            ("effective_volume", "By effective volume (software gain)"),
            ("inter_cue_interval", "By time since the previous cue"),
            ("cue_index", "By cue index in its block"),
        )
        for key, title in titles:
            lines += ["", f"### {title}", "", "| Bin | Cues | Any flag |", "| --- | ---: | ---: |"]
            rows = list(self.dose_tables.get(key, ()) or ())
            if not rows:
                lines.append("| none | 0 | n/a |")
            for row in rows:
                lines.append(
                    f"| {row['bin']} | {row['cues']} | "
                    f"{_count_text({'flagged': row['any_flagged'], 'scored': row['any_scored']})} |"
                )
        lines += ["", "## Limitations"]
        if self.limitations:
            lines.extend(f"- `{item.code}`: {item.message}" for item in self.limitations)
        else:
            lines.append("- No limitations recorded.")
        return "\n".join(lines) + "\n"


def load_cue_arousal_report(input_path: Path) -> CueArousalReport:
    return CueArousalReport.load(input_path)


# --------------------------------------------------------------------------- #
# Scoring a night from its logs
# --------------------------------------------------------------------------- #
async def score_cue_arousal(
    recording_dir: Path,
    *,
    config: Optional[CueArousalConfig] = None,
    pilot3_report: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    scheduler_events_path: Optional[Path] = None,
    audio_log_path: Optional[Path] = None,
    gate_log_path: Optional[Path] = None,
    awakening_events_path: Optional[Path] = None,
    replay_input: Optional[Path] = None,
    optics_capable: Optional[bool] = None,
) -> CueArousalReport:
    """Score every cue of a night, and a quiet control paired with each.

    ``recording_dir`` is a Pilot 4/5 output directory; its ``scheduler_events.jsonl``,
    ``audio_playback.jsonl``, ``rem_gate_decisions.jsonl``, ``awakening_events.jsonl`` and
    summary are read unless another path is given. With ``pilot3_report`` the cues are
    that report's planned plays and injected cues, its ``epochs`` are the gate log and
    ``recording_dir`` is the recording it replayed. The windows are cut from
    ``replay_input`` (default ``recording_dir``) in one replay pass. ``optics_capable``
    None reads the preset from the recording's ``metadata.json``.
    """
    config = config or CueArousalConfig()
    config.validate()
    recording_dir = Path(recording_dir).expanduser()
    raw_path = resolve_raw_path(Path(replay_input).expanduser() if replay_input else recording_dir)
    inputs = _night_inputs(
        recording_dir,
        pilot3_report=pilot3_report,
        summary_path=summary_path,
        scheduler_events_path=scheduler_events_path,
        audio_log_path=audio_log_path,
        gate_log_path=gate_log_path,
        awakening_events_path=awakening_events_path,
    )
    first_epoch_start = inputs.first_epoch_start
    time_base_inferred = first_epoch_start is None
    if first_epoch_start is None:
        first_epoch_start = await _first_frame_time(raw_path)
    if optics_capable is None:
        optics_capable = _optics_capable_from_metadata(raw_path.parent)

    runs = gate_open_runs(inputs.gate_records)
    cues = _cue_targets(inputs, first_epoch_start, config)
    controls = _control_targets(cues, inputs.gate_records, first_epoch_start, config)
    cues = _pair(cues, controls)
    targets = cues + controls
    windows: List[CueArousalWindow] = []
    if targets:
        windows = await _score_targets_from_recording(raw_path, targets, config, optics_capable)
    awakening_offsets, awakening_notes = _awakening_offsets(inputs.awakenings, first_epoch_start)
    windows = [
        _with_night_context(window, runs, awakening_offsets, config)
        for window in sorted(windows, key=lambda window: (window.time, window.kind))
    ]
    outcomes = _outcomes(windows, runs, awakening_offsets, inputs, config)
    synthetic = _says_synthetic(raw_path.parent / "metadata.json")
    limitations = _limitations(
        windows,
        inputs=inputs,
        runs=runs,
        config=config,
        optics_capable=bool(optics_capable),
        synthetic=synthetic,
        time_base_inferred=time_base_inferred,
        awakening_notes=awakening_notes,
    )
    return CueArousalReport(
        recording=recording_dir.name,
        windows=tuple(windows),
        outcomes=outcomes,
        dose_tables=_dose_tables(windows, config),
        limitations=limitations,
        rem_periods=tuple(_rem_period_row(run, windows, config) for run in runs),
        config=config.to_dict(),
        metadata={
            "first_epoch_start": first_epoch_start,
            "time_base": "inferred_from_first_frame" if time_base_inferred else "logged",
            "optics_capable": bool(optics_capable),
            "replay_input": raw_path.parent.name,
            "inputs": dict(inputs.sources),
        },
        synthetic=synthetic,
    )


async def score_windows(
    replay_input: Path,
    times: Sequence[float],
    *,
    kind: str = "cue",
    config: Optional[CueArousalConfig] = None,
    optics_capable: bool = True,
) -> Tuple[CueArousalWindow, ...]:
    """Score windows at ``times`` (Unix seconds, the recording's clock) without any log,
    as the night scorer scores each cue and control; one replay pass."""
    config = config or CueArousalConfig()
    config.validate()
    targets = [_Target(kind=kind, index=index, time=float(time)) for index, time in enumerate(times)]
    windows = await _score_targets_from_recording(
        resolve_raw_path(Path(replay_input).expanduser()), targets, config, optics_capable
    )
    return tuple(sorted(windows, key=lambda window: window.index))


def score_frame_windows(
    frames: Iterable[MuseFrame],
    times: Sequence[float],
    *,
    kind: str = "cue",
    config: Optional[CueArousalConfig] = None,
    optics_capable: bool = True,
) -> Tuple[CueArousalWindow, ...]:
    """``score_windows()`` over frames in memory, in time order."""
    config = config or CueArousalConfig()
    config.validate()
    targets = [_Target(kind=kind, index=index, time=float(time)) for index, time in enumerate(times)]
    cutter = _WindowCutter(targets, config, _scorer(config, optics_capable))
    for frame in frames:
        cutter.feed(frame)
    return tuple(sorted(cutter.finish(), key=lambda window: window.index))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _NightInputs:
    plays: Tuple[Mapping[str, object], ...]
    audio: Tuple[Optional[Mapping[str, object]], ...]
    audio_log_found: bool
    audio_log_mismatch: bool
    gate_records: Tuple[GateDecisionRecord, ...]
    awakenings: Optional[Tuple[AwakeningEvent, ...]]
    first_epoch_start: Optional[float]
    pilot3: bool
    sources: Mapping[str, Optional[str]]


@dataclass(frozen=True)
class _Target:
    kind: str
    index: int
    time: float
    offset_seconds: Optional[float] = None
    gate_open: Optional[bool] = None
    paired_index: Optional[int] = None
    cue: Mapping[str, object] = field(default_factory=dict)


def _night_inputs(
    recording_dir: Path,
    *,
    pilot3_report: Optional[Path],
    summary_path: Optional[Path],
    scheduler_events_path: Optional[Path],
    audio_log_path: Optional[Path],
    gate_log_path: Optional[Path],
    awakening_events_path: Optional[Path],
) -> _NightInputs:
    sources: Dict[str, Optional[str]] = {}
    if pilot3_report is not None:
        report_path = Path(pilot3_report).expanduser()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        plays = tuple(
            event for event in report.get("scheduler_events", ()) or () if event.get("event_type") == "play"
        ) + tuple(event for event in report.get("injected_cues", ()) or () if event.get("event_type") == "play")
        metrics = dict(report.get("metrics", {}) or {})
        sources["pilot3_report"] = report_path.name
        gate_records = load_rem_gate_decisions(
            Path(gate_log_path).expanduser() if gate_log_path else report
        )
        audio_path = Path(audio_log_path).expanduser() if audio_log_path else None
        audio, found, mismatch = _audio_for_plays(plays, audio_path)
        awakenings = _awakenings(awakening_events_path, None)
        return _NightInputs(
            plays=plays,
            audio=audio,
            audio_log_found=found,
            audio_log_mismatch=mismatch,
            gate_records=gate_records,
            awakenings=awakenings,
            first_epoch_start=_optional_float(metrics.get("first_epoch_start")),
            pilot3=True,
            sources=sources,
        )

    summary, summary_name = _read_summary(recording_dir, summary_path)
    events_path = Path(scheduler_events_path).expanduser() if scheduler_events_path else recording_dir / "scheduler_events.jsonl"
    plays = tuple(event for event in _read_jsonl(events_path) if event.get("event_type") == "play")
    sources["scheduler_events"] = events_path.name if events_path.exists() else None
    audio_path = Path(audio_log_path).expanduser() if audio_log_path else recording_dir / "audio_playback.jsonl"
    audio, found, mismatch = _audio_for_plays(plays, audio_path)
    sources["audio_log"] = audio_path.name if found else None
    gate_source = Path(gate_log_path).expanduser() if gate_log_path else recording_dir
    gate_records = load_rem_gate_decisions(gate_source)
    sources["gate_log"] = (
        (Path(gate_log_path).name if gate_log_path else REM_GATE_DECISIONS_FILENAME) if gate_records else None
    )
    awakenings = _awakenings(awakening_events_path, recording_dir)
    sources["summary"] = summary_name
    return _NightInputs(
        plays=plays,
        audio=audio,
        audio_log_found=found,
        audio_log_mismatch=mismatch,
        gate_records=gate_records,
        awakenings=awakenings,
        first_epoch_start=_optional_float(summary.get("first_epoch_start")) if summary else None,
        pilot3=False,
        sources=sources,
    )


def find_night_summary(recording_dir: Path, summary_path: Optional[Path] = None) -> Optional[Path]:
    """The Pilot 4/5 summary the scorer reads for a night: ``summary_path``, else the
    directory's ``pilot4_summary.json`` or ``pilot5_summary.json``; None without one."""
    candidates = [Path(summary_path).expanduser()] if summary_path else [
        Path(recording_dir).expanduser() / name for name in SUMMARY_FILENAMES
    ]
    return next((path for path in candidates if path.exists()), None)


def write_summary_block(summary_path: Path, block: Mapping[str, object]) -> bool:
    """Put ``block`` (``CueArousalReport.summary_block()``) into a Pilot 4/5 summary as its
    ``cue_arousal``, as the night does when it ends, so that scoring the night again,
    after awakening markers were added, leaves no older figure there. A summary without
    the field, from before #227, is left as it is. Returns whether it was written."""
    summary_path = Path(summary_path).expanduser()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "cue_arousal" not in payload:
        return False
    payload["cue_arousal"] = _json_safe(dict(block))
    _write_text(summary_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return True


def _read_summary(recording_dir: Path, summary_path: Optional[Path]) -> Tuple[Dict[str, object], Optional[str]]:
    """The night's Pilot 4/5 summary and its file name; empty without one."""
    path = find_night_summary(recording_dir, summary_path)
    if path is None:
        return {}, None
    return json.loads(path.read_text(encoding="utf-8")), path.name


def _awakenings(path: Optional[Path], recording_dir: Optional[Path]) -> Optional[Tuple[AwakeningEvent, ...]]:
    if path is None:
        if recording_dir is None:
            return None
        path = recording_dir / "awakening_events.jsonl"
    path = Path(path).expanduser()
    if not path.exists():
        return None
    return load_awakening_events(path)


def _audio_for_plays(
    plays: Sequence[Mapping[str, object]],
    audio_path: Optional[Path],
) -> Tuple[Tuple[Optional[Mapping[str, object]], ...], bool, bool]:
    """Each scheduler play's line of ``audio_playback.jsonl``, in order.

    The player logs one line per play, in the order of the plays, and one more for an
    emergency stop. A play whose line names another cue ends the matching: it and the
    plays after it get none, and the mismatch is reported.
    """
    if audio_path is None or not audio_path.exists():
        return tuple(None for _ in plays), False, False
    entries = [
        entry
        for entry in _read_jsonl(audio_path)
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


def _cue_targets(inputs: _NightInputs, first_epoch_start: float, config: CueArousalConfig) -> List[_Target]:
    records = sorted(inputs.gate_records, key=lambda record: (record.timestamp_seconds, record.index))
    starts = [record.timestamp_seconds for record in records]
    placed = []
    for play, audio in zip(inputs.plays, inputs.audio):
        metadata = dict(play.get("metadata", {}) or {})
        scheduled = float(play["timestamp_seconds"])
        record = emitting_record(records, scheduled, starts=starts)
        if metadata.get("decision_seconds") is not None:
            offset = float(metadata["decision_seconds"])
        elif record is not None:
            offset = record.decision_seconds
        else:
            offset = scheduled + _DEFAULT_EPOCH_SECONDS
        placed.append((offset, play, audio, record))
    placed.sort(key=lambda item: item[0])
    targets = []
    previous_played: Optional[float] = None
    for index, (offset, play, audio, record) in enumerate(placed):
        cue = _cue_info(play, audio)
        cue["inter_cue_interval_seconds"] = None if previous_played is None else offset - previous_played
        if cue["played"]:
            previous_played = offset
        targets.append(
            _Target(
                kind="cue",
                index=index,
                time=first_epoch_start + offset,
                offset_seconds=offset,
                gate_open=None if record is None else record.gate_open,
                cue=cue,
            )
        )
    return targets


def _cue_info(play: Mapping[str, object], audio: Optional[Mapping[str, object]]) -> Dict[str, object]:
    metadata = dict(play.get("metadata", {}) or {})
    synthetic_cue = bool(metadata.get("synthetic_cue"))
    if audio is not None:
        volumes = {key: audio.get(key) for key in ("requested_volume", "effective_volume", "volume_multiplier")}
        volume_source = "audio_playback.jsonl"
    elif metadata.get("effective_volume") is not None:
        volumes = {key: metadata.get(key) for key in ("requested_volume", "effective_volume", "volume_multiplier")}
        volume_source = "scheduler_event"
    else:
        volumes = {"requested_volume": None, "effective_volume": None, "volume_multiplier": None}
        volume_source = None
    status = None if audio is None else audio.get("status")
    backend = None if audio is None else audio.get("backend_name")
    return {
        "cue_id": play.get("cue_id"),
        "protocol": play.get("protocol"),
        "puzzle_id": play.get("puzzle_id"),
        "scheduler_timestamp_seconds": play.get("timestamp_seconds"),
        "synthetic_cue": synthetic_cue,
        "guard_action": metadata.get("arousal_guard_action", "allow"),
        "guard_volume_multiplier": metadata.get("volume_multiplier", 1.0),
        "puzzle_cue_index": metadata.get("puzzle_cue_index"),
        "tlr_event_index": metadata.get("tlr_event_index"),
        "audio_status": status,
        "audio_backend": backend,
        "volume_source": volume_source,
        **volumes,
        "played": status in (None, "played"),
        "audible": status == "played" and backend not in SILENT_BACKENDS and not synthetic_cue,
    }


def _control_targets(
    cues: Sequence[_Target],
    records: Sequence[GateDecisionRecord],
    first_epoch_start: float,
    config: CueArousalConfig,
) -> List[_Target]:
    """One quiet control per played cue where one exists.

    The candidates are the gate decisions of the night more than
    ``control_exclusion_seconds`` from every play, with the whole window inside the
    epochs. Each played cue, in time order, takes the nearest one in its own gate state
    (the earlier on a tie), and no two controls are closer than
    ``control_spacing_seconds``. A play the player blocked or skipped is left out of the
    cue rates, so it gets no control in the control rates either.
    """
    if not records or not cues:
        return []
    plays = sorted(cue.offset_seconds for cue in cues if cue.offset_seconds is not None)
    first = min(record.timestamp_seconds for record in records)
    last = max(record.decision_seconds for record in records)
    candidates = []
    for record in records:
        offset = record.decision_seconds
        if offset - config.baseline_seconds < first or offset + config.post_window_seconds > last:
            continue
        if _near(plays, offset, config.control_exclusion_seconds):
            continue
        candidates.append((offset, record.gate_open))
    chosen: List[float] = []
    controls = []
    for cue in sorted(cues, key=lambda target: target.offset_seconds or 0.0):
        if cue.gate_open is None or cue.offset_seconds is None or not cue.cue.get("played", True):
            continue
        best = None
        for offset, gate_open in candidates:
            if gate_open != cue.gate_open:
                continue
            if any(abs(offset - other) < config.control_spacing_seconds for other in chosen):
                continue
            key = (abs(offset - cue.offset_seconds), offset)
            if best is None or key < best:
                best = key
        if best is None:
            continue
        offset = best[1]
        chosen.append(offset)
        controls.append(
            _Target(
                kind="control",
                index=len(controls),
                time=first_epoch_start + offset,
                offset_seconds=offset,
                gate_open=cue.gate_open,
                paired_index=cue.index,
            )
        )
    return controls


def _pair(cues: Sequence[_Target], controls: Sequence[_Target]) -> List[_Target]:
    """The cues with the index of the control paired with each."""
    paired = {control.paired_index: control.index for control in controls}
    return [
        _Target(
            kind=cue.kind,
            index=cue.index,
            time=cue.time,
            offset_seconds=cue.offset_seconds,
            gate_open=cue.gate_open,
            paired_index=paired.get(cue.index),
            cue=cue.cue,
        )
        for cue in cues
    ]


def _near(sorted_offsets: Sequence[float], offset: float, distance: float) -> bool:
    position = bisect.bisect_left(sorted_offsets, offset - distance - _TIME_TOLERANCE_SECONDS)
    return position < len(sorted_offsets) and sorted_offsets[position] <= offset + distance + _TIME_TOLERANCE_SECONDS


def _awakening_offsets(
    awakenings: Optional[Sequence[AwakeningEvent]],
    first_epoch_start: float,
) -> Tuple[Optional[List[Tuple[float, Optional[str]]]], Dict[str, int]]:
    """Each awakening on the night's time base with its attribution, and how many were
    placed by their UTC time or could not be placed."""
    if awakenings is None:
        return None, {}
    placed = []
    notes = {"from_utc": 0, "unplaced": 0}
    for event in awakenings:
        if event.timestamp_seconds is not None:
            placed.append((event.timestamp_seconds, event.attributed_to_cue))
            continue
        try:
            moment = dt.datetime.fromisoformat(event.timestamp_utc)
        except ValueError:
            notes["unplaced"] += 1
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=dt.timezone.utc)
        placed.append((moment.timestamp() - first_epoch_start, event.attributed_to_cue))
        notes["from_utc"] += 1
    return sorted(placed, key=lambda item: item[0]), notes


# --------------------------------------------------------------------------- #
# The replay pass and the windows
# --------------------------------------------------------------------------- #
class _WindowCutter:
    """Hands each target the frames around it, in one pass over a time-ordered stream,
    and keeps only the frames a pending target still needs."""

    def __init__(
        self,
        targets: Sequence[_Target],
        config: CueArousalConfig,
        scorer: Callable[[_Target, Tuple[MuseFrame, ...]], CueArousalWindow],
    ) -> None:
        self._pending: Deque[_Target] = deque(sorted(targets, key=lambda target: target.time))
        self._before = config.span_before_seconds + _FRAME_LEAD_SECONDS
        self._after = config.span_after_seconds + _FRAME_LEAD_SECONDS
        self._scorer = scorer
        self._buffer: Deque[MuseFrame] = deque()
        self._windows: List[CueArousalWindow] = []

    def feed(self, frame: MuseFrame) -> None:
        if not self._pending:
            return
        self._buffer.append(frame)
        while self._pending and frame.timestamp > self._pending[0].time + self._after:
            self._score_next()
        if not self._pending:
            self._buffer.clear()
            return
        earliest = self._pending[0].time - self._before
        while self._buffer and self._buffer[0].timestamp < earliest:
            self._buffer.popleft()

    def finish(self) -> List[CueArousalWindow]:
        while self._pending:
            self._score_next()
        self._buffer.clear()
        return self._windows

    def _score_next(self) -> None:
        target = self._pending.popleft()
        start, end = target.time - self._before, target.time + self._after
        frames = tuple(frame for frame in self._buffer if start <= frame.timestamp <= end)
        self._windows.append(self._scorer(target, frames))


async def _score_targets_from_recording(
    raw_path: Path,
    targets: Sequence[_Target],
    config: CueArousalConfig,
    optics_capable: bool,
) -> List[CueArousalWindow]:
    """One replay pass over ``raw_path``, from the first target's span to the last's."""
    cutter = _WindowCutter(targets, config, _scorer(config, optics_capable))
    session_start = _raw_session_start(raw_path)
    lead = config.span_before_seconds + 2 * _FRAME_LEAD_SECONDS
    tail = config.span_after_seconds + 2 * _FRAME_LEAD_SECONDS
    start_seconds = min(target.time for target in targets) - lead - session_start
    end_seconds = max(target.time for target in targets) + tail - session_start
    replay = ReplaySession(
        ReplayConfig(
            input_path=raw_path,
            speed=0.0,
            start_seconds=max(0.0, start_seconds),
            end_seconds=max(max(0.0, start_seconds), end_seconds),
        )
    )
    await replay.connect()
    try:
        await _feed_stream(cutter, replay.stream())
    finally:
        await replay.stop()
    return cutter.finish()


async def _feed_stream(cutter: _WindowCutter, frames: AsyncIterable[MuseFrame]) -> None:
    async for frame in frames:
        cutter.feed(frame)


async def _first_frame_time(raw_path: Path) -> float:
    replay = ReplaySession(ReplayConfig(input_path=raw_path, speed=0.0))
    await replay.connect()
    stream = replay.stream()
    try:
        async for frame in stream:
            return float(frame.timestamp)
    finally:
        await stream.aclose()
        await replay.stop()
    raise ValueError(f"{raw_path} holds no frames to take the time base from")


def _raw_session_start(raw_path: Path) -> float:
    stream = MuseRawStream(str(raw_path))
    stream.open_read()
    try:
        return stream.session_start.timestamp()
    finally:
        stream.close()


def _scorer(config: CueArousalConfig, optics_capable: bool):
    def score(target: _Target, frames: Tuple[MuseFrame, ...]) -> CueArousalWindow:
        return _score_target(target, frames, config, optics_capable)

    return score


def _score_target(
    target: _Target,
    frames: Tuple[MuseFrame, ...],
    config: CueArousalConfig,
    optics_capable: bool,
) -> CueArousalWindow:
    flags: Dict[str, Optional[bool]] = {}
    values: Dict[str, object] = {}
    quality: List[str] = []

    rises, eeg_values, eeg_quality = _eeg_rises(frames, target.time, config)
    values.update(eeg_values)
    quality.extend(eeg_quality)
    for band, ratio in (
        ("alpha", config.alpha_rise_ratio),
        ("beta", config.beta_rise_ratio),
        ("emg", config.emg_rise_ratio),
    ):
        rise = rises.get(band)
        values[f"{band}_rise"] = rise
        flags[band] = None if rise is None else bool(rise >= ratio)

    span_start = target.time - config.span_before_seconds
    span_end = target.time + config.span_after_seconds
    flags["imu"], imu_values, imu_quality = _imu_flag(frames, target.time, span_start, span_end, config)
    values.update(imu_values)
    quality.extend(imu_quality)
    flags["hr"], hr_values, hr_quality = _hr_flag(frames, target.time, span_start, span_end, config, optics_capable)
    values.update(hr_values)
    quality.extend(hr_quality)

    scored = [flag for flag in (flags[name] for name in AROUSAL_FLAGS) if flag is not None]
    flags["any"] = None if not scored else any(scored)
    return CueArousalWindow(
        kind=target.kind,
        index=target.index,
        time=target.time,
        offset_seconds=target.offset_seconds,
        paired_index=target.paired_index,
        gate_open=target.gate_open,
        flags=flags,
        values=values,
        quality_flags=tuple(quality),
        cue=dict(target.cue),
    )


def _eeg_rises(
    frames: Sequence[MuseFrame],
    cue_time: float,
    config: CueArousalConfig,
) -> Tuple[Dict[str, Optional[float]], Dict[str, object], Tuple[str, ...]]:
    """Post-over-baseline power of alpha, beta and the EMG band; None where unscorable."""
    fs = config.eeg_sample_rate_hz
    n_base = int(round(config.baseline_seconds * fs))
    n_post = int(round(config.post_window_seconds * fs))
    window = _eeg_window(frames, cue_time - config.baseline_seconds, n_base + n_post, config)
    present = np.isfinite(window)
    seen = present.any(axis=0)
    none: Dict[str, Optional[float]] = {"alpha": None, "beta": None, "emg": None}
    if not seen.any():
        return none, {"eeg_coverage_baseline": 0.0, "eeg_coverage_post": 0.0}, ("eeg_gap",)
    base_coverage = float(present[:n_base, seen].mean())
    post_coverage = float(present[n_base:, seen].mean())
    values: Dict[str, object] = {
        "eeg_coverage_baseline": base_coverage,
        "eeg_coverage_post": post_coverage,
    }
    if min(base_coverage, post_coverage) < config.min_window_coverage:
        return none, values, ("eeg_gap",)

    usable = [
        column
        for column in range(window.shape[1])
        if _usable_channel(window[:, column], present[:, column], n_base, config)
    ]
    values["eeg_channels_used"] = [config.eeg_channels[column] for column in usable]
    if not usable:
        return none, values, ("eeg_unusable",)

    bands = {"alpha": config.alpha_band_hz, "beta": config.beta_band_hz, "emg": config.emg_band_hz}
    powers = {band: [] for band in bands}
    valid = None
    for column in usable:
        freqs, psd, segment_valid = _segment_spectra(window[:, column], present[:, column], config)
        emg_psd = np.vstack(
            [bridge_lines(freqs, row, config.line_hz, config.line_half_width_hz) for row in psd]
        )
        for band, (low, high) in bands.items():
            powers[band].append(_band_power_rows(freqs, emg_psd if band == "emg" else psd, low, high))
        valid = segment_valid if valid is None else valid & segment_valid
    segment_starts = np.arange(valid.size) * config.segment_step_seconds
    rises: Dict[str, Optional[float]] = {}
    for band in bands:
        mean_power = np.mean(np.vstack(powers[band]), axis=0)
        baseline = _sub_window_levels(mean_power, valid, segment_starts, 0.0, config.baseline_seconds, config)
        after = _sub_window_levels(
            mean_power,
            valid,
            segment_starts,
            config.baseline_seconds,
            config.baseline_seconds + config.post_window_seconds,
            config,
        )
        base_level = _enough(baseline, np.nanmedian)
        post_level = _enough(after, np.nanmax)
        values[f"{band}_baseline_power"] = base_level
        values[f"{band}_post_power"] = post_level
        if base_level is None or post_level is None or base_level <= 0:
            rises[band] = None
        else:
            rises[band] = float(post_level / base_level)
    quality = ("eeg_gap",) if any(rise is None for rise in rises.values()) else ()
    return rises, values, quality


def _usable_channel(values: np.ndarray, present: np.ndarray, n_base: int, config: CueArousalConfig) -> bool:
    """Read in both windows, not flat and not clipping before the cue (the thresholds
    of ``EEGFeatureConfig``). Clipping after the cue counts: a movement does that."""
    if present[:n_base].mean() < config.min_window_coverage or present[n_base:].mean() < config.min_window_coverage:
        return False
    baseline = values[:n_base][present[:n_base]]
    if baseline.size < 2 or float(np.std(baseline)) <= config.flat_std_uv:
        return False
    centred = np.abs(baseline - float(np.median(baseline)))
    return float(np.mean(centred >= config.clipping_uv)) < config.clipping_fraction


def _segment_spectra(
    values: np.ndarray,
    present: np.ndarray,
    config: CueArousalConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hann-windowed power spectral density of each segment, after a linear detrend and
    with the optics line fitted at its own frequency and taken out of each segment, and
    whether few enough of the segment's samples were missing (bridged linearly)."""
    fs = config.eeg_sample_rate_hz
    index = np.arange(values.size)
    filled = np.interp(index, index[present], values[present])
    length = int(round(config.segment_seconds * fs))
    step = int(round(config.segment_step_seconds * fs))
    starts = np.arange(0, values.size - length + 1, step)
    segments = np.lib.stride_tricks.sliding_window_view(filled, length)[starts]
    segments = detrend(segments, axis=1, type="linear")
    missing = np.lib.stride_tricks.sliding_window_view(~present, length)[starts].mean(axis=1)
    whole = detrend(filled, type="linear")
    t = np.arange(length) / fs
    for line in config.line_hz:
        frequency = _line_frequency(whole, fs, line, config.line_search_hz)
        if frequency is None:
            continue
        basis = np.column_stack([np.cos(2 * np.pi * frequency * t), np.sin(2 * np.pi * frequency * t)])
        coefficients, *_ = np.linalg.lstsq(basis, segments.T, rcond=None)
        segments = segments - (basis @ coefficients).T
    taper = np.hanning(length)
    spectra = np.fft.rfft(segments * taper, axis=1)
    psd = (np.abs(spectra) ** 2) / (fs * float(np.sum(taper ** 2)))
    psd[:, 1:] *= 2.0
    if length % 2 == 0:
        psd[:, -1] /= 2.0
    freqs = np.fft.rfftfreq(length, 1.0 / fs)
    return freqs, psd, missing <= config.max_segment_missing_fraction


def _line_frequency(values: np.ndarray, fs: float, line: float, search: float) -> Optional[float]:
    """The spectral peak within ``search`` Hz of ``line`` on a 0.01 Hz grid, as
    ``complexity_features.remove_lines`` finds it."""
    if not 0 < line < fs / 2 or values.size < 2:
        return None
    size = max(values.size, int(round(fs / 0.01)))
    spectrum = np.abs(np.fft.rfft(values * np.hanning(values.size), size))
    freqs = np.fft.rfftfreq(size, 1.0 / fs)
    band = (freqs >= line - search) & (freqs <= line + search)
    if not np.any(band):
        return None
    return float(freqs[band][int(np.argmax(spectrum[band]))])


def _band_power_rows(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    if np.count_nonzero(mask) < 2:
        return np.full(psd.shape[0], np.nan)
    return trapezoid(psd[:, mask], freqs[mask], axis=1)


def _sub_window_levels(
    power: np.ndarray,
    valid: np.ndarray,
    segment_starts: np.ndarray,
    region_start: float,
    region_end: float,
    config: CueArousalConfig,
) -> np.ndarray:
    """Mean power of each sub-window inside [region_start, region_end), NaN where one of
    its segments was missing too much."""
    levels = []
    start = region_start
    eps = _TIME_TOLERANCE_SECONDS
    while start + config.subwindow_seconds <= region_end + eps:
        inside = (segment_starts >= start - eps) & (
            segment_starts + config.segment_seconds <= start + config.subwindow_seconds + eps
        )
        chosen = np.flatnonzero(inside)
        if chosen.size and bool(np.all(valid[chosen])):
            levels.append(float(np.mean(power[chosen])))
        else:
            levels.append(math.nan)
        start += config.subwindow_step_seconds
    return np.asarray(levels, dtype=float)


def _enough(levels: np.ndarray, reduce) -> Optional[float]:
    """``reduce`` of the sub-window levels when at least half of them could be read."""
    finite = np.isfinite(levels)
    if levels.size == 0 or np.count_nonzero(finite) * 2 < levels.size:
        return None
    return float(reduce(levels[finite]))


def _eeg_window(frames: Sequence[MuseFrame], start: float, count: int, config: CueArousalConfig) -> np.ndarray:
    """(count, channels) µV from ``start`` on at the nominal rate, NaN where nothing arrived.

    The frames go on one sample clock that starts at the first EEG frame's timestamp and
    moves one sample a row, as the optics clock of ``ppg_features`` does: samples the
    decoder reports lost stay missing, and a frame that arrives later than the clock
    expects starts after a gap only when no reported loss can explain it
    (``_gap_lateness()``). A BLE gap is therefore missing samples, never zeros counted as
    signal, and a frame that came late with nothing lost keeps its place.
    """
    fs = config.eeg_sample_rate_hz
    channels = config.eeg_channels
    out = np.full((count, len(channels)), np.nan)
    clock_start = None
    length = 0
    positions = []
    blocks = []
    for frame in frames:
        eeg = frame.eeg
        if eeg is None or not eeg.channels_uv:
            continue
        rows = max(len(values) for values in eeg.channels_uv.values())
        if rows == 0:
            continue
        lead, offsets = _row_offsets(rows, eeg.gaps)
        if clock_start is None:
            clock_start = float(eeg.timestamp)
            lead = 0
        else:
            late = float(eeg.timestamp) - (clock_start + (length + lead) / fs)
            if late > _gap_lateness(eeg, fs, config):
                lead += int(round(late * fs))
        block = np.full((rows, len(channels)), np.nan)
        for column, name in enumerate(channels):
            series = eeg.channels_uv.get(name)
            if series:
                taken = min(rows, len(series))
                block[:taken, column] = series[:taken]
        positions.append(length + lead + offsets)
        blocks.append(block)
        length += lead + int(offsets[-1]) + 1
    if clock_start is None:
        return out
    clock = np.full((length, len(channels)), np.nan)
    clock[np.concatenate(positions)] = np.vstack(blocks)
    first = int(round((start - clock_start) * fs))
    low, high = max(0, first), min(length, first + count)
    if high > low:
        out[low - first:high - first] = clock[low:high]
    return out


def _gap_lateness(eeg: EEGSample, fs: float, config: CueArousalConfig) -> float:
    """How much later than the clock expects a frame must come to start after a gap.

    The subpacket counter reports any loss of up to ``_REPORTED_LOSS_SUBPACKETS``
    subpackets, so a frame that is later by less signal than that, with no loss reported
    before it, lost nothing: it came late. The BLE client stamps each packet when the
    event loop takes it, and a loop that was busy stamps all the packets it held with
    the moment it is free. Only a loss too long for the counter is read from the clock.
    An EEG layout the protocol does not know falls back to ``clock_slack_seconds``.
    """
    samples = _EEG_SUBPACKET_SAMPLES.get(len(eeg.channels_uv))
    reach = 0.0 if samples is None else _REPORTED_LOSS_SUBPACKETS * samples / fs
    return max(config.clock_slack_seconds, reach)


def _row_offsets(rows: int, gaps: Sequence[Tuple[int, int]]) -> Tuple[int, np.ndarray]:
    """Samples lost before a frame's first row, and each row's clock offset from it."""
    offsets = np.arange(rows)
    lead = 0
    for index, missing in gaps:
        if missing <= 0:
            continue
        if index <= 0:
            lead += int(missing)
        elif index < rows:
            offsets[index:] += int(missing)
    return lead, offsets


def _span_epoch(frames: Sequence[MuseFrame], start: float, end: float, coverage: Mapping[str, float]) -> SleepEpoch:
    """The frames around a cue as a ``SleepEpoch``, so the epoch feature code reads them."""
    return SleepEpoch(
        index=0,
        start_time=start,
        end_time=end,
        frames=tuple(frames),
        modality_counts={},
        sample_counts={},
        coverage=dict(coverage),
        quality_flags=(),
    )


def _imu_flag(
    frames: Sequence[MuseFrame],
    cue_time: float,
    span_start: float,
    span_end: float,
    config: CueArousalConfig,
) -> Tuple[Optional[bool], Dict[str, object], Tuple[str, ...]]:
    """An arousal-proxy movement (``IMUFeatureConfig`` thresholds) in the cue window of
    ``imu_features``, 2 s before the cue to 10 s after it."""
    imu = config.imu
    low = cue_time - imu.cue_pre_window_seconds
    high = cue_time + imu.cue_post_window_seconds
    rows = 0
    for frame in frames:
        if frame.imu is None:
            continue
        count = max(len(frame.imu.accelerometer_g or ()), len(frame.imu.gyroscope_dps or ()))
        if not count:
            continue
        _, offsets = _row_offsets(count, frame.imu.gaps)
        times = frame.imu.timestamp + offsets / imu.sample_rate_hz
        rows += int(np.count_nonzero((times >= low) & (times <= high)))
    coverage = min(1.0, rows / ((high - low) * imu.sample_rate_hz)) if high > low else 0.0
    values: Dict[str, object] = {"imu_coverage": coverage}
    if coverage < config.min_window_coverage:
        return None, values, ("imu_gap",)
    row = extract_imu_features(
        _span_epoch(frames, span_start, span_end, {"imu": coverage}),
        imu,
        cue_timestamps=(cue_time,),
    )
    log = row.cue_movement_logs[0]
    values["movement_events"] = log.movement_event_count
    values["arousal_movement_events"] = log.arousal_event_count
    values["peak_motion_level"] = log.peak_motion_level
    return bool(log.arousal_event_count > 0), values, ()


def _hr_flag(
    frames: Sequence[MuseFrame],
    cue_time: float,
    span_start: float,
    span_end: float,
    config: CueArousalConfig,
    optics_capable: bool,
) -> Tuple[Optional[bool], Dict[str, object], Tuple[str, ...]]:
    """A used value of the optics heart-rate series (#251) centred in the post window at
    least ``hr_jump_bpm`` above the median of the used values centred in the baseline.
    Never the decoder's ``heart_rate`` stream, which jumps between values 36 to 77 ms apart."""
    if not optics_capable or not any(frame.ppg is not None and frame.ppg.channels for frame in frames):
        return None, {}, ("ppg_absent",)
    series = heart_rate_series(_span_epoch(frames, span_start, span_end, {}), config.ppg)
    in_window = [
        point
        for point in series.points
        if cue_time - config.baseline_seconds <= point.time <= cue_time + config.post_window_seconds
    ]
    used = [point for point in in_window if point.used]
    before = [point for point in used if point.time < cue_time]
    after = [point for point in used if point.time >= cue_time]
    values: Dict[str, object] = {
        "hr_values_baseline": len(before),
        "hr_values_post": len(after),
        "hr_values_skipped": len(in_window) - len(used),
    }
    if len(before) < config.min_hr_values or len(after) < config.min_hr_values:
        return None, values, ("hr_unavailable",)
    baseline = float(np.median([point.bpm for point in before]))
    peak = float(max(point.bpm for point in after))
    values.update(
        {
            "hr_baseline_bpm": baseline,
            "hr_post_max_bpm": peak,
            "hr_jump_bpm": peak - baseline,
            "hr_quality": float(np.median([point.quality for point in used])),
        }
    )
    return bool(peak - baseline >= config.hr_jump_bpm), values, ()


# --------------------------------------------------------------------------- #
# The night around each window, outcomes, tables and limitations
# --------------------------------------------------------------------------- #
def _with_night_context(
    window: CueArousalWindow,
    runs: Sequence[GateOpenRun],
    awakenings: Optional[Sequence[Tuple[float, Optional[str]]]],
    config: CueArousalConfig,
) -> CueArousalWindow:
    run = run_at(runs, window.offset_seconds) if window.gate_open else None
    awakening = None
    if window.kind == "cue" and awakenings is not None and window.offset_seconds is not None:
        later = [
            offset - window.offset_seconds
            for offset, _attribution in awakenings
            if 0 < offset - window.offset_seconds <= config.awakening_window_seconds
        ]
        awakening = min(later) if later else None
    return CueArousalWindow(
        kind=window.kind,
        index=window.index,
        time=window.time,
        flags=window.flags,
        values=window.values,
        quality_flags=window.quality_flags,
        offset_seconds=window.offset_seconds,
        paired_index=window.paired_index,
        gate_open=window.gate_open,
        rem_period_index=None if run is None else run.index,
        seconds_since_gate_open=None if run is None else window.offset_seconds - run.start_seconds,
        rem_period_ends_in_seconds=(
            None if run is None or not run.closed else run.end_seconds - window.offset_seconds
        ),
        awakening_in_seconds=awakening,
        cue=window.cue,
    )


def _flag_counts(windows: Sequence[CueArousalWindow]) -> Dict[str, Dict[str, object]]:
    counts = {}
    for flag in SUMMARY_FLAGS:
        scored = [window.flags.get(flag) for window in windows if window.flags.get(flag) is not None]
        flagged = sum(1 for value in scored if value)
        counts[flag] = {"flagged": flagged, "scored": len(scored), "rate": _rate(flagged, len(scored))}
    return counts


def _outcomes(
    windows: Sequence[CueArousalWindow],
    runs: Sequence[GateOpenRun],
    awakenings: Optional[Sequence[Tuple[float, Optional[str]]]],
    inputs: _NightInputs,
    config: CueArousalConfig,
) -> Dict[str, object]:
    cues = [window for window in windows if window.kind == "cue"]
    played = [window for window in cues if window.played]
    controls = [window for window in windows if window.kind == "control"]
    cue_counts = _flag_counts(played)
    control_counts = _flag_counts(controls)
    per_cue = {flag: cue_counts[flag]["rate"] for flag in SUMMARY_FLAGS}
    per_control = {flag: control_counts[flag]["rate"] for flag in SUMMARY_FLAGS}
    by_index = {window.index: window for window in controls}
    pairs = [
        (cue.flags.get("any"), by_index[cue.paired_index].flags.get("any"))
        for cue in played
        if cue.paired_index is not None and cue.paired_index in by_index
    ]
    pairs = [(cue, control) for cue, control in pairs if cue is not None and control is not None]

    cued_runs = {window.rem_period_index for window in played if window.rem_period_index is not None}
    disrupted = {
        window.rem_period_index
        for window in played
        if window.rem_period_index is not None
        and (window.flags.get("any") or _ended_within(window, config.rem_termination_seconds))
    }
    awakening_rate = None
    attribution_counts: Dict[str, int] = {}
    if awakenings is not None:
        awakening_rate = _rate(sum(1 for window in played if window.awakening_in_seconds is not None), len(played))
        for offset, attribution in awakenings:
            if any(
                window.offset_seconds is not None
                and 0 < offset - window.offset_seconds <= config.awakening_window_seconds
                for window in played
            ):
                key = attribution or "unset"
                attribution_counts[key] = attribution_counts.get(key, 0) + 1
    return {
        "cue_count": len(cues),
        "played_cue_count": len(played),
        "synthetic_cue_count": sum(1 for window in cues if window.cue.get("synthetic_cue")),
        "audible_cue_count": sum(1 for window in played if window.cue.get("audible")),
        "scored_cue_count": cue_counts["any"]["scored"],
        "control_count": len(controls),
        "scored_control_count": control_counts["any"]["scored"],
        "paired_count": sum(1 for window in played if window.paired_index is not None),
        "arousal_proxy_per_cue": per_cue,
        "arousal_proxy_per_control": per_control,
        "arousal_proxy_difference": {
            flag: _difference(per_cue[flag], per_control[flag]) for flag in SUMMARY_FLAGS
        },
        "flag_counts": {"cue": cue_counts, "control": control_counts},
        "paired_any": {
            "pairs": len(pairs),
            "both": sum(1 for cue, control in pairs if cue and control),
            "cue_only": sum(1 for cue, control in pairs if cue and not control),
            "control_only": sum(1 for cue, control in pairs if control and not cue),
            "neither": sum(1 for cue, control in pairs if not cue and not control),
        },
        "movement_events_per_cue": _mean_value(played, "movement_events"),
        "movement_events_per_control": _mean_value(controls, "movement_events"),
        "cue_attributed_awakening_rate": awakening_rate,
        "awakening_attribution_counts": attribution_counts,
        "awakening_count": None if awakenings is None else len(awakenings),
        "rem_period_count": len(runs) if inputs.gate_records else None,
        "cued_rem_period_count": len(cued_runs) if inputs.gate_records else None,
        "rem_periods_with_disruption_fraction": (
            _rate(len(disrupted), len(cued_runs)) if inputs.gate_records else None
        ),
        "rem_termination_within_60s_of_cue_rate": _termination_rate(played, config),
        "rem_termination_within_60s_of_control_rate": _termination_rate(controls, config),
    }


def _ended_within(window: CueArousalWindow, seconds: float) -> bool:
    return window.rem_period_ends_in_seconds is not None and window.rem_period_ends_in_seconds <= seconds


def _termination_rate(windows: Sequence[CueArousalWindow], config: CueArousalConfig) -> Optional[float]:
    in_runs = [window for window in windows if window.rem_period_index is not None]
    return _rate(sum(1 for window in in_runs if _ended_within(window, config.rem_termination_seconds)), len(in_runs))


def _mean_value(windows: Sequence[CueArousalWindow], key: str) -> Optional[float]:
    values = [float(window.values[key]) for window in windows if window.values.get(key) is not None]
    return float(np.mean(values)) if values else None


def _rem_period_row(run: GateOpenRun, windows: Sequence[CueArousalWindow], config: CueArousalConfig) -> Dict[str, object]:
    inside = [window for window in windows if window.rem_period_index == run.index]
    cues = [window for window in inside if window.played]
    return {
        **run.to_dict(),
        "cues": len(cues),
        "controls": sum(1 for window in inside if window.kind == "control"),
        "cues_with_any_flag": sum(1 for window in cues if window.flags.get("any")),
        "disrupted": (
            any(window.flags.get("any") or _ended_within(window, config.rem_termination_seconds) for window in cues)
            if cues
            else None
        ),
    }


def _dose_tables(windows: Sequence[CueArousalWindow], config: CueArousalConfig) -> Dict[str, object]:
    """Played cues by effective volume (a software gain), by the time since the cue
    played before them and by their index in a REM block (``puzzle_cue_index``,
    ``tlr_event_index``); the ``any`` rate and each flag's rate per bin."""
    played = [window for window in windows if window.played]
    volume_edges = tuple(config.volume_bin_edges)
    interval_edges = (0.0,) + tuple(config.interval_bin_edges_seconds)
    volume_labels = _bin_labels(volume_edges, "")
    interval_labels = _bin_labels(interval_edges, " s")
    by_volume: Dict[str, List[CueArousalWindow]] = {}
    by_interval: Dict[str, List[CueArousalWindow]] = {}
    by_index: Dict[str, List[CueArousalWindow]] = {}
    for window in played:
        volume = window.cue.get("effective_volume")
        label = "unknown" if volume is None else _bin_label(float(volume), volume_edges, volume_labels)
        by_volume.setdefault(label, []).append(window)
        interval = window.cue.get("inter_cue_interval_seconds")
        label = "first" if interval is None else _bin_label(float(interval), interval_edges, interval_labels)
        by_interval.setdefault(label, []).append(window)
        by_index.setdefault(_index_label(window.cue), []).append(window)
    return {
        "effective_volume": _table(by_volume, list(volume_labels) + ["unknown"]),
        "inter_cue_interval": _table(by_interval, ["first"] + list(interval_labels)),
        "cue_index": _table(by_index, sorted(by_index, key=_index_sort_key)),
    }


def _bin_labels(edges: Sequence[float], unit: str) -> Tuple[str, ...]:
    """``low-high`` for each pair of edges and ``>=last`` above them."""
    labels = [f"{low:g}-{high:g}{unit}" for low, high in zip(edges, edges[1:])]
    labels.append(f">={edges[-1]:g}{unit}")
    return tuple(labels)


def _bin_label(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    """The label of the bin [edge, next edge) that holds ``value``; below the first edge
    counts in the first bin."""
    position = bisect.bisect_right(edges, value) - 1
    return labels[min(max(position, 0), len(labels) - 1)]


def _index_label(cue: Mapping[str, object]) -> str:
    if cue.get("synthetic_cue"):
        return "synthetic"
    if cue.get("tlr_event_index") is not None:
        return f"tlr {int(cue['tlr_event_index'])}"
    if cue.get("puzzle_cue_index") is not None:
        return f"puzzle {int(cue['puzzle_cue_index'])}"
    return "unindexed"


def _index_sort_key(label: str):
    kind, _, number = label.partition(" ")
    order = {"tlr": 0, "puzzle": 1, "synthetic": 2, "unindexed": 3}
    return (order.get(kind, 4), int(number) if number.isdigit() else 0)


def _table(groups: Mapping[str, Sequence[CueArousalWindow]], order: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for label in order:
        windows = groups.get(label)
        if not windows:
            continue
        counts = _flag_counts(windows)
        rows.append(
            {
                "bin": label,
                "cues": len(windows),
                "any_flagged": counts["any"]["flagged"],
                "any_scored": counts["any"]["scored"],
                "any_rate": counts["any"]["rate"],
                "flag_rates": {flag: counts[flag]["rate"] for flag in AROUSAL_FLAGS},
            }
        )
    return rows


def _limitations(
    windows: Sequence[CueArousalWindow],
    *,
    inputs: _NightInputs,
    runs: Sequence[GateOpenRun],
    config: CueArousalConfig,
    optics_capable: bool,
    synthetic: bool,
    time_base_inferred: bool,
    awakening_notes: Mapping[str, int],
) -> Tuple[AnalysisLimitation, ...]:
    cues = [window for window in windows if window.kind == "cue"]
    played = [window for window in cues if window.played]
    controls = [window for window in windows if window.kind == "control"]
    items = [
        AnalysisLimitation(
            code="thresholds_unfitted",
            severity="warning",
            message=(
                "The rise ratios and the heart-rate jump are starting values set by hand and "
                "not fitted on any night; flags move with them."
            ),
            metadata={name: getattr(config, name) for name in UNFITTED_THRESHOLDS},
        ),
        AnalysisLimitation(
            code="arousal_proxy_only",
            severity="info",
            message=(
                "Every flag is an arousal proxy from forehead EEG, the IMU and optics, not an "
                "AASM-scored arousal: no occipital alpha, no chin EMG, and no comparison with "
                "PSG-scored arousals."
            ),
        ),
        AnalysisLimitation(
            code="software_gain",
            severity="info",
            message="Effective volume is a software gain from 0 to 1, not a sound level in dB SPL.",
        ),
        AnalysisLimitation(
            code="descriptive_only",
            severity="info",
            message="Rates and differences are descriptive; no significance test is run.",
        ),
    ]
    if synthetic:
        items.append(
            AnalysisLimitation(
                code="synthetic_input",
                severity="warning",
                message="The recording is a synthetic night: synthetic input, not sleep.",
            )
        )
    silent_night = inputs.audio_log_found and played and not any(window.cue.get("audible") for window in played)
    if inputs.pilot3 or silent_night:
        items.append(
            AnalysisLimitation(
                code="no_audio_played",
                severity="warning",
                message=(
                    "No cue made a sound (a Pilot 3 replay, mock or dry-run audio, or synthetic "
                    "cues), so nothing flagged at a cue can come from the cue."
                ),
            )
        )
    if time_base_inferred:
        items.append(
            AnalysisLimitation(
                code="time_base_inferred",
                severity="warning",
                message=(
                    "The summary has no first_epoch_start, so the cue times count from the "
                    "recording's first frame, where the first epoch starts on a full replay."
                ),
            )
        )
    if not cues:
        items.append(AnalysisLimitation(code="no_cues", severity="warning", message="The night has no play event to score."))
    elif len(played) < config.min_cues_for_rates:
        items.append(
            AnalysisLimitation(
                code="small_n",
                severity="warning",
                message=f"Fewer than {config.min_cues_for_rates} played cues; read the rates as counts.",
                metadata={"played_cues": len(played)},
            )
        )
    not_played = [window.index for window in cues if not window.played]
    if not_played:
        items.append(
            AnalysisLimitation(
                code="cue_not_played",
                severity="info",
                message="Some play events were blocked or skipped by the player and are left out of the rates.",
                metadata={"cue_indexes": not_played},
            )
        )
    if not inputs.gate_records:
        items.append(
            AnalysisLimitation(
                code="gate_log_missing",
                severity="warning",
                message=(
                    "No gate log: no controls and no REM periods, and each cue is placed at the "
                    "end of a 30 s epoch starting at its scheduler time."
                ),
            )
        )
    unpaired = [window.index for window in played if window.paired_index is None]
    if inputs.gate_records and unpaired:
        items.append(
            AnalysisLimitation(
                code="unpaired_cues",
                severity="info",
                message="Some cues had no quiet moment in the same gate state more than the exclusion away from every play.",
                metadata={"cue_indexes": unpaired, "control_exclusion_seconds": config.control_exclusion_seconds},
            )
        )
    span = config.baseline_seconds + config.post_window_seconds
    offsets = sorted(window.offset_seconds for window in played if window.offset_seconds is not None)
    close = sum(1 for before, after in zip(offsets, offsets[1:]) if after - before < span)
    if close:
        items.append(
            AnalysisLimitation(
                code="cue_windows_overlap",
                severity="info",
                message=(
                    "Some cues came less than a baseline plus a post window after the one before, "
                    "so a baseline can hold the earlier cue's response."
                ),
                metadata={"overlapping_pairs": close},
            )
        )
    if cues and inputs.audio_log_found is False and not inputs.pilot3:
        items.append(
            AnalysisLimitation(
                code="audio_log_missing",
                severity="warning",
                message="No audio_playback.jsonl: the dose tables use the play events' volumes or none.",
            )
        )
    if inputs.audio_log_mismatch:
        items.append(
            AnalysisLimitation(
                code="audio_log_mismatch",
                severity="warning",
                message="audio_playback.jsonl does not line up with the play events; unmatched cues have no volume.",
            )
        )
    _count_limitation(items, windows, "eeg_gap", "EEG windows could not be scored: too much of them was missing.")
    _count_limitation(items, windows, "eeg_unusable", "EEG windows had no channel that was read, not flat and not clipping.")
    _count_limitation(items, windows, "imu_gap", "IMU windows could not be scored: too much of them was missing.")
    _count_limitation(items, windows, "hr_unavailable", "Windows had optics but too few used heart-rate series values on one side of the cue.")
    if windows and all("ppg_absent" in window.quality_flags for window in windows):
        items.append(
            AnalysisLimitation(
                code="ppg_absent",
                severity="info",
                message=(
                    "No window had optics, so no heart-rate flag was scored"
                    + (" (the preset streams none)." if not optics_capable else ".")
                ),
                metadata={"by_design": not optics_capable},
            )
        )
    if inputs.awakenings is None and played:
        items.append(
            AnalysisLimitation(
                code="awakening_log_missing",
                severity="info",
                message="No awakening_events.jsonl: the cue-attributed awakening rate is not available.",
            )
        )
    if awakening_notes.get("from_utc"):
        items.append(
            AnalysisLimitation(
                code="awakening_time_from_utc",
                severity="info",
                message="Some awakenings have no timestamp_seconds and were placed by their UTC time against the frame clock.",
                metadata={"awakenings": awakening_notes["from_utc"]},
            )
        )
    if awakening_notes.get("unplaced"):
        items.append(
            AnalysisLimitation(
                code="awakening_unplaced",
                severity="warning",
                message="Some awakenings have neither timestamp_seconds nor a readable UTC time and were left out.",
                metadata={"awakenings": awakening_notes["unplaced"]},
            )
        )
    return tuple(items)


def _count_limitation(items: List[AnalysisLimitation], windows: Sequence[CueArousalWindow], code: str, message: str) -> None:
    count = sum(1 for window in windows if code in window.quality_flags)
    if count:
        items.append(
            AnalysisLimitation(
                code=code,
                severity="info",
                message=f"{count} {message}",
                metadata={"windows": count},
            )
        )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _optics_capable_from_metadata(recording_dir: Path) -> bool:
    """Whether the recording's preset streams optics, as the CLI decides it; an unknown
    preset counts as optics, as everywhere else."""
    from muse_tmr.presets import preset_provides_optics

    try:
        payload = json.loads((recording_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    source = payload.get("source", {}) if isinstance(payload, dict) else {}
    preset = dict(source.get("metadata", {}) or {}).get("preset") if isinstance(source, dict) else None
    return bool(preset_provides_optics(preset))


def _says_synthetic(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("synthetic") is True


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    path = Path(path).expanduser()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_text(output_path: Path, text: str) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator <= 0 else numerator / denominator


def _difference(left: Optional[float], right: Optional[float]) -> Optional[float]:
    return None if left is None or right is None else left - right


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_int(value: object) -> Optional[int]:
    return None if value is None else int(value)


def _json_safe(value):
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _count_text(counts: Mapping[str, object]) -> str:
    scored = int(counts.get("scored", 0) or 0)
    if not scored:
        return "n/a"
    flagged = int(counts.get("flagged", 0) or 0)
    return f"{flagged} of {scored} ({flagged / scored:.0%})"


def _rate_text(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.0%}"


def _points_text(value: object) -> str:
    return "n/a" if value is None else f"{100 * float(value):+.0f} pp"


def _number_text(value: object, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
