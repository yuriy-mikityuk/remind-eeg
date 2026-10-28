"""Validation helpers for M8 pilot workflows.

The names below load on first use, so that importing one module of this package, such
as ``rem_metrics``, ``rem_reference`` or ``external_labels``, does not import the pilot
runners and, through them, the audio and protocol packages (#226).
"""

from __future__ import annotations

import importlib
from typing import Dict, List

_EXPORTS: Dict[str, str] = {
    "GATE_LOG_SCHEMA_VERSION": "gate_log",
    "REM_GATE_DECISIONS_FILENAME": "gate_log",
    "GateDecisionRecord": "gate_log",
    "GateOpenRun": "gate_log",
    "emitting_record": "gate_log",
    "gate_decision_record": "gate_log",
    "gate_open_runs": "gate_log",
    "load_gate_open_runs": "gate_log",
    "load_rem_gate_decisions": "gate_log",
    "run_at": "gate_log",
    "write_rem_gate_decisions": "gate_log",
    "DEFAULT_PILOT1_MIN_DURATION_SECONDS": "pilot1",
    "DEFAULT_PILOT1_REQUIRED_MODALITIES": "pilot1",
    "Pilot1Criterion": "pilot1",
    "Pilot1ValidationReport": "pilot1",
    "validate_pilot1_recording": "pilot1",
    "DEFAULT_PILOT2_HARD_MAX_VOLUME": "pilot2",
    "Pilot2Criterion": "pilot2",
    "Pilot2ValidationReport": "pilot2",
    "validate_pilot2_calibration": "pilot2",
    "Pilot3Criterion": "pilot3",
    "Pilot3EpochResult": "pilot3",
    "Pilot3ReplayCueSimulationReport": "pilot3",
    "simulate_cue_plan_from_epochs": "pilot3",
    "simulate_replay_cue_plan": "pilot3",
    "AWAKENING_ATTRIBUTIONS": "pilot4",
    "AwakeningEvent": "pilot4",
    "Pilot4Criterion": "pilot4",
    "Pilot4CueingConfig": "pilot4",
    "Pilot4CueingSummary": "pilot4",
    "append_awakening_event": "pilot4",
    "audio_for_plays": "pilot4",
    "load_awakening_events": "pilot4",
    "run_pilot4_cueing_night": "pilot4",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_EXPORTS))
