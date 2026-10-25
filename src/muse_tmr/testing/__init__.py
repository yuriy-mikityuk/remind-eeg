"""Synthetic test data for the replay, detector, gate and scheduler path.

Synthetic input, not sleep: these nights exercise the code, they are never evidence
about REM detection or cueing.
"""

from muse_tmr.testing.scenarios import SCENARIOS, Scenario, scenario_config, scenario_names
from muse_tmr.testing.synthetic import (
    SYNTHETIC_NOTE,
    Arousal,
    ScriptedNight,
    SyntheticNight,
    SyntheticNightConfig,
    generate_night,
    heartbeat_onsets,
    script_night,
)

__all__ = [
    "SCENARIOS",
    "SYNTHETIC_NOTE",
    "Arousal",
    "Scenario",
    "ScriptedNight",
    "SyntheticNight",
    "SyntheticNightConfig",
    "generate_night",
    "heartbeat_onsets",
    "scenario_config",
    "scenario_names",
    "script_night",
]
