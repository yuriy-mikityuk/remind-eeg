"""Muse preset capability helpers.

Single source of truth for what a device preset actually streams. The REM gate
uses this to decide whether a missing cardiac (PPG/HR) signal reflects a sensor
that is *off by design* (EEG-only presets) versus one that is *expected but
dropped out* -- only the latter should lower REM confidence.

Note: this is preset-derived, not source-derived. Sources advertise a static
capability map (e.g. amused reports ``heart_rate: True``) regardless of the
preset in use, so ``MuseSourceMetadata.capabilities`` cannot tell you that a p21
night carries no optics. The preset is the authoritative signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Presets that stream EEG (+ IMU) only, with the optics/PPG LED stack powered
# off. Everything else drives the full 8-channel optics payload, from which both
# PPG heart rate and fNIRS are derived (they are not separable by preset).
NO_OPTICS_PRESETS = frozenset({"p21"})

# Recalibrated (enter, exit) REM-gate thresholds by preset. The heuristic
# detector runs hot (p_rem pinned high in NREM), so the live gate over-fires at
# the historical 0.70/0.45. Raising EXIT is the dominant lever: NREM p_rem rarely
# drops below ~0.6, so a low exit never closes the gate.
#
#   - p21 (EEG-only): tuned in-sample on one EEG-only night (overnight_20260707,
#     n=1) against GSSC, an automated proxy, not PSG; fitted and reported on the
#     same night, so this is not held-out validation. 0.90/0.85 cut gate-open from
#     ~75% to ~36% of that night while holding REM recall ~0.80. Kappa and epoch
#     counts are only in PR #115 / #116 (see docs/numbers.md). p21 thresholds were
#     fitted with the pre-reweighting weights; re-grid pending (#116).
#   - Optics presets: PROVISIONAL. The only p1034 recording is a 0-REM nap, so
#     p1034 REM *recall* at these thresholds is UNVERIFIED (false-alarm collapse
#     only). 0.80/0.70 is the recall-protective (lower-enter) choice on purpose;
#     re-grid once a REM-bearing optics night is recorded.
_PRESET_GATE_THRESHOLDS = {
    "p21": (0.90, 0.85),
}
_DEFAULT_OPTICS_GATE_THRESHOLDS = (0.80, 0.70)


@dataclass(frozen=True)
class GateThresholdSet:
    """A named (enter, exit) REM-gate pair that no preset uses, with the sha256 of the
    splits.json on whose fit folds it was fitted, so that nothing scores it on a
    participant it was fitted on."""

    enter_threshold: float
    exit_threshold: float
    splits_sha256: str


# Named threshold sets beside the presets' pairs (#226). They are opt-in, through
# `muse-tmr validate-external --gate-thresholds`: preset_gate_thresholds() never returns
# one and no live command reads them. A preset also says whether optics stream, so a pair
# fitted on another headband cannot be one.
#
#   - boas_v1: the pair validate-external fitted on the fit fold of BOAS (OpenNeuro
#     ds005555 v1.1.3, a Bitbrain Ikon headband referenced to the mastoid, EEG only) in
#     the run of 2026-10-28 at commit d7485ce: of the fit grid, the pair with the highest
#     pooled κ of the gate against the PSG consensus on the 64 fit nights (κ 0.110). It
#     belongs to data/validation_public/boas/splits.json, whose sha256 it carries, and
#     validate-external uses it on those splits only. A transfer to a frontal montage,
#     not a pair for a Muse night.
_NAMED_GATE_THRESHOLDS: Dict[str, Optional[GateThresholdSet]] = {
    "boas_v1": GateThresholdSet(
        enter_threshold=0.40,
        exit_threshold=0.35,
        splits_sha256="d7642b15b42e00cb735141ab45a8d73ff5bd19a31633644673b4eee11ad9617c",
    ),
}


def preset_provides_optics(preset: Optional[str]) -> bool:
    """Whether ``preset`` streams the optics/PPG stack.

    Unknown or empty presets conservatively return ``True`` (assume optics is
    present), so the REM gate keeps its cardiac-coverage confidence cap unless we
    positively know the preset is EEG-only.
    """
    if not preset:
        return True
    return preset.strip().lower() not in NO_OPTICS_PRESETS


def preset_gate_thresholds(preset: Optional[str]) -> Tuple[float, float]:
    """Default ``(enter, exit)`` REM-gate thresholds for ``preset``.

    Unknown/None or any optics preset returns the conservative, recall-protective
    provisional point; only p21 (EEG-only) has its own pair, tuned in-sample on
    one night against GSSC, an automated proxy.
    """
    if not preset:
        return _DEFAULT_OPTICS_GATE_THRESHOLDS
    return _PRESET_GATE_THRESHOLDS.get(preset.strip().lower(), _DEFAULT_OPTICS_GATE_THRESHOLDS)


def named_gate_threshold_sets() -> Tuple[str, ...]:
    """The names of the opt-in threshold sets, ``boas_v1`` among them."""
    return tuple(_NAMED_GATE_THRESHOLDS)


def named_gate_thresholds(name: str) -> Optional[GateThresholdSet]:
    """The named set ``name``; None for one that has no values yet. KeyError for a name
    that is not one (a preset's pair comes from preset_gate_thresholds())."""
    key = name.strip().lower()
    if key not in _NAMED_GATE_THRESHOLDS:
        raise KeyError(f"unknown gate threshold set {name!r}; known: {', '.join(_NAMED_GATE_THRESHOLDS)}")
    return _NAMED_GATE_THRESHOLDS[key]
