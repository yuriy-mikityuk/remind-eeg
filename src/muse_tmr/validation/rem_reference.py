"""The replay, the reference stagers and the provenance of the REM check (#226).

``scripts/validate_rem_detection.py`` and ``muse-tmr validate-external`` share these, so
that the detector's input, the EEG a stager is given and the provenance block come from
one implementation. They moved here from the script without a change in what they do.

- ``collect_from_source()``: one pass over a source's frames, the heuristic ``p_rem`` per
  30 s epoch and the continuous EEG a stager reads (``STAGER_CONTINUOUS_CONFIG``).
- ``night_source()``: a recording directory replayed through ``ReplaySession``, or a BOAS
  night (OpenNeuro ds005555) through ``BoasHeadbandSession`` with its EEG alone.
- ``resample_for_stager()``, ``run_reference()`` and ``stage_eeg()``: YASA or GSSC, each in
  its own venv, through ``scripts/_yasa_stage.py`` and ``scripts/_gssc_stage.py``. Those
  scripts ship with the repository, not the wheel.
- ``build_provenance()``: the command, versions, channels, thresholds, weights, labels
  and stager behind a set of metrics.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from muse_tmr import __version__ as MUSE_TMR_VERSION
from muse_tmr.data.continuous import ContinuousBuilder, ContinuousConfig
from muse_tmr.features import new_rem_event_tracker
from muse_tmr.features.epochs import EpochBuilder, EpochConfig
from muse_tmr.models import HeuristicRemDetector
from muse_tmr.models.heuristic_rem_detector import DEFAULT_REM_WEIGHTS, HeuristicRemConfig
from muse_tmr.models.rem_detector import RemPrediction
from muse_tmr.sources.base_source import BaseMuseSource

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
EEG_CHANNELS = ("AF7", "AF8", "TP9", "TP10")
NOMINAL_EEG_RATE_HZ = 256.0
GAP_FILL_THRESHOLD_SECONDS = 0.5
# The reconstruction the stager has always been given: frames without AF7 are dropped,
# a dropout over 0.5 s is zero-filled by host time, and lost subpackets are not filled
# (an export fills them; keeping them out keeps these metrics comparable with #115/#116).
STAGER_CONTINUOUS_CONFIG = ContinuousConfig(
    gap_policy="zero_fill",
    gap_threshold_seconds=GAP_FILL_THRESHOLD_SECONDS,
    nominal_rate_hz=NOMINAL_EEG_RATE_HZ,
    channels=EEG_CHANNELS,
    reference_channel="AF7",
    fill_lost_samples=False,
)

# --eog when it is not given. GSSC's README documents horizontal or left/right EOG
# only, and AF8 is a second EEG channel, so GSSC stages from AF7 alone unless asked;
# YASA keeps AF8 as its EOG derivation.
DEFAULT_EOG = {"gssc": "none", "yasa": "AF8"}
# Reference stagers, each shelled into its own isolated venv (schemas identical).
REFERENCE_SCRIPTS = {"yasa": "_yasa_stage.py", "gssc": "_gssc_stage.py"}
REFERENCES = tuple(REFERENCE_SCRIPTS)


class StagerError(RuntimeError):
    """A reference stager failed or could not be started."""


# --------------------------------------------------------------------------- #
# One replay pass: heuristic p_rem per epoch AND continuous EEG for the stager
# --------------------------------------------------------------------------- #
async def collect_from_source(
    source: BaseMuseSource,
    epoch_seconds: float,
    weights: Optional[Mapping[str, float]] = None,
    *,
    keep_eeg: bool = True,
) -> Dict[str, object]:
    """Single pass over the frames of ``source``.

    Feeds every MuseFrame to the EpochBuilder (-> heuristic p_rem per epoch)
    while side-accumulating per-channel EEG samples (uV) for the reference
    stager. The heuristic epochs are anchored to wall-clock frame timestamps, so
    to keep the stager's sample-indexed 30s epochs on the SAME clock, we zero-fill any
    BLE-dropout / AF7-loss gap: array position then maps to real elapsed time and
    stager epoch j covers real [t0 + j*30, ...). Without the fill, a mid-night gap
    would slide every later stager epoch earlier and mispair the two grids.
    ``weights`` replaces the detector's DEFAULT_REM_WEIGHTS (see the script's
    resolve_weights). The reconstruction is muse_tmr.data.continuous with
    STAGER_CONTINUOUS_CONFIG. Without ``keep_eeg`` no EEG is kept and ``channels`` is
    None; ``t0`` is then the first epoch's start. ValueError when no EEG was read.
    """
    continuous = ContinuousBuilder(STAGER_CONTINUOUS_CONFIG) if keep_eeg else None

    async def teed(stream):
        async for frame in stream:
            if continuous is not None:
                continuous.add(frame)
            yield frame

    await source.connect()
    detector = HeuristicRemDetector(
        None if weights is None else HeuristicRemConfig(weights=dict(weights))
    )
    rem_tracker = new_rem_event_tracker()
    builder = EpochBuilder(EpochConfig(epoch_seconds=epoch_seconds, stride_seconds=epoch_seconds))
    predictions: List[Tuple[float, RemPrediction]] = []
    try:
        async for epoch in builder.build(teed(source.stream())):
            prediction = detector.predict_epoch(epoch, rem_tracker=rem_tracker)
            predictions.append((epoch.start_time, prediction))
    finally:
        await source.stop()

    if continuous is None:
        if not predictions:
            raise ValueError("no usable EEG samples reconstructed from the recording")
        return {"predictions": predictions, "channels": None, "sfreq": None, "t0": predictions[0][0], "n_samples": 0}
    try:
        eeg = continuous.result().eeg
    except ValueError:
        eeg = None
    if eeg is None or eeg.n_samples == 0 or eeg.t_last <= eeg.t0:
        raise ValueError("no usable EEG samples reconstructed from the recording")
    return {
        "predictions": predictions,
        "channels": {name: eeg.channels[name] for name in EEG_CHANNELS},
        "sfreq": float(eeg.measured_rate_hz),
        "t0": float(eeg.t0),
        "n_samples": int(eeg.n_samples),
    }


def night_source(path: Path) -> BaseMuseSource:
    """The source the REM check replays: a BOAS night through ``BoasHeadbandSession``
    with ``muse_tmr.data.boas.DETECTOR_MODALITIES``, anything else through
    ``ReplaySession`` at full speed."""
    from muse_tmr.data.boas import DETECTOR_MODALITIES, BoasHeadbandSession, is_boas_night
    from muse_tmr.data.replay import ReplayConfig, ReplaySession

    if is_boas_night(path):
        return BoasHeadbandSession(Path(path), modalities=DETECTOR_MODALITIES)
    return ReplaySession(ReplayConfig(input_path=Path(path), speed=0.0))


# --------------------------------------------------------------------------- #
# Reference stager (subprocess into its isolated venv)
# --------------------------------------------------------------------------- #
def resample_for_stager(channels: Dict[str, np.ndarray], sfreq: float, target: float = 100.0) -> Tuple[Dict[str, np.ndarray], float]:
    """Resample to ~target Hz with a fast polyphase FIR, preserving real duration.

    The Muse's empirical rate is a non-integer, measured per night as ``sfreq``. YASA resamples to 100
    Hz internally, and doing that from an irrational ratio on a multi-million
    sample array triggers a pathologically slow full-length FFT resample. Pre-
    resampling here (polyphase, O(n)) shrinks the array to ~target*duration
    samples and makes YASA's own step trivial. Duration is preserved, so YASA
    epoch j still maps to real time [t0 + j*30, ...).
    """
    from fractions import Fraction

    from scipy.signal import resample_poly

    frac = Fraction(target / sfreq).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    resampled = {
        name: resample_poly(arr.astype(np.float64), up, down).astype(np.float32)
        for name, arr in channels.items()
    }
    # Declare exactly `target` Hz. The polyphase result is ~target (e.g. 99.999),
    # but a near-unity sfreq makes YASA's internal resample-to-100 pathologically
    # slow; declaring exactly 100 makes it a no-op. The <0.01% rate error is ~0.4s
    # of drift over a full night -- far below the 30s epoch grid.
    return resampled, float(target)


def stager_script(reference: str) -> Path:
    """The stager script of ``reference`` in the repository's ``scripts/``."""
    if reference not in REFERENCE_SCRIPTS:
        raise ValueError(f"reference must be one of: {', '.join(REFERENCES)}")
    return SCRIPTS_DIR / REFERENCE_SCRIPTS[reference]


def run_reference(
    reference: str, npz_path: Path, eeg: str, eog: str, python_path: Path
) -> Dict[str, object]:
    """Run the stager of ``reference`` with ``python_path`` (its venv's python, not
    resolved: resolving the symlink would drop the venv) on ``npz_path``; its hypnogram
    JSON. StagerError when it fails or its script is not there."""
    stager = stager_script(reference)
    if not stager.is_file():
        raise StagerError(
            f"{stager.name} not found in {stager.parent}: the stager scripts come with a clone of the repository"
        )
    out_path = npz_path.with_suffix(".hypno.json")
    cmd = [
        str(python_path), str(stager),
        "--npz", str(npz_path),
        "--eeg", eeg,
        "--eog", eog,
        "--out", str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise StagerError(f"{reference} staging could not start: {exc}\ncmd: {' '.join(cmd)}") from exc
    if result.returncode != 0:
        raise StagerError(
            f"{reference} staging failed (exit {result.returncode}).\n"
            f"cmd: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(out_path.read_text())


def stage_eeg(
    channels: Mapping[str, np.ndarray],
    sfreq: float,
    reference: str,
    eeg: str,
    eog: str,
    python_path: Path,
    *,
    runner=None,
) -> Tuple[Dict[str, object], float]:
    """Resample ``channels`` for the stager, write them as the .npz the stager scripts
    read and run it; the hypnogram and the rate it was given. ``runner`` stands in for
    ``run_reference`` (tests)."""
    resampled, stage_sfreq = resample_for_stager(dict(channels), sfreq)
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "eeg.npz"
        np.savez(
            npz_path,
            sfreq=np.float64(stage_sfreq),
            ch_names=np.array(list(EEG_CHANNELS)),
            **{name: resampled[name] for name in EEG_CHANNELS},
        )
        hypno = (runner or run_reference)(reference, npz_path, eeg, eog, python_path)
    return hypno, stage_sfreq


def resolve_eog(reference: str, eog: Optional[str]) -> str:
    """The --eog value for ``reference``: the one given, or that reference's default."""
    return DEFAULT_EOG[reference] if eog is None else eog


def eog_channel(eog: Optional[str]) -> Optional[str]:
    """The channel the stager gets as EOG; None for 'none' or '' (as the stagers read it)."""
    return None if eog is None or str(eog).lower() in ("", "none") else str(eog)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_provenance(
    *,
    argv: Sequence[str],
    reference: Optional[str],
    hypno: Mapping[str, object],
    eeg: Optional[str],
    eog: Optional[str],
    preset: Optional[str],
    enter_threshold: float,
    exit_threshold: float,
    resample_hz: Optional[float],
    epoch_seconds: float,
    weights: Optional[Mapping[str, float]] = None,
    weight_overrides: Optional[Mapping[str, float]] = None,
    labels_source: Optional[str] = None,
    labels_file_sha256: Optional[str] = None,
    dataset_version: Optional[str] = None,
    stride_seconds: Optional[float] = None,
) -> Dict[str, object]:
    """What produced these metrics: command, versions, channels, thresholds, detector
    weights (``weight_overrides`` as given to --weights, ``weights`` as used), stager,
    and the external labels the metrics are against, if any.

    The stager versions come from the ``stager`` block the stager script writes into
    its hypnogram JSON, since it runs in its own venv. They are None when the
    hypnogram has no such block, and the stager's ``tool`` is None when none ran.
    ``labels_source`` names the labels (``boas:stage_hum``, ``labels:<file>``),
    ``labels_file_sha256`` the file they were read from and ``dataset_version`` the
    version of the dataset it belongs to. ``stride_seconds`` defaults to the epoch.
    """
    import scipy

    eog_name = eog_channel(eog)
    passed = set() if eeg is None else ({eeg} if eog_name is None else {eeg, eog_name})
    stager = hypno.get("stager") or {}
    return {
        "command_line": shlex.join(str(arg) for arg in argv),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "muse_tmr_version": MUSE_TMR_VERSION,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "resample_hz": None if resample_hz is None else float(resample_hz),
        "epoch_seconds": float(epoch_seconds),
        "stride_seconds": float(epoch_seconds if stride_seconds is None else stride_seconds),
        "preset": preset,
        "enter_threshold": float(enter_threshold),
        "exit_threshold": float(exit_threshold),
        "weights": {name: float(w) for name, w in (weights or DEFAULT_REM_WEIGHTS).items()},
        "weight_overrides": {name: float(w) for name, w in (weight_overrides or {}).items()},
        "eeg": eeg,
        "eog": eog_name,
        "channels_not_passed": [name for name in EEG_CHANNELS if name not in passed] if reference else [],
        "labels_source": labels_source,
        "labels_file_sha256": labels_file_sha256,
        "dataset_version": dataset_version,
        "stager": {
            "tool": reference,
            "package_version": stager.get("package_version"),
            "torch_version": stager.get("torch_version"),
            "mne_version": stager.get("mne_version"),
            "returns_probabilities": stager.get("returns_probabilities"),
        },
    }
