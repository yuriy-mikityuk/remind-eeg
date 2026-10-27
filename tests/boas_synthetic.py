"""A dataset laid out like BOAS (OpenNeuro ds005555), written into a temporary directory.

Synthetic input, not sleep (#226). Each night's headband EEG is the AF7 and AF8 of a
synthetic night of ``muse_tmr.testing``, written as ``HB_1`` and ``HB_2`` of an EDF at
256 Hz; its ``stage_hum`` is the generated hypnogram (W 0, N1 1, N2 2, N3 3, R 4), and the
events files, sidecars, ``participants.tsv`` and ``dataset_description.json`` follow the
layout of the BOAS README and sidecars. Nothing of BOAS itself is read or copied, and
nothing here is committed: the files are made in-test. Writing the EDF needs the
``[edf]`` extra; ``write_layout(..., edf=False)`` writes everything else.
"""

from __future__ import annotations

import asyncio
import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from muse_tmr.data.continuous import ContinuousConfig, reconstruct_recording
from muse_tmr.testing import SyntheticNightConfig, generate_night
from muse_tmr.testing.synthetic import SYNTHETIC_NOTE

STAGE_CODES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
BOAS_DOI_TEXT = "doi:10.18112/openneuro.ds005555.v1.1.3"
RATE_HZ = 256

# A night of 66 epochs (33 min): sleep onset, NREM down to N3 and back, two REM periods
# with N2 between them, and a brief awakening at the end.
NIGHT_STAGES = (
    ("W",) * 6 + ("N1",) * 3 + ("N2",) * 10 + ("N3",) * 8 + ("N2",) * 5 + ("R",) * 12
    + ("N2",) * 8 + ("R",) * 10 + ("W",) * 4
)


@dataclass(frozen=True)
class NightSpec:
    """One night of the layout: its recording id, participant, hypnogram and seed;
    ``with_imu`` also writes made-up HB_IMU_1 to 6 and HB_PULSE channels, ``unscored``
    gives those epoch indices stage_hum 8 (a PSG disconnection), and
    ``stage_ai_changes`` sets the headband stage_ai of those epochs."""

    recording_id: str
    pid: str
    stages: Tuple[str, ...] = NIGHT_STAGES
    seed: int = 0
    with_imu: bool = False
    unscored: Tuple[int, ...] = ()
    stage_ai_changes: Dict[int, int] = field(default_factory=dict)


TWO_NIGHTS = (
    NightSpec("sub-1", "1", seed=1, with_imu=True, unscored=(0,)),
    NightSpec("sub-2", "2", seed=2, stage_ai_changes={20: 4, 21: 4}),
)


def write_layout(root: Path, nights: Sequence[NightSpec] = TWO_NIGHTS, *, edf: bool = True, work: Optional[Path] = None):
    """Write the BOAS-shaped dataset of ``nights`` under ``root``; return ``root``.

    ``work`` holds the synthetic nights the EEG comes from (default ``root.parent``)."""
    root = Path(root)
    work = Path(work) if work is not None else root.parent / f"{root.name}_synthetic_nights"
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "dataset_description.json",
        {
            "synthetic": True,
            "note": SYNTHETIC_NOTE,
            "Name": "BOAS-shaped test layout",
            "BIDSVersion": "1.8.0",
            "DatasetDOI": BOAS_DOI_TEXT,
            "License": "CC0",
        },
    )
    with (root / "participants.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["participant_id", "age", "sex", "bmi", "pid"])
        for spec in nights:
            writer.writerow([spec.recording_id, "n/a", "n/a", "n/a", spec.pid])
    for spec in nights:
        write_night(root, spec, edf=edf, work=work)
    return root


def write_night(root: Path, spec: NightSpec, *, edf: bool = True, work: Optional[Path] = None) -> Path:
    eeg_dir = Path(root) / spec.recording_id / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    name = f"{spec.recording_id}_task-Sleep"
    _write_events(eeg_dir / f"{name}_acq-psg_events.tsv", spec, psg=True)
    _write_events(eeg_dir / f"{name}_acq-headband_events.tsv", spec, psg=False)
    channels = ["HB_1", "HB_2"] + ([f"HB_IMU_{i}" for i in range(1, 7)] + ["HB_PULSE"] if spec.with_imu else [])
    units = {"HB_1": "uV", "HB_2": "uV", "HB_PULSE": "nA"}
    with (eeg_dir / f"{name}_acq-headband_channels.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["name", "type", "units", "sampling_frequency"])
        for channel in channels:
            unit = units.get(channel, "g" if channel in ("HB_IMU_1", "HB_IMU_2", "HB_IMU_3") else "deg/s")
            writer.writerow([channel, "EEG" if channel.startswith("HB_") and channel[3].isdigit() else "MISC", unit, RATE_HZ])
    seconds = len(spec.stages) * 30
    _write_json(
        eeg_dir / f"{name}_acq-headband_eeg.json",
        {"EEGChannelCount": 2, "EEGReference": "M1 (left mastoid)", "RecordingDuration": seconds,
         "SamplingFrequency": RATE_HZ, "TaskName": "Sleep", "synthetic": True, "note": SYNTHETIC_NOTE},
    )
    if edf:
        _write_edf(eeg_dir / f"{name}_acq-headband_eeg.edf", spec, Path(work) if work else Path(root).parent)
    return eeg_dir


def headband_eeg(spec: NightSpec, work: Path) -> Dict[str, np.ndarray]:
    """AF7 and AF8 in uV of the synthetic night behind ``spec``, one sample per 1/256 s
    from its first frame, exactly ``len(stages) * 30`` s long."""
    hours = len(spec.stages) * 30.0 / 3600.0
    config = SyntheticNightConfig(
        seed=spec.seed, hours=hours, preset="p21", scenario="boas_layout_test", stages=tuple(spec.stages)
    )
    night = generate_night(config, Path(work) / spec.recording_id)
    recording = asyncio.run(
        reconstruct_recording(night.recording_dir, ContinuousConfig(channels=("AF7", "AF8"), reference_channel="AF7"))
    )
    n = len(spec.stages) * 30 * RATE_HZ
    out = {}
    for name in ("AF7", "AF8"):
        values = np.asarray(recording.eeg.channels[name], dtype=float)[:n]
        out[name] = np.pad(values, (0, max(0, n - values.size)))
    return out


def _write_edf(path: Path, spec: NightSpec, work: Path) -> None:
    import edfio

    eeg = headband_eeg(spec, work)
    signals = [
        edfio.EdfSignal(eeg[channel], RATE_HZ, label=label, physical_dimension="uV",
                        physical_range=(-3000.0, 3000.0), prefiltering="HP:0.15Hz")
        for label, channel in (("HB_1", "AF7"), ("HB_2", "AF8"))
    ]
    if spec.with_imu:
        rng = np.random.default_rng(spec.seed)
        n = eeg["AF7"].size
        for index in range(1, 7):
            unit = "g" if index <= 3 else "deg/s"
            values = rng.normal(0.0, 0.05 if index <= 3 else 2.0, n) + (1.0 if index == 3 else 0.0)
            limit = 4.0 if index <= 3 else 250.0
            signals.append(edfio.EdfSignal(values, RATE_HZ, label=f"HB_IMU_{index}", physical_dimension=unit,
                                           physical_range=(-limit, limit)))
        pulse = 5.0 * np.sin(2 * np.pi * 1.1 * np.arange(n) / RATE_HZ)
        signals.append(edfio.EdfSignal(pulse, RATE_HZ, label="HB_PULSE", physical_dimension="nA",
                                       physical_range=(-60.0, 60.0)))
    edfio.Edf(
        signals,
        patient=edfio.Patient(additional=tuple(SYNTHETIC_NOTE.split())),
        recording=edfio.Recording(startdate=dt.date(2023, 1, 1)),
        starttime=dt.time(23, 0, 0),
        data_record_duration=1.0,
    ).write(path)


def _write_events(path: Path, spec: NightSpec, *, psg: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["onset", "duration", "begsample", "endsample", "offset"] + (["stage_hum"] if psg else []) + ["stage_ai"])
        for index, stage in enumerate(spec.stages):
            code = STAGE_CODES[stage]
            hum = 8 if index in spec.unscored else code
            # stage_ai_changes are the headband network's; the PSG's agrees with stage_hum.
            ai = code if psg else spec.stage_ai_changes.get(index, code)
            begin = index * 30 * RATE_HZ + 1
            row = [index * 30, 30, begin, begin + 30 * RATE_HZ - 1, 0] + ([hum] if psg else []) + [ai]
            writer.writerow(row)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
