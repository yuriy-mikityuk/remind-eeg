"""Short labelled nights for the data exchange tests of #230. Synthetic input, not sleep.

``labelled_night()`` writes a night from ``muse_tmr.testing`` with a fixed hypnogram into
a temporary directory; nothing here is committed or read from a recording. ``frames()``
replays one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Sequence

from muse_tmr.data.replay import ReplayConfig, ReplaySession
from muse_tmr.testing import SyntheticNightConfig, generate_night

# Two minutes of wake, two of N2, two of REM and one of N1: seven minutes, every stage the
# lossy map has a text for. p1034, so EEG, IMU, optics and battery are all there.
STAGES = ("W",) * 4 + ("N2",) * 4 + ("R",) * 4 + ("N1",) * 2


def labelled_night(
    output_dir: Path,
    *,
    stages: Sequence[str] = STAGES,
    preset: str = "p1034",
    ble_gaps_per_hour: float = 0.0,
    seed: int = 0,
):
    hours = len(stages) * 30.0 / 3600.0
    config = SyntheticNightConfig(
        seed=seed,
        hours=hours,
        preset=preset,
        scenario="exchange_test",
        stages=tuple(stages),
        ble_gaps_per_hour=ble_gaps_per_hour,
    )
    return generate_night(config, Path(output_dir))


def frames(recording: Path) -> List:
    async def collect():
        session = ReplaySession(ReplayConfig(input_path=Path(recording), speed=0.0))
        return [frame async for frame in session.stream()]

    return asyncio.run(collect())
