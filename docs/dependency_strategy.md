# amused-py Dependency Strategy

Decision: use a forked-source strategy for now.

This repository was created from `Amused-EEG/amused-py` and keeps the Muse BLE/data-source implementation in the repository while the REM-TMR layer is built. Do not add `amused-py` as a submodule or PyPI dependency yet.

Pinned upstream baseline:

- remote: `https://github.com/Amused-EEG/amused-py`
- ref: `bce20f98ddc7fa2efe3219d1b5d2f7554a55eb97`

## Rationale

- The current code already contains the Athena BLE protocol implementation.
- Local changes need to coordinate closely with recording, replay, and test fixtures.
- A submodule would add workflow complexity before the project has stable adapter boundaries.
- A PyPI dependency would make it harder to patch protocol-level behavior quickly.

## Sync Plan

1. Keep the `upstream` remote pointed at `Amused-EEG/amused-py`.
2. Periodically run `git fetch upstream --prune`.
3. Review upstream changes before merging or cherry-picking.
4. Port small protocol fixes intentionally with tests.
5. Avoid broad upstream merges during active feature work.

## Contribution Policy

Upstream first. A fix to how the Muse is connected to or decoded goes to the project that owns that code before, or at the same time as, it lands here:

- the BLE and packet decoding in the top-level `amused.py` and `muse_*.py` modules: [amused-py](https://github.com/Amused-EEG/amused-py)
- the BrainFlow board code behind `muse_tmr.sources.brainflow_source`: [BrainFlow](https://github.com/brainflow-dev/brainflow)
- the OpenMuse decoder behind `muse_tmr.sources.openmuse_lsl_source`: [OpenMuse](https://github.com/DominiqueMakowski/OpenMuse)

A fix carried here first is recorded in the commit and in this file, so the fork can shrink once upstream has it. Keep REM detection, cue scheduling, reports, validation, and sleep-protocol code in this repository.

Fixes already upstream:

- BrainFlow [PR #837](https://github.com/brainflow-dev/brainflow/pull/837), "Fix Muse Athena macOS reconnect hangs", merged 2026-05-21 and released in BrainFlow 5.22.2.

Fixes carried here first:

- amused-py: `MuseRealtimeDecoder` keeps the rows of every ACCGYRO subpacket in a notification, not only the last one's, and marks lost ones in `IMUSample.gaps` ([#244](https://github.com/yuriy-mikityuk/remind-eeg/issues/244)). On 2026-10-25 `muse_realtime_decoder.py` on amused-py `main` (last changed in `ca52abf`) still assigns the IMU rows per subpacket, and the EEG and optics rows too, which this fork already appends.

Nothing has been sent to amused-py or OpenMuse yet.

## BrainFlow Policy

BrainFlow is an optional acquisition adapter, not a replacement foundation for this
repository. Use the released `brainflow>=5.22.1` package by default. A personal
BrainFlow fork may be used to prototype or validate Muse S Athena fixes, but only pin
this project to a git dependency temporarily when a live-smoke blocker cannot wait for
an upstream BrainFlow release.
