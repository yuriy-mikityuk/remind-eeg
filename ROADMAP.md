# Roadmap

Dated 2026-10-27. Each item's issue has its live state; this page sums them up and is updated by hand.

## Where REMind stands

REMind is n = 1 research software, not a medical device, with one maintainer. Where the evidence stands:

- Every accuracy figure comes from one EEG-only night, scored against GSSC, an automated stager used as a proxy, not PSG. The `p21` gate thresholds were tuned on that night and reported on it, so they are not a held-out result.
- No night with sleep-time audio has been run. Pilots 4 and 5 exist as tooling only.
- The heuristic REM detector is effectively one feature, `low_delta_power`. The count of eye-movement deflections meant to replace the saturated eye-movement proxy is computed at weight 0 and has only been checked on synthetic input. [#116](https://github.com/yuriy-mikityuk/remind-eeg/issues/116), open, reweights the detector and re-grids the `p21` thresholds once there are REM-bearing nights to fit them on.
- How long after REM onset the first cue comes is measured on synthetic input only. [#167](https://github.com/yuriy-mikityuk/remind-eeg/issues/167) was closed by [#242](https://github.com/yuriy-mikityuk/remind-eeg/pull/242) on 2026-10-23, once the gate counted real time at any stride; no real night has measured it.
- REMind is not on PyPI and has no DOI yet; both wait on the maintainer ([#232](https://github.com/yuriy-mikityuk/remind-eeg/issues/232)).

The figures and their sources are in the README's [project status](README.md#project-status-2026-10-23) and in [`docs/numbers.md`](docs/numbers.md).

## The ten initiatives

The research of October 2026 ([`docs/research/2026-10-uniqueness/`](docs/research/2026-10-uniqueness/report.md)) ended in ten initiatives, filed as sub-issues of [EPIC #1](https://github.com/yuriy-mikityuk/remind-eeg/issues/1).

| Initiative | State on 2026-10-27 |
|---|---|
| 1. Replace the saturated eye-movement proxy, refit the detector, publish the first-cue latency ([#225](https://github.com/yuriy-mikityuk/remind-eeg/issues/225)) | 1a, 1b and 1c merged ([#239](https://github.com/yuriy-mikityuk/remind-eeg/pull/239), [#241](https://github.com/yuriy-mikityuk/remind-eeg/pull/241), [#242](https://github.com/yuriy-mikityuk/remind-eeg/pull/242)). The refit, 1d, waits for two or more REM-bearing `p21` nights. |
| 2. Validate the detector and the gate against PSG consensus on the open BOAS dataset ([#226](https://github.com/yuriy-mikityuk/remind-eeg/issues/226)) | Not started. |
| 3. Cue-locked arousal scoring with quiet control windows ([#227](https://github.com/yuriy-mikityuk/remind-eeg/issues/227)) | The scorer, the gate log and the Pilot 3, 4 and 5 wiring in [#261](https://github.com/yuriy-mikityuk/remind-eeg/pull/261). Scoring a night with sound waits on the maintainer. |
| 4. Sham and no-cue nights, N-of-1 blocks, delayed retest, blind incorporation coding ([#228](https://github.com/yuriy-mikityuk/remind-eeg/issues/228)) | Not started. |
| 5. Honesty, provenance and positioning pass ([#229](https://github.com/yuriy-mikityuk/remind-eeg/issues/229)) | Merged ([#236](https://github.com/yuriy-mikityuk/remind-eeg/pull/236), [#237](https://github.com/yuriy-mikityuk/remind-eeg/pull/237)). Open only for a re-run of the REM check on the night behind the figures, if that recording turns up. |
| 6. EDF+/BIDS export, MNE, and import of hypnograms and other recordings ([#230](https://github.com/yuriy-mikityuk/remind-eeg/issues/230)) | Not started. |
| 7. A loadable, cited protocol-preset registry with a fidelity report ([#231](https://github.com/yuriy-mikityuk/remind-eeg/issues/231)) | Not started. |
| 8. Install, cite, govern ([#232](https://github.com/yuriy-mikityuk/remind-eeg/issues/232)) | 8a, 8b and 8d merged ([#253](https://github.com/yuriy-mikityuk/remind-eeg/pull/253), [#257](https://github.com/yuriy-mikityuk/remind-eeg/pull/257), [#240](https://github.com/yuriy-mikityuk/remind-eeg/pull/240)), and 8c with `CITATION.cff`, `CONTRIBUTING.md`, this page and the issue templates. Publishing waits on the maintainer. |
| 9. Synthetic night corpus and a demo without a headband ([#233](https://github.com/yuriy-mikityuk/remind-eeg/issues/233)) | Done ([#243](https://github.com/yuriy-mikityuk/remind-eeg/pull/243), [#245](https://github.com/yuriy-mikityuk/remind-eeg/pull/245)). |
| 10. Optics heart rate against a Polar H10 ECG ([#234](https://github.com/yuriy-mikityuk/remind-eeg/issues/234)) | 10a, 10b and 10c merged ([#238](https://github.com/yuriy-mikityuk/remind-eeg/pull/238), [#252](https://github.com/yuriy-mikityuk/remind-eeg/pull/252), [#254](https://github.com/yuriy-mikityuk/remind-eeg/pull/254)). Paired sessions wait on the maintainer. |

The open ones come in this order: 3, 7, 4, 6, 2. [EPIC #1](https://github.com/yuriy-mikityuk/remind-eeg/issues/1) has the dependencies between them.

## Waiting on the maintainer

These need the maintainer's own nights, hardware or accounts:

- two or more REM-bearing `p21` nights without audio, for the refit of [#225](https://github.com/yuriy-mikityuk/remind-eeg/issues/225) and [#116](https://github.com/yuriy-mikityuk/remind-eeg/issues/116)
- a PyPI trusted publisher for `remind-eeg` and the repository variable `PYPI_PUBLISH`, the Zenodo integration, and the release tag ([`docs/release.md`](docs/release.md))
- one listen to `muse-tmr play-test-cue --backend system` through the intended headphones
- one low-volume Pilot 4 night with sound after a Pilot 2 calibration, scored with `muse-tmr score-cue-arousal` ([#227](https://github.com/yuriy-mikityuk/remind-eeg/issues/227))
- paired Muse and H10 sessions in three postures, and a `p1034` night with the H10 ([#234](https://github.com/yuriy-mikityuk/remind-eeg/issues/234))

One ask stands for anyone with a sleep lab: a night of the Muse S Athena next to PSG would give REMind its first reference that is not an automated stager.

## Not planned

From the same research ([`strategy_judgement.md`](docs/research/2026-10-uniqueness/notes/strategy_judgement.md)):

- claims of efficacy of any kind, for lucid dreaming, memory, sleep or health
- network upload, telemetry or cloud sync
- a real-audio backend or a night runner for Linux or Windows, since sound is only tested on macOS
- a documentation site, Discussions, a code of conduct, a plugin registry or a public benchmark table, for now
- competing on raw streaming, NREM slow-oscillation stimulation, general sleep staging or fNIRS
