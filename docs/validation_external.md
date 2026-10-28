# The REM detector against PSG consensus: BOAS

`muse-tmr validate-external` scores the REM detector and the stable gate against the sleep stages that three human scorers agreed on, on the open BOAS dataset ([OpenNeuro ds005555](https://github.com/OpenNeuroDatasets/ds005555), CC0). It is the repository's first comparison with PSG; every other agreement figure here is against an automated stager. What it can show is a detector transfer-validated on a frontal montage (Bitbrain Ikon, mastoid reference): another headband than the Muse, referenced to the mastoid instead of near Fpz, against PSG. It says nothing about a Muse night and nothing about cueing. The run over all 128 nights was made on 2026-10-28; its results are [below](#results-of-the-run-of-2026-10-28).

## The dataset

BOAS v1.1.3 ([doi:10.18112/openneuro.ds005555.v1.1.3](https://doi.org/10.18112/openneuro.ds005555.v1.1.3); Esparza-Iaizzo et al. 2026, [doi:10.1016/j.compbiomed.2026.111560](https://doi.org/10.1016/j.compbiomed.2026.111560)) holds 128 nights of 100 participants, each recorded at once with a Bitbrain Ikon headband and a Micromed PSG. `participants.tsv` lists every night (`sub-1` to `sub-128`) with `pid`, the participant: 20 of them have two or three nights. REMind reads, per night under `sub-<n>/eeg/`:

| File | What REMind takes from it |
| --- | --- |
| `sub-<n>_task-Sleep_acq-headband_eeg.edf` | `HB_1` and `HB_2`, forehead EEG near AF7 and AF8 in µV against the left mastoid, at 256 Hz, as AF7 and AF8 |
| `sub-<n>_task-Sleep_acq-psg_events.tsv` | `stage_hum`, the consensus: each epoch has the stage at least two of three AASM scorers gave it, a fourth scorer deciding when all three differed; and `stage_ai`, Bitbrain's network on the PSG |
| `sub-<n>_task-Sleep_acq-headband_events.tsv` | `stage_ai` of that network on the headband EEG, kept as `reference_stage_ai` |
| `sub-<n>_task-Sleep_acq-headband_channels.tsv` | which headband channels the night has |
| `participants.tsv`, `dataset_description.json` | `pid` of each night; the version, from `DatasetDOI` |

The PSG EDFs are not read: the consensus is in the events file, and the headband is the device being checked. Stages are 0 W, 1 to 3 N1 to N3, 4 REM, 8 a PSG disconnection (human-scored only) and −2 artefact or missing data (AI-scored only). `load_boas_hypnogram()` maps `stage_hum` to the REM annotation labels: 0 `wake`, 1 to 3 `nrem`, 4 `probable_rem`, and 8, −2 or a missing value `unknown`, which is never scored.

Headband and PSG are on one clock in the dataset: both events files of a night list the same epochs, from 0 s in steps of 30 s, and the headband and PSG sidecars give the same `RecordingDuration`. The loader checks the two files of every night it reads and refuses one where onsets or the number of epochs differ, or where the epochs do not follow one another, so the headband's epoch k is always paired with the PSG's epoch k.

`reference_stage_ai` is the headband's `stage_ai`, not the PSG's: it is a stager given the same forehead EEG REMind reads, so it shows what a frontal-only model trained for this headband reaches. It was trained on BOAS itself, so it is context in the results, not a baseline to beat.

## What the detector is given

The EEG alone, as AF7 and AF8: there is no TP9 or TP10, and the headband IMU is not in the units its files name. On the 44 nights with IMU on disk on 2026-10-28, the night median of the accelerometer's magnitude was 10.4 to 70.4 in channels labelled g, where gravity alone is 1 g, and the largest per-axis night median of the gyroscope was 12.5 to 969 deg/s in channels labelled deg/s, where a sleeper lies nearly still. The motion features and the arousal guard's motion rules would read noise or offsets. So the detector runs without its stillness term, and the gate with `optics_capable` and `imu_capable` off (`rem_metrics.simulate_gate`), the way an EEG-only preset turns off the cardiac codes: missing IMU and missing optics do not cap its confidence. A Muse night has an IMU, so this is the detector's EEG part with the gate's timing, not the whole live system. `BoasHeadbandSession` can still emit the IMU and `HB_PULSE` for other uses.

GSSC is given `HB_1` (AF7) alone and YASA `HB_1` with `HB_2` (AF8) as EOG, as `scripts/validate_rem_detection.py` gives them a Muse night, resampled to 100 Hz.

## Getting the data

With [DataLad](https://www.datalad.org/), the headband files and the scoring tables, without the PSG EDFs, which are not needed:

```bash
datalad clone https://github.com/OpenNeuroDatasets/ds005555 ~/Datasets/openneuro/ds005555
cd ~/Datasets/openneuro/ds005555
datalad get sub-*/eeg/*_acq-headband_* sub-*/eeg/*_acq-psg_events.tsv
```

`participants.tsv` and `dataset_description.json` come with the clone. In the OpenNeuro S3 listing of 2026-10-28 these files, with the sidecars and the dataset's top-level files, are 13.45 GB (1,289 files); the smoke run of that day fetched them from that bucket. Keep the dataset outside the repository: nothing of it may be committed, and `*.edf` is refused by `scripts/check_forbidden_files.py` anyway.

The stagers each need their own venv beside the checkout, as for the REM check (`docs/USAGE.md`): `python -m venv .venv-gssc && .venv-gssc/bin/pip install gssc==0.0.9`, and `.venv-yasa` with `yasa`. The heuristic, the fit and the personal model need only the `[edf]` extra.

## The run

One command does all of it, and can be stopped and started again:

```bash
muse-tmr validate-external --dataset boas --root ~/Datasets/openneuro/ds005555 \
  --splits data/validation_public/boas/splits.json --make-splits --seed 0 \
  --output-dir data/validation_public/boas \
  --reference gssc --gssc-python .venv-gssc/bin/python \
  --reference yasa --yasa-python .venv-yasa/bin/python --jobs 4
```

1. Splits. `--make-splits` shuffles the 100 participants with `--seed` and puts them into the `fit` fold until it holds half of the nights (`--fit-fraction`), the rest into `report`. Every night of a participant is in one fold. The splits come from `participants.tsv` alone, so they do not depend on which files are on disk; an existing `--splits` file must hold the same splits, or the command stops. Their sha256 is that of the text `--make-splits` writes, so a checkout that turned its line ends into CRLF has the same. Without `--make-splits` the file is read and checked: a participant with nights in both folds, a recording that is not in the dataset or has another `pid`, or another dataset version, and nothing runs.
2. Each night, once. The headband EEG is replayed through the heuristic detector (`epochs.csv`: per 30 s epoch the consensus, `p_rem`, reason codes and feature scores and values, no EEG and no clock time), and on report nights GSSC and YASA stage it (`gssc.json`, `yasa.json`). Each file is written to a temporary name and moved into place, and `night.json` last, with the sha256 of the splits and the code version (package version and git commit). A night already there with the same key is skipped, so a run that stopped goes on where it stopped; `--rescore` replays every night again.
3. The fit, on the fit fold only. `boas_v1` is the gate pair with the highest pooled κ of the gate against the consensus REM, over a grid of enter 0.05 to 0.95 and exit 0.05 up to enter, in steps of 0.05 (190 pairs). It is wider than the REM check's 0.70 to 0.95, since `p_rem` need not run as high on another montage as on a Muse night and a personal model's probabilities are calibrated; ties go to the higher pair. The personal model (`boas_v1_personal_rem_model.json`) is trained on the fit nights' epochs with the EEG features of `EEG_PERSONAL_REM_FEATURES`, and gets its own pair from the same grid. Both go into `fit.json` with the recordings and participants they were fitted on.
4. The report, on the report fold only: each threshold set of `--gate-thresholds` (default `p21` and `boas_v1`) over the heuristic, the personal model, the stagers, and the headband `stage_ai` for context, per night (`report.json`, `report_epochs.csv`) and pooled (`metrics.json`). A report night of a participant the fit used, or that the personal model was trained on, stops the run.

`boas_v1` is opt-in. `presets.py` holds the pair the run of 2026-10-28 fitted, enter 0.40 and exit 0.35, with the sha256 of the committed `splits.json`, and `validate-external` scores that pair on those splits only, with or without a fit step in the output directory. On other splits, such as another seed or the synthetic layouts of the tests, `boas_v1` is the pair the run's own fit step wrote (`fit.json`), and the log says so. A run with no fit night selected uses that `fit.json` only when it was fitted on the same splits with the same code version and the personal model beside it is the one it records; otherwise, or with no fit there, `boas_v1` on other splits stops the command before it replays anything, and `p21` alone is reported without the personal model. `preset_gate_thresholds("p21")` and every live default stay as they are; whether `boas_v1` replaces the `p21` pair is for the maintainer to decide after a check on a Muse night.

`--nights sub-3,sub-4` limits a run to some recordings, for a smoke test on part of the dataset. On this Mac one night took 7 to 9 s for the heuristic, 26 to 30 s for GSSC and about 7 s for YASA (three report nights, 2026-10-28), so all 128 nights take about an hour in one process.

For one night, `scripts/validate_rem_detection.py <root>/sub-3 --labels <root>/sub-3 --preset p21 [--reference gssc]` prints the REM check against the consensus, and the stager's agreement with it with `--reference`.

## The files

All of them go into `--output-dir`. Under `data/validation_public/boas/` only the first two may be committed: `.gitignore` admits nothing else there and `scripts/check_forbidden_files.py` refuses everything else, by the maintainer's decision.

| File | Content | Committed |
| --- | --- | --- |
| `splits.json` | Seed, fit fraction, dataset version and DOI, sha256 of `participants.tsv`, and the recordings and `pid`s of each fold | yes, of the run of 2026-10-28 |
| `metrics.json` | Pooled metrics per system, per-feature AUC, the thresholds used and where they came from, the splits' sha256, the code version, provenance, the stagers' versions and the limitations. It names no local path: the command in its provenance has `<root>`, `<splits>`, `<output-dir>` and `<gssc-python>` or `<yasa-python>` for the paths, and a night left out gets the first line of its error with every path cut to its last part. Its only clock time is the run's own, `provenance.timestamp_utc` | yes, of the run of 2026-10-28 |
| `fit.json` | `boas_v1` and the personal model's pair with their grids, the fit recordings and participants | no |
| `boas_v1_personal_rem_model.json` | The personal model, version 2, with `training_data` | no |
| `per_night_metrics.json` | Every system's metrics per report night, and the whole error of each night left out | no |
| `nights/<sub>/` | `epochs.csv`, `night.json`, the stagers' hypnograms, `report.json` (with the command as given) and `report_epochs.csv` | no |

A dataset whose `dataset_description.json` says `synthetic: true`, as the layouts of the tests do, gets `synthetic input, not sleep` in every JSON it writes.

## The metrics

For every system and report night, on the epochs whose consensus is scored:

- κ (binary, REM against not REM, not five-class), F1, precision and recall of the epochs the system calls REM: the gate's open epochs, a stager's REM epochs, `stage_ai` 4. Precision here is the gate-open precision.
- Precision at recall 0.80 of the system's score (`p_rem`, the personal model's probability, a stager's `proba_rem`): the best precision of a threshold that finds 80 % of the REM epochs. GSSC 0.0.9 and `stage_ai` have hard labels, so they have one operating point.
- REM-onset latency: from the start of each run of consensus REM epochs to the system's first open decision inside it, a decision counting at the end of its epoch. Median and interquartile range over all runs, with how many runs had none. GSSC and YASA read the whole night, so theirs is not the latency of a causal gate.
- False-open minutes per night: open epochs outside consensus REM; mean, median and interquartile range over the nights.
- Per-feature AUC: for every `score_*` and `value_*` column of the heuristic, its ROC-AUC against the consensus REM over the pooled report epochs, as the REM check reports it against a stager.

Pooled figures put the report nights' epochs together; κ also comes as its median over nights. Every prediction of the detector, the personal model and the gate for an epoch depends on that epoch and the ones before it only (`tests/test_external_labels.py` checks it). The IMU and pulse `BoasHeadbandSession` can emit are resampled over the whole file and so read a little past each frame, but the detector does not get them.

`metrics.json` also counts, under `nights.headband`, the scored epochs on which the detector had no EEG score at all (flat, clipped or missing headband EEG), and the nights where that was more than half of them; those epochs stay in the metrics, as a live night would have them. A night whose files cannot be read, or on which a stager fails, is left out of the fit and the report and named under `nights.left_out`, so one bad file does not end an unattended run. When every report-fold night the run selected is left out, it writes no `metrics.json`, names them and exits with code 1.

## Results of the run of 2026-10-28

The command above ran over all 128 nights on 2026-10-28 at commit `d7485ce`, with `--jobs 3`, on BOAS v1.1.3 and the splits of seed 0 (sha256 `d7642b15b42e00cb735141ab45a8d73ff5bd19a31633644673b4eee11ad9617c`). Every night was scored and none was left out. The fit used the 64 nights of the 51 participants in the fit fold; the report is on the other 64 nights, of 49 participants, with 60,012 scored epochs, 10,024 of them REM by the consensus. `splits.json` and `metrics.json` are committed under `data/validation_public/boas/`; the per-night files, `fit.json` and the personal model stay on the machine that ran it.

The fit chose `boas_v1` enter 0.40, exit 0.35 (κ 0.110 on the fit fold) and the gate pair 0.25/0.15 for the personal model. Pooled over the report fold, binary REM κ against the consensus:

| System | κ | F1 | Gate-open precision | Precision at recall 0.80 | REM-onset latency, median (IQR) s | False-open min per night, mean |
|---|---|---|---|---|---|---|
| Heuristic, `p21` gate 0.90/0.85 | −0.000 | 0.00 | 0.08 | 0.19 | 270, in 1 of 398 REM runs | 0.2 |
| Heuristic, `boas_v1` gate 0.40/0.35 | 0.155 | 0.29 | 0.30 | 0.19 | 60 (30 to 360), in 165 of 398 | 51.5 |
| Personal model, gate 0.25/0.15 | 0.391 | 0.49 | 0.49 | 0.28 | 90 (30 to 180), in 323 of 398 | 39.8 |
| GSSC 0.0.9, AF7 | 0.802 | 0.84 | 0.82 | 0.82 | 30 (30 to 30), in 352 of 398 | 15.1 |
| YASA 0.7.0, AF7 with AF8 as EOG | 0.672 | 0.72 | 0.82 | 0.71 | 30 (30 to 90), in 317 of 398 | 10.8 |
| Headband `stage_ai`, context | 0.838 | 0.87 | 0.84 | 0.84 | 30 (30 to 60), in 360 of 394 | 12.7 |

Per night the spread is wide. The median κ over the 64 report nights, with its interquartile range, is 0.081 (−0.011 to 0.332) for the heuristic at `boas_v1`, 0.404 (0.238 to 0.546) for the personal model, 0.880 (0.790 to 0.922) for GSSC and 0.742 (0.496 to 0.848) for YASA (`kappa_per_night`).

- At the `p21` pair the gate was open on 24 of the 60,012 epochs. `p_rem` from this headband's EEG rarely reaches 0.90: the pair was tuned on a Muse night, where the detector runs hot, and does not carry over. The row says that, not how well `p_rem` ranks REM: its ROC-AUC is 0.613.
- With the pair fitted on the fit fold, the heuristic reaches κ 0.155. On this montage, with its EEG alone, it is weak.
- The personal model, a logistic regression on the same per-epoch EEG features, trained on the fit fold, reaches κ 0.391.
- GSSC and YASA, staging the same forehead EEG, reach κ 0.802 and 0.672, but they read the whole night, so their latency is not that of a gate. Neither was trained on BOAS: GSSC 0.0.9 was published in September 2023 and YASA's classifiers in August 2021, before BOAS came out in October 2024. Bitbrain's own network (`stage_ai`) was trained on BOAS.

Per-feature AUC of the heuristic's scores against the consensus REM, over the report epochs: `low_delta_power` 0.553, `theta_alpha_ratio` 0.573, `rem_event_density` 0.637 (weight 0, not fitted yet) and `eye_movement_proxy` 0.500. Of the values behind them, the relative delta power has 0.395, lower in REM, and the theta/alpha ratio 0.586. They are input for the refit of the detector ([#225](https://github.com/yuriy-mikityuk/remind-eeg/issues/225)); this run changes no weight. The detector had no EEG score on 1,801 of the 60,012 scored epochs (flat, clipped or missing headband EEG), and on one night on more than half of them.

## What this cannot say

- A headband other than the Muse, with another reference and amplifier. The detector's band thresholds were set by hand for the Muse; on this montage the score of `low_delta_power` ranks REM less well (AUC 0.553) than the relative delta power it is computed from (0.605 read the other way round, as 1 − 0.395: it is lower in REM).
- No IMU: the stillness term and the motion blocks of the live system are not tested here.
- The splits keep a participant's nights in one fold, but the consensus scorers and the lab are the same for all nights.
- `boas_v1` and the personal model are fitted on about half of one dataset's nights; nothing here says how they do on a Muse night.
