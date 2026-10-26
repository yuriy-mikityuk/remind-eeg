# Usage reference

Complete command and code reference for `muse-tmr`, organized by workflow stage. The [README](../README.md) gives the high-level picture; this page contains every invocation with flags and the embeddable Python APIs.

## Contents

- [Installation & local development](#installation--local-development)
- [Acquisition: discover, stream, record](#acquisition-discover-stream-record)
- [Alternative acquisition backends](#alternative-acquisition-backends)
- [macOS BLE troubleshooting](#macos-ble-troubleshooting)
- [Blink and artifact diagnostics](#blink-and-artifact-diagnostics)
- [Offline replay](#offline-replay)
- [Synthetic nights and the demo](#synthetic-nights-and-the-demo)
- [Epoch building](#epoch-building)
- [Feature extraction: EEG](#feature-extraction-eeg)
- [Feature extraction: IMU](#feature-extraction-imu)
- [Feature extraction: PPG / HR / HRV](#feature-extraction-ppg--hr--hrv)
- [Heuristic REM baseline](#heuristic-rem-baseline)
- [Manual REM annotation templates](#manual-rem-annotation-templates)
- [Personal REM classifier](#personal-rem-classifier)
- [REM stable gate](#rem-stable-gate)
- [Checking the REM detector against a reference stager](#checking-the-rem-detector-against-a-reference-stager)
- [Low-volume test cue player](#low-volume-test-cue-player)
- [Pre-sleep volume calibration](#pre-sleep-volume-calibration)
- [Cue library metadata](#cue-library-metadata)
- [Pre-sleep puzzle session manager](#pre-sleep-puzzle-session-manager)
- [TLR cue module](#tlr-cue-module)
- [REM-gated scheduler](#rem-gated-scheduler)
- [Morning dream report](#morning-dream-report)
- [Morning puzzle retest](#morning-puzzle-retest)
- [Cued-vs-uncued analysis](#cued-vs-uncued-analysis)
- [Cue-locked arousal proxies](#cue-locked-arousal-proxies)
- [Meditation complexity analysis](#meditation-complexity-analysis)
- [Polar H10 companion recording](#polar-h10-companion-recording)
- [Optics heart rate against the H10](#optics-heart-rate-against-the-h10)
- [Local app](#local-app)
- [Validation plan and pilots](#validation-plan-and-pilots)

## Installation & local development

The pip distribution is `remind-eeg`, with the `muse-tmr` command and the `muse_tmr` import. It is not on PyPI yet, so install it from a checkout; [`release.md`](release.md) says how a release gets there. CI runs the tests and the installed wheel on Python 3.11 to 3.13 on Linux, macOS and Windows. The app launcher and real audio (`afplay`) are macOS only, and the local app starts and stops recordings with POSIX signals, which Windows lacks, so the tests of those parts are skipped on Windows. Windows before Python 3.13 has a 15.6 ms clock, on which a recording can take its last wait for a stall and reconnect right at its end, and a Pilot 4 night can end as `no_data_timeout`; the tests that time this are skipped there too. On 3.9 and 3.10 a Stop could be lost: `asyncio.wait_for` can swallow its cancel, and the recorder needs `Task.cancelling()`, new in 3.11, to notice.

```bash
pip install -e .
python -m muse_tmr.cli.main --help
muse-tmr --help
muse-tmr status
python scripts/check_forbidden_files.py
```

`muse-tmr status` prints `REMind <version>: n = 1 research software, not a medical device`, and `muse-tmr --help` ends with the same line. `muse-tmr --version` prints only `muse-tmr <version>`.

A venv set up before the rename has the checkout installed as `athena-tmr-lucid`. Run `pip uninstall athena-tmr-lucid` before installing again, so that two distributions do not both own the `muse-tmr` command.

The BLE layer is the package `muse_tmr.ble`. Code that imports its modules by their old top-level names, such as `from muse_stream_client import MuseStreamClient` or `import amused` in the checkout, works until 0.5 with a `DeprecationWarning`; import them from `muse_tmr.ble` instead.

## Acquisition: discover, stream, record

```bash
muse-tmr discover --source amused
muse-tmr stream --source amused --duration-seconds 3600
muse-tmr record --source amused --duration-hours 8
```

Short smoke-test recordings require `--allow-short`; normal overnight recordings are constrained to 2-8 hours.
Recordings are written to `data/recordings/<timestamp>/` by default. Pass `--output-dir` to pin a specific session directory.

After discovery, prefer the discovered BLE address for live checks:

```bash
MUSE_ADDR="<discovered-address>"
muse-tmr stream --source amused --address "$MUSE_ADDR" --duration-seconds 60
muse-tmr record --source amused --address "$MUSE_ADDR" --duration-seconds 60 --allow-short
```

## Alternative acquisition backends

### OpenMuse LSL source

```bash
pip install -e ".[openmuse]"
muse-tmr discover --source openmuse --require-lsl-stream eeg
muse-tmr stream --source openmuse --duration-seconds 60 --require-lsl-stream eeg
```

OpenMuse must run separately and publish LSL streams such as `Muse_EEG` and `Muse_ACCGYRO`. The adapter reads available LSL modalities into the same `MuseFrame` contract as the BLE source; `mne_lsl` or `pylsl` plus local `liblsl` remains optional.

### BrainFlow source

```bash
pip install -e ".[brainflow]"
muse-tmr discover --source brainflow
muse-tmr stream --source brainflow --address "$MUSE_ADDR" --duration-seconds 60 --debug-stats
```

BrainFlow remains an optional acquisition backend, not the project foundation. The adapter targets `MUSE_S_ATHENA_BOARD`, uses `p1041` with low latency by default, and maps BrainFlow EEG, IMU, optics, and battery rows into the existing `MuseFrame` contract. Keep `amused` as the default until live smoke tests show BrainFlow is more reliable for Muse S Athena.

If BrainFlow connect/reconnect is flaky, keep the process bounded and leave BLE time to settle between attempts:

```bash
muse-tmr stream --source brainflow --address "$MUSE_ADDR" \
  --duration-seconds 10 \
  --brainflow-connect-timeout 20 \
  --brainflow-stream-start-timeout 10 \
  --brainflow-stop-timeout 10 \
  --brainflow-session-cooldown 5 \
  --debug-stats
```

### Official SDK policy

```bash
muse-tmr stream --source sdk --sdk-path /local/path/to/sdk
```

The SDK source is currently a policy-enforcing stub only. It imports without any SDK installed, but runtime operations fail until a local-only adapter is explicitly built. Do not commit official SDK binaries, headers, frameworks, archives, installers, docs, or copied vendor code. Run `python scripts/check_forbidden_files.py` before publishing SDK-adjacent changes.

## macOS BLE troubleshooting

If direct `python3 -m muse_tmr.cli.main discover --source amused` or `muse-tmr discover --source amused` aborts before printing Python logs, macOS TCC is likely blocking CoreBluetooth for the Python bundle. Use a virtual environment and launch the framework `Python.app` through LaunchServices:

```bash
cd /path/to/remind-eeg
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/python -m pip install -e .

PYAPP=/opt/homebrew/Cellar/python@3.12/3.12.3/Frameworks/Python.framework/Versions/3.12/Resources/Python.app
PYTHONPATH="$PWD/src:$PWD/.venv/lib/python3.12/site-packages"

open -W -n \
  --stdout /tmp/muse-discover.out \
  --stderr /tmp/muse-discover.err \
  --env "PYTHONPATH=$PYTHONPATH" \
  "$PYAPP" \
  --args -m muse_tmr.cli.main discover --source amused

cat /tmp/muse-discover.out
cat /tmp/muse-discover.err
```

If using the `Python.app` workaround, keep the same `open ... "$PYAPP" --args` wrapper and replace only the command after `--args` with the `stream` or `record` invocation.

## Blink and artifact diagnostics

```bash
muse-tmr diagnose-blink-artifacts --source amused --address "$MUSE_ADDR" [--output data/reports/<name>.json]
muse-tmr compare-source-diagnostics data/reports/<a>.json data/reports/<b>.json [--output compare.md]
muse-tmr inspect-blink-channels data/reports/<a>.json [--phase blink] [--output inspect.csv]
```

`diagnose-blink-artifacts` prompts through a short live protocol (settle, eyes open, blinks, recovery, jaw clench, head movement, eyes closed; about 3.5 min with the defaults) and writes a JSON report of amplitude and rate metrics per phase. Its `--source` defaults to `brainflow`, so pass `--source amused` for the default backend. `compare-source-diagnostics` puts several reports side by side as a source quality table, and `inspect-blink-channels` shows the per-channel metrics of the chosen phases. Both print markdown and can also write `.md`, `.csv` or `.json`.

## Offline replay

```bash
muse-tmr replay data/recordings/<session>
muse-tmr replay data/recordings/<session> --speed 1.0
muse-tmr replay data/recordings/<session> --speed 20 --start-seconds 1800 --end-seconds 3600
```

Replay accepts either a recording directory or a `raw_amused.bin` path. `--speed 1.0` is real time; `--speed 0.0` replays as fast as possible for tests and batch feature extraction.

## Synthetic nights and the demo

```bash
muse-tmr demo
muse-tmr demo --scenario fragmented_rem --hours 8 --seed 1 --output data/synthetic/fragmented --no-open
muse-tmr synthesize-night --list
muse-tmr synthesize-night --scenario clean_night --hours 0.5 --output data/synthetic/clean_night
```

`synthesize-night` writes one labelled synthetic night from `muse_tmr.testing` into `--output`: `raw_amused.bin`, `metadata.json`, `events.jsonl` and `summary.json` as `record` writes them, next to the ground truth `hypnogram.csv`, `arousals.csv`, `rem_events.csv` and `generation.json`. `--hours` replaces the scenario's length, and the same `--seed` (default 0) gives the same bytes. It writes only into a new or empty directory (hidden files aside) or over an earlier synthetic night, one whose `metadata.json` says `synthetic: true`, and refuses any other. The commands that read a recording take the result, `replay`, `annotate-template`, `validate-pilot1-recording` and `simulate-replay-cues` among them.

`demo` (defaults `--scenario clean_night --hours 2 --seed 0`) generates a night into `data/synthetic/demo_<timestamp>/`, or `--output`, and runs `validate-pilot1-recording` on it. It writes Pilot 3 inputs under `protocol/` (four synthetic puzzles with a generated tone each, two of them cued), runs `simulate-replay-cues --preset <the night's preset>` with mock audio, builds the REM report, scores the gate against `hypnogram.csv`, writes `demo_summary.json` and opens the report unless `--no-open`. It exits 0 when Pilot 1 and Pilot 3 pass, no uncued puzzle got a `play`, the audio stayed mock and the report was written, and 1 otherwise. It writes only into a new or empty `--output` or over an earlier demo, one whose `demo_summary.json` says `synthetic: true`, and refuses any other with exit code 2, so it never lands in a recording's folder or replaces protocol files such as `data/protocol/puzzle_catalog.json`. `--baseline FILE` also compares the run with a saved one and exits 1 on a difference; the CI `end_to_end` job runs it with `tests/fixtures/synthetic/agreement_baseline.json`. Synthetic input, not sleep: the numbers check the plumbing and say nothing about REM detection. Scenarios, output files, sizes and limits: `docs/synthetic_data.md`.

## Epoch building

```python
import asyncio
from pathlib import Path
from muse_tmr.data.replay import ReplayConfig, ReplaySession
from muse_tmr.features.epochs import EpochBuilder, EpochConfig


async def collect_epochs(recording):
    session = ReplaySession(ReplayConfig(Path(recording), speed=0.0))
    builder = EpochBuilder(EpochConfig(epoch_seconds=30, stride_seconds=30))
    return [epoch async for epoch in builder.build(session.stream())]


epochs = asyncio.run(collect_epochs("data/recordings/<session>"))
for epoch in epochs:
    print(epoch.index, epoch.coverage, epoch.quality_flags)
```

The same builder accepts live `MuseFrame` streams from `AmusedSource`; missing modalities are represented as coverage/quality flags instead of errors. The feature and REM examples below take this `epochs` list. Without Python, `muse-tmr annotate-template` writes the same per-epoch features with `P_REM` to a CSV.

A `stride_seconds` below `epoch_seconds` gives overlapping epochs that are each still 30 s long. Code that keeps time per epoch, like the REM gate below, moves on by `epoch_step_seconds(epoch.start_time, previous_start_time, stride_seconds)` from `muse_tmr.features`: the time between epoch starts, and one stride for the first epoch.

## Feature extraction: EEG

```python
from pathlib import Path
from muse_tmr.features.eeg_features import (
    export_eeg_feature_rows,
    extract_eeg_feature_rows,
)

rows = extract_eeg_feature_rows(epochs)
export_eeg_feature_rows(rows, Path("data/reports/eeg_features.csv"))
```

EEG rows include band powers for delta/theta/alpha/beta/gamma, relative powers, theta-alpha and slow-fast ratios, frontal/posterior asymmetry, the deprecated frontal eye-movement proxy, REM events on the band-passed AF7-AF8 difference (`rem_event_count`, `rem_density_per_min`, `last_rem_event_age_s`, `rem_burst_active`, `rem_event_times`, `rem_event_threshold_uv`, `rem_event_warmup`), and artifact flags. CSV export is always available; Parquet export uses the installed pandas Parquet engine when available.

The REM-event fields come from one `RemEventTracker` that carries its filter and running threshold from epoch to epoch, so pass all epochs of a recording in one `extract_eeg_feature_rows()` call, or keep a tracker yourself:

```python
from muse_tmr.features import extract_eeg_features, new_rem_event_tracker

tracker = new_rem_event_tracker()
rows = [extract_eeg_features(epoch, rem_tracker=tracker) for epoch in epochs]
```

How the events are found and what the defaults are: `docs/data_model.md`. They are starting values, not fitted on any night (#225).

## Feature extraction: IMU

```python
from pathlib import Path
from muse_tmr.features.imu_features import (
    export_imu_feature_rows,
    extract_imu_feature_rows,
)

rows = extract_imu_feature_rows(epochs)
export_imu_feature_rows(rows, Path("data/reports/imu_features.csv"))
```

Pass `cue_timestamps_by_epoch={epoch_index: [timestamp, ...]}` to also log movement in a window around each cue.

IMU rows include motion level, stillness score, accelerometer and gyroscope peaks, movement events, arousal proxy counts, arousal guard reason codes, and cue-window movement logs. CSV export is always available; Parquet export uses the installed pandas Parquet engine when available.

## Feature extraction: PPG / HR / HRV

```python
from pathlib import Path
from muse_tmr.features.ppg_features import (
    export_ppg_feature_rows,
    extract_ppg_feature_rows,
)

rows = extract_ppg_feature_rows(epochs)
export_ppg_feature_rows(rows, Path("data/reports/ppg_features.csv"))
```

PPG rows include PPG-derived heart-rate estimates when raw optics are available, mean/min/max HR from `HeartRateSample`, HR trend, HRV proxy metrics, the sudden HR-change logs the arousal guard reads with the heart-rate series they come from (`hr_series_json`, one value a second from the epoch's optics; see `docs/data_model.md`), and missing-modality flags. CSV export is always available; Parquet export uses the installed pandas Parquet engine when available. `estimate_channel_heart_rates(epoch)` gives the peak-interval estimate of every optics channel, with its confidence, quality and periodicity.

## Heuristic REM baseline

```python
from muse_tmr.models import HeuristicRemDetector

detector = HeuristicRemDetector()

for epoch, prediction in zip(epochs, detector.predict_epochs(epochs)):
    print(epoch.index, prediction.probability, prediction.reason_codes)
```

`prediction.probability` is `P_REM` in the 0-1 range. The heuristic detector is a non-ML baseline over EEG, IMU, and PPG/HR feature rows. It returns reason codes and feature scores only; it does not play audio or make cue decisions. Its weights are `DEFAULT_REM_WEIGHTS` in `muse_tmr.models.heuristic_rem_detector`; the eye-movement proxy is still scored but weighted 0, because it is saturated on real nights (#116, #225). Its replacement, `rem_event_density`, is scored and exported with the reason codes `eeg_rem_events_support`/`eeg_rem_events_low`, also at weight 0 until it is fitted on real nights. `predict_epochs()` runs one REM-event tracker over all its epochs; when you call `predict_epoch()` per epoch, pass `rem_tracker=` the same `new_rem_event_tracker()` each time. HRV from PPG peaks with `ppg_confidence` under 0.4 adds the reason code `low_ppg_quality` without changing any score. The weights and why: `docs/data_model.md`. To see how each feature separates REM on a recorded night, and what other weights would do, use the REM check below with `--weights`.

## Manual REM annotation templates

```bash
muse-tmr annotate-template data/recordings/<session> \
  --output data/annotations/<session>_rem_labels.csv
```

The template overlays each epoch with `P_REM`, reason codes, and feature columns, including `feature_score_rem_event_density` and the `feature_value_rem_event_*` columns from one REM-event tracker over the whole recording; the first epochs carry `feature_value_rem_event_warmup` 1. The default label is `unknown`; manually edit labels to `wake`, `nrem`, or `probable_rem` before using them for training. Templates written before these columns existed still load.

## Personal REM classifier

```bash
muse-tmr train-rem-classifier data/annotations/<session>_rem_labels.csv \
  --output data/models/personal_rem_model.json
```

The trainer skips `unknown` labels, treats `probable_rem` as the positive class, and treats `wake`/`nrem` as the negative class. The saved JSON artifact is versioned and includes class-balanced logistic coefficients, calibration metrics, feature importance, and training metrics. Personal model probabilities still do not trigger audio directly; stable gates and safety layers decide whether a cue is allowed.

## REM stable gate

```python
from muse_tmr.models import RemGateConfig, StableRemGate

gate = StableRemGate(RemGateConfig(min_stable_seconds=60.0))

for prediction in rem_predictions:
    decision = gate.update(prediction)
    print(decision.gate_open, decision.state, decision.reason_codes)
```

The gate requires stable REM confidence over time, uses hysteresis for closing, blocks motion/arousal reasons, and applies cooldown after arousal blocks. It still does not play audio; cue code must consume `gate_open` plus reason codes.

Each `update()` counts `RemGateConfig.epoch_seconds` (30 s) towards the stable window and off the cooldown unless you pass `duration_seconds`. With overlapping epochs pass the time between epoch starts, `duration_seconds=epoch_step_seconds(epoch.start_time, previous_start_time, stride_seconds)`, as `simulate-replay-cues`, `run-pilot4-cueing` and `run-pilot5-full-night` do; then the 60 s window and the 120 s cooldown take as long at a 10 s stride as at 30 s.

`RemGateConfig()` opens at `P_REM` 0.80 and closes below 0.70. The pilot commands take the pair from `--preset` (`muse_tmr.presets.preset_gate_thresholds`): 0.90/0.85 on `p21`, 0.80/0.70 on an optics preset; `--enter-threshold` and `--exit-threshold` override it. The `p21` pair was fitted with the detector's earlier weights and has not been re-gridded since the eye-movement proxy went to 0 (#116). On `p21` the gate also stops treating missing PPG as low confidence, and the arousal guard stops treating it as an artifact and skips its HR rules (`docs/safety.md`). Why these values and how far they have been checked: `docs/data_model.md`.

## Checking the REM detector against a reference stager

```bash
python -m venv .venv-gssc && .venv-gssc/bin/pip install gssc==0.0.9
.venv/bin/python scripts/validate_rem_detection.py data/recordings/<session> --reference gssc --preset p21 \
  [--output data/reports/rem_validation.json] [--html data/reports/rem_validation.html] [--dump-epochs epochs.csv]
```

Replays a recording through the heuristic detector and the gate, stages the same EEG with YASA (the default) or GSSC (`--reference gssc`), each from its own venv (`.venv-yasa`, `.venv-gssc`, both gitignored), and reports REM precision, recall, F1 and kappa at a few thresholds, a threshold sweep, ROC-AUC, and how often the reference calls REM while the gate is open. Both stagers are automated proxies, not PSG. GSSC was trained mostly on C3, C4, F3, F4 (NSRR PSG) and stages from one EEG channel; there is no published evaluation on forehead-only EEG. YASA expects central electrodes and mis-stages this montage. The `p21` thresholds were tuned in-sample on one EEG-only night (`overnight_20260707`, n=1) against GSSC, an automated proxy: fitted and reported on the same night, so this is not held-out validation. Kappa and epoch counts are only in PR #115 / #116; [`numbers.md`](numbers.md) lists them with their source.

`--eog` defaults to `none` for GSSC, so it stages from AF7 alone: its README documents horizontal or left/right EOG only, and AF8 is a second EEG channel. YASA keeps AF8 as its EOG. Before this change GSSC got AF8 as EOG too, and the PR #115 run most likely did (its flags were not logged). To see what AF8 changes, run GSSC both ways on the same night:

```bash
.venv/bin/python scripts/validate_rem_detection.py data/recordings/<session> --reference gssc --preset p21 \
  --eog none --output data/reports/rem_validation_eog_none.json
.venv/bin/python scripts/validate_rem_detection.py data/recordings/<session> --reference gssc --preset p21 \
  --eog AF8 --output data/reports/rem_validation_eog_af8.json
```

`gssc` is pinned to 0.0.9, the last PyPI release, whose `mne_infer` returns stage labels only, so `proba_rem` is 0 or 1. `scripts/_gssc_stage.py` also takes the three-value return of gssc's main branch and then uses its class probabilities when they are valid. The metrics JSON carries `kappa_definition` (binary REM-vs-not κ against an automated stager; not five-class) and a `provenance` block: the command line, UTC time, `muse_tmr` version, git commit, Python, numpy and scipy versions, the resample rate, the epoch length, the preset and its gate thresholds, the EEG and EOG channels, the Muse channels the stager was not given, and a `stager` block with the package, torch and mne versions and whether it returned probabilities. Each stager script writes that `stager` block into its hypnogram JSON, because it runs in its own venv.

For refitting the detector (#225) the summary, the metrics JSON and the HTML also carry two tables. `feature_metrics` has, for every per-epoch `score_*` and `value_*` column (the ones `--dump-epochs` writes), its ROC-AUC against the reference's REM, its minimum and maximum, the number of epochs where it is not NaN, and `saturated` when it has one value in every epoch, as the eye-movement proxy had on `overnight_20260707` (#116). An AUC below 0.5 means the column runs against REM. `gate_grid` runs the real gate over the night for each enter threshold from 0.70 to 0.95 and each exit from 0.45 up to the enter value, in steps of 0.05, with the preset's `optics_capable`, and gives the gate-open fraction, REM recall and precision (with F1 and κ) per pair; `preset_pair` marks the pair the preset uses. `--weights` replays the night with other detector weights:

```bash
.venv/bin/python scripts/validate_rem_detection.py data/recordings/<session> --reference gssc --preset p21 \
  --weights theta_alpha_ratio=0,rem_event_density=0.2 --output data/reports/rem_validation_weights.json
```

The names are those of `DEFAULT_REM_WEIGHTS`; the ones not named keep their default, and then all are divided by their sum so they add up to 1, like the defaults. `provenance.weights` has the weights used and `provenance.weight_overrides` the ones given. The pilots always use the defaults. All of these are agreement with an automated proxy, not REM accuracy.

## Low-volume test cue player

```bash
muse-tmr play-test-cue --volume 0.05 --max-volume 0.20
```

`play-test-cue` uses conservative low-volume defaults, applies fade in/out, caps any requested volume at `--max-volume`, supports backend/device metadata, and can write JSONL playback logs with `--log-path`. The `system` and `afplay` backends play through macOS `afplay`. When `afplay` is not on PATH, the command exits with code 2 and says how to fix it instead of playing nothing; `--allow-dry-run` runs it silently on purpose, and the output then shows `backend=dry-run requested_backend=system`. `train-tlr-cue`, `run-pilot4-cueing` and `run-pilot5-full-night` check their `--backend` the same way. Use `--backend dry-run` for non-audible smoke tests and CI. Finding `afplay` does not prove the cue reaches the intended headphones: play one while awake and listen.

## Pre-sleep volume calibration

```bash
muse-tmr calibrate-volume \
  --device-name "Bedroom Headphones" \
  --detectable-volume 0.02 \
  --identifiable-volume 0.04 \
  --comfortable-volume 0.08 \
  --output data/calibration/volume.json

muse-tmr play-test-cue \
  --backend dry-run \
  --device-name "Bedroom Headphones" \
  --calibration data/calibration/volume.json \
  --volume 0.20
```

Volume calibration stores the detectable, identifiable, and comfortable volumes for one playback device. Scheduler code must use `comfortable_volume` as the calibrated maximum, still capped by the hard session max. Calibration files under `data/calibration/` are gitignored because they can reveal personal devices and sleep setup.

## Cue library metadata

```bash
muse-tmr create-cue-library --output data/cues/starter.json
muse-tmr validate-cue-library data/cues/starter.json
muse-tmr list-cues data/cues/starter.json --protocol puzzle
```

Cue libraries are JSON metadata catalogs for `sound`, `generated_tone`, and `silence` cues. They carry duration, protocol role (`puzzle`, `tlr`, `test`, or `generic`), tags, optional volume hints, and private sound file paths. Sound cue files under `cues/private/` or `data/cues/audio/` are gitignored by default; validation detects missing files before a sleep session.

## Pre-sleep puzzle session manager

```bash
muse-tmr import-puzzles puzzles.csv --output data/protocol/puzzle_catalog.json

muse-tmr record-puzzle-attempt data/protocol/puzzle_catalog.json \
  --puzzle-id p001 \
  --response "my answer" \
  --duration-seconds 90

muse-tmr generate-puzzle-session data/protocol/puzzle_catalog.json \
  --session-id night-001 \
  --count 4 \
  --seed 17 \
  --output data/protocol/night-001_puzzles.json

muse-tmr assign-puzzle-cues data/protocol/night-001_puzzles.json \
  --seed 23 \
  --output data/protocol/night-001_assignment.json

muse-tmr record-association-check data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --puzzle-id p001 \
  --response "remembered answer"
```

Puzzle catalogs track prompts, solutions, cue IDs, solved/known/retired flags, timed pre-sleep attempts, and cue-to-solution association checks. Session generation filters out solved, known, and retired tasks and produces a reproducible night session with four eligible unsolved puzzles by default. `data/protocol/` is gitignored because it can contain private puzzle content and responses.

`assign-puzzle-cues` randomizes half of the night-session puzzles into cued and uncued groups with a seed. Scheduler code must consume `scheduled_puzzle_ids` from the saved assignment, which contains only cued puzzles, so uncued tasks are never scheduled.

## TLR cue module

```bash
muse-tmr create-tlr-cue --output data/cues/tlr_default.json

muse-tmr train-tlr-cue data/cues/tlr_default.json \
  --output data/protocol/tlr_training.json \
  --event-log data/protocol/tlr_training.jsonl \
  --backend dry-run \
  --repetitions 3

muse-tmr plan-tlr-block data/cues/tlr_default.json \
  --output data/protocol/tlr_block.json \
  --repetitions 3 \
  --interval-seconds 8 \
  --post-block-pause-seconds 10
```

`create-tlr-cue` writes a default generated TLR cue library. `train-tlr-cue` runs pre-sleep TLR cue familiarization through the audio player facade and writes structured JSONL training events. `plan-tlr-block` produces the configurable TLR cue block that the REM-gated scheduler runs before puzzle cues.

## REM-gated scheduler

```python
from muse_tmr.protocol import ArousalGuard, TmrCueScheduler, TmrSchedulerConfig

guard = ArousalGuard()

scheduler = TmrCueScheduler(
    assignment=puzzle_cue_assignment,
    catalog=puzzle_catalog,
    cue_library=cue_library,
    tlr_block_plan=tlr_block_plan,
    config=TmrSchedulerConfig(max_puzzle_cues_per_block=4),
)

for timestamp_seconds, gate_decision, feature_rows in replayed_gate_decisions:
    guard_decision = guard.evaluate(
        timestamp_seconds=timestamp_seconds,
        eeg=feature_rows.eeg,
        imu=feature_rows.imu,
        ppg=feature_rows.ppg,
    )
    events = scheduler.update(
        gate_decision,
        timestamp_seconds=timestamp_seconds,
        guard_decision=guard_decision,
    )
```

The scheduler consumes stable REM gate decisions, optionally emits a TLR block first, then schedules only cued puzzle cues from `scheduled_puzzle_ids`. It enforces cue interval, cooldown, and max-per-block limits, and logs structured `play`, `skip`, `pause`, and `stop` events. `ArousalGuard` consumes EEG/IMU/PPG feature rows and can allow cueing, lower volume, pause cueing, or stop a session from motion, alpha, HR-jump, and artifact-quality proxies. Neither layer calls audio playback directly.

## Morning dream report

```bash
muse-tmr record-dream-report data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --output data/reports/night-001_dream_report.json \
  --lucid yes \
  --cues-heard no \
  --confidence 0.7 \
  --dream-text "I found the answer written in a notebook." \
  --puzzle-link "p1=the first puzzle answer appeared in the notebook"
```

Dream reports save local JSON with lucid yes/no, cues-heard yes/no, confidence, free-text recall, and optional per-puzzle links. Each `--puzzle-link` is validated against the generated night puzzle session so reports can connect dream content back to the experimental puzzle IDs without making uncued assumptions.

## Morning puzzle retest

```bash
muse-tmr record-puzzle-retest data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --assignment data/protocol/night-001_assignment.json \
  --output data/reports/night-001_retest.json \
  --result "p1=Answer from morning" \
  --result "p2=" \
  --solved p1 \
  --duration "p1=42" \
  --duration "p2=30" \
  --confidence "p1=0.8" \
  --confidence "p2=0.2"
```

Retests save one result per generated session puzzle with response, solved/unsolved, duration, confidence, cue ID, blind order, and `cue_condition` (`cued` or `uncued`) from the assignment for later analysis. During administration, do not reveal the cued/uncued condition to the subject.

## Cued-vs-uncued analysis

```bash
muse-tmr analyze-cued-uncued data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --retest data/reports/night-001_retest.json \
  --dream-report data/reports/night-001_dream_report.json \
  --scheduler-events data/reports/night-001_scheduler.jsonl \
  --output data/reports/night-001_analysis.json \
  --markdown-output data/reports/night-001_analysis.md
```

The analysis report compares cued and uncued solve rates, dream incorporation rates, mean retest duration/confidence, and scheduler cue timing. It is intentionally descriptive and records limitations such as small sample size or missing cue logs.

## Cue-locked arousal proxies

```bash
muse-tmr score-cue-arousal data/recordings/pilot4_low_volume_<timestamp>
```

For every scheduler `play` of a Pilot 4 or 5 night, and for a quiet control moment paired with each, it scores the 10 s from the moment against the 20 s before it: an alpha (8 to 12 Hz), beta (15 to 30 Hz) or 55 to 95 Hz EMG-band rise of at least 2 times, with the 64 Hz optics line taken out, an arousal-proxy movement from 2 s before to 10 s after, and on optics presets a jump of 10 bpm in the optics heart-rate series. A control is a gate decision in the same gate state more than 60 s from every `play`. It reads the night's `scheduler_events.jsonl`, `audio_playback.jsonl` (the effective volume of each cue, #256), `rem_gate_decisions.jsonl`, `awakening_events.jsonl` and summary, replays `raw_amused.bin` once, and writes `cue_arousal_events.jsonl`, `cue_arousal_report.json` and `cue_arousal_report.md` into the directory, or `--output-dir`. `--summary`, `--scheduler-events`, `--audio-log`, `--gate-log`, `--awakening-events` and `--replay-input` point at other files; `--preset` says whether optics are expected, otherwise the preset in `metadata.json` does. Pilot 4 and 5 run it on their own night when it ends and put the rates into the summary as `cue_arousal`. With `--pilot3-report` it scores a Pilot 3 report's planned and injected cues on the recording that report replayed (see Pilot 3 below). The flags are arousal proxies, not scored arousals, the thresholds are UNFITTED, and the command exits 0 whatever the rates are. Definitions, the study figures beside them and every limitation code: `docs/safety_outcomes.md`; the fields: `docs/data_model.md`.

```python
import asyncio
from pathlib import Path

from muse_tmr.reports.cue_arousal import score_cue_arousal

report = asyncio.run(score_cue_arousal(Path("data/recordings/pilot4_low_volume_<timestamp>")))
report.save_outputs(Path("data/recordings/pilot4_low_volume_<timestamp>"))
print(report.outcomes["arousal_proxy_per_cue"], report.outcomes["arousal_proxy_per_control"])
```

## Meditation complexity analysis

```bash
muse-tmr meditation-plan --conditions focus,open --blocks 4 --block-minutes 8 \
  --settle-seconds 60 --seed 17 --output data/protocol/meditation/2026-10-09.json
muse-tmr analyze-meditation data/recordings/session/<name> \
  --blocks data/protocol/meditation/2026-10-09.json [--no-lyapunov] [--trim-block-start 30]
muse-tmr aggregate-meditation data/reports/meditation/*/summary.json \
  --output data/reports/meditation/aggregate.json
```

A/B meditation blocks with LZC, entropies, Hjorth, 1/f fits, Lyapunov, band-envelope DFA and an EMG check per session; p-values only across 8+ sessions. Protocol, metric definitions and limitations: `docs/meditation_metrics.md`.

```bash
muse-tmr calibration-run [--address <muse>] [--polar-address <h10>] [--language en|ru] [--voice <name>] [--rate 170] [--silent] \
  [--preset p1034|p21] [--battery-pull] [--line-check]
muse-tmr calibration-guide data/recordings/session/<name> [--recorder-pid <pid>] [--battery-pull] [--language en|ru]
muse-tmr calibration-report data/recordings/session/<name> [--output-dir data/reports/calibration/<name>]
```

A 23 min Muse + H10 recording led by voice (macOS `say`, English or Russian): slight jaw and forehead tension, clench pulses, 6/min and 12/min paced breathing and an H10 unclip. The report gives the EMG dB change per channel group with the LZC and 1/f shift, the three breathing estimates against the known pace, and the H10 reconnect. Also in the setup app as "Calibration run". `--line-check` first records 2 min each of p1034, p21, p1034, p21 to see whether the 64 Hz line comes from the optics; `--battery-pull` takes the H10 battery out in the unclip minute. Details: `docs/calibration_run.md`.

## Polar H10 companion recording

```bash
muse-tmr record --source amused --duration-hours 8 --with-polar [--polar-address <ble-address>]
muse-tmr record-polar --output-dir data/recordings/session/<name> --duration-seconds 600 \
  [--address <ble-address>] [--no-ecg] [--no-acc] [--acc-rate 50] [--acc-range 2]
muse-tmr decode-polar data/recordings/session/<name>
```

Records HR/RR, 130 Hz ECG and chest ACC from a Polar H10 into `<session>/polar/` in a separate process, so a strap problem never touches the Muse recording. Raw BLE payloads are kept and `decode-polar` rebuilds everything from them. `muse_tmr.data.polar_session.load_polar_session` maps the data onto the Muse wall-clock time base; `muse_tmr.features.cardio_resp_features` gives RR/HRV, breathing rate and RSA per window. Details and limitations: `docs/polar_h10.md`.

## Optics heart rate against the H10

```bash
muse-tmr analyze-optics-agreement data/recordings/session/<name> \
  [--epoch-seconds 30] [--output-dir data/reports/optics_agreement/<name>/]
```

Compares the heart rate the Muse takes from its optics with the H10's ECG R-peaks in a `p1034` session recorded with `--with-polar`: the decoder's `heart_rate` stream, `ppg_estimated_hr_bpm` and each optics channel per epoch, and the heart-rate series the sudden-change rule reads per 10 s window. It writes `optics_agreement.json` and `optics_agreement.md` with bias, SD, 95 % limits of agreement and MAE by quality level, the rule's false and missed jumps against the ECG change, and what a gate on `ppg_confidence` or on the series' quality would keep. Without `polar/`, without ECG or on `p21` it exits with code 2. In code, `analyze_optics_agreement(recording_dir)` is async; `optics_agreement_from_epochs(epochs, polar_session)` takes epochs you built yourself. Agreement with a chest ECG, a proxy; how to record and read it: `docs/optics_agreement.md`.

## Local app

```bash
muse-tmr app --source mock --host 127.0.0.1 --port 8765
muse-tmr app --source amused --address "$MUSE_ADDR" --host 127.0.0.1 --port 8765 [--auto-update]
muse-tmr stream --source amused --address "$MUSE_ADDR" --duration-seconds 30 --debug-stats
.venv/bin/python scripts/install_macos_launcher.py --address "$MUSE_ADDR"
```

The app is local-only. It shows live contact for `TP9`, `AF7`, `AF8` and `TP10` and records a session (`p1034`, up to 1 h), a night (`p21`, 8 h), a guided meditation, a meditation series or the calibration run. A start waits until all four channels have been good for 5 s (`--contact-stability-seconds`), and the server refuses a start before that. Each recording runs as its own `muse-tmr record` process under `caffeinate`, so closing the app or its terminal does not stop it, and its report is built from the app when it ends. `--auto-update` pulls a clean `main` checkout and restarts the app when it is idle. Live diagnostics: `/api/muse/diagnostics`. On macOS the launcher installer creates `~/Desktop/REMind.app`, which starts the app with `--auto-update`. The Recordings tab lists every recording with its reports, the EEG rate and lost subpackets, and the Morning tab takes the dream report and the blind retest of a puzzle night. Runbook: `docs/contact_setup_local_app.md`.

## Validation plan and pilots

The validation plan lives in `docs/validation.md`. Validation separates protocol fidelity and technical quality from exploratory behavioral deltas. Metrics are computed from generated session artifacts such as `summary.json`, feature rows, scheduler JSONL, dream reports, morning retests, and cued-vs-uncued analysis outputs. Reports should describe observed differences and limitations; do not claim clinical, medical, or guaranteed lucid-dreaming efficacy.

### Pilot 1: no-audio recording validation

```bash
muse-tmr validate-pilot1-recording data/recordings/<pilot1-session> \
  --output data/reports/pilot1_no_audio_validation.json
```

See `docs/pilot1_no_audio.md` for the 6h+ no-audio recording runbook. The validator checks that `summary.json` exists, duration meets the target, EEG/IMU/PPG counts are nonzero, raw packet capture is present, downtime is within target, and no audio or scheduler sidecar logs were produced.

A synthetic night from `synthesize-night` reads the same way. Half an hour of `clean_night` passes with `--min-duration-hours 0.5 --required-modality eeg --required-modality imu` (the `p21` scenarios have no PPG), and `battery_death_2h29` fails on its duration and stop reason, as a night that lost its battery should. Because its `metadata.json` says `synthetic: true`, the report, saved and printed, carries `"synthetic": true` and the note "synthetic input, not sleep", so a synthetic pass does not read as a recorded one.

### Pilot 2: audio calibration validation

```bash
muse-tmr validate-pilot2-calibration data/calibration/volume_calibration.json \
  --device-name "Sleep Headphones" \
  --playback-log data/calibration/volume_calibration_test.jsonl \
  --output data/reports/pilot2_audio_calibration_validation.json
```

See `docs/pilot2_audio_calibration.md` for the daytime calibration runbook. The validator checks that saved thresholds are ordered, scheduler max volume equals the comfortable volume, and a dry-run cap probe proves later playback code uses the calibration cap. Generated calibration files and playback logs stay local.

### Pilot 3: replay cue simulation

```bash
muse-tmr simulate-replay-cues data/recordings/<pilot-session> \
  --catalog data/protocol/puzzle_catalog.json \
  --session data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --preset p21 \
  --output data/reports/pilot3_replay_cue_plan.json \
  --scheduler-events-output data/reports/pilot3_scheduler_events.jsonl
```

See `docs/pilot3_replay_cue_simulation.md` for the replay-only runbook. The simulator runs epochs, REM detection, the stable gate, arousal guard, and scheduler on a recording with mocked audio. It writes an inspectable cue plan and fails if any uncued puzzle receives a scheduler `play` event. No real audio is played. Pass the preset the night was recorded with (`source.metadata.preset` in its `metadata.json`); without it the gate and the arousal guard assume an optics preset. Without a recording, `muse-tmr demo` runs this command on a synthetic night with protocol files it writes itself (see above). On a night whose `metadata.json` says `synthetic: true` the report carries `"synthetic": true` and the note, and the line the command prints ends with it; the `--scheduler-events-output` lines are written as for any night.

`--stride-seconds` (default 30) sets how far apart the 30 s epochs start; the gate counts the time between epoch starts, so its stable window and cooldown run in real time at any stride. The report's `metrics` carry `stride_seconds`, `first_gate_open_seconds` and `first_play_seconds`, counted from the first epoch's start to the end of the epoch whose decision first opened the gate or first planned a cue (`null` when it never happened). How they come out at strides of 30, 10 and 5 s on synthetic input: `docs/pilot3_replay_cue_simulation.md`.

`--guard-min-ppg-confidence` and `--guard-min-hr-series-quality` (both 0 to 1, default 0: off) turn the arousal guard's two HR rules off on epochs whose `ppg_confidence`, or the median quality of whose heart-rate series, is below the value; each such decision logs `hr_rule_skipped_by` (`docs/safety.md`). Set one only from an `analyze-optics-agreement` table.

`--inject-cues-every-seconds N` adds a `synthetic_cue` play event to a gate-open epoch whenever N seconds have passed since the last one, decided like a scheduler play at the end of the epoch, with mock audio and outside the scheduler: the report keeps them in `injected_cues` and its `scheduler_events` are what they are without the flag. It is there to check where `score-cue-arousal` puts its windows on a replayed night before a night with sound:

```bash
muse-tmr simulate-replay-cues <recording> --catalog ... --session ... --assignment ... --cue-library ... \
  --preset p21 --inject-cues-every-seconds 180 --output data/reports/pilot3_injected.json
muse-tmr score-cue-arousal <recording> --pilot3-report data/reports/pilot3_injected.json
```

The report's `metrics` also carry `first_epoch_start`, the first epoch's start in the recording's clock, and `injected_cue_count`. At the default 30 s stride, cues 180 s apart or more leave a gate decision between them more than 60 s from both, where a control can go.

### Pilot 4: low-volume REM-gated cueing

```bash
muse-tmr run-pilot4-cueing \
  --source amused \
  --address "$MUSE_ADDR" \
  --duration-hours 2 \
  --output-dir data/recordings/pilot4_low_volume_<timestamp> \
  --catalog data/protocol/puzzle_catalog.json \
  --session data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --calibration data/calibration/volume_calibration.json \
  --device-name "Sleep Headphones" \
  --backend system \
  --default-volume 0.02
```

See `docs/pilot4_low_volume_cueing.md` before using `--backend system`. Pilot 4 requires volume calibration, enforces the comfortable-volume cap, logs scheduler, arousal, audio, awakening, and emergency-stop artifacts, and only sends playback requests for scheduler `play` events after stable REM. With `--backend system` and no `afplay` on PATH it exits with code 2 before it reads an input, connects the headband or writes a file. `--allow-dry-run` runs the night silently instead; `metadata.json` and the summary record the requested and the resolved backend. `--preset` (default `p1034`) is the headband preset for the night and also picks the gate thresholds, as above. `--stride-seconds`, `--guard-min-ppg-confidence` and `--guard-min-hr-series-quality` work as in Pilot 3, and `pilot4_summary.json` has the same `stride_seconds`, `first_gate_open_seconds` and `first_play_seconds`. Each epoch's gate decision goes to `rem_gate_decisions.jsonl`, the summary records `first_epoch_start`, and when the night ends the cue arousal scorer runs on it (above) and adds `cue_arousal` to the summary. Mark an awakening with:

```bash
muse-tmr log-pilot4-awakening data/recordings/pilot4_low_volume_<timestamp>/awakening_events.jsonl \
  --attributed-to-cue unsure \
  --notes "woke briefly; no clear cue recall"
```

`--attributed-to-cue` (`yes`, `no`, `unsure`) records whether you think a cue woke you, and `--timestamp-seconds` places the marker at that many seconds from the night's first epoch instead of its UTC time (`--timestamp-utc`, by default now). Older markers without them still load.

### Pilot 5: full-night TLR plus puzzle cueing

```bash
muse-tmr run-pilot5-full-night \
  --source amused \
  --address "$MUSE_ADDR" \
  --duration-hours 8 \
  --output-dir data/recordings/pilot5_full_night_<timestamp> \
  --catalog data/protocol/puzzle_catalog.json \
  --session data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --tlr-block data/protocol/night-001_tlr_block.json \
  --calibration data/calibration/volume_calibration.json \
  --device-name "Sleep Headphones" \
  --backend system \
  --default-volume 0.02
```

See `docs/pilot5_full_night.md` for the full runbook. Pilot 5 requires the Pilot 4 safety setup plus a TLR block plan, writes `pilot5_summary.json`, and expects the morning dream report, blind puzzle retest, and cued-vs-uncued analysis before any behavioral interpretation. `--preset`, `--allow-dry-run` and the `--guard-min-*` gates work as in Pilot 4.
