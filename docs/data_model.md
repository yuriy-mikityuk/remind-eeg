# Unified Muse Data Model

`muse_tmr.data.sample_types` defines the common frame format used by live sources, recorders, replay, feature extraction, and future REM detection.

## Units

- `MuseFrame.timestamp`: Unix epoch seconds as a float.
- `EEGSample.channels_uv`: microvolts per channel. Values are stored as sample arrays, even when a packet contains a single sample.
- `IMUSample.accelerometer_g`: acceleration in g, keyed by `x`, `y`, and `z`.
- `IMUSample.gyroscope_dps`: angular velocity in degrees per second, keyed by `x`, `y`, and `z`.
- `PPGSample.channels`: raw optics values from the Muse protocol decoder.
- `HeartRateSample.bpm`: beats per minute.
- `BatterySample.percent`: battery percentage.
- `MuseFrame.raw_packet`: optional raw BLE packet bytes, serialized as `raw_packet_hex`.

Missing modalities are valid. A frame may contain only EEG, only IMU, only PPG, or any combination available from a packet.

## Serialization

Each sample type supports `to_dict()` and `from_dict()`. `MuseFrame` additionally supports `to_json()` and `from_json()`. Raw bytes are encoded as hex so frame JSON remains portable.

## Source Attribution

Every sample and frame carries a `source` string. The amused-py adapter uses `amused`;
the optional OpenMuse LSL adapter uses `openmuse`; the optional BrainFlow adapter uses
`brainflow`; future SDK adapters should use distinct source names.

## OpenMuse LSL Source

`muse_tmr.sources.openmuse_lsl_source.OpenMuseLslSource` is an optional adapter for
OpenMuse LSL streams. It imports neither OpenMuse nor LSL packages at module import
time; `mne_lsl` or `pylsl` is loaded only when the source is used.

The adapter maps known OpenMuse stream names into `MuseFrame` modalities:

- `Muse_EEG`: `EEGSample`
- `Muse_ACCGYRO`: `IMUSample`
- `Muse_PPG`: `PPGSample`
- `Muse_HR`: `HeartRateSample`
- `Muse_BATT`: `BatterySample`

LSL samples do not contain raw BLE packets, so `raw_packets` capability is false and
recordings from this source may have `raw_packet_count=0`.

## BrainFlow Source

`muse_tmr.sources.brainflow_source.BrainFlowSource` is an optional adapter for
BrainFlow's Muse S Athena support. It imports no BrainFlow package at module import
time; install `brainflow>=5.22.1` through the `brainflow` extra before running
`--source brainflow`.

The adapter uses `MUSE_S_ATHENA_BOARD`, `p1041`, and low latency by default. It maps
BrainFlow default, auxiliary, and ancillary presets into `MuseFrame` modalities:

- EEG rows: `EEGSample`
- BrainFlow `other_channels` from the default preset: `EEGSample` channels named `OTHER_*`
- accelerometer and gyroscope rows: `IMUSample`
- optical rows: `PPGSample` channels named `OPTICAL_*`
- battery row: `BatterySample`

BrainFlow samples do not contain raw BLE packets, so `raw_packets` capability is
false. BrainFlow does not currently supply a derived heart-rate sample through this
adapter; HR features should come from optics-derived PPG processing downstream.

BrainFlow session operations are guarded by configurable connect/start/stop timeouts
and a post-release cooldown because repeated BLE sessions can otherwise stall during
headset reconnect tests.

## Muse SDK Source Stub

`muse_tmr.sources.muse_sdk_source_stub.MuseSdkSourceStub` reserves the official SDK
source contract without importing or bundling any proprietary SDK files. It implements
`BaseMuseSource`, exposes a metadata template with `source_name="muse-sdk"`, and raises
`MuseSdkUnavailableError` from runtime methods with a pointer to `docs/sdk_policy.md`.
SDK binaries, headers, frameworks, archives, installers, copied docs, and vendor code
must remain outside committed files.

## Offline Replay

`muse_tmr.data.replay.ReplaySession` reads a recording directory or `raw_amused.bin`,
decodes raw packets through the same TAG decoder used by live streaming, and emits
`MuseFrame` objects for downstream epoch building and feature extraction.

Replay speed is explicit:

- `speed=1.0` preserves real-time packet spacing.
- `speed>1.0` accelerates replay.
- `speed=0.0` disables sleeps for tests and batch processing.

`start_seconds` and `end_seconds` are relative to the raw recording start, allowing
feature code to replay a specific sleep segment without loading unrelated packets.

## Repeated And Lost Subpackets

The first byte of every Athena subpacket header is a counter the device keeps per TAG
and wraps at 256. BLE sometimes delivers a notification twice, so `MuseRealtimeDecoder`,
which both the live amused source and replay use, drops a subpacket whose counter and
data bytes repeat one of the last 16 of its TAG. A counter that skips ahead means the
device lost subpackets. Nothing is filled in for them: the rows stay joined, and
`EEGSample.gaps`, `PPGSample.gaps` and `IMUSample.gaps` list `(index, missing)` pairs,
meaning `missing` samples belong right before row `index` of that frame's series. `gaps`
is written to `decoded_frames.jsonl` only when it is not empty. The decoder stats count
both per sensor type as `repeated_subpackets` and `lost_subpackets`, which reach
`progress.json` under `source_diagnostics.decoder`.

A frame holds every kept subpacket of its notification in the order they came: the EEG
and PPG rows of each, per channel, and three IMU rows for each ACCGYRO subpacket. Until
[#244](https://github.com/yuriy-mikityuk/remind-eeg/issues/244) the IMU rows were only
the last ACCGYRO subpacket's, while `imu_samples` counted them all: the frames of
`tests/test_data/muse_20250824_172756.bin` got 249 of its 357 IMU rows. A
`decoded_frames.jsonl` written before that fix lacks the others.

`raw_amused.bin` keeps every notification as received, so replaying an older recording
applies the same rules and gets every IMU row.

## Synthetic Recordings

`muse_tmr.testing.synthetic.generate_night()` writes a recording directory from a seed and
a `SyntheticNightConfig`, or from one of the scenarios in `muse_tmr.testing.scenarios`
(`clean_night`, `fragmented_rem`, `no_rem_nap`, `battery_death_2h29`, `optics_line`,
`bad_contact_af7`, `movement_heavy`, `short_allow_short`, each with an `hours` override).
Synthetic input, not sleep: the stage signals are assumptions, and nothing measured on
these nights is evidence of REM accuracy. The same seed and config give the same bytes.
Generated nights go to a temporary directory or the gitignored `data/synthetic/`.

The directory holds what `muse-tmr record` writes, without `decoded_frames.jsonl` and
`progress.json`:

- `raw_amused.bin` in the format above, written with `MuseRawStream.write_packet`; the
  header holds a fixed start, 2026-01-14 23:00 UTC. Byte 0 of each notification is its
  length and byte 1 a running count; a notification is at most 244 bytes. Notifications
  leave 29.03 times a second, each with the subpackets made since the one before, in the
  order they were made. The IMU is read out one, two or three ACCGYRO subpackets at a
  time, as often as notifications of the committed capture carry them, and a readout goes
  into one notification unless that would pass 244 bytes; then the rest of it goes into
  the next. Some subpackets are lost, some notifications arrive twice and short BLE gaps
  lose everything in them, as `SyntheticNightConfig` says.
- `metadata.json`: `source.source_name` is `synthetic` and `source.metadata.preset` the
  preset (`p21` or `p1034`); top level `synthetic: true`, `note`, `scenario` and `seed`.
- `events.jsonl`: `recording_started` and `recording_stopped` with the stop reason;
  timestamps are seconds from the start.
- `summary.json`: every `RecordingSummary` field, so `validate-pilot1-recording` reads it
  as it is, plus `synthetic` and `note`. `decoded_frames_path` is empty and
  `decoded_frame_count` is 0. `modality_counts` counts the `eeg`, `imu`, `ppg` and
  `battery` frames from the packet layout; `heart_rate`, which the decoder derives from
  PPG peaks during replay, is not counted. `stop_reason` is `source_ended` when the
  battery empties before the night ends.

The ground truth next to it, which no pipeline step reads:

| File | Content |
| ---- | ------- |
| `hypnogram.csv` | `epoch`, `start_s`, `end_s`, `stage` (`W`, `N1`, `N2`, `N3`, `R`), one row per 30 s epoch. |
| `arousals.csv` | `onset_s`, `duration_s`, `stage` of the epoch it starts in, `movement` (1 with an IMU burst), `position_change` (1 when the head ends up in another position). |
| `rem_events.csv` | Each eye-movement deflection: `onset_s`, `duration_s`, `epoch`, `amplitude_uv` of the AF7 lobe, `af8_ratio` (AF8 carries minus that share of it) and `sign` of the first AF7 lobe. All of them lie inside R epochs. |
| `generation.json` | `synthetic`, `note`, `seed`, `scenario`, `preset`, the whole `config`, `duration_seconds`, `stop_reason`, `hypnogram` (stage counts, REM periods), `events` (counts), `line_uv`, `battery`, `ble` and `files`, the bytes and sha256 of every file but `summary.json`, which holds absolute paths. |

`ble` is what the decoder should report: `notifications`, `repeated_notifications`,
`frames` (both), the `gaps`, `frames_with` per modality and, per sensor type, the
subpackets `produced`, `delivered`, `lost`, `lost_in_gaps` and `repeated` and the sample
`rows` delivered. The decoder's `lost_subpackets` is `lost` plus `lost_in_gaps` and its
`repeated_subpackets` is `repeated`. Times in the CSV files are seconds from the start;
replayed frames carry the Unix time of the fixed start plus that.

`muse_tmr.testing.agreement` scores gate decisions against `hypnogram.csv`: binary κ,
REM recall and precision of the gate, gate-open fraction and, per REM period, the
seconds to the first open gate, all from `muse_tmr.validation.rem_metrics`. Its output
carries `"synthetic": true` and the note. The numbers show whether the replay, detector
and gate path still behaves the same on the same input; no detector weight or threshold
is fitted to them.

`muse-tmr synthesize-night` writes this directory. `muse-tmr demo` writes it as
`<output>/<scenario>_seed<seed>/` and puts next to it:

| File | Content |
| ---- | ------- |
| `protocol/` | Pilot 3 inputs in their usual formats: `puzzle_catalog.json` (four synthetic puzzles), `night_puzzles.json`, `assignment.json` (two cued, two uncued) and `cue_library.json` (a generated tone per puzzle, the uncued ones included). |
| `pilot1.json` | The `validate-pilot1-recording` report, with the night's planned length as the minimum duration. |
| `pilot3_report.json` | The `simulate-replay-cues` report (epochs, gate and guard decisions, scheduler events, `cue_plan`). |
| `report.html` | The REM report of `muse_tmr.reports.nightly_report`; a recording whose `metadata.json` says `synthetic: true` gets a "Synthetic input, not sleep" callout and the note in its subtitle. |
| `agreement.json` | The output of `muse_tmr.testing.agreement`. |
| `demo_summary.json` | `scenario`, `seed`, `hours`, `preset`, `raw_sha256`, Pilot 1 and Pilot 3 results, `cued_plays` with the generated stage of the epoch that planned each one, `agreement` (κ, recall, precision, gate-open fraction, latency per REM period), `checks`, `baseline` when `--baseline` was given, and `seconds`. |

Every JSON file the demo writes carries the note: the reports as top-level `synthetic`
and `note`, with NaN written as `null`, the protocol files in their `metadata`.
`validate-pilot1-recording` and `simulate-replay-cues` mark their reports the same way
when the recording's `metadata.json` says `synthetic: true`.

## Polar H10 Raw Notifications

`muse-tmr record-polar` writes `<session_dir>/polar/raw_notifications.jsonl`, one JSON
object per line, appended and flushed as each BLE payload arrives:

| Field | Meaning |
| ----- | ------- |
| `seq` | Running number within the file. |
| `dir` | `rx` for notifications from the strap, `tx` for PMD control-point commands we wrote. |
| `char` | `hr` (Heart Rate Measurement 0x2A37), `pmd_cp` (PMD control point), `pmd_data` (PMD data). |
| `uuid` | Full characteristic UUID, lower case. |
| `wall` | Host `time.time()` at receipt or write, seconds. |
| `mono` | Host `time.monotonic()` at the same moment. |
| `b64` | Payload bytes, base64. |

The logged `tx` start commands (rate, resolution, range, channels) and `rx` start
responses (conversion factor) are enough to decode the PMD data frames again, so
`muse-tmr decode-polar` needs nothing but this file. A torn last line after a hard kill
is skipped. Decoded files, alignment and features are described in `docs/polar_h10.md`.

## Sleep Epochs

`muse_tmr.features.epochs.EpochBuilder` consumes any async stream of `MuseFrame`
objects, including live `AmusedSource.stream()` and offline `ReplaySession.stream()`.
The default configuration builds 30-second epochs with a 30-second stride. Strides
of 5 or 10 seconds create overlapping windows for smoother downstream REM features.
Every epoch is still 30 seconds long, so anything that keeps time per epoch, such as the
REM gate, has to move on by the time between epoch starts, not by the epoch length:
`epoch_step_seconds(start_time, previous_start_time, stride_seconds)` gives that time,
and one stride for the first epoch.

Each `SleepEpoch` contains:

- `frames`: frames inside `[start_time, end_time)`.
- `modality_counts`: number of frames carrying each modality.
- `sample_counts`: estimated sample totals per modality.
- `coverage`: observed samples divided by expected samples for EEG, IMU, PPG, and HR.
- `quality_flags`: `missing_*` or `low_*_coverage` flags instead of exceptions.

Default expected rates are EEG 256 Hz, IMU 52 Hz, PPG 64 Hz, and HR 1 Hz.

## EEG Feature Rows

`muse_tmr.features.eeg_features.extract_eeg_features()` converts one `SleepEpoch`
into an `EEGFeatureRow`. `extract_eeg_feature_rows()` handles an iterable of epochs,
and `export_eeg_feature_rows()` writes `.csv`, `.parquet`, or `.pq` through pandas.

Current EEG features include:

- absolute and relative band powers for delta, theta, alpha, beta, and gamma
- theta/alpha, delta/beta, and slow/fast ratios
- frontal and posterior alpha asymmetry
- frontal theta asymmetry
- a simple frontal eye-movement proxy from AF7-AF8 changes (deprecated: kept as a column
  so older CSVs and model artifacts load, weighted 0 in the heuristic detector, see below)
- REM events on the band-passed AF7-AF8 difference: `rem_event_count`,
  `rem_density_per_min`, `last_rem_event_age_s`, `rem_burst_active`, `rem_event_times`,
  `rem_event_threshold_uv` and `rem_event_warmup` (see below)
- artifact flags for missing EEG, low coverage, flatline, clipping, and non-finite channels

Feature rows carry epoch timing, channel counts, sample counts, coverage, and quality
flags so downstream REM detectors can filter noisy epochs before inference.

### REM events

`muse_tmr.features.rem_event_features.RemEventTracker` counts eye-movement deflections
on AF7 - AF8 (#225). Horizontal eye movements reach AF7 and AF8, both referenced near
Fpz, with opposite polarity and add up in the difference; blinks and frontal slow waves
are in phase on the two sites and mostly cancel there. The tracker is causal and keeps
its state between epochs, so a run needs one tracker for the whole session:
`extract_eeg_feature_rows()`, `HeuristicRemDetector.predict_epochs()`,
`annotate-template`, `simulate-replay-cues`, `run-pilot4-cueing`,
`run-pilot5-full-night` and `scripts/validate_rem_detection.py` each make one.
`extract_eeg_features(epoch)` on its own gets a fresh tracker, which has no threshold for
the epoch's first 7.5 s and stays in warm-up; pass `rem_tracker=new_rem_event_tracker()`
to keep one across calls.

- The difference is filtered to 0.3-5 Hz: a first-order high-pass, so a saccade decays
  without ringing, and a fourth-order low-pass, which takes the 40 µV optics line at
  about 64 Hz (#154) below 0.001 µV, so no line removal runs first.
- The threshold is `max(5 × MAD, 10 µV)`, with the MAD of the filtered difference over
  the last 5 min, refreshed every second from earlier samples only.
  `rem_event_threshold_uv` is its value at the end of the epoch and `rem_event_warmup`
  is true until the MAD covers 60 s.
- A stretch of one sign that clears the threshold starts an event; the next stretch
  joins it when it follows within 0.25 s with the other sign, so a deflection and its
  return count once and a saccade that stays out counts on its own. An event counts
  when it is 0.3-1.5 s wide at half its peak and, with the blink veto on, when
  (AF7 + AF8) / 2 carries no more power than the difference over those samples. An
  after-swing below half the event's peak does not count.
- An epoch counts the events confirmed inside it; `rem_event_times` are their peak
  times in the stream's clock. `rem_density_per_min` is per minute the tracker actually
  read, and all of these are NaN when AF7 or AF8 is missing or flagged bad, or was read
  for less than half the epoch. `rem_burst_active` means at least 2 events in the last
  10 s; `last_rem_event_age_s` counts from the last one.
- A gap of more than 0.5 s in the frame clock or in `EEGSample.gaps`, a frame without
  AF7 or AF8, or a non-finite sample restarts the filter, which then needs 2.5 s.

These are starting values in `RemEventConfig`, not fitted on any night. The feature is a
supporting term, not a REM detector of its own: blinks (wake, N1) and frontal slow waves
(N3) share the band, a slow pendular movement lasting a few seconds can still count, and
the reference near Fpz attenuates horizontal EOG.

## IMU Feature Rows

`muse_tmr.features.imu_features.extract_imu_features()` converts one `SleepEpoch`
into an `IMUFeatureRow`. `extract_imu_feature_rows()` handles an iterable of epochs,
and `export_imu_feature_rows()` writes `.csv`, `.parquet`, or `.pq` through pandas.

Current IMU features include:

- dimensionless `motion_level` from accelerometer delta and gyroscope magnitude
- `stillness_score` as the quiet sample fraction inside the epoch
- accelerometer RMS/peak delta from the epoch baseline magnitude
- gyroscope RMS/peak angular velocity
- grouped movement events with duration, peak motion, and arousal-proxy flag
- `arousal_guard_reason_codes` for downstream cue safety decisions
- cue-window movement logs when cue timestamps are supplied per epoch
- artifact flags for missing IMU, low coverage, missing axes, and non-finite values

Row `i` of a frame is taken at the frame's timestamp plus `i / sample_rate_hz` (52 Hz by
default), as the REM-event tracker does with EEG rows. Samples lost inside the frame, an
`IMUSample.gaps` entry after row 0, move the rows after them by that many samples.

IMU rows preserve noisy epochs and expose guard reasons instead of dropping data.
Future cue schedulers can turn `arousal_guard_reason_codes` into a `CueDecision`
through `muse_tmr.protocol.tmr_scheduler.arousal_guard_decision()`.

## PPG/HR/HRV Feature Rows

`muse_tmr.features.ppg_features.extract_ppg_features()` converts one `SleepEpoch`
into a `PPGFeatureRow`. `extract_ppg_feature_rows()` handles an iterable of epochs,
and `export_ppg_feature_rows()` writes `.csv`, `.parquet`, or `.pq` through pandas.

Current PPG/HR features include:

- PPG-derived heart-rate estimate from the strongest optics channel when raw PPG is present
- mean, median, min, max, and trend for `HeartRateSample.bpm`
- HRV proxy metrics: mean RR, SDNN, RMSSD, and pNN50
- sudden HR-change logs from the epoch's own heart-rate series (below), or from adjacent
  heart-rate samples when the epoch has no optics
- source fields showing whether HR/HRV came from PPG peaks, heart-rate samples, or missing data
- artifact flags for missing PPG, low coverage, flatline, non-finite values, and out-of-range HR

Raw PPG and heart-rate samples are optional. Missing modalities must produce flags and
`NaN` feature values where needed, not exceptions.

### Optics channels and the heart-rate estimates

On `p1034` the decoder names the eight optics channels after `OPTICS_CHANNELS_8` in
`muse_athena_protocol.py` (other channel counts become `opt0`, `opt1`, ...). Side, depth
and wavelength below are what the comments there say; no vendor document in the
repository confirms them, so all of it is vendor-unverified.

| Channel | Side | Depth | Wavelength (comment) |
|---|---|---|---|
| `LO_NIR` | left | outer | about 850 nm |
| `RO_NIR` | right | outer | about 850 nm |
| `LO_IR` | left | outer | about 735 nm |
| `RO_IR` | right | outer | about 735 nm |
| `LI_NIR` | left | inner | about 850 nm |
| `RI_NIR` | right | inner | about 850 nm |
| `LI_IR` | left | inner | about 735 nm |
| `RI_IR` | right | inner | about 735 nm |

The decoder's `HeartRateSample` stream comes from the first channel only (`LO_NIR` on
`p1034`): on every optics notification it takes the last 2 to 5 s of it (128 to 320
samples), finds peaks at least 40 samples apart, so never above 96 bpm, and keeps 60 over
their mean interval when that lies between 40 and 200 bpm. `mean_hr_bpm` and the other
summary fields, the HR trend and `heart_rate_out_of_range` come from that stream.

`ppg_estimated_hr_bpm` is a second estimate: peak intervals on the channel with the
largest variance over the epoch (`primary_ppg_channel`), band-passed to 0.5 to 4 Hz.
`ppg_confidence` is 1 minus the SD of its beat intervals over their mean, clamped to 0..1,
so it falls when the heart rate changes inside the epoch, and band-passed noise reaches
0.6 to 0.8. `ppg_signal_quality` names it: `Excellent` above 0.8, `Good` above 0.6, `Fair`
above 0.4, else `Poor`; `Irregular rhythm`, `Too few peaks detected`, `Flatline`,
`Insufficient data` or `Missing` when there is no estimate
(`muse_tmr.features.ppg_features.PPG_SIGNAL_QUALITIES`).
`estimate_channel_heart_rates(epoch)` gives the same estimate for every channel as
`ChannelHeartRate` records (`channel`, `estimated_hr_bpm`, `confidence`, `peak_count`,
`signal_quality`) plus `periodicity`, the highest autocorrelation peak of the channel's
band-passed epoch in the heart-rate range, by which the series below picks its channel
(0 for a flat channel or one shorter than a series window). The primary channel's entry
carries the row's own values. `PPGFeatureRow.hr_series_median_quality` is the median
quality of the series values, skipped ones included, the `hr_series_median_quality`
column.

### Heart-rate series for the sudden-change rule

The arousal guard's sudden heart-rate change rule reads a series that
`heart_rate_series(epoch)` builds from the optics the epoch holds, not the decoder's
`heart_rate` stream ([#247](https://github.com/yuriy-mikityuk/remind-eeg/issues/247)).
That stream is re-estimated from the last 5 s of `LO_NIR` on every optics notification, so
consecutive values share almost all their samples and jump when one beat enters or
leaves the window. `mean_hr_bpm` and the other summary fields still come from it.

Every optics channel goes on one sample clock at `ppg_sample_rate_hz` (64 Hz) from the
first optics frame's timestamp, one row after the other. Samples reported lost in
`PPGSample.gaps`, and the time a frame arrives more than 1 s after the clock says it
should, stay missing and are bridged linearly. Each channel is detrended and
band-passed to `ppg_low_hz`..`ppg_high_hz` (0.5 to 4 Hz) without phase shift. The series
comes from the channel whose band-passed epoch is the most periodic, the highest
autocorrelation peak between `min_valid_hr_bpm` and `max_valid_hr_bpm`, the first by name
on a tie. Each window of `hr_series_window_seconds` (10 s), stepped by
`hr_series_step_seconds` (1 s), gives one `HeartRatePoint`:

| Field | Meaning |
|---|---|
| `time` | Centre of the window, Unix seconds. |
| `bpm` | The beat period the window correlates best with itself at (Pearson correlation over the overlap at each lag; the shortest lag whose peak reaches 0.9 of the highest, so one beat wins over two), as bpm. NaN when there is no positive peak. |
| `peak_bpm` | 60 over the median interval between the window's light maxima (the pulse foot, prominence at least `hr_series_peak_prominence` 0.5 of the window's SD), the check on `bpm`. |
| `quality` | 0 to 1: the autocorrelation peak when `bpm` and `peak_bpm` agree within `hr_series_max_disagreement_bpm` (5 bpm), else 0; also 0 when more than `hr_series_max_missing_fraction` (0.2) of the window is missing. |
| `used` | `quality` reaches `hr_series_min_quality` (0.5). Skipped values stay in the series. |
| `run_seconds` | 0, or the length of the run of regular beats that gave `bpm`, `peak_bpm` and `quality` (below). |

A window that holds two rates, where a step or a short surge falls inside it, peaks at
neither beat period: its autocorrelation often settles near two fast beats while
`peak_bpm` reads the fast rate. When a window's quality stays under `hr_series_min_quality`
and it is not missing data, its longest run of at least 5 beat intervals, each within
15 % of the window's median interval, stands in for it if the run spans
`hr_series_min_run_seconds` (4 s) or more and the autocorrelation of that stretch alone
agrees with the median of its intervals. A short surge then reads its own rate, while
noise rarely holds such a run.

The rule compares used values `hr_series_compare_seconds` to
`hr_series_max_compare_seconds` (10 to 20 s) apart. Windows that far apart share no
samples, so a step held between them shows in full, while a missed or extra beat moves
neither estimate. Each change of `sudden_hr_change_bpm` (10) less `hr_series_margin_bpm`
(1), so 9 bpm or more, is logged once per direction by its largest pair, in
`sudden_hr_changes`, `sudden_hr_change_count` and `max_sudden_hr_change_bpm` as before.
The margin covers the error of a value against the heart rate of its window, so a change
of 10 bpm is not lost to it. A window stays inside its epoch, so a step in the first or
last 10 s of an epoch shows only when the arousal guard compares the previous epoch's
values with this one's (below), and a surge shorter than about 8 s that straddles an
epoch boundary, with under 4 s of it in either epoch, often shows in neither. An epoch
with heart-rate samples but no optics compares consecutive samples, as every epoch did
before.

The rows add `sudden_hr_change_source` (`optics`, `heart_rate` or `missing`: what fed the
rule), `hr_series_channel`, `hr_series_used_count`, `hr_series_skipped_count`,
`hr_series_median_quality` and `hr_series_json` (every point, skipped ones included). The
rule's own column names stay, so earlier CSVs and models load.

## REM Predictions

`muse_tmr.models.heuristic_rem_detector.HeuristicRemDetector` is the first non-ML
REM baseline. It can consume a `SleepEpoch` directly or precomputed EEG, IMU, and
PPG feature rows.

Each `RemPrediction` contains:

- `probability`: `P_REM` clamped to 0-1
- `reason_codes`: support, missing-feature, low-coverage, and guard-related explanations
- `feature_scores`: normalized heuristic component scores
- `feature_values`: raw feature values used by the heuristic
- `source`: currently `heuristic`

Before the caps, `P_REM` is the weighted mean of the component scores that are
available, with `DEFAULT_REM_WEIGHTS`: `low_delta_power` 0.333, `theta_alpha_ratio`
0.200, `stillness` 0.267, `hr_variability` 0.133, `hr_trend` 0.067 and
`eye_movement_proxy` 0. The proxy is still scored and stays in `feature_scores`,
`feature_values` and the annotation columns, but it is 1.0 in every epoch of a real night
(AUC 0.500 against GSSC REM, #116), so it no longer moves `P_REM`. The other five are the
earlier weights divided by 0.75, so they still sum to 1 and a prediction with less than
0.40 of the weight available still gets `limited_feature_support` and the 0.55 cap; an
EEG and IMU epoch on `p21` has 0.80. The weights are set by hand; the refit waits for
REM-bearing `p21` nights (#225). For that refit `scripts/validate_rem_detection.py`
reports, against its reference stager, the ROC-AUC, minimum and maximum of every
`feature_scores` and `feature_values` entry, with `saturated` when an entry has one
value in every epoch, and replays the night with other weights given as
`--weights name=w,...` (see `docs/USAGE.md`).

`rem_event_density`, the proxy's replacement, is scored from `rem_density_per_min`
between `rem_event_density_min` 1 and `rem_event_density_strong` 6 events per minute,
with the reason codes `eeg_rem_events_support` and `eeg_rem_events_low`, and exported as
`feature_values` `rem_event_density`, `rem_event_count`, `rem_event_threshold_uv` and
`rem_event_warmup` (1 or 0). Its weight in `DEFAULT_REM_WEIGHTS` is 0 and both thresholds
are UNFITTED starting values, so it does not move `P_REM` until it has been fitted on
real nights; it is never fitted on synthetic data. An epoch whose REM-event fields are
NaN gets no score and no reason code for it.

When the HRV comes from PPG peaks (`hrv_source` `ppg`) and `ppg_confidence` is below
`HeuristicRemConfig.low_ppg_quality_confidence` (0.4), the prediction adds the reason code
`low_ppg_quality`. The HRV score, every other score and `P_REM` stay as they are, and the
gate does not read the code (#234).

The REM detector does not play audio and does not return `CueDecision`. Stable gates,
cooldowns, arousal guards, and cue scheduling remain separate downstream layers.

## REM Annotation Rows

`muse_tmr.annotations.rem_annotations` defines manual annotation rows for later REM
classifier training. Valid labels are:

- `wake`
- `nrem`
- `probable_rem`
- `unknown`

`muse-tmr annotate-template <recording> --output <labels.csv|labels.json>` replays a
recording, builds epochs, runs the heuristic REM detector, and exports an editable
annotation template. Each row includes epoch timing, `label`, `notes`, `P_REM`, reason
codes, normalized feature scores, and raw feature values. The default label is
`unknown` so the file can be created before manual review.

`load_rem_annotations()` validates labels from CSV or JSON. `rem_training_rows()`
returns training-ready rows and excludes `unknown` labels by default.

## Personal REM Models

`muse_tmr.models.ml_rem_detector` trains a personal REM classifier from labeled
annotation rows. The default model is a small class-balanced logistic classifier over
the annotation feature columns, so it stays dependency-light and can be trained from
manually labeled nights.

`muse-tmr train-rem-classifier <labels.csv|labels.json> --output <model.json>` writes a
versioned JSON artifact with:

- feature names, means, scales, coefficients, and calibration intercept
- feature importance based on normalized coefficient magnitude
- training metrics, Brier/log-loss calibration metrics, and calibration bins
- optional leave-one-recording-out metrics when multiple recording IDs are present

`probable_rem` is the positive class. `wake` and `nrem` are negative classes.
`unknown` rows are skipped. A loaded `PersonalRemModel` returns `RemPrediction` with
`source="personal"` and remains separate from cue playback decisions.

## REM Stable Gate

`muse_tmr.models.rem_gate.StableRemGate` converts a stream of `RemPrediction` values
into a stateful REM gate decision. The gate keeps REM detection separate from cue
playback: it only emits `RemGateDecision`, never `CueDecision` or audio calls.

Each decision contains:

- `gate_open`: true only after stable REM confidence exceeds the entry threshold for
  the configured stability window
- `state`: `closed`, `warming`, `open`, or `blocked`
- `confidence`: the original `P_REM`, capped confidence, active threshold, source, and
  prediction reason codes
- `stable_seconds`: consecutive stable REM time tracked by the gate
- `cooldown_remaining_seconds`: remaining cooldown after motion/arousal blocks
- `reason_codes`: threshold, stability, hysteresis, cooldown, and block reasons

`update(prediction, duration_seconds=...)` adds `duration_seconds` to the stable time
and takes it off the cooldown. Pilot 3, 4 and 5 pass the time between epoch starts
(`epoch_step_seconds`), so both run in real time at any `--stride-seconds`: at a 10 s
stride the gate opens on the sixth REM-like epoch in a row instead of the second, and a
cooldown lasts twelve epochs instead of four. At the default 30 s stride the step is the
epoch length, as before. Before #225 the pilots passed the 30 s epoch length at every
stride, so at 10 s the window and the cooldown ran three times too fast. What this means
for the first cue after REM onset: `docs/pilot3_replay_cue_simulation.md`.

`RemGateConfig` defaults to an entry threshold of `0.80`, exit threshold of `0.70`, a
60-second stability window, and a 120-second cooldown. The pilot commands and
`scripts/validate_rem_detection.py` take their pair from the recording preset instead
(`muse_tmr.presets.preset_gate_thresholds`): `p21` uses `0.90`/`0.85`, any optics preset
`0.80`/`0.70`. The heuristic detector runs hot, so the earlier `0.70`/`0.45` left the
gate open about 75% of a night. The `p21` pair was tuned in-sample on one EEG-only night
(`overnight_20260707`, n=1) against GSSC, an automated proxy, not PSG; it was fitted and
reported on the same night, so it is not held-out validation. It was also fitted with the
detector's earlier weights, before the eye-movement proxy went to 0; re-grid pending
(#116). Kappa and epoch counts are only in PR #115 / #116 (see [`numbers.md`](numbers.md)).
The optics pair is provisional until a night with REM on an optics preset is recorded
(#116).

Motion/arousal reason codes close the gate and start cooldown. Low feature-support
reason codes cap confidence so single high-probability spikes do not open the gate by
themselves. On an EEG-only preset (`p21`) missing PPG is expected, so the gate is built
with `optics_capable=False` and missing-cardiac reason codes do not lower confidence.

### Gate log and REM periods

Pilot 4 and 5 append each epoch's decision to `rem_gate_decisions.jsonl` in the output
directory (#227); the scheduler drops a decision once it has used it. A Pilot 3 report
keeps the same decisions in `epochs[].gate_decision`. `muse_tmr.validation.gate_log` reads
both: `load_rem_gate_decisions(source)` takes an output directory, the JSONL file, a
Pilot 3 report or its dict and returns `GateDecisionRecord`s, `write_rem_gate_decisions()`
writes or appends them. One line:

| Field | Content |
|---|---|
| `schema_version` | 1. |
| `index` | The epoch index. |
| `timestamp_seconds`, `duration_seconds` | The epoch's start, seconds from the first epoch's start (the scheduler's time base), and its length. |
| `decision_seconds` | Their sum: when the decision exists. |
| `gate_open`, `state` | As in `RemGateDecision`. |
| `p_rem`, `confidence` | The prediction's probability and the gate's capped confidence. |
| `stable_seconds`, `cooldown_remaining_seconds`, `reason_codes` | As in `RemGateDecision`. |

`gate_open_runs(records)` and `load_gate_open_runs(source)` give the gate-open runs, the
REM periods of the cue arousal scorer and of the protocol fidelity report: consecutive
epochs whose decision left the gate open. A `GateOpenRun` starts at the first one's
`decision_seconds` and ends at the `decision_seconds` of the first decision after them
that closed the gate (`closed` true), or at the last open one's when the log ends open
(`closed` false). It is the gate's REM, not a scored REM period. Both reports place a
scheduler `play` with `emitting_record()`, the first epoch starting at or after its
`timestamp_seconds` (a puzzle cue is logged at its epoch's start, a TLR cue at its
planned time), and find its run with `run_at()`.

## Audio Playback

`muse_tmr.audio.audio_player.AudioCuePlayer` is the low-volume playback facade for
sleep cues. It does not inspect REM probabilities or make cue eligibility decisions;
M4/M5 protocol code must call it only after stable gate and safety layers allow a cue.

`muse-tmr play-test-cue` plays or dry-runs a generated test tone. The playback request
captures:

- cue ID, frequency, and duration
- `requested_volume`, the request at the max-volume cap (`capped_volume`), the
  `volume_multiplier` applied after the cap (1.0 unless the arousal guard lowered the
  volume), the `effective_volume` the backend plays (capped times multiplier), the
  `max_volume` cap, and `volume_capped` when the cap lowered the request
- fade-in/fade-out seconds
- optional device name metadata
- backend name and reason codes, `volume_lowered` when the multiplier is below 1

`CuePlaybackResult` records `played`, `skipped`, `blocked`, or `stopped` outcomes and
can be appended to a JSONL log. On a sham night (#228) each line adds `arm: "sham"`; the
cue arm, the default, is left out, so a cue night's log reads as before. `AudioPlaybackConfig.cue_volume()` computes the four
volumes for the player and for the scheduler. `capped_volume` and `volume_multiplier`
are in the log since #255; the other fields keep their values on a cue the guard did not
lower. Before, a lowered cue was logged with the halved request and no multiplier.
`create_audio_backend("system")` uses macOS `afplay`
when it is on PATH and otherwise returns `dry-run`; tests use `MockAudioBackend` so CI
does not need an audio device.

The commands that can play a cue (`play-test-cue`, `train-tlr-cue`, `run-pilot4-cueing`,
`run-pilot5-full-night`) go through `resolve_audio_backend(name, *,
allow_dry_run_fallback=False)` instead. It returns a `BackendResolution` with
`requested_name`, `resolved_name` and `backend`, and raises `AudioBackendFallbackError`
when `system` or `afplay` is asked for and `afplay` is not on PATH, because every cue
would then be silent. With `allow_dry_run_fallback=True` (`--allow-dry-run`) it returns
`dry-run` instead. Pilot 4 and 5 resolve right after validating
`Pilot4CueingConfig`, before `connect()` and before any file is written, and take the
flag as `Pilot4CueingConfig.allow_dry_run_fallback`. Their `metadata.json` `audio`
block records `requested_backend_name` and `resolved_backend_name` next to
`backend_name`, which stays the requested name. The summary records
`requested_audio_backend_name` and `resolved_audio_backend_name` next to
`audio_backend_name`, which stays the resolved one.

## Volume Calibration

`muse_tmr.audio.volume_calibration.VolumeCalibration` stores the pre-sleep audible
thresholds for one playback device:

- `detectable_volume`: lowest volume the subject can notice
- `identifiable_volume`: lowest volume the subject can recognize as the intended cue
- `comfortable_volume`: highest volume accepted for sleep-time scheduling
- cue/backend metadata, timestamp, and optional notes

`VolumeCalibrationStore` writes versioned JSON metadata with a `calibrations` array.
Saving a new calibration for the same `device_name` replaces the prior record for that
device while preserving other devices. `muse-tmr calibrate-volume --output <file>`
creates or updates this metadata.

Scheduler-facing code should use `comfortable_volume` through
`calibrated_max_volume()` or `calibrated_cue_decision()`. The resulting max remains
bounded by the configured hard cap, and missing calibration should block planned
sleep-time cue sessions rather than falling back to an unverified speaker level.

## Cue Libraries

`muse_tmr.audio.cue_library.CueLibrary` stores cue metadata in JSON without committing
private audio files. A cue has:

- `cue_id`: stable identifier used by protocol and scheduler layers
- `cue_type`: `sound`, `generated_tone`, or `silence`
- `protocol`: `puzzle`, `tlr`, `test`, `generic`, or `sham` (an untrained sound a sham
  night plays in place of the trained cues, #228)
- `duration_seconds`, tags, optional description, and optional volume hint
- `path` for private sound files or `frequency_hz` for generated tones

`muse-tmr create-cue-library --output <catalog.json>` creates a starter metadata
catalog with puzzle, TLR, silence-control and sham (`sham_soft_tone`, a 330 Hz tone
tagged `untrained`) cues; with `--protocol-preset NAME` its `tlr_soft_tone` is the cue
`create-tlr-cue --protocol-preset NAME` makes, with the preset block under
`metadata.protocol` (#264). `validate-cue-library` checks
metadata and, by default, verifies that `sound` cue paths exist relative to the catalog
directory before a session starts. `list-cues` filters by protocol or tag.

Private cue files should live under gitignored folders such as `cues/private/`,
`data/cues/private/`, or `data/cues/audio/`. Commit cue metadata only when it does not
include private labels or personal audio paths.

## Puzzle Protocol

`muse_tmr.protocol.puzzle_protocol` defines versioned JSON records for the pre-sleep
puzzle workflow:

- `PuzzleCatalog`: puzzle tasks plus timed pre-sleep attempts
- `PuzzleTask`: puzzle ID, prompt, solution, cue ID, source/tags, and solved/known/retired flags
- `PuzzleAttempt`: response, duration, solved flag, known-after flag, timestamps, and notes
- `NightPuzzleSession`: generated session ID, selected puzzle IDs, seed, and association results
- `AssociationResult`: cue ID, response, expected solution, normalized match flag, timestamp, and notes

`muse-tmr import-puzzles` accepts CSV or JSON rows and writes a catalog. `muse-tmr
generate-puzzle-session` filters solved, known, and retired tasks and selects four
eligible unsolved puzzles by default. `record-puzzle-attempt` appends timed pre-sleep
attempts to the catalog, and `record-association-check` appends cue-to-solution checks
to the generated night session.

## Puzzle Cue Assignments

`muse_tmr.protocol.randomization.PuzzleCueAssignment` stores reproducible cued vs
uncued assignment for one `NightPuzzleSession`:

- `session_id` and randomization `seed`
- `cued_puzzle_ids`
- `uncued_puzzle_ids`
- derived `scheduled_puzzle_ids`, which contains only cued puzzle IDs
- generation timestamp and metadata

`muse-tmr assign-puzzle-cues <session.json> --seed <n> --output <assignment.json>`
splits half of the session puzzles into the cued group by default, or as many as the
protocol preset says (`--protocol-preset`); its `metadata.protocol` is the preset block,
and so is a session's from `generate-puzzle-session`. The assignment must
cover exactly the session puzzle IDs with no overlap. Scheduler code should call
`scheduled_puzzle_ids` or `scheduled_cue_ids(catalog)` so uncued controls are never
eligible for cue playback.

## TLR Protocol

`muse_tmr.protocol.tlr_protocol` defines versioned records for targeted lucidity
reactivation:

- `TlrCueConfig`: default generated cue settings
- `TlrTrainingConfig`: pre-sleep repetition count, interval, volume, and backend name
- `TlrTrainingSession`: training summary with one event per cue presentation and, since
  schema 2, the `tone` it played
- `TlrTrainingEvent`: playback status, requested/effective volume, reason codes, and scheduled offset
- `TlrBlockConfig`: REM block repetitions, interval, post-block pause, and enabled flag
- `TlrBlockPlan`: cue offsets, `puzzle_cue_start_offset_seconds` and `metadata`, which
  holds the protocol preset block the plan was made with (`metadata.protocol`, #231)
  and, with `plan-tlr-block --training`, the training it was checked against
  (`metadata.training`: `session_id`, `completed_at_utc`); since schema 2, the `tone` it
  was made with
- `TlrTone` (#264): `cue_id`, `frequency_hz` (`null` for a TLR cue that is not a
  generated tone), `duration_seconds` and `library_id`, the cue library the cue came from

Plans and training summaries of schema 1 have no `tone` and still load; written again,
an old plan stays as it was.

`muse-tmr create-tlr-cue` writes a `CueLibrary` containing a generated TLR cue.
`train-tlr-cue` plays or dry-runs the cue through `AudioCuePlayer` and writes both a
summary JSON and JSONL training events. It plays `--volume`, else the cue's
`volume_hint`, else the player's default, as Pilot 4 and 5 choose a cue's volume; a hint
of 0 counts as a hint since #255. `plan-tlr-block` writes the TLR block that
future scheduler code should insert before puzzle cues when the REM gate opens.

A night plays the cue its own cue library has under each block event's cue ID, for that
cue's length. `tlr_tone_mismatch(plan, cue_library)` says why that would not be the
plan's tone, checking a schema 2 plan's frequency and length and a schema 1 plan's
event lengths; `run_pilot4_cueing_night()` raises with it, and `run-pilot5-full-night`
exits with code 2 with it, before any file is written. `training_tone_mismatch()` is the
same check of a training summary against a cue, for `plan-tlr-block --training`.

## TMR Scheduler Events

`muse_tmr.protocol.tmr_scheduler.TmrCueScheduler` emits `TmrSchedulerEvent` records:

- `event_type`: `play`, `skip`, `pause`, or `stop`
- `timestamp_seconds`: replay/session-relative time
- optional `cue_id`, `protocol`, and `puzzle_id`
- `reason_codes`
- metadata such as duration, volume hint, next allowed cue time, or cooldown end time
- `arm`: the night's arm, written only when it is not `cue` (#228), so a cue night's
  events read as before and an event without it is a cue night's

On a sham night a `play` names the sham cue as `cue_id`, keeps the `puzzle_id`,
`protocol` and metadata of the cue it stands in for and adds the reason `sham_arm`. On a
no-cue night there is no `play`: in its place a `skip` with the same `cue_id`,
`protocol`, `puzzle_id` and metadata, the added reason `no_cue_arm` and
`metadata.virtual_play: true` (`TmrSchedulerEvent.virtual_play`).

The scheduler consumes `RemGateDecision`, `PuzzleCueAssignment`, `PuzzleCatalog`,
`CueLibrary`, and optional `TlrBlockPlan`. It uses only `scheduled_puzzle_ids` from
the assignment, so uncued puzzle controls are not eligible for playback. Events can be
appended to JSONL with `append_tmr_scheduler_events()` and read with
`load_tmr_scheduler_events()`.

A play event's `volume_hint` is the cue's own. Under a `lower_volume` guard decision the
event adds `arousal_guard_action`, `arousal_guard_reason_codes` and the guard's
`volume_multiplier`. Given the player's config (`playback_config`, as Pilot 4 and 5 pass
it), every play event also logs `requested_volume`, `capped_volume`, `volume_multiplier`
and `effective_volume`, the values its line in `audio_playback.jsonl` has. Pilot 3 has no
player, so its plan carries only the hint and the multiplier. Before #255 a lowered play
event carried the halved hint as `volume_hint` and the cue's own as
`original_volume_hint`.

## Arousal Guard Decisions

`muse_tmr.protocol.arousal_guard.ArousalGuard` emits `ArousalGuardDecision` records:

- `action`: `allow`, `lower_volume`, `pause`, or `stop`
- `timestamp_seconds`: replay/session-relative time
- `reason_codes`: motion, alpha, HR-jump, or artifact-quality explanations
- `volume_multiplier`: what a `lower_volume` decision multiplies the cue volume by, after
  the player's cap
- `pause_seconds`: requested pause duration for pause decisions
- `metadata`: feature values, artifact flags, and consecutive guard counters;
  `ppg_absent_by_design: true` when the guard runs with `optics_capable=False`

`ArousalGuardConfig.optics_capable` (default `true`) says whether the session preset
streams optics at all. The CLI sets it from `preset_provides_optics(preset)`, the same
call that sets `RemGateConfig.optics_capable`, and the Pilot 3 and Pilot 4 runners copy
the gate's value when no guard config is passed. When it is `false` the PPG-absence
flags `ppg_missing`, `missing_ppg`, `heart_rate_missing`, `missing_heart_rate`,
`low_heart_rate_coverage` and `low_ppg_coverage` are dropped before the PPG checks,
`sudden_heart_rate_change` and `heart_rate_out_of_range` are not raised, and every
decision, `allow` included, carries `metadata.ppg_absent_by_design`. On optics presets
an `allow` decision keeps an empty `metadata` unless the epoch's heart-rate series
skipped values; then it carries `hr_series_used_count` and `hr_series_skipped_count`,
since the pilots keep the decisions but not the PPG rows, and `hr_rule_blind` or a quality
gate's `hr_rule_skipped_by` with the quality fields when they apply (below).

On optics presets the HR rule pauses when it finds at least
`sudden_hr_change_pause_count` (1) heart-rate changes of `sudden_hr_change_pause_bpm`
(10) less `sudden_hr_change_margin_bpm` (1) or more: the row's `sudden_hr_changes` at or
above that, and the largest change from the previous row's used series values into this
row's, for values at most `sudden_hr_change_link_seconds` (20 s) apart (0 leaves that
comparison out). That change across the boundary is not counted again when a counted row
change in the same direction overlaps it in time: it is the same step, seen from both
sides. The guard keeps the previous row's series for it and `reset()` clears it. Before
#247 any logged change paused through the count, so raising the bpm alone changed
nothing. The row logs changes from `PPGFeatureConfig.sudden_hr_change_bpm` less its
`hr_series_margin_bpm` up, so a lower pause bpm only affects the comparison across
epochs. A row built with the counts but without the changes counts all of them when the
largest reaches the bpm. Non-`allow` decisions on optics presets add
`sudden_hr_change_source`, `hr_series_used_count`, `hr_series_skipped_count`,
`ppg_confidence`, `ppg_signal_quality`, `hr_series_median_quality` (when the row has series
values), `sudden_hr_change_pause_steps` (the changes counted) and, when there was a
previous series to compare with, `sudden_hr_change_across_epochs_bpm`.

Three quality gates can turn the two HR rules off for an epoch on optics presets (#234):
`min_ppg_confidence_for_hr_rule` (0.0), `ppg_quality_for_hr_rule` (the
`ppg_signal_quality` levels the rules run on; empty, all of them) and
`min_hr_series_quality_for_hr_rule` (0.0; a row without series values fails it). A row
that fails one raises neither `sudden_heart_rate_change` nor `heart_rate_out_of_range`,
its series does not feed the next row's comparison across the boundary, and the decision,
`allow` included, logs `hr_rule_skipped_by` (the config fields it failed) with
`ppg_confidence`, `ppg_signal_quality` and `hr_series_median_quality`. Its PPG artifact
flags still count. The defaults gate nothing; `docs/optics_agreement.md` has the tables
to set one from.

An optics row whose series skipped more than `hr_rule_blind_skipped_share` (0.5) of its
values is `hr_rule_blind`: the HR rule could not see that epoch, which no artifact flag
covers. Every decision then logs `hr_rule_blind: true`, `allow` included.
`hr_rule_blind_action` says what else happens: `lower_volume` (the default, the
maintainer's choice on #234) adds the reason code `hr_rule_blind` as a lower-volume
reason, so the decision is `lower_volume` with `lower_volume_multiplier` unless a pause or
stop reason in the same epoch wins; `pause` adds it as a pause reason; `allow` adds
nothing. It is never an artifact reason, so blind epochs in a row do not lead to a
`repeated_artifact_quality` pause. A row without series values is not counted; its
coverage and flatline flags act.

Decisions can be appended to JSONL with `append_arousal_guard_decisions()` and read
with `load_arousal_guard_decisions()`. The scheduler can consume a decision directly;
`lower_volume` puts its multiplier on the play events, which the player applies after
its cap, `pause` records a pause and cooldown, and `stop` records a stop event.

## Dream Reports

`muse_tmr.reports.dream_report` defines versioned morning dream report records:

- `DreamReport`: session ID, lucid yes/no, cues-heard yes/no, confidence, free-text
  dream recall, notes, metadata, and puzzle incorporation links
- `DreamPuzzleIncorporation`: puzzle ID, optional cue ID, dream-content excerpt, and
  link confidence

`build_dream_report()` validates links against `NightPuzzleSession` and can enrich links
with cue IDs from `PuzzleCatalog`. `muse-tmr record-dream-report` writes the report JSON.

Schema 2 (#228) adds `woke_from_sound` (`yes`, `no`, `unknown`), `cue_perception`
(`not_perceived`, `in_dream`, `awake`, `unsure` for heard but not sure whether dreaming,
`unknown`) and `eye_signal_attempt_times` (clock times `HH:MM[:SS]` or ISO times, as the
sleeper marks them; nothing detects an eye signal). `cues_heard` stays and equals
`cue_perception` in `in_dream`, `awake` or `unsure`; the two must agree unless the
perception is `unknown`. A schema 1 file loads as it was: `cue_perception` from
`cues_heard` (true: `unsure`, false: `not_perceived`), `woke_from_sound` `unknown`, no
attempt times, and `schema_version` 1.
Dream text and puzzle-link excerpts are private morning data and should stay in
gitignored report locations.

## Morning Retests

`muse_tmr.reports.morning_retest` defines versioned morning puzzle retest records:

- `MorningRetest`: session ID, result counts, solved/unsolved counts, mean duration,
  notes, metadata, and per-puzzle results
- `MorningRetestResult`: puzzle ID, cue ID, blind order, `cue_condition`, response,
  solved flag, duration, confidence, and notes

`build_morning_retest()` validates results against `NightPuzzleSession`, enriches cue
IDs from `PuzzleCatalog`, and stores cued/uncued condition from `PuzzleCueAssignment`
for analysis. Schema 2 (#228) adds `retest_wave`: `morning`, `day7` or `day10`, with
`retest_id` `<session>-<wave>-retest`; the app saves a delayed wave as
`<night>_retest_<wave>.json`. A schema 1 file loads as a `morning` retest. `muse-tmr record-puzzle-retest` writes the retest JSON. The retest should
be administered blind; `cue_condition` is for downstream analysis, not subject-facing
prompts.

## Cued-vs-Uncued Analysis

`muse_tmr.reports.analysis` defines versioned analysis records:

- `CuedUncuedAnalysisReport`: session ID, per-puzzle rows, condition metrics, effect
  summary, cue timing summary, limitations, and metadata
- `PuzzleAnalysisRow`: puzzle ID, cue condition, retest solved/duration/confidence,
  dream incorporation flag, and scheduler puzzle cue timing
- `ConditionMetrics`: cued or uncued group counts, solve rate, incorporation rate,
  mean duration/confidence, and cue play counts

`build_cued_uncued_analysis()` validates `NightPuzzleSession`, `PuzzleCueAssignment`,
`MorningRetest`, and optional `DreamReport`; scheduler events are optional but used to
audit puzzle cue timing. The output is descriptive only and includes machine-readable
limitations such as `small_n`, `missing_scheduler_events`, or `uncued_cue_play_observed`.

Schema 2 (#231) adds `cue_recall`, the morning sound check by condition: `available`,
`trial_count`, `answered_count`, `correct_count`, `recall_rate` (a sound not placed counts
as wrong), `by_condition` (`trial_count`, `correct_count`, `recall_rate` for `cued` and
`uncued`), `recall_rate_difference_cued_minus_uncued`, `completed_at_utc`,
`retest_reported_at_utc` and `before_retest`; each row's `cue_recalled`; the limitations
`cue_recall_missing` and `cue_recall_after_retest`; and in `metadata` the night's
`protocol` block with `protocol_source` (the `--protocol-metadata` file's name, or
`assignment`).

Schema 3 (#228) reads one retest per wave. `rows`, `condition_metrics` and
`effect_summary` are those of the `morning` wave, or of the earliest wave given without
one (`metadata.primary_retest_wave`, with `metadata.retest_waves`). New fields:

| Field | Content |
|---|---|
| `wave_metrics` | Per wave given: `retest_id`, `reported_at_utc`, `condition_metrics` (`cued`, `uncued`) and `effect_summary`, computed as the top-level ones from that wave's results. |
| `incorporation` | `source` (`raters` with rater coding sheets of the night, else `self_report` with a dream report, else `none`), `rater_ids`, `rater_count`, `incorporated_count`, `self_linked_count`, `split_puzzle_ids` (raters disagreed), `uncoded_puzzle_ids`. |
| `incorporation_solved` | `available` (false with source `none`), `source`, and per wave `incorporated` and `not_incorporated` (each `puzzles`, `solved`, `unsolved`, `solve_rate`), also `by_condition`. |
| rows: `wave_results` | Per wave the puzzle's `solved`, `duration_seconds`, `confidence`. |
| rows: `self_linked`, `rater_codes` | Whether the dream report links the puzzle; `rater_count` and the share of raters per category and for `any`. `incorporated` follows the raters when the night has codes: more than half of them put the puzzle in some category. |
| rows: `sham_play_count` | A sham night's plays in the puzzle's place: the sham sound played, so they are not in `cue_play_count`. |
| `cue_timing` | Adds `arm` (the arm the scheduler events carry, `null` without events), `sham_puzzle_play_count` and `virtual_puzzle_play_count`. `uncued_puzzle_play_count` counts every play that names an uncued puzzle, a sham play included. |

New limitations: `missing_delayed_retest` (no `day7` or no `day10` wave; `metadata.missing_waves`),
`sham_arm` (in place of `missing_cued_timing` on a sham night) and `no_cue_arm`. A schema
1 or 2 report loads with empty new fields, each row's `self_linked` read from its
`incorporated`.

## Blind Coding Packets And Coding Sheets

`muse_tmr.reports.incorporation_coding` (#228) writes two kinds of file, which hold dream
text and stay in the gitignored `data/reports/`.

The packet of `export-blind-coding`, version 1 (`BlindCodingPacket`):

| Field | Content |
|---|---|
| `kind`, `schema_version`, `packet_id` | `blind_coding_packet`, 1, `<session>-blind-coding`. |
| `session_id`, `generated_at_utc` | The night and when the packet was written. |
| `instructions`, `categories` | The rater's instructions and the definitions of `sound_cue`, `direct_conceptual` and `indirect_conceptual`. |
| `dream_text` | The dream report's text, and nothing else of the report. |
| `puzzles` | `blind_index` (1 to N in the session's order, as the retest numbers them), `puzzle_id`, `prompt`. |

A rater's coding sheet from `record-incorporation-coding`, version 1
(`IncorporationCodingSheet`): `kind` `incorporation_coding`, `schema_version`,
`packet_id`, `session_id`, `rater_id`, `coded_at_utc`, `coding_count`,
`incorporated_count`, `notes` and `codings`, one `IncorporationCoding` per puzzle:
`rater_id`, `puzzle_id`, the three categories as true or false, and `notes`.
`compare-incorporation-codings --output` writes `rater_ids`, `session_ids`,
`item_count`, `unmatched` (puzzles one rater only coded) and per category and `any`:
`items`, `agreements`, `percent_agreement`, `kappa` (Cohen's, `null` when chance
agreement is 1) with `kappa_note`, and `yes_counts` per rater.

## N-of-1 Series Analyses

`analyze-nof1` (`muse_tmr.reports.nof1_analysis`, #228) writes version 1:

| Field | Content |
|---|---|
| `report`, `schema_version`, `analysis_id`, `generated_at_utc` | `nof1_analysis`, 1, `<schedule-id>-nof1-analysis`. With a synthetic night, `synthetic: true` and the note. |
| `schedule` | `schedule_id`, `created_at_utc`, `seed`, `blocks`, `washout_nights`, `arms`, `sham_cue_id`, `night_count`, `commitment_sha256`, `seal` (`matches` once `analyze-nof1` has checked the schedule against its sealed companion, `not_checked` from `build_nof1_analysis()` alone), `unsealed_at_utc`, and the nights' distinct `protocol_effective_hashes`. |
| `settings` | `seed`, `permutations`, `interval` (0.95), `min_nights_per_arm`, `primary_outcome`, `primary_contrast`, `reference_set` (`schedule_design`, or `within_block` for a schedule `plan-night-schedule` could not have drawn), `possible_orders`, `method` (`exact` or `monte_carlo`), `relabelings`, `outside_possible`, `software_version`. |
| `outcomes` | Each outcome's definition. |
| `nights` | Each block night: `night_id`, `position`, `block`, `arm` (the schedule's), `first_night`, `has_data`, `synthetic`, `summary_arm`, `summary_kind`, `started_at`, `summary_passed`, `failed_criteria`, `protocol_effective_hash`, `plays`, `virtual_plays`, `inputs` (what was found), `incorporation_source`, `arousal_proxy_per_cue` and `values`, one per outcome, `null` when missing. |
| `descriptives` | Per outcome and arm: `nights`, `mean`, `nights_without_first_night`, `mean_without_first_night`. |
| `differences` | Per outcome and contrast (`cue-no_cue`, `cue-sham`, `sham-no_cue`): `difference` (first arm's mean minus the second's), `nights` per arm, `interval` (low, high), `observed_outside_interval`, `relabelings` and `primary`. |
| `autocorrelation` | Per outcome: `lag1` of the values once the arm means are taken out, over consecutive positions, with `pairs` and `nights`. |
| `cue_counts`, `arousal` | Per arm: nights with a summary, `plays`, `plays_per_night`, `virtual_plays`; nights with arousal proxies and the mean of each flag's rate per cue. |
| `limitations` | `code`, `severity`, `message`, `metadata`, as in the cued-vs-uncued analysis; the codes are in `docs/nof1_protocol.md`. |
| `reporting_checklist`, `reporting_checklist_counts` | Each SCRIBE 2016 item with its CENT 2015 items, `status` (`auto`, `partial`, `manual`) and what fills it; the count per status. |

No dream text is copied into it.

## Cue Recall Tests

`muse_tmr.reports.cue_recall` (`record-cue-recall`, and the app's Morning tab) writes
`<night>_cue_recall.json`, version 1 (`CueRecallTest`, `load_cue_recall_test()`):

| Field | Content |
|---|---|
| `schema_version`, `test_id`, `session_id` | 1, `<session>-cue-recall`, the night's session. |
| `started_at_utc`, `completed_at_utc` | When the first sound played and when the check was saved. |
| `order_seed` | The seed of the play order, made from the session ID alone (`cue_recall_order()`). |
| `trial_count`, `answered_count`, `correct_count`, `recall_rate`, `by_condition` | As in the analysis. |
| `trials` | One per puzzle in play order: `trial` (the number shown), `puzzle_id` and `cue_id` of the sound, `response_puzzle_id` (`null`: not sure), `correct`, `cue_condition` (for the analysis only), `play_count` and `plays` (each play's `status`, `backend_name`, `requested_volume`, `capped_volume`, `effective_volume`, `max_volume`, `reason_codes`, `started_at`). |
| `notes`, `metadata` | Free text; the audio log's name, the requested and resolved backend, and a note that the condition was not shown. |

Each play also goes to `<night>_cue_recall_audio.jsonl` through `AudioCuePlayer`'s log.
The app's Morning tab sends the page only the sound numbers, whether they can play and,
once saved, the answers by number.

## Protocol Presets

`muse_tmr.protocol.protocol_presets` defines the records of a protocol preset and
`muse_tmr.protocol.protocol_registry` the presets (#231, [`protocol_presets.md`](protocol_presets.md)):

- `ProtocolPreset`: `name`, `version`, `title`, `runnable` (with `not_runnable_reason`),
  `design_note`, `caveats`, `citations`, `cues`, `timing`, `pauses`, `ramp`, `training`,
  `morning`, `runner`, `deviations`
- `Citation`: authors, year, title, venue, URL, DOI, arm, `source_check`
  (`verified_full_text` or `notes_only`) and a note
- `CueSpec`: a cue as published, by stream (`tlr`, `puzzle` or `any`): description,
  duration, frequencies
- `CueTiming`: a stream's published interval range and limit of cues per REM period
  (the fidelity report compares the limit per scheduler REM block)
- `PauseRules`: REM before the first cue and before resuming, pauses by trigger
  (movement, alpha, blink, poor_signal, heart_rate, arousal), pauses that lasted until a
  sign was gone, the pause after an awakening, what stopped cueing
- `IntensityRamp`: the published level, start, step per cue or per cycle, cap and what an
  arousal did; documentation and a fidelity-report input only
- `PlatformDeviation`: a parameter, the published value and what REMind does
- `RunnerSettings`: what the runners use; the only part that reaches them, through
  `scheduler_config()`, `tlr_block_config()`, `tlr_cue_config()`, `gate_overrides()` and
  `puzzle_counts()`

`content_hash()` is the sha256 of the preset's canonical JSON (sorted keys, no spaces).
`manifest(overrides)` is the `protocol` block a run records: `schema_version` (1),
`name`, `version`, `registry_version`, `content_hash`, `runnable`, `overrides` (the
explicit flags, by setting), `effective_hash` (the sha256 of the content hash and the
overrides) and `settings` (the runner settings with the overrides). A run whose
assignment or TLR plan was made with a preset adds `inputs`, their `name`, `version`,
`content_hash` and `effective_hash`.

## Protocol Fidelity Reports

`muse_tmr.reports.protocol_fidelity` (`protocol-fidelity`) writes
`protocol_fidelity.json`, version 1, and `protocol_fidelity.md` into the night's
directory or the Pilot 3 report's (#231):

| Field | Content |
|---|---|
| `schema_version`, `recording`, `generated_at_utc` | 1, the directory's name, the time it was written. |
| `synthetic`, `note` | `true` and "synthetic input, not sleep" for a synthetic recording. |
| `parameter_fidelity_is_not_result_fidelity` | Always there: `statement`, `no_psg_verified_rem`, `no_eog`, `gain_not_db_spl`, `first_cue_latency`. |
| `preset` | The preset compared with: `name`, `version`, `title`, `content_hash`, `runnable`, `source` (the night's, the one asked for, or the default) and its `citations`. |
| `night_protocol`, `settings` | The night's `protocol` block and the runner settings it ran with. |
| `inputs` | Which files were read. |
| `summary` | `rem_periods`, `rem_periods_with_cues`, `plays` by stream (the plays the player played), `plays_blocked_or_skipped` by stream (left out of every check), `plays_outside_rem_periods`, `uncued_puzzle_play_count`, `stride_seconds`, the night's `arm` (from its summary or `metadata.json`, `null` without one) and `virtual_plays` by stream, a no-cue night's skips where a cue night would have played. |
| `rem_periods` | Each gate-open run with its plays by stream, `first_cue_seconds_after_gate_open` and `stable_rem_seconds_at_first_cue`. |
| `checks` | `inter_cue_intervals`, `cues_per_rem_period`, `stable_rem_before_first_cue`, `pause_rules`, `volume_ramp`, `cue_duration`, `tlr_block` and, since #264, `tlr_tone` (the TLR plan's `tone` in `metadata.json` against the preset's `tlr_tone_frequency_hz` and `tlr_tone_duration_seconds`): each a `status` with `observed` and `published` and its detail. |
| `deviations` | The preset's platform deviations. |
| `limitations` | Codes: `synthetic_input`, `no_protocol_block`, `preset_changed_since_night`, `night_overrode_preset_settings`, `compared_with_another_preset`, `unknown_preset_in_night`, `no_gate_log`, `no_scheduler_events`, `no_audio_log`, `audio_log_mismatch`, `plays_blocked_or_skipped`, `pilot3_plan_not_played`, `awakening_markers_unplaced`, `sham_arm` and `no_cue_arm` (#228). |

Puzzle protocol files may contain private puzzle content, responses, and night/session
metadata, so `data/protocol/` is gitignored.

## Night Schedules

`muse_tmr.protocol.night_schedule` (`plan-night-schedule`, #228) writes a series'
schedule and its sealed companion into the gitignored `data/protocol/night_schedules/`.
The schedule, `<schedule-id>.json`, version 1 (`NightSchedule`, `load_night_schedule()`):

| Field | Content |
|---|---|
| `schema_version`, `schedule_id`, `created_at_utc` | 1, the series' name, when it was drawn. |
| `seed`, `blocks`, `washout_nights`, `arms` | What drew it: `random.Random(seed)` orders the `arms` (`cue`, `sham`, `no_cue`) of each block; within every group of three blocks the orders are rows of one Latin square. |
| `sham_cue_id` | The untrained sound of the sham nights. |
| `nights` | In series order: `night_id`, `position` (1, 2, ...), `block` (`null` for a washout night) and `arm` (`cue`, `sham`, `no_cue` or `washout`). |
| `salt` | 32 hex characters from `secrets`. |
| `commitment_sha256` | sha256 of the canonical JSON (sorted keys, no spaces, UTF-8) of the fields from `schema_version` to `nights`, followed by the salt. |
| `unsealed_at_utc` | `null` until `unseal-night-schedule`, then the first time it ran. |

The sealed companion, `<schedule-id>.sealed.json`, holds only `{"commitment_sha256": ...}`.
`unseal_night_schedule()` and every runner check the schedule against it first
(`SealMismatch` otherwise); `arm_for(night_id)` gives a night's arm. `prereg_template()`
(`prereg-template`) renders a preregistration from the schedule's design, its
commitment and a protocol block, never from its arms, seed or salt.

A night run with an arm writes a `night_arm` block into its `metadata.json` (Pilot 4/5):
`arm`, `night_id`, `sham_cue_id`, `sealed_schedule` (whether the arm came from a
schedule), `schedule_id`, `schedule_commitment_sha256` and `schedule_unsealed_at_utc`
(the schedule's unsealing time when the night started, `null` while sealed). An
open-label `--arm` night has `sealed_schedule: false` and no schedule fields.

## Pilot Time Base And Awakening Markers

`pilot4_summary.json` and `pilot5_summary.json` are schema 4 since #228, which adds
`arm`, `night_id` and `schedule_commitment_sha256` (all `null` on a night without an
arm), `virtual_play_count` and, on a night with an arm, the criteria
`no_play_in_no_cue_arm` and, with a sealed schedule, `arm_matches_sealed_schedule`; such
a night's `metadata.json` adds the `night_arm` block (Night Schedules, above). A Pilot 3 report is
schema 2 with the same three fields, criteria and `metrics.virtual_play_count`. Schema 3
(#231) added `protocol`, the night's preset block (also in `metadata.json`, with the TLR plan the night
used as `tlr_block_plan`); a Pilot 3 report has the same `protocol`. Schema 2 (#227) added
`first_epoch_start`, the first epoch's start in the frames' clock (Unix seconds, `null`
without an epoch), which every `timestamp_seconds` of the night counts from;
`rem_gate_decisions_path`; and `cue_arousal`, the scorer's block below. Schema 1
summaries have none of the three and still load: the scorer then takes the recording's
first frame, where the first epoch of a full replay starts, and says
`time_base_inferred`. A Pilot 3 report's `metrics` add `first_epoch_start` the same way,
`injected_cue_count` and `inject_cues_every_seconds`, and the report adds
`injected_cues`: with `--inject-cues-every-seconds N`, a `play` event with
`cue_id` and reason code `synthetic_cue`, `protocol` `synthetic` and
`metadata.decision_seconds` for a gate-open epoch whenever N seconds have passed since
the last one. The scheduler never sees them; `scheduler_events` and `cue_plan` stay as
they are without them.

An `awakening_events.jsonl` row (`AwakeningEvent`) has `event_type`, `timestamp_utc` and
`notes`, and since #227 may add `timestamp_seconds` (on the night's time base) and
`attributed_to_cue` (`yes`, `no` or `unsure`), written only when given;
`load_awakening_events()` reads old and new rows. `log-pilot4-awakening` takes them as
`--timestamp-seconds` and `--attributed-to-cue`.

## Cue Arousal Reports

`muse_tmr.reports.cue_arousal` (`score-cue-arousal`, and Pilot 4 and 5 at the end of a
night) writes three files, by default into the night's directory (#227;
`docs/safety_outcomes.md` has the definitions). They stay local, like the night.

`cue_arousal_events.jsonl` holds one `CueArousalWindow` a line, cues and controls in time
order:

| Field | Content |
|---|---|
| `kind`, `index` | `cue` or `control`, and its number among them. `paired_index` is the other half of its pair, `null` for an unpaired cue. |
| `offset_seconds`, `time` | The moment scored, from the first epoch's start and in the recording's clock (Unix seconds). For a cue, the end of the epoch whose decision played it. |
| `gate_open`, `rem_period_index`, `seconds_since_gate_open`, `rem_period_ends_in_seconds` | The gate state at that moment, its gate-open run, the time since the run began and, when the run closed, the time until it did. |
| `awakening_in_seconds` | For a cue, the first awakening marker within 120 s after it, else `null`. |
| `flags` | `alpha`, `beta`, `emg`, `imu`, `hr`: `true` or `false`, `null` where the window could not be scored; `any`. |
| `values` | `eeg_coverage_baseline`, `eeg_coverage_post`, `eeg_channels_used`, `<band>_baseline_power`, `<band>_post_power` (µV²) and `<band>_rise` for `alpha`, `beta`, `emg`; `imu_coverage`, `movement_events`, `arousal_movement_events`, `peak_motion_level`; `hr_values_baseline`, `hr_values_post`, `hr_values_skipped`, `hr_baseline_bpm`, `hr_post_max_bpm`, `hr_jump_bpm`, `hr_quality`. |
| `quality_flags` | `eeg_gap`, `eeg_unusable`, `imu_gap`, `ppg_absent`, `hr_unavailable`. |
| `cue` | A cue's own logs, empty for a control: `cue_id`, `protocol`, `puzzle_id`, `scheduler_timestamp_seconds`, `synthetic_cue`, `guard_action` and `guard_volume_multiplier` from its play event, `puzzle_cue_index`, `tlr_event_index`, `inter_cue_interval_seconds`, `audio_status` and `audio_backend` from its line of `audio_playback.jsonl`, `requested_volume`, `effective_volume`, `volume_multiplier` and their `volume_source` (`audio_playback.jsonl`, `scheduler_event` or `null`), `played` and `audible`. |

`cue_arousal_report.json`, version 1 (`CueArousalReport`, `load_cue_arousal_report()`):

| Field | Content |
|---|---|
| `schema_version`, `report`, `recording`, `generated_at_utc` | 1, `cue_arousal`, the directory's name, the time it was written. |
| `synthetic`, `note` | `true` and "synthetic input, not sleep" when the replayed recording's `metadata.json` says `synthetic: true`. |
| `outcomes` | `cue_count`, `played_cue_count`, `synthetic_cue_count`, `audible_cue_count`, `scored_cue_count`, `control_count`, `scored_control_count`, `paired_count`; `arousal_proxy_per_cue`, `arousal_proxy_per_control` and `arousal_proxy_difference` by flag; `flag_counts` (flagged and scored per flag, cues and controls); `paired_any`; `movement_events_per_cue`, `movement_events_per_control`; `cue_attributed_awakening_rate`, `awakening_attribution_counts`, `awakening_count`; `rem_period_count`, `cued_rem_period_count`, `rem_periods_with_disruption_fraction`; `rem_termination_within_60s_of_cue_rate`, `rem_termination_within_60s_of_control_rate`. A rate is `null` without a denominator. |
| `dose_tables` | `effective_volume`, `inter_cue_interval`, `cue_index`: rows of `bin`, `cues`, `any_flagged`, `any_scored`, `any_rate` and `flag_rates`. |
| `rem_periods` | Each gate-open run with its cues, controls, cues with `any` and whether it was disrupted. |
| `limitations` | `code`, `severity`, `message`, `metadata`, as in the cued-vs-uncued analysis. Always `thresholds_unfitted`, `arousal_proxy_only`, `software_gain`, `descriptive_only`; the others in `docs/safety_outcomes.md`. |
| `config` | `CueArousalConfig` with its IMU and PPG configs, and `unfitted_thresholds`. |
| `metadata` | `first_epoch_start`, `time_base` (`logged` or `inferred_from_first_frame`), `optics_capable`, the replayed recording's name and which input files were read. |
| `windows` | The lines of `cue_arousal_events.jsonl`. |

`cue_arousal_report.md` has the summary table, the dose tables and the limitations. The
`cue_arousal` block of a Pilot 4/5 summary has `status` (`scored` or `failed` with
`error`), the counts and rates of `outcomes` named above, the limitation codes and the
three paths. `score-cue-arousal` on the night's directory rewrites it with the report it
writes there (`write_summary_block()`); a schema 1 summary has no block and is left as it
is.

## Optics Agreement Reports

`muse-tmr analyze-optics-agreement` (`muse_tmr.reports.optics_agreement`) writes
`optics_agreement.json` and `optics_agreement.md`, by default into
`data/reports/optics_agreement/<recording name>/` (gitignored). The JSON, version 1
(`OpticsAgreementReport`, `load_optics_agreement_report()` reads it back):

| Field | Content |
|---|---|
| `schema_version`, `report`, `recording`, `generated_at_utc` | 1, `optics_agreement`, the recording directory's name, the time it was written. |
| `synthetic`, `note` | `true` and "synthetic input, not sleep" when the recording's `metadata.json` says `synthetic: true`. |
| `config` | `OpticsAgreementConfig`, the PPG feature and cardio configs inside it. |
| `counts` | `epochs`, `optics_epochs`, `ecg_reference_epochs`, `compared_epochs`, `series_values`, `series_values_with_reference`, `blind_epochs`, `r_peaks`, `ecg_intervals`, `ecg_intervals_corrected`, `duration_seconds`, `no_contact_seconds`. |
| `agreement` | One row per estimator (`decoder_heart_rate`, `ppg_estimated_hr_bpm`, `channel`, `hr_series`, `hr_series_skipped`), `channel`, `stratum` (`all`, `ppg_signal_quality`, `ppg_confidence`, `series_quality`, `periodicity`) and `level`: `n_reference`, `n`, `coverage`, `bias_bpm`, `sd_bpm`, `loa_low_bpm`, `loa_high_bpm`, `mae_bpm` (estimate minus ECG; `null` without enough values). |
| `jumps` | The sudden heart-rate change rule against the ECG per `stratum` (`all`, the three quality fields, `hr_rule_blind`) and `level`: `epochs`, `fired`, `false_jumps`, `false_jump_rate`, `ecg_jumps` (each ECG change of 10 bpm or more once, in the epoch it starts in, also when the next epoch sees it again), `missed`, `miss_rate`. |
| `gates` | The rule with a gate `field` (`none`, `ppg_confidence`, `hr_series_median_quality`) at `threshold`: `gated_epochs`, `fired`, `false_jumps`, `false_jump_rate`, `ecg_jumps`, `caught`, `miss_rate`. |
| `separation` | Per gate field over the fired epochs: `true_fires`, `false_fires`, `median_true`, `median_false`, `auc`. |
| `events` | Each change the rule counted: `epoch_index`, `kind` (`row` or `across` the boundary), `start_seconds` and `end_seconds` from the first epoch, `optics_delta_bpm`, `ecg_delta_bpm` between the same windows. |
| `limitations` | `code`, `message`, `metadata`: `synthetic_input`, `no_contact_seconds`, `n_epochs_small` (fewer than `min_epochs`, 60, compared epochs), `few_jump_events` (fewer than `min_jump_events`, 10, fired epochs and ECG jumps together). |
| `metadata` | `preset`, and `polar`: the H10 session's R-peak count, no-contact seconds, ECG noise spans, clock segments, drift, median BLE delay, NTP step and RR matched to the ECG. No device name or address. |

How each figure is computed and read: `docs/optics_agreement.md`.
