# Importing nights from other apps

`muse-tmr import FILE --format FORMAT` turns a night recorded by another app into a recording directory, so the REM report and a replay simulation can run on it. How to run it: [`USAGE.md`](USAGE.md#importing-a-night-from-another-app). The other way, from REMind to EDF, BIDS and MNE: [`formats.md`](formats.md).

## Report and simulation only

An import writes `decoded_frames.jsonl`, `metadata.json` and `summary.json` into `data/recordings/imported/<name>/` (gitignored), where the name is the input's, or for a `decoded_frames.jsonl` its recording directory's, or into `--output`: a new or empty directory, or an earlier import of the same file:

- `metadata.json`: `source.source_name` `import:<format>`, `capabilities.raw_packets` false, and an `import` block with the scale, offset, who declared them, their evidence and status, the clock, the input's name, size and sha256, and for an EDF its signals, channel map and any resampling. Nothing from the input's header that could name a person or a device is copied.
- `summary.json`: `raw_packet_count` 0 and `stop_reason` `imported`.

`replay`, the REM report (`muse_tmr.reports.nightly_report`, `scripts/generate_nightly_report.py`), `annotate-template` and `simulate-replay-cues` read the directory like any recording, and each says `imported: <format>, scale <x>; report and simulation only` in what it prints or writes. They refuse an import whose metadata has lost its scale. `validate-pilot1-recording` fails an import: Pilot 1 needs raw packets (`raw_capture_present`, `raw_packet_count > 0`) and a `duration_complete` stop. Nothing cues from an import: the runners that play sound, `run-pilot4-cueing` and `run-pilot5-full-night`, take a headband, not a recording, and neither the importer nor an export writes a `play`. A P_REM curve of an imported night says what the detector makes of values another decoder scaled; it is no evidence about REM.

## Scales

EEG values become centred microvolts as `value × scale + offset`. The scale comes from the table below, or from `--eeg-scale` with `--eeg-offset` (0 by default), which replaces the table's row. A format without a verified row needs `--eeg-scale`, and BrainFlow needs `--source-version`, since its two rows differ 4.55 times.

| Format | Applies to | Scale | Offset (µV) | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| `decoded-frames` | `decoded_frames.jsonl` of `muse-tmr record` | 1 | 0 | this repository's decoder, centred µV | same decoder |
| `edf` | EDF or BDF with EEG in µV, mV or V | 1 | 0 | the unit the header names; `--eeg-scale` stands in for it | header units |
| `muselsl-csv` | `muselsl record` CSV of a Muse S Athena | 1 | 0 | `muselsl/athena.py` decodes (raw − 8192) × 1450/16383 since [muse-lsl PR #228](https://github.com/alexandrebarachant/muse-lsl/pull/228) (merged 2026-06-30) | source code read 2026-10-27 |
| `openmuse-csv` | the EEG DataFrame of `OpenMuse.decode_rawdata()`, saved as CSV | 1 | −725.04 | [`OpenMuse/decode.py`](https://github.com/DominiqueMakowski/OpenMuse/blob/main/OpenMuse/decode.py) scales raw × 1450/16383 without subtracting 8192 | source code read 2026-10-27 |
| `brainflow-csv` | BrainFlow 5.23.0 or later | 1 | 0 | [PR #855](https://github.com/brainflow-dev/brainflow/pull/855) (scale 1450/16383) and [PR #856](https://github.com/brainflow-dev/brainflow/pull/856) (centred on 8192), merged 2026-08-18, first released in 5.23.0 | source code read 2026-10-27 |
| `brainflow-csv` | BrainFlow before 5.23.0 | 0.2197 (÷ 4.55) | −725.04 | before PR #855 Athena counts were scaled by 1650/4095 µV, 4.55 times 1450/16383, and before PR #856 not centred | source code read 2026-10-27 |
| `mindmonitor-csv` | Mind Monitor CSV, `RAW_TP9` to `RAW_TP10` | none | 0 | no primary document of Mind Monitor's EEG units for the Athena was found; its site answered 403 to the [research notes](research/2026-10-uniqueness/notes/muse_ecosystem.md) | unverified: `--eeg-scale` required |

All of them rest on the 1450/16383 µV per count of the decoders, whose 1450 µV full scale is vendor-stated and unverified ([`formats.md`](formats.md#edfc-layout)).

## What each format keeps and loses

| Format | EEG | IMU, optics, heart rate, battery | Times | Gaps |
| --- | --- | --- | --- | --- |
| `decoded-frames` | as decoded | as decoded; a file written before #244 lacks the IMU rows of all but the last ACCGYRO subpacket of a notification | the recorder's frame timestamps, Unix | lost subpackets as `gaps`, dropouts as time |
| `edf` | the channels the map names, converted to µV | accelerometer, gyroscope and a pulse or optics channel when the map names them | the header's start read as UTC, plus the sample index; an anonymised date counts as 1 January 1985 | `BAD_ACQ_SKIP` stretches come back as gaps; a stream at another rate than 256, 52 or 64 Hz is resampled to it, which the `import` block records; annotations are not imported (stages go through `annotate-template --from-hypnogram`) |
| `muselsl-csv` | `TP9`, `AF7`, `AF8`, `TP10`, to 0.001 µV as muse-lsl writes them | none | `timestamps`, LSL clock to the millisecond | time jumps only |
| `openmuse-csv` | `EEG_TP9` to `EEG_TP10` | none | `time`, seconds from the stream start | time jumps only |
| `brainflow-csv` | rows 1 to 4 of the default preset, to 6 decimals | none: the auxiliary and ancillary presets are other files | row 9, Unix seconds | time jumps only |
| `mindmonitor-csv` | `RAW_TP9` to `RAW_TP10` | none | `TimeStamp` (`2019-06-10 16:46:53.081` in the exports read), local time without a zone | time jumps only |

The CSV formats are read in frames of up to 12 rows, split where the times jump by more than 0.1 s. A clock that counts from the stream start (muse-lsl's LSL clock, OpenMuse's `time`, any time before 2000) needs `--start-time`, the moment the first sample was taken; without it the start is the file's modification time less its length, recorded as approximate. Mind Monitor writes local time without a zone, which is read in this machine's zone unless `--start-time` anchors it. Its layout (`TimeStamp`, the band powers, `RAW_TP9` to `RAW_TP10`, ..., `Elements`) is that of the Mind Monitor exports found in public repositories; Mind Monitor's own documentation could not be read.

## Channel maps for EDF

Without `--channel-map` an EDF is read by the labels `export-edf` writes. `--channel-map boas-headband` reads the headband files of the BOAS dataset ([OpenNeuro ds005555](https://github.com/OpenNeuroDatasets/ds005555)), `sub-*_task-Sleep_acq-headband_eeg.edf`: `HB_1` and `HB_2` (near AF7 and AF8, against the left mastoid) as AF7 and AF8, `HB_IMU_1` to `3` as the accelerometer, `HB_IMU_4` to `6` as the gyroscope and `HB_PULSE` as a pulse channel, all recorded at 256 Hz; nights without IMU or pulse read the same way. Any other file takes `LABEL=TARGET,...` with targets `eeg:NAME`, `accel:x`, `gyro:z`, `ppg:NAME` or `skip`; a target without a kind is EEG.

## Hypnograms from other tools

`annotate-template --from-hypnogram FILE` labels each epoch by the stage at its middle: W `wake`, N, N1, N2 and N3 `nrem`, R `probable_rem`, anything else `unknown` (the map in [`formats.md`](formats.md#stages-and-the-lossy-map)). It reads an EDF+ file's stage annotations, a BIDS events file (an export's `acq-scoring` file, or BOAS's `stage_hum`, else `stage_ai`, column, `--hypnogram-column` to choose), a CSV with `start_s`/`end_s`, `onset`/`duration` or 30 s `epoch` numbers, and `annotate-template` rows, which are host times. Onsets count from the recording's first frame on its sample clock, as in an `export-edf` file. Every row gets `prediction_source` `hypnogram:<file>` and loads into `train-rem-classifier`, so an expert's scoring of an exported night, or the hypnogram of a device recorded alongside, becomes training labels.

## Recordings from the BrainFlow and OpenMuse sources

`muse-tmr record --source brainflow` and `--source openmuse` write no raw packets, only `decoded_frames.jsonl`, and `replay` now reads that file when `raw_amused.bin` holds no packet (#108). Their EEG has the scale of the decoder that streamed it: BrainFlow before 5.23.0 gives counts 4.55 times too large and not centred, and OpenMuse leaves its pedestal. To run the report on such a night in centred µV, import its `decoded_frames.jsonl` with `--format decoded-frames --eeg-scale 0.2197 --eeg-offset -725.04` (BrainFlow before 5.23.0) or `--eeg-offset -725.04` with `--eeg-scale 1` (OpenMuse).
