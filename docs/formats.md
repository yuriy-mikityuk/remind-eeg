# Exported formats: EDF+, BIDS and MNE

`muse-tmr export-edf` writes a recording as one EDF+ file, `muse-tmr export-bids` as one EEG-BIDS session, and `muse_tmr.data.mne_export.to_raw()` returns it as an MNE `Raw`, so the night can go into YASA, Luna, U-Sleep, MNE or EDFbrowser with its cue, gate and guard events as annotations. Both commands need the `[edf]` extra (`pip install -e ".[edf]"`), which installs [edfio](https://github.com/the-siesta-group/edfio); `to_raw()` needs MNE-Python, which REMind never installs. How to run them: [`USAGE.md`](USAGE.md#exporting-a-night-edf-bids-and-mne). Nights from other apps go the other way: [`import.md`](import.md).

An export is a night's data. `export-edf` writes next to the recording unless `--output` says otherwise, `export-bids` under `data/bids/`; both are gitignored, and `scripts/check_forbidden_files.py` refuses `*.edf` and `data/bids/` in a commit.

## From frames to arrays

`muse_tmr.data.continuous` lays the samples of each stream end to end at its nominal rate: EEG 256 Hz, accelerometer and gyroscope 52 Hz, optics 64 Hz. A frame's samples are taken to start at the frame's timestamp, the host time its notification arrived, so sample 0 of the export is the night's first frame. The arrays then run on the headband's own sample clock, not on host time. The EEG rate the night actually had, samples per host second from the first frame to the last, goes into the EDF header and the BIDS sidecar; on the one session measured it was about 257.2 Hz (#160), so a night's export runs a little longer than the night. Events are placed through the frames they arrived with, so each one lands on the samples that came in at that moment, wherever the clock has drifted.

What the arrays lost is kept:

- A subpacket that never arrived (the decoder's `gaps`, #182) becomes zeros at its place, so the index follows the device's sample counter. They were 1.0 % of the EEG subpackets on the one session counted (#160) and are not annotated one by one; the export's `lost_samples` counts them.
- A frame more than 0.5 s after the one before starts a dropout. Its zeros are as many samples as the device counter says went missing when the frame after it reports a skip at its start that agrees with host time to within a quarter second, and as many as host time says otherwise: a counter that wrapped while the stream was away is off by whole wraps. On a synthetic night with BLE gaps the counter put every sample back in its place, where host time was a sample or two off after each gap. The zeros are annotated `BAD_ACQ_SKIP`, which MNE leaves out of its epochs and filters by default.

That is the `zero_fill` gap policy, the only one the EDF is written with. `discontinuous` fills nothing and starts a new segment at each dropout; `to_raw()` takes it and marks each join `BAD boundary` and `EDGE boundary` as `mne.concatenate_raws` does. EDF+D is not written: neither edfio nor pyedflib writes it, and `read_edf()` refuses an EDF+D file rather than join its records as if they followed each other.

`scripts/validate_rem_detection.py` reads the night through the same module with its own settings (frames without AF7 left out, host time for every dropout, lost subpackets not filled), so the reference stager gets the same arrays, rate and sample count as before this module existed.

## EDF+C layout

One EDF+C file with 1 s data records. Its start is the first frame, in UTC; the microseconds go into the first record's time-keeping annotation, as EDF+ allows.

| Signals | Unit | Rate | Digital range | Physical range | Lossless |
| --- | --- | --- | --- | --- | --- |
| EEG `TP9`, `AF7`, `AF8`, `TP10` (and `FPz`, `AUX_R`, `AUX_L`, `CH8` in 8-channel mode) | uV | 256 Hz | −8192 to 8191 | −725.044 to 724.9557 | yes: the 14-bit counts unchanged |
| `ACC_X`, `ACC_Y`, `ACC_Z` | g | 52 Hz | −32768 to 32767 | the decoder's 16-bit counts | yes |
| `GYRO_X`, `GYRO_Y`, `GYRO_Z` | deg/s | 52 Hz | −32768 to 32767 | the decoder's 16-bit counts | yes |
| optics (`LO_NIR` ... `RI_IR`, or `opt0` ...) | none | 64 Hz | −32768 to 32767 | each channel's own range | no: 20-bit values quantised to 16 bits |

The EEG keeps the decoder's 1450/16383 µV per count, centred on count 8192. The 1450 µV full scale is vendor-stated and unverified: no primary InteraXon document was found, only BrainFlow's "published 1.45 mVpp range" ([research note](research/2026-10-uniqueness/notes/gap_4_the_ocular_channel_on_muse_why_the_eye_m.md)). The transducer field of each EEG signal says so. A signal that is not on its count grid, as an imported one, is quantised to 16 bits over its own range like the optics; `export-edf --json` gives each signal's resolution and whether it is lossless.

The header's patient field is `X X X X`, with `synthetic input, not sleep` after it on a synthetic night, and the recording field `Startdate <date> X X Muse_S_Athena measured_rate=<Hz>Hz gaps=zero_fill`. No device name, address or serial goes into the file. The streams can end a little apart; the file is long enough for the longest, and the EEG's padding at its end is annotated `BAD_ACQ_SKIP` too.

## Annotations

The annotations come from `muse_tmr.data.night_events`, which reads what the recording directory holds. Every text is built from names and codes, never from free text, paths or device names.

| Log | Annotation |
| --- | --- |
| `events.jsonl` | Recorder and runner events by name, with `reason`, `attempt`, `modality`, `backoff_seconds`, `companion`, `returncode` or `pilot_id` when they have one: `recording_started`, `reconnect_attempt attempt=1 backoff_seconds=1.0`, `emergency_stop_triggered`, `pilot_stopped reason=duration_complete pilot_id=...` |
| `scheduler_events.jsonl` | `scheduler play cue=<cue> protocol=<protocol> puzzle=<puzzle> reasons=<codes>`, and `pause` and `stop` the same way. A run of skips with the same reasons is one annotation from the first to the next scheduler event, with `n=<count>` |
| `rem_gate_decisions.jsonl` | `rem_gate_open` from the decision that opened the gate to the one that closed it, the gate-open runs of `muse_tmr.validation.gate_log`. A night from before that log gets them from the scheduler's `rem_gate_open` and `rem_gate_closed` codes, which find an opening only at the next play |
| `arousal_guard_events.jsonl` | Every decision but `allow`: `arousal_guard pause reasons=<codes> pause_seconds=60`, runs of equal decisions merged |
| `audio_playback.jsonl` | `audio played cue=<cue> volume=<v> backend=<backend>`, or `blocked`, `skipped`, and `audio emergency_stop`. A dry-run or mock `played` made no sound; the backend says which it was |
| `awakening_events.jsonl` | The marker's type and `attributed_to_cue`; never its notes |
| a synthetic night's `hypnogram.csv`, `--hypnogram` files | Stage texts, below |

At 0 s come the notes: `synthetic input, not sleep` on a synthetic night and the banner of an imported one. The night logs count their `timestamp_seconds` from the first epoch's start (`first_epoch_start` of the Pilot 4/5 summary); the export places each by host time on the sample clock, and reading the EDF back puts each annotation within a microsecond of where it was written.

An export of a cueing night names the puzzles that were cued, so make it after the morning retest. A night whose arm came from a sealed night schedule names its arm in its logs (`sham_arm`, `no_cue_arm`, `virtual_play`, `arm=sham`), so `export-edf` and `export-bids` refuse it, naming `unseal-night-schedule`, until the schedule has been unsealed; then pass it with `--night-schedule`, or keep it at `data/protocol/night_schedules/<schedule-id>.json`. An open-label night (`--arm`) exports as it is.

## Stages and the lossy map

Stages go through the four REM annotation labels and out as EDF+ standard texts ([EDF+ specification](https://www.edfplus.info/specs/edfplus.html)):

| Label | EDF+ text |
| --- | --- |
| `wake` | `Sleep stage W` |
| `nrem` | `Sleep stage N` |
| `probable_rem` | `Sleep stage R` |
| `unknown` | `Sleep stage ?` |

The map loses the NREM stage: an export never writes `Sleep stage N1`, `N2` or `N3`, also not for a hypnogram that has them, since REMind's labels do not say which. Read back (`read_hypnogram()`, `annotate-template --from-hypnogram`), W, Wake and 0 are `wake`; N, N1 to N3, S1 to S4, `Sleep stage 1` to `4` and 1 to 3 are `nrem`; R, REM and 4 are `probable_rem`; anything else, artefact and movement codes among it, is `unknown`. Integers are read as YASA and the BOAS dataset write them (0 W, 1 to 3 N1 to N3, 4 R), not as the [EDF FAQ](https://www.edfplus.info/specs/edffaq.html) codes them (4 stage 4, 5 R, 6 movement time).

Luna defines 30 s epochs from the EDF start and expects stage annotations to line up with them; others give conflicts in `HYPNO` until `EPOCH align` ([Luna walkthrough](https://github.com/remnrem/luna-walkthrough-docs/blob/main/docs/p3/hypno.md)). A synthetic night's stages and a hypnogram scored on the exported EDF line up; a hypnogram from another device usually does not.

## BIDS layout

BIDS has no sleep extension yet; the nearest proposals are BEP044 (stimuli) and BEP045 (physiological recordings) ([BEP list](https://bids.neuroimaging.io/extensions/beps.html)). The ear-EEG sleep datasets of Mikkelsen et al. 2025 ([doi:10.1038/s41597-025-04579-8](https://doi.org/10.1038/s41597-025-04579-8)) keep their stages in a second events file with `acq-scoring`, which passes the validator ([NeuroStars thread](https://neurostars.org/t/how-to-add-sleep-scorings/21965)), and so does REMind:

```text
<root>/
  dataset_description.json      Name, BIDSVersion, DatasetType raw, GeneratedBy
  participants.tsv              sub-<id>, the pseudonym given with --subject
  participants.json
  README.md
  sub-<id>/ses-<YYYYMMDD>/eeg/
    sub-<id>_ses-<date>_task-sleep_eeg.edf                  the EDF+C above
    sub-<id>_ses-<date>_task-sleep_eeg.json
    sub-<id>_ses-<date>_task-sleep_channels.tsv             EEG, ACCEL, GYRO, MISC (optics), units, rates
    sub-<id>_ses-<date>_task-sleep_events.tsv (+ .json)     cue, gate, guard, audio and recording events, fills
    sub-<id>_ses-<date>_task-sleep_acq-scoring_events.tsv (+ .json)   stages, with their REM label
```

The session is the night's UTC date unless `--session` gives one. `_eeg.json` has the required fields, `EEGReference` "Fpz (vendor-stated, unverified)", `PowerLineFrequency` "n/a" unless `--power-line-frequency` says 50 or 60, and three keys of REMind's own: `EEGFullScale` (the 1450 µV note above), `MeasuredSamplingFrequency` and `GapPolicy`. HED tags, HED-SCORE among them, are not written. A synthetic night's dataset says "synthetic input, not sleep" in its name, README, `_eeg.json` and EDF, and a root holds synthetic nights or recorded ones, never both. A session that already has files is refused.

The `bids` job of CI exports 30 min of a synthetic night and runs the BIDS project's validator on it, `bids-validator-deno` 3.0.2: no errors, and 16 warnings, all for recommended fields REMind cannot fill or never writes, such as `Authors`, `License`, `HeadCircumference` and `DeviceSerialNumber`. The validator's npm package stopped at 1.15.0 and is deprecated; it rejects any key its schema does not list, REMind's own among them.

## MNE

`to_raw(recording_dir)` returns an `mne.io.RawArray` at 256 Hz: the EEG in volts as `eeg` channels, the accelerometer, gyroscope and optics as `misc` channels interpolated onto the EEG's grid, since one `Raw` has one rate. Its annotations are the EDF's, and `meas_date` is the first frame in UTC. It refuses a sealed night like the exports.

## Reading an EDF back

`read_edf()` turns an EDF or BDF into MuseFrames of 0.25 s through a channel map: without one it takes the labels an export writes, and `boas-headband` reads the headband files of the BOAS dataset ([OpenNeuro ds005555](https://github.com/OpenNeuroDatasets/ds005555)): `HB_1` and `HB_2` as AF7 and AF8, `HB_IMU_1` to `3` as the accelerometer, `4` to `6` as the gyroscope, `HB_PULSE` as a pulse channel. EEG in mV or V becomes µV; another unit needs a factor. It leaves out what `BAD_ACQ_SKIP` covers and reports the skipped samples in the next frame's `gaps`, so an exported dropout comes back in its place. `read_edf_hypnogram()` returns the stage annotations of an EDF+ file.
