# Architecture

The project keeps the existing amused-py BLE implementation as the low-level Muse S Athena data source and adds a separate `muse_tmr` layer for REM-TMR/TLR workflows.

## Layers

1. Source adapters: convert live or external streams into common Muse frames.
2. Data layer: recording, metadata sidecars, and offline replay.
3. Feature layer: sleep epochs and EEG/IMU/PPG/HR features.
4. REM layer: heuristic and personal REM predictions.
5. Gate and safety layer: stable REM checks, motion/arousal checks, cooldowns, and emergency stop.
6. Protocol layer: puzzle assignment, TLR cues, TMR scheduler, and session state.
7. Report layer: dream report, puzzle retest, and cued-vs-uncued analysis.

The M0 scaffold intentionally avoids moving the existing top-level `muse_*.py` modules. Later issues can introduce adapters one component at a time.

## M1 Source And Recording Flow

`BaseMuseSource` defines the source contract: discover devices, connect, stream `MuseFrame` objects, and stop. `AmusedSource` adapts the existing `MuseStreamClient` callback model into that contract.

`OpenMuseLslSource` is the optional M7 source. OpenMuse runs outside this process and
publishes Lab Streaming Layer streams; the adapter resolves stream names such as
`Muse_EEG` and `Muse_ACCGYRO`, converts pulled samples into `MuseFrame` objects, and
does not make OpenMuse or LSL packages mandatory for normal installation.

`BrainFlowSource` is the optional BrainFlow acquisition source. It keeps BrainFlow as
an adapter behind `BaseMuseSource`, targets `MUSE_S_ATHENA_BOARD`, and converts
BrainFlow EEG, IMU, optics, and battery batches into the same `MuseFrame` contract.
The direct amused-py BLE source remains the default until live smoke tests prove a
BrainFlow path is more reliable for Muse S Athena.

`MuseSdkSourceStub` reserves the future official SDK source interface without importing
or bundling any proprietary SDK files. Runtime SDK operations fail with a policy error
until a local-only integration is explicitly implemented.

`OvernightRecorder` consumes any `BaseMuseSource`, writes `raw_amused.bin`, `metadata.json`, `events.jsonl`, and `summary.json`, and uses `RecordingWatchdog` to detect no-data timeouts and modality dropouts. While it runs it rewrites `progress.json` about every 2 s with the contact quality and source diagnostics.

## Polar H10

With `--with-polar` the recorder starts `record-polar` as a child process. `muse_tmr.sources.polar_h10` and `muse_tmr.data.polar_recorder` write the raw BLE notifications and the decoded HR/RR, ECG and chest ACC to `<recording>/polar/`, and a strap failure is only logged, never stops the Muse recording. `muse_tmr.data.polar_session` puts the data on the Muse time base and `muse_tmr.features.cardio_resp_features` derives RR/HRV, breathing rate and RSA. None of it feeds REM detection, the gate, the arousal guard or the scheduler.

## Local App

`muse_tmr.app.server` serves the page in `muse_tmr/app/static/` (the Record, Morning and Recordings tabs) and a JSON API on `127.0.0.1:8765`. Its own BLE connection only feeds the contact view (`muse_tmr.contact`), and a start goes through the armed `ContactGate`: the server refuses it until all four channels have been good for the stability window. To record, it drops that connection, since the Muse takes one, and starts `caffeinate -s python -m muse_tmr.cli.main record ...` as a separate process group writing to `data/recordings/{session,night}/<timestamp>/`, so the recording outlives the app. During the recording the page shows what the recorder writes to `progress.json`. `launch.json` keeps the recorder's PID and command, so after a restart the app can still tell whether a folder is being written. Report builds, meditation analyses and the calibration guide run as further background processes with their logs next to the recording. The recordings list reads the EEG rate and lost subpackets from each folder's `progress.json`, and `muse_tmr.app.morning` saves a puzzle night's dream report and blind retest through the same builders as the CLI without telling the page which puzzles were cued. `muse_tmr.app.auto_update` checks `origin/main` every two minutes, fast-forwards a clean `main` checkout and restarts the app once it is idle. The page reloads when it sees the new build, except while the Morning tab holds unsaved answers, and shows a banner when the server stops answering.

## Reports

- `muse_tmr.reports.nightly_report`, run by `scripts/generate_nightly_report.py` and `muse-tmr demo`: the REM report of a night or session as `data/reports/<kind>/<name>.html`, with a heart and breathing section from `muse_tmr.reports.night_cardio` when the recording has `polar/` and a "Synthetic input, not sleep" callout when its `metadata.json` says `synthetic: true`.
- `muse_tmr.reports.meditation_analysis` and `muse_tmr.reports.meditation_report`: cut a recording into the blocks of its `blocks.json`, compute the metrics of `muse_tmr.features.complexity_features` per epoch, compare conditions and write `data/reports/meditation/<recording>/`; `aggregate-meditation` pools the sessions of one series.
- `muse_tmr.protocol.calibration` speaks the calibration run and logs when each cue was spoken; `muse_tmr.reports.calibration_report` cuts its blocks at those logged times, and `muse_tmr.reports.line_check` measures the 64 Hz line for `--line-check`.
- `muse_tmr.reports.dream_report`, `muse_tmr.reports.morning_retest` and `muse_tmr.reports.analysis`: the morning side of the TMR protocol.

The three HTML reports share `muse_tmr.reports.report_style`: the app's colours and type, a dark mode and the REMind line, with nothing loaded from the network.

`scripts/validate_rem_detection.py` sits outside the package: it replays a recording through the detector and the gate and scores them against an automated sleep stager (YASA or GSSC) run in its own venv.
