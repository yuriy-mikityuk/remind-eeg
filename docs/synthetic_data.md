# Synthetic nights

`muse_tmr.testing` writes nights in the Athena TAG format from a seed: a `raw_amused.bin`
the decoders read like a recording, the files `muse-tmr record` writes next to it, and
the hypnogram the night was generated from. They exist so the path from BLE
notifications to a cue plan can be run and checked without a headband
([#233](https://github.com/yuriy-mikityuk/remind-eeg/issues/233)).

Synthetic input, not sleep. Nothing measured on these nights is evidence about sleep, REM
detection or cueing, no detector weight or threshold may be fitted to them, and no
fragment of a real night goes into them. Every JSON and HTML file and every line of
output they lead to says so; the CSV ground truth sits next to a `generation.json` that
does.

## Commands

```bash
muse-tmr demo
muse-tmr synthesize-night --list
muse-tmr synthesize-night --scenario clean_night --output data/synthetic/clean_night
```

`demo` (defaults `--scenario clean_night --hours 2 --seed 0`) generates a night, runs
`validate-pilot1-recording` on it, writes Pilot 3 inputs (four synthetic puzzles with a
generated tone each, two of them cued), runs `simulate-replay-cues` with mock audio and
the gate of the night's preset, builds the REM report, scores the gate against the
hypnogram and opens the report; `--no-open` leaves it closed. It exits 0 when Pilot 1
and Pilot 3 pass, no uncued puzzle got a `play`, the audio stayed mock and the report
was written, 1 when one of these fails and 2 on a usage error or a refused `--output`.
`--hours` sets the length of any scenario, so `battery_death_2h29` only runs out of
battery with more than 2.48 h, and then fails Pilot 1 as a night that lost its battery
should.

It writes to `data/synthetic/demo_<timestamp>/` unless `--output` says otherwise:

| Path | Content |
|---|---|
| `<scenario>_seed<seed>/` | The night, laid out as in [`data_model.md`](data_model.md#synthetic-recordings). |
| `protocol/` | `puzzle_catalog.json`, `night_puzzles.json`, `assignment.json`, `cue_library.json`. The uncued puzzles have tones in the library too, so only the assignment keeps them from playing. |
| `pilot1.json`, `pilot3_report.json` | The two validators' reports. |
| `report.html` | The REM report with a "Synthetic input, not sleep" callout. |
| `agreement.json` | Binary κ, REM recall and precision of the gate, gate-open fraction and the seconds from each REM onset to the first open gate, from `muse_tmr.validation.rem_metrics`. |
| `demo_summary.json` | All of that in short, the generated stage of the epoch that planned each cued play, the checks and the timings. |

`synthesize-night` writes the night alone. It refuses a directory that holds a
`raw_amused.bin` whose `metadata.json` does not say `synthetic: true`, so it cannot
overwrite a recording. `demo` refuses such a directory as `--output` as well, so its
reports never land next to a real night.

## What the demo shows, and what it does not

The demo and the CI `end_to_end` job check plumbing: a night in the device format
replays, Pilot 1 reads it, the detector, the gate, the arousal guard and the scheduler
run on it, no uncued puzzle gets a `play`, the report builds, and the same seed gives the
same bytes and the same decisions. They are no evidence about REM gating.

On `clean_night` the cued plays come at sleep onset. With the `p21` gate at 0.90/0.85,
84 % of the synthetic N1 epochs of the 8 h night reach P_REM 0.90, while its synthetic
REM epochs have a median P_REM of 0.89. So the gate opens on N1 a few minutes into the
night, both cued puzzles play there (decided at 690 and 720 s, on N1 epochs, in the
default 2 h demo and in the 8 h night alike), and the scheduler, which cues each puzzle
once a night, has none left for REM. The `end_to_end` check that `clean_night` gives at
least one cued `play` passes for that reason, not because REM was found. The agreement
figures say what the current detector makes of signals written to look like stages.
They move with the generator's assumptions, not with anyone's sleep.

## What is modelled

Every default is in `SyntheticNightConfig` and `script_night()` in
`src/muse_tmr/testing/synthetic.py`, with a row in [`numbers.md`](numbers.md#synthetic-nights).

- Hypnogram: 30 s epochs of W, N1, N2, N3 and R in sleep cycles of 90 to 110 min, N3
  shrinking and REM growing across the night, a brief awakening after each REM period but
  the last, arousals with and without movement.
- EEG on TP9, AF7, AF8 and TP10, in µV as the decoder returns it: a 1/f background of
  10 to 20 µV RMS, delta in N3, theta with little alpha in REM, alpha and blinks awake,
  spindles in N2, EMG at arousals, and in REM eye-movement deflections of 20 to 60 µV and
  0.3 to 1.5 s, opposite in sign on AF7 and AF8. On optics presets the line at 63.93 Hz,
  20 to 40 µV on AF7.
- IMU: gravity for three head positions, sensor noise, a burst at every movement.
- Optics, `p1034` only: a pulse at a heart rate set per stage, faster after arousals.
- Battery: falling at the preset's measured drain until it is empty; the stream ends there.
- BLE: notifications of at most 244 bytes, each carrying the subpackets made since the one
  before, subpackets lost and notifications repeated at the rates of #160, short gaps.

## Where the parameters come from

The device side comes from the repository's own measurements: the battery drain and the
2 h 29 min death from [#108](https://github.com/yuriy-mikityuk/remind-eeg/issues/108),
the notification rate from the completed night of #108 (835,926 notifications over
7.998 h), the lost and repeated subpackets from
[#160](https://github.com/yuriy-mikityuk/remind-eeg/issues/160), the optics line from
[#154](https://github.com/yuriy-mikityuk/remind-eeg/issues/154).

The stage signals are textbook shapes, set by hand: an assumption, not a measurement.
The eye-movement deflections share their shape with the fixture of
[#225](https://github.com/yuriy-mikityuk/remind-eeg/issues/225). On a forehead montage they
are an assumption twice over. On ZMax, whose F7 and F8 are referenced to Fpz much as the
Muse's AF7 and AF8 are, an automatic detector's REM eye-movement counts correlated
r = 0.66 with its counts on PSG EOG, and it found about 1.8 times as many on the
headband, on average and with a wide spread (Jafarzadeh Esfahani et al. 2023, a preprint,
in [`references.md`](references.md)). A real Athena night may look nothing like these
deflections.

## Formats and sizes

The files are described in [`data_model.md`](data_model.md#synthetic-recordings). Seed 0,
each scenario at its own length, one run each on an M4 Pro with other work running (load
average 5 to 10 on 14 cores), so the seconds are rough. Generation is `generate_night()`,
replay a `ReplaySession` at speed 0 over every frame, and `demo` the whole command, which
replays the night twice, for Pilot 3 and for the report:

| Scenario | Preset | Length | `raw_amused.bin`, MB | Generation, s | Replay, s | `demo`, s |
|---|---|---|---|---|---|---|
| `clean_night` | `p21` | 8 h | 96.5 | 3.9 | 50 | 171 |
| `fragmented_rem` | `p21` | 8 h | 96.5 | 3.6 | 47 | 152 |
| `no_rem_nap` | `p1034` | 1 h | 17.2 | 0.5 | 14 | 35 |
| `battery_death_2h29` | `p1034` | 8 h, empty after 2 h 29 min | 42.8 | 1.6 | 36 | 88 |
| `optics_line` | `p1034` | 2 h | 34.5 | 1.0 | 29 | 71 |
| `bad_contact_af7` | `p21` | 2 h | 24.1 | 0.9 | 12 | 32 |
| `movement_heavy` | `p21` | 2 h | 24.1 | 0.8 | 12 | 33 |
| `short_allow_short` | `p1034` | 10 min | 2.9 | 0.1 | 2 | 6 |

On all eight the decoder's lost and repeated subpackets per sensor type equal what
`generation.json` scripted, and no uncued puzzle gets a `play`. The default `demo`, 2 h
of `clean_night`, took 39 s.

The corpus is generated on demand and never committed: `data/synthetic/` is gitignored
and `scripts/check_forbidden_files.py` refuses anything under it. There is no pre-rendered
archive on a release. The same seed gives the same bytes, which two tests pin: the
sha256 of one minute of `short_allow_short` in `tests/test_synthetic_night.py`, and the
sha256 of the 2 h demo night in `tests/fixtures/synthetic/agreement_baseline.json`.

`short_allow_short` is 10 min and 2.9 MB. #233 hoped for under 1 MB, which ten minutes at
the device's rates cannot give: the EEG payload alone is 64 subpackets a second × 28 bytes
× 600 s = 1.08 MB. Its test bounds it at 3.5 MB and 2 s.

## The CI check

The `end_to_end` job runs

```bash
muse-tmr demo --hours 2 --no-open --baseline tests/fixtures/synthetic/agreement_baseline.json
```

and fails when Pilot 1 or Pilot 3 fails, an uncued puzzle plays, no cued puzzle plays,
the audio is not mock, the report is missing, or the run differs from the baseline: the
sha256 of `raw_amused.bin` and the Pilot 3 counts exactly, κ, recall, precision,
gate-open fraction and REM-onset latency within 0.01. The `package` job runs
`muse-tmr demo --hours 0.25 --no-open` from the built wheel. A change to the generator,
the detector, the gate, the guard or the scheduler that moves these numbers on purpose
renews the baseline in the same commit: run the command without `--baseline` and copy
`raw_sha256`, `agreement` and `pilot3` from its `demo_summary.json`.

## Current limitations

- The decoder keeps only the last ACCGYRO subpacket of a notification
  ([#244](https://github.com/yuriy-mikityuk/remind-eeg/issues/244)). The device does put
  more than one in a notification (34 of the 120 notifications of the committed capture
  `tests/test_data/muse_20250824_172756.bin`), but the generator never does, and
  `SyntheticNightConfig.validate()` refuses a notification rate that would need it, so the
  synthetic nights cannot catch that loss until #244 is fixed.
- No cue plays on the four `p1034` scenarios. The arousal guard pauses every epoch for
  `sudden_heart_rate_change`: the heart rate the decoder derives from the synthetic
  pulse jumps by about 35 bpm within an epoch (the median of the per-epoch maximum),
  where 10 bpm pauses. Whether the pulse model or the decoder's beat detection is at
  fault is not known yet. On optics presets the corpus checks the guard's pauses and the
  cued-only contract, not a play.
- Not modelled: DRL/REF telemetry, the device's own sample clock (#160 counted about
  257.2 EEG samples a second), sawtooth waves and K-complexes, dropouts long enough for a
  reconnect, fNIRS. No `decoded_frames.jsonl` is written, and `summary.json` leaves out
  the heart-rate frames the decoder derives from PPG during replay.
- The stage signals are one author's assumptions, and every figure on this page comes
  from one seed.

## For other decoders

The nights are in the device's own TAG format, and `generation.json` says what a decoder
should get out of each one: per sensor type the subpackets produced, delivered, lost and
repeated and the sample rows, and the sha256 of every file. That makes them usable as a
test oracle for other Athena decoders, such as OpenMuse, BrainFlow or muse-rs, as long as
the stage content is not read as sleep. Maintainers who want a fixed set of seeds for
that can open an issue here.
