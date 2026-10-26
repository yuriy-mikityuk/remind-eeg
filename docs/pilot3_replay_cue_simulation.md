# M8 Pilot 3: Replay Cue Simulation

Pilot 3 validates the REM gate, arousal guard, and TMR scheduler on an existing
recording without playing real audio. It produces an inspectable cue plan and
scheduler event stream from replayed Muse frames.

## Goal

- Replay a completed recording through 30-second epochs and feature extraction.
- Run heuristic REM detection, `StableRemGate`, `ArousalGuard`, and
  `TmrCueScheduler`.
- Save a JSON report containing epochs, gate decisions, guard decisions,
  scheduler events, and `cue_plan`.
- Prove this step uses mocked audio only: `audio_playback_executed=false`.
- Confirm zero uncued puzzle `play` events.

This pilot is not sleep-time cueing. It does not call an audio backend and does not
test whether cues are audible or effective.

To try it without a recording, `muse-tmr demo` generates a synthetic night, writes the
protocol files below for four synthetic puzzles with generated tones and runs this
command on them with the night's preset (`p21` for the default `clean_night`).
Synthetic input, not sleep: on that night the gate opens on the N1 epochs at sleep
onset, so both cued puzzles play there and not in REM (`docs/synthetic_data.md`).

## Run

Use a previous no-audio recording, a puzzle catalog, a generated night puzzle
session, a cued/uncued assignment, and a cue library:

```bash
cd /path/to/remind-eeg

.venv/bin/python -m muse_tmr.cli.main simulate-replay-cues \
  data/recordings/<pilot-session> \
  --catalog data/protocol/puzzle_catalog.json \
  --session data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --preset p21 \
  --output data/reports/pilot3_replay_cue_plan.json \
  --scheduler-events-output data/reports/pilot3_scheduler_events.jsonl
```

Pass the preset the night was recorded with; it is in the recording's `metadata.json`
under `source.metadata.preset` (the app's night sessions use `p21`; `record` and the
Pilot 4 and 5 commands default to `p1034`). It sets the gate thresholds, `0.90`/`0.85`
for `p21` and `0.80`/`0.70` for an optics preset, and on `p21` keeps the missing PPG
from capping REM confidence. On `p21` the arousal guard also stops treating the missing
PPG as an artifact and skips its HR rules, and each guard decision carries
`ppg_absent_by_design`. Without that, a `p21` replay is paused with
`repeated_artifact_quality` from the third epoch on and plans no cues. Without
`--preset` the replay assumes an optics preset.

On an optics preset the guard's `sudden_heart_rate_change` reads the heart-rate series
each epoch's optics give (one value a second from 10 s windows), not the decoder's
`heart_rate` stream, whose beat-in, beat-out jumps used to pause epochs where nothing
changed (#247). A pause for it records in the guard metadata how many of the epoch's
values were used and skipped (`hr_series_used_count`, `hr_series_skipped_count`) and the
largest change from the previous epoch's values into this one's
(`sudden_hr_change_across_epochs_bpm`); an `allow` decision records the two counts when
a value was skipped. When more than half of them were, the decision says `hr_rule_blind`
and lowers the volume, so a cue planned in that epoch carries half its volume hint (#234).
`docs/data_model.md` has the details.

`--guard-min-ppg-confidence` and `--guard-min-hr-series-quality` (default 0, off) turn
the guard's HR rules off on epochs whose `ppg_confidence` or series median quality is
below the value, with `hr_rule_skipped_by` in the decision. Replaying a night with and
without one shows what it would change in the cue plan; which value to use, if any, is
for a paired optics and ECG table to say (`docs/optics_agreement.md`).

Optional replay and gate tuning flags:

```bash
  --start-seconds 1800 \
  --end-seconds 7200 \
  --epoch-seconds 30 \
  --stride-seconds 30 \
  --min-stable-seconds 60
```

`--enter-threshold` and `--exit-threshold` override the preset's pair. The old
`0.70`/`0.45` kept the gate open about three quarters of a night (#115).

## Stride and first-cue latency

`--stride-seconds` sets how far apart the 30-second epochs start. Below 30 s the epochs
overlap, and the gate is fed the time between epoch starts, so its 60 s stable window
and its 120 s cooldown take as long as at 30 s (#225). At 30 s nothing changed: a replay
gives the same scheduler events as before.

The report's `metrics` carry `stride_seconds`, `first_gate_open_seconds` and
`first_play_seconds`. The last two count from the start of the first epoch to the end of
the epoch whose decision first opened the gate or first planned a cue, which is when
that decision exists; the scheduler events and `epochs[].timestamp_seconds` give the
epoch start, 30 s earlier. They are `null` when it never happened.

On a scripted stream, synthetic input, not sleep (a 40 µV 1.5 Hz wave, then 6 Hz theta
with eye-movement deflections, a still IMU, no optics, the `p21` gate 0.90/0.85), the
first gate opening, which was also the first planned cue, came this many seconds after
REM onset, for onsets 0, 5, 10, 15, 20 and 25 s past an epoch boundary:

| Stride | Seconds after REM onset |
|---|---|
| 30 s | 60 to 85 (60 with the onset on an epoch boundary) |
| 10 s | 80 to 85 |
| 5 s | 85 |

With the gate as it is, a shorter stride does not bring the first cue earlier. A window
that was still partly NREM did not look like REM, so the first REM-like window ended at
least about 30 s after onset at every stride, and at a 30 s stride that window counts
30 s towards the 60 s, at 10 s only 10 s. What the shorter stride removes is the spread
that comes from where the onset falls in the epoch grid. The table comes from
`tests/test_first_cue_latency.py`. For a real night, run the command above with
`--preset p21` at `--stride-seconds 30` and `10` and compare the two metrics.

## Validate

The command exits `0` only when:

- replay produced at least one epoch;
- no real audio playback was executed;
- scheduler events were generated for inspection;
- no uncued puzzle received a `play` scheduler event.

Inspect `data/reports/pilot3_replay_cue_plan.json`:

- `cue_plan`: scheduler `play` events only;
- `scheduler_events`: all `play`, `skip`, `pause`, and `stop` decisions;
- `epochs`: per-epoch prediction, gate, guard, and scheduler context;
- `metrics.stride_seconds`, `metrics.first_gate_open_seconds`,
  `metrics.first_play_seconds`: the stride and the first-cue timing above;
- `metrics.uncued_puzzle_play_count`: must be `0`;
- `audio_backend`: must be `mock`;
- `audio_playback_executed`: must be `false`.

If `cue_plan` is empty but validation passes, the replay did not meet stable REM and
safety conditions during the selected time range. That is an acceptable Pilot 3
outcome; use scheduler `skip` reasons and gate decisions to understand why.

Keep generated reports under `data/reports/` local unless you intentionally sanitize
and share them.
