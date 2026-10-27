# N-of-1 series: cue, sham and no-cue nights

A cued-vs-uncued delta from one night mixes two things: what the cues did, and what a
night with sounds does. A series of nights in three arms separates them a little:

- `cue`: the night plays its cues as planned, like any Pilot 4 or 5 night.
- `sham`: an untrained sound plays wherever a cue would, at the same times, lengths and
  volumes. Only the sound differs.
- `no_cue`: nothing plays. The scheduler logs a `skip` with `metadata.virtual_play`
  wherever the cue arm would have played.
- `washout`: a night between blocks without a cueing run.

The one randomised, participant-blind home TLR study interleaved untrained-sound and
no-cue nights with TLR nights in this way (Konkoly et al. 2024, in
[`references.md`](references.md)). This page is the runbook for such a series with
REMind. It describes what was run, not whether cueing works: at n = 1 the experimenter
cannot be blind, a sham still makes sound, the untrained sound may be recognised, and
every output is a single-case description.

## 1. Plan and seal the series

```bash
muse-tmr plan-night-schedule --blocks 2 --washout-nights 1
```

Each block has one night of each arm. Within every group of three blocks each arm takes
each place in its block once, so no arm is always the first night. Washout nights sit
between blocks. The order comes from `--seed`, or from a seed drawn at random and kept
only in the file; the command prints neither the seed nor an arm.

It writes two files under the gitignored `data/protocol/night_schedules/`:

- `<schedule-id>.json`, the schedule: arms, seed and a random salt. Do not open it
  before the series ends.
- `<schedule-id>.sealed.json`, its sealed companion. It holds only `commitment_sha256`,
  the sha256 of the schedule's canonical JSON followed by the salt, so you can share or
  preregister it before the first night without giving an arm away.

The printed `nights=` line lists the night IDs in order (`night-001`, ...). Name each
night's puzzle session after its night ID (`data/protocol/night-001_puzzles.json` with
session ID `night-001`), so the morning files and the night's summary carry the same
name.

`--sham-cue-id` (default `sham_soft_tone`) names the untrained sound. It must be in the
night's cue library with `protocol: "sham"`, and it must not be a trained cue: not the
cue of any puzzle in the catalog, not a TLR cue, and not a generated tone of the same
frequency as one. `muse-tmr create-cue-library` writes a starter library that has
`sham_soft_tone`, a 330 Hz tone that no other starter cue uses. A runner checks the sham
cue on every night of the series, whatever its arm, so a refusal says nothing about the
arm.

`muse-tmr show-night-schedule <schedule.json>` shows the nights, their blocks and the
washout nights, and checks the schedule against its companion. While the series is
sealed it prints `sealed` for every other night.

## 2. Each night

Set up the night as Pilot 5 does (`docs/pilot5_full_night.md`): puzzles, assignment,
TLR training, calibration. Then start it with the schedule and the night's ID:

```bash
muse-tmr run-pilot5-full-night \
  --night-schedule data/protocol/night_schedules/<schedule-id>.json \
  --night-id night-001 \
  ...the usual Pilot 5 flags...
```

`run-pilot4-cueing` and `simulate-replay-cues` take the same two flags. The runner checks
the schedule against its sealed companion before it reads anything else, refuses a
washout night and a night the schedule does not hold, and exits with code 2 if the
schedule changed after it was sealed. It never prints the arm: on a night with an arm its
last line leaves out the play counts, the first play, the volumes and the scorer's
status, which would give the arm away, and says where they are.

`--arm cue|sham|no_cue` runs a night as one arm without a schedule, open label, for a
replay or a smoke test. `--night-id` then only labels the night, and `--sham-cue-id`
replaces the sham cue of either.

The arm changes what a cue becomes, never when one comes or how loud. The gate, the
arousal guard, the scheduler's timing and limits and the protocol preset are the same in
every arm, and so is the night's `protocol` block. A sham play keeps the length and the
volume hint of the cue it stands in for and plays through the same `AudioCuePlayer`,
backend and calibrated cap; a no-cue night sends nothing to the player.

Do not open the night's summary or logs before the morning forms are saved: they name the
arm (`arm` in `pilot5_summary.json`, `night_arm` in `metadata.json`, `arm` on each event
of a sham or no-cue night and each audio line of a sham night, and reasons such as
`sham_arm`).

## 3. The morning

Use the app's Morning tab or the CLI as in Pilot 5. The page never learns which puzzles
were cued or which arm the night had: it reads no schedule and no night log, and the sound
check plays every puzzle's sound in an order made from the session ID alone, the same in
every arm.

The dream report (schema 2) asks, besides lucidity and recall:

- whether a sound was heard, and where: `not_perceived`, `in_dream`, `awake`, `unsure`
  (heard, but not sure whether dreaming) or `unknown`;
- whether a sound woke the sleeper: `yes`, `no` or `unknown`;
- the times of any eye-signal attempts (left-right-left-right), as marked by the
  sleeper. Nothing detects or checks an eye signal.

```bash
muse-tmr record-dream-report data/protocol/night-001_puzzles.json \
  --output data/reports/night-001_dream_report.json \
  --lucid no --cue-perception in_dream --woke-from-sound no \
  --eye-signal-attempt 04:20 \
  --confidence 0.6 --dream-text "..."
```

## 4. Delayed retests

Effects of cueing during sleep have shown up only at a delayed test in some studies: a
week later for cueing in REM sleep (Pereira et al. 2023), ten days later for cueing in
NREM sleep (Rakowska et al. 2021). So retest the same puzzles again then. A delayed retest is blind like the morning one and saves its own
file:

```bash
muse-tmr record-puzzle-retest data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --assignment data/protocol/night-001_assignment.json \
  --wave day7 \
  --output data/reports/night-001_retest_day7.json \
  ...the results...
```

In the Morning tab pick "A week later" or "Ten days later" above the retest; it saves
`<night>_retest_day7.json` or `<night>_retest_day10.json` next to the morning's
`<night>_retest.json`.

## 5. Unsealing

After the last morning form and the last delayed retest:

```bash
muse-tmr unseal-night-schedule data/protocol/night_schedules/<schedule-id>.json
```

It checks the schedule against its sealed companion, records `unsealed_at_utc` in the
schedule (the first time only) and prints the arms. A schedule that changed after it was
sealed is not unsealed, and the command exits with code 1. A night run after the
unsealing records it in its `metadata.json` (`night_arm.schedule_unsealed_at_utc`).

`protocol-fidelity` and `score-cue-arousal` run on every arm. On a sham night they read
the sham plays like any others and add `sham_arm`; on a no-cue night they find no cues,
say `no_cue_arm`, and the fidelity summary counts the night's `virtual_plays`. Both
print the arm, so run them after the morning forms.

What does not exist yet (#228): a per-arm analysis of a series, blind coding of dream
incorporation by a second rater, and a preregistration template. `analyze-cued-uncued`
still reads one night: on a sham night it counts the sham plays as the cued puzzles'
plays, since each keeps its puzzle.

Keep everything a series writes under `data/` and out of git: the schedule, the nights,
the reports and the retests.
