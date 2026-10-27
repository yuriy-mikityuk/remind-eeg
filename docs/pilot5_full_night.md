# M8 Pilot 5: Full-Night TLR and Puzzle Cueing

Pilot 5 is the first full protocol night. It keeps Pilot 4 safety controls, adds a
required TLR block, and then allows REM-gated puzzle cues from the cued half of the
night puzzle session.

## Goal

- Run a normal overnight Muse recording with epochs, REM detection, stable gate,
  arousal guard, scheduler, and audio playback logs.
- Present the TLR block after the first stable REM gate opening.
- Present only cued puzzle cues after the TLR block and scheduler interval rules.
- Preserve uncued puzzles as controls and verify they never receive scheduler `play`
  events.
- Generate the morning dream report, blind puzzle retest, and cued-vs-uncued analysis.

Pilot 5 is exploratory. Treat results as a protocol-fidelity and descriptive analysis
artifact, not evidence of efficacy.

## Preflight

Required before using `--backend system`:

- Pilot 2 volume calibration exists and passed validation.
- Pilot 3 replay simulation passed on a recent recording.
- Pilot 4 dry-run or live low-volume cueing smoke completed without permanent safety
  stops from ordinary recoverable artifacts.
- Puzzle catalog has at least four eligible unsolved puzzles.
- Night session and cue assignment exist, with cued and uncued puzzles randomized.
- Cue library contains the puzzle cue IDs and the TLR cue the block was planned with,
  under the same ID and with the same tone: training, the plan and the night use one
  library.
- TLR training was run before sleep, and a TLR block plan exists.
- macOS output device is set to the intended sleep headphones.
- You know the emergency stop path for the run.

## Prepare Protocol Files

```bash
cd /path/to/remind-eeg

.venv/bin/python -m muse_tmr.cli.main create-cue-library \
  --output data/cues/starter.json

.venv/bin/python -m muse_tmr.cli.main train-tlr-cue data/cues/starter.json \
  --output data/protocol/night-001_tlr_training.json \
  --event-log data/protocol/night-001_tlr_training.jsonl \
  --backend dry-run \
  --repetitions 3

.venv/bin/python -m muse_tmr.cli.main plan-tlr-block data/cues/starter.json \
  --output data/protocol/night-001_tlr_block.json \
  --training data/protocol/night-001_tlr_training.json \
  --repetitions 3 \
  --interval-seconds 8 \
  --post-block-pause-seconds 10
```

Use `--backend system` for `train-tlr-cue` only if you intentionally want audible
pre-sleep TLR familiarization. Without `afplay` on PATH it then exits with code 2
instead of training in silence.

To run a night with a protocol preset (`docs/protocol_presets.md`), give the same
`--protocol-preset` to `create-cue-library`, `plan-tlr-block`, `generate-puzzle-session`,
`assign-puzzle-cues` and `run-pilot5-full-night`, and leave out the flags it sets, for
example the `--repetitions`, `--interval-seconds` and `--post-block-pause-seconds` above.
`create-cue-library` then puts the preset's TLR tone in `data/cues/starter.json` under
`tlr_soft_tone`, the same cue `create-tlr-cue --protocol-preset` makes, so training, the
plan and the night all read that one library. Without the flag the library keeps the
starter's tone, and `plan-tlr-block --protocol-preset` warns when that is not the
preset's. Each file records the preset it was made with; a night whose plan or
assignment was made with another preset says so in a warning and in `metadata.json`.

The night plays the TLR cue of its own `--cue-library`, whatever library the plan was
made from. So the plan records the tone it was made with (`tone`), and
`run-pilot5-full-night` exits with code 2, before it connects or writes a file, when the
night's library has another tone under the plan's cue ID; for a plan made before it kept
its tone, when the cue's length differs from the plan's. The error says how to fix it.
`--training` makes `plan-tlr-block` refuse in the same way when the library's cue is not
the tone the training played, and records the training in the plan. Before this check
(#264) a TLR cue made by `create-tlr-cue --protocol-preset` into a file of its own was
trained and planned, while the night played the starter library's `tlr_soft_tone`.

## Dry-Run Smoke

```bash
OUTDIR="data/recordings/pilot5_dry_run_$(date +%Y%m%d_%H%M%S)"

.venv/bin/python -m muse_tmr.cli.main run-pilot5-full-night \
  --source amused \
  --address "$MUSE_ADDR" \
  --duration-seconds 300 \
  --allow-short \
  --output-dir "$OUTDIR" \
  --catalog data/protocol/puzzle_catalog.json \
  --session data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --tlr-block data/protocol/night-001_tlr_block.json \
  --calibration data/calibration/volume_calibration.json \
  --device-name "Sleep Headphones" \
  --backend dry-run
```

The dry-run should produce `pilot5_summary.json`, `scheduler_events.jsonl`,
`arousal_guard_events.jsonl`, and `audio_playback.jsonl`. The summary should include
`tlr_block_required=true`, `tlr_cue_play_count`, `puzzle_cue_play_count`, and the
`tlr_block_played` criterion.

## Run With Audio

Use real audio only when ready for a sleep run:

```bash
OUTDIR="data/recordings/pilot5_full_night_$(date +%Y%m%d_%H%M%S)"

.venv/bin/python -m muse_tmr.cli.main run-pilot5-full-night \
  --source amused \
  --address "$MUSE_ADDR" \
  --duration-hours 8 \
  --output-dir "$OUTDIR" \
  --catalog data/protocol/puzzle_catalog.json \
  --session data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --tlr-block data/protocol/night-001_tlr_block.json \
  --calibration data/calibration/volume_calibration.json \
  --device-name "Sleep Headphones" \
  --backend system \
  --default-volume 0.02 \
  --hard-max-volume 0.20
```

The default duration rules require 2-8 hours unless `--allow-short` is present.
`--preset` sets the headband preset and the REM gate thresholds, as in Pilot 4. Also as
in Pilot 4, the command exits with code 2 before it connects or creates `$OUTDIR` when
`afplay` is not on PATH, `--allow-dry-run` runs the night silently instead, and
`pilot5_summary.json` names the requested and the resolved backend.

The arousal guard is Pilot 4's. On `p1034` an epoch whose heart-rate series skipped more
than half of its values is `hr_rule_blind` and lowers the volume of the TLR and puzzle
cues played in it (#234): their play events in `scheduler_events.jsonl` and their lines
in `audio_playback.jsonl` carry `volume_multiplier` 0.5 and an `effective_volume` of half
the `capped_volume`, the volume after the calibration and hard caps (#255). A pause or
stop reason in the same epoch still wins.

## A Night Of A Sealed Series

A night of an N-of-1 series (`docs/nof1_protocol.md`) adds the schedule and the night's
ID to the command above:

```bash
  --night-schedule data/protocol/night_schedules/<schedule-id>.json \
  --night-id night-001
```

The night then runs the arm the sealed schedule gives it: `cue` as above, `sham` with the
untrained sound in place of every TLR and puzzle cue, or `no_cue` with nothing played.
The command never prints the arm. Leave `pilot5_summary.json`, `metadata.json`, the logs
and the `cue_arousal_report.*` and `protocol_fidelity.*` files closed, and do not run
`score-cue-arousal` or `protocol-fidelity` on the night, until the last delayed retest of
the series is saved and `unseal-night-schedule` has run. They name the arm, and the arms
of earlier nights narrow down the later ones (`docs/nof1_protocol.md`). On a no-cue night
`tlr_block_played` passes on the block's virtual plays.

## Emergency Stop

Creating the emergency stop file blocks future playback while the recording continues:

```bash
touch "$OUTDIR/EMERGENCY_STOP"
```

## Morning Workflow

Log awakenings or cue recall notes, with `--attributed-to-cue yes|no|unsure` when you
have a view on whether a cue woke you:

```bash
.venv/bin/python -m muse_tmr.cli.main log-pilot4-awakening \
  "$OUTDIR/awakening_events.jsonl" \
  --notes "woke briefly; cue recall unclear"
```

As in Pilot 4, the night ends by scoring what followed each TLR and puzzle cue against a
paired quiet control (`docs/safety_outcomes.md`) and puts a `cue_arousal` block in
`pilot5_summary.json`. The cue index table separates the TLR cues (`tlr 1`, `tlr 2`,
...) from the puzzle cues. After logging awakenings, score it again so the
cue-attributed awakening rate in the report and in that block counts them (on a night of
a sealed series, only after unsealing):

```bash
.venv/bin/python -m muse_tmr.cli.main score-cue-arousal "$OUTDIR"
```

In the morning, the app's Morning tab does the next three steps without flags: pick the
night, write the dream report, then the sound check (play each sound and pick its
puzzle), then the blind retest with a timer per puzzle; above the retest, pick "A week
later" or "Ten days later" for a delayed one. The night's logs name the cued puzzles, so
open them only after the last retest you plan, a delayed one included. The tab finds the
night's catalog and assignment in `data/protocol`, the cues in `data/cues` and the
calibration in `data/calibration`, and saves to the same
`data/reports/<night>_dream_report.json`, `<night>_cue_recall.json` and
`<night>_retest.json` as the commands below. Its sounds
play on `afplay`; start the app with `--cue-backend dry-run` (or `mock`) to go through
the step without sound. `--allow-dry-run` only matters where `afplay` is missing: it
then plays the sounds silently instead of refusing them. From the terminal:

Record dream report:

```bash
.venv/bin/python -m muse_tmr.cli.main record-dream-report \
  data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --output data/reports/night-001_dream_report.json \
  --lucid no \
  --cue-perception not_perceived \
  --woke-from-sound no \
  --confidence 0.5 \
  --dream-text ""
```

`--cue-perception` says whether a sound was heard and where (`not_perceived`,
`in_dream`, `awake`, `unsure`, `unknown`), `--woke-from-sound` whether one woke you, and
`--eye-signal-attempt 04:20`, repeated, marks when you think you made the eye signal.
`--cues-heard yes|no` still works and is read as `unsure` or `not_perceived`.

Play each puzzle's sound and say which puzzle it belongs to, before the retest:

```bash
.venv/bin/python -m muse_tmr.cli.main record-cue-recall \
  data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --calibration data/calibration/volume_calibration.json \
  --output data/reports/night-001_cue_recall.json \
  --backend system
```

The sounds come in an order that does not depend on which puzzles were cued, and the
command never prints which answers were right.

Record the blind morning puzzle retest before looking at cue conditions:

```bash
.venv/bin/python -m muse_tmr.cli.main record-puzzle-retest \
  data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --assignment data/protocol/night-001_assignment.json \
  --output data/reports/night-001_retest.json \
  --result "p1=" \
  --result "p2=" \
  --result "p3=" \
  --result "p4=" \
  --duration "p1=30" \
  --duration "p2=30" \
  --duration "p3=30" \
  --duration "p4=30" \
  --confidence "p1=0.0" \
  --confidence "p2=0.0" \
  --confidence "p3=0.0" \
  --confidence "p4=0.0"
```

A week and ten days later, retest the same puzzles again with `--wave day7` and
`--wave day10`, saved as `data/reports/night-001_retest_day7.json` and
`night-001_retest_day10.json` (the Morning tab's wave selector writes the same files).

Generate analysis:

```bash
.venv/bin/python -m muse_tmr.cli.main analyze-cued-uncued \
  data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --retest data/reports/night-001_retest.json \
  --dream-report data/reports/night-001_dream_report.json \
  --scheduler-events "$OUTDIR/scheduler_events.jsonl" \
  --cue-recall data/reports/night-001_cue_recall.json \
  --protocol-metadata "$OUTDIR/metadata.json" \
  --output data/reports/night-001_analysis.json \
  --markdown-output data/reports/night-001_analysis.md
```

Compare the night with its protocol preset:

```bash
.venv/bin/python -m muse_tmr.cli.main protocol-fidelity "$OUTDIR"
```

Its `tlr_block` check shows how each TLR block played against the plan: at a 30 s stride
cues planned closer than 30 s play back to back, and after an arousal guard pause the
scheduler plays the block again in the same REM period. Its `tlr_tone` check compares the
plan's tone, the one the night played, with the preset's.

## Outputs

The output directory contains:

- `raw_amused.bin`, `metadata.json`, `events.jsonl`;
- `scheduler_events.jsonl`;
- `arousal_guard_events.jsonl`;
- `rem_gate_decisions.jsonl`;
- `audio_playback.jsonl`;
- `awakening_events.jsonl`;
- `cue_arousal_events.jsonl`, `cue_arousal_report.json`, `cue_arousal_report.md`;
- `pilot5_summary.json`;
- `protocol_fidelity.json`, `protocol_fidelity.md` once `protocol-fidelity` ran on it.

The summary passes only when the Pilot 4 safety criteria pass and at least one TLR
block cue was logged as a scheduler `play` event. A night with no puzzle cues can still
be valid if REM gate or safety conditions did not allow puzzle cueing after the TLR
block.
