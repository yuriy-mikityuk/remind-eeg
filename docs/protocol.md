# Protocol Notes

The project aims to preserve key REM-TMR/TLR protocol controls:

- pre-sleep puzzle assignment
- cued-vs-uncued randomization
- unique cue metadata per puzzle
- optional TLR block before puzzle cues
- REM-gated cue playback
- arousal and motion guardrails
- morning dream report and puzzle retest
- explicit analysis of limitations

Cue metadata lives in `muse_tmr.audio.cue_library` and should be validated before a
session starts. Protocol layers should reference cue IDs from a validated catalog
rather than hard-coded file paths.

The cue protocol is one-way and REM-timed: a cue is paired with a puzzle or a lucidity
intention before sleep and replayed when the stable REM gate opens. Nothing checks for
eye signals (no LRLR verification), so this is not the two-way communication protocol
of Konkoly et al. 2021; the papers it draws on are listed in
[`references.md`](references.md).

Protocol settings are the dataclass defaults in code, versioned with the release
(`muse_tmr.__version__`); no YAML file is loaded:

- REM gate: `RemGateConfig` in `src/muse_tmr/models/rem_gate.py`, with the per-preset
  enter/exit pair from `preset_gate_thresholds()` in `src/muse_tmr/presets.py`
- arousal guard: `ArousalGuardConfig` in `src/muse_tmr/protocol/arousal_guard.py`
- scheduler: `TmrSchedulerConfig` in `src/muse_tmr/protocol/tmr_scheduler.py`
- TLR cue, training and block: `TlrCueConfig`, `TlrTrainingConfig` and `TlrBlockConfig`
  in `src/muse_tmr/protocol/tlr_protocol.py`
- audio caps and fades: `AudioPlaybackConfig` in `src/muse_tmr/audio/audio_player.py`
- epochs: `EpochConfig` in `src/muse_tmr/features/epochs.py`
- puzzle session size and cued split: `src/muse_tmr/protocol/puzzle_protocol.py` and
  `src/muse_tmr/protocol/randomization.py`

[`numbers.md`](numbers.md) lists each default with its source and status.

## Pre-Sleep Puzzle Sessions

Puzzle session management lives in `muse_tmr.protocol.puzzle_protocol`. It is the M5
foundation for later cued-vs-uncued randomization and REM-gated scheduling.

The puzzle catalog stores:

- puzzle ID, prompt, solution, and cue ID
- solved, known, and retired flags
- tags/source metadata
- timed pre-sleep attempts

Night sessions are generated from eligible unsolved tasks. By default, eligibility
excludes solved, known, and retired tasks, then selects four tasks. Passing a seed makes
selection reproducible.

## Cued Vs Uncued Randomization

`muse_tmr.protocol.randomization` assigns the generated night-session puzzle IDs into
cued and uncued groups. The default split cues half of the session tasks, uses a seed
for reproducibility, and saves a versioned JSON assignment.

The assignment exposes `scheduled_puzzle_ids`, which is intentionally identical to the
cued group. Scheduler code must use this field or `scheduled_cue_ids()` rather than
iterating over every puzzle in the night session. Calling `ensure_schedulable()` on an
uncued puzzle raises an error, making accidental scheduling of control tasks a tested
contract.

## TLR Cue Module

Targeted lucidity reactivation support lives in `muse_tmr.protocol.tlr_protocol`.
The module provides three pieces for the later REM-gated scheduler:

- a default generated TLR cue metadata library
- a pre-sleep training routine that plays or dry-runs repeated TLR cues and writes JSONL events
- a configurable REM TLR block plan to run before puzzle cues

`muse-tmr create-tlr-cue` creates the default generated cue. `muse-tmr train-tlr-cue`
uses `AudioCuePlayer` with a selected backend and records one `tlr_training_cue` event
per repetition. `muse-tmr plan-tlr-block` records cue offsets and exposes
`puzzle_cue_start_offset_seconds`, so scheduler code can place puzzle cues after the
TLR block and its post-block pause.

## REM-Gated TMR Scheduler

`muse_tmr.protocol.tmr_scheduler.TmrCueScheduler` is the deterministic scheduling layer
between `StableRemGate` and audio playback. It consumes replayable `RemGateDecision`
objects, never raw REM probabilities, and emits structured scheduler events rather than
playing sounds directly.

On a stable open REM gate, the scheduler:

- emits the optional TLR block first
- waits until the TLR block's `puzzle_cue_start_offset_seconds`
- schedules puzzle cues only from `PuzzleCueAssignment.scheduled_puzzle_ids`
- enforces cue interval, cooldown, and max puzzle cues per REM block

Closed gates produce `skip` events until a block has started. If the gate closes during
an active block, the scheduler emits `pause` and starts cooldown. `stop()` emits a
`stop` event and future updates emit `skip` with `scheduler_stopped`.

Event logs are JSONL records with event type, timestamp, cue ID, protocol, puzzle ID,
reason codes, and metadata. Tests replay synthetic REM gate decisions to verify TLR
ordering, puzzle-cue intervals, cooldown, max-per-block behavior, and the rule that
uncued controls are never scheduled.

## Arousal Guard

`muse_tmr.protocol.arousal_guard.ArousalGuard` is the cue safety layer between feature
extraction and `TmrCueScheduler`. It consumes `EEGFeatureRow`, `IMUFeatureRow`, and
`PPGFeatureRow` objects and emits an `ArousalGuardDecision` with one action:

- `allow`
- `lower_volume`
- `pause`
- `stop`

The conservative default watches motion/arousal guard reason codes, stillness, relative
alpha power, sudden HR changes, out-of-range HR, and artifact/coverage flags. Mild alpha,
artifact-quality concerns and, on optics presets, an epoch whose heart-rate series is too
noisy for the HR rule (`hr_rule_blind`, #234) lower volume. Motion arousal, low
stillness, strong alpha, HR jumps, and out-of-range HR pause cueing. Repeated pause or
artifact epochs stop the session plan. Decisions can be appended to JSONL for replay analysis and passed into
`TmrCueScheduler.update(..., guard_decision=decision)`.

## Morning Dream Report

`muse_tmr.reports.dream_report` captures the morning self-report for the generated night
session. A report records lucid yes/no, cues heard yes/no, self-report confidence,
free-text recall, and optional puzzle incorporation links. Puzzle links are validated
against `NightPuzzleSession.puzzle_ids`; with a `PuzzleCatalog`, links also include the
stable cue ID for analysis.

The CLI command is:

```bash
muse-tmr record-dream-report data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --output data/reports/night-001_dream_report.json \
  --lucid no \
  --cues-heard yes \
  --confidence 0.6 \
  --dream-text "Free-text recall" \
  --puzzle-link "p2=the second puzzle appeared as a sign"
```

## Morning Puzzle Retest

`muse_tmr.reports.morning_retest` captures the morning retest for the previous night's
generated unsolved puzzle set. The workflow records a response, solved/unsolved flag,
duration, and confidence for every `NightPuzzleSession` puzzle. It validates the full
set against the session and stores cue condition from `PuzzleCueAssignment` for
cued-vs-uncued analysis.

The CLI command is:

```bash
muse-tmr record-puzzle-retest data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --assignment data/protocol/night-001_assignment.json \
  --output data/reports/night-001_retest.json \
  --result "p1=morning answer" \
  --result "p2=" \
  --solved p1 \
  --duration "p1=42" \
  --duration "p2=30" \
  --confidence "p1=0.8" \
  --confidence "p2=0.2"
```

The retest should remain blind during administration: do not reveal which puzzles were
cued. The output stores `cue_condition` only for later analysis.

## Cued-vs-Uncued Analysis

`muse_tmr.reports.analysis` combines the generated night puzzle session, cue assignment,
morning retest, optional dream report, and optional scheduler JSONL events into a
descriptive analysis report. It reports cued vs uncued solve rates, dream incorporation
rates, retest duration/confidence, and puzzle cue timing from `TmrSchedulerEvent`
records.

The CLI command is:

```bash
muse-tmr analyze-cued-uncued data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --retest data/reports/night-001_retest.json \
  --dream-report data/reports/night-001_dream_report.json \
  --scheduler-events data/reports/night-001_scheduler.jsonl \
  --output data/reports/night-001_analysis.json \
  --markdown-output data/reports/night-001_analysis.md
```

Reports must be interpreted as exploratory home-run data, not statistical proof. The
JSON output includes limitations such as `small_n`, missing dream report, missing
scheduler events, missing cued timing, or any accidental uncued cue play.

Association checks compare a remembered response with the expected solution using a
case-insensitive whitespace-normalized match and append the result to the night session
metadata.

Private catalogs and generated sessions should stay under gitignored locations such as
`data/protocol/`.
