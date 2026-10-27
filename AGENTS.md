# Agent Instructions

This repository is the implementation workspace for the Muse S Athena REM-TMR/TLR project tracked under GitHub EPIC #1.

## Project Goal

Build an open-source, home-run Muse S Athena system for overnight recording, offline replay, REM-gated cue scheduling, TLR cues, puzzle cues, morning reporting, and cued-vs-uncued analysis.

The target is protocol fidelity and measurable validation. Do not claim clinical, medical, or guaranteed lucid-dreaming efficacy.

## Current Planning Source

- EPIC: #1
- Milestones: 0 through 10
- Child issues: #3 through #42 are attached to the EPIC as sub-issues.
- Issues from #100 on have no milestone.
- Work one GitHub issue at a time unless the user explicitly asks for a cross-cutting change.
- Keep labels minimal. Do not create a large label taxonomy unless requested.

Milestone themes:

- M0: repository and governance
- M1: data source and overnight recording
- M2: replay and feature extraction
- M3: REM detection
- M4: audio cue system
- M5: TMR/TLR protocol
- M6: morning reports and experiment analysis
- M7: OpenMuse and SDK adapters
- M8: validation and effectiveness criteria
- M9: documentation and agent handoff
- M10: local app UI and Muse contact gate

## Required Git Preflight

Before editing anything:

```bash
git fetch origin --prune
git status -sb
git rev-list --left-right --count origin/main...HEAD
```

If the first number from `rev-list` is greater than 0, run:

```bash
git rebase origin/main
```

If the rebase conflicts, stop and ask the user how to resolve each conflict chunk. Do not guess.

Repeat the same preflight before every commit and before every push.

## Commit And Push Rules

- Keep diffs small and scoped to the active issue.
- Avoid broad formatting, reordering, or unrelated refactors.
- Stage only files that belong to the current task.
- Use normal `git push` unless you rebased; after a rebase, use `git push --force-with-lease`.
- Never use plain `--force`.
- If the worktree contains unrelated changes, leave them alone.

## Repository Shape

- `src/muse_tmr/` is the engine and everything built on it: `sources`, `data`,
  `features`, `models`, `protocol`, `audio`, `reports`, `validation`, `annotations`,
  the synthetic nights in `testing`, the local app in `app` (page files in
  `app/static/`), and the `muse-tmr` CLI in `cli/main.py`.
- `src/muse_tmr/ble/` (`muse_tmr.ble`) is the forked amused-py BLE layer, wrapped by
  `muse_tmr.sources.amused_source`. Change it only for BLE or protocol fixes, so that
  upstream patches keep applying; `docs/dependency_strategy.md` says how upstream's
  paths map onto it. The top-level `muse_*.py` and `amused.py` are deprecated shims
  that re-export it until 0.5: import `muse_tmr.ble` in new code and add nothing to
  them. `examples/` holds the amused-py examples.
- `scripts/` holds the wrapper that runs the nightly REM report (the report itself is
  `muse_tmr.reports.nightly_report`, so `muse-tmr demo` can build it from the wheel), the
  REM detector check against a reference stager, the macOS launcher installer, the
  forbidden-files check, the wheel check and the page check.
- `docs/` holds the runbooks and references; `docs/USAGE.md` covers every command.
- Tests live in `tests/`. CI runs `python -m pytest tests/`; `run_tests.py` runs a
  fast unittest subset, and `run_tests.py --all` everything.
- `pyproject.toml` finds the `muse_tmr` packages under `src/` and ships every file in
  `app/static/`; until 0.5 the top-level shims are listed in `py-modules`. CI builds the
  wheel, and `scripts/check_wheel.py` fails if any file under `src/muse_tmr` is not in
  it, which an editable install would hide, or if it ships a top-level `amused*`.

Put new engine code in the matching `muse_tmr` subpackage and new commands in
`muse_tmr.cli.main`.

## Safety And Data Rules

- Do not commit proprietary Muse SDK binaries, headers, archives, installers, docs, or copied SDK code.
- SDK must be downloaded separately and placed locally; do not commit.
- See `docs/sdk_policy.md` and run `python scripts/check_forbidden_files.py` before publishing SDK-adjacent work.
- Do not commit private overnight recordings, personal sleep data, dream reports, calibration files, or device identifiers.
- Do not commit generated cue audio by default unless the issue explicitly asks for tiny test fixtures.
- Keep raw recordings and session outputs gitignored.
- Do not rotate, print, or modify secrets or tokens.
- Do not add network upload, telemetry, cloud sync, or remote reporting without explicit user approval.

## Implementation Guidance

- Prefer deterministic replay and synthetic fixtures before live BLE tests.
- Separate live device access from offline processing so tests can run without a Muse device.
- Preserve timestamps, sample rates, modality names, units, and source metadata.
- Treat missing modalities as normal. Muse sessions may lack PPG, HR, IMU, or clean EEG for periods.
- Use conservative defaults for sleep-time audio: low volume, fade in/out, cooldowns, arousal guards, and emergency stop.
- Log cue decisions as structured events: play, skip, block reason, pause, stop, volume, and timing.
- Keep REM detection and cue scheduling separate. REM probability must not directly trigger audio without the stable gate and safety checks.
- Keep validation language precise: report metrics and limitations instead of promising effectiveness.
- On macOS, direct Python BLE commands may abort in TCC before Python logs if the launching
  bundle lacks Bluetooth privacy metadata. For live smoke tests, prefer a Homebrew Python
  virtualenv launched through `Python.app` with `open -W -n --env PYTHONPATH=... --args -m muse_tmr.cli.main ...`.
- For live `record` smoke tests, use `--allow-short` and either an explicit `--output-dir`
  under `data/recordings/` or the CLI default. Never commit raw recordings, metadata with
  device identifiers, or generated session summaries.
- M1 live BLE smoke criteria: `discover` finds the Muse, `stream` reports nonzero frames
  across expected modalities, and `record` writes `raw_amused.bin`, `metadata.json`,
  `events.jsonl`, and `summary.json` with `stop_reason=duration_complete`.
- OpenMuse LSL source support lives in `muse_tmr.sources.openmuse_lsl_source`. Keep
  `mne_lsl`/`pylsl` optional, require OpenMuse to run separately, and convert LSL
  samples into normal `MuseFrame` objects with `source="openmuse"`. Do not make LSL
  dependencies part of the base install.
- Official Muse SDK support is currently only `muse_tmr.sources.muse_sdk_source_stub`.
  Keep it importable without SDK files, make runtime operations fail with the SDK
  policy message, and update `scripts/check_forbidden_files.py` when adding any
  SDK-adjacent paths. Never commit proprietary SDK binaries, headers, docs, or code.
- For M2 replay/features, use `muse_tmr.data.replay.ReplaySession` as the offline source.
  It emits the same `MuseFrame` type as live BLE and supports real-time, accelerated,
  fastest-possible, and relative time-range replay. Keep feature tests synthetic or replay-based.
- Use `muse_tmr.features.epochs.EpochBuilder` for 30-second sleep windows. It must accept
  both live source streams and replay streams, tolerate missing modalities, and expose
  coverage plus quality flags for downstream EEG/IMU/PPG feature modules.
- EEG features live in `muse_tmr.features.eeg_features`. Keep band-power tests synthetic
  with known sine waves, and preserve artifact flags rather than dropping noisy epochs.
- IMU motion features live in `muse_tmr.features.imu_features`. Keep movement/arousal
  tests synthetic with known accelerometer or gyroscope bursts, preserve noisy epochs,
  and expose arousal guard reason codes rather than coupling features directly to audio.
- PPG/HR/HRV features live in `muse_tmr.features.ppg_features`. Keep pulse/HR tests
  synthetic and deterministic, treat PPG and heart-rate samples as independently
  optional, and make HRV terminology explicit as a proxy until beat-level validation exists.
- REM predictions live in `muse_tmr.models`. Keep heuristic REM tests synthetic and
  deterministic, expose `P_REM` plus reason codes, and never couple REM probability
  directly to audio playback or cue decisions.
- Manual REM annotations live in `muse_tmr.annotations`. Preserve the supported labels
  `wake`, `nrem`, `probable_rem`, and `unknown`; generated annotation templates should
  default to `unknown`, overlay feature columns, and keep training export separate from
  raw personal recordings.
- Personal REM classifier code lives in `muse_tmr.models.ml_rem_detector`. Train from
  annotation rows, skip `unknown` labels by default, keep artifacts versioned and
  loadable, and report imbalance/calibration/feature-importance metrics. Do not commit
  private trained models or real annotation files unless the user explicitly asks and
  confirms the data is shareable.
- REM confidence and stable gate code lives in `muse_tmr.models.rem_gate`. Keep it
  stateful, deterministic, replay-testable, and separate from audio. Gate decisions may
  expose `gate_open` and block reasons, but must not expose cue playback decisions.
- Low-volume audio playback lives in `muse_tmr.audio.audio_player`. Keep playback
  behind an explicit player facade with volume caps, fade metadata, emergency stop, and
  JSONL logging. Tests must use mock or dry-run backends and must not require speakers.
- Cue library metadata lives in `muse_tmr.audio.cue_library`. Keep catalogs separate
  from private audio files, validate missing files before sessions, and do not commit
  real cue audio or private cue paths unless the user explicitly confirms they are safe.
- Pre-sleep volume calibration lives in `muse_tmr.audio.volume_calibration`. Store
  detectable, identifiable, and comfortable volumes per playback device, keep
  calibration files out of git, and make scheduler code use the calibrated comfortable
  volume as the max while still respecting a hard safety cap. Missing calibration
  should block planned sleep-time cue sessions.
- Pre-sleep puzzle session management lives in `muse_tmr.protocol.puzzle_protocol`.
  Keep puzzle catalogs, timed attempts, solved/known/retired filters, and association
  checks separate from cued-vs-uncued randomization. Do not commit private puzzle
  content, user responses, or generated sessions; `data/protocol/` is gitignored.
- Cued-vs-uncued randomization lives in `muse_tmr.protocol.randomization`. Use seeded
  `PuzzleCueAssignment` records and make scheduler code consume `scheduled_puzzle_ids`
  or `scheduled_cue_ids()`, never all session puzzle IDs, so uncued controls cannot be
  scheduled.
- Night arms and sealed N-of-1 schedules live in `muse_tmr.protocol.night_schedule`
  (see `docs/nof1_protocol.md`). The scheduler's `arm` decides only what a gated cue
  becomes (`cue`, `sham` with an untrained `sham_cue_id`, `no_cue` with `virtual_play`
  skips), never when it comes, and uncued puzzles stay unschedulable in every arm. A
  runner never prints a night's arm or the counts that would give it away, checks the
  sham cue in every arm, and sends nothing to the player in `no_cue`. Keep schedules,
  their salts and everything a series writes under the gitignored `data/`.
  `prereg_template()` renders the design and the commitment, never an arm, the seed or
  the salt; a blind coding packet (`muse_tmr.reports.incorporation_coding`) holds the
  dream text and the puzzles and nothing about cues, the assignment or the arm; and
  `muse_tmr.reports.nof1_analysis` refuses a sealed schedule, reads the records that
  exist (night summaries, dream report and retest schema 2, `arousal_proxy_per_cue`)
  instead of new ones, moves arm labels only the way the schedule draws them, and stays
  pure Python and descriptive, with no p-value.
- TLR cue support lives in `muse_tmr.protocol.tlr_protocol`. Keep default cue metadata,
  pre-sleep training events, and REM TLR block planning separate from the final scheduler.
  Training should use `AudioCuePlayer` with mock or dry-run backends in tests and must
  write structured events.
- REM-gated cue scheduling lives in `muse_tmr.protocol.tmr_scheduler`. It should consume
  `RemGateDecision`, optional `TlrBlockPlan`, and `PuzzleCueAssignment.scheduled_puzzle_ids`;
  never schedule uncued puzzles. Keep it deterministic and replay-testable, log
  `play`, `skip`, `pause`, and `stop` events, and do not call real audio playback inside
  the scheduler.
- Arousal guard logic lives in `muse_tmr.protocol.arousal_guard`. Keep it feature-row
  based, configurable, conservative by default, and replay-testable. It may return
  `allow`, `lower_volume`, `pause`, or `stop` decisions from motion, alpha, HR-jump,
  and artifact-quality proxies, but it must not play audio directly.
- Morning dream report capture lives in `muse_tmr.reports.dream_report`. Keep reports
  structured, local-first, and versioned; link dream content to generated session puzzle
  IDs only after validating against `NightPuzzleSession`. Do not commit real dream
  reports, private recall text, or morning session outputs.
- The app's Morning tab (`muse_tmr.app.morning`) saves through `build_dream_report`,
  `build_cue_recall_test` and `build_morning_retest`, into the paths the Pilot 5 runbook
  uses. Nothing it sends to the page may say which puzzles were cued or which arm the
  night had: no assignment IDs, no `cue_condition` and no arm, also not when it shows a
  saved retest or sound check again, and not through the sound check's sounds, their
  order, their number or their labels.
- Morning puzzle retest capture lives in `muse_tmr.reports.morning_retest`. Keep the
  retest blind during administration, but save `cue_condition` from `PuzzleCueAssignment`
  in the output for later analysis. Require solved/unsolved, duration, and confidence
  for each generated session puzzle.
- Cued-vs-uncued analysis lives in `muse_tmr.reports.analysis`. Keep it descriptive,
  versioned, and local-first; combine session, assignment, retest, optional dream report,
  and scheduler events without copying raw dream text into summary outputs. Always record
  limitations such as small-N or missing timing data.
- Protocol presets live in `muse_tmr.protocol.protocol_presets` (the records) and
  `muse_tmr.protocol.protocol_registry` (the presets); see `docs/protocol_presets.md`.
  Name a preset after a paper's parameters and arm, never after a lab's protocol; set
  `source_check` to `verified_full_text` only for a paper whose full text was read, else
  `notes_only`. Keep `remind_default_v1` equal to the dataclass defaults and the default
  of every runner. No preset may set a volume or lower the gate's 60 s stable window. A
  change to a preset raises its `version` and renews its pinned hash in
  `tests/test_protocol_presets.py` in the same commit.
- The protocol fidelity report (`muse_tmr.reports.protocol_fidelity`,
  `protocol-fidelity`) reads REM periods only through `muse_tmr.validation.gate_log`,
  plays nothing, and always writes its `parameter_fidelity_is_not_result_fidelity` block.
  The morning sound check (`muse_tmr.reports.cue_recall`, `record-cue-recall`) plays
  through `AudioCuePlayer` with the backend resolved as for a night and the calibrated
  cap; its play order comes from the session ID alone. Never commit its output.
- M8 validation planning lives in `docs/validation.md`. Keep validation metrics tied to
  concrete logs or artifact fields, separate protocol fidelity from behavioral deltas,
  and include limitations instead of claiming clinical or lucid-dreaming efficacy.
- M8 Pilot 1 is a no-audio overnight recording workflow. Keep its runbook in
  `docs/pilot1_no_audio.md`, validate completed sessions with
  `muse-tmr validate-pilot1-recording`, and never commit the generated recording or
  validation output.
- M8 Pilot 2 is an awake audio-calibration-only workflow. Keep its runbook in
  `docs/pilot2_audio_calibration.md`, validate calibration cap probes with
  `muse-tmr validate-pilot2-calibration`, and never commit local calibration JSON,
  playback logs, or private device names unless explicitly confirmed safe.
- M8 Pilot 3 is a replay-only cue simulation workflow. Keep its runbook in
  `docs/pilot3_replay_cue_simulation.md`, generate mocked-audio cue plans with
  `muse-tmr simulate-replay-cues`, require zero uncued puzzle `play` events, and
  never treat replay simulation as real sleep-time cueing.
- M8 Pilot 4 is the first low-volume REM-gated cueing workflow. Keep its runbook in
  `docs/pilot4_low_volume_cueing.md`, require volume calibration before playback,
  use `muse-tmr run-pilot4-cueing`, and make real audio explicit through the selected
  backend while logging scheduler, arousal, audio, awakening, and emergency-stop
  artifacts.
- M8 Pilot 5 is the full-night TLR plus puzzle-cueing workflow. Keep its runbook in
  `docs/pilot5_full_night.md`, require the Pilot 4 safety setup plus a TLR block plan,
  use `muse-tmr run-pilot5-full-night`, and do not interpret results until dream
  report, blind retest, scheduler logs, and cued-vs-uncued analysis are generated.

- Meditation complexity analysis is a side track: metrics in
  `muse_tmr.features.complexity_features`, blocks/analysis/aggregation in
  `muse_tmr.reports.meditation_analysis` (see `docs/meditation_metrics.md`). Keep it
  decoupled from REM detection, the gate, the scheduler, the arousal guard and audio,
  keep the inference unit the session, and never commit blocks files or reports.
- The voice-guided calibration run (`muse_tmr.protocol.calibration`,
  `muse_tmr.reports.calibration_report`, `docs/calibration_run.md`) cuts its blocks from
  the logged cue times, not the plan. Keep the cue log append-only and the protocol
  step names stable: the report and the pair definitions key on them.
- Polar H10 companion recording lives in `muse_tmr.sources.polar_h10`,
  `muse_tmr.data.polar_recorder`, `muse_tmr.data.polar_session` and
  `muse_tmr.features.cardio_resp_features` (see `docs/polar_h10.md`). It runs as its own
  process, keeps raw BLE payloads first, never puts chest ACC into `MuseFrame.imu`, and
  stays out of REM detection, the gate, the arousal guard, the scheduler and audio. Never
  commit device addresses, serials or Polar recordings.
- Synthetic nights live in `muse_tmr.testing` (see `docs/synthetic_data.md`): `synthetic`
  writes a night in the Athena TAG format with its ground truth, `scenarios` names eight
  of them and `agreement` scores the gate against `hypnogram.csv` with
  `muse_tmr.validation.rem_metrics`. `muse-tmr synthesize-night` writes one, and
  `muse-tmr demo` runs Pilot 1, Pilot 3 with mock audio, the REM report and the agreement
  on it. Every JSON and HTML file they write says "synthetic input, not sleep", and so do
  the REM report and the Pilot 1 and Pilot 3 reports of a synthetic night (they read
  `synthetic: true` in its `metadata.json`). Never fit a detector weight or
  threshold to them, never present their agreement as REM accuracy (on `clean_night` the
  cues play at sleep onset), never mix in a fragment of a real night, and never commit
  what they write (`data/synthetic/` is gitignored and forbidden). A generator change
  moves the pinned sha256 in `tests/test_synthetic_night.py`; a change to the generator,
  detector, gate, guard or scheduler can move `tests/fixtures/synthetic/agreement_baseline.json`,
  which the CI `end_to_end` job compares `demo --hours 2` with. Renew them in the same commit.

## Testing Expectations

For Python changes, run the fastest relevant checks:

```bash
python run_tests.py
```

For targeted changes, prefer focused tests first, for example:

```bash
python -m unittest tests.test_realtime_decoder
python -m unittest tests.test_raw_stream
python -m unittest tests.test_ppg_fnirs_fast
```

Run all tests when shared behavior, binary formats, parser logic, replay, or scheduler contracts change:

```bash
python run_tests.py --all
```

For changes to the app's page (`app/static/`), also open it the way CI does: start
`muse-tmr app --source mock --port 8799` and run
`node scripts/check_page.mjs http://127.0.0.1:8799/` (Node 22, or Node 20 with
`--experimental-websocket`). It goes through the tabs in headless Chrome and fails on any
page error; the app tests only check the strings the server sends.

For docs-only changes, at minimum run:

```bash
git diff --check
```

## Documentation Rules

- Update README or project docs when adding a user-visible command, protocol step, file format, or safety behavior.
- Keep setup instructions reproducible from a clean checkout.
- Distinguish required dependencies from optional visualization, OpenMuse, or SDK-related adapters.
- Document any binary output format and generated artifact location before relying on it.
- A new figure quoted in the README, `TEST_STATUS.md`, `docs/` or a config dataclass needs a row in `docs/numbers.md` with its source and status (code default, in-sample, GitHub only, self-reported, literature).

## Issue Handoff

When finishing an issue, leave enough context for the next agent:

- what changed
- how it was tested
- remaining risks or limitations
- any follow-up issue numbers

Prefer small PRs or commits tied to a single issue number.

The ten uniqueness initiatives of 2026-10 are sub-issues of EPIC #1 titled `Initiative N: …`.
Their evidence base is `docs/research/2026-10-uniqueness/` (the report, the research notes
and the issue drafts). A local Claude Code session drives them from issue to merged PR with
the `initiatives` skill in `.claude/skills/initiatives/SKILL.md`; each issue's "Needs from
the maintainer" section lists what only a person can do.
