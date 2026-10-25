# Validation

Validation for this project is descriptive and local-first. It checks whether the
recording, replay, REM gate, cue scheduler, safety guard, and morning analysis pipeline
ran as designed before looking at behavioral outcomes. It must not be used to claim
clinical efficacy, diagnostic accuracy, therapy, or guaranteed lucid dreaming.

## Required Artifacts

Each night or replay validation bundle should keep the following artifacts together:

- `summary.json` from `OvernightRecorder`, with duration, frame counts, modality counts,
  reconnect attempts, downtime, and stop reason.
- `metadata.json` and `events.jsonl` from the recording directory.
- The raw packet file when the selected source produces one, for example
  `raw_amused.bin`.
- Replay, epoch, and feature outputs for EEG, IMU, PPG, and HR/HRV rows.
- REM detector outputs and `StableRemGate` decisions with reason codes.
- `TmrSchedulerEvent` JSONL from the REM-gated scheduler.
- TLR training or block-plan events when TLR was part of the session.
- Puzzle session, cued-vs-uncued assignment, dream report, morning retest, and final
  cued-vs-uncued analysis JSON or Markdown.

Missing artifacts should be reported as limitations. Do not fill missing measurements
with assumptions.

## Protocol Fidelity Metrics

| Metric | Computation | Source | Flag when |
| --- | --- | --- | --- |
| Duration completeness | `summary.duration_seconds / requested_duration_seconds` | `summary.json`, CLI/session config | The run ended before the planned duration without an expected manual stop. |
| Stop reason | Exact `summary.stop_reason` | `summary.json` | Anything other than `duration_complete` for planned fixed-duration tests. |
| Source metadata present | Required source name, device/address metadata, and start time are present | `metadata.json` | Metadata is missing or cannot be tied to the session. |
| Modality coverage | Per-modality counts and per-epoch coverage fractions | `summary.modality_counts`, epoch/feature rows | Expected modalities are absent or below the pilot threshold. |
| Raw capture availability | `summary.raw_packet_count` and raw path existence | `summary.json`, raw packet file | Raw capture is expected for the source but missing or empty. |
| Reconnect burden | `summary.reconnect_attempts`, `summary.downtime_seconds / summary.duration_seconds` | `summary.json`, `events.jsonl` | Reconnects or downtime exceed the configured pilot threshold. |
| Replay determinism | Same raw path and time range produce the same frame counts, epoch indexes, and feature row counts | Replay and feature outputs | Re-running replay changes deterministic counts or row identities. |
| Cued-only scheduler contract | Count puzzle `play` events whose `puzzle_id` is not in `PuzzleCueAssignment.scheduled_puzzle_ids` | Scheduler JSONL, assignment JSON | Any uncued puzzle receives a scheduler `play` event. |
| Safety separation | Audio decisions are derived from stable gate plus scheduler/guard state, not raw `P_REM` alone | Gate decisions, scheduler events, guard decisions | A cue can be traced directly from REM probability without gate and guard checks. |
| Synthetic end-to-end regression | `muse-tmr demo --hours 2` on `clean_night`, seed 0: Pilot 1 and Pilot 3 pass, zero uncued `play` events, mock audio, the report written, the raw sha256 and Pilot 3 counts equal to `tests/fixtures/synthetic/agreement_baseline.json` and the agreement within its 0.01 | CI `end_to_end` job, `demo_summary.json` | Anything differs from the baseline. It checks plumbing on synthetic input, not sleep, not validity: its cued plays come on N1 epochs at sleep onset, not in REM (`docs/synthetic_data.md`). |

## Technical Quality Metrics

| Metric | Computation | Source | Interpretation |
| --- | --- | --- | --- |
| EEG feature coverage | Rows with valid EEG bands divided by total epochs | EEG feature rows | Measures how much of the night can feed REM detection. |
| EEG artifact burden | Epochs flagged for clipping, flatline, missing channels, or high artifact proxy | EEG feature rows | High burden limits REM confidence and should be called out. |
| IMU stillness and arousal events | Stillness score distribution, movement event count, cue-window motion count | IMU feature rows | Used by arousal guard and to explain blocked cues. |
| PPG/HR availability | PPG-derived HR rows, `HeartRateSample` rows, missing-modality flags | PPG feature rows | HRV metrics are proxies until beat-level validation exists. |
| REM-event density | `rem_event_count` and `rem_density_per_min` per epoch, epochs in warm-up, epochs where they are NaN | EEG feature rows, `feature_value_rem_event_*` annotation columns | A supporting term at weight 0 until fitted on REM-bearing nights; against a reference stager it gives an AUC against an automated proxy, not REM accuracy. |
| REM gate stability | Open/closed counts, consecutive-open streaks, reason codes, and transition times | `StableRemGate` decisions | Stable REM eligibility is separate from raw classifier probability. |
| Per-feature separation and gate grid | ROC-AUC, min, max and `saturated` of each `score_*`/`value_*` column against a reference stager's REM; gate-open fraction, REM recall and precision for each enter/exit pair | `scripts/validate_rem_detection.py` metrics JSON: `feature_metrics`, `gate_grid` | Agreement with an automated proxy, not REM accuracy. A saturated column carries no information. |
| First-cue latency | `stride_seconds`, `first_gate_open_seconds`, `first_play_seconds` (to the end of the deciding epoch); against a reference, minus its REM onset | Pilot 3 report `metrics`, `pilot4_summary.json` | On synthetic input a shorter stride did not bring the first cue earlier; not yet measured on a real night. |
| Scheduler behavior | Counts of `play`, `skip`, `pause`, `stop`; cooldown and interval reason codes | Scheduler JSONL | Audits whether cue timing rules were enforced. |
| Arousal guard burden | Counts by guard action and reason code: motion, alpha, HR jump, artifact quality | Guard decisions, scheduler metadata | Explains cue suppression or volume lowering. |
| Audio calibration status | Comfortable volume, detectable volume, scheduler max volume, device name | Volume calibration JSON, scheduler metadata | Missing calibration should block planned sleep-time cue sessions. |
| TLR block fidelity | TLR block repetitions, inter-cue intervals, post-block pause, training event count | TLR training JSONL, block plan | Verifies that TLR was administered as planned before sleep or before puzzle cues. |

## Behavioral And Effect Metrics

Behavioral metrics are exploratory. Report deltas and confidence context; do not report
them as proof of effectiveness.

| Metric | Computation | Source | Notes |
| --- | --- | --- | --- |
| Dream report completion | Dream report exists and validates against the night puzzle session | Dream report JSON | Keep raw self-report text separate from summary metrics. |
| Lucid report rate | `lucid == true` count divided by completed reports | Dream report JSON across sessions | Self-report only. |
| Cues-heard rate | `cues_heard == true` count divided by completed reports | Dream report JSON across sessions | Use as awareness/context, not as a success claim. |
| Puzzle incorporation rate | Linked puzzle incorporations divided by generated session puzzle count | Dream report JSON, session JSON | Count only links that validate against session puzzle IDs. |
| Retest completion | Retest rows divided by generated session puzzle count | Morning retest JSON, session JSON | Rows must include solved/unsolved, duration, and confidence. |
| Solve rate by condition | `solved_count / puzzle_count` for cued and uncued groups | Analysis JSON `condition_metrics` | Cued-vs-uncued comparison is descriptive. |
| Incorporation rate by condition | `incorporated_count / puzzle_count` for cued and uncued groups | Analysis JSON `condition_metrics` | Requires a dream report. |
| Timing and confidence by condition | Mean duration and mean confidence for cued and uncued groups | Analysis JSON `condition_metrics` | Report missing or outlier values as limitations. |
| Cue exposure | Cue play count, puzzles with cues, first and last cue time | Scheduler JSONL, analysis `cue_timing` | Needed before interpreting cued-vs-uncued deltas. |
| Effect deltas | Cued minus uncued solve rate, incorporation rate, duration, and confidence summaries | Analysis JSON `effect_summary` | Use terms like "observed difference" or "delta". |

## Required Limitations

Every validation report should include a limitations section. At minimum, consider:

- `small_n` when either cued or uncued group is below the configured minimum group size.
- Missing dream report, morning retest, scheduler events, feature rows, or raw recording.
- Missing cued timing for any cued puzzle.
- Any observed uncued cue play event.
- Consumer EEG and PPG limitations compared with laboratory PSG.
- Movement, poor sensor contact, audio audibility, waking, or environment confounds.
- Self-report and recall bias in dream incorporation and lucidity fields.
- HRV values based on proxy features rather than validated beat-level intervals.

## Pilot Gates

M8 validation should progress through increasingly risky pilots:

1. Overnight no-audio recording: verify duration, metadata, modality coverage, raw
   capture, reconnect/downtime limits, and replayability. Use
   `docs/pilot1_no_audio.md` and `muse-tmr validate-pilot1-recording` for the M8 Pilot 1
   acceptance report.
2. Audio calibration only: verify detectable and comfortable volume values, scheduler
   max volume, and device metadata without sleep-time cueing. Use
   `docs/pilot2_audio_calibration.md` and `muse-tmr validate-pilot2-calibration` to
   confirm a dry-run cap probe uses the saved calibration.
3. Replay cue simulation: run REM gate, arousal guard, and scheduler on replay outputs;
   require zero uncued puzzle plays and auditable `skip` reason codes. Use
   `docs/pilot3_replay_cue_simulation.md` and `muse-tmr simulate-replay-cues` to
   produce the mocked-audio cue plan and scheduler event stream.
4. Low-volume REM-gated cueing: require calibration, emergency stop availability,
   arousal guard events, scheduler events, and clear limitations. Use
   `docs/pilot4_low_volume_cueing.md` and `muse-tmr run-pilot4-cueing`; real audio
   requires an explicit backend such as `system` and a saved volume calibration.
5. Full-night exploratory session: require the Pilot 4 safety setup plus a TLR block,
   recording bundle, scheduler logs, dream report, retest, and cued-vs-uncued analysis
   output before interpreting any deltas. Use `docs/pilot5_full_night.md` and
   `muse-tmr run-pilot5-full-night`; real audio requires an explicit backend such as
   `system`.

## Reporting Language

Use precise language:

- "Protocol fidelity", "technical quality", "observed delta", "descriptive analysis",
  and "exploratory session" are acceptable.
- "Effective", "proven", "validated lucid dreaming", "diagnostic", "therapeutic", and
  "not less effective than PSG/lab protocols" are not acceptable.
- "Not less effective" criteria, if used for M8 planning, must mean internal operational
  thresholds against previous local runs or planned controls, not a claim against clinical
  or laboratory systems.
- A κ from `scripts/validate_rem_detection.py` is a binary REM-vs-not κ against an
  automated stager, as its `kappa_definition` says. Call it that, and do not put it next to
  a five-class κ against PSG or expert scoring, as published for other headbands, as if the
  two were comparable.
