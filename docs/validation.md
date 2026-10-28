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
- `rem_gate_decisions.jsonl` from Pilot 4 and 5, one gate decision per epoch.
- `cue_arousal_events.jsonl`, `cue_arousal_report.json` and `.md` from the cue arousal
  scorer, on a night with cues.
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
| Night arm contract | A `no_cue` night: scheduler `play` events plus `status == "played"` lines of `audio_playback.jsonl`; a `sham` night: its plays against the same night's plan in the cue arm | `pilot4_summary.json`/`pilot5_summary.json` criterion `no_play_in_no_cue_arm`, scheduler JSONL, audio JSONL | A no-cue night with a play or a played line, or a sham play at another time or volume than the cue it stands in for. |
| Sealed arm | The night's `arm` against its sealed schedule, and the schedule's commitment against its companion, at the end of the night | Summary criterion `arm_matches_sealed_schedule`, `metadata.json` `night_arm` | The schedule changed, or the night ran another arm. A night run after `unseal-night-schedule` has `schedule_unsealed_at_utc` set and was not blind. |
| Synthetic end-to-end regression | `muse-tmr demo --hours 2` on `clean_night`, seed 0: Pilot 1 and Pilot 3 pass, zero uncued `play` events, mock audio, the report written, the raw sha256 and Pilot 3 counts equal to `tests/fixtures/synthetic/agreement_baseline.json` and the agreement within its 0.01 | CI `end_to_end` job, `demo_summary.json` | Anything differs from the baseline. It checks plumbing on synthetic input, not sleep, not validity: its cued plays come on N1 epochs at sleep onset, not in REM (`docs/synthetic_data.md`). |

## Technical Quality Metrics

| Metric | Computation | Source | Interpretation |
| --- | --- | --- | --- |
| EEG feature coverage | Rows with valid EEG bands divided by total epochs | EEG feature rows | Measures how much of the night can feed REM detection. |
| EEG artifact burden | Epochs flagged for clipping, flatline, missing channels, or high artifact proxy | EEG feature rows | High burden limits REM confidence and should be called out. |
| IMU stillness and arousal events | Stillness score distribution, movement event count, cue-window motion count | IMU feature rows; the cue-window count from `cue_arousal_events.jsonl` | Used by arousal guard and to explain blocked cues. The live guard never sees cue times, so its `cue_related_arousal` rule stays off; the scorer counts the window afterwards. |
| PPG/HR availability | PPG-derived HR rows, `HeartRateSample` rows, missing-modality flags | PPG feature rows | HRV metrics are proxies. A beat-level reference exists when a Polar H10 records with the Muse; see the next row. |
| Optics HR agreement | Bias, SD, 95 % limits of agreement and MAE of each optics heart-rate estimate against H10 ECG R-peaks, by `ppg_signal_quality`, `ppg_confidence` and the series' quality; the sudden heart-rate change rule's false and missed jumps against the ECG change | `optics_agreement.json` from `analyze-optics-agreement` (`docs/optics_agreement.md`) | Agreement with a chest ECG, a proxy for heart rate. Awake desk sessions say little about sleep; no `p1034` night with the H10 has been recorded yet. |
| REM-event density | `rem_event_count` and `rem_density_per_min` per epoch, epochs in warm-up, epochs where they are NaN | EEG feature rows, `feature_value_rem_event_*` annotation columns | A supporting term at weight 0 until fitted on REM-bearing nights; against a reference stager it gives an AUC against an automated proxy, not REM accuracy. |
| REM gate stability | Open/closed counts, consecutive-open streaks, reason codes, and transition times | `StableRemGate` decisions | Stable REM eligibility is separate from raw classifier probability. |
| Per-feature separation and gate grid | ROC-AUC, min, max and `saturated` of each `score_*`/`value_*` column against a reference stager's REM; gate-open fraction, REM recall and precision for each enter/exit pair | `scripts/validate_rem_detection.py` metrics JSON: `feature_metrics`, `gate_grid` | Agreement with an automated proxy, not REM accuracy. A saturated column carries no information. |
| REM agreement with PSG consensus (BOAS) | On the report fold of participant-grouped splits: binary REM κ, F1, precision (the gate-open precision) and recall of the open epochs; precision at recall 0.80 of the score; REM-onset latency median and IQR; false-open minutes per night; per-feature AUC; for the heuristic at `p21` and `boas_v1`, the personal model, GSSC and YASA, with the headband `stage_ai` for context | `metrics.json` of `validate-external`, its `splits.json` and their sha256 (`docs/validation_external.md`) | Against three PSG scorers' consensus, but on a Bitbrain Ikon referenced to the mastoid and with the EEG alone: a frontal-montage transfer, not a Muse result, and the stillness term and motion blocks are not in it. The run over the 128 nights is pending. |
| First-cue latency | `stride_seconds`, `first_gate_open_seconds`, `first_play_seconds` (to the end of the deciding epoch); against a reference, minus its REM onset | Pilot 3 report `metrics`, `pilot4_summary.json` | On synthetic input a shorter stride did not bring the first cue earlier; not yet measured on a real night. |
| Scheduler behavior | Counts of `play`, `skip`, `pause`, `stop`; cooldown and interval reason codes | Scheduler JSONL | Audits whether cue timing rules were enforced. |
| Arousal guard burden | Counts by guard action and reason code: motion, alpha, HR jump, artifact quality | Guard decisions, scheduler metadata | Explains cue suppression or volume lowering. |
| Audio calibration status | Comfortable volume, detectable volume, scheduler max volume, device name | Volume calibration JSON, scheduler metadata | Missing calibration should block planned sleep-time cue sessions. |
| TLR block fidelity | TLR block repetitions, inter-cue intervals, post-block pause, training event count; per REM period the TLR cues played against the plan, and their spacing as played (at the decisions of the epochs that played them) against the planned spacing | TLR training JSONL, block plan, `protocol_fidelity.json` (`checks.tlr_block`) | Verifies that TLR was administered as planned before sleep or before puzzle cues. At a 30 s stride cues planned closer than 30 s play back to back. |
| Protocol preset fidelity | Intervals between cues, cues per REM period, stable REM before the first cue, pause rules, cue gain and length, each against the night's protocol preset | `protocol_fidelity.json` from `protocol-fidelity`, the `protocol` block of `metadata.json` | How closely the night's timing followed a paper's parameters. Parameter fidelity, not result fidelity: no PSG, no EOG, gain not dB SPL (`docs/protocol_presets.md`). |
| Sound check (manipulation check) | Recall rate of the sound-to-puzzle check by condition, and whether it came before the retest | `<night>_cue_recall.json`, analysis `cue_recall` | Whether the pairings still held in the morning. A low rate limits what any cued-vs-uncued delta can mean; it is not an outcome. |

## Safety Outcome Metrics

What follows a cue, scored afterwards by `muse_tmr.reports.cue_arousal` against quiet
control moments of the same night (`docs/safety_outcomes.md`, #227). They are arousal
proxies, not scored arousals, the thresholds are UNFITTED, and no validator fails on any
of them. No night with sleep-time audio has been scored yet.

| Metric | Computation | Source | Interpretation |
| --- | --- | --- | --- |
| Arousal proxy per cue and per control | Windows with a flag over windows where it could be scored, per flag (`alpha`, `beta`, `emg`, `imu`, `hr`, `any`), and cue minus control | `cue_arousal_report.json` `outcomes`, `cue_arousal_events.jsonl` | A cue rate means something only against the control rate of the same night; the study figures in `docs/safety_outcomes.md` come from other devices and scoring. |
| Movement events per cue and per control | Mean movement events from 2 s before to 10 s after | same | The measure of Konkoly et al. 2024 (0.19 against 0.10), from the headband's IMU instead of a phone. |
| Cue-attributed awakening rate | Played cues followed within 120 s by an awakening marker | `awakening_events.jsonl`, the report | Only as complete as the markers; `attributed_to_cue` adds the sleeper's own call. |
| REM periods with any disruption | Gate-open runs with a played cue in which a cue raised `any` or the run closed within 60 s of one | `rem_gate_decisions.jsonl`, the report | A gate-open run is the gate's REM, not a scored REM period. |
| REM termination within 60 s | Cues, and controls, after which their gate-open run closed within 60 s | same | The control rate is how often the gate closes anyway. |
| Dose tables | The `any` rate by effective volume, time since the previous cue and cue index in a block | `audio_playback.jsonl`, scheduler play events, the report | Volume is a software gain, not dB SPL; at n = 1 a bin holds a few cues. |

## Behavioral And Effect Metrics

Behavioral metrics are exploratory. Report deltas and confidence context; do not report
them as proof of effectiveness.

| Metric | Computation | Source | Notes |
| --- | --- | --- | --- |
| Dream report completion | Dream report exists and validates against the night puzzle session | Dream report JSON | Keep raw self-report text separate from summary metrics. |
| Lucid report rate | `lucid == true` count divided by completed reports | Dream report JSON across sessions | Self-report only. |
| Cues-heard rate | `cues_heard == true` count divided by completed reports | Dream report JSON across sessions | Use as awareness/context, not as a success claim. |
| Cue perception and sound awakenings | Reports by `cue_perception` (`not_perceived`, `in_dream`, `awake`, `unsure`, `unknown`) and by `woke_from_sound`, per night arm | Dream report JSON (schema 2), the night's summary `arm` | Self-report. On a sham night a perceived sound is the sham sound; at n = 1 the sleeper may tell the sounds apart. |
| Delayed retest | The retest metrics and cued-minus-uncued deltas of the `day7` and `day10` waves next to the morning one | Retest JSON `retest_wave` (schema 2), analysis `wave_metrics` (schema 3) | A missing wave is the limitation `missing_delayed_retest`, not a zero. |
| Puzzle incorporation rate | Linked puzzle incorporations divided by generated session puzzle count | Dream report JSON, session JSON | Count only links that validate against session puzzle IDs. |
| Blind incorporation codes | Per puzzle `sound_cue`, `direct_conceptual`, `indirect_conceptual` from a rater who saw only the dream text and the puzzles; incorporated when more than half of the raters coded it; incorporation against solving per wave | Coding sheets of `record-incorporation-coding`, analysis `incorporation` and `incorporation_solved` | Preferred over the sleeper's links. A sleeper who codes their own dreams is blind to which puzzles were cued, not to the dream. |
| Rater agreement | Percent agreement and Cohen's kappa per category between two raters | `compare-incorporation-codings` JSON | Agreement between two people's codes, not their accuracy; undefined when both gave one answer to every puzzle. |
| Arm differences of a series | Per outcome the difference of two arms' means with its permutation interval over the orders the schedule could have drawn; lag-1 autocorrelation; the means without the first night | `analyze-nof1` JSON `differences`, `autocorrelation`, `descriptives` | Single-case description after unsealing. With two or three blocks nothing can fall outside an interval; a difference outside one describes the series and is not evidence of an effect. |
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
- Arousal proxies after cues that are not PSG-scored arousals, with `thresholds_unfitted`
  until a pull request carries real-night numbers.
- For nights with an arm: which nights were sham or no-cue, whether the schedule was
  sealed and when it was unsealed, that a sham still makes sound and may be recognised,
  and that at n = 1 the experimenter cannot be blind.
- For a series: the codes of `analyze-nof1` (`docs/nof1_protocol.md`), among them
  `unblinded_early`, `carry_over_possible`, `first_night_effect_possible`,
  `missing_delayed_retest` and `missing_arousal_scoring`, and that every arm comparison
  inherits the REM detector's open issues (#116, #167).

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
  two were comparable. A κ from `compare-incorporation-codings` is the agreement of two
  raters' codes; call it that.
- An N-of-1 series is described, not tested. Report a difference between arms with its
  permutation interval and the number of orders behind it, as "observed difference" or
  "outside the permutation interval"; give no p-value as evidence for an effect at
  n = 1, and do not call a difference significant.
