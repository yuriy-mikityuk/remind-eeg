# Protocol presets

A protocol preset is the cueing parameters of one paper and arm, written down as a versioned, hashed record that the runners can load (#231). It says what the paper reports, where it comes from, how that was checked, the settings REMind runs it with, and where REMind on a Muse headband cannot follow it. The code is `muse_tmr.protocol.protocol_presets` (the records) and `muse_tmr.protocol.protocol_registry` (the presets themselves).

A preset is named after the paper's parameters: "parameters from Konkoly et al. 2021, Northwestern arm", never a lab's protocol. Running one does not reproduce a paper: there is no PSG-verified REM, no EOG and no sound level, and no night with sleep-time audio has been run yet. What a preset gives is provenance, and with `protocol-fidelity` a measurement of how closely a night followed it.

## Using a preset

```bash
muse-tmr protocol-presets                         # name, version, runnable, hash, how checked, title
muse-tmr protocol-presets konkoly2026_puzzle      # the whole preset as JSON, its hash and manifest
```

`--protocol-preset NAME` takes the protocol settings from a preset on the commands below. Without it they use `remind_default_v1`, which is the repository's defaults, so nothing changes for a run that does not ask for a preset. A flag given explicitly wins over the preset and is recorded as an override.

| Command | Flags the preset sets | Setting |
| --- | --- | --- |
| `simulate-replay-cues`, `run-pilot4-cueing`, `run-pilot5-full-night` | `--min-stable-seconds`, `--puzzle-cue-interval-seconds`, `--scheduler-cooldown-seconds`, `--max-puzzle-cues-per-block` | `gate_min_stable_seconds`, `puzzle_cue_interval_seconds`, `scheduler_cooldown_seconds`, `max_puzzle_cues_per_block` |
| `plan-tlr-block` | `--repetitions`, `--interval-seconds`, `--post-block-pause-seconds`, `--disabled` | `tlr_repetitions`, `tlr_interval_seconds`, `tlr_post_block_pause_seconds`, `tlr_block_enabled` |
| `create-tlr-cue` | `--frequency-hz`, `--duration-seconds` | `tlr_tone_frequency_hz`, `tlr_tone_duration_seconds` |
| `generate-puzzle-session` | `--count` | `puzzle_count` |
| `assign-puzzle-cues` | `--cued-count` | `cued_puzzle_count` |

A preset that documents a protocol REMind cannot run (`konkoly2021_paris`, `sanders2019_puzzle_sws`) is refused with exit code 2 before anything is read or written, and so is an unknown name.

What a run records is the preset's manifest, a `protocol` block:

```json
{
  "schema_version": 1,
  "name": "konkoly2026_puzzle",
  "version": 1,
  "registry_version": 1,
  "content_hash": "<sha256 of the preset's canonical JSON>",
  "runnable": true,
  "overrides": {"cued_puzzle_count": 3},
  "effective_hash": "<sha256 of the content hash and the overrides>",
  "settings": {"gate_min_stable_seconds": 60.0, "...": "..."}
}
```

`metadata.json` of a Pilot 4 or 5 night, `pilot4_summary.json` or `pilot5_summary.json`, a Pilot 3 report and `analyze-cued-uncued` (from `--protocol-metadata`, or the assignment's block) keep it as `protocol`; a TLR block plan, a cue assignment, a night puzzle session and a TLR cue made by `create-tlr-cue` keep it under `metadata.protocol`. When the assignment or the TLR plan a night uses was made with another preset, the night prints a warning, runs, and names those files' presets under `protocol.inputs`.

Two rules hold for every preset, and the tests check them:

- No preset sets a volume. The builders never touch `max_volume`, `default_volume`, `volume_hint`, the 0.20 hard cap or the calibrated comfortable volume. A paper's dB SPL figures and intensity ramps are text and fidelity-report inputs; REMind plays every cue at the gain its player computes under the calibration, lowered only by the arousal guard.
- No preset lowers the REM gate's 60 s stable window. A paper that cued sooner keeps the 60 s and says so as a deviation.

Changing anything in a preset, a number or a sentence, changes its content hash. Raise its `version` and renew the pinned hash in `tests/test_protocol_presets.py` in the same commit.

## The presets

| Preset | Parameters from | Runs | Checked in |
| --- | --- | --- | --- |
| `remind_default_v1` | the repository's defaults, no published basis | yes | |
| `konkoly2021_northwestern` | Konkoly et al. 2021, Northwestern arm (USA group) | yes | full text |
| `konkoly2021_nijmegen` | Konkoly et al. 2021, Nijmegen arm (Netherlands group) | yes | full text |
| `konkoly2021_osnabrueck` | Konkoly et al. 2021, Osnabrück arm (Germany group) | yes | full text |
| `konkoly2021_paris` | Konkoly et al. 2021, Paris arm (France group) | no | full text |
| `konkoly2024_exp1` | Konkoly et al. 2024, Experiment 1 | yes | full text |
| `konkoly2024_exp2` | Konkoly et al. 2024, Experiment 2 | yes | full text |
| `konkoly2026_puzzle` | Konkoly et al. 2026 | yes | full text |
| `borghese2022_rem_home` | Borghese et al. 2022 | yes | full text |
| `dresler2024_multisensory` | Jafarzadeh Esfahani, Picard-Deland et al. 2024, preprint version 1, auditory channel only | yes | full text |
| `schwartz2022_irt_rem` | Schwartz et al. 2022 | yes | research notes only |
| `sanders2019_puzzle_sws` | Sanders et al. 2019 | no | full text |

"Full text" means the methods were read in the paper's full text for this registry (`source_check: verified_full_text`); "research notes only" means the figures come from `docs/research/2026-10-uniqueness/notes/` (`notes_only`). The papers are in [`references.md`](references.md); every figure below has a row in [`numbers.md`](numbers.md).

### `remind_default_v1`

Today's defaults: the gate's 60 s stable window, puzzle cues at least 30 s apart and at most 4 per REM block, a 120 s scheduler cooldown, a TLR block of 3 cues 8 s apart and a 10 s pause, a TLR tone of 396 Hz for 1 s, 4 puzzles a night with half of them cued. None comes from a paper.

The TLR tone has no rationale either. 396 Hz first appears on 2026-05-12 in commit `b932d08`, as the starter tone "for TLR cue smoke tests" next to a 528 Hz starter puzzle tone, and moved into `TlrCueConfig` with the TLR module (issue #25); nothing in the repository says why 396 Hz, 1 s or 3 repetitions. The cited presets that replace the tone use the paper's cue where one generated tone can stand in for it: 600 Hz for 0.65 s for the three rising beeps of 400, 600 and 800 Hz that Konkoly et al. 2021, 2024 and 2026 used.

### `konkoly2021_northwestern`

Published: three pure-tone beeps rising in pitch (400, 600, 800 Hz), about 650 ms in all, at about 40 to 45 dB SPL; before the nap the cue up to 15 times over 15 min, the first 4 with spoken guidance; in REM the cue about every 30 s while REM stayed stable, and after about 10 cues without a lucid signal the math problems began. REMind runs a TLR block of 10 cues 30 s apart with a 600 Hz tone of 0.65 s. It differs in the cue (one tone, not three), the level (gain, not dB SPL), REM (detector, not PSG) and what follows the cues (no eye signals, no math problems). The paper is about two-way communication in lucid REM, not about how often cues induce lucidity.

### `konkoly2021_nijmegen`

Published: TLR cues, beeps with blinking lights on white noise of at most 45 dBA, after the first 30 s epoch of REM, one every 10 to 15 s, up to 10 per REM period, paused while the EEG showed an arousal and resumed at REM onset. REMind runs 10 TLR cues 15 s apart. Its first cue comes later: the gate's 60 s window holds, 60 to 85 s after a REM onset at a 30 s stride. At a 30 s stride the scheduler emits TLR cues once per epoch, so cues planned 15 s apart play two at a time; `--stride-seconds 15` comes closer.

The research report put these timings under the Paris arm. The paper's STAR Methods give them for the Netherlands group (Donders Institute, Nijmegen); the Paris procedure is `konkoly2021_paris`.

### `konkoly2021_osnabrueck`

Published: after 5 min of stable REM, Morse-coded math problems as tones at 470 or 600 Hz or LED flashes, louder with each new problem, until the sleeper answered, had an arousal or went into NREM, which stopped the stimulation. Only the start and stop rules transfer: REMind waits for 300 s of stable REM (`gate_min_stable_seconds` 300) and plays its own cues. A guard pause lasts 120 s and cueing can resume in the same REM period, where the paper stopped; the fidelity report counts cues after a pause.

### `konkoly2021_paris` (documentation only)

One participant with narcolepsy: stimulation started 1 min into a daytime nap while still awake, in 10 blocks of 6 spoken words every 10 s with a jitter, 1 min of white noise between blocks. It was not gated on a sleep stage, so REMind cannot run it.

### `konkoly2024_exp1`

A phone app without sleep staging: the TLR cue (rising beeps of about 650 ms or a 1000 ms violin sound) every 20 to 40 s from 6 h after the evening training, each cue 0.16 % louder from the perceptual discrimination threshold, paused until 5 min after a movement and until 45 min after the sleeper said a cue woke them. REMind gates the cue on REM, which the authors name as the obvious improvement, plays TLR cues 30 s apart, and holds every guard pause at least 300 s through the scheduler cooldown. It does not ramp the volume, and it does not apply the 45 min pause during the night: awakening markers are logged, not read by the scheduler; the fidelity report checks them against the next cue.

### `konkoly2024_exp2`

The public version of the app: two harp melodies of about 3 s with fades, movement pausing cues for 1 min, the intensity capped where the sleeper reported the sound in a dream, and an optional slow mode of 0.08 % per cue that stopped for the night after an arousal. The cue interval and the awakening pause are not restated for this experiment; Experiment 1's are used. REMind plays a generated 396 Hz tone of 3 s, 30 s apart.

### `konkoly2026_puzzle`

Lab nights with PSG: four unsolved puzzles a night, each with a 15 s soundtrack (8 to 12 s of sound, 3 to 7 s of silence) and a spoken title, two of them cued in REM after about 10 TLR cues, cues every 15 to 30 s from the lowest detectable volume, paused and lowered on any sign of arousal. In the morning a puzzle-sound association test came first, then a 4 min attempt per puzzle, and a follow-up about 10 days later. REMind runs 4 puzzles with 2 cued, a TLR block of 10 cues and puzzle cues 30 s apart, and the morning sound check below. Its scheduler plays each cued puzzle's sound once a night, where the paper repeated them in blocks, and Pilot 4 and 5 play generated tones, not soundtracks. The paper found no overall benefit of the cues on solving.

### `borghese2022_rem_home`

A home headband with automatic staging: once REM had lasted more than 5 min, a 1 s piano chord every 10 s; a new stage stopped the sounds until 5 min of REM again; movement paused them 90 s, alpha 45 s, a blink 10 s and a poor signal 4 s; each participant chose the volume. REMind waits for 300 s of stable REM and plans puzzle cues 10 s apart, which only a `--stride-seconds 10` run can reach. Its arousal guard pauses 120 s, longer than the 90 s and 45 s; it has no blink rule, and a poor signal lowers the volume instead of a 4 s pause.

### `dresler2024_multisensory`

Lab naps with a ZMax headband (preprint): cues about 20 s after the first rapid eye movement with low EMG, visual, auditory and tactile in turn about every 20 s, each cycle 5 dBA louder from the lowest level judged not arousing (from 40 dBA down in 5 dBA steps), on white noise of about 35 dBA; a micro-arousal halted cueing until it passed, then cueing went on one step lower. On a Muse headband only the auditory cue is left: REMind plays its TLR tone 20 s apart. The first cue comes 60 to 85 s after a REM onset instead of about 20 s, and there is no ramp.

### `schwartz2022_irt_rem`

One sound paired with a rehearsed positive version of a nightmare during imagery rehearsal therapy, then played in REM at home for two weeks by a headband that detected sleep stages; nightmares per week was the outcome, with a follow-up at 3 months. From the research notes only: the cue timing, volume and pause rules are not in the abstract, so REMind's defaults stand.

### `sanders2019_puzzle_sws` (documentation only)

The cued and uncued puzzle design REMind follows: each evening puzzles until 6 stayed unsolved, each with a 15 s clip looping during a 2 min attempt, then three rounds of sound-to-puzzle recall with feedback; at home a Zeo headband played a puzzle's sound for 3 min after 1.5 min of slow-wave sleep within 2 min, for 3 of the 6 puzzles, over pink noise from sleep onset; in the morning the sound-to-puzzle recall test, then one 4 min attempt per puzzle. Its cues target slow-wave sleep and REMind has no NREM gate, so it does not run. A correction to the paper was published in 2020; its text was not read.

## The fidelity report

```bash
muse-tmr protocol-fidelity data/recordings/pilot5_full_night_<timestamp>
muse-tmr protocol-fidelity data/recordings/<night> --pilot3-report data/reports/<night>_pilot3.json
```

It compares a night with its preset: with the one in `metadata.json`, or with `--protocol-preset` another one; a night without a block is compared with `remind_default_v1`. It reads `scheduler_events.jsonl`, `audio_playback.jsonl`, `rem_gate_decisions.jsonl`, `awakening_events.jsonl` and `metadata.json`, or a Pilot 3 report's plan and epochs, and writes `protocol_fidelity.json` and `protocol_fidelity.md` into the night's directory, the Pilot 3 report's, or `--output-dir`.

A REM period is a gate-open run as `muse_tmr.validation.gate_log` defines it, the same as for the cue arousal scorer, and a cue is placed at the decision of the epoch that played it. The checks:

- `inter_cue_intervals`: the time between cues of one stream within a REM period against the published range;
- `cues_per_rem_period`: against a published limit;
- `stable_rem_before_first_cue`: the gate's stable count when the first cue of a period played, against the published start (or resume) rule;
- `pause_rules`: every arousal guard pause against the published pause for its trigger, every awakening marker against a published awakening pause, cues outside REM periods, and cues after a guard pause in the same period where the paper stopped;
- `volume_ramp`: the gain of the cues within each period against a published ramp (`not_implemented` when the paper ramped and the night did not);
- `cue_duration`: against the published cue length;
- `tlr_block`: the TLR cues per period and their spacing against the plan in `metadata.json`, the played spacing against the planned one, and whether a period played the whole block before it ended (`complete`).

Each check reports `as_published` (or `as_planned`), `differs`, `later_than_published`, `earlier_than_published` or `not_implemented`, or why there was nothing to compare (`not_published`, `not_checkable` for a rule without a duration, `nothing_to_check`, `no_cues`, `no_intervals`, `no_plan`, `not_available`, `not_used`). For `remind_default_v1`, which cites no paper, "published" stands for its own settings. Every report carries `parameter_fidelity_is_not_result_fidelity`: no PSG-verified REM, no EOG, gain and not dB SPL, and a first cue 60 to 85 s after a REM onset, later than the Nijmegen arm of Konkoly et al. 2021 and than Jafarzadeh Esfahani, Picard-Deland et al. 2024 (#167). Matching a paper's parameters says nothing about matching its results.

## The morning sound check

Sanders et al. 2019 and Konkoly et al. 2026 played each puzzle's sound in the morning before the retest and asked which puzzle it belonged to. Sanders et al. report 71.9 % of the pairings recalled, 70.3 % for cued and 73.5 % for uncued puzzles. REMind's version is `muse-tmr record-cue-recall`, or the "Which puzzle does this sound belong to?" step of the app's Morning tab:

```bash
muse-tmr record-cue-recall data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --calibration data/calibration/volume_calibration.json \
  --output data/reports/night-001_cue_recall.json \
  --backend system
```

Every puzzle of the night gets one sound, cued and uncued alike, in an order shuffled with a seed made from the session ID alone, so the order, the number of sounds and their labels ("Sound 1" to "Sound N") are the same whichever puzzles were cued. The sounds play through `AudioCuePlayer`, with the backend resolved as for a night (`system` without `afplay` exits with code 2 unless `--allow-dry-run`) and each cue's volume hint capped by the calibrated comfortable volume and 0.20. The command asks for a puzzle number after each sound (`r` plays it again, Enter means "not sure"), prints only how many were answered, and saves `cue_condition` per trial for the analysis. `analyze-cued-uncued --cue-recall` adds the recall rate by condition and the limitations `cue_recall_missing` and `cue_recall_after_retest`.
