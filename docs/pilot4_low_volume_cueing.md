# M8 Pilot 4: Low-Volume REM-Gated Cueing Night

Pilot 4 is the first sleep-time cueing workflow. It uses real audio only when the
operator explicitly selects a playback backend such as `system`, and it requires a
saved Pilot 2 volume calibration before any cue can be played.

## Goal

- Record a normal Muse session while running epochs, REM detection, stable REM gate,
  arousal guard, and TMR scheduler.
- Play only scheduler-approved puzzle cues after stable REM.
- Enforce the calibration comfortable-volume cap and the hard max-volume cap.
- Write auditable logs for scheduler events, arousal guard decisions, audio playback,
  emergency stop, and manual awakenings.
- Preserve raw recording outputs locally; do not commit generated data.

This is still an exploratory pilot. Do not use it to claim effectiveness or lucid
dream induction.

## Preflight

Required before using `--backend system`:

- Pilot 2 calibration exists and passed validation.
- Pilot 3 replay simulation passed on a recent recording.
- macOS output device is set to the intended sleep headphones.
- You know the emergency stop file path for the run.
- Cue metadata uses `generated_tone` puzzle cues with low `volume_hint` values.

Do a dry-run first:

```bash
cd /path/to/remind-eeg

.venv/bin/python -m muse_tmr.cli.main run-pilot4-cueing \
  --source amused \
  --address "$MUSE_ADDR" \
  --duration-seconds 300 \
  --allow-short \
  --output-dir data/recordings/pilot4_dry_run \
  --catalog data/protocol/puzzle_catalog.json \
  --session data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --calibration data/calibration/volume_calibration.json \
  --device-name "Sleep Headphones" \
  --backend dry-run
```

## Run With Audio

Use `--backend system` only when you are ready for real low-volume playback:

```bash
OUTDIR="data/recordings/pilot4_low_volume_$(date +%Y%m%d_%H%M%S)"

.venv/bin/python -m muse_tmr.cli.main run-pilot4-cueing \
  --source amused \
  --address "$MUSE_ADDR" \
  --duration-hours 2 \
  --output-dir "$OUTDIR" \
  --catalog data/protocol/puzzle_catalog.json \
  --session data/protocol/night-001_puzzles.json \
  --assignment data/protocol/night-001_assignment.json \
  --cue-library data/cues/starter.json \
  --calibration data/calibration/volume_calibration.json \
  --device-name "Sleep Headphones" \
  --backend system \
  --default-volume 0.02 \
  --hard-max-volume 0.20
```

The default duration rules require 2-8 hours unless `--allow-short` is present.

If `afplay` is not on PATH, the command exits with code 2 before it connects to the
headband or creates `$OUTDIR`, instead of running the night without a sound. To run it
silently on purpose, add `--allow-dry-run`. `metadata.json` then records
`requested_backend_name` `system` and `resolved_backend_name` `dry-run`, and
`pilot4_summary.json` has the same pair as `requested_audio_backend_name` and
`resolved_audio_backend_name`.

`--preset` (default `p1034`) is the headband preset and also picks the REM gate
thresholds: 0.80/0.70 on `p1034`, still provisional, and 0.90/0.85 on `p21`, where
the gate does not count the missing PPG against REM confidence. `--enter-threshold`
and `--exit-threshold` override the pair; see `docs/data_model.md`. The same preset
sets the arousal guard's `optics_capable`: on `p21` missing PPG neither lowers volume
nor leads to a `repeated_artifact_quality` pause, the HR rules are skipped, and
`arousal_guard_events.jsonl` records `ppg_absent_by_design` on each decision. Motion,
alpha and EEG artifact rules still apply. Pilot 5 uses the same wiring.

On `p1034` the HR rule (`sudden_heart_rate_change`) compares heart rates from each
epoch's optics, one value a second from 10 s windows, 10 to 20 s apart, and the last
values of the previous epoch with the first of this one (#247). Each decision in
`arousal_guard_events.jsonl`, an `allow` too once a value was skipped, says how many
values were used and how many were too noisy to use (`hr_series_used_count`,
`hr_series_skipped_count`). Skipped values never pause for heart rate. Under a movement
the motion rules still pause, but on a still stretch with poor optical contact no artifact
flag acts: none covers noisy optics. When more than half of an epoch's values were
skipped the decision says `hr_rule_blind` and lowers the volume (#234,
`ArousalGuardConfig.hr_rule_blind_action`), unless a pause or stop reason wins in the same
epoch. A cue played then shows it in `scheduler_events.jsonl` (`arousal_guard_action`
`lower_volume`, `arousal_guard_reason_codes`, `volume_multiplier` 0.5) and in
`audio_playback.jsonl`. The halving comes after the calibration and hard caps: both logs
give the cue's `requested_volume` (its `volume_hint`, or without one the smallest of
`--default-volume`, the comfortable volume and `--hard-max-volume`), the
`capped_volume`, the multiplier and the `effective_volume` that plays, half the capped
one (#255).
`--guard-min-ppg-confidence` and `--guard-min-hr-series-quality` (default 0,
off) turn the HR rules off on epochs of low PPG quality, logged as `hr_rule_skipped_by`;
leave them off for a night until `docs/optics_agreement.md` has a paired table that
supports a value.

## Emergency Stop

The summary prints an `emergency_stop` path. Creating that file blocks future playback
while allowing the run to keep logging:

```bash
touch "$OUTDIR/EMERGENCY_STOP"
```

The audio player records an emergency-stop event in `audio_playback.jsonl` and future
cue attempts are logged as blocked.

## Manual Awakening Markers

If you wake during or after the pilot and want to mark it:

```bash
.venv/bin/python -m muse_tmr.cli.main log-pilot4-awakening \
  "$OUTDIR/awakening_events.jsonl" \
  --attributed-to-cue unsure \
  --notes "woke briefly; no clear cue recall"
```

`--attributed-to-cue` (`yes`, `no` or `unsure`) records whether you think a cue woke
you. The marker gets the current time in UTC, or `--timestamp-utc`; `--timestamp-seconds`
places it on the night's time base instead, seconds from the first epoch as in
`scheduler_events.jsonl`. Use short factual notes. Keep the file local unless you
intentionally sanitize it.

## Cue Arousal Proxies

When the night ends, Pilot 4 scores what followed each cue (`docs/safety_outcomes.md`):
for every scheduler `play`, the 10 s after it against the 20 s before it, for an alpha,
beta or 55 to 95 Hz EMG-band rise, an arousal-proxy movement and, on `p1034`, a jump of
the optics heart-rate series; and the same at a quiet control moment, in the same gate
state and more than 60 s from every cue, paired with each cue. It replays the night's
own `raw_amused.bin` once, which takes a few minutes for a full night, after the
summary is written. `pilot4_summary.json` then gets a `cue_arousal` block with the
per-cue and per-control rates, the limitation codes and the paths of
`cue_arousal_events.jsonl`, `cue_arousal_report.json` and `cue_arousal_report.md`; if
the scorer fails, the block says `failed` and why, and the rest of the summary stands.
The block counts the awakening markers logged by the end of the night. Run it again
after adding markers in the morning; it rewrites the report and the block:

```bash
.venv/bin/python -m muse_tmr.cli.main score-cue-arousal "$OUTDIR"
```

The flags are arousal proxies with UNFITTED thresholds, and the effective volume in the
dose tables is a software gain. No pass criterion reads them.

## Outputs

The output directory contains:

- `raw_amused.bin`, `metadata.json`, `events.jsonl`;
- `scheduler_events.jsonl`;
- `arousal_guard_events.jsonl`;
- `rem_gate_decisions.jsonl`, the gate's decision for each epoch;
- `audio_playback.jsonl`;
- `awakening_events.jsonl`;
- `cue_arousal_events.jsonl`, `cue_arousal_report.json`, `cue_arousal_report.md`;
- `pilot4_summary.json`.

The summary passes only when:

- at least one epoch was processed;
- scheduler events were generated;
- every scheduler `play` event contains `rem_gate_open`;
- uncued puzzles received zero scheduler `play` events;
- effective playback volume stayed within calibration and hard caps;
- arousal guard and emergency stop artifacts are available.

An empty cue night can still pass if stable REM or safety conditions were not met.
That means the system refused to cue rather than forcing audio.

None of these criteria shows that a cue was heard. `resolved_audio_backend_name` in the
summary says what played the cues: `afplay` for sound, `dry-run` for a night run with
`--allow-dry-run`.

The summary also has `stride_seconds`, `first_gate_open_seconds` and
`first_play_seconds`: seconds from the start of the first epoch to the end of the epoch
whose decision first opened the gate or first played a cue, `null` when that never
happened. `first_epoch_start` is that first epoch's start in the recording's clock, so
every `timestamp_seconds` of the night can be put on the raw recording. The gate counts the time between epoch starts, so its 60 s stable window and
120 s cooldown take as long at `--stride-seconds 10` as at the default 30; see
`docs/pilot3_replay_cue_simulation.md` for what a shorter stride does to the first cue.
