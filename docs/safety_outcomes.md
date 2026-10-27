# Safety outcomes: arousal proxies after cues

The arousal guard decides before a cue, from 30 s epochs. This page is about what the
recording shows after one. `muse_tmr.reports.cue_arousal` scores every cue of a night
against a quiet control moment, from the night's own logs and its raw recording, and
reports how often an arousal proxy followed a cue and how often one followed a moment
without a cue (#227). It is the safety outcome the cueing studies report, and the first
night with sound is hard to read without it.

Every flag below is a proxy. The Muse has forehead and temporal EEG only: no occipital
alpha, no chin EMG, no EOG, and nothing here was compared with arousals scored on PSG.
The thresholds are UNFITTED starting values. A cue's volume is a software gain from 0 to
1, not a sound level. None of these figures measures a benefit of cueing.

## How a window is scored

A cue is a scheduler `play`. Its moment is the end of the epoch whose decision played it,
the time Pilot 4 hands it to the player; the sound starts a fraction of a second later.
Pilot 4 plays it on a worker thread, so the event loop keeps taking the headband's
packets while the tone plays and they keep the time they came.
A TLR cue logged at its planned time is placed at the end of the first epoch that starts
at or after it, the epoch that plays it. Times count from the first epoch's start, which
Pilot 3, 4 and 5 now record as `first_epoch_start`.

For each cue the scorer cuts the 20 s before it (the baseline) and the 10 s from it (the
post window) out of one replay pass over `raw_amused.bin`, and sets five flags:

| Flag | Signal | Rule |
| --- | --- | --- |
| `alpha` | EEG 8 to 12 Hz | The largest 3 s mean power after the cue is at least 2 times the median 3 s mean power of the baseline. |
| `beta` | EEG 15 to 30 Hz | The same, ratio 2. |
| `emg` | EEG 55 to 95 Hz | The same, ratio 2, with the optics line near 64 Hz fitted and taken out of each 1 s segment and its bins bridged (`complexity_features.bridge_lines`). |
| `imu` | Accelerometer and gyroscope | A movement that `imu_features` calls an arousal proxy (0.20 g, 60 deg/s or motion level 2) from 2 s before the cue to 10 s after it. |
| `hr` | Optics heart-rate series (#251) | A used value centred in the post window at least 10 bpm above the median of the used values centred in the baseline. |

`any` is set when one of them is. The EEG is put on one sample clock per window, as the
optics clock of `ppg_features` is: samples the decoder reports lost, and a BLE gap, stay
missing and are never read as zeros. The decoder's subpacket counter reports any loss of
up to 127 subpackets (1.98 s of 4-channel EEG, 0.99 s of 8-channel), so a frame that
comes later than the clock expects by less than that, with no loss reported, only came
late and keeps its place; a longer lateness counts as a gap. A window that misses more
than 20 % of its baseline or post window gets `eeg_gap` and no EEG flags; a channel that
clips or is flat before the cue is left out. The band powers come from 1 s Hann segments
every 0.5 s, averaged over the usable channels and over 3 s, the shortest arousal AASM
scoring counts. The heart-rate flag reads only values the series used; a window with
fewer than three on either side gets `hr_unavailable`, and one without optics, by preset
or by dropout, `ppg_absent`. Neither is a flag. The decoder's `heart_rate` stream is not
read: it jumps by 10 bpm or more between values 36 to 77 ms apart.

The ratios are taken against the moment's own baseline, so a slow change of contact or
stage moves both. Absolute band power is used, not relative: relative alpha rises when
delta falls, which a move from N3 to N2 does without any arousal.

## Controls

A control is a gate decision in the same gate state as its cue, more than 60 s from every
`play` of the night, with its whole window inside the recorded epochs. Each played cue, in
time order, takes the nearest such moment that is at least 30 s from the controls already
taken, so no two control windows overlap. A play the player blocked or skipped is left
out of the cue rates and gets no control. A cue without one stays unpaired and the
report lists it (`unpaired_cues`). Controls are scored with the same windows and rules,
so `arousal_proxy_per_control` is the night's background rate for the same proxies, and
the difference is what can be read against the literature, not the cue rate alone. The
background is not zero: on synthetic input the max-over-median rule passes 2 now and then
on quiet stretches, and in REM eye movements and theta raise frontal power without an
arousal; a control in the same gate state carries both.

## Outcomes

| Outcome | Definition |
| --- | --- |
| `arousal_proxy_per_cue`, `arousal_proxy_per_control` | Per flag and `any`: flagged windows over windows where that flag could be scored. Cues the player blocked or skipped are left out. |
| `arousal_proxy_difference` | Cue rate minus control rate, per flag. `paired_any` counts the pairs where both, only the cue, only the control or neither raised `any`. |
| `movement_events_per_cue`, `movement_events_per_control` | Mean count of movement events in the IMU window, the measure of Konkoly et al. 2024. |
| `cue_attributed_awakening_rate` | Played cues followed within 120 s by a marker in `awakening_events.jsonl`. `awakening_attribution_counts` adds what the sleeper said (`--attributed-to-cue`). |
| `rem_periods_with_disruption_fraction` | REM periods with a played cue in which some cue raised `any` or the period ended within 60 s of a cue, over REM periods with a played cue. A REM period is a gate-open run (`muse_tmr.validation.gate_log`), not a scored REM period. |
| `rem_termination_within_60s_of_cue_rate`, `..._of_control_rate` | Cues, and controls, after which their gate-open run closed within 60 s. A run that ends with the night does not count. |
| Dose tables | Played cues by `effective_volume` from `audio_playback.jsonl` (#256), by the time since the cue before and by `puzzle_cue_index` / `tlr_event_index`: cues and the `any` rate per bin. Each cue also carries the guard's action and multiplier from its play event. |

## What the studies report, and what is measured here

| Study | Their figure | What REMind computes instead |
| --- | --- | --- |
| Whitmore & Paller 2023 (lab, PSG) | AASM-scored arousals within 10 s after 57.9 % of loud (about 43 dBA) and 43.0 % of soft (about 28 dBA) cues | `arousal_proxy_per_cue`: EEG, IMU and optics proxies in the 10 s after a cue, against a control rate. No AASM scoring and no dBA. |
| Konkoly et al. 2024 (home app, no EEG) | 0.19 movement events per cue against 0.10 in silent periods | `movement_events_per_cue` against `movement_events_per_control`, from the headband's IMU in REM gate-open time. |
| Picard-Deland et al. 2024 (lab naps, preprint) | Sleep disrupted in 23.7 % of cued REM periods | `rem_periods_with_disruption_fraction` over gate-open runs, which are not PSG-scored REM periods. |

The study figures come from different devices, stages and scoring; they are a sense of
scale, not a target or a ceiling. No validator fails on any of these outcomes, and no
threshold here may be changed to fit a synthetic result (`docs/numbers.md`).

## Running it

`run-pilot4-cueing` and `run-pilot5-full-night` run the scorer on their own recording when
the night ends and put a `cue_arousal` block with its rates, limitation codes and file
paths into the summary; when it fails the block says `failed` and why, and the night's
summary is kept. The same scoring on any night:

```bash
muse-tmr score-cue-arousal data/recordings/pilot4_low_volume_<timestamp>
```

On a replayed night, Pilot 3 can add a synthetic cue to gate-open epochs every N seconds,
outside the scheduler and with mock audio, to check where the windows fall before a night
with sound:

```bash
muse-tmr simulate-replay-cues <recording> ... --inject-cues-every-seconds 180 \
  --output data/reports/pilot3_injected.json
muse-tmr score-cue-arousal <recording> --pilot3-report data/reports/pilot3_injected.json
```

Each writes `cue_arousal_events.jsonl` (one line per cue and control window),
`cue_arousal_report.json` and `cue_arousal_report.md`; their fields are in
[`data_model.md`](data_model.md). They are local outputs and stay out of git.

## Limitations the report always carries

- `thresholds_unfitted`: the rise ratios (2) and the heart-rate jump (10 bpm) are set by
  hand, not fitted on any night.
- `arousal_proxy_only`: forehead EEG, IMU and optics proxies; no PSG comparison.
- `software_gain`: the effective volume is a gain from 0 to 1.
- `descriptive_only`: no significance test.

Others appear when they apply: `synthetic_input`, `no_audio_played` (Pilot 3, mock or
dry-run audio, synthetic cues), `time_base_inferred` (a summary from before
`first_epoch_start`, where the first replayed frame stands in for the first epoch),
`gate_log_missing` (no controls and no REM periods), `small_n` (fewer than 10 played
cues), `unpaired_cues`, `cue_windows_overlap`, `cue_not_played`, `audio_log_missing`,
`audio_log_mismatch`, `eeg_gap`, `eeg_unusable`, `imu_gap`, `hr_unavailable`,
`ppg_absent`, `awakening_log_missing`, `awakening_time_from_utc`, `awakening_unplaced`,
`no_cues` (the night logged no play), and on a night with an arm (#228) `sham_arm` (every
cue was the untrained sham sound) or, on a no-cue night instead of `no_cues`,
`no_cue_arm` (nothing played by design; its metadata counts the virtual plays).

## On synthetic input, not sleep

No night with sleep-time audio has been run yet, so there are no figures from a real
night here. The synthetic nights only show that the windows sit where they should:

- `tests/test_cue_arousal.py` puts cues at the 7 arousals outside wake of a 45 min
  `clean_night` (seed 0) and at 18 quiet N2 and N3 moments. All 7 arousal cues raise
  `any` (alpha on 7, EMG on 6, IMU on the one arousal with movement), and 1 of the 18
  quiet ones does (an alpha rise of 2.08).
- The 8 h `clean_night` (seed 0) through `simulate-replay-cues --preset p21
  --inject-cues-every-seconds 180` gives 25 cues (the 2 planned puzzle cues, at sleep
  onset, and 23 injected) and 14 controls. `any` comes up after 3 of 25 cues and 2 of
  14 controls: two cues and one control where an arousal of `arousals.csv` starts inside
  the post window, and one of each without one. No cue made a sound and nothing is
  expected to differ; the dose table by volume has only `unknown`, as Pilot 3 has no
  player.

Neither result may change a threshold (`AGENTS.md`).
