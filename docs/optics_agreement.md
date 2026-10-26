# Optics heart rate against a Polar H10 ECG

The Muse S Athena estimates heart rate from its forehead optics in four ways, and until
[#234](https://github.com/yuriy-mikityuk/remind-eeg/issues/234) none of them had been
compared with a beat-level reference. `muse-tmr analyze-optics-agreement` compares them
with the heart rate from the R-peaks of a Polar H10 ECG recorded in the same session,
and scores the arousal guard's sudden heart-rate change rule against the ECG. It is
agreement with a chest ECG, a proxy for heart rate: no PSG takes part, nothing here
says how the optics behave asleep unless the session was a night, and nothing is
estimated about blood oxygen or sleep stages.

Code: `muse_tmr.reports.optics_agreement`. It reads the recording and the feature rows
only; it imports nothing from `muse_tmr.audio`, `muse_tmr.models` or
`muse_tmr.protocol`, and the Polar data never reaches `MuseFrame`, the detector, the
gate, the guard, the scheduler or audio (`docs/polar_h10.md`).

## Recording a paired session

The optics run only on optics presets, so record on `p1034` with the H10 (wet strap,
logo upright; setup in `docs/polar_h10.md`):

```bash
muse-tmr record --source amused --preset p1034 --duration-seconds 1800 --allow-short \
  --output-dir data/recordings/session/<name> --with-polar
```

The app's session kind with Polar records the same. `calibration-run` defaults to `p21`,
which has no optics: pass `--preset p1034` to it when the session should also serve
here. Sessions of 20 to 60 min in three postures (for example sitting, on the back and
on the side) give the comparison some spread; a `p1034` night with the H10 until the
battery ends ([#108](https://github.com/yuriy-mikityuk/remind-eeg/issues/108)) is the only
way to see it asleep.

## Running the report

```bash
muse-tmr analyze-optics-agreement data/recordings/session/<name> \
  [--epoch-seconds 30] [--output-dir data/reports/optics_agreement/<name>/]
```

It writes `optics_agreement.json` and `optics_agreement.md` into the output directory,
by default under `data/reports/`, which is gitignored, and prints the main rows. It
exits with code 2 and says why when the recording has no `polar/` folder, the H10
recorded no ECG (`--no-ecg`), the recording's preset streams no optics (`p21`) or the
replay holds no optics samples. When `polar/` has only the raw log, loading it writes the
decoded files next to it, as `decode-polar` does. The Markdown holds no device name or
address; paste only its tables, not the recording's `metadata.json`.

## What is compared

The ECG reference is the H10's R-peaks outside the spans it had no skin contact or
noise like it (`PolarSession.no_contact`). Beat intervals of 300 to 2000 ms that do not
cross such a span are cleaned with `correct_rr` (an ectopic or missed beat), and a
window's heart rate is 60 over their mean when the intervals inside the window span at
least `min_ecg_coverage` (0.7) of it.

| Estimator | What it is | Compared per |
|---|---|---|
| `decoder_heart_rate` | The epoch mean of the decoder's `heart_rate` stream: peaks on the first optics channel (`LO_NIR` on `p1034`) over its last 2 to 5 s, re-estimated on every optics notification | 30 s epoch |
| `ppg_estimated_hr_bpm` | The PPG row's peak-interval estimate on the channel with the largest variance | 30 s epoch |
| `channel` | The same estimate on each optics channel, `estimate_channel_heart_rates()` | 30 s epoch |
| `hr_series` | The used values of the heart-rate series the sudden-change rule reads (`heart_rate_series()`) | its 10 s window |
| `hr_series_skipped` | The values the series skipped, where they have a number | its 10 s window |

An epoch whose optics cover less than `min_optics_coverage` (0.9) of it is left out of
the epoch rows. For each estimator the report gives the number of windows with a
reference, the coverage (the share of them with an estimate), the bias (optics minus
ECG), its SD, the 95 % limits of agreement (bias ± 1.96 SD, Bland and Altman 1986, see
`docs/references.md`) and the mean absolute error. Each is split by quality: the epoch
estimators by the epoch's `ppg_signal_quality`, its `ppg_confidence` (bins of 0.2) and
the median quality of its series values (`hr_series_median_quality`); each channel by
its own estimate's quality, confidence and periodicity (the autocorrelation peak the
series picks its channel by); the series by its epoch's `ppg_signal_quality` and
`ppg_confidence`, the fields the guard's gates read, and by each value's own quality.

## The sudden heart-rate change rule against the ECG

The report runs the rule over the epochs as the arousal guard runs it on an optics
preset by default: series changes of 9 bpm or more (`sudden_hr_change_pause_bpm` 10 less
the 1 bpm margin) between values 10 to 20 s apart, and the largest change from the
previous epoch's values within 20 s, counted once when a row change holds the same step.
The ECG change of an epoch is its largest between the windows the rule pairs. A fired
epoch is a false jump when neither it nor the epoch before changed by 9 bpm or more on
the ECG: a step at an epoch boundary pauses the epoch after it, and #251 counted steps
found the same way. An ECG change of 10 bpm or more is one ECG jump, also when the next
epoch sees it again: the ECG window over a step averages both rates, so the next epoch's
windows, paired with it, can change by 10 bpm or more in the same direction over
overlapping windows. The jump is missed when none of its epochs and not the one after
them fired. Epochs without an ECG change to compare are not scored.

The table splits the epochs by the same quality fields and by `hr_rule_blind` (more
than half the series values skipped). A second table runs the rule with a gate on
`ppg_confidence` or on `hr_series_median_quality` at 0.2 to 0.9, as
`min_ppg_confidence_for_hr_rule` and `min_hr_series_quality_for_hr_rule` make the guard
do: a gated epoch neither fires nor feeds the next epoch's comparison. A third gives,
for the epochs that fired, the median of each field over the true and the false fires
and their AUC, the chance that a true fire has the higher value. The JSON also lists
every change the rule counted with the ECG change between the same two windows; over a
window that holds a step the ECG averages both rates while the series reads its longer
run of regular beats, so those two can differ more than the epochs do.

## Example (synthetic input, not sleep)

From the fixtures of `tests/test_optics_agreement.py`: optics frames and an H10 session
written by `tests/polar_synthetic.py` from the same scripted beats, ten minutes around
60 bpm, with `RO_IR` replaced by a 63.93 Hz line (folded to 0.07 Hz at 64 Hz) and white
noise. The frames carry no decoder stream, so `decoder_heart_rate` has no estimate.

| Estimator | Channel | Windows | Coverage | Bias | SD | 95 % limits | MAE |
|---|---|---:|---:|---:|---:|---|---:|
| `ppg_estimated_hr_bpm` |  | 20 | 100 % | +0.0 | 0.1 | -0.1 to +0.2 | 0.1 |
| channel | LI_NIR | 20 | 100 % | +0.0 | 0.1 | -0.1 to +0.2 | 0.1 |
| channel | RO_IR | 20 | 100 % | +28.4 | 5.5 | +17.7 to +39.1 | 28.4 |
| `heart_rate_series`, used |  | 420 | 100 % | -0.0 | 0.2 | -0.4 to +0.3 | 0.2 |

The peak-interval confidence behind `ppg_signal_quality` rates the noise channel Good
(0.64 to 0.78): band-passed noise has peaks almost as regular as a pulse's. Its
periodicity is 0.10 to 0.19, against 0.95 to 0.97 on `LI_NIR`, and the series never
reads it.

A 15 bpm step held 60 s, 15 s into epoch 4 and back 15 s into epoch 6 of eight minutes:

| Gate | Threshold | Gated | Fired | False jumps | ECG jumps caught |
|---|---:|---:|---:|---:|---:|
| none |  | 0 | 3 | 0 | 2 of 2 |
| ppg_confidence | 0.9 | 2 | 0 | 0 | 0 of 2 |
| hr_series_median_quality | 0.9 | 0 | 3 | 0 | 2 of 2 |

The rule pauses the two step epochs and, from the window over the step into the next
epoch's values, epoch 5. `ppg_confidence` is 1 minus the spread of the beat intervals
over their mean, so a real step lowers it: 0.89 on the step epochs against 0.99 on the
others, while their series quality stays at 0.98 or more. A gate on `ppg_confidence`
holds back the true jumps first.

## What the tables are for

Three guard settings wait for them; all of them default to doing nothing:

- `min_ppg_confidence_for_hr_rule` (`--guard-min-ppg-confidence`) and
  `min_hr_series_quality_for_hr_rule` (`--guard-min-hr-series-quality`) on
  `simulate-replay-cues`, `run-pilot4-cueing` and `run-pilot5-full-night`, and
  `ppg_quality_for_hr_rule` in `ArousalGuardConfig` (the `ppg_signal_quality` levels the
  HR rules run on). The gate table shows what each would keep; set one only from a
  paired-session table.
- `hr_rule_blind`: an optics epoch whose series skipped more than half of its values is
  logged in the guard's metadata, with action `allow`. The jump table's `hr_rule_blind`
  rows show how many ECG jumps fell in such epochs before `hr_rule_blind_action` is set
  to `lower_volume` or `pause`.
- The series rows by quality show how far the values the series uses, from quality 0.5,
  are from the ECG at each level, for `PPGFeatureConfig.hr_series_min_quality`. By
  `ppg_confidence` and `ppg_signal_quality` they show the same for the epochs a gate on
  those fields would hold back.

The detector adds the reason code `low_ppg_quality` when its HRV comes from PPG peaks
with `ppg_confidence` below 0.4; its scores and `P_REM` do not change.

## Limitations

- A desk session is awake; agreement awake says little about the optics asleep, with
  other postures, pressure on the band and movement.
- Synthetic agreement checks the code: the optics and the ECG come from the same
  scripted beats.
- The optics channel names and their side, depth and wavelength come from the comments
  in `muse_athena_protocol.py` and no vendor document in the repository confirms them
  (`docs/data_model.md`).
- The jump scoring credits a fire to an ECG change in its own or the previous epoch, and
  an ECG jump to a fire in its epochs or the next one. On a recording where the heart
  rate changes by 10 bpm or more in most epochs, that credit makes the rule look better
  than it is. Each epoch scores only its largest ECG change, so a surge up and back down
  inside one epoch is one jump.
- Below 60 compared epochs the report carries `n_epochs_small`, below 10 fired epochs and
  ECG jumps together `few_jump_events`.
