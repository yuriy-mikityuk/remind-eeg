# Polar H10 companion recording

The Muse heart signal is forehead PPG at 64 Hz, so `ppg_features` labels its
HRV a proxy. A Polar H10 chest strap recorded in the same session adds
beat-level RR, a 130 Hz single-lead ECG and chest acceleration for breathing.
It does nothing for the EEG side. Nothing here feeds REM detection, the gate,
the arousal guard, the scheduler or audio.

Code: `muse_tmr.sources.polar_h10` (BLE, parsing), `muse_tmr.data.polar_recorder`
(recorder, decoder), `muse_tmr.data.polar_session` (alignment, loader),
`muse_tmr.features.cardio_resp_features` (features).

## Setup

- Wet both electrode areas of the strap and wear it just below the chest
  muscles, the H10 logo upright and centred. A dry strap gives HR dropouts and a
  noisy ECG for the first minutes.
- The H10 accepts only a limited number of connections and phones/watches grab
  it. If `record-polar` says the strap is not found, turn off Bluetooth on the
  phone or close the Polar/Strava app, then retry.
- macOS Bluetooth permission (TCC) works the same way as for the Muse: launch
  through the same Terminal or `Python.app` setup that already records the Muse
  (see the BLE notes in `AGENTS.md`). `record --with-polar` starts the Polar
  recorder as a child of the Muse recorder, so it inherits that permission.
- Polar known issue: an H10 that is never told to stop ECG/ACC streaming stays
  on until its battery is empty, even off the strap. The recorder always sends
  the stop commands, including on Ctrl-C / the app's Stop button. If a run is
  ever killed hard (`kill -9`), reconnect once with `record-polar` for a few
  seconds to stop it.

## Commands

```bash
# Muse + H10 together; the H10 runs in its own process
muse-tmr record --source amused --preset p1034 --duration-seconds 600 --allow-short \
  --output-dir data/recordings/session/smoke_h10 --with-polar

# H10 alone into an existing or new session directory
muse-tmr record-polar --output-dir data/recordings/session/<name> --duration-seconds 600 \
  [--address <ble-address>] [--no-ecg] [--acc-rate 50] [--acc-range 2]

# rebuild decoded files from the raw log
muse-tmr decode-polar data/recordings/session/<name>
```

`--with-polar` is fully isolated from the Muse run: a Polar child that crashes,
never finds the strap or exits early only shows up as `companion_*` events in
the Muse `events.jsonl`. The Muse `stop_reason` and summary never depend on it.
At the end the Muse recorder writes its own summary first, then sends the child
SIGINT and waits up to 30 s.

## Files

Everything lands in `<session_dir>/polar/` (under `data/recordings/`, gitignored):

| File | Content |
| ---- | ------- |
| `raw_notifications.jsonl` | Every BLE payload as received plus the control-point commands we sent. See `docs/data_model.md`. |
| `events.jsonl` | recording_started, connected (with stream settings), connect_failed, disconnected, recording_stopped. |
| `clock_anchors.jsonl` | `time.time()` / `time.monotonic()` pairs at start, every 60 s and at stop. |
| `metadata.json` | Device name, model, firmware, negotiated stream settings. No BLE address or serial. |
| `summary.json` | stop_reason, reconnects, downtime, notification counts, decode counts, gaps, errors. |
| `hr_rr.jsonl` | One row per HR notification: receive times, HR, contact bits, RR list (ms). |
| `ecg.jsonl`, `acc.jsonl` | One row per PMD frame: receive times, sensor timestamp, first-sample time, sample period, samples (µV, mG). |

## Time alignment

Muse replay timestamps are host wall-clock, so Polar data is mapped onto the
host wall-clock too.

1. PMD frames carry a sensor timestamp: ns since 2000-01-01Z for the **last**
   sample of the frame (after a power cycle the H10 restarts its clock at
   2019-01-01, which does not matter here). Per-sample times are spread back
   from it using the gap to the previous frame, or the nominal rate after a gap.
2. Sensor time is mapped to host monotonic time with an offset and a linear
   drift fitted to the **lower envelope** of (receive time - sensor time): a
   linear program that keeps every frame on or above the line. BLE only ever
   adds delay, so a least-squares line would sit late by the mean delay. The
   residuals above the line are the BLE delays and are reported
   (`delay_median_ms`, `delay_p95_ms`). An H10 that powers down during a
   reconnect restarts its clock, so frames are split into clock segments
   wherever receive time minus sensor time jumps by more than 10 s or sensor
   time goes back by more than 1 s, and each segment gets its own fit
   (`clock_segments`). On the second calibration run the battery went out, then
   the H10 restarted again as the cover was closed, 25 s later: the offset moved
   only 25 s, which the old 30 s rule missed, and the merged run fitted a 22 %
   "drift" that stretched every later time. A fit beyond 1000 ppm now falls
   back to the offset alone.
3. Monotonic is converted to wall-clock through the clock anchors; a jump in
   wall - monotonic between anchors (an NTP step) is reported as `ntp_step_ms`.
4. HR-service RR intervals have no sensor timestamp. With ECG on, each
   notification's RR batch is matched to a run of consecutive ECG R-peaks and
   takes their times (`aligned_to = ecg_r_peak`). With ECG off, beats keep a
   receive-time estimate (`aligned_to = host_receive`). On a live H10 a
   notification arrived a median 2 s (up to ~3 s) after its last beat, so
   these estimates are that far off.

With the electrodes off the skin the H10 keeps streaming ECG and even sends RR,
all noise; only the HR notification's contact bit says so. On the calibration run
the ECG swung rail to rail (±19.6 mV) from about 13 s before that bit went false to
about 7 s after it came back, and the RR in between were noise (390-3800 ms). So
spans run from 15 s before the first no-contact notification to 10 s after contact
returns, or to the end of the data if it never does. RR beats inside are dropped,
and R-peaks are searched separately in each stretch of ECG outside them, because
that noise would otherwise pick the detector's polarity for the whole recording.
`quality` reports the spans, their total seconds and the dropped beats. Splining or
differencing across the hole would invent data, so `extract_cardio_resp_features`
takes the longest part of the window with contact: `window_seconds` is that part
and `no_contact_seconds` what was cut. When that part is shorter than half the
window (`min_contact_fraction`) there are no features at all.

The contact bit is not enough on its own. With the battery pulled the H10 lost
power before it reported anything, and while unclipping and walking it said
"contact" over rail-to-rail ECG and RR between 260 and 5900 ms. So any second whose
ECG swings more than 4000 µV around its median also starts a span, from 5 s before
to 15 s after, and spans less than 10 s apart are joined. Normal seconds peaked at
1200-1900 µV on three sessions (p99.9 3800 on a quiet 10 min recording).

Expected error: on synthetic data with 50 ppm drift and random BLE delays the
mapped times are within a few ms of the truth over 8 h. A constant minimum BLE
latency (a few ms, one connection interval) cannot be told apart from a clock
offset and stays in the result as a small fixed lag.

## Features

`extract_cardio_resp_features(session, start, end)` per epoch or block:

- R-peaks on the 130 Hz ECG (5-25 Hz detection band, adaptive threshold),
  refined with a parabola through the apex because raw resolution is ~7.7 ms.
  Device RR vs ECG RR agreement is reported.
- RR correction: a beat is flagged when outside 300-2000 ms or more than 20 %
  away from the median of the surrounding 10 beats; flagged beats are linearly
  interpolated. An ectopic usually flags as a short-long pair. The percent of
  corrected beats is reported.
- Mean HR always; RMSSD, SDNN and pNN50 only for windows of at least 60 s.
- Breathing from chest ACC: 0.05-0.7 Hz band-pass of the first principal
  component, both spectral peak and breath-by-breath median, plus ECG-derived
  respiration from R-amplitude modulation as a cross-check.
- RSA: LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz) RR power, and power in a ±0.03 Hz
  band around the measured breathing frequency. Below about 9 breaths/min the
  RSA sits in LF, so `hf_band_valid` is 0 and HF must not be read as vagal tone.
- Exhale/inhale ratio is not implemented: the sign of the dominant ACC
  component does not say which phase is inhalation without a reference, so the
  ratio could silently come out inverted.

## In the nightly REM report

`scripts/generate_nightly_report.py` (and the app's "Build report") adds a "Heart and
breathing" chart under the REM curve when the recording has `polar/`: HR and breathing per
5-minute window on the same time axis, untrusted breathing windows drawn hollow, plus a
night summary (mean and lowest HR, RMSSD, median trusted breathing, RR-to-ECG match) and a
table. A Polar log that cannot be read is reported in that section instead of failing the
report.

## Against the Muse optics

`muse-tmr analyze-optics-agreement <session>` compares the heart rate the Muse takes
from its optics with the heart rate from these R-peaks, per epoch and per window of the
heart-rate series, and scores the arousal guard's sudden heart-rate change rule against
the ECG. It needs a `p1034` session with ECG on. The H10 is the reference there and feeds
nothing: the report reads the session, and what the guard reads stays the Muse's own.
Runbook and how to read it: `docs/optics_agreement.md`.

## Limitations

- BLE delay and dropouts, especially with two peripherals on one Mac adapter.
  The first live session must compare Muse frame counts and gaps against a
  Muse-only recording of the same length.
- 130 Hz ECG (±2 % rate) limits raw R-peak timing to ~7.7 ms before
  interpolation.
- ACC respiration breaks down with posture changes, movement and talking;
  breath-by-breath and spectral rates disagreeing is the warning sign.
- The HF band is invalid during slow breathing, see above.
- Breathing from ACC is gated by `resp_reliable` (no posture changes, spectral and
  breath-by-breath rates agree); `analyze-meditation` only compares trusted blocks.
