# Local App Runbook

REMind, the local web app, connects to the Muse, shows the contact of each sensor,
starts and stops recordings (a session, a night, a guided meditation, a meditation
series, the calibration run) and builds their reports. Its Record tab has the headband
on the left and the recording modes on the right; the Recordings tab lists what has
been recorded. It is a local-only tool served from your
machine. It is not a cloud service and does not upload recordings, contact data,
device identifiers, or reports.

## Required Channels

The app tracks these Muse EEG contact channels:

- `TP9`: left rear/temporal contact.
- `AF7`: left front contact.
- `AF8`: right front contact.
- `TP10`: right rear/temporal contact.

A recording starts only after all four channels have been `good` for 5 s in a row.

## Mock UI Smoke Test

This command requires no Muse, BLE access, or macOS Bluetooth permission:

```bash
cd /path/to/remind-eeg
.venv/bin/muse-tmr app --source mock --host 127.0.0.1 --port 8765 \
  --mock-scenario mixed_fair_good
```

Open:

```text
http://127.0.0.1:8765
```

Useful mock scenarios:

- `all_missing`
- `one_channel_poor`
- `mixed_fair_good`
- `all_good`
- `flapping_af7`
- `disconnect_after_good`
- `stale_data`

Nothing can be recorded in mock mode; recording needs the live source.

Stop the app with `Ctrl-C` in the terminal running the command.

## Live Muse UI Smoke Test

Prepare the headset before launching the live app:

- Close the mobile Muse app so it does not hold the Bluetooth connection.
- Keep the Mac near the headset.
- Put the headset on before contact testing.
- Move hair away from the forehead sensors and the temporal/rear contacts.
- If the headband was charging, unplug it and wait briefly before discovery.

Run:

```bash
cd /path/to/remind-eeg
export MUSE_ADDR="<MUSE_BLE_ADDRESS>"
.venv/bin/muse-tmr app --source amused --address "$MUSE_ADDR" \
  --host 127.0.0.1 --port 8765
```

If you do not know the address, discover first:

```bash
.venv/bin/muse-tmr discover --source amused
```

Then open:

```text
http://127.0.0.1:8765
```

The default bind address is `127.0.0.1`. Do not expose the app on the LAN unless
you intentionally pass an explicit host such as `--host 0.0.0.0`.

Stop the app with `Ctrl-C`.

## Morning tab

After a puzzle night (see `docs/pilot5_full_night.md`), the Morning tab lists the night
puzzle sessions in `data/protocol`, newest first. A night's cue assignment is the one
with the same session ID, and its catalog is one that holds all of its puzzles, so the
files can be named either way (`night-001_puzzles.json`, `pilot5_smoke_session.json`).
The dream report records lucidity, whether cues were heard, confidence, the free recall
and how the dream touched any of the puzzles. The blind retest shows each puzzle's
prompt with a timer (one runs at a time), an answer, solved, confidence and the
seconds. Both save through the same code as `record-dream-report` and
`record-puzzle-retest`, to `data/reports/<night>_dream_report.json` and
`<night>_retest.json`; saving again asks first and replaces the file.

Between the two, the sound check asks "Which puzzle does this sound belong to?": one
Play button per sound, numbered 1 to N in an order the session ID alone decides, and a
puzzle to pick for each. It needs a cue library in `data/cues` or `data/protocol` with
every puzzle's tone and a calibration in `data/calibration`. The sounds play through
`afplay`, each at its cue's volume hint under the calibrated comfortable volume and the
0.20 cap. Start the app with `--cue-backend dry-run` (or `mock`) to go through it without
sound. Without `afplay`, the default `--cue-backend system` refuses every play and names
the fix, unless the app runs with `--allow-dry-run`, which plays them silently. It saves
through the same code as `record-cue-recall`, to `<night>_cue_recall.json`, and logs
every play to `<night>_cue_recall_audio.jsonl`.

The page is never told which puzzles were cued; the retest and sound check files keep
`cue_condition` for `analyze-cued-uncued`.

## Recordings tab

The Recordings tab lists the newest folders under `data/recordings/{session,night}/`,
filtered by kind if you like, with their start, length, stop reason, H10 data, the EEG
rate and lost subpackets from the recorder's last heartbeat (lost subpackets only for
recordings made since the decoder counted them), and links each one's REM, meditation or
calibration report, or offers to build it. It is read from disk, so it survives
app restarts (auto-update restarts the app after every merge).

## Guided meditation

"Set up…" on the Meditation card opens a small form: two practice labels, number of blocks,
minutes per block and settle-in time. Starting it builds the same counterbalanced plan as
`muse-tmr meditation-plan` (random seed), starts a session recording long enough for it
(plus a minute of slack) and stores `blocks.json` in the recording folder. While it runs the
panel shows the current block, practice and time left, counted from the first Muse frame
like the analysis. At every block change it plays a soft tone, then a voice names the
practice (focus and open get a short phrase, other labels are read as typed), so the order
need not be kept in mind with eyes closed. It speaks English or Russian, as picked under
"Voice language" in the form (until then the browser's language decides); the same choice
sets the language of the calibration run. The voice has to be installed on the Mac (Samantha
or Milena, else another local voice of that language); a browser's network voice is never
used, since it would send the text out, so without a local one the change is just the tone.
The browser needs one click on the page for sound.
The Sound button next to the timer mutes or unmutes both mid-session, and unmuting plays them
once to check the volume; it is the same setting as the option in the start form. Finished
blocks get Stability, Sensory fading and Drowsiness selectors that save straight into
`blocks.json` (Stability under the old field name `depth`). When the
recording is done, "Analyze meditation" runs `analyze-meditation` in the background and links its
`report.html`. The Polar H10 checkbox applies here too.

The meditation and series forms link a guide for newcomers, `/guide.html`, in English and
Russian (it opens in the voice language). It explains the settle-in, what to do in focus on
the breath and in open awareness, what the tones and the voice mean, why a practice repeats,
and the three ratings. Its quotes of the voice are checked against `app.js` in the tests.

## Meditation series

The Meditation series card is for the sessions that get pooled by `aggregate-meditation`,
which needs the same two practices in every session and reports p-values only from 8
sessions. The first session of a series sets the practices, block count and block
length, and they stay locked after that. The settle-in can still change between
sessions, since it is read off the EMG timeline of the reports. Every series session
records on `p21` with the H10, whatever the checkbox says, and draws its own ABBA or
BAAB order.

The series lives in `data/protocol/meditation/series.json`, and each session's
`blocks.json` carries its id in `series`. A session counts once it has headband data up
to the end of its last block: the recorded time from the first Muse frame, minus the
time the recorder spent reconnecting, give or take one 10 s epoch. The recording runs a
minute past the last block, so a short dropout still counts. The form lists the
sessions, numbers the counted ones and says why the others did not count: stopped early
or lost the headband, no headband data at all, or the recorder did not finish.
`aggregate-meditation` refuses to pool sessions from different series. The button shows the
count, e.g. "Meditation series 3/8…". "New series" renames the file to
`series_<id>.json`; the old sessions stay on disk but stop counting.

## Reports

When a recording ends, the Record tab says "Recording finished" and offers "Build report". It runs
`scripts/generate_nightly_report.py` for that recording in the background with the app's
own Python and writes `data/reports/<kind>/<name>.html`; when it is done the box shows an
"Open report" link served by the app (only `.html` files from `data/reports/`). If the
build fails, the box points to `report.log` next to the recording and shows the command to
run by hand. "Back to menu" in the same box puts the finished recording away, with its
meditation or calibration panel, until the next one (it stays on the Recordings tab). The
app lets go of the headband for every recording, so the next start needs "Connect Muse"
first.

## Recording with a Polar H10

Tick "Also record the Polar H10 chest strap" under the modes to start the recorder with
`--with-polar` (the choice is remembered in this browser). During the recording the panel
shows the strap's state: streaming with the last HR, no skin contact, no data for N s,
reconnecting, or failed (see `polar/record-polar.log`). Stop works as before; with Polar on
the app waits up to 50 s instead of 5 s before force-killing, so the child can stop the
strap's streams and write its summary. See `docs/polar_h10.md`.

## macOS Launcher Icon

To create a clickable launcher on your Desktop:

```bash
cd /path/to/remind-eeg
export MUSE_ADDR="<MUSE_BLE_ADDRESS>"
.venv/bin/python scripts/install_macos_launcher.py --address "$MUSE_ADDR"
```

This installs:

```text
~/Desktop/REMind.app
```

Double-clicking the app starts the local setup server through Terminal and opens
`http://127.0.0.1:8765` when the server is ready. If the server is already running,
the launcher just opens the browser UI. Stop the app with `Ctrl-C` in the Terminal
window.

The launcher runs the app with `--auto-update`. Every two minutes the app fetches
`origin/main`; if the checkout is on `main` with no tracked changes and can
fast-forward, it pulls and restarts itself once it is idle (no recording, not
scanning or connected), and the open page reloads. If the Morning tab holds answers
that are not saved yet, the reload waits until they are, and the bottom of the page
says so. A checkout on another branch or with local edits is never touched. New
dependencies still need a manual `pip install`. Pass `--no-auto-update` to the
installer to turn this off.

When the app stops answering for more than 5 s (quit, crashed, or a restart that
takes long), the page says it is out of date and dims what it last saw. A running
recording keeps going on its own; the page catches up once the app is back.

Use `--force` to replace an existing launcher after changing the Muse address,
repo path, or port:

```bash
.venv/bin/python scripts/install_macos_launcher.py --address "$MUSE_ADDR" --force
```

## macOS Bluetooth Permission Troubleshooting

On macOS, CoreBluetooth can terminate or block BLE access when the launching app
does not have Bluetooth permission metadata. If live discovery or connection fails
before useful Python logs appear:

- Confirm the mobile Muse app is closed.
- Toggle the headset off/on and retry discovery.
- Check System Settings -> Privacy & Security -> Bluetooth for the terminal or
  Python launcher you are using.
- Prefer a Homebrew Python virtualenv launched through `Python.app` when direct
  terminal Python is blocked by TCC.

Example `Python.app` launch pattern:

```bash
cd /path/to/remind-eeg
export MUSE_ADDR="<MUSE_BLE_ADDRESS>"
open -W -n /opt/homebrew/Frameworks/Python.framework/Versions/3.12/Resources/Python.app \
  --args -m muse_tmr.cli.main app --source amused --address "$MUSE_ADDR" \
  --host 127.0.0.1 --port 8765
```

Adjust the Python.app path to your installed Homebrew Python version.

## Per-Channel Contact Troubleshooting

`TP9`:

- Check the left rear/temporal contact point.
- Move hair away from the left rear pad.
- Reseat the band so the rear-left segment rests flat.

`AF7`:

- Check the left forehead contact.
- Move hair and skin oils away from the sensor area.
- Slightly lower or level the front band if the contact remains poor.

`AF8`:

- Check the right forehead contact.
- Move hair away and confirm the right front pad is not lifted.
- Level the front band if only the right front contact is fair or poor.

`TP10`:

- Check the right rear/temporal contact point.
- Move hair away from the right rear pad.
- Reseat the band so the rear-right segment is not floating.

## Starting a Recording

The Record tab shows five modes and what each records: Night (8 h on `p21`), One-hour
session (`p1034`, with the optics), Meditation, Meditation series and Calibration run.
Their buttons work once the Muse is connected. A start does not need good contact yet. It arms the contact gate (`/api/session/record/arm`), and the page
shows what it is waiting for: all four channels `good`, then a countdown over the
stability window (5 s, `--contact-stability-seconds`). Any channel that turns poor,
missing or stale, or a disconnect, starts the count again. When the window is full the
page sends the start; Cancel start disarms the gate (`/api/session/record/disarm`).

The server enforces this, not only the page: `/api/session/record`,
`/api/meditation/start`, `/api/meditation/series/start` and `/api/calibration/start`
answer 409 with the gate state and its reason codes unless the armed gate is ready.
Arming always starts the window from zero, so a start comes at least 5 s after the
press.

Starting a recording releases the app's own connection, since the Muse takes only
one, and launches `muse-tmr record` as a separate process under `caffeinate`, so
closing the app or its terminal does not stop it. A session records on `p1034` for
up to an hour, a night on `p21` for 8 hours, and the meditation, series and
calibration recordings use `p21`. While it runs, the recording panel shows the elapsed
time against the planned length, the battery, the EEG rate, lost subpackets and
reconnects, and the contact view and the diagnostics show what the recorder reports in
its `progress.json` heartbeat, and the contact
turns stale when the heartbeat is more than 10 s old. Contact drops never stop a
recording. Stop recording sends SIGINT, so the recorder still writes `summary.json`.

The older in-app session flow (`/api/muse/start-when-ready`, `/api/session/start`)
is still in the server, but the page no longer uses it.

## UI Smoke Checklist

- "Find headband" finds the Muse, or says what to try.
- Connect succeeds.
- Contact data updates live.
- The source badge clearly shows `MOCK` or `LIVE Muse`.
- The device card shows the headset name, address, source, connected duration,
  last packet age and battery when available.
- Poor, fair, and good visual states are visible.
- Fair/poor channels show their primary reason code in human-readable form.
- Each channel maps to the expected segment: `TP9`, `AF7`, `AF8`, `TP10`.
- Per-channel sparklines update for recent contact fill history.
- All-good contact shows the centered check mark.
- A start with poor contact waits and says so, counts down 5 s once all four
  channels are good, restarts the count when one drops, and Cancel start stops it.
- Once recording, the panel shows the kind and preset, elapsed time and progress,
  headband battery, EEG rate, lost subpackets, reconnects, the H10 state when it is
  on, and the last recorder event.
- Stop recording ends in "Recording finished." with "Build report" and "Back to
  menu".
- The diagnostics panel exposes notification rate, EEG rows/sec, EEG effective
  rate, decode errors, unknown TAG counts, and last packet age.

Run this checklist before Pilot 4 or Pilot 5 overnight sessions.

## Live Diagnostics

The local app exposes live source diagnostics at:

```text
GET /api/muse/diagnostics
```

For the `amused` source, the payload includes packet/frame counts, decoder
TAG counts, unknown TAG counts, decode errors, telemetry/battery summaries,
cumulative sample rates, and rolling sample rates over the decoder window. Use
this when checking whether a short live smoke is seeing startup jitter or
steady-state EEG/PPG rates.
