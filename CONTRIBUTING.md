# Contributing to REMind

REMind is n = 1 research software, not a medical device. It has one maintainer, has been tried on one person's nights, and makes no clinical, diagnostic, therapeutic or lucid-dreaming claims. Bug reports, fixes, tests and docs are welcome. [`AGENTS.md`](AGENTS.md) holds the full rules, for people and coding agents alike; this page is the part a person needs first, and [`ROADMAP.md`](ROADMAP.md) says what is being worked on.

## Set up

You need Python 3.11 to 3.13. A Muse S Athena is only needed to record; everything else runs on synthetic nights and replays.

```bash
git clone https://github.com/yuriy-mikityuk/remind-eeg.git
cd remind-eeg
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
muse-tmr demo
```

`muse-tmr demo` writes a 2 h synthetic night and runs it through the detector, the gate, the guard and the scheduler with mock audio, so it is a quick check that the setup works. The optional OpenMuse and BrainFlow sources and the macOS Bluetooth workaround for live recording are in [`docs/USAGE.md`](docs/USAGE.md).

## Tests

```bash
python -m pytest tests/
python scripts/check_forbidden_files.py
git diff --check
```

`python -m pytest tests/` is what CI runs, on Linux, macOS and Windows. `python run_tests.py` runs a faster subset; add a new test module to its fast list. Tests use synthetic fixtures or replays and mock or dry-run audio, never a headband or a speaker. A change to the app's page (`src/muse_tmr/app/static/`) also needs the page check: start `muse-tmr app --source mock --port 8799` and run `node scripts/check_page.mjs http://127.0.0.1:8799/` (Node 22, or Node 20 with `--experimental-websocket`).

## Pull requests

One issue per pull request, with a small diff and no unrelated formatting. For anything bigger than a fix, open an issue first. Before you push, rebase onto `origin/main` as the [preflight](AGENTS.md#required-git-preflight) says; after a rebase push with `--force-with-lease`, never plain `--force`. The pull request template has the checklist.

A new command, protocol step, file format or safety behaviour goes into [`docs/USAGE.md`](docs/USAGE.md) or its runbook, and into the Unreleased part of [`CHANGELOG.md`](CHANGELOG.md). A figure newly quoted in the README, `TEST_STATUS.md` or `docs/` needs a row in [`docs/numbers.md`](docs/numbers.md) with its source.

Fixes to how the headband is connected to or decoded go upstream first: the BLE layer in `muse_tmr.ble` to [amused-py](https://github.com/Amused-EEG/amused-py), the BrainFlow and OpenMuse decoders to their projects ([`docs/dependency_strategy.md`](docs/dependency_strategy.md)).

## Claims

Nothing in the code, the docs, an issue or a pull request may say that REMind diagnoses or treats anything, improves sleep or memory, or induces lucid dreams. Report what was measured and its limits: how many nights, against which reference, and whether a threshold was tuned on the data it is reported on ([`docs/validation.md`](docs/validation.md)). Anything computed on a synthetic night says "synthetic input, not sleep".

## Sleep-time audio

Audio during sleep keeps its conservative defaults: a low volume under the calibrated and the hard cap, fades, cooldowns, the arousal guard and the emergency stop. REM probability reaches audio only through the stable REM gate. A change to the gate, the guard, the scheduler or the player comes with replay or synthetic tests ([`docs/safety.md`](docs/safety.md)). If a cue played outside an open gate, above a cap or after an emergency stop, open a bug report with "safety" in its title.

## Data, privacy and the Muse SDK

Your nights stay on your machine. Never commit, attach or paste into an issue or a pull request:

- recordings: `raw_amused.bin`, `decoded_frames.jsonl`, `events.jsonl`, the `polar/` folder, and epoch tables or reports of your own nights
- dream reports or dream text, puzzle catalogs, your answers and retests
- volume calibration files and playback logs
- names, BLE addresses or serials of the headband, the H10 or your headphones
- cue audio, recorded or generated
- keys, tokens and `.env` files
- anything from the proprietary Muse SDK ([`docs/sdk_policy.md`](docs/sdk_policy.md))

REMind has no network upload, telemetry or cloud sync, and a contribution must not add any.

### Sharing numbers from a night

An issue about a night needs aggregate numbers pasted as text, not the files. From `summary.json` keep `duration_seconds`, `frame_count`, `raw_packet_count`, `decoded_frame_count`, `modality_counts`, `reconnect_attempts`, `downtime_seconds` and `stop_reason`; leave out `output_dir`, the `*_path` fields, `started_at` and `ended_at`, which hold your user name, folder names and clock times. From `metadata.json` the source name and the preset are enough, never `device_name` or `device_id`. Counts from a validator or a pilot summary, such as gate openings, plays, skips, pauses, stops and awakenings, are fine. The night report template asks for exactly these.
