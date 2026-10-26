# Test Status

## Current Test Results
- **1003 tests passing**, 6 skipped, 0 failing on Ubuntu and macOS with Python 3.11, 3.12 and 3.13
- On Windows **989 passing**, 20 skipped with Python 3.11 and 3.12, **992 passing**, 17 skipped with 3.13, 0 failing
- **82 % line coverage** on the `muse_tmr` package (15,481 of 18,964 lines) on Ubuntu with Python 3.11

Measured in CI run [40352898510](https://github.com/yuriy-mikityuk/remind-eeg/actions/runs/40352898510)
on 2026-10-26 (commit `1ab3824`). The line counts come from the `coverage.xml` artifact of
that cell, `coverage-ubuntu-latest-py3.11`; the nine cells range from 81.5 to 81.7 %.
The package includes the BLE layer since #232 moved it into `muse_tmr.ble`, and that
layer has 1,138 of its 3,029 lines covered (37.6 %); the rest of the package is at 90.0 %
(14,343 of 15,935 lines), as before the move.
Six tests are skipped on every cell: three require an optional pandas Parquet engine
(`pyarrow` or `fastparquet`) for the Parquet export paths, two cross-check the complexity
metrics against `antropy` and `neurokit2` and run only when those are installed, and one
checks the version in `CITATION.cff`, which 8c of #232 adds. Windows skips 11 more, the
tests of POSIX signals and process groups, the macOS launcher, `afplay` and the app's
shell report command, and before Python 3.13 another 3 that time waits shorter than its
15.6 ms clock tick. Each skip prints its reason.

## Coverage Overview
Written by hand at release 0.3.0 and not re-measured since; the per-file figures of a
run are in its `coverage.xml` artifacts.

Well covered: REM detection (heuristic + personal classifier), the
`StableRemGate` and `ArousalGuard` safety gates, the TMR cue scheduler,
randomization, the staged pilots, audio playback safety branches
(volume caps, fades, backend fallbacks), the ring buffer, the
acquisition sources' error/timeout paths, the recorder, contact quality,
the meditation analysis and report, the calibration protocol, and the
Polar H10 session and cardio-respiratory features (92-98% each). The local
app server is at 87%.

Lower coverage worth improving:
- `cli/main.py` (66%), the largest module
- `data/watchdog.py` no-data timeout / reconnect logic (78%)
- `app/auto_update.py` (79%), `reports/source_diagnostics.py` (80%) and
  `sources/polar_h10.py` (81%)
- The BLE layer in `muse_tmr.ble` (`muse_sleep_parser`, `muse_integrated_parser`,
  `muse_stream_client`, `muse_discovery`) if those modules remain on the live path

## Running Tests

### Full suite under pytest (matches CI):
```bash
pip install -e ".[dev]"
python -m pytest tests/
```

### With coverage:
```bash
python -m pytest tests/ --cov=muse_tmr --cov-report=term-missing
```

### Legacy unittest runner (no install required):
```bash
PYTHONPATH=src python run_tests.py --all
```

### Specific test file:
```bash
python -m pytest tests/test_rem_gate.py -v
```

## Notes
- A repository-root `conftest.py` puts `src/` on `sys.path` so `pytest` works
  from a checkout without an editable install.
- CI (`.github/workflows/guardrails.yml`) runs the full suite with coverage on
  every pull request and every push to `main`, on Python 3.11 to 3.13 on Ubuntu, macOS
  and Windows, prints each cell's total and uploads its `coverage.xml`.
