<!-- One issue per pull request. Say what changed, how you tested it, what is left, and end with "Refs #<issue>". AGENTS.md has the full rules: https://github.com/yuriy-mikityuk/remind-eeg/blob/main/AGENTS.md -->

What changed:

How it was tested:

What is left, and follow-up issues:

Refs #

Checklist:

- [ ] Git preflight run; rebased onto `origin/main` if it was behind, and pushed after a rebase with `--force-with-lease`, never plain `--force`
- [ ] One issue, a small diff, no unrelated formatting or refactors
- [ ] `python -m pytest tests/` passes
- [ ] `python scripts/check_forbidden_files.py` and `git diff --check` are clean
- [ ] A change to `src/muse_tmr/app/static/` passed `node scripts/check_page.mjs` against `muse-tmr app --source mock`
- [ ] No recordings, sleep data, dream reports, calibration files, device identifiers, cue audio or Muse SDK files committed or attached
- [ ] No network upload, telemetry, cloud sync or remote reporting added
- [ ] Sleep-time audio keeps its low volume, fades, cooldowns, arousal guard and emergency stop, and REM probability reaches audio only through the stable gate
- [ ] No clinical, medical or lucid-dreaming efficacy claims; results on synthetic nights say "synthetic input, not sleep"
- [ ] Docs and the Unreleased part of `CHANGELOG.md` updated for a new command, protocol step, file format or safety behaviour, and a newly quoted figure has a row in `docs/numbers.md`
- [ ] A BLE or decoder fix also goes upstream (`docs/dependency_strategy.md`)
