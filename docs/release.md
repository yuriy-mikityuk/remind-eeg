# Releasing remind-eeg

The pip distribution is `remind-eeg`; it installs the `muse-tmr` command and the
`muse_tmr` package. Nothing is on PyPI yet. `.github/workflows/release.yml` builds and
publishes a release from a `v*` tag, and only the maintainer pushes one: a version
uploaded to PyPI cannot be uploaded again, even after deleting it, so a bad one gets
yanked and replaced by the next version
([#232](https://github.com/yuriy-mikityuk/remind-eeg/issues/232)).

## What the workflow does

On a tag `vX.Y.Z` it builds the sdist and the wheel with `python -m build`, runs
`scripts/check_wheel.py` on the wheel, installs the wheel into a fresh venv and runs
`muse-tmr --version`, and fails unless the tag is `v` followed by
`muse_tmr.__version__`. Then it attaches `dist/*` to the tag's GitHub release, creating
the release if there is none. Last, and only when the repository variable `PYPI_PUBLISH`
is `true`, it uploads `dist/*` to PyPI with `pypa/gh-action-pypi-publish` from the `pypi`
environment, through trusted publishing, so no PyPI token is stored anywhere. Without
the variable the PyPI job is skipped and the tag only makes a GitHub release.

Run by hand with `dry_run` (the default), and on a pull request that changes
`release.yml`, it builds, checks and uploads the files as the `dist` artifact of the run
and publishes nothing.

The wheel holds `muse_tmr` and the top-level `muse_*` modules of the BLE layer. It does
not hold `amused.py`, and an editable install does not map it either: `amused` is the
name of the upstream package on PyPI, and the shim only serves `import amused` with the
repository root on `sys.path`.

## Once, before the first release

1. On PyPI, add a pending trusted publisher for the project `remind-eeg`: owner
   `yuriy-mikityuk`, repository `remind-eeg`, workflow `release.yml`, environment `pypi`
   (<https://docs.pypi.org/trusted-publishers/>).
2. In the repository's Actions settings, set the variable `PYPI_PUBLISH` to `true`.

## Each release

1. Set `__version__` in `src/muse_tmr/__init__.py`; the wheel takes its version from
   there. Move the Unreleased part of `CHANGELOG.md` under that version.
2. Merge that to `main` and check the build without publishing:
   `gh workflow run release.yml -f dry_run=true`. The run's `dist` artifact holds the
   files that a tag would publish.
3. Tag the merge commit and push the tag:
   `git tag vX.Y.Z origin/main && git push origin vX.Y.Z`.

If the PyPI job was skipped or failed after the GitHub release was made, fix the cause and
run `gh workflow run release.yml --ref vX.Y.Z -f dry_run=false`. That builds the same tag
again, uploads the files to its release again and then publishes. With `dry_run=false`
and a branch as the ref, the workflow stops before building.
