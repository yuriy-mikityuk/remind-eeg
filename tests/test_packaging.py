"""Packaging metadata against the CI matrix (Initiative 8, #232).

pyproject.toml is read with tomllib; guardrails.yml and CITATION.cff as text, since
PyYAML is not a dependency. Nothing is built or installed.
"""

import re
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "guardrails.yml"
CITATION = REPO_ROOT / "CITATION.cff"
VERSION_FILE = REPO_ROOT / "src" / "muse_tmr" / "__init__.py"

MATRIX_JOBS = ("tests", "package")
MATRIX_OSES = {"ubuntu-latest", "macos-latest", "windows-latest"}
OLD_NAMES = ("athena-tmr-lucid", "athena_tmr_lucid")
OLD_NAME_PATHS = ("pyproject.toml", "setup.py", "src", "scripts", ".github")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _old_name_hits(root: Path) -> list:
    """The old name in the files git tracks under OLD_NAME_PATHS.

    Untracked files are left out: a personal script in a checkout is not part of the
    project, and the checkout's own folder may still carry the old name.
    """
    if shutil.which("git") is None:
        raise unittest.SkipTest("git is not on PATH, and the scan reads the tracked files from git ls-files")
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", *OLD_NAME_PATHS],
        capture_output=True,
    )
    if listed.returncode != 0:
        reason = listed.stderr.decode("utf-8", errors="replace").strip()
        raise unittest.SkipTest(f"git ls-files failed in {root}: {reason}")
    hits = []
    for name in sorted(item.decode("utf-8") for item in listed.stdout.split(b"\0") if item):
        path = root / name
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        hits.extend(f"{name}: {old}" for old in OLD_NAMES if old in text)
    return hits


def _version(text: str) -> tuple:
    return tuple(int(part) for part in text.split("."))


def _job_block(workflow: str, job: str) -> str:
    match = re.search(rf"^  {re.escape(job)}:[ \t]*\n(.*?)(?=^  [A-Za-z_][\w-]*:[ \t]*$|\Z)", workflow, re.M | re.S)
    if match is None:
        raise AssertionError(f"guardrails.yml has no job {job}")
    return match.group(1)


def _matrix_list(block: str, key: str) -> list:
    match = re.search(rf"^\s+{re.escape(key)}:\s*\[([^\]]*)\]", block, re.M)
    if match is None:
        raise AssertionError(f"no {key} list in the job's matrix")
    return [item.strip().strip("\"'") for item in match.group(1).split(",") if item.strip()]


class DistributionTest(unittest.TestCase):
    def test_name_and_command(self):
        pyproject = _pyproject()
        self.assertEqual(pyproject["project"]["name"], "remind-eeg")
        self.assertEqual(pyproject["project"]["scripts"]["muse-tmr"], "muse_tmr.cli.main:main")
        self.assertEqual(pyproject["tool"]["setuptools"]["packages"]["find"]["include"], ["muse_tmr*"])

    def test_old_name_is_not_looked_up(self):
        self.assertEqual(_old_name_hits(REPO_ROOT), [])

    def test_old_name_scan_reads_tracked_files_only(self):
        if shutil.which("git") is None:
            self.skipTest("git is not on PATH")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "scripts").mkdir()
            (root / "scripts" / "build.py").write_text("print('athena-tmr-lucid')\n", encoding="utf-8")
            (root / "scripts" / "personal.sh").write_text("cd ~/athena-tmr-lucid\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "scripts/build.py"], check=True)

            self.assertEqual(_old_name_hits(root), ["scripts/build.py: athena-tmr-lucid"])

    def test_no_amused_in_py_modules(self):
        modules = _pyproject()["tool"]["setuptools"]["py-modules"]
        self.assertTrue(modules)
        self.assertNotIn("amused", modules)


class PythonVersionsTest(unittest.TestCase):
    def setUp(self):
        project = _pyproject()["project"]
        match = re.fullmatch(r">=\s*(3\.\d+)", project["requires-python"])
        self.assertIsNotNone(match, f"requires-python {project['requires-python']!r} is not a plain floor")
        self.floor = _version(match.group(1))
        self.classifiers = sorted(
            _version(found.rsplit(" :: ", 1)[1])
            for found in project["classifiers"]
            if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", found)
        )
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.matrices = {job: _job_block(workflow, job) for job in MATRIX_JOBS}

    def test_floor_is_the_lowest_classifier(self):
        self.assertTrue(self.classifiers)
        self.assertEqual(self.floor, self.classifiers[0])

    def test_matrix_runs_exactly_the_classified_pythons(self):
        for job, block in self.matrices.items():
            with self.subTest(job=job):
                pythons = sorted(_version(found) for found in _matrix_list(block, "python-version"))
                self.assertEqual(pythons, self.classifiers)
                self.assertEqual(pythons[0], self.floor)

    def test_matrix_covers_three_operating_systems(self):
        for job, block in self.matrices.items():
            with self.subTest(job=job):
                self.assertEqual(set(_matrix_list(block, "os")), MATRIX_OSES)
                self.assertRegex(block, r"(?m)^\s+fail-fast:\s*false\s*$")


class CitationTest(unittest.TestCase):
    @unittest.skipUnless(CITATION.is_file(), "CITATION.cff does not exist yet; 8c of #232 adds it")
    def test_citation_version_is_the_package_version(self):
        version = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", VERSION_FILE.read_text(encoding="utf-8"), re.M)
        cited = re.search(r"""^version:\s*["']?([^"'\s#]+)""", CITATION.read_text(encoding="utf-8"), re.M)
        self.assertIsNotNone(version)
        self.assertIsNotNone(cited, "CITATION.cff has no version")
        self.assertEqual(cited.group(1), version.group(1))


if __name__ == "__main__":
    unittest.main()
