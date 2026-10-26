"""Packaging metadata against the CI matrix (Initiative 8, #232).

pyproject.toml is read with tomllib; guardrails.yml and CITATION.cff as text, since
PyYAML is not a dependency. Nothing is built or installed.
"""

import re
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


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


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
        paths = [PYPROJECT, REPO_ROOT / "setup.py"]
        for root in ("src", "scripts", ".github"):
            paths.extend(path for path in sorted((REPO_ROOT / root).rglob("*")) if path.is_file())
        hits = []
        for path in paths:
            if "__pycache__" in path.parts:
                continue
            text = path.read_bytes().decode("utf-8", errors="ignore")
            hits.extend(f"{path.relative_to(REPO_ROOT)}: {name}" for name in OLD_NAMES if name in text)
        self.assertEqual(hits, [])

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
