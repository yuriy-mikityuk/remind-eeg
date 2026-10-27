"""Wording checks from Initiative 5 (#229).

The first test is the issue's acceptance grep: the retired phrases may only appear in
the history rows of docs/numbers.md and docs/faq.md, and there only inside backticks.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RETIRED_PHRASES = (
    "VALIDATED",
    "frontal-trained",
    "far better fit",
    "Konkoly-like",
    "Konkoly-style",
    # Presets say "parameters from <paper, arm>" (#231).
    "Konkoly protocol",
    "configs/",
)
SEARCH_ROOTS = ("README.md", "docs", "src", "scripts")
EXCLUDED_DIRS = {"research", "__pycache__"}
HISTORY_FILES = {Path("docs/numbers.md"), Path("docs/faq.md")}

UPSTREAM_URL = "https://github.com/Amused-EEG/amused-py"
OLD_UPSTREAM = "github.com/nexon33/amused"

_CODE_SPAN = re.compile(r"`[^`\n]*`")


def _searched_files():
    for name in SEARCH_ROOTS:
        root = REPO_ROOT / name
        if root.is_file():
            yield root
            continue
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or EXCLUDED_DIRS.intersection(relative.parts[:-1]):
                continue
            yield path


def _read_text(path: Path) -> str:
    # A Windows checkout can have CRLF line ends, which the ^...$ patterns below would miss.
    return path.read_bytes().decode("utf-8", errors="ignore").replace("\r\n", "\n")


def _outside_code_spans(line: str, phrase: str):
    spans = [match.span() for match in _CODE_SPAN.finditer(line)]
    start = line.find(phrase)
    while start != -1:
        if not any(left < start and start + len(phrase) < right for left, right in spans):
            yield start
        start = line.find(phrase, start + 1)


class RetiredWordingTest(unittest.TestCase):
    def test_retired_phrases_only_in_history_rows(self):
        hits = []
        for path in _searched_files():
            relative = path.relative_to(REPO_ROOT)
            for number, line in enumerate(_read_text(path).splitlines(), start=1):
                for phrase in RETIRED_PHRASES:
                    if phrase not in line:
                        continue
                    if relative in HISTORY_FILES and not list(_outside_code_spans(line, phrase)):
                        continue
                    hits.append(f"{relative}:{number}: {phrase}")
        self.assertEqual(hits, [], "retired wording found:\n" + "\n".join(hits))

    def test_history_files_exist(self):
        for relative in HISTORY_FILES:
            self.assertTrue((REPO_ROOT / relative).is_file(), relative)

    def test_configs_directory_is_gone(self):
        self.assertFalse((REPO_ROOT / "configs").exists())


class UpstreamUrlTest(unittest.TestCase):
    def test_pyproject_pins_the_one_upstream_url(self):
        pyproject = _read_text(REPO_ROOT / "pyproject.toml")
        pinned = re.findall(r'^pinned_upstream_remote = "([^"]+)"$', pyproject, re.MULTILINE)
        self.assertEqual(pinned, [UPSTREAM_URL])

    def test_docs_link_the_same_upstream(self):
        for relative in ("README.md", "docs/amused_upstream.md", "docs/dependency_strategy.md"):
            self.assertIn(UPSTREAM_URL, _read_text(REPO_ROOT / relative), relative)

    def test_old_upstream_url_is_gone(self):
        hits = [
            str(path.relative_to(REPO_ROOT))
            for path in _searched_files()
            if OLD_UPSTREAM in _read_text(path)
        ]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
