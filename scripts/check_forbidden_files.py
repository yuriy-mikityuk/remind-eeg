#!/usr/bin/env python3
"""Fail when tracked or publishable files violate SDK/private-data policy."""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


FORBIDDEN_PATTERNS = (
    ".env",
    "*.bdf",
    "*.bin",
    "*.a",
    "*.aar",
    "*.apk",
    "*.dll",
    "*.dylib",
    "*.edf",
    "*.exe",
    "*.framework",
    "*.framework/*",
    "*.ipa",
    "*.jar",
    "*.key",
    "*.lib",
    "*.muse",
    "*.msi",
    "*.pem",
    "*.pkg",
    "*.private.json",
    "*.so",
    "*.xcframework",
    "*.xcframework/*",
    "*Muse-SDK*.tar*",
    "*Muse-SDK*.zip",
    "*MuseSDK*.tar*",
    "*MuseSDK*.zip",
    "*muse-sdk*.tar*",
    "*muse-sdk*.zip",
    "*muse_sdk*.tar*",
    "*muse_sdk*.zip",
    "audio/private/*",
    "cues/private/*",
    "data/bids/*",
    "data/raw/*",
    "data/recordings/*",
    "data/reports/*",
    "data/sleep/*",
    "data/synthetic/*",
    "data/validation_public/*",
    "dream_reports/*",
    "sdk/*",
    "vendor/MuseSDK/*",
    "vendor/muse-sdk/*",
    "vendor/muse_sdk/*",
)

# Existing tiny fixtures are allowed for now. New binary recordings should not
# be committed unless they are explicitly reviewed and added here.
ALLOWLIST = {
    "examples/recorded_sessions/muse_20250824_202226.bin",
    "muse_data/muse_20250824_172756.bin",
    "tests/test_data/muse_20250824_172756.bin",
}

# Of what validate-external writes under data/validation_public/, only the BOAS splits
# and the pooled metrics may be published (#226, the maintainer's decision): the
# per-night epoch files, the fit and the personal model artifact stay local.
PUBLIC_VALIDATION_FILES = {
    "data/validation_public/boas/splits.json",
    "data/validation_public/boas/metrics.json",
}


def git_files() -> List[str]:
    cmd = [
        "git",
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    return [
        Path(item.decode()).as_posix()
        for item in proc.stdout.split(b"\0")
        if item
    ]


def is_forbidden(path: str) -> bool:
    if path in ALLOWLIST or path in PUBLIC_VALIDATION_FILES:
        return False
    return any(fnmatch.fnmatch(path, pattern) for pattern in FORBIDDEN_PATTERNS)


def violations(paths: Iterable[str]) -> List[str]:
    return sorted(path for path in paths if is_forbidden(path))


def main() -> int:
    bad = violations(git_files())
    if not bad:
        print("No forbidden SDK, secret, recording, report, or private audio files found.")
        return 0

    print("Forbidden files detected:")
    for path in bad:
        print(f"- {path}")
    print("\nSee docs/sdk_policy.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
