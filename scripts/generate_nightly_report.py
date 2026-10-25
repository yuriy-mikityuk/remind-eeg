#!/usr/bin/env python3
"""Generate a self-contained HTML REM-probability report for one recording.

Runs muse_tmr.reports.nightly_report from a checkout, as the app's "Build report"
does. The report lives in the package so `muse-tmr demo` can build it from an
installed wheel too.

Usage:
    python scripts/generate_nightly_report.py data/recordings/night/20260707_010000
    python scripts/generate_nightly_report.py data/recordings/session/20260707_140000 --output /tmp/report.html
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from muse_tmr.reports.nightly_report import TEMPLATE, build_report, main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
