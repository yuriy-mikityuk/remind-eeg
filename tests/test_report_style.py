import contextlib
import datetime as dt
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from muse_tmr.reports import nightly_report
from muse_tmr.reports.report_style import TOKENS_CSS, brand_line, local_minute_of_day, readable_time
from muse_tmr.testing import SYNTHETIC_NOTE, generate_night

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_nightly_report.py"


def _load_nightly_report_script():
    spec = importlib.util.spec_from_file_location("generate_nightly_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReportStyle(unittest.TestCase):
    def test_readable_time_is_local_and_drops_the_microseconds(self):
        moment = dt.datetime(2026, 10, 17, 11, 22, 28, 281028, tzinfo=dt.timezone.utc)
        local = moment.astimezone()
        self.assertEqual(
            readable_time(moment.isoformat()),
            f"{local:%a} {local.day} {local:%b %Y, %H:%M}",
        )
        self.assertEqual(readable_time("2026-10-17T16:22:00"), "Sat 17 Oct 2026, 16:22")
        self.assertEqual(readable_time("not a time"), "not a time")
        self.assertEqual(readable_time(None), "")

    def test_the_chart_clock_is_the_same_local_time_as_the_heading(self):
        moment = dt.datetime(2026, 10, 17, 11, 22, 28, tzinfo=dt.timezone.utc)
        local = moment.astimezone()
        self.assertEqual(local_minute_of_day(moment.isoformat()), local.hour * 60 + local.minute)
        self.assertTrue(readable_time(moment.isoformat()).endswith(f"{local:%H:%M}"))
        self.assertEqual(local_minute_of_day("2026-10-17T16:22:00"), 16 * 60 + 22)
        self.assertEqual(local_minute_of_day(""), 0)
        self.assertEqual(local_minute_of_day("not a time"), 0)

    def test_meditation_and_calibration_reports_give_a_local_time(self):
        from muse_tmr.reports import meditation_report
        from muse_tmr.reports.calibration_report import render_calibration_report

        generated = "2026-10-17T11:22:28.281028+00:00"
        expected = f"generated {readable_time(generated)}"
        self.assertIn(expected, meditation_report._header("s1", {"generated_at": generated}, "focus", "open"))
        page = render_calibration_report({"generated_at": generated})
        self.assertIn(expected, page)
        self.assertNotIn(" UTC", page)

    def test_brand_line_names_the_app_and_escapes_the_kind(self):
        line = brand_line("<b>REM</b> report")
        self.assertIn("REMind · &lt;b&gt;REM&lt;/b&gt; report", line)
        self.assertIn("<svg", line)
        self.assertNotIn("http", TOKENS_CSS + line.replace("http://www.w3.org/2000/svg", ""))

    def test_the_rem_report_is_a_whole_document(self):
        values = dict(
            page_title="REM report · 20261017_162227",
            brand=brand_line("REM report"),
            title="REM report · Sat 17 Oct 2026, 16:22",
            subtitle="20261017_162227 · amused",
            synthetic_callout="",
            mean_p=0.41,
            duration_h=0.1,
            n_epochs=12,
            reconnects=0,
            callout="",
            data_json="[]",
            hourly_json="[]",
            t_max=1,
            start_min=0,
            cardio_section="",
        )
        page = nightly_report.TEMPLATE.format(**values)
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn('<meta charset="utf-8">', page)
        self.assertIn("<title>REM report · 20261017_162227</title>", page)
        self.assertIn("REMind · REM report", page)
        self.assertTrue(page.rstrip().endswith("</html>"))


class TestNightlyReport(unittest.TestCase):
    def test_the_script_runs_the_report_of_the_package(self):
        script = _load_nightly_report_script()
        self.assertIs(script.build_report, nightly_report.build_report)
        self.assertIs(script.main, nightly_report.main)

    def test_the_script_writes_the_report_when_run(self):
        # The app's Build report runs the file as a script, from the checkout.
        with tempfile.TemporaryDirectory() as tmp:
            night = generate_night(output_dir=Path(tmp) / "night", scenario="short_allow_short", hours=0.05)
            output = Path(tmp) / "report.html"
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), str(night.recording_dir), "--output", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"report written: {output}", result.stdout)
            self.assertIn("<strong>Synthetic input, not sleep.</strong>", output.read_text(encoding="utf-8"))

    def test_a_synthetic_night_says_so_and_a_recorded_one_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = generate_night(output_dir=Path(tmp) / "night", scenario="short_allow_short", hours=0.05)
            nightly_report.build_report(night.recording_dir, Path(tmp) / "synthetic.html")
            synthetic = (Path(tmp) / "synthetic.html").read_text(encoding="utf-8")

            metadata_path = night.recording_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("synthetic", "note", "scenario", "seed"):
                metadata.pop(key)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                code = nightly_report.main([str(night.recording_dir), "--output", str(Path(tmp) / "recorded.html")])
            self.assertEqual(code, 0)
            recorded = (Path(tmp) / "recorded.html").read_text(encoding="utf-8")

        self.assertIn("<strong>Synthetic input, not sleep.</strong>", synthetic)
        self.assertIn("scenario short_allow_short, seed 0", synthetic)
        self.assertIn(SYNTHETIC_NOTE, synthetic)
        self.assertIn("<title>REM report · night (synthetic)</title>", synthetic)
        self.assertNotIn("Synthetic input", recorded)
        self.assertNotIn(SYNTHETIC_NOTE, recorded)
        self.assertIn("<title>REM report · night</title>", recorded)


if __name__ == "__main__":
    unittest.main()
