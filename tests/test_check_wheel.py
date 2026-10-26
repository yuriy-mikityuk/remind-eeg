import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


def _load_check_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_wheel.py"
    spec = importlib.util.spec_from_file_location("check_wheel", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCheckWheel(unittest.TestCase):
    def test_reports_the_files_a_wheel_left_out(self):
        check = _load_check_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            for name in (
                "muse_tmr/__init__.py",
                "muse_tmr/validation/__init__.py",
                "muse_tmr/app/static/index.html",
                "muse_tmr/__pycache__/__init__.cpython-312.pyc",
                "muse_tmr/app/.DS_Store",
            ):
                (src / name).parent.mkdir(parents=True, exist_ok=True)
                (src / name).write_text("")
            wheel = Path(tmp) / "pkg.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("muse_tmr/__init__.py", "")
                archive.writestr("muse_tmr/app/static/index.html", "")

            self.assertEqual(check.missing_files(wheel, src), ["muse_tmr/validation/__init__.py"])

            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr("muse_tmr/validation/__init__.py", "")
            self.assertEqual(check.missing_files(wheel, src), [])

    def test_reports_a_top_level_amused(self):
        check = _load_check_script()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "pkg.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("muse_tmr/sources/amused_source.py", "")
                archive.writestr("muse_raw_stream.py", "")
                archive.writestr("remind_eeg-0.3.0.dist-info/RECORD", "")
            self.assertEqual(check.top_level_amused(wheel), [])

            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr("amused.py", "")
                archive.writestr("amused/__init__.py", "")
            self.assertEqual(check.top_level_amused(wheel), ["amused", "amused.py"])


if __name__ == "__main__":
    unittest.main()
