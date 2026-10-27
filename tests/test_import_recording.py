"""Nights recorded by other apps (#230): muse_tmr.data.import_recording and `muse-tmr import`.

Every input file is written in-test, in each app's layout, from a labelled synthetic
night (synthetic input, not sleep) into a temporary directory; ``*.csv`` is gitignored and
nothing here is committed. The EDF import is skipped without the ``[edf]`` extra.
"""

import contextlib
import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from muse_tmr.ble import muse_athena_protocol as proto
from muse_tmr.cli.main import _write_demo_protocol
from muse_tmr.cli.main import main as cli_main
from muse_tmr.data.edf_io import edf_available, export_edf
from muse_tmr.data.import_recording import (
    IMPORT_FORMATS,
    IMPORT_SCALE_TABLE,
    MIDSCALE_UV,
    default_import_name,
    import_banner,
    import_recording,
    require_declared_scale,
    resolve_scale,
)
from muse_tmr.validation import validate_pilot1_recording
from tests.exchange_synthetic import frames as replay_frames
from tests.exchange_synthetic import labelled_night

CHANNELS = ("TP9", "AF7", "AF8", "TP10")
STEP = proto.EEG_SCALE_UV_PER_COUNT
BRAINFLOW_OLD = 1650.0 / 4095.0


def eeg_rows(frames):
    """Each EEG sample of ``frames`` in order, with the time an app that stamps every
    sample would give it: the first frame's timestamp plus its place at 256 Hz."""
    first, values = None, {name: [] for name in CHANNELS}
    for frame in frames:
        if frame.eeg is None or not frame.eeg.channels_uv.get("AF7"):
            continue
        first = frame.timestamp if first is None else first
        for name in CHANNELS:
            values[name].extend(frame.eeg.channels_uv[name])
    count = len(values["AF7"])
    return first + np.arange(count) / 256.0, {name: np.asarray(series) for name, series in values.items()}


def imported_samples(directory):
    """The EEG samples of an import in order, and its first frame's timestamp."""
    frames = [frame for frame in replay_frames(directory) if frame.eeg is not None and frame.eeg.channels_uv]
    values = {name: np.concatenate([frame.eeg.channels_uv[name] for frame in frames]) for name in CHANNELS}
    return frames[0].timestamp, values


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        yield out, err


class TestImportFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.night = labelled_night(cls.root / "night")
        cls.frames = replay_frames(cls.night.recording_dir)
        cls.times, cls.values = eeg_rows(cls.frames)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _compare(self, directory, tolerance_uv, *, time_tolerance=1e-6):
        first, values = imported_samples(directory)
        for name in CHANNELS:
            self.assertEqual(values[name].size, self.values[name].size, name)
            self.assertLess(float(np.max(np.abs(values[name] - self.values[name]))), tolerance_uv, name)
        self.assertAlmostEqual(first, self.times[0], delta=time_tolerance)

    def _check_directory(self, directory, fmt):
        self.assertEqual(sorted(path.name for path in directory.iterdir()),
                         ["decoded_frames.jsonl", "metadata.json", "summary.json"])
        metadata = json.loads((directory / "metadata.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        self.assertEqual(metadata["source"]["source_name"], f"import:{fmt}")
        self.assertFalse(metadata["source"]["capabilities"]["raw_packets"])
        self.assertEqual(summary["raw_packet_count"], 0)
        self.assertEqual(metadata["import"]["format"], fmt)
        self.assertIn("report and simulation only", metadata["import"]["banner"])
        self.assertTrue(metadata["import"]["scale_provenance"])
        return metadata

    def test_muselsl_csv_on_an_lsl_clock_anchored_at_start_time(self):
        path = self.root / "muselsl.csv"
        table = pd.DataFrame({"timestamps": self.times - self.times[0] + 5000.0,
                              **{name: self.values[name] for name in CHANNELS}, "Right AUX": 0.0})
        table.to_csv(path, float_format="%.3f", index=False)

        result = import_recording(path, "muselsl-csv", self.root / "imp_muselsl", start_time=float(self.times[0]))

        self._compare(self.root / "imp_muselsl", 0.0006, time_tolerance=0.002)
        metadata = self._check_directory(self.root / "imp_muselsl", "muselsl-csv")
        self.assertEqual(metadata["import"]["clock"]["kind"], "relative")
        self.assertEqual(result.banner, "imported: muselsl-csv, scale 1; report and simulation only")

    def test_openmuse_csv_loses_its_pedestal(self):
        path = self.root / "openmuse.csv"
        table = pd.DataFrame({"time": self.times - self.times[0],
                              **{f"EEG_{name}": self.values[name] + MIDSCALE_UV for name in CHANNELS}})
        table.to_csv(path)  # DataFrame.to_csv keeps its index column, as a user's would

        import_recording(path, "openmuse-csv", self.root / "imp_openmuse", start_time=float(self.times[0]))

        self._compare(self.root / "imp_openmuse", 1e-6, time_tolerance=1e-6)
        metadata = self._check_directory(self.root / "imp_openmuse", "openmuse-csv")
        self.assertAlmostEqual(metadata["import"]["eeg_offset_uv"], -725.044, places=3)

    def _brainflow_file(self, path, values):
        rows = np.zeros((self.times.size, 11))
        rows[:, 0] = np.arange(self.times.size) % 256
        for column, name in enumerate(CHANNELS, start=1):
            rows[:, column] = values[name]
        rows[:, 9] = self.times
        np.savetxt(path, rows, fmt="%.6f", delimiter="\t")  # DataFilter.write_file: %lf, tabs

    def test_brainflow_csv_needs_its_version(self):
        new = self.root / "brainflow_new.csv"
        self._brainflow_file(new, self.values)
        old = self.root / "brainflow_old.csv"
        self._brainflow_file(old, {name: (self.values[name] / STEP + 8192) * BRAINFLOW_OLD for name in CHANNELS})

        with self.assertRaises(ValueError) as caught:
            import_recording(new, "brainflow-csv", self.root / "imp_bf_none")
        self.assertIn("--source-version", str(caught.exception))
        import_recording(new, "brainflow-csv", self.root / "imp_bf_new", source_version="5.23.0")
        result = import_recording(old, "brainflow-csv", self.root / "imp_bf_old", source_version="5.22.1")

        self._compare(self.root / "imp_bf_new", 1e-5)
        self._compare(self.root / "imp_bf_old", 1e-5)
        self._check_directory(self.root / "imp_bf_old", "brainflow-csv")
        self.assertAlmostEqual(1 / (STEP / BRAINFLOW_OLD), 4.5526, places=4)
        self.assertEqual(result.banner, f"imported: brainflow-csv, scale {STEP / BRAINFLOW_OLD:g}; report and simulation only")

    def test_mind_monitor_needs_a_declared_scale(self):
        path = self.root / "mindmonitor.csv"
        stamps = [dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] for t in self.times]
        table = pd.DataFrame({"TimeStamp": stamps, "Delta_TP9": np.nan,
                              **{f"RAW_{name}": self.values[name] + 800.0 for name in CHANNELS},
                              "AUX_RIGHT": 0.0, "Elements": ""})
        marker = pd.DataFrame({"TimeStamp": [stamps[100]], "Elements": ["/muse/elements/blink"]})
        pd.concat([table.iloc[:101], marker, table.iloc[101:]]).to_csv(path, index=False)

        with self.assertRaises(ValueError) as caught:
            import_recording(path, "mindmonitor-csv", self.root / "imp_mm_none")
        self.assertIn("--eeg-scale", str(caught.exception))
        self.assertFalse((self.root / "imp_mm_none").exists())
        import_recording(path, "mindmonitor-csv", self.root / "imp_mm", eeg_scale=1.0, eeg_offset=-800.0)

        self._compare(self.root / "imp_mm", 1e-6, time_tolerance=0.001)
        metadata = self._check_directory(self.root / "imp_mm", "mindmonitor-csv")
        self.assertEqual(metadata["import"]["scale_declared_by"], "user")

    def test_decoded_frames_keep_every_modality(self):
        recorded = self.root / "recorded"
        recorded.mkdir()
        (recorded / "decoded_frames.jsonl").write_text(
            "".join(frame.to_json(include_raw=False) + "\n" for frame in self.frames), encoding="utf-8"
        )
        (recorded / "metadata.json").write_text(json.dumps({"synthetic": True, "note": "synthetic input, not sleep"}))

        result = import_recording(recorded / "decoded_frames.jsonl", "decoded-frames", self.root / "imp_decoded")

        self._compare(self.root / "imp_decoded", 1e-9, time_tolerance=0.0)
        self.assertTrue(result.synthetic)
        imported = replay_frames(self.root / "imp_decoded")
        self.assertEqual([frame.modalities() for frame in imported], [frame.modalities() for frame in self.frames])
        self.assertTrue(all(frame.raw_packet is None for frame in imported))
        metadata = self._check_directory(self.root / "imp_decoded", "decoded-frames")
        self.assertTrue(metadata["synthetic"])

    @unittest.skipUnless(edf_available(), 'needs the [edf] extra: pip install "remind-eeg[edf]"')
    def test_an_exported_edf_comes_back(self):
        edf_path = self.root / "night.edf"
        export_edf(self.night.recording_dir, edf_path)

        import_recording(edf_path, "edf", self.root / "imp_edf")

        first, values = imported_samples(self.root / "imp_edf")
        # Lost subpackets were zeros in the EDF, so the import holds them as zeros too.
        self.assertGreaterEqual(values["AF7"].size, self.values["AF7"].size)
        self.assertAlmostEqual(first, self.frames[0].timestamp, places=5)
        metadata = self._check_directory(self.root / "imp_edf", "edf")
        self.assertTrue(metadata["synthetic"])
        self.assertEqual(metadata["import"]["resampled"], {})
        self.assertNotIn("recording_field", metadata["import"]["edf"])

    def test_the_output_must_be_new_or_an_earlier_import(self):
        decoded = self.root / "again.jsonl"
        decoded.write_text("".join(frame.to_json(include_raw=False) + "\n" for frame in self.frames[:200]))
        import_recording(decoded, "decoded-frames", self.root / "imp_again")
        import_recording(decoded, "decoded-frames", self.root / "imp_again")  # over an earlier import

        with self.assertRaises(ValueError):
            import_recording(decoded, "decoded-frames", self.night.recording_dir)
        with self.assertRaises(ValueError):
            import_recording(decoded, "decoded-frames", self.root / "x", channel_map={"HB_1": "eeg:AF7"})
        self.assertTrue((self.night.recording_dir / "raw_amused.bin").exists())

        # Another file never replaces an import.
        other = self.root / "other.jsonl"
        other.write_text("".join(frame.to_json(include_raw=False) + "\n" for frame in self.frames[200:400]))
        with self.assertRaises(ValueError) as caught:
            import_recording(other, "decoded-frames", self.root / "imp_again")
        self.assertIn("again.jsonl", str(caught.exception))
        kept = json.loads((self.root / "imp_again" / "metadata.json").read_text())["import"]
        self.assertEqual(kept["input_name"], "again.jsonl")

    def test_each_recordings_decoded_frames_import_under_its_name(self):
        self.assertEqual(default_import_name(Path("data/recordings/night_a/decoded_frames.jsonl")), "night_a")
        self.assertEqual(default_import_name(Path("exports/muselsl_0114.csv")), "muselsl_0114")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name, part in (("night_a", self.frames[:150]), ("night_b", self.frames[150:300])):
                (base / name).mkdir()
                (base / name / "decoded_frames.jsonl").write_text(
                    "".join(frame.to_json(include_raw=False) + "\n" for frame in part)
                )
            with patch("muse_tmr.cli.main._default_path_base", return_value=base), quiet():
                for name in ("night_a", "night_b"):
                    code = cli_main(["import", str(base / name / "decoded_frames.jsonl"), "--format", "decoded-frames"])
                    self.assertEqual(code, 0)

            imported = base / "data" / "recordings" / "imported"
            self.assertEqual(sorted(path.name for path in imported.iterdir()), ["night_a", "night_b"])
            digests = {json.loads((imported / name / "metadata.json").read_text())["import"]["input_sha256"]
                       for name in ("night_a", "night_b")}
            self.assertEqual(len(digests), 2)


class TestScaleTable(unittest.TestCase):
    def test_every_row_names_its_evidence(self):
        self.assertEqual({row.fmt for row in IMPORT_SCALE_TABLE}, set(IMPORT_FORMATS))
        for row in IMPORT_SCALE_TABLE:
            with self.subTest(fmt=row.fmt, applies_to=row.applies_to):
                self.assertTrue(row.provenance and row.status)
        (mind_monitor,) = [row for row in IMPORT_SCALE_TABLE if row.fmt == "mindmonitor-csv"]
        self.assertIsNone(mind_monitor.eeg_scale)
        self.assertEqual(mind_monitor.status, "unverified")

    def test_resolving_a_scale(self):
        self.assertEqual(resolve_scale("openmuse-csv")[0].eeg_offset_uv, -MIDSCALE_UV)
        self.assertEqual(resolve_scale("brainflow-csv", source_version="v5.23.1")[0].eeg_scale, 1.0)
        self.assertAlmostEqual(resolve_scale("brainflow-csv", source_version="5.22.2")[0].eeg_scale, STEP / BRAINFLOW_OLD)
        user, declared_by = resolve_scale("mindmonitor-csv", eeg_scale=0.5)
        self.assertEqual((user.eeg_scale, user.eeg_offset_uv, declared_by), (0.5, 0.0, "user"))
        for bad in ({"fmt": "xdf"}, {"fmt": "mindmonitor-csv"}, {"fmt": "muselsl-csv", "eeg_scale": 0.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_scale(bad.pop("fmt"), **bad)


class TestImportedNightInThePipeline(unittest.TestCase):
    """replay, the REM report, annotate-template and simulate-replay-cues take an import as
    it is, every one of them says it is one, and Pilot 1 never passes it."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        night = labelled_night(cls.root / "night")
        decoded = cls.root / "decoded_frames.jsonl"
        decoded.write_text("".join(frame.to_json(include_raw=False) + "\n" for frame in replay_frames(night.recording_dir)))
        cls.imported = cls.root / "imported"
        import_recording(decoded, "decoded-frames", cls.imported)
        cls.banner = "imported: decoded-frames, scale 1; report and simulation only"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_replay_and_the_report_say_it_is_an_import(self):
        with quiet() as (out, _):
            self.assertEqual(cli_main(["replay", str(self.imported)]), 0)
        self.assertIn(self.banner, out.getvalue())
        self.assertEqual(import_banner(self.imported), self.banner)

        from muse_tmr.reports.nightly_report import build_report

        html_path = self.root / "report.html"
        build_report(self.imported, html_path)
        page = html_path.read_text(encoding="utf-8")
        self.assertIn(self.banner, page)
        self.assertIn("Imported night.", page)

    def test_annotate_template_and_simulate_replay_cues_take_it(self):
        with quiet() as (out, _):
            code = cli_main(["annotate-template", str(self.imported), "--output", str(self.root / "rows.csv")])
        self.assertEqual(code, 0)
        self.assertIn(self.banner, out.getvalue())

        protocol = _write_demo_protocol(self.root / "protocol", seed=0)
        output = self.root / "pilot3.json"
        with quiet() as (out, _):
            code = cli_main([
                "simulate-replay-cues", str(self.imported),
                "--catalog", str(protocol["catalog"]), "--session", str(protocol["session"]),
                "--assignment", str(protocol["assignment"]), "--cue-library", str(protocol["cue_library"]),
                "--output", str(output), "--preset", "p1034",
            ])
        self.assertEqual(code, 0)
        self.assertIn(self.banner, out.getvalue())
        self.assertEqual(json.loads(output.read_text())["imported"], self.banner)

    def test_pilot1_fails_it_on_its_raw_capture(self):
        report = validate_pilot1_recording(self.imported, min_duration_seconds=60)

        self.assertFalse(report.passed)
        self.assertIn("raw_capture_present", report.failed_criteria)
        with quiet() as (out, _):
            self.assertEqual(cli_main(["validate-pilot1-recording", str(self.imported)]), 1)
        self.assertEqual(json.loads(out.getvalue())["imported"], self.banner)

    def test_an_import_without_a_declared_scale_is_refused(self):
        bare = self.root / "bare"
        bare.mkdir()
        (bare / "decoded_frames.jsonl").write_text((self.imported / "decoded_frames.jsonl").read_text())
        metadata = json.loads((self.imported / "metadata.json").read_text())
        metadata["import"]["eeg_scale"] = None
        (bare / "metadata.json").write_text(json.dumps(metadata))

        with self.assertRaises(ValueError):
            require_declared_scale(bare, "test")
        with quiet() as (_, err):
            self.assertEqual(cli_main(["annotate-template", str(bare), "--output", str(self.root / "x.csv")]), 2)
        self.assertIn("--eeg-scale", err.getvalue())
        protocol = _write_demo_protocol(self.root / "protocol_bare", seed=0)
        simulated = self.root / "bare_pilot3.json"
        with quiet() as (_, err):
            code = cli_main([
                "simulate-replay-cues", str(bare),
                "--catalog", str(protocol["catalog"]), "--session", str(protocol["session"]),
                "--assignment", str(protocol["assignment"]), "--cue-library", str(protocol["cue_library"]),
                "--output", str(simulated), "--preset", "p1034",
            ])
        self.assertEqual(code, 2)
        self.assertIn("--eeg-scale", err.getvalue())
        self.assertFalse(simulated.exists())
        # replay only lists the frames, and says the scale is missing.
        with quiet() as (out, _):
            self.assertEqual(cli_main(["replay", str(bare)]), 0)
        self.assertIn("imported: decoded-frames, scale undeclared; report and simulation only", out.getvalue())

        from muse_tmr.reports.nightly_report import build_report

        with self.assertRaises(SystemExit):
            build_report(bare, self.root / "x.html")
        self.assertFalse((self.root / "x.html").exists())


if __name__ == "__main__":
    unittest.main()
