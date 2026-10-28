"""validate-external and the external labels of the REM check (#226).

Every dataset here is a BOAS-shaped layout written in-test (tests.boas_synthetic), with
synthetic nights or made-up rows: synthetic input, not sleep. Nothing is fitted for
use: the fits only exercise the code. The stagers are stand-ins, so no GSSC or YASA venv
is needed; the runs that replay a headband EDF need the [edf] extra.
"""

import asyncio
import contextlib
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest import mock

from muse_tmr.cli.main import main
from muse_tmr.data.boas import BoasNight, iter_boas_nights
from muse_tmr.data.edf_io import edf_available
from muse_tmr.data.replay import ReplayConfig, ReplaySession
from muse_tmr.models import EEG_PERSONAL_REM_FEATURES, PersonalRemModel, RemGateDecision
from muse_tmr.presets import GateThresholdSet, named_gate_threshold_sets, named_gate_thresholds, preset_gate_thresholds
from muse_tmr.sources.base_source import BaseMuseSource, MuseSourceMetadata
from muse_tmr.testing import SyntheticNightConfig, generate_night
from muse_tmr.validation import external_labels as ext
from muse_tmr.validation.rem_metrics import simulate_gate
from muse_tmr.validation.rem_reference import collect_from_source, file_sha256
from tests.boas_synthetic import TWO_NIGHTS, NightSpec, write_layout

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_REASON = 'needs the [edf] extra: pip install "remind-eeg[edf]"'
SHORT = ("W", "N1", "N2", "N2", "R", "R", "R", "N2")


def _nights(pids):
    return [BoasNight(f"sub-{index}", pid, Path(f"/nowhere/sub-{index}")) for index, pid in enumerate(pids, 1)]


def _cli(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["validate-external", *args])
    return code, out.getvalue(), err.getvalue()


class SplitsTest(unittest.TestCase):
    def test_no_seed_separates_a_participant(self):
        rng = random.Random(1)
        pids = [str(pid) for pid in range(100) for _ in range(rng.choice((1, 1, 1, 2, 3)))]
        nights = _nights(pids)
        for seed in range(25):
            with self.subTest(seed=seed):
                splits = ext.make_splits(nights, seed=seed, fit_fraction=0.5)
                fit = {entry["pid"] for entry in splits["folds"]["fit"]}
                report = {entry["pid"] for entry in splits["folds"]["report"]}
                self.assertFalse(fit & report)
                self.assertEqual(fit | report, set(pids))
                self.assertEqual(len(splits["folds"]["fit"]) + len(splits["folds"]["report"]), len(nights))
                self.assertLessEqual(abs(len(splits["folds"]["fit"]) - len(nights) / 2), 3)
                ext.check_splits(splits, nights, version=None)

    def test_the_seed_decides_and_the_same_seed_repeats(self):
        nights = _nights([str(pid) for pid in range(40)])
        first = ext.make_splits(nights, seed=3)
        self.assertEqual(ext.splits_text(first), ext.splits_text(ext.make_splits(nights, seed=3)))
        self.assertNotEqual(first["folds"], ext.make_splits(nights, seed=4)["folds"])
        self.assertEqual(first["grouping"], "pid")
        self.assertEqual(first["seed"], 3)

    def test_inputs_it_refuses(self):
        with self.assertRaisesRegex(ValueError, "no pid"):
            ext.make_splits(_nights(["1", None]))
        with self.assertRaises(ValueError):
            ext.make_splits(_nights(["1", "2"]), fit_fraction=1.0)

    def test_check_splits_refuses_a_participant_in_both_folds(self):
        nights = _nights(["7", "7", "8"])
        splits = ext.make_splits(nights, seed=0)
        splits["folds"] = {
            "fit": [{"recording_id": "sub-1", "pid": "7"}],
            "report": [{"recording_id": "sub-2", "pid": "7"}, {"recording_id": "sub-3", "pid": "8"}],
        }
        with self.assertRaisesRegex(ext.SplitOverlapError, "pid 7"):
            ext.check_splits(splits, nights, version=None)

    def test_check_splits_refuses_what_does_not_fit_the_dataset(self):
        nights = _nights(["7", "8"])
        good = ext.make_splits(nights, seed=0, dataset_version_text="1.1.3")
        cases = {
            "twice": [{"recording_id": "sub-1", "pid": "7"}, {"recording_id": "sub-1", "pid": "7"}],
            "not in this dataset": [{"recording_id": "sub-9", "pid": "7"}],
            "not 9": [{"recording_id": "sub-1", "pid": "9"}],
        }
        for message, fit in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ext.SplitOverlapError, message):
                ext.check_splits({**good, "folds": {"fit": fit, "report": []}}, nights, version="1.1.3")
        with self.assertRaisesRegex(ext.SplitOverlapError, "version"):
            ext.check_splits(good, nights, version="1.0.0")

    def test_write_and_load(self):
        nights = _nights(["7", "8"])
        splits = ext.make_splits(nights, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            sha = ext.write_splits(splits, path)
            self.assertEqual(sha, file_sha256(path))
            self.assertEqual(ext.write_splits(splits, path), sha)
            with self.assertRaisesRegex(ValueError, "other splits"):
                ext.write_splits(ext.make_splits(nights, seed=1, fit_fraction=0.9), path)
            self.assertEqual(ext.load_splits(path), splits)
            path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ext.load_splits(path)

    def test_a_crlf_checkout_of_the_splits_has_their_sha256(self):
        splits = ext.make_splits(_nights(["7", "8"]), seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            lf = Path(tmp) / "lf" / "splits.json"
            sha = ext.write_splits(splits, lf)
            crlf = Path(tmp) / "crlf" / "splits.json"
            crlf.parent.mkdir()
            crlf.write_bytes(lf.read_bytes().replace(b"\n", b"\r\n"))

            self.assertEqual(sha, file_sha256(lf))
            self.assertNotEqual(file_sha256(crlf), sha)
            self.assertEqual(ext.write_splits(splits, crlf), sha)
            self.assertEqual(ext.splits_sha256(ext.load_splits(crlf)), sha)
            stored = {"boas_v1": GateThresholdSet(0.7, 0.6, sha)}
            with mock.patch.dict("muse_tmr.presets._NAMED_GATE_THRESHOLDS", stored):
                pair = ext.resolve_named_pair("boas_v1", None, ext.splits_sha256(ext.load_splits(crlf)))
            self.assertEqual(pair["source"], "presets.py")


class NamedThresholdsTest(unittest.TestCase):
    def test_boas_v1_is_opt_in_and_has_no_values_yet(self):
        self.assertEqual(named_gate_threshold_sets(), ("boas_v1",))
        self.assertIsNone(named_gate_thresholds("boas_v1"))
        self.assertEqual(preset_gate_thresholds("p21"), (0.90, 0.85))
        self.assertEqual(ext.GATE_THRESHOLD_CHOICES, ("p21", "boas_v1"))
        with self.assertRaises(KeyError):
            named_gate_thresholds("p21")

    def test_where_a_pair_comes_from(self):
        self.assertEqual(ext.resolve_named_pair("p21", None, "abc")["enter"], 0.90)
        with self.assertRaisesRegex(ext.NotFittedError, "boas_v1 has no values yet"):
            ext.resolve_named_pair("boas_v1", None, "abc")
        fit = {"gate_thresholds": {"boas_v1": {"enter": 0.6, "exit": 0.5}}}
        self.assertEqual(ext.resolve_named_pair("boas_v1", fit, "abc"), {"enter": 0.6, "exit": 0.5, "source": "fit.json"})

        stored = {"boas_v1": GateThresholdSet(0.7, 0.6, "abc")}
        with mock.patch.dict("muse_tmr.presets._NAMED_GATE_THRESHOLDS", stored):
            self.assertEqual(ext.resolve_named_pair("boas_v1", fit, "abc")["source"], "presets.py")
            with self.assertRaisesRegex(ext.SplitOverlapError, "fitted on the fit fold of splits"):
                ext.resolve_named_pair("boas_v1", fit, "other")


class RefusalsTest(unittest.TestCase):
    """What validate-external refuses before it replays a night (no EDF needed)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root = write_layout(
            self.tmp / "ds",
            (NightSpec("sub-1", "7", stages=SHORT), NightSpec("sub-2", "7", stages=SHORT), NightSpec("sub-3", "8", stages=SHORT)),
            edf=False,
        )
        self.out = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def _args(self, *extra):
        return ["--root", str(self.root), "--splits", str(self.tmp / "splits.json"), "--output-dir", str(self.out), *extra]

    def test_a_participant_in_both_folds_is_refused(self):
        splits = ext.make_splits(list(iter_boas_nights(self.root)), seed=0, dataset_version_text="1.1.3")
        splits["folds"] = {
            "fit": [{"recording_id": "sub-1", "pid": "7"}],
            "report": [{"recording_id": "sub-2", "pid": "7"}, {"recording_id": "sub-3", "pid": "8"}],
        }
        ext.write_splits(splits, self.tmp / "splits.json")

        code, _, err = _cli(*self._args())

        self.assertEqual(code, 2)
        self.assertIn("pid 7", err)
        self.assertFalse((self.out / "nights").exists())

    def test_nights_not_on_disk_are_named(self):
        code, _, err = _cli(*self._args("--make-splits"))
        self.assertEqual(code, 2)
        self.assertIn("not on disk", err)
        self.assertIn("sub-1: sub-1_task-Sleep_acq-headband_eeg.edf", err)
        self.assertTrue((self.tmp / "splits.json").is_file())

    def test_boas_v1_without_a_fit_is_refused(self):
        _cli(*self._args("--make-splits"))
        folds = ext.load_splits(self.tmp / "splits.json")["folds"]
        report = ",".join(entry["recording_id"] for entry in folds["report"])
        for night in ("sub-1", "sub-2", "sub-3"):
            edf = self.root / night / "eeg" / f"{night}_task-Sleep_acq-headband_eeg.edf"
            edf.write_bytes(b"")

        code, _, err = _cli(*self._args("--nights", report, "--gate-thresholds", "boas_v1"))

        self.assertEqual(code, 2)
        self.assertIn("needs the fit step", err)

    def test_unknown_nights_missing_splits_and_stager_pythons(self):
        code, _, err = _cli(*self._args())
        self.assertEqual(code, 2)
        self.assertIn("--make-splits", err)
        _cli(*self._args("--make-splits"))
        code, _, err = _cli(*self._args("--nights", "sub-9"))
        self.assertEqual(code, 2)
        self.assertIn("sub-9", err)
        code, _, err = _cli(*self._args("--reference", "gssc", "--gssc-python", str(self.tmp / "no-python")))
        self.assertEqual(code, 2)
        self.assertIn("gssc venv python not found", err)


def _fake_stager(stages):
    """A stand-in for the stager subprocess: GSSC calls REM exactly where ``stages``
    has R, YASA never does."""
    names = {"W": "Wake", "N1": "N1", "N2": "N2", "N3": "N3", "R": "REM"}

    def runner(reference, npz_path, eeg, eog, python_path):
        epochs = [
            {"epoch": index, "stage": names[stage] if reference == "gssc" else "N2", "proba_rem": 1.0 if stage == "R" and reference == "gssc" else 0.0}
            for index, stage in enumerate(stages)
        ]
        return {
            "eeg": eeg,
            "eog": None if eog == "none" else eog,
            "epoch_seconds": 30,
            "stager": {"tool": reference, "package_version": "fake", "torch_version": None, "mne_version": None,
                       "returns_probabilities": False},
            "stage_counts": {},
            "epochs": epochs,
        }

    return runner


@unittest.skipUnless(edf_available(), SKIP_REASON)
class HarnessRunTest(unittest.TestCase):
    """The whole run on the two-night layout (synthetic input, not sleep)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.root = write_layout(cls.tmp / "ds", TWO_NIGHTS)
        cls.out = cls.tmp / "out"
        cls.config = ext.ExternalValidationConfig(
            root=cls.root,
            output_dir=cls.out,
            splits_path=cls.out / "splits.json",
            make_splits=True,
            seed=0,
            references=("gssc", "yasa"),
            reference_pythons={"gssc": Path(sys.executable), "yasa": Path(sys.executable)},
            argv=("muse-tmr", "validate-external"),
        )
        cls.splits = ext.make_splits(list(iter_boas_nights(cls.root)), seed=0)
        cls.report_night = cls.splits["folds"]["report"][0]["recording_id"]
        cls.fit_night = cls.splits["folds"]["fit"][0]["recording_id"]
        stages = {spec.recording_id: spec.stages for spec in TWO_NIGHTS}[cls.report_night]
        cls.runner = staticmethod(_fake_stager(stages))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = ext.run_external_validation(cls.config, log=lambda line: None, stager_runner=cls.runner)
        cls.metrics = json.loads((cls.out / "metrics.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_one_night_in_each_fold(self):
        self.assertEqual(len(self.splits["folds"]["fit"]), 1)
        self.assertEqual(len(self.splits["folds"]["report"]), 1)
        self.assertEqual(self.metrics["nights"]["report_recordings"], [self.report_night])
        self.assertEqual(self.metrics["nights"]["fit_recordings"], [self.fit_night])
        self.assertEqual(self.result["splits_sha256"], file_sha256(self.out / "splits.json"))

    def test_splits_read_from_a_crlf_checkout_have_the_same_sha256(self):
        out = self.tmp / "crlf"
        out.mkdir()
        (out / "splits.json").write_bytes((self.out / "splits.json").read_bytes().replace(b"\n", b"\r\n"))
        config = replace(self.config, output_dir=out, splits_path=out / "splits.json", make_splits=False,
                         nights=(self.fit_night,), references=())

        result = ext.run_external_validation(config, log=lambda line: None)

        self.assertEqual(result["splits_sha256"], self.result["splits_sha256"])

    def test_everything_says_synthetic(self):
        report, fit = self.out / "nights" / self.report_night, self.out / "nights" / self.fit_night
        for path in (self.out / "metrics.json", self.out / "fit.json", self.out / "splits.json",
                     self.out / "per_night_metrics.json", report / "report.json", report / "night.json",
                     report / "gssc.json", report / "yasa.json", fit / "night.json"):
            with self.subTest(path=str(path.relative_to(self.out))):
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertIs(payload["synthetic"], True)
                self.assertEqual(payload["note"], "synthetic input, not sleep")
        model = json.loads((self.out / "boas_v1_personal_rem_model.json").read_text(encoding="utf-8"))
        self.assertEqual(model["training_data"]["note"], "synthetic input, not sleep")
        self.assertTrue(self.result["synthetic"])

    def test_metrics_cover_every_system_and_are_strict_json(self):
        systems = self.metrics["systems"]
        self.assertEqual(set(systems), {"heuristic_p21", "heuristic_boas_v1", "personal", "gssc", "yasa", "stage_ai"})
        for name, block in systems.items():
            pooled = block["pooled"]
            with self.subTest(system=name):
                self.assertEqual(pooled["nights"], 1)
                for key in ("kappa", "f1", "precision", "recall", "precision_at_recall_0.80", "rem_onset_latency",
                            "false_open_minutes_per_night", "kappa_per_night"):
                    self.assertIn(key, pooled)
        # The stand-in GSSC is the consensus itself; the first epoch of the report night
        # may be unscored, so compare on what was scored.
        self.assertEqual(systems["gssc"]["pooled"]["kappa"], 1.0)
        self.assertEqual(systems["gssc"]["pooled"]["rem_onset_latency"]["median_s"], 30.0)
        self.assertEqual(systems["yasa"]["pooled"]["open_epochs"], 0)
        self.assertIn("score_low_delta_power", self.metrics["per_feature_auc"])
        self.assertEqual(self.metrics["kappa_definition"], ext.CONSENSUS_KAPPA_DEFINITION)
        self.assertEqual(self.metrics["dataset"]["version"], "1.1.3")
        self.assertEqual(self.metrics["detector"]["modalities"], ["eeg"])
        self.assertEqual(self.metrics["gate_thresholds"]["p21"]["enter"], 0.90)
        self.assertEqual(self.metrics["stagers"]["gssc"]["package_version"], "fake")
        self.assertEqual(self.metrics["provenance"]["labels_source"], "boas:stage_hum")
        self.assertEqual(self.metrics["provenance"]["dataset_version"], "1.1.3")
        self.assertIn("transfer_not_muse", " ".join(self.metrics["limitations"]))
        text = (self.out / "metrics.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        json.loads(text, parse_constant=lambda name: self.fail(f"{name} in metrics.json"))

    def test_the_fit_uses_the_fit_fold_only(self):
        fit = json.loads((self.out / "fit.json").read_text(encoding="utf-8"))
        pid = {spec.recording_id: spec.pid for spec in TWO_NIGHTS}
        self.assertEqual(fit["fit_recordings"], [self.fit_night])
        self.assertEqual(fit["fit_pids"], [pid[self.fit_night]])
        self.assertEqual(len(fit["gate_thresholds"]["boas_v1"]["grid"]), 190)
        self.assertEqual(fit["gate"], {"optics_capable": False, "imu_capable": False})
        model = PersonalRemModel.load(self.out / "boas_v1_personal_rem_model.json")
        self.assertEqual(model.feature_names, EEG_PERSONAL_REM_FEATURES)
        self.assertEqual(model.training_data["recording_ids"], [self.fit_night])
        self.assertEqual(model.training_data["pids"], [pid[self.fit_night]])
        self.assertEqual(model.training_data["splits_sha256"], self.result["splits_sha256"])
        self.assertEqual(fit["personal"]["sha256"], file_sha256(self.out / "boas_v1_personal_rem_model.json"))
        self.assertEqual(self.metrics["gate_thresholds"]["boas_v1"]["source"], "fit.json")
        self.assertEqual(
            (self.metrics["gate_thresholds"]["boas_v1"]["enter"], self.metrics["gate_thresholds"]["boas_v1"]["exit"]),
            (fit["gate_thresholds"]["boas_v1"]["enter"], fit["gate_thresholds"]["boas_v1"]["exit"]),
        )

    def test_the_cached_night_holds_no_eeg_and_no_clock_time(self):
        epochs = (self.out / "nights" / self.report_night / "epochs.csv").read_text(encoding="utf-8")
        header = epochs.splitlines()[0].split(",")
        self.assertEqual(header[:8], ["epoch", "onset_s", "stage_hum", "label", "reference_stage_ai", "psg_stage_ai",
                                      "p_rem", "reason_codes"])
        self.assertFalse([name for name in header if "time" in name and name != "onset_s"])
        manifest = json.loads((self.out / "nights" / self.report_night / "night.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["parts"]), {"heuristic", "gssc", "yasa"})
        fit_manifest = json.loads((self.out / "nights" / self.fit_night / "night.json").read_text(encoding="utf-8"))
        self.assertEqual(set(fit_manifest["parts"]), {"heuristic"})

    def test_a_rerun_skips_every_night_and_gives_the_same_numbers(self):
        out = self.tmp / "rerun"
        config = ext.ExternalValidationConfig(**{**{f.name: getattr(self.config, f.name) for f in fields(self.config)},
                                                 "output_dir": out, "splits_path": out / "splits.json"})
        runner_calls = []

        def counting(*args):
            runner_calls.append(args[0])
            return self.runner(*args)

        first = ext.run_external_validation(config, log=lambda line: None, stager_runner=counting)
        self.assertEqual(sorted(runner_calls), ["gssc", "yasa"])
        second = ext.run_external_validation(config, log=lambda line: None, stager_runner=counting)
        self.assertEqual(sorted(runner_calls), ["gssc", "yasa"])
        self.assertTrue(all(night["cached"] for night in second["nights"].values()))
        self.assertFalse(any(night["cached"] for night in first["nights"].values()))
        self.assertEqual(json.dumps(first["systems"], sort_keys=True), json.dumps(second["systems"], sort_keys=True))

        rescored = ext.run_external_validation(
            ext.ExternalValidationConfig(**{**{f.name: getattr(config, f.name) for f in fields(config)}, "rescore": True}),
            log=lambda line: None, stager_runner=counting,
        )
        self.assertFalse(any(night["cached"] for night in rescored["nights"].values()))
        self.assertEqual(len(runner_calls), 4)

    def test_another_code_version_scores_the_night_again(self):
        out = self.tmp / "version"
        config = ext.ExternalValidationConfig(**{**{f.name: getattr(self.config, f.name) for f in fields(self.config)},
                                                 "output_dir": out, "splits_path": out / "splits.json",
                                                 "references": ()})
        ext.run_external_validation(config, log=lambda line: None)
        with mock.patch.object(ext, "code_version", lambda: {"muse_tmr_version": "x", "git_commit": "y", "harness_schema": 1}):
            again = ext.run_external_validation(config, log=lambda line: None)
        self.assertFalse(any(night["cached"] for night in again["nights"].values()))

    def test_the_report_refuses_participants_the_fit_used(self):
        fit = json.loads((self.out / "fit.json").read_text(encoding="utf-8"))
        pid = {spec.recording_id: spec.pid for spec in TWO_NIGHTS}[self.report_night]
        tainted = {**fit, "fit_pids": fit["fit_pids"] + [pid]}
        with mock.patch.object(ext, "_fit", lambda *args, **kwargs: tainted), self.assertRaisesRegex(
            ext.SplitOverlapError, f"pid {pid}"
        ):
            ext.run_external_validation(self.config, log=lambda line: None, stager_runner=self.runner)

    def test_the_cli_prints_the_table(self):
        out_dir = self.tmp / "cli"
        with mock.patch("muse_tmr.validation.rem_reference.run_reference", self.runner):
            code, out, _ = _cli(
                "--root", str(self.root), "--splits", str(out_dir / "splits.json"), "--make-splits",
                "--output-dir", str(out_dir), "--reference", "gssc", "--gssc-python", sys.executable,
                "--gate-thresholds", "p21",
            )
        self.assertEqual(code, 0)
        self.assertIn("heuristic_p21", out)
        self.assertIn("gssc", out)
        self.assertNotIn("heuristic_boas_v1", out)
        self.assertIn("(synthetic input, not sleep)", out)
        self.assertEqual((out_dir / "splits.json").read_text(encoding="utf-8"),
                         (self.out / "splits.json").read_text(encoding="utf-8"))


@unittest.skipUnless(edf_available(), SKIP_REASON)
class FailedNightTest(unittest.TestCase):
    def test_a_night_that_cannot_be_read_is_left_out_and_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = tuple(NightSpec(f"sub-{index}", str(index), stages=SHORT, seed=index) for index in (1, 2, 3))
            root = write_layout(Path(tmp) / "ds", specs)
            splits = ext.make_splits(list(iter_boas_nights(root)), seed=0)
            fit_nights = [entry["recording_id"] for entry in splits["folds"]["fit"]]
            self.assertEqual(len(fit_nights), 2)
            broken = fit_nights[0]
            events = root / broken / "eeg" / f"{broken}_task-Sleep_acq-headband_events.tsv"
            events.write_text("\n".join(events.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
            out = Path(tmp) / "out"
            config = ext.ExternalValidationConfig(
                root=root, output_dir=out, splits_path=out / "splits.json", make_splits=True, seed=0
            )
            lines = []

            result = ext.run_external_validation(config, log=lines.append)

            self.assertEqual(list(result["failed"]), [broken])
            self.assertIn("ValueError", result["failed"][broken])
            self.assertTrue(any("failed, left out" in line for line in lines))
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(list(metrics["nights"]["left_out"]), [broken])
            fit = json.loads((out / "fit.json").read_text(encoding="utf-8"))
            self.assertEqual(fit["fit_recordings"], [fit_nights[1]])


class _FrameSource(BaseMuseSource):
    """Frames from a list, as a source."""

    def __init__(self, frames):
        self.frames = list(frames)

    async def discover(self):
        return []

    async def connect(self, device=None):
        return MuseSourceMetadata("list", "list", "list", {"eeg": True})

    async def stream(self):
        for frame in self.frames:
            yield frame

    async def stop(self):
        return None


class CausalityTest(unittest.TestCase):
    """What is decided for epoch k does not change when later epochs come or not."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        stages = ("W",) * 2 + ("N2",) * 4 + ("R",) * 6 + ("N2",) * 2 + ("R",) * 4
        config = SyntheticNightConfig(seed=5, hours=len(stages) * 30 / 3600, preset="p21", scenario="causality",
                                      stages=stages)
        night = generate_night(config, Path(cls._tmp.name) / "night")

        async def frames():
            session = ReplaySession(ReplayConfig(input_path=night.recording_dir, speed=0.0))
            return [frame async for frame in session.stream()]

        cls.frames = asyncio.run(frames())
        cls.stages = stages

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _rows(self, frames):
        collected = asyncio.run(collect_from_source(_FrameSource(frames), 30.0, keep_eeg=False))
        labels = [
            ext.BoasEpochLabel(k, 30.0 * k, 30.0, None, "probable_rem" if stage == "R" else "nrem", None, None)
            for k, stage in enumerate(self.stages)
        ]
        return ext.epoch_rows(collected["predictions"], labels, t0=float(collected["t0"]))

    def test_predictions_and_gate_for_epoch_k_ignore_later_epochs(self):
        full = self._rows(self.frames)
        model = ext.train_personal_rem_classifier(
            [{**row, "recording_id": "night"} for row in full],
            ext.PersonalRemClassifierConfig(feature_names=EEG_PERSONAL_REM_FEATURES, compute_group_holdout=False),
        )
        t0 = self.frames[0].timestamp
        for k in (3, 8, 13):
            with self.subTest(k=k):
                cut = [frame for frame in self.frames if frame.timestamp < t0 + 30.0 * (k + 1)]
                partial = self._rows(cut)
                self.assertGreaterEqual(len(partial), k + 1)
                for a, b in zip(full[: k + 1], partial[: k + 1]):
                    self.assertEqual({key: a[key] for key in a if not _is_nan(a[key])},
                                     {key: b[key] for key in b if not _is_nan(b[key])})
                heuristic_full = [ext.row_prediction(row) for row in full]
                heuristic_cut = [ext.row_prediction(row) for row in partial]
                for system in ("heuristic", "personal"):
                    if system == "personal":
                        heuristic_full = ext.personal_predictions(model, full)
                        heuristic_cut = ext.personal_predictions(model, partial)
                        self.assertEqual([p.probability for p in heuristic_full[: k + 1]],
                                         [p.probability for p in heuristic_cut[: k + 1]])
                    gate_full = simulate_gate(heuristic_full, 30.0, enter_threshold=0.5, exit_threshold=0.4,
                                              **ext.gate_kwargs())
                    gate_cut = simulate_gate(heuristic_cut, 30.0, enter_threshold=0.5, exit_threshold=0.4,
                                             **ext.gate_kwargs())
                    self.assertEqual(gate_full[: k + 1], gate_cut[: k + 1], system)


def _is_nan(value):
    return isinstance(value, float) and math.isnan(value)


class LabelsOnASyntheticNightTest(unittest.TestCase):
    """scripts/validate_rem_detection.py --labels on a synthetic recording (synthetic input, not sleep)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("vrd_for_external", REPO_ROOT / "scripts" / "validate_rem_detection.py")
        cls.vrd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.vrd)
        cls._tmp = tempfile.TemporaryDirectory()
        stages = ("W",) * 4 + ("N2",) * 10 + ("R",) * 10 + ("N2",) * 4
        config = SyntheticNightConfig(seed=7, hours=len(stages) * 30 / 3600, preset="p21", scenario="labels", stages=stages)
        cls.night = generate_night(config, Path(cls._tmp.name) / "night")
        cls.stages = stages

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_hypnogram_of_a_synthetic_night_as_labels(self):
        labels = self.night.recording_dir / "hypnogram.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.vrd.run(
                self.night.recording_dir, "AF7", None, Path(sys.executable), 30.0, "p21", None, None,
                argv=["validate_rem_detection.py", "night", "--labels", str(labels)], labels=labels,
            )

        self.assertEqual(result["reference"]["tool"], "external")
        self.assertEqual(result["provenance"]["labels_source"], "labels:hypnogram.csv")
        self.assertEqual(result["provenance"]["labels_file_sha256"], file_sha256(labels))
        self.assertGreaterEqual(result["epochs_matched"], len(self.stages) - 1)
        self.assertEqual(result["ref_rem_epochs"], self.stages.count("R"))
        self.assertIn("rem_onset_latency", result["labels_metrics"])
        self.assertTrue(math.isfinite(result["roc_auc"]))

    def test_boas_labels_read_from_a_night_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_layout(Path(tmp) / "ds", (NightSpec("sub-4", "4", stages=SHORT, unscored=(1,)),), edf=False)
            labels = ext.load_external_labels(root / "sub-4")
            psg = root / "sub-4" / "eeg" / "sub-4_task-Sleep_acq-psg_events.tsv"

            self.assertEqual(labels.source, "boas:stage_hum")
            self.assertEqual(labels.sha256, file_sha256(psg))
            self.assertEqual(labels.dataset_version, "1.1.3")
            self.assertEqual(len(labels.intervals), len(SHORT))
            self.assertEqual(labels.stage_counts(), {"WAKE": 1, "UNKNOWN": 1, "N2": 3, "REM": 3})
            interval = labels.interval_for(1000.0 + 120.0, t0=1000.0, epoch_seconds=30.0)
            self.assertEqual((interval.stage, interval.label, ext.display_stage(interval)), ("R", "probable_rem", "REM"))
            self.assertIsNone(labels.interval_for(1000.0 + 300.0, t0=1000.0, epoch_seconds=30.0))
            with self.assertRaises(ValueError):
                ext.load_external_labels(root)


class LayeringTest(unittest.TestCase):
    def test_the_harness_reaches_neither_audio_nor_protocol(self):
        code = (
            "import importlib, json, sys\n"
            "for name in sys.argv[1:]:\n"
            "    importlib.import_module(name)\n"
            "print(json.dumps(sorted(n for n in sys.modules if n.startswith(('muse_tmr.audio', 'muse_tmr.protocol')))))\n"
        )
        modules = ["muse_tmr.validation.external_labels", "muse_tmr.validation.rem_metrics", "muse_tmr.validation.rem_reference"]
        result = subprocess.run(
            [sys.executable, "-c", code, *modules],
            capture_output=True, text=True, check=True, cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        self.assertEqual(json.loads(result.stdout), [])

    def test_the_gate_decision_says_nothing_about_cues(self):
        names = {field.name for field in fields(RemGateDecision)}
        self.assertEqual(names, {"gate_open", "state", "confidence", "stable_seconds", "cooldown_remaining_seconds",
                                 "reason_codes"})
        self.assertFalse([name for name in names if any(word in name for word in ("cue", "play", "volume", "audio"))])


if __name__ == "__main__":
    unittest.main()
