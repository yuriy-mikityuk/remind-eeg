"""Protocol presets and their registry (#231)."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

from muse_tmr.audio import AudioPlaybackConfig, VolumeCalibration, audio_config_with_calibration
from muse_tmr.cli.main import build_parser
from muse_tmr.cli import main as cli_main
from muse_tmr.protocol import (
    DEFAULT_PROTOCOL_PRESET,
    SOURCE_CHECKS,
    Citation,
    ProtocolPreset,
    ProtocolPresetNotRunnable,
    RunnerSettings,
    TlrBlockConfig,
    TlrCueConfig,
    TmrSchedulerConfig,
    get_protocol_preset,
    list_protocol_presets,
    protocol_block_of,
    resolve_protocol_preset,
)
from muse_tmr.protocol.protocol_presets import canonical_json

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every preset's version and content hash. A change to a preset changes its hash: raise
# its version and renew the hash here in the same commit.
PINNED = {
    "remind_default_v1": (1, "4c1b2084b275ab1b80405e710f0cae5f4b16edeeb23c1d45446f07269ea40a3a"),
    "konkoly2021_northwestern": (1, "f09fdf79a5bd44e30ee67e6f84be5e31d5a7dab64cef27cc6b8440529522ed82"),
    "konkoly2021_nijmegen": (1, "6e72782374330017b5439fd37b4c9c2121e7e442e705f0e125b070f61c1fb304"),
    "konkoly2021_osnabrueck": (1, "341ea54e01c3980bb2d891a5d0dabc6a9ea7e715019299c0a5a0b60a0b159061"),
    "konkoly2021_paris": (1, "e716c9b9f7b13b8daf71938790ebd851eaf4029d973b0452001a5d1de04985a7"),
    "konkoly2024_exp1": (1, "bc40f60e609d3a58bbb7f1a88fb9e45aacc54ba73d1832252c09670beebaccba"),
    "konkoly2024_exp2": (1, "e5ba322df61a6c2fd933f9ec10ddbd8b15d4df91a65ae1248ec46b77516f14b8"),
    "konkoly2026_puzzle": (1, "e262ddb0cbc8196aeed8341697e0b1ba606371bb991e95b2f8388e215f1269a4"),
    "borghese2022_rem_home": (1, "d8ad891134173d24805b64838022375d6a388b14356ed8d10b99e2d00e3eeb7b"),
    "dresler2024_multisensory": (1, "752ffd89c0415434d47dac20b59163f9187a3a00f79c2b12f8d2006dac99fe3e"),
    "schwartz2022_irt_rem": (1, "ba1372c9d732e4d68a15b5b34f98b9514cfeb4b7215987fd588d89f4425eccf5"),
    "sanders2019_puzzle_sws": (1, "d9a8c6c194b4a5698149cc74a308cbf51b8f92b080007480f6ace44eb493fff2"),
}

NAMES = (
    "remind_default_v1",
    "konkoly2021_northwestern",
    "konkoly2021_nijmegen",
    "konkoly2021_osnabrueck",
    "konkoly2021_paris",
    "konkoly2024_exp1",
    "konkoly2024_exp2",
    "konkoly2026_puzzle",
    "borghese2022_rem_home",
    "dresler2024_multisensory",
    "schwartz2022_irt_rem",
    "sanders2019_puzzle_sws",
)
NOT_RUNNABLE = {"konkoly2021_paris", "sanders2019_puzzle_sws"}
VOLUME_WORDS = ("volume", "gain")


def runnable_presets():
    return [preset for preset in list_protocol_presets() if preset.runnable]


class RegistryTest(unittest.TestCase):
    def test_the_registry_holds_the_presets_default_first(self):
        self.assertEqual(tuple(preset.name for preset in list_protocol_presets()), NAMES)
        self.assertEqual(DEFAULT_PROTOCOL_PRESET, "remind_default_v1")
        self.assertEqual(resolve_protocol_preset().name, "remind_default_v1")
        self.assertEqual({preset.name for preset in list_protocol_presets() if not preset.runnable}, NOT_RUNNABLE)
        with self.assertRaises(KeyError):
            get_protocol_preset("the_konkoly_protocol")

    def test_presets_that_cannot_run_are_refused(self):
        for name in sorted(NOT_RUNNABLE):
            with self.subTest(name=name):
                with self.assertRaises(ProtocolPresetNotRunnable):
                    resolve_protocol_preset(name)
                self.assertFalse(resolve_protocol_preset(name, require_runnable=False).runnable)
                self.assertTrue(get_protocol_preset(name).not_runnable_reason)

    def test_the_cli_refuses_them_before_reading_an_input(self):
        commands = [
            ["simulate-replay-cues", "rec", "--catalog", "c", "--session", "s", "--assignment", "a",
             "--cue-library", "l", "--output", "o.json"],
            ["plan-tlr-block", "cues.json", "--output", "plan.json"],
            ["create-tlr-cue", "--output", "cue.json"],
            ["generate-puzzle-session", "catalog.json", "--output", "s.json", "--session-id", "n"],
            ["assign-puzzle-cues", "s.json", "--output", "a.json", "--seed", "1"],
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli_main, "_default_path_base", return_value=Path(tmp)):
            for argv in commands:
                for name in ("sanders2019_puzzle_sws", "nonexistent_preset"):
                    with self.subTest(command=argv[0], preset=name):
                        stderr = io.StringIO()
                        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(cli_main.main([*argv, "--protocol-preset", name]), 2)
                        self.assertIn(name, stderr.getvalue())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_every_cited_preset_names_its_paper_and_how_it_was_checked(self):
        for preset in list_protocol_presets():
            with self.subTest(preset=preset.name):
                self.assertTrue(preset.title.startswith("Parameters from") or preset.name == "remind_default_v1")
                self.assertNotIn("Konkoly protocol", json.dumps(preset.to_dict()))
                if preset.name == "remind_default_v1":
                    self.assertEqual(preset.citations, ())
                    self.assertIn("no published basis", preset.title)
                    continue
                self.assertTrue(preset.citations)
                for citation in preset.citations:
                    self.assertIn(citation.source_check, SOURCE_CHECKS)
                    self.assertTrue(citation.url.startswith("https://"))
                    self.assertTrue(citation.doi.startswith("10."))
        checks = {preset.name: preset.citations[0].source_check for preset in list_protocol_presets() if preset.citations}
        self.assertEqual(checks["schwartz2022_irt_rem"], "notes_only")
        self.assertEqual(
            {name for name, check in checks.items() if check == "verified_full_text"},
            set(NAMES) - {"remind_default_v1", "schwartz2022_irt_rem"},
        )
        with self.assertRaises(ValueError):
            Citation(authors="A", year=2020, title="T", venue="V", url="https://x", source_check="read_it")


class HashTest(unittest.TestCase):
    def test_hashes_are_pinned(self):
        for name, (version, content_hash) in PINNED.items():
            with self.subTest(name=name):
                preset = get_protocol_preset(name)
                self.assertEqual((preset.version, preset.content_hash()), (version, content_hash))
        self.assertEqual(set(PINNED), set(NAMES))

    def test_the_hash_is_the_same_in_another_process(self):
        script = (
            "import json; from muse_tmr.protocol import list_protocol_presets; "
            "print(json.dumps({p.name: p.content_hash() for p in list_protocol_presets()}))"
        )
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), PYTHONHASHSEED="12345")
        output = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
        ).stdout
        self.assertEqual(json.loads(output), {preset.name: preset.content_hash() for preset in list_protocol_presets()})

    def test_any_change_changes_the_hash(self):
        preset = get_protocol_preset("borghese2022_rem_home")
        changed = (
            replace(preset, version=2),
            replace(preset, runner=replace(preset.runner, puzzle_cue_interval_seconds=11.0)),
            replace(preset, caveats=preset.caveats + ("one more",)),
            replace(preset, citations=(replace(preset.citations[0], source_check="notes_only"),)),
            replace(preset, pauses=replace(preset.pauses, pause_seconds=(("movement", 91.0),))),
        )
        hashes = {preset.content_hash()} | {item.content_hash() for item in changed}
        self.assertEqual(len(hashes), len(changed) + 1)
        self.assertEqual(preset.content_hash(), get_protocol_preset("borghese2022_rem_home").content_hash())

    def test_overrides_change_the_effective_hash_only(self):
        preset = get_protocol_preset("konkoly2021_northwestern")
        plain = preset.manifest()
        overridden = preset.manifest({"tlr_repetitions": 4, "gate_min_stable_seconds": 90})
        same = preset.manifest({"gate_min_stable_seconds": 90.0, "tlr_repetitions": "4"})
        self.assertEqual(plain["content_hash"], overridden["content_hash"])
        self.assertNotEqual(plain["effective_hash"], overridden["effective_hash"])
        self.assertEqual(overridden, same)
        self.assertEqual(overridden["overrides"], {"gate_min_stable_seconds": 90.0, "tlr_repetitions": 4})
        self.assertEqual((overridden["settings"]["tlr_repetitions"], plain["settings"]["tlr_repetitions"]), (4, 10))
        self.assertEqual(plain["overrides"], {})
        with self.assertRaises(ValueError):
            preset.manifest({"max_volume": 0.5})


class RoundTripTest(unittest.TestCase):
    def test_presets_and_manifests_are_plain_json(self):
        for preset in list_protocol_presets():
            with self.subTest(preset=preset.name):
                payload = preset.to_dict()
                self.assertEqual(json.loads(json.dumps(payload)), payload)
                self.assertEqual(canonical_json(payload), canonical_json(json.loads(json.dumps(payload))))
                manifest = preset.manifest()
                self.assertEqual(json.loads(json.dumps(manifest)), manifest)
                self.assertEqual(
                    set(manifest),
                    {"schema_version", "name", "version", "registry_version", "content_hash", "runnable",
                     "overrides", "effective_hash", "settings"},
                )
                self.assertEqual(RunnerSettings(**manifest["settings"]), preset.runner)

    def test_the_block_is_found_where_each_record_keeps_it(self):
        block = get_protocol_preset("konkoly2026_puzzle").manifest()
        self.assertEqual(protocol_block_of({"protocol": block}), block)
        self.assertEqual(protocol_block_of({"metadata": {"protocol": block}}), block)
        self.assertIsNone(protocol_block_of({"protocol": None, "metadata": {}}))
        self.assertIsNone(protocol_block_of({"protocol": {"name": "x"}}))
        self.assertIsNone(protocol_block_of([block]))


class BuilderTest(unittest.TestCase):
    def test_the_default_preset_is_todays_defaults(self):
        preset = get_protocol_preset("remind_default_v1")
        self.assertEqual(preset.runner, RunnerSettings())
        self.assertEqual(preset.scheduler_config(enable_tlr_block=False), TmrSchedulerConfig(enable_tlr_block=False))
        self.assertEqual(preset.tlr_block_config(), TlrBlockConfig())
        self.assertEqual(preset.tlr_cue_config(), TlrCueConfig())
        self.assertEqual(preset.gate_overrides(), {"min_stable_seconds": 60.0})
        self.assertEqual(preset.puzzle_counts(), (4, None))
        parser_defaults = build_parser().parse_args(["create-tlr-cue", "--output", "x.json"])
        self.assertEqual(parser_defaults.volume_hint, TlrCueConfig().volume_hint)

    def test_builders_take_the_preset_and_the_overrides(self):
        preset = get_protocol_preset("konkoly2021_nijmegen")
        self.assertEqual(
            preset.tlr_block_config(),
            TlrBlockConfig(repetitions=10, interval_seconds=15.0, post_block_pause_seconds=10.0),
        )
        cue = preset.tlr_cue_config(cue_id="lucid")
        self.assertEqual((cue.cue_id, cue.frequency_hz, cue.duration_seconds), ("lucid", 600.0, 0.65))
        self.assertIn("konkoly2021_nijmegen", cue.description)
        self.assertEqual(preset.tlr_block_config({"tlr_block_enabled": False}).enabled, False)
        borghese = get_protocol_preset("borghese2022_rem_home")
        self.assertEqual(borghese.gate_overrides(), {"min_stable_seconds": 300.0})
        self.assertEqual(borghese.gate_overrides({"gate_min_stable_seconds": 90}), {"min_stable_seconds": 90.0})
        self.assertEqual(borghese.scheduler_config(enable_tlr_block=True).puzzle_cue_interval_seconds, 10.0)
        self.assertEqual(get_protocol_preset("konkoly2026_puzzle").puzzle_counts(), (4, 2))
        self.assertEqual(get_protocol_preset("konkoly2026_puzzle").puzzle_counts({"cued_puzzle_count": 3}), (4, 3))

    def test_no_preset_lowers_the_stable_window(self):
        for preset in list_protocol_presets():
            self.assertGreaterEqual(preset.runner.gate_min_stable_seconds, 60.0, preset.name)
        default = get_protocol_preset("remind_default_v1")
        with self.assertRaises(ValueError):
            replace(default, runner=replace(default.runner, gate_min_stable_seconds=30.0))
        with self.assertRaises(ValueError):
            replace(default, runner=replace(default.runner, cued_puzzle_count=9))


class VolumeTest(unittest.TestCase):
    """No preset sets a volume (AGENTS.md audio rule)."""

    def test_no_setting_is_a_volume(self):
        for item in fields(RunnerSettings):
            self.assertFalse(any(word in item.name for word in VOLUME_WORDS), item.name)
        for preset in runnable_presets():
            with self.subTest(preset=preset.name):
                self.assertEqual(preset.tlr_cue_config().volume_hint, TlrCueConfig().volume_hint)
                manifest_keys = set(preset.manifest()["settings"])
                self.assertFalse(any(word in key for key in manifest_keys for word in VOLUME_WORDS))

    def test_every_runnable_preset_leaves_the_caps_alone(self):
        calibration = VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04)
        base = [
            "run-pilot5-full-night", "--output-dir", "out", "--catalog", "c.json", "--session", "s.json",
            "--assignment", "a.json", "--cue-library", "l.json", "--tlr-block", "t.json",
            "--calibration", "v.json", "--duration-seconds", "60", "--allow-short",
        ]
        for preset in runnable_presets():
            with self.subTest(preset=preset.name):
                args = build_parser().parse_args([*base, "--protocol-preset", preset.name])
                config = cli_main._live_cueing_config(
                    args,
                    pilot_id="m8_pilot5_full_night_tlr_puzzle_cueing",
                    summary_filename="pilot5_summary.json",
                    enable_tlr_block=True,
                    require_tlr_block=True,
                )
                self.assertEqual((config.hard_max_volume, config.default_volume), (0.20, 0.02))
                self.assertEqual((config.fade_in_seconds, config.fade_out_seconds), (0.25, 0.25))
                audio = audio_config_with_calibration(
                    AudioPlaybackConfig(max_volume=config.hard_max_volume, default_volume=config.default_volume),
                    calibration,
                )
                self.assertEqual(audio.max_volume, 0.04)
                self.assertEqual(config.protocol["name"], preset.name)
                self.assertEqual(config.gate_config.min_stable_seconds, preset.runner.gate_min_stable_seconds)


class CliLoaderTest(unittest.TestCase):
    def _run(self, *argv):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli_main.main(list(argv)), 0)
        return stdout.getvalue()

    def test_tlr_files_take_the_preset_and_record_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            cues = Path(tmp) / "cues.json"
            output = self._run("create-tlr-cue", "--output", str(cues), "--protocol-preset", "konkoly2021_northwestern")
            self.assertIn("protocol=konkoly2021_northwestern", output)
            cue = json.loads(cues.read_text(encoding="utf-8"))["cues"][0]
            self.assertEqual((cue["frequency_hz"], cue["duration_seconds"], cue["volume_hint"]), (600.0, 0.65, 0.05))
            self.assertEqual(cue["metadata"]["protocol"]["name"], "konkoly2021_northwestern")

            plan_path = Path(tmp) / "plan.json"
            self._run("plan-tlr-block", str(cues), "--output", str(plan_path), "--protocol-preset",
                      "konkoly2021_northwestern")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual((plan["config"]["repetitions"], plan["config"]["interval_seconds"]), (10, 30.0))
            block = plan["metadata"]["protocol"]
            self.assertEqual((block["name"], block["overrides"]), ("konkoly2021_northwestern", {}))
            self.assertEqual(block["inputs"]["tlr_cue"]["content_hash"], block["content_hash"])

            self._run("plan-tlr-block", str(cues), "--output", str(plan_path), "--protocol-preset",
                      "konkoly2021_northwestern", "--repetitions", "4", "--disabled")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["config"]["enabled"], False)
            self.assertEqual(
                plan["metadata"]["protocol"]["overrides"], {"tlr_block_enabled": False, "tlr_repetitions": 4}
            )

    def test_explicit_night_flags_reach_the_gate_and_the_scheduler(self):
        argv = [
            "run-pilot4-cueing", "--output-dir", "night", "--catalog", "catalog.json", "--session", "session.json",
            "--assignment", "assignment.json", "--cue-library", "cues.json", "--calibration", "calibration.json",
            "--protocol-preset", "konkoly2024_exp1",
        ]

        def live_config(*flags):
            return cli_main._live_cueing_config(
                build_parser().parse_args(argv + list(flags)),
                pilot_id="m8_pilot4_low_volume_rem_gated_cueing",
                summary_filename="pilot4_summary.json",
                enable_tlr_block=False,
                require_tlr_block=False,
            )

        preset = get_protocol_preset("konkoly2024_exp1")
        plain = live_config()
        self.assertEqual(plain.gate_config.min_stable_seconds, preset.runner.gate_min_stable_seconds)
        self.assertEqual(plain.scheduler_config, preset.scheduler_config(enable_tlr_block=False))
        self.assertEqual(plain.protocol["overrides"], {})
        flagged = live_config(
            "--min-stable-seconds", "90", "--max-puzzle-cues-per-block", "1",
            "--puzzle-cue-interval-seconds", "45", "--scheduler-cooldown-seconds", "60",
        )
        self.assertEqual(flagged.gate_config.min_stable_seconds, 90.0)
        scheduler = flagged.scheduler_config
        self.assertEqual(
            (scheduler.max_puzzle_cues_per_block, scheduler.puzzle_cue_interval_seconds, scheduler.cooldown_seconds),
            (1, 45.0, 60.0),
        )
        self.assertEqual(
            flagged.protocol["overrides"],
            {
                "gate_min_stable_seconds": 90.0,
                "max_puzzle_cues_per_block": 1,
                "puzzle_cue_interval_seconds": 45.0,
                "scheduler_cooldown_seconds": 60.0,
            },
        )

    def test_without_a_preset_the_tlr_files_are_todays(self):
        with tempfile.TemporaryDirectory() as tmp:
            cues = Path(tmp) / "cues.json"
            self._run("create-tlr-cue", "--output", str(cues))
            cue = json.loads(cues.read_text(encoding="utf-8"))["cues"][0]
            self.assertEqual((cue["frequency_hz"], cue["duration_seconds"], cue["description"]),
                             (396.0, 1.0, TlrCueConfig().description))
            plan_path = Path(tmp) / "plan.json"
            self._run("plan-tlr-block", str(cues), "--output", str(plan_path))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["config"], TlrBlockConfig().to_dict())
            self.assertEqual(plan["metadata"]["protocol"]["name"], "remind_default_v1")

    def test_puzzle_counts_come_from_the_preset(self):
        from muse_tmr.protocol import PuzzleCatalog, PuzzleTask

        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog.json"
            PuzzleCatalog(
                puzzles=tuple(PuzzleTask(f"p{index}", f"Prompt {index}", f"Answer {index}") for index in range(8))
            ).save(catalog)
            session = Path(tmp) / "session.json"
            self._run("generate-puzzle-session", str(catalog), "--output", str(session), "--session-id", "night-9",
                      "--seed", "3", "--protocol-preset", "konkoly2026_puzzle")
            payload = json.loads(session.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["puzzle_ids"]), 4)
            self.assertEqual(payload["metadata"]["protocol"]["name"], "konkoly2026_puzzle")
            assignment = Path(tmp) / "assignment.json"
            self._run("assign-puzzle-cues", str(session), "--output", str(assignment), "--seed", "5",
                      "--protocol-preset", "konkoly2026_puzzle")
            payload = json.loads(assignment.read_text(encoding="utf-8"))
            self.assertEqual((len(payload["cued_puzzle_ids"]), len(payload["uncued_puzzle_ids"])), (2, 2))
            self.assertEqual(payload["metadata"]["protocol"]["inputs"]["session"]["name"], "konkoly2026_puzzle")

            self._run("generate-puzzle-session", str(catalog), "--output", str(session), "--session-id", "night-9",
                      "--seed", "3", "--count", "6")
            self._run("assign-puzzle-cues", str(session), "--output", str(assignment), "--seed", "5",
                      "--protocol-preset", "konkoly2026_puzzle", "--cued-count", "3")
            payload = json.loads(assignment.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["cued_puzzle_ids"]), 3)
            self.assertEqual(payload["metadata"]["protocol"]["overrides"], {"cued_puzzle_count": 3})

    def test_protocol_presets_lists_and_prints(self):
        listing = self._run("protocol-presets")
        self.assertEqual(len(listing.strip().splitlines()), len(NAMES))
        self.assertIn("sanders2019_puzzle_sws\tv1\tdocumentation only", listing)
        rows = json.loads(self._run("protocol-presets", "--json"))
        self.assertEqual([row["name"] for row in rows], list(NAMES))
        payload = json.loads(self._run("protocol-presets", "borghese2022_rem_home"))
        self.assertEqual(payload["content_hash"], get_protocol_preset("borghese2022_rem_home").content_hash())
        self.assertEqual(payload["preset"]["pauses"]["pause_seconds"][0], ["movement", 90.0])

    def test_docs_describe_every_preset(self):
        text = (REPO_ROOT / "docs" / "protocol_presets.md").read_text(encoding="utf-8")
        for name in NAMES:
            self.assertIn(f"`{name}`", text)
        self.assertNotIn("Konkoly protocol", text)


if __name__ == "__main__":
    unittest.main()
