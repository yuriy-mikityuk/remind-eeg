"""The protocol fidelity report (#231): synthetic gate sequences and a synthetic night."""

import ast
import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from muse_tmr.audio import CueLibrary, CueMetadata
from muse_tmr.cli.main import main as cli_main
from muse_tmr.models import RemGateConfig, RemPrediction, StableRemGate
from muse_tmr.protocol import (
    ArousalGuardDecision,
    NightPuzzleSession,
    PuzzleCatalog,
    PuzzleCueAssignment,
    PuzzleTask,
    TmrCueScheduler,
    default_tlr_cue,
    get_protocol_preset,
    plan_tlr_block,
)
from muse_tmr.reports.protocol_fidelity import (
    PARAMETER_FIDELITY_IS_NOT_RESULT_FIDELITY,
    ProtocolFidelityReport,
    build_protocol_fidelity,
)
from muse_tmr.testing import SYNTHETIC_NOTE, generate_night
from muse_tmr.validation import (
    AwakeningEvent,
    append_awakening_event,
    gate_decision_record,
    load_gate_open_runs,
    write_rem_gate_decisions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_MODULES = (
    "src/muse_tmr/protocol/protocol_presets.py",
    "src/muse_tmr/protocol/protocol_registry.py",
    "src/muse_tmr/reports/protocol_fidelity.py",
    "src/muse_tmr/reports/cue_recall.py",
)
BACKEND_NAMES = {
    "AudioBackend",
    "MacOSAfplayBackend",
    "DryRunAudioBackend",
    "MockAudioBackend",
    "create_audio_backend",
    "resolve_audio_backend",
}


def protocol_fixture():
    catalog = PuzzleCatalog(
        puzzles=tuple(PuzzleTask(f"p{index}", f"Puzzle {index}", f"Answer {index}", cue_id=f"cue-p{index}") for index in (1, 2, 3, 4))
    )
    session = NightPuzzleSession(session_id="night-001", puzzle_ids=("p1", "p2", "p3", "p4"), puzzle_count=4)
    assignment = PuzzleCueAssignment(
        session_id="night-001", cued_puzzle_ids=("p1", "p3"), uncued_puzzle_ids=("p2", "p4"), seed=17
    )
    cues = tuple(
        CueMetadata(f"cue-p{index}", "generated_tone", 1.0, protocol="puzzle", frequency_hz=400.0 + 50 * index)
        for index in (1, 2, 3, 4)
    )
    return catalog, session, assignment, cues


def write_night(
    directory: Path,
    probabilities,
    *,
    preset="remind_default_v1",
    overrides=None,
    tlr=False,
    guard=None,
):
    """Run the gate and the scheduler over ``probabilities`` (one per 30 s epoch) with the
    preset's settings and write the logs a Pilot 4/5 night writes."""
    directory.mkdir(parents=True, exist_ok=True)
    preset = get_protocol_preset(preset)
    settings = preset.runner_settings(overrides)
    catalog, session, assignment, cues = protocol_fixture()
    plan = None
    if tlr:
        tlr_cue = default_tlr_cue(preset.tlr_cue_config(overrides=overrides))
        cues = cues + (tlr_cue,)
        plan = plan_tlr_block(tlr_cue, config=preset.tlr_block_config(overrides))
    gate = StableRemGate(
        RemGateConfig(
            enter_threshold=0.70,
            exit_threshold=0.45,
            min_stable_seconds=settings.gate_min_stable_seconds,
            epoch_seconds=30.0,
            cooldown_seconds=0.0,
        )
    )
    scheduler = TmrCueScheduler(
        assignment=assignment,
        catalog=catalog,
        cue_library=CueLibrary(cues=cues),
        config=preset.scheduler_config(enable_tlr_block=tlr, overrides=overrides),
        tlr_block_plan=plan,
        event_log_path=directory / "scheduler_events.jsonl",
    )
    records = []
    for index, probability in enumerate(probabilities):
        timestamp = index * 30.0
        decision = gate.update(RemPrediction(probability=probability, source="script"), duration_seconds=30.0)
        records.append(gate_decision_record(index=index, timestamp_seconds=timestamp, duration_seconds=30.0, decision=decision))
        scheduler.update(decision, timestamp_seconds=timestamp, guard_decision=(guard or {}).get(index))
    write_rem_gate_decisions(records, directory / "rem_gate_decisions.jsonl")
    metadata = {
        "protocol": preset.manifest(overrides),
        "tlr_block_plan": plan.to_dict() if plan is not None else None,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def motion_pause(index):
    return ArousalGuardDecision(
        action="pause",
        timestamp_seconds=index * 30.0,
        reason_codes=("motion_arousal_proxy",),
        pause_seconds=120.0,
    )


class GateSequenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_known_rem_periods_give_known_counts(self):
        # Two REM periods: the gate opens on the third and on the ninth epoch.
        night = write_night(self.root / "night", [0.2, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9])
        report = build_protocol_fidelity(night)

        self.assertEqual(report.preset["name"], "remind_default_v1")
        self.assertEqual(report.preset["source"], "the night's preset")
        self.assertEqual(report.summary["rem_periods"], 2)
        self.assertEqual(report.summary["plays"], {"tlr": 0, "puzzle": 2})
        self.assertEqual(report.summary["plays_outside_rem_periods"], 0)
        self.assertEqual([period["plays"]["puzzle"] for period in report.rem_periods], [2, 0])
        self.assertEqual([(period["start_seconds"], period["end_seconds"]) for period in report.rem_periods], [(90.0, 180.0), (270.0, 330.0)])
        intervals = report.checks["inter_cue_intervals"]["streams"]["puzzle"]
        self.assertEqual((intervals["interval_count"], intervals["within"], intervals["min_seconds"]), (1, 1, 30.0))
        self.assertEqual(report.checks["inter_cue_intervals"]["status"], "as_published")
        self.assertEqual(report.checks["cues_per_rem_period"]["streams"]["puzzle"]["per_period"], [2, 0])
        self.assertEqual(report.checks["cues_per_rem_period"]["status"], "as_published")
        stable = report.checks["stable_rem_before_first_cue"]
        self.assertEqual((stable["status"], stable["observed"]), ("as_published", [60.0]))
        self.assertEqual(report.checks["pause_rules"]["status"], "as_published")
        self.assertEqual(report.checks["tlr_block"]["status"], "not_used")
        self.assertEqual(report.limitations, ("no_audio_log",))

    def test_a_pause_shorter_than_published_is_found(self):
        probabilities = [0.9] * 12
        # konkoly2024_exp1 pauses 5 min after a movement; overriding the scheduler's
        # cooldown to 120 s lets the first cue come 120 s after the guard's pause.
        short = write_night(
            self.root / "short",
            probabilities,
            preset="konkoly2024_exp1",
            overrides={"scheduler_cooldown_seconds": 120.0},
            guard={1: motion_pause(1)},
        )
        report = build_protocol_fidelity(short)
        pauses = report.checks["pause_rules"]
        self.assertEqual(pauses["status"], "differs")
        self.assertEqual(
            pauses["guard_pauses"],
            [{"at_seconds": 60.0, "triggers": ["movement"], "published_pause_seconds": 300.0,
              "seconds_to_next_cue": 120.0, "honoured": False}],
        )
        self.assertIn("night_overrode_preset_settings", report.limitations)

        kept = write_night(self.root / "kept", probabilities, preset="konkoly2024_exp1", guard={1: motion_pause(1)})
        pauses = build_protocol_fidelity(kept).checks["pause_rules"]
        self.assertEqual(pauses["status"], "as_published")
        self.assertEqual(pauses["guard_pauses"][0]["seconds_to_next_cue"], 300.0)
        self.assertTrue(pauses["guard_pauses"][0]["honoured"])

        # A night without a cue honours every pause and shows nothing.
        quiet = write_night(self.root / "quiet", [0.2] * 6, preset="konkoly2024_exp1", guard={1: motion_pause(1)})
        self.assertEqual(build_protocol_fidelity(quiet).checks["pause_rules"]["status"], "no_cues")

    def test_a_cue_soon_after_a_reported_awakening_is_found(self):
        night = write_night(self.root / "night", [0.9] * 6, preset="konkoly2024_exp1")
        append_awakening_event(night / "awakening_events.jsonl", AwakeningEvent(timestamp_seconds=70.0, attributed_to_cue="yes"))
        append_awakening_event(night / "awakening_events.jsonl", AwakeningEvent(timestamp_utc="2026-10-27T03:00:00+00:00"))
        report = build_protocol_fidelity(night)
        pauses = report.checks["pause_rules"]
        self.assertEqual(pauses["status"], "differs")
        self.assertEqual(
            pauses["awakenings"],
            [{"at_seconds": 70.0, "published_pause_seconds": 2700.0, "seconds_to_next_cue": 20.0, "honoured": False}],
        )
        # The second marker has no time on the night's base and no summary to place it by.
        self.assertIn("awakening_markers_unplaced", report.limitations)

    def test_tlr_cues_planned_closer_than_the_stride_play_back_to_back(self):
        night = write_night(self.root / "default", [0.9] * 6, tlr=True)
        tlr = build_protocol_fidelity(night).checks["tlr_block"]
        self.assertEqual(tlr["status"], "differs")
        self.assertEqual(tlr["plan"], {"repetitions": 3, "interval_seconds": 8.0})
        self.assertEqual(
            tlr["periods"],
            [{"rem_period": 0, "cues": 3, "complete": True, "planned_intervals_seconds": [8.0, 8.0],
              "played_intervals_seconds": [30.0, 0.0], "back_to_back": 1}],
        )

        night = write_night(self.root / "northwestern", [0.9] * 14, preset="konkoly2021_northwestern", tlr=True)
        report = build_protocol_fidelity(night)
        tlr = report.checks["tlr_block"]
        self.assertEqual(tlr["status"], "as_planned")
        self.assertEqual(tlr["periods"][0]["cues"], 10)
        self.assertEqual(set(tlr["periods"][0]["played_intervals_seconds"]), {30.0})
        self.assertEqual(report.checks["inter_cue_intervals"]["streams"]["tlr"]["status"], "as_published")
        self.assertEqual(report.checks["cues_per_rem_period"]["streams"]["tlr"]["per_period"], [10])
        self.assertEqual(report.checks["cue_duration"]["streams"]["tlr"]["status"], "as_published")

        # A REM period that ends before the plan's tenth cue cuts the block short.
        night = write_night(self.root / "short", [0.9] * 8 + [0.1] * 2, preset="konkoly2021_northwestern", tlr=True)
        tlr = build_protocol_fidelity(night).checks["tlr_block"]
        self.assertEqual(tlr["status"], "differs")
        self.assertEqual([(row["cues"], row["complete"]) for row in tlr["periods"]], [(7, False)])
        self.assertEqual(set(tlr["periods"][0]["played_intervals_seconds"]), {30.0})

    def test_the_stable_window_against_the_published_start(self):
        probabilities = [0.2, 0.9, 0.9, 0.9, 0.9]
        nijmegen = build_protocol_fidelity(write_night(self.root / "n", probabilities, preset="konkoly2021_nijmegen"))
        stable = nijmegen.checks["stable_rem_before_first_cue"]
        self.assertEqual((stable["status"], stable["observed"]), ("later_than_published", [60.0]))

        borghese = build_protocol_fidelity(write_night(self.root / "b", [0.9] * 14, preset="borghese2022_rem_home"))
        stable = borghese.checks["stable_rem_before_first_cue"]
        self.assertEqual((stable["status"], stable["observed"]), ("as_published", [300.0]))
        # Its 10 s cue interval cannot be met at a 30 s stride.
        self.assertEqual(borghese.checks["inter_cue_intervals"]["streams"]["puzzle"]["longer"], 1)
        self.assertEqual(borghese.checks["inter_cue_intervals"]["status"], "differs")

    def test_the_gain_against_the_published_intensity(self):
        def ramp(name, preset, gains=None):
            night = write_night(self.root / name, [0.9] * 6, preset=preset, tlr=True)
            events = [json.loads(line) for line in (night / "scheduler_events.jsonl").read_text().splitlines()]
            plays = [event for event in events if event["event_type"] == "play"]
            gains = gains or [0.04] * len(plays)
            (night / "audio_playback.jsonl").write_text(
                "".join(json.dumps({"effective_volume": gain}) + "\n" for gain in gains[: len(plays)]),
                encoding="utf-8",
            )
            return build_protocol_fidelity(night).checks["volume_ramp"]

        # The default's own rule: one gain under the calibration cap.
        self.assertEqual(ramp("default", "remind_default_v1")["status"], "as_published")
        self.assertEqual(ramp("lowered", "remind_default_v1", [0.04, 0.02, 0.04, 0.04, 0.04])["status"], "differs")
        # Konkoly et al. 2024 raised the intensity with each cue; REMind does not.
        rising = ramp("exp1", "konkoly2024_exp1")
        self.assertEqual((rising["status"], rising["observed"]["constant_gain"]), ("not_implemented", True))
        # Schwartz et al. 2022, from the notes, say nothing about intensity.
        self.assertEqual(ramp("schwartz", "schwartz2022_irt_rem")["status"], "not_published")

    def test_another_preset_can_be_asked_for(self):
        night = write_night(self.root / "night", [0.9] * 5)
        report = build_protocol_fidelity(night, preset_name="konkoly2021_nijmegen")
        self.assertEqual(report.preset["name"], "konkoly2021_nijmegen")
        self.assertIn("compared_with_another_preset", report.limitations)
        self.assertEqual(report.night_protocol["name"], "remind_default_v1")

    def test_a_night_without_a_block_is_compared_with_the_default(self):
        night = write_night(self.root / "night", [0.9] * 5)
        (night / "metadata.json").write_text("{}", encoding="utf-8")
        report = build_protocol_fidelity(night)
        self.assertEqual(report.preset["name"], "remind_default_v1")
        self.assertIn("no_protocol_block", report.limitations)

    def test_the_cli_writes_json_and_markdown(self):
        night = write_night(self.root / "night", [0.2, 0.9, 0.9, 0.9])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cli_main(["protocol-fidelity", str(night)]), 0)
        self.assertIn("preset=remind_default_v1", stdout.getvalue())
        payload = json.loads((night / "protocol_fidelity.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["parameter_fidelity_is_not_result_fidelity"], PARAMETER_FIDELITY_IS_NOT_RESULT_FIDELITY)
        markdown = (night / "protocol_fidelity.md").read_text(encoding="utf-8")
        self.assertIn("Parameter fidelity is not result fidelity", markdown)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli_main(["protocol-fidelity", str(self.root / "missing")]), 2)
            self.assertEqual(cli_main(["protocol-fidelity", str(night), "--protocol-preset", "nope"]), 2)


class MandatoryBlockTest(unittest.TestCase):
    def test_the_block_cannot_be_left_out(self):
        report = ProtocolFidelityReport(
            recording="x",
            preset={},
            night_protocol=None,
            settings={},
            inputs={},
            summary={},
            rem_periods=(),
            checks={},
            deviations=(),
            limitations=(),
            parameter_fidelity_is_not_result_fidelity={},
        )
        block = report.to_dict()["parameter_fidelity_is_not_result_fidelity"]
        self.assertEqual(block, PARAMETER_FIDELITY_IS_NOT_RESULT_FIDELITY)
        for key in ("no_psg_verified_rem", "no_eog", "gain_not_db_spl", "first_cue_latency"):
            self.assertIn(key, block)
        self.assertIn("60 to 85 s", block["first_cue_latency"])
        self.assertIn("#167", block["first_cue_latency"])


class CleanNightTest(unittest.TestCase):
    """Half an hour of clean_night, seed 0 (synthetic input, not sleep), replayed as Pilot 3."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        night = generate_night(output_dir=root / "night", scenario="clean_night", hours=0.5, seed=0)
        cls.recording = night.recording_dir
        catalog, session, assignment, cues = protocol_fixture()
        for name, record in (("catalog", catalog), ("session", session), ("assignment", assignment)):
            record.save(root / f"{name}.json")
        CueLibrary(cues=cues).save(root / "cues.json")
        cls.report_path = root / "pilot3.json"
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main([
                "simulate-replay-cues", str(cls.recording),
                "--catalog", str(root / "catalog.json"),
                "--session", str(root / "session.json"),
                "--assignment", str(root / "assignment.json"),
                "--cue-library", str(root / "cues.json"),
                "--output", str(cls.report_path),
                "--preset", "p21",
                "--protocol-preset", "konkoly2026_puzzle",
            ])
        assert exit_code == 0
        cls.pilot3 = json.loads(cls.report_path.read_text(encoding="utf-8"))
        cls.fidelity = build_protocol_fidelity(cls.recording, pilot3_report=cls.report_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_rem_periods_and_cues_are_those_of_the_replay(self):
        runs = load_gate_open_runs(self.pilot3)
        self.assertGreaterEqual(len(runs), 1)
        self.assertEqual(self.fidelity.summary["rem_periods"], len(runs))
        self.assertEqual(self.fidelity.summary["plays"]["puzzle"], self.pilot3["metrics"]["cue_plan_count"])
        self.assertEqual(self.fidelity.summary["plays_outside_rem_periods"], 0)
        self.assertEqual(self.fidelity.summary["uncued_puzzle_play_count"], 0)
        self.assertEqual(self.fidelity.preset["name"], "konkoly2026_puzzle")
        self.assertEqual(self.fidelity.preset["content_hash"], self.pilot3["protocol"]["content_hash"])

    def test_a_synthetic_night_says_so(self):
        payload = self.fidelity.to_dict()
        self.assertTrue(payload["synthetic"])
        self.assertEqual(payload["note"], SYNTHETIC_NOTE)
        self.assertEqual(self.fidelity.limitations[0], "synthetic_input")
        self.assertIn("pilot3_plan_not_played", self.fidelity.limitations)
        self.assertIn("Synthetic input, not sleep", self.fidelity.to_markdown())
        # A Pilot 3 plan has no player, so there is no volume to compare.
        self.assertEqual(self.fidelity.checks["volume_ramp"]["status"], "not_available")


class NoBackendTest(unittest.TestCase):
    """No new module builds or picks an audio backend; the fidelity report and the
    presets import nothing from muse_tmr.audio."""

    def test_new_modules_import_no_backend(self):
        for relative in NEW_MODULES:
            tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported.update(alias.name for alias in node.names)
                    if relative.endswith(("protocol_fidelity.py", "protocol_presets.py", "protocol_registry.py")):
                        self.assertFalse((node.module or "").startswith("muse_tmr.audio"), relative)
                elif isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
            with self.subTest(module=relative):
                self.assertFalse(imported & BACKEND_NAMES, relative)
                self.assertNotIn("subprocess", imported)


if __name__ == "__main__":
    unittest.main()
