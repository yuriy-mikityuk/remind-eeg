import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from muse_tmr.cli.main import main
from muse_tmr.protocol import (
    NightPuzzleSession,
    PuzzleCatalog,
    PuzzleCueAssignment,
    PuzzleTask,
    TmrSchedulerEvent,
    append_tmr_scheduler_events,
    get_protocol_preset,
)
from muse_tmr.reports import (
    DreamPuzzleIncorporation,
    DreamReport,
    MorningRetest,
    MorningRetestResult,
    build_cue_recall_test,
    build_cued_uncued_analysis,
    cue_recall_order,
    load_cued_uncued_analysis,
)


class TestCuedUncuedAnalysis(unittest.TestCase):
    def test_analysis_computes_group_rates_effects_and_cue_timing(self):
        session, assignment, dream_report, retest = _analysis_inputs()
        events = (
            TmrSchedulerEvent("play", 100.0, cue_id="cue-p1", protocol="puzzle", puzzle_id="p1"),
            TmrSchedulerEvent("play", 130.0, cue_id="cue-p2", protocol="puzzle", puzzle_id="p2"),
        )

        report = build_cued_uncued_analysis(
            session,
            assignment,
            retest,
            dream_report=dream_report,
            scheduler_events=events,
            min_group_size=3,
        )

        metrics = {item.cue_condition: item for item in report.condition_metrics}
        self.assertEqual(metrics["cued"].puzzle_count, 2)
        self.assertEqual(metrics["uncued"].puzzle_count, 2)
        self.assertEqual(metrics["cued"].solve_rate, 0.5)
        self.assertEqual(metrics["uncued"].solve_rate, 0.0)
        self.assertEqual(report.effect_summary["solve_rate_difference_cued_minus_uncued"], 0.5)
        self.assertEqual(report.effect_summary["incorporation_rate_difference_cued_minus_uncued"], 0.0)
        self.assertEqual(report.cue_timing["puzzle_play_event_count"], 2)
        self.assertEqual(report.cue_timing["first_puzzle_cue_seconds"], 100.0)
        self.assertEqual(report.rows[0].cue_play_count, 1)
        self.assertEqual(report.rows[2].cue_condition, "uncued")
        self.assertIn("small_n", report.limitation_codes)

    def test_analysis_marks_missing_optional_inputs_as_limitations(self):
        session, assignment, _, retest = _analysis_inputs()

        report = build_cued_uncued_analysis(session, assignment, retest)

        self.assertIn("missing_dream_report", report.limitation_codes)
        self.assertIn("missing_scheduler_events", report.limitation_codes)
        self.assertIn("cue_recall_missing", report.limitation_codes)
        self.assertEqual(report.cue_recall, {"available": False})
        self.assertEqual(report.cue_timing["puzzle_play_event_count"], 0)

    def test_the_sound_check_gives_recall_by_condition(self):
        session, assignment, dream_report, retest = _analysis_inputs()
        order = cue_recall_order(session)
        # Every sound placed except p4's (uncued), which was not known: the two conditions
        # differ, so a swap of cued and uncued shows.
        answers = {index: (None if puzzle == "p4" else puzzle) for index, puzzle in enumerate(order, start=1)}
        recall = build_cue_recall_test(
            session, _catalog(), assignment, answers, completed_at_utc="2026-10-27T07:00:00+00:00"
        )
        report = build_cued_uncued_analysis(
            session, assignment, retest, dream_report=dream_report, cue_recall=recall, min_group_size=1
        )
        self.assertEqual(report.cue_recall["recall_rate"], 0.75)
        self.assertEqual(report.cue_recall["by_condition"]["cued"], {"trial_count": 2, "correct_count": 2, "recall_rate": 1.0})
        self.assertEqual(report.cue_recall["by_condition"]["uncued"], {"trial_count": 2, "correct_count": 1, "recall_rate": 0.5})
        self.assertEqual(report.cue_recall["recall_rate_difference_cued_minus_uncued"], 0.5)
        self.assertTrue(report.cue_recall["before_retest"])
        self.assertEqual({row.puzzle_id: row.cue_recalled for row in report.rows},
                         {"p1": True, "p2": True, "p3": True, "p4": False})
        self.assertNotIn("cue_recall_missing", report.limitation_codes)
        self.assertNotIn("cue_recall_after_retest", report.limitation_codes)
        self.assertIn("Sound check", report.to_markdown())
        self.assertEqual(load_cued_uncued_analysis_round_trip(report).cue_recall, report.cue_recall)

    def test_a_sound_check_after_the_retest_is_a_limitation(self):
        session, assignment, _, retest = _analysis_inputs()
        answers = {index: puzzle for index, puzzle in enumerate(cue_recall_order(session), start=1)}
        late = build_cue_recall_test(session, _catalog(), assignment, answers, completed_at_utc="2026-10-27T09:00:00+00:00")
        report = build_cued_uncued_analysis(session, assignment, retest, cue_recall=late)
        self.assertFalse(report.cue_recall["before_retest"])
        self.assertIn("cue_recall_after_retest", report.limitation_codes)

    def test_analysis_detects_uncued_scheduler_play_events(self):
        session, assignment, dream_report, retest = _analysis_inputs()
        events = (
            TmrSchedulerEvent("play", 100.0, cue_id="cue-p3", protocol="puzzle", puzzle_id="p3"),
        )

        report = build_cued_uncued_analysis(
            session,
            assignment,
            retest,
            dream_report=dream_report,
            scheduler_events=events,
        )

        self.assertIn("uncued_cue_play_observed", report.limitation_codes)
        self.assertEqual(report.cue_timing["uncued_puzzle_play_count"], 1)

    def test_analysis_round_trips_json_and_markdown(self):
        session, assignment, dream_report, retest = _analysis_inputs()
        report = build_cued_uncued_analysis(
            session,
            assignment,
            retest,
            dream_report=dream_report,
            min_group_size=1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "analysis.json"
            loaded = load_cued_uncued_analysis(report.save(report_path))

        self.assertEqual(loaded.session_id, "night-001")
        self.assertEqual(loaded.rows[0].puzzle_id, "p1")
        self.assertIn("Cued vs Uncued Analysis", loaded.to_markdown())

    def test_cli_writes_analysis_report_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session, assignment, dream_report, retest = _analysis_inputs()
            session_path = tmp_path / "session.json"
            assignment_path = tmp_path / "assignment.json"
            dream_path = tmp_path / "dream.json"
            retest_path = tmp_path / "retest.json"
            events_path = tmp_path / "scheduler.jsonl"
            output_path = tmp_path / "analysis.json"
            markdown_path = tmp_path / "analysis.md"
            session.save(session_path)
            assignment.save(assignment_path)
            dream_report.save(dream_path)
            retest.save(retest_path)
            append_tmr_scheduler_events(
                (
                    TmrSchedulerEvent("play", 100.0, cue_id="cue-p1", protocol="puzzle", puzzle_id="p1"),
                    TmrSchedulerEvent("play", 130.0, cue_id="cue-p2", protocol="puzzle", puzzle_id="p2"),
                ),
                events_path,
            )

            with redirect_stdout(io.StringIO()):
                code = main([
                    "analyze-cued-uncued",
                    str(session_path),
                    "--assignment",
                    str(assignment_path),
                    "--retest",
                    str(retest_path),
                    "--dream-report",
                    str(dream_path),
                    "--scheduler-events",
                    str(events_path),
                    "--output",
                    str(output_path),
                    "--markdown-output",
                    str(markdown_path),
                    "--min-group-size",
                    "3",
                ])
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            markdown_exists = markdown_path.exists()

        self.assertEqual(code, 0)
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["condition_metrics"]["cued"]["solve_rate"], 0.5)
        self.assertTrue(markdown_exists)
        self.assertNotIn("protocol", payload["metadata"])

    def test_cli_takes_the_sound_check_and_the_protocol_block(self):
        session, assignment, _, retest = _analysis_inputs()
        block = get_protocol_preset("konkoly2026_puzzle").manifest()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session.save(tmp_path / "session.json")
            assignment.save(tmp_path / "assignment.json")
            retest.save(tmp_path / "retest.json")
            answers = {index: puzzle for index, puzzle in enumerate(cue_recall_order(session), start=1)}
            build_cue_recall_test(
                session, _catalog(), assignment, answers, completed_at_utc="2026-10-27T07:00:00+00:00"
            ).save(tmp_path / "recall.json")
            (tmp_path / "metadata.json").write_text(json.dumps({"protocol": block}), encoding="utf-8")
            (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
            base = [
                "analyze-cued-uncued", str(tmp_path / "session.json"),
                "--assignment", str(tmp_path / "assignment.json"),
                "--retest", str(tmp_path / "retest.json"),
                "--output", str(tmp_path / "analysis.json"),
            ]
            with redirect_stdout(io.StringIO()):
                code = main([*base, "--cue-recall", str(tmp_path / "recall.json"),
                             "--protocol-metadata", str(tmp_path / "metadata.json")])
            payload = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                refused = main([*base, "--protocol-metadata", str(tmp_path / "empty.json")])

        self.assertEqual(code, 0)
        self.assertEqual(payload["metadata"]["protocol"]["content_hash"], block["content_hash"])
        self.assertEqual(payload["metadata"]["protocol_source"], "metadata.json")
        self.assertEqual(payload["cue_recall"]["recall_rate"], 1.0)
        self.assertEqual(refused, 2)


def load_cued_uncued_analysis_round_trip(report):
    with tempfile.TemporaryDirectory() as tmp:
        return load_cued_uncued_analysis(report.save(Path(tmp) / "analysis.json"))


def _catalog():
    return PuzzleCatalog(
        puzzles=tuple(PuzzleTask(puzzle, f"Prompt {puzzle}", f"Answer {puzzle}", cue_id=f"cue-{puzzle}") for puzzle in ("p1", "p2", "p3", "p4"))
    )


def _analysis_inputs():
    session = NightPuzzleSession(
        session_id="night-001",
        puzzle_ids=("p1", "p2", "p3", "p4"),
        puzzle_count=4,
    )
    assignment = PuzzleCueAssignment(
        session_id="night-001",
        cued_puzzle_ids=("p1", "p2"),
        uncued_puzzle_ids=("p3", "p4"),
        seed=17,
    )
    dream_report = DreamReport(
        session_id="night-001",
        lucid=False,
        cues_heard=True,
        confidence=0.6,
        dream_text="I saw puzzle fragments.",
        puzzle_incorporations=(
            DreamPuzzleIncorporation("p1", "first cue fragment", cue_id="cue-p1", confidence=0.8),
            DreamPuzzleIncorporation("p3", "control fragment", cue_id="cue-p3", confidence=0.4),
        ),
    )
    retest = MorningRetest(
        session_id="night-001",
        reported_at_utc="2026-10-27T08:00:00+00:00",
        results=(
            MorningRetestResult("p1", "answer 1", True, 12.0, 0.9, cue_id="cue-p1", cue_condition="cued"),
            MorningRetestResult("p2", "wrong", False, 24.0, 0.4, cue_id="cue-p2", cue_condition="cued"),
            MorningRetestResult("p3", "wrong", False, 18.0, 0.5, cue_id="cue-p3", cue_condition="uncued"),
            MorningRetestResult("p4", "", False, 20.0, 0.2, cue_id="cue-p4", cue_condition="uncued"),
        ),
    )
    return session, assignment, dream_report, retest


if __name__ == "__main__":
    unittest.main()
