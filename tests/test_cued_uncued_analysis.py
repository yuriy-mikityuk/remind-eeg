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
    CuedUncuedAnalysisReport,
    DreamPuzzleIncorporation,
    DreamReport,
    IncorporationCodingSheet,
    MorningRetest,
    MorningRetestResult,
    build_cue_recall_test,
    build_cued_uncued_analysis,
    build_incorporation_coding,
    cue_recall_order,
    export_blind_coding_packet,
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
        self.assertEqual(payload["schema_version"], 3)
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


class TestSchema3(unittest.TestCase):
    """Retest waves, rater codes and sham plays (#228)."""

    def test_each_wave_gets_its_metrics_and_deltas(self):
        session, assignment, dream_report, retest = _analysis_inputs()
        day7 = _retest_wave("day7", solved=("p2", "p3"))
        report = build_cued_uncued_analysis(session, assignment, (day7, retest), dream_report=dream_report, min_group_size=1)

        self.assertEqual(list(report.wave_metrics), ["morning", "day7"])
        morning = report.wave_metrics["morning"]
        self.assertEqual(morning["effect_summary"], report.effect_summary)
        self.assertEqual(morning["condition_metrics"]["cued"], report.condition_metrics[0].to_dict())
        later = report.wave_metrics["day7"]
        self.assertEqual(later["condition_metrics"]["cued"]["solve_rate"], 0.5)
        self.assertEqual(later["condition_metrics"]["uncued"]["solve_rate"], 0.5)
        self.assertEqual(later["effect_summary"]["solve_rate_difference_cued_minus_uncued"], 0.0)
        self.assertEqual(later["reported_at_utc"], day7.reported_at_utc)
        self.assertEqual(report.rows[0].wave_results["day7"]["solved"], False)
        self.assertTrue(report.rows[0].solved)
        missing = next(item for item in report.limitations if item.code == "missing_delayed_retest")
        self.assertEqual(missing.metadata["missing_waves"], ["day10"])
        self.assertIn("## Retest Waves", report.to_markdown())
        with self.assertRaises(ValueError):
            build_cued_uncued_analysis(session, assignment, (retest, _retest_wave("morning")))
        only_later = build_cued_uncued_analysis(session, assignment, (day7,))
        self.assertEqual(only_later.metadata["primary_retest_wave"], "day7")
        # Without a morning retest the rows are the earliest wave's, whatever the order given.
        delayed = build_cued_uncued_analysis(session, assignment, (_retest_wave("day10", solved=("p4",)), day7))
        self.assertEqual(delayed.metadata["primary_retest_wave"], "day7")
        self.assertEqual(list(delayed.wave_metrics), ["day7", "day10"])
        self.assertEqual({row.puzzle_id: row.solved for row in delayed.rows},
                         {"p1": False, "p2": True, "p3": True, "p4": False})
        complete = build_cued_uncued_analysis(session, assignment, (retest, day7, _retest_wave("day10")))
        self.assertNotIn("missing_delayed_retest", complete.limitation_codes)

    def test_the_cross_tab_counts_each_waves_own_solves(self):
        # The dream report links p1 and p3; the morning retest solved p1, the day7 one p2 and p3.
        session, assignment, dream_report, retest = _analysis_inputs()
        day7 = _retest_wave("day7", solved=("p2", "p3"))
        report = build_cued_uncued_analysis(session, assignment, (retest, day7), dream_report=dream_report, min_group_size=1)
        waves = report.incorporation_solved["waves"]
        self.assertEqual(waves["morning"]["incorporated"], {"puzzles": 2, "solved": 1, "unsolved": 1, "solve_rate": 0.5})
        self.assertEqual(waves["morning"]["not_incorporated"], {"puzzles": 2, "solved": 0, "unsolved": 2, "solve_rate": 0.0})
        self.assertEqual(waves["day7"]["incorporated"], {"puzzles": 2, "solved": 1, "unsolved": 1, "solve_rate": 0.5})
        self.assertEqual(waves["day7"]["not_incorporated"], {"puzzles": 2, "solved": 1, "unsolved": 1, "solve_rate": 0.5})
        self.assertEqual(waves["day7"]["by_condition"]["uncued"]["incorporated"]["solved"], 1)
        self.assertEqual(waves["morning"]["by_condition"]["uncued"]["incorporated"]["solved"], 0)

    def test_rater_codes_win_over_the_sleepers_links(self):
        session, assignment, dream_report, retest = _analysis_inputs()
        packet = export_blind_coding_packet(session, dream_report, _catalog())
        codes = {"p1": (), "p2": ("direct_conceptual",), "p3": (), "p4": ("sound_cue",)}
        rater_a = build_incorporation_coding(packet, "rater-a", codes)
        rater_b = build_incorporation_coding(packet, "rater-b", dict(codes, p4=()))
        report = build_cued_uncued_analysis(
            session, assignment, retest, dream_report=dream_report, incorporation_codings=(rater_a, rater_b), min_group_size=1
        )

        self.assertEqual({row.puzzle_id: row.incorporated for row in report.rows}, {"p1": False, "p2": True, "p3": False, "p4": False})
        self.assertEqual({row.puzzle_id: row.self_linked for row in report.rows}, {"p1": True, "p2": False, "p3": True, "p4": False})
        self.assertEqual(report.rows[3].rater_codes["any"], 0.5)
        self.assertEqual(report.incorporation["source"], "raters")
        self.assertEqual(report.incorporation["rater_ids"], ["rater-a", "rater-b"])
        self.assertEqual(report.incorporation["split_puzzle_ids"], ["p4"])
        self.assertEqual(report.effect_summary["incorporation_rate_difference_cued_minus_uncued"], 0.5)
        cross = report.incorporation_solved["waves"]["morning"]
        self.assertEqual(cross["incorporated"], {"puzzles": 1, "solved": 0, "unsolved": 1, "solve_rate": 0.0})
        self.assertEqual(cross["not_incorporated"]["solved"], 1)
        self.assertEqual(cross["by_condition"]["cued"]["not_incorporated"]["solved"], 1)
        self.assertIn("rater codes (2 raters)", report.to_markdown())
        self.assertIn("## Incorporation And Solving", report.to_markdown())
        other = IncorporationCodingSheet(session_id="night-002", rater_id="rater-c", codings=())
        with self.assertRaises(ValueError):
            build_cued_uncued_analysis(session, assignment, retest, incorporation_codings=(other,))

    def test_without_a_dream_report_or_codes_there_is_no_cross_tab(self):
        session, assignment, _, retest = _analysis_inputs()
        report = build_cued_uncued_analysis(session, assignment, retest)
        self.assertEqual(report.incorporation["source"], "none")
        self.assertEqual(report.incorporation_solved, {"available": False, "source": "none"})

    def test_a_sham_nights_plays_are_not_plays_of_the_cued_sounds(self):
        session, assignment, dream_report, retest = _analysis_inputs()
        events = (
            TmrSchedulerEvent("play", 100.0, cue_id="sham_soft_tone", protocol="puzzle", puzzle_id="p1",
                              reason_codes=("sham_arm",), arm="sham"),
            TmrSchedulerEvent("play", 130.0, cue_id="sham_soft_tone", protocol="puzzle", puzzle_id="p2",
                              reason_codes=("sham_arm",), arm="sham"),
            TmrSchedulerEvent("play", 160.0, cue_id="sham_soft_tone", protocol="tlr", reason_codes=("sham_arm",), arm="sham"),
        )
        report = build_cued_uncued_analysis(session, assignment, retest, dream_report=dream_report, scheduler_events=events)

        self.assertEqual([row.cue_play_count for row in report.rows], [0, 0, 0, 0])
        self.assertEqual([row.sham_play_count for row in report.rows], [1, 1, 0, 0])
        self.assertEqual(report.cue_timing["puzzle_play_event_count"], 0)
        self.assertEqual(report.cue_timing["cued_puzzle_play_count"], 0)
        self.assertEqual(report.cue_timing["sham_puzzle_play_count"], 2)
        self.assertEqual(report.cue_timing["arm"], "sham")
        self.assertIsNone(report.cue_timing["first_puzzle_cue_seconds"])
        self.assertEqual(report.condition_metrics[0].cue_play_count, 0)
        self.assertIn("sham_arm", report.limitation_codes)
        self.assertNotIn("missing_cued_timing", report.limitation_codes)
        self.assertNotIn("uncued_cue_play_observed", report.limitation_codes)
        self.assertIn("Sham plays in place of puzzle cues: 2", report.to_markdown())
        # A sham play that names an uncued puzzle still breaks the contract.
        stray = events + (
            TmrSchedulerEvent("play", 190.0, cue_id="sham_soft_tone", protocol="puzzle", puzzle_id="p3",
                              reason_codes=("sham_arm",), arm="sham"),
        )
        broken = build_cued_uncued_analysis(session, assignment, retest, scheduler_events=stray)
        self.assertIn("uncued_cue_play_observed", broken.limitation_codes)
        self.assertEqual(broken.cue_timing["uncued_puzzle_play_count"], 1)

    def test_a_no_cue_night_counts_its_virtual_plays(self):
        session, assignment, _, retest = _analysis_inputs()
        events = (
            TmrSchedulerEvent("skip", 100.0, cue_id="cue-p1", protocol="puzzle", puzzle_id="p1",
                              reason_codes=("no_cue_arm",), metadata={"virtual_play": True}, arm="no_cue"),
            TmrSchedulerEvent("skip", 130.0, reason_codes=("gate_closed",), arm="no_cue"),
        )
        report = build_cued_uncued_analysis(session, assignment, retest, scheduler_events=events)
        self.assertEqual(report.cue_timing["virtual_puzzle_play_count"], 1)
        self.assertEqual(report.cue_timing["arm"], "no_cue")
        self.assertIn("no_cue_arm", report.limitation_codes)
        self.assertNotIn("missing_cued_timing", report.limitation_codes)
        with self.assertRaises(ValueError):
            build_cued_uncued_analysis(
                session, assignment, retest,
                scheduler_events=events + (TmrSchedulerEvent("play", 160.0, cue_id="cue-p2", protocol="puzzle", puzzle_id="p2"),),
            )

    def test_a_schema_2_report_still_loads(self):
        session, assignment, dream_report, retest = _analysis_inputs()
        payload = build_cued_uncued_analysis(session, assignment, retest, dream_report=dream_report).to_dict()
        payload["schema_version"] = 2
        for key in ("wave_metrics", "incorporation", "incorporation_solved"):
            payload.pop(key)
        for row in payload["rows"]:
            for key in ("self_linked", "rater_codes", "sham_play_count", "wave_results"):
                row.pop(key)
        loaded = CuedUncuedAnalysisReport.from_dict(payload)
        self.assertEqual(loaded.schema_version, 2)
        self.assertEqual(loaded.wave_metrics, {})
        self.assertEqual([row.self_linked for row in loaded.rows], [True, False, True, False])

    def test_cli_takes_waves_and_codings(self):
        session, assignment, dream_report, retest = _analysis_inputs()
        packet = export_blind_coding_packet(session, dream_report, _catalog())
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session.save(tmp_path / "session.json")
            assignment.save(tmp_path / "assignment.json")
            retest.save(tmp_path / "retest.json")
            _retest_wave("day7").save(tmp_path / "retest_day7.json")
            build_incorporation_coding(packet, "rater-a", {"p1": ("sound_cue",), "p2": (), "p3": (), "p4": ()}).save(
                tmp_path / "coding.json"
            )
            base = [
                "analyze-cued-uncued", str(tmp_path / "session.json"),
                "--assignment", str(tmp_path / "assignment.json"),
                "--retest", str(tmp_path / "retest.json"),
                "--output", str(tmp_path / "analysis.json"),
            ]
            out = io.StringIO()
            with redirect_stdout(out):
                code = main([*base, "--retest", str(tmp_path / "retest_day7.json"), "--coding", str(tmp_path / "coding.json")])
            payload = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                twice = main([*base, "--retest", str(tmp_path / "retest.json")])

        self.assertEqual(code, 0)
        self.assertIn("waves=morning,day7 incorporation=raters", out.getvalue())
        self.assertEqual(sorted(payload["wave_metrics"]), ["day7", "morning"])
        self.assertEqual(payload["incorporation"]["rater_ids"], ["rater-a"])
        self.assertEqual(twice, 2)


def _retest_wave(wave, solved=()):
    return MorningRetest(
        session_id="night-001",
        retest_wave=wave,
        reported_at_utc="2026-11-03T08:00:00+00:00",
        results=tuple(
            MorningRetestResult(puzzle, "x", puzzle in solved, 30.0, 0.5, cue_id=f"cue-{puzzle}")
            for puzzle in ("p1", "p2", "p3", "p4")
        ),
    )


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
