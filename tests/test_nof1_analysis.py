"""The per-arm description of an N-of-1 series (#228), on hand-made synthetic series:
synthetic input, not sleep."""

import ast
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from muse_tmr.cli import main as cli_main
from muse_tmr.protocol import BLOCK_ARMS, NightSchedule, ScheduledNight, generate_night_schedule
from muse_tmr.reports import (
    Nof1NightInputs,
    ScheduleStillSealed,
    build_nof1_analysis,
    load_series_inputs,
)
from muse_tmr.reports import nof1_analysis
from muse_tmr.reports.nof1_analysis import (
    CONTRASTS,
    DEFAULT_PERMUTATIONS,
    DEFAULT_SEED,
    OUTCOMES,
    checklist_counts,
    schedule_design_orders,
)
from tests.nof1_synthetic import SYNTHETIC_NOTE, arms_by_night, block_orders, write_series

SRC = Path(__file__).resolve().parents[1] / "src" / "muse_tmr"
MORNING_DELTA = "cued_minus_uncued_solve_rate_morning"
GENERATED = "2026-12-31T00:00:00+00:00"


def _analyse(series, **kwargs):
    nights = load_series_inputs(
        series.schedule,
        protocol_dir=series.protocol_dir,
        reports_dir=series.reports_dir,
        recordings_dir=series.recordings_dir,
    )
    return build_nof1_analysis(series.schedule, nights, generated_at_utc=GENERATED, **kwargs)


class _SeriesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def series(self, name="series", **kwargs):
        return write_series(self.tmp / name, **kwargs)


class TestSealAndInputs(_SeriesTest):
    def test_a_sealed_schedule_is_refused_and_names_unseal(self):
        series = self.series(unseal=False)
        with self.assertRaises(ScheduleStillSealed) as caught:
            build_nof1_analysis(series.schedule, ())
        self.assertIn("unseal-night-schedule", str(caught.exception))

        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = cli_main.main(["analyze-nof1", str(series.schedule_path), "--output", str(self.tmp / "out.json")])
        self.assertEqual(code, 2)
        self.assertIn("unseal-night-schedule", stderr.getvalue())
        self.assertFalse((self.tmp / "out.json").exists())

    def test_a_changed_schedule_is_refused(self):
        series = self.series()
        payload = json.loads(series.schedule_path.read_text(encoding="utf-8"))
        payload["nights"][0]["arm"], payload["nights"][1]["arm"] = payload["nights"][1]["arm"], payload["nights"][0]["arm"]
        series.schedule_path.write_text(json.dumps(payload), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli_main.main(["analyze-nof1", str(series.schedule_path), "--output", str(self.tmp / "out.json")])
        self.assertEqual(code, 1)

    def test_finds_every_nights_files_by_its_id(self):
        series = self.series(raters=("rater-a",))
        nights = load_series_inputs(
            series.schedule,
            protocol_dir=series.protocol_dir,
            reports_dir=series.reports_dir,
            recordings_dir=series.recordings_dir,
        )
        self.assertEqual([item.night_id for item in nights], [n.night_id for n in series.schedule.nights])
        for item in nights:
            self.assertIsNotNone(item.session)
            self.assertIsNotNone(item.assignment)
            self.assertIsNotNone(item.dream_report)
            self.assertEqual(item.retest_waves, ("morning", "day7", "day10"))
            self.assertEqual([sheet.rater_id for sheet in item.codings], ["rater-a"])
            self.assertEqual(item.summary["night_id"], item.night_id)
            self.assertEqual(item.arousal_source, "summary")
            self.assertTrue(item.synthetic)

    def test_a_summary_of_another_series_is_ignored_or_refused(self):
        series = self.series()
        other = self.series("other", schedule_seed=1)
        # The other series has the same night IDs but another commitment.
        stray = series.recordings_dir / "stray" / "pilot5_summary.json"
        stray.parent.mkdir()
        shutil.copy(other.recordings_dir / "night-001" / "pilot5_summary.json", stray)
        nights = load_series_inputs(series.schedule, recordings_dir=series.recordings_dir)
        self.assertEqual(nights[0].summary_path, str(series.recordings_dir / "night-001" / "pilot5_summary.json"))
        with self.assertRaises(ValueError):
            load_series_inputs(series.schedule, summary_paths=(stray,))

    def test_two_files_of_one_kind_for_a_night_raise(self):
        series = self.series()
        shutil.copy(
            series.reports_dir / "night-002_dream_report.json", series.reports_dir / "night-002_dream_report_copy.json"
        )
        with self.assertRaises(ValueError) as caught:
            load_series_inputs(series.schedule, reports_dir=series.reports_dir)
        self.assertIn("night-002", str(caught.exception))

    def test_a_pilot3_report_can_stand_for_a_night(self):
        series = self.series()
        night_dir = series.recordings_dir / "night-001"
        summary = json.loads((night_dir / "pilot5_summary.json").read_text(encoding="utf-8"))
        shutil.rmtree(night_dir)
        replay = self.tmp / "replays" / "night-001_pilot3.json"
        replay.parent.mkdir()
        replay.write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "pilot_id": "m8_pilot3_replay_cue_simulation",
                    "night_id": "night-001",
                    "arm": summary["arm"],
                    "schedule_commitment_sha256": summary["schedule_commitment_sha256"],
                    "metrics": {"cue_plan_count": 3, "virtual_play_count": 0},
                }
            ),
            encoding="utf-8",
        )
        nights = load_series_inputs(series.schedule, recordings_dir=series.recordings_dir, summary_paths=(replay,))
        report = build_nof1_analysis(series.schedule, nights)
        first = report.nights[0]
        self.assertEqual((first["summary_kind"], first["plays"], first["virtual_plays"]), ("pilot3_report", 3, 0))
        self.assertIsNone(first["started_at"])

    def test_reads_arousal_from_the_scorers_report_when_the_summary_has_none(self):
        series = self.series(arousal_scored=False)
        night_dir = series.recordings_dir / "night-001"
        (night_dir / "cue_arousal_report.json").write_text(
            json.dumps({"synthetic": True, "outcomes": {"arousal_proxy_per_cue": {"any": 0.25, "imu": 0.1}}}),
            encoding="utf-8",
        )
        nights = load_series_inputs(series.schedule, recordings_dir=series.recordings_dir)
        self.assertEqual(nights[0].arousal_proxy_per_cue, {"any": 0.25, "imu": 0.1})
        self.assertEqual(nights[0].arousal_source, "cue_arousal_report.json")
        self.assertIsNone(nights[1].arousal_proxy_per_cue)


class TestPermutationIntervals(_SeriesTest):
    def test_the_reference_set_is_every_order_the_schedule_can_draw(self):
        for blocks in (1, 2, 3, 4):
            drawn = {
                tuple(night.arm for night in generate_night_schedule(seed, blocks, schedule_id="s", salt="0" * 32).nights)
                for seed in range(3000)
            }
            self.assertEqual(schedule_design_orders(blocks, BLOCK_ARMS), len(drawn), blocks)
        report = _analyse(self.series())
        self.assertEqual(report.settings["reference_set"], "schedule_design")
        self.assertEqual(report.settings["possible_orders"], 72)
        self.assertEqual(report.settings["method"], "exact")
        self.assertEqual(report.settings["relabelings"], 72)

    def test_the_same_seed_gives_the_same_report(self):
        series = self.series()
        first = _analyse(series).to_dict()
        again = _analyse(series).to_dict()
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(again, sort_keys=True))
        # Past --permutations orders are drawn, from the seed alone.
        long = self.series("long", blocks=10)
        drawn = _analyse(long, permutations=500, seed=3)
        self.assertEqual(drawn.settings["method"], "monte_carlo")
        self.assertEqual(drawn.settings["possible_orders"], 10368)
        self.assertEqual(drawn.settings["relabelings"], 500)
        self.assertEqual(drawn.to_dict(), _analyse(long, permutations=500, seed=3).to_dict())

    def test_a_planted_difference_needs_enough_blocks_to_fall_outside(self):
        planted = _analyse(self.series("planted", blocks=8, planted=True))
        plain = _analyse(self.series("plain", blocks=8))
        self.assertEqual(planted.settings["possible_orders"], 1728)
        self.assertTrue(planted.difference(MORNING_DELTA, "cue-no_cue")["observed_outside_interval"])
        self.assertFalse(plain.difference(MORNING_DELTA, "cue-no_cue")["observed_outside_interval"])
        # Two blocks: the schedule has 12 orders, too few for anything to fall outside.
        short = _analyse(self.series("short", blocks=2, planted=True))
        self.assertEqual(short.settings["possible_orders"], 12)
        self.assertFalse(short.settings["outside_possible"])
        self.assertTrue(all(item["observed_outside_interval"] is not True for item in short.differences))
        small_n = next(item for item in short.limitations if item.code == "small_n")
        self.assertFalse(small_n.metadata["outside_possible"])

    def test_a_twelve_night_series_with_and_without_a_planted_difference(self):
        """The figures the PR reports, synthetic input, not sleep. Four blocks allow 72
        orders, and with a share of four puzzles or a yes/no per night ties among them keep
        even the planted difference on the edge of its interval, not outside it."""
        plain = _analyse(self.series("plain"))
        planted = _analyse(self.series("planted", planted=True))
        for report in (plain, planted):
            self.assertEqual(len(report.nights), 12)
            self.assertEqual((report.settings["possible_orders"], report.settings["method"]), (72, "exact"))

        def figures(report, outcome):
            item = report.difference(outcome, "cue-no_cue")
            return item["difference"], item["interval"], item["observed_outside_interval"]

        self.assertEqual(figures(plain, MORNING_DELTA), (0.625, [-0.625, 0.625], False))
        self.assertEqual(figures(planted, MORNING_DELTA), (1.0, [-1.0, 1.0], False))
        self.assertEqual(figures(plain, "lucid"), (0.25, [-0.75, 0.75], False))
        self.assertEqual(figures(planted, "lucid"), (0.5, [-1.0, 1.0], False))

    def test_a_schedule_the_planner_could_not_draw_uses_every_order_of_every_block(self):
        nights = tuple(
            ScheduledNight(f"night-{3 * (block - 1) + place:03d}", 3 * (block - 1) + place, arm, block)
            for block in (1, 2, 3)
            for place, arm in enumerate(BLOCK_ARMS, start=1)
        )
        schedule = NightSchedule(
            schedule_id="same-order", seed=1, blocks=3, washout_nights=0, nights=nights, salt="0" * 32
        )
        design = nof1_analysis._design(schedule)
        self.assertEqual((design.reference_set, design.size), ("within_block", 216))
        drawn = generate_night_schedule(1, 3, schedule_id="s", salt="0" * 32)
        self.assertEqual(nof1_analysis._design(drawn).reference_set, "schedule_design")

    def test_graded_values_by_hand(self):
        """cue 1, sham 0.5, no_cue 0 on every block: the observed cue - no_cue difference is
        the one largest of the 72 orders, so it is outside; equal values give 0 inside."""
        schedule = generate_night_schedule(2028, 4, schedule_id="s", salt="0" * 32)
        design = nof1_analysis._design(schedule)
        relabelings, method = nof1_analysis._relabelings(design, seed=0, permutations=10000)
        self.assertEqual((len(relabelings), method), (72, "exact"))
        grade = {"cue": 1.0, "sham": 0.5, "no_cue": 0.0}
        values = tuple(tuple(grade[arm] for arm in block) for block in design.observed)
        observed = nof1_analysis._contrast(values, design.observed, "cue", "no_cue")
        permuted = [nof1_analysis._contrast(values, orders, "cue", "no_cue") for orders in relabelings]
        low, high = nof1_analysis._interval(permuted)
        self.assertEqual(observed, 1.0)
        self.assertEqual(sorted(permuted)[-2:], [0.875, 1.0])
        self.assertEqual(high, 0.875)
        self.assertLess(low, 0.0)
        flat = tuple(tuple(0.5 for _ in block) for block in design.observed)
        self.assertEqual({nof1_analysis._contrast(flat, orders, "cue", "no_cue") for orders in relabelings}, {0.0})
        # A schedule plan-night-schedule could not draw falls back to every order of every block.
        same = tuple(("cue", "sham", "no_cue") for _ in range(4))
        self.assertNotIn(same, {tuple(orders) for orders in relabelings})


class TestDescription(_SeriesTest):
    def test_means_by_arm_and_the_first_night(self):
        series = self.series()
        report = _analyse(series)
        arms = arms_by_night(series.schedule)
        lucid = {
            night_id: json.loads((series.reports_dir / f"{night_id}_dream_report.json").read_text(encoding="utf-8"))["lucid"]
            for night_id in arms
        }
        for arm in BLOCK_ARMS:
            nights = [night_id for night_id, other in arms.items() if other == arm]
            expected = sum(1.0 for night_id in nights if lucid[night_id]) / len(nights)
            self.assertAlmostEqual(report.descriptives["lucid"][arm]["mean"], expected)
            self.assertEqual(report.descriptives["lucid"][arm]["nights"], 4)
        first = report.nights[0]
        self.assertTrue(first["first_night"])
        self.assertEqual([night["first_night"] for night in report.nights].count(True), 1)
        first_arm = first["arm"]
        self.assertEqual(report.descriptives["lucid"][first_arm]["nights_without_first_night"], 3)
        self.assertIn("first_night_effect_possible", report.limitation_codes)
        self.assertEqual(
            next(item for item in report.limitations if item.code == "first_night_effect_possible").metadata["night_id"],
            "night-001",
        )
        self.assertEqual(set(report.descriptives), set(OUTCOMES))
        self.assertEqual({item["contrast"] for item in report.differences}, set(CONTRASTS))

    def test_lag1_autocorrelation_by_hand(self):
        rows = [
            {"position": position, "arm": arm, "values": {outcome: None for outcome in OUTCOMES}}
            for position, arm in enumerate(("cue", "sham", "no_cue", "cue", "sham", "no_cue"), start=1)
        ]
        for row, value in zip(rows, (1.0, 0.0, 1.0, 0.0, 1.0, None)):
            row["values"]["lucid"] = value
        result = nof1_analysis._autocorrelation(rows)["lucid"]
        # Arm means: cue 0.5, sham 0.5, no_cue 1.0; residuals 0.5, -0.5, 0, -0.5, 0.5 (mean 0).
        residuals = [0.5, -0.5, 0.0, -0.5, 0.5]
        numerator = sum(left * right for left, right in zip(residuals, residuals[1:]))
        denominator = sum(value * value for value in residuals)
        self.assertEqual(result["pairs"], 4)
        self.assertAlmostEqual(result["lag1"], numerator / denominator)
        self.assertIsNone(nof1_analysis._autocorrelation(rows[:2])["lucid"]["lag1"])

    def test_counts_plays_and_arousal_by_arm(self):
        report = _analyse(self.series())
        self.assertEqual(report.cue_counts["no_cue"]["plays"], 0)
        self.assertEqual(report.cue_counts["no_cue"]["virtual_plays"], 20)
        self.assertGreater(report.cue_counts["cue"]["plays"], 0)
        self.assertEqual(report.arousal["cue"]["nights_scored"], 4)
        self.assertIsNotNone(report.arousal["sham"]["mean_rate"]["any"])
        self.assertIsNone(report.arousal["no_cue"]["mean_rate"]["any"])

    def test_rater_codes_decide_incorporation(self):
        report = _analyse(self.series(raters=("rater-a",)))
        self.assertEqual({night["incorporation_source"] for night in report.nights}, {"raters"})
        plain = _analyse(self.series("plain"))
        self.assertEqual({night["incorporation_source"] for night in plain.nights}, {"self_report"})

    def test_the_report_holds_no_dream_text_and_says_synthetic(self):
        series = self.series()
        report = _analyse(series, primary_outcome="lucid")
        text = json.dumps(report.to_dict())
        self.assertNotIn("Synthetic dream text", text)
        self.assertTrue(report.to_dict()["synthetic"])
        self.assertEqual(report.to_dict()["note"], SYNTHETIC_NOTE)
        self.assertIn("synthetic_input", report.limitation_codes)
        markdown = report.to_markdown()
        self.assertIn("Synthetic input, not sleep", markdown)
        self.assertIn("SCRIBE", markdown)
        self.assertIn("4 items auto, 13 partial, 12 manual", markdown)
        primary = [item for item in report.differences if item["primary"]]
        self.assertEqual([(item["outcome"], item["contrast"]) for item in primary], [("lucid", "cue-no_cue")])
        self.assertNotIn("p-value =", markdown)

    def test_the_checklist_marks_every_item(self):
        counts = checklist_counts()
        self.assertEqual(counts, {"auto": 4, "partial": 13, "manual": 12})
        rows = nof1_analysis.checklist_rows()
        self.assertEqual([row["scribe"] for row in rows if row["scribe"]], [str(item) for item in range(1, 27)])
        self.assertTrue(all(row["status"] in ("auto", "partial", "manual") for row in rows))
        # CENT keeps CONSORT 2010's item numbers and adds 4c (ethics approval): each one
        # sits in exactly one row.
        consort_2010 = (
            "1a 1b 2a 2b 3a 3b 4a 4b 5 6a 6b 7a 7b 8a 8b 9 10 11a 11b 12a 12b 13a 13b 14a 14b "
            "15 16 17a 17b 18 19 20 21 22 23 24 25"
        ).split()
        named = [item for row in rows for item in row["cent"].split(", ") if item]
        self.assertEqual(sorted(named), sorted(consort_2010 + ["4c"]))
        self.assertEqual([row["cent"] for row in rows if not row["scribe"]], ["7a, 7b", "10", "18"])


class TestLimitations(_SeriesTest):
    def test_a_complete_series_has_only_the_standing_limitations(self):
        report = _analyse(self.series())
        self.assertEqual(
            report.limitation_codes,
            (
                "synthetic_input",
                "descriptive_only",
                "single_subject",
                "small_n",
                "sham_detectable",
                "carry_over_possible",
                "first_night_effect_possible",
                "inherits_detector_limits",
            ),
        )

    def test_unsealing_before_the_last_retest_is_unblinded_early(self):
        series = self.series(unsealed_at="2026-11-10T12:00:00+00:00")
        report = _analyse(series)
        item = next(item for item in report.limitations if item.code == "unblinded_early")
        forms = {(form["night_id"], form["form"]) for form in item.metadata["forms_after_unsealing"]}
        self.assertIn(("night-012", "retest_day10"), forms)
        self.assertIn(("night-012", "dream_report"), forms)
        self.assertNotIn(("night-001", "dream_report"), forms)
        self.assertTrue(item.metadata["nights_after_unsealing"])

    def test_missing_pieces_are_named(self):
        series = self.series(delayed_retests=False, arousal_scored=False, skip_nights=("night-005",))
        report = _analyse(series)
        codes = report.limitation_codes
        for code in ("missing_delayed_retest", "missing_arousal_scoring", "missing_night_data", "unbalanced_arms"):
            self.assertIn(code, codes)
        by_code = {item.code: item for item in report.limitations}
        self.assertEqual(by_code["missing_delayed_retest"].metadata["missing_waves"]["night-001"], ["day7", "day10"])
        playing = sorted(
            night for night, arm in arms_by_night(series.schedule).items() if arm in ("cue", "sham") and night != "night-005"
        )
        self.assertEqual(sorted(by_code["missing_arousal_scoring"].metadata["night_ids"]), playing)
        self.assertEqual(
            by_code["missing_night_data"].metadata["missing"]["night-005"], ["summary", "dream_report", "morning_retest"]
        )
        self.assertIsNone(report.nights[4]["values"]["lucid"])

    def test_no_sham_nights_and_an_arm_mismatch(self):
        series = self.series()
        sham_nights = [night for night, arm in arms_by_night(series.schedule).items() if arm == "sham"]
        for night_id in sham_nights:
            (series.reports_dir / f"{night_id}_dream_report.json").unlink()
            for suffix in ("", "_day7", "_day10"):
                (series.reports_dir / f"{night_id}_retest{suffix}.json").unlink()
            shutil.rmtree(series.recordings_dir / night_id)
        cue_night = next(night for night, arm in arms_by_night(series.schedule).items() if arm == "cue")
        summary_path = series.recordings_dir / cue_night / "pilot5_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["arm"] = "no_cue"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        report = _analyse(series)
        self.assertIn("no_sham_nights", report.limitation_codes)
        self.assertNotIn("sham_detectable", report.limitation_codes)
        mismatch = next(item for item in report.limitations if item.code == "arm_mismatch")
        self.assertEqual(mismatch.severity, "error")
        self.assertEqual(mismatch.metadata["nights"][cue_night], {"scheduled": "cue", "summary": "no_cue"})
        # The schedule's arm stays the night's arm in the analysis.
        self.assertEqual(next(n for n in report.nights if n["night_id"] == cue_night)["arm"], "cue")


class TestCommand(_SeriesTest):
    def test_analyze_nof1_writes_json_and_markdown(self):
        series = self.series(raters=("rater-a",))
        output = self.tmp / "out" / "nof1.json"
        markdown = self.tmp / "out" / "nof1.md"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli_main.main(
                [
                    "analyze-nof1",
                    str(series.schedule_path),
                    "--protocol-dir", str(series.protocol_dir),
                    "--reports-dir", str(series.reports_dir),
                    "--recordings-dir", str(series.recordings_dir),
                    "--primary-outcome", MORNING_DELTA,
                    "--output", str(output),
                    "--markdown-output", str(markdown),
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["report"], "nof1_analysis")
        self.assertEqual(payload["schedule"]["commitment_sha256"], series.schedule.commitment_sha256())
        # The command checked the seal; the library cannot know unless told.
        self.assertEqual(payload["schedule"]["seal"], "matches")
        self.assertEqual(_analyse(series).schedule["seal"], "not_checked")
        self.assertEqual(payload["settings"]["primary_contrast"], "cue-no_cue")
        self.assertEqual(payload["reporting_checklist_counts"], {"auto": 4, "partial": 13, "manual": 12})
        self.assertTrue(markdown.exists())
        self.assertIn("orders=72 (exact)", stdout.getvalue())
        self.assertIn(SYNTHETIC_NOTE, stdout.getvalue())

    def test_command_defaults_are_the_modules(self):
        args = cli_main.build_parser().parse_args(["analyze-nof1", "s.json", "--output", "o.json"])
        self.assertEqual((args.seed, args.permutations), (DEFAULT_SEED, DEFAULT_PERMUTATIONS))
        self.assertEqual(args.min_nights_per_arm, nof1_analysis.DEFAULT_MIN_NIGHTS_PER_ARM)

    def test_an_unknown_primary_outcome_exits_2(self):
        series = self.series()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli_main.main(
                ["analyze-nof1", str(series.schedule_path), "--primary-outcome", "nope", "--output", str(self.tmp / "o.json")]
            )
        self.assertEqual(code, 2)


class TestPurePython(unittest.TestCase):
    def test_the_modules_import_no_numeric_library(self):
        for name in ("nof1_analysis.py", "incorporation_coding.py"):
            tree = ast.parse((SRC / "reports" / name).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertFalse(imported & {"numpy", "scipy", "pandas", "statsmodels"}, name)
            allowed = {"__future__", "dataclasses", "datetime", "itertools", "json", "math", "pathlib", "random", "typing"}
            self.assertTrue(imported <= allowed | {"muse_tmr"}, (name, imported))

    def test_inputs_of_a_night_outside_the_schedule_raise(self):
        schedule = generate_night_schedule(1, 1, schedule_id="s", salt="0" * 32)
        from dataclasses import replace

        unsealed = replace(schedule, unsealed_at_utc="2026-12-01T00:00:00+00:00")
        with self.assertRaises(ValueError):
            build_nof1_analysis(unsealed, (Nof1NightInputs("night-999"),))
        report = build_nof1_analysis(unsealed, ())
        self.assertIn("missing_night_data", report.limitation_codes)
        self.assertEqual(block_orders(unsealed), (tuple(night.arm for night in unsealed.nights),))


if __name__ == "__main__":
    unittest.main()
