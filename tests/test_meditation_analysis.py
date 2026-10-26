import asyncio
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from muse_tmr.features.complexity_features import ComplexityConfig
from muse_tmr.reports.meditation_analysis import (
    MEDITATION_SUMMARY_SCHEMA_VERSION,
    MeditationAnalysisConfig,
    MeditationBlocks,
    aggregate_meditation_summaries,
    analyze_meditation_frames,
    assign_block,
    bootstrap_ci,
    build_meditation_plan,
    check_drift_cancelling,
    load_meditation_blocks,
    sign_flip_p_value,
    write_meditation_blocks,
)

from tests.meditation_synthetic import (
    FS,
    aiter_frames,
    band_noise,
    condition_signal,
    pink_noise,
    session_frames,
)

FAST = ComplexityConfig(lyapunov_enabled=False)


def contrast(summary, metric, group="all", variant="clean"):
    for item in summary["contrasts"]:
        if (item["metric"], item["group"], item["variant"]) == (metric, group, variant):
            return item
    raise KeyError(metric)


def residualized(summary, metric, group="all", variant="clean"):
    for item in summary["emg"]["residualized_contrasts"]:
        if (item["metric"], item["group"], item["variant"]) == (metric, group, variant):
            return item
    raise KeyError(metric)


class MeditationPlanTest(unittest.TestCase):
    def test_plan_is_deterministic_and_counterbalanced(self):
        first = build_meditation_plan(["focus", "open"], blocks=4, block_minutes=8, settle_seconds=60, seed=3)
        again = build_meditation_plan(["focus", "open"], blocks=4, block_minutes=8, settle_seconds=60, seed=3)
        self.assertEqual(first, again)
        self.assertIn(first.order, (("focus", "open", "open", "focus"), ("open", "focus", "focus", "open")))
        orders = {build_meditation_plan(["focus", "open"], seed=seed).order for seed in range(20)}
        self.assertEqual(len(orders), 2)
        self.assertEqual([block.start_s for block in first.blocks], [60.0, 540.0, 1020.0, 1500.0])
        self.assertTrue(all(block.depth is None and block.sensory_fading is None for block in first.blocks))

    def test_order_cancels_drift(self):
        # Block means of a metric that only drifts with time: A - B must be 0
        # for linear drift over 4 blocks and for quadratic drift over 8.
        for blocks, power in ((4, 1), (8, 1), (8, 2)):
            for seed in range(6):
                plan = build_meditation_plan(["focus", "open"], blocks=blocks, seed=seed)
                drift = {"focus": [], "open": []}
                for block in plan.blocks:
                    drift[block.condition].append(((block.start_s + block.end_s) / 2) ** power)
                difference = sum(drift["focus"]) / len(drift["focus"]) - sum(drift["open"]) / len(drift["open"])
                self.assertAlmostEqual(difference, 0.0, delta=1e-6 * max(drift["focus"]), msg=(blocks, power, plan.order))
        for blocks in (2, 6, 10):
            with self.assertRaises(ValueError):
                check_drift_cancelling(blocks)
        for blocks in (4, 8, 12):
            check_drift_cancelling(blocks)
        eight = build_meditation_plan(["a", "b"], blocks=8, seed=3).order
        self.assertIn("".join(eight), ("abbabaab", "baababba"))

    def test_json_round_trip(self):
        plan = build_meditation_plan(["focus", "open"], seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_meditation_blocks(plan, Path(tmp) / "blocks.json")
            payload = json.loads(path.read_text())
            self.assertEqual(payload["time_base"], "seconds_from_recording_start")
            self.assertEqual(payload["blocks"][0]["depth"], None)
            self.assertEqual(payload["blocks"][0]["drowsiness"], None)
            self.assertEqual(payload["ratings_scale"], "stability")
            # A file from before ratings_scale was written rated depth itself.
            del payload["ratings_scale"]
            self.assertEqual(MeditationBlocks.from_dict(payload).ratings_scale, "depth")
            with self.assertRaises(ValueError):
                MeditationBlocks.from_dict({**payload, "ratings_scale": "deep"})
            self.assertNotIn("series", payload)
            self.assertEqual(load_meditation_blocks(path), plan)
            tagged = replace(plan, series="20261010_190000_ab12cd")
            self.assertEqual(load_meditation_blocks(write_meditation_blocks(tagged, path)), tagged)

    def test_rejects_bad_plans(self):
        with self.assertRaises(ValueError):
            build_meditation_plan(["focus"], seed=1)
        with self.assertRaises(ValueError):
            build_meditation_plan(["focus", "focus"], seed=1)
        overlapping = {
            "schema_version": 1,
            "time_base": "seconds_from_recording_start",
            "blocks": [
                {"index": 0, "condition": "a", "start_s": 0, "end_s": 100},
                {"index": 1, "condition": "b", "start_s": 50, "end_s": 150},
            ],
        }
        with self.assertRaises(ValueError):
            MeditationBlocks.from_dict(overlapping)

    def test_epochs_mapped_to_blocks_with_trim_and_settle(self):
        plan = build_meditation_plan(["a", "b"], blocks=2, block_minutes=2, settle_seconds=60, seed=0)
        first, second = plan.blocks
        self.assertIsNone(assign_block(0, 10, plan, 30))  # settle
        self.assertIsNone(assign_block(60, 70, plan, 30))  # trimmed block start
        self.assertIsNone(assign_block(80, 90, plan, 30))
        self.assertEqual(assign_block(90, 100, plan, 30), first)
        self.assertIsNone(assign_block(175, 185, plan, 30))  # straddles the boundary
        self.assertEqual(assign_block(210, 220, plan, 30), second)
        self.assertIsNone(assign_block(300, 310, plan, 30))  # after the last block


class MeditationAnalysisEndToEndTest(unittest.TestCase):
    def analyze(self, plan, signal_for, config=None, seed=5):
        rng = np.random.default_rng(seed)
        frames = session_frames(plan, signal_for, rng)
        config = config or MeditationAnalysisConfig(emg_indicator="emg_power_55_95", complexity=FAST)
        return asyncio.run(analyze_meditation_frames(aiter_frames(frames), plan, config, recording="synthetic"))

    def test_emg_timeline_covers_the_whole_recording(self):
        from muse_tmr.reports.meditation_report import render_meditation_report

        plan = build_meditation_plan(["busy", "calm"], blocks=4, block_minutes=1, settle_seconds=30, seed=3)

        def signal_for(condition, seconds, rng):
            return condition_signal("busy" if condition == "settle" else condition, seconds, rng)

        import muse_tmr.reports.meditation_analysis as module

        full_runs = []
        original = module.extract_complexity_features

        def counting(*args, **kwargs):
            full_runs.append(1)
            return original(*args, **kwargs)

        module.extract_complexity_features = counting
        try:
            analysis = self.analyze(plan, signal_for)
        finally:
            module.extract_complexity_features = original
        timeline = analysis.summary["timeline"]
        # Settle, trimmed block starts and the padding at the end are all there,
        # with only the EMG computed for them.
        self.assertEqual(timeline[0]["start_s"], 0.0)
        self.assertEqual(timeline[0]["end_s"], 10.0)
        self.assertIsNone(timeline[0]["block_index"])
        self.assertIsNone(timeline[0]["artifact"])
        self.assertEqual(len(full_runs), analysis.summary["counts"]["epochs_in_blocks"])
        self.assertGreaterEqual(timeline[-1]["start_s"] + 10.0, plan.blocks[-1].end_s)
        self.assertEqual({row["block_index"] for row in timeline} - {None}, {block.index for block in plan.blocks})
        self.assertTrue(all(math.isfinite(row["emg_55_95_frontal_db"]) for row in timeline))
        self.assertEqual(analysis.summary["counts"]["epochs_in_blocks"], sum(row["block_index"] is not None for row in timeline))

        page = render_meditation_report(analysis.summary, analysis.blocks.to_dict("records"))
        self.assertIn("Muscle (EMG) over the session", page)
        self.assertIn("55–95 Hz per 10 s epoch", page)
        condition_a = analysis.summary["conditions"][0]
        self.assertEqual(page.count('class="band"'), sum(block.condition == condition_a for block in plan.blocks))

    def test_ratings_reach_the_summary_and_the_report(self):
        from muse_tmr.reports.meditation_report import render_meditation_report

        plan = build_meditation_plan(["busy", "calm"], blocks=4, block_minutes=1, settle_seconds=30, seed=3)
        plan = replace(
            plan,
            blocks=tuple(
                replace(block, depth=6.0, sensory_fading=3.0, drowsiness=float(block.index)) for block in plan.blocks
            ),
        )

        def signal_for(condition, seconds, rng):
            return condition_signal("busy" if condition == "settle" else condition, seconds, rng)

        analysis = self.analyze(plan, signal_for)
        for condition in ("busy", "calm"):
            indices = [block.index for block in plan.blocks if block.condition == condition]
            self.assertEqual(
                analysis.summary["ratings"][condition],
                {"depth": 6.0, "sensory_fading": 3.0, "drowsiness": sum(indices) / len(indices)},
            )
        rows = analysis.blocks.to_dict("records")
        self.assertEqual({row["drowsiness"] for row in rows}, {0.0, 1.0, 2.0, 3.0})
        page = render_meditation_report(analysis.summary, rows)
        # depth is shown as Stability, in the ratings by condition and in the blocks table.
        self.assertEqual(page.count("<th>Stability</th>"), 2)
        self.assertEqual(page.count("<th>Drowsiness</th>"), 2)
        self.assertNotIn("<th>Depth</th>", page)
        # An older blocks file rated depth itself, and its report says so.
        older = {**analysis.summary, "blocks_file": {**analysis.summary["blocks_file"], "ratings_scale": "depth"}}
        page = render_meditation_report(older, rows)
        self.assertEqual(page.count("<th>Depth</th>"), 2)
        self.assertNotIn("<th>Stability</th>", page)

    def test_timeline_traces_use_epoch_midpoints_and_break_at_gaps(self):
        from muse_tmr.reports.meditation_report import trace_segments

        rows = [
            {"start_s": 0.0, "end_s": 30.0, "v": 1.0},
            {"start_s": 30.0, "end_s": 60.0, "v": 2.0},
            {"start_s": 60.0, "end_s": 90.0, "v": float("nan")},
            {"start_s": 90.0, "end_s": 120.0, "v": 3.0},
            {"start_s": 120.0, "end_s": 150.0, "v": 4.0},
            {"start_s": 180.0, "end_s": 210.0, "v": 5.0},  # the epoch before it is missing
        ]
        self.assertEqual(
            trace_segments(rows, "v"),
            [[(15.0, 1.0), (45.0, 2.0)], [(105.0, 3.0), (135.0, 4.0)], [(195.0, 5.0)]],
        )
        self.assertEqual(trace_segments([{"start_s": 0.0, "v": 1.0}], "v"), [[(5.0, 1.0)]])

    def test_conditions_that_differ_by_construction_give_expected_signs(self):
        plan = build_meditation_plan(["busy", "calm"], blocks=4, block_minutes=2, settle_seconds=20, seed=2)

        def signal_for(condition, seconds, rng):
            return condition_signal("busy" if condition == "settle" else condition, seconds, rng)

        analysis = self.analyze(plan, signal_for)
        summary = analysis.summary

        self.assertEqual(summary["conditions"], ["busy", "calm"])
        self.assertEqual(summary["counts"]["blocks_with_epochs"], 4)
        lzc = contrast(summary, "lzc")
        self.assertTrue(lzc["primary"])
        primaries = [item for item in summary["contrasts"] if item["primary"]]
        self.assertEqual(len(primaries), 1)
        self.assertFalse(contrast(summary, "lzc", variant="all")["primary"])
        self.assertFalse(any(item["primary"] for item in summary["emg"]["residualized_contrasts"]))
        self.assertGreater(lzc["difference"], 0.2)
        self.assertGreater(contrast(summary, "permutation_entropy")["difference"], 0)
        self.assertLess(contrast(summary, "aperiodic_exponent_2_40")["difference"], -0.5)
        self.assertLess(contrast(summary, "band_power_alpha")["difference"], 0)
        self.assertEqual(contrast(summary, "sample_entropy")["label"], "exploratory")
        self.assertTrue(math.isfinite(contrast(summary, "dfa_alpha")["a_mean"]))
        self.assertEqual(set(analysis.blocks["variant"]), {"all", "clean"})
        self.assertEqual(len(analysis.blocks), 8)
        # 2 min blocks minus 30 s trim = 9 epochs of 10 s each.
        self.assertEqual(len(analysis.epochs), 4 * 9)

        with tempfile.TemporaryDirectory() as tmp:
            paths = analysis.write(Path(tmp))
            written = json.loads(paths["summary"].read_text())
            self.assertIn("limitations", written)
            self.assertEqual(written["primary_metric"]["metric"], "lzc")
            self.assertIn("lyapunov_enabled", written["config"]["complexity"])
            self.assertTrue(paths["epochs"].exists() and paths["blocks"].exists())
            page = paths["report"].read_text(encoding="utf-8")
            self.assertIn("Primary result", page)
            self.assertIn("Muscle (EMG) check", page)
            self.assertIn("<svg", page)
            self.assertNotIn("http://", page)
            exploratory = page.split("Metrics from the paper", 1)[1].split("</section>", 1)[0]
            self.assertNotIn("Lempel-Ziv", exploratory)
            self.assertNotIn("https://", page)

    def test_emg_only_difference_is_flagged_and_residualized_contrast_shrinks(self):
        # B is A plus broadband EMG-like 30-95 Hz bursts. Pure 60-90 Hz noise
        # cannot move the 0.5-40 Hz complexity metrics, real EMG is broadband.
        plan = build_meditation_plan(["relaxed", "tense"], blocks=4, block_minutes=2, settle_seconds=20, seed=4)

        def signal_for(condition, seconds, rng):
            emg = 12.0 if condition == "tense" else 0.0
            return condition_signal("busy", seconds, rng, emg_uv=emg)

        summary = self.analyze(plan, signal_for).summary
        emg = summary["emg"]
        self.assertEqual(emg["indicator"], "emg_power_55_95")
        self.assertTrue(emg["emg_confounded"])
        self.assertLess(emg["condition_difference"]["clean"]["difference_log10"], -0.1)

        raw = residualized(summary, "lzc")
        self.assertLess(raw["raw_difference"], 0)  # tense has higher LZC
        self.assertLess(abs(raw["residualized_difference"]), 0.5 * abs(raw["raw_difference"]))
        rho = next(
            item["spearman_rho"]
            for item in emg["correlations"]
            if (item["metric"], item["group"], item["variant"]) == ("lzc", "all", "clean")
        )
        self.assertGreater(rho, 0.5)

    def test_no_emg_difference_is_not_flagged(self):
        plan = build_meditation_plan(["a", "b"], blocks=4, block_minutes=2, settle_seconds=20, seed=6)

        def signal_for(condition, seconds, rng):
            signal = condition_signal("busy", seconds, rng)
            if condition == "b":
                t = np.arange(int(seconds * FS)) / FS
                signal = {name: values + 15.0 * np.sin(2 * np.pi * 10.0 * t) for name, values in signal.items()}
            return signal

        summary = self.analyze(plan, signal_for).summary
        self.assertFalse(summary["emg"]["emg_confounded"])
        self.assertLess(contrast(summary, "band_power_alpha")["difference"], -0.3)

    def test_auto_indicator_falls_back_when_high_band_is_at_the_floor(self):
        plan = build_meditation_plan(["a", "b"], blocks=2, block_minutes=1.5, settle_seconds=0, seed=1)

        def white(condition, seconds, rng):
            n = int(seconds * FS)
            return {name: 10.0 * rng.standard_normal(n) for name in ("TP9", "AF7", "AF8", "TP10")}

        config = MeditationAnalysisConfig(complexity=FAST)
        summary = self.analyze(plan, white, config=config).summary
        self.assertEqual(summary["emg"]["indicator"], "emg_power_30_45")
        self.assertTrue(any("30-45 Hz" in item for item in summary["limitations"]))

        def low_passed(condition, seconds, rng):
            n = int(seconds * FS)
            return {name: 10.0 * band_noise(n, (1.0, 100.0), rng) for name in ("TP9", "AF7", "AF8", "TP10")}

        summary = self.analyze(plan, low_passed, config=config).summary
        self.assertEqual(summary["emg"]["indicator"], "emg_power_55_95")

    def test_artifact_epochs_are_flagged_and_dropped_only_from_clean_variant(self):
        plan = build_meditation_plan(["a", "b"], blocks=2, block_minutes=1.5, settle_seconds=0, seed=1)

        def with_clipping(condition, seconds, rng):
            n = int(seconds * FS)
            signal = {name: 10.0 * pink_noise(n, 1.0, rng) for name in ("TP9", "AF7", "AF8", "TP10")}
            if condition == "a":
                signal["TP9"][int(42 * FS) : int(44 * FS)] = 2000.0
            return signal

        analysis = self.analyze(plan, with_clipping)
        flagged = analysis.epochs[analysis.epochs["is_artifact"]]
        self.assertGreaterEqual(len(flagged), 1)
        self.assertTrue(flagged["artifact_flags"].str.contains("TP9").all())
        rows = analysis.blocks
        block_a = rows[rows["condition"] == "a"]
        self.assertLess(
            int(block_a.loc[block_a["variant"] == "clean", "epochs"].iloc[0]),
            int(block_a.loc[block_a["variant"] == "all", "epochs"].iloc[0]),
        )


class MeditationWithPolarTest(unittest.TestCase):
    def test_polar_breathing_lands_in_blocks_contrasts_and_confound_flag(self):
        from muse_tmr.data.polar_session import load_polar_session
        from tests.polar_synthetic import write_raw_session

        plan = build_meditation_plan(["slow", "normal"], blocks=4, block_minutes=2, settle_seconds=20, seed=3)
        wall0 = 1_790_000_000.0
        rng = np.random.default_rng(9)

        def signal_for(condition, seconds, rng):
            return condition_signal("busy", seconds, rng)

        def breathing(t):
            for block in plan.blocks:
                if block.start_s <= t < block.end_s:
                    return 6.0 if block.condition == "slow" else 15.0
            return 12.0

        frames = session_frames(plan, signal_for, rng)
        offset = wall0 - frames[0].timestamp
        frames = [
            type(frame)(timestamp=frame.timestamp + offset, eeg=type(frame.eeg)(timestamp=frame.eeg.timestamp + offset, channels_uv=frame.eeg.channels_uv), source=frame.source)
            for frame in frames
        ]
        seconds = plan.blocks[-1].end_s + 10.0
        with tempfile.TemporaryDirectory() as tmp:
            write_raw_session(tmp, seconds, rng, wall0=wall0, acc_breaths_per_min=breathing)
            polar = load_polar_session(Path(tmp))
            config = MeditationAnalysisConfig(emg_indicator="emg_power_55_95", complexity=FAST)
            analysis = asyncio.run(analyze_meditation_frames(aiter_frames(frames), plan, config, polar=polar))

        summary = analysis.summary
        cardio = summary["cardio"]
        self.assertTrue(cardio["available"])
        self.assertTrue(cardio["breathing_confounded"])
        self.assertAlmostEqual(cardio["breathing_difference_bpm"], -9.0, delta=1.0)
        # The synthetic ECG keeps breathing at 12/min in both conditions, so EDR
        # sees no difference while both ACC estimates see -9: reported as disagreement.
        methods = cardio["breathing_methods"]
        self.assertEqual(set(methods), {"acc_spectral", "acc_breath", "edr"})
        self.assertAlmostEqual(methods["acc_spectral"]["difference"], -9.0, delta=1.0)
        self.assertAlmostEqual(methods["acc_breath"]["difference"], -9.0, delta=1.0)
        self.assertAlmostEqual(methods["edr"]["difference"], 0.0, delta=1.0)
        self.assertTrue(cardio["breathing_methods_disagree"])
        self.assertEqual(summary["schema_version"], 5)
        self.assertEqual(cardio["breathing_difference_method"], "median_of_methods")
        self.assertGreater(cardio["breathing_methods_spread_bpm"], 7.0)
        self.assertTrue(any("estimates disagree" in item for item in summary["limitations"]))
        self.assertEqual(set(summary["emg"]["group_difference_db"]["clean"]), {"all", "frontal", "temporal"})
        self.assertEqual(cardio["hf_band_invalid_blocks"], 2)
        rows = analysis.blocks[analysis.blocks["variant"] == "all"].set_index("block_index")
        for block in plan.blocks:
            expected = 6.0 if block.condition == "slow" else 15.0
            self.assertAlmostEqual(rows.loc[block.index, "cardio_resp_rate_bpm"], expected, delta=0.7)
            self.assertAlmostEqual(rows.loc[block.index, "cardio_mean_hr_bpm"], 60.0, delta=3.0)
        chest = [item for item in summary["contrasts"] if item["group"] == "chest"]
        self.assertTrue(chest and not any(item["primary"] for item in chest))
        self.assertTrue(any("breathing rate" in item for item in summary["limitations"]))

        self.assertEqual(cardio["breathing_unreliable_blocks"], [])

        # The chest contrasts flow into the cross-session aggregate like any other,
        # and a session recorded without Polar keeps its slot as a gap.
        without_polar = {**summary, "contrasts": [item for item in summary["contrasts"] if item["group"] != "chest"]}
        aggregate = aggregate_meditation_summaries([summary, without_polar, summary], labels=["s1", "s2", "s3"])
        breathing = next(row for row in aggregate["rows"] if row["metric"] == "cardio_resp_rate_bpm")
        self.assertEqual(aggregate["sessions"], ["s1", "s2", "s3"])
        self.assertEqual(len(breathing["differences"]), 3)
        self.assertTrue(math.isnan(breathing["differences"][1]))
        self.assertEqual(breathing["n_sessions"], 2)

    def test_block_with_movement_is_left_out_of_breathing_contrasts(self):
        from muse_tmr.data.polar_session import load_polar_session
        from tests.polar_synthetic import write_raw_session

        plan = build_meditation_plan(["a", "b"], blocks=2, block_minutes=2, settle_seconds=20, seed=0)
        wall0 = 1_790_000_000.0
        rng = np.random.default_rng(10)
        frames = session_frames(plan, lambda c, s, r: condition_signal("busy", s, r), rng)
        offset = wall0 - frames[0].timestamp
        frames = [
            type(frame)(timestamp=frame.timestamp + offset, eeg=type(frame.eeg)(timestamp=frame.eeg.timestamp + offset, channels_uv=frame.eeg.channels_uv), source=frame.source)
            for frame in frames
        ]
        with tempfile.TemporaryDirectory() as tmp:
            write_raw_session(tmp, plan.blocks[-1].end_s + 10.0, rng, wall0=wall0, acc_breaths_per_min=10.0)
            polar = load_polar_session(Path(tmp))
            moving = plan.blocks[1]
            times = polar.acc["time"].to_numpy() - wall0
            for start in np.arange(moving.start_s + 35, moving.end_s - 10, 20.0):
                window = (times >= start) & (times < start + 8)
                polar.acc.loc[window, "x"] += 350.0 * np.sin(np.pi * (times[window] - start) / 8.0)
            config = MeditationAnalysisConfig(emg_indicator="emg_power_55_95", complexity=FAST)
            summary = asyncio.run(analyze_meditation_frames(aiter_frames(frames), plan, config, polar=polar)).summary

        cardio = summary["cardio"]
        self.assertEqual(cardio["breathing_unreliable_blocks"], [moving.index])
        self.assertFalse(cardio["breathing_confounded"])
        breathing = next(item for item in summary["contrasts"] if item["metric"] == "cardio_resp_rate_bpm")
        self.assertEqual(breathing["n_blocks_a"] + breathing["n_blocks_b"], 1)
        self.assertTrue(any("not trusted" in item for item in summary["limitations"]))

    def test_without_polar_the_section_says_so(self):
        plan = build_meditation_plan(["a", "b"], blocks=2, block_minutes=1.5, settle_seconds=0, seed=1)
        rng = np.random.default_rng(2)
        frames = session_frames(plan, lambda c, s, r: condition_signal("busy", s, r), rng)
        config = MeditationAnalysisConfig(emg_indicator="emg_power_55_95", complexity=FAST)
        summary = asyncio.run(analyze_meditation_frames(aiter_frames(frames), plan, config)).summary
        self.assertEqual(summary["cardio"], {"available": False})
        self.assertFalse(any(item["group"] == "chest" for item in summary["contrasts"]))


class MeditationReportTest(unittest.TestCase):
    def test_condition_names_are_escaped_and_cardio_shown(self):
        from muse_tmr.reports.meditation_report import render_meditation_report

        summary = {
            "conditions": ["<b>focus</b>", "open & wide"],
            "recording": "/data/recordings/session/x",
            "counts": {"blocks": 2},
            "contrasts": [
                {"metric": "lzc", "group": "all", "variant": "clean", "a_mean": 0.5, "b_mean": 0.4, "difference": 0.1}
            ],
            "emg": {},
            "cardio": {
                "available": True,
                "breathing_difference_bpm": None,
                "breathing_unreliable_blocks": [1],
                "breathing_methods": {
                    "acc_spectral": {"difference": 1.1},
                    "acc_breath": {"difference": -0.4},
                    "edr": {"difference": 3.2},
                },
                "breathing_methods_spread_bpm": 3.6,
                "breathing_methods_disagree": True,
            },
            "limitations": ["n = 1"],
        }
        rows = [
            {"variant": variant, "block_index": index, "condition": condition, "start_s": 60, "end_s": 540,
             "epochs": 40, "lzc_all": 0.5 - index * 0.1, "cardio_resp_rate_bpm": 6.0,
             "cardio_resp_rate_breath_bpm": 5.5, "cardio_edr_rate_bpm": 3.8, "cardio_resp_reliable": float(index == 0)}
            for index, condition in enumerate(summary["conditions"])
            for variant in ("all", "clean")
        ]
        page = render_meditation_report(summary, rows)
        self.assertNotIn("<b>focus</b>", page)
        self.assertIn("&lt;b&gt;focus&lt;/b&gt;", page)
        self.assertIn("open &amp; wide", page)
        self.assertIn("Breathing was not compared", page)
        self.assertIn("block(s) 1", page)
        self.assertIn("ECG-derived +3.2", page)
        self.assertIn("disagree by 3.6", page)
        self.assertIn("<td>2.2</td>", page)  # per-block spread 6.0 - 3.8

    def test_emg_by_channel_group(self):
        from muse_tmr.reports.meditation_report import render_meditation_report

        summary = {
            "conditions": ["relaxed", "jaw"],
            "counts": {},
            "contrasts": [],
            "emg": {
                "indicator": "emg_power_55_95",
                "condition_difference": {"clean": {"ratio": 0.2}},
                "group_difference_db": {"clean": {"all": -7.0, "frontal": -1.5, "temporal": -12.25}},
            },
        }
        page = render_meditation_report(summary, [])
        self.assertIn("AF7/AF8 (forehead) -1.5 dB", page)
        self.assertIn("TP9/TP10 (jaw) -12.2 dB", page)


class AggregateMeditationTest(unittest.TestCase):
    def summary(self, conditions, lzc_difference, recording):
        return {
            "schema_version": MEDITATION_SUMMARY_SCHEMA_VERSION,
            "recording": recording,
            "conditions": list(conditions),
            "contrasts": [
                {"metric": "lzc", "group": "all", "variant": "clean", "difference": lzc_difference},
                {"metric": "sample_entropy", "group": "all", "variant": "clean", "difference": None},
            ],
            "emg": {
                "emg_confounded": recording == "s2",
                "residualized_contrasts": [
                    {"metric": "lzc", "group": "all", "variant": "clean", "residualized_difference": lzc_difference / 2}
                ],
            },
        }

    def row(self, result, metric="lzc", kind="raw"):
        return next(item for item in result["rows"] if item["metric"] == metric and item["kind"] == kind)

    def test_few_sessions_give_descriptives_and_warning_only(self):
        summaries = [
            self.summary(("focus", "open"), 0.10, "s1"),
            self.summary(("open", "focus"), -0.20, "s2"),  # reversed labels: sign flipped
            self.summary(("focus", "open"), 0.30, "s3"),
        ]
        result = aggregate_meditation_summaries(summaries)

        lzc = self.row(result)
        self.assertEqual(lzc["differences"], [0.10, 0.20, 0.30])
        self.assertAlmostEqual(lzc["mean_difference"], 0.2)
        self.assertTrue(lzc["primary"])
        self.assertNotIn("p_sign_flip", lzc)
        self.assertNotIn("ci95", lzc)
        self.assertTrue(result["warnings"])
        self.assertEqual(result["emg_confounded_sessions"], ["s2"])
        residual = self.row(result, kind="emg_residualized")
        self.assertEqual(residual["differences"], [0.05, 0.10, 0.15])
        self.assertFalse(residual["primary"])
        self.assertEqual(self.row(result, metric="sample_entropy")["n_sessions"], 0)

    def test_eight_sessions_get_sign_flip_and_bootstrap(self):
        summaries = [self.summary(("focus", "open"), 0.1 + 0.01 * index, f"s{index}") for index in range(8)]
        result = aggregate_meditation_summaries(summaries)
        lzc = self.row(result)
        self.assertFalse(result["warnings"])
        # All 8 positive: only the all-plus and all-minus sign patterns are as extreme.
        self.assertAlmostEqual(lzc["p_sign_flip"], 2 / 256)
        low, high = lzc["ci95"]
        self.assertLess(low, lzc["mean_difference"])
        self.assertGreater(high, lzc["mean_difference"])

    def test_summaries_with_the_old_emg_indicator_are_rejected(self):
        old = {**self.summary(("focus", "open"), 0.2, "s2"), "schema_version": 2}
        unversioned = self.summary(("focus", "open"), 0.3, "s3")
        del unversioned["schema_version"]
        with self.assertRaisesRegex(ValueError, "rebuild them with analyze-meditation: s2, s3"):
            aggregate_meditation_summaries([self.summary(("focus", "open"), 0.1, "s1"), old, unversioned])
        # Every summary is checked, not only the labelled ones.
        with self.assertRaisesRegex(ValueError, "1 labels for 2 summaries"):
            aggregate_meditation_summaries([self.summary(("focus", "open"), 0.1, "s1"), old], labels=["s1"])

    def test_sessions_with_different_feature_settings_are_rejected(self):
        default = {**self.summary(("focus", "open"), 0.1, "s1"), "config": {"complexity": ComplexityConfig().to_dict()}}
        unbridged = {
            **self.summary(("focus", "open"), 0.2, "s2"),
            "config": {"complexity": ComplexityConfig(emg_exclude_hz=()).to_dict()},
        }
        other_floor = {
            **self.summary(("focus", "open"), 0.3, "s3"),
            "config": {"complexity": ComplexityConfig(emg_floor_band_hz=(100.0, 120.0)).to_dict()},
        }
        with self.assertRaisesRegex(ValueError, "differ from s1 in: s2, s3"):
            aggregate_meditation_summaries([default, unbridged, other_floor])
        aggregate_meditation_summaries([default, json.loads(json.dumps(default))])  # tuples vs JSON lists
        # Without Lyapunov that metric is just missing; the rest still pools.
        no_lyapunov = {
            **self.summary(("focus", "open"), 0.4, "s4"),
            "config": {"complexity": ComplexityConfig(lyapunov_enabled=False).to_dict()},
        }
        aggregate_meditation_summaries([default, no_lyapunov])
        line_kept = {
            **self.summary(("focus", "open"), 0.5, "s5"),
            "config": {"complexity": ComplexityConfig(line_hz=()).to_dict()},
        }
        with self.assertRaisesRegex(ValueError, "differ from s1 in: s5"):
            aggregate_meditation_summaries([default, line_kept])

    def test_sessions_from_different_series_are_rejected(self):
        def tagged(summary, series):
            return {**summary, "blocks_file": {"series": series}}

        first = tagged(self.summary(("focus", "open"), 0.1, "s1"), "a")
        other = tagged(self.summary(("focus", "open"), 0.2, "s2"), "b")
        standalone = self.summary(("focus", "open"), 0.3, "s3")
        with self.assertRaisesRegex(ValueError, "than s1: s2, s3"):
            aggregate_meditation_summaries([first, other, standalone])
        result = aggregate_meditation_summaries([first, tagged(self.summary(("focus", "open"), 0.2, "s2"), "a")])
        self.assertEqual(result["series"], "a")
        self.assertIsNone(aggregate_meditation_summaries([standalone])["series"])

    def test_residualized_contrasts_pool_only_within_one_emg_indicator(self):
        summaries = [self.summary(("focus", "open"), 0.1 * (index + 1), f"s{index + 1}") for index in range(3)]
        for summary, indicator in zip(summaries, ("emg_power_55_95", "emg_power_30_45", "emg_power_55_95")):
            summary["emg"]["indicator"] = indicator
        result = aggregate_meditation_summaries(summaries)
        self.assertEqual(self.row(result)["n_sessions"], 3)
        residual = {row["emg_indicator"]: row for row in result["rows"] if row["kind"] == "emg_residualized"}
        self.assertEqual(residual["emg_power_55_95"]["n_sessions"], 2)
        self.assertEqual(residual["emg_power_30_45"]["n_sessions"], 1)
        self.assertTrue(math.isnan(residual["emg_power_55_95"]["differences"][1]))

    def test_mismatched_conditions_are_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_meditation_summaries(
                [self.summary(("focus", "open"), 0.1, "s1"), self.summary(("focus", "body"), 0.1, "s2")]
            )

    def test_sign_flip_and_bootstrap_helpers(self):
        self.assertAlmostEqual(sign_flip_p_value(np.array([1.0, -1.0, 1.0, -1.0])), 1.0)
        low, high = bootstrap_ci(np.array([1.0, 1.0, 1.0]))
        self.assertEqual((low, high), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
