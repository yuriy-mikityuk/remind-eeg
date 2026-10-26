import asyncio
import contextlib
import io
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from muse_tmr.cli.main import _calibration_speaker, build_parser
from muse_tmr.protocol.calibration import (
    END_SAY,
    PROTOCOL,
    PROTOCOL_SECONDS,
    Pace,
    SaySpeaker,
    Segment,
    SilentSpeaker,
    calibration_protocol,
    first_frame_time,
    pair_blocks,
    read_state,
    run_guide,
    schedule,
    segments_from_cues,
)
from muse_tmr.reports.calibration_report import (
    _h10_unclip,
    build_calibration_report_from_frames,
    inhale_exhale_check,
    render_calibration_report,
)
from muse_tmr.reports.meditation_analysis import load_meditation_blocks

from tests.meditation_synthetic import CHANNELS, aiter_frames, band_noise, condition_signal, frames_from_segments

ORIGIN = 1_790_000_000.0


class FakeClock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.001)


def recording_with_first_frame(tmp, timestamp=ORIGIN):
    folder = Path(tmp)
    (folder / "decoded_frames.jsonl").write_text(json.dumps({"timestamp": timestamp, "eeg": None}) + "\n")
    return folder


class ScheduleTest(unittest.TestCase):
    def test_protocol_matches_the_plan(self):
        self.assertEqual(PROTOCOL_SECONDS, 23 * 60)
        names = [segment.name for segment in PROTOCOL]
        self.assertEqual(len(set(names)), len(names))
        for before, after in zip(PROTOCOL, PROTOCOL[1:]):
            self.assertEqual(before.end_s, after.start_s)

    def test_paced_cycles_stay_inside_their_step(self):
        cues = schedule()
        self.assertEqual([cue.planned_s for cue in cues], sorted(cue.planned_s for cue in cues))
        by_segment = {segment.name: segment for segment in PROTOCOL}
        for name, cycles in (("clench", 18), ("breath_6", 17), ("breath_12", 35)):
            segment = by_segment[name]
            starts = [cue.planned_s for cue in cues if cue.segment == name and cue.cycle_start]
            self.assertEqual(len(starts), cycles, name)
            self.assertEqual(starts[0], segment.start_s + segment.pace.lead_s)
            self.assertLessEqual(starts[-1] + segment.pace.period_s, segment.end_s)
        exhale = [cue.planned_s for cue in cues if cue.segment == "breath_6" and cue.text == "выдох"]
        self.assertEqual(exhale[0] - 786.0, 4.0)
        self.assertEqual(cues[-1].kind, "end")

    def test_english_says_the_same_plan(self):
        english = calibration_protocol(language="en")
        self.assertEqual(
            [(s.name, s.label, s.start_s, s.end_s, s.condition) for s in english],
            [(s.name, s.label, s.start_s, s.end_s, s.condition) for s in PROTOCOL],
        )
        cues = schedule(english, "en")
        self.assertEqual(
            [(cue.planned_s, cue.kind, cue.segment, cue.cycle_start) for cue in cues],
            [(cue.planned_s, cue.kind, cue.segment, cue.cycle_start) for cue in schedule()],
        )
        self.assertEqual([cue.text for cue in cues if re.search("[а-яё]", cue.text, re.IGNORECASE)], [])
        self.assertEqual({cue.text for cue in cues if cue.segment == "breath_6" and cue.kind == "beat"}, {"in", "out"})
        self.assertEqual(cues[-1].text, END_SAY["en"])


class GuideTest(unittest.TestCase):
    def test_full_run_logs_cues_and_writes_the_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = recording_with_first_frame(tmp)
            clock = FakeClock(ORIGIN - 3.0)
            speaker = SilentSpeaker()
            result = run_guide(folder, speaker, clock=clock, sleep=clock.sleep)

            self.assertEqual(result.stop_reason, "completed")
            self.assertEqual(len(speaker.spoken), len(schedule()))
            cues_path = folder / "calibration" / "cues.jsonl"
            cues = [json.loads(line) for line in cues_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(cues), len(schedule()))
            self.assertTrue(all(abs(cue["elapsed_s"] - cue["planned_s"]) < 0.06 for cue in cues))
            self.assertTrue(all(item["completed"] for item in result.segments))
            self.assertEqual(sorted(result.blocks_files), ["breathing", "clench", "forehead", "jaw"])

            jaw = load_meditation_blocks(Path(result.blocks_files["jaw"]))
            self.assertEqual(jaw.conditions, ("jaw", "relaxed"))
            self.assertEqual([round(block.start_s) for block in jaw.blocks], [60, 180, 300])
            self.assertEqual([round(block.end_s) for block in jaw.blocks], [180, 300, 420])
            clench = load_meditation_blocks(Path(result.blocks_files["clench"]))
            # Relaxed windows as long as the 1 min clench step, on both sides.
            self.assertEqual([(round(b.start_s), round(b.end_s)) for b in clench.blocks], [(600, 660), (660, 720), (720, 780)])
            breathing = load_meditation_blocks(Path(result.blocks_files["breathing"]))
            self.assertEqual(breathing.conditions, ("breath_6", "breath_12"))
            self.assertEqual(read_state(folder)["state"], "done")
            segments = json.loads((folder / "calibration" / "segments.json").read_text(encoding="utf-8"))
            paced = next(item for item in segments["segments"] if item["name"] == "breath_6")
            self.assertEqual(len(paced["cycle_starts_s"]), 17)
            self.assertEqual(paced["known_rate_bpm"], 6.0)

    def test_english_run_speaks_english(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = recording_with_first_frame(tmp)
            clock = FakeClock(ORIGIN)
            speaker = SilentSpeaker()
            english = calibration_protocol(language="en")
            result = run_guide(folder, speaker, protocol=english, language="en", clock=clock, sleep=clock.sleep)
            self.assertEqual(result.stop_reason, "completed")
            self.assertEqual(speaker.spoken[0], english[0].say)
            self.assertEqual(speaker.spoken[-1], END_SAY["en"])
            self.assertEqual(sorted(result.blocks_files), ["breathing", "clench", "forehead", "jaw"])

    def test_stops_when_the_recording_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = recording_with_first_frame(tmp)
            clock = FakeClock(ORIGIN)

            class StopAtForehead(SilentSpeaker):
                def speak(self, text):
                    super().speak(text)
                    if text.startswith("Чуть напряги лоб"):
                        (folder / "summary.json").write_text("{}")

            result = run_guide(folder, StopAtForehead(), clock=clock, sleep=clock.sleep)
            self.assertEqual(result.stop_reason, "recording_ended")
            self.assertEqual(sorted(result.blocks_files), ["jaw"])
            forehead = next(item for item in result.segments if item["name"] == "forehead")
            self.assertFalse(forehead["completed"])
            self.assertEqual(read_state(folder)["state"], "stopped")

    def test_interrupt_keeps_what_was_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = recording_with_first_frame(tmp)
            clock = FakeClock(ORIGIN)

            class CtrlC(SilentSpeaker):
                def speak(self, text):
                    if text.startswith("Сейчас сжимай"):
                        raise KeyboardInterrupt
                    super().speak(text)

            result = run_guide(folder, CtrlC(), clock=clock, sleep=clock.sleep)
            self.assertEqual(result.stop_reason, "interrupted")
            # relaxed_3 was cut at the stop, so the forehead pair has no "after" window.
            self.assertEqual(sorted(result.blocks_files), ["jaw"])
            relaxed_3 = next(item for item in result.segments if item["name"] == "relaxed_3")
            self.assertFalse(relaxed_3["completed"])
            self.assertAlmostEqual(relaxed_3["end_s"], 660.0, delta=0.1)

    def test_late_start_announces_only_the_current_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = recording_with_first_frame(tmp)
            clock = FakeClock(ORIGIN + 200.0)  # in the middle of the jaw step
            speaker = SilentSpeaker()
            run_guide(folder, speaker, clock=clock, sleep=clock.sleep)
            self.assertTrue(speaker.spoken[0].startswith("Чуть напряги челюсть"))
            self.assertFalse(any(text.startswith("Закрой глаза") for text in speaker.spoken))

    def test_waits_for_the_first_frame_and_gives_up_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            clock = FakeClock(ORIGIN)
            result = run_guide(folder, SilentSpeaker(), clock=clock, sleep=clock.sleep, recorder_alive=lambda: False)
            self.assertEqual(result.stop_reason, "recording_ended")
            self.assertFalse((folder / "calibration" / "segments.json").exists())
            result = run_guide(folder, SilentSpeaker(), clock=clock, sleep=clock.sleep, first_frame_timeout_s=30)
            self.assertEqual(result.stop_reason, "no_first_frame")

    def test_first_frame_needs_a_complete_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decoded_frames.jsonl"
            path.write_text('{"timestamp": 12.5')
            self.assertIsNone(first_frame_time(Path(tmp)))
            path.write_text('{"timestamp": 12.5}\n{"timest')
            self.assertEqual(first_frame_time(Path(tmp)), 12.5)


class SpeakerTest(unittest.TestCase):
    def test_say_command_and_no_overlap(self):
        calls = []

        class Proc:
            def __init__(self):
                self.waited = False

            def poll(self):
                return 0 if self.waited else None

            def wait(self, timeout=None):
                self.waited = True

        def popen(argv, **kwargs):
            calls.append(argv)
            return Proc()

        speaker = SaySpeaker(voice="Milena", rate=170, popen=popen)
        speaker.speak("вдох")
        first = speaker._process
        speaker.speak("выдох")
        self.assertTrue(first.waited)
        self.assertEqual(calls[0], ["say", "-v", "Milena", "-r", "170", "вдох"])
        self.assertEqual(SaySpeaker(voice=None, popen=popen).voice, None)

    def test_voice_follows_the_language(self):
        def speaker(*argv):
            return _calibration_speaker(build_parser().parse_args(["calibration-guide", "/tmp/x", *argv]))

        with patch("muse_tmr.protocol.calibration.available_voices", return_value=["Milena", "Samantha", "Yuri"]):
            self.assertEqual(speaker().voice, "Samantha")
            self.assertEqual(speaker("--language", "ru").voice, "Milena")
            self.assertEqual(speaker("--language", "ru", "--voice", "Yuri").voice, "Yuri")
        out = io.StringIO()
        with patch("muse_tmr.protocol.calibration.available_voices", return_value=["Samantha"]), contextlib.redirect_stdout(out):
            self.assertIsNone(speaker("--language", "ru").voice)
        self.assertIn("Russian voices here: none installed", out.getvalue())


def asymmetric_breathing_session(period_s, inhale_s, seconds=200.0, first_cue_s=5.0, rng=None, exhale_s=None):
    """Chest ACC rising for inhale_s and falling for exhale_s (the rest of the cycle
    by default), resting after that; HR in phase."""
    rng = rng or np.random.default_rng(0)
    exhale_s = exhale_s or period_s - inhale_s

    def wave(t):
        phase = np.mod(t - first_cue_s, period_s)
        falling = np.clip(1.0 - (phase - inhale_s) / exhale_s, 0.0, 1.0)
        return np.where(phase < inhale_s, phase / inhale_s, falling)

    t = np.arange(0.0, seconds, 1 / 50.0)
    direction = np.array([0.3, -0.8, 0.5]) / np.linalg.norm([0.3, -0.8, 0.5])
    # Negative sign: the raw component points "down" on the inhale, the check must flip it.
    xyz = np.array([20.0, -40.0, 990.0]) - 8.0 * np.outer(wave(t), direction) + rng.normal(0, 0.5, (t.size, 3))
    acc = pd.DataFrame({"time": ORIGIN + t, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    beats = [0.0]
    while beats[-1] < seconds:
        hr = 60.0 + 4.0 * float(wave(np.array([beats[-1]]))[0])
        beats.append(beats[-1] + 60.0 / hr)
    beats = np.array(beats)
    rr = pd.DataFrame({"time": ORIGIN + beats[1:], "rr_ms": np.diff(beats) * 1000.0})
    cycles = [first_cue_s + period_s * k for k in range(1, int((seconds - first_cue_s) // period_s) - 1)]
    return SimpleNamespace(acc=acc, rr=rr), cycles


class InhaleExhaleTest(unittest.TestCase):
    def test_acc_peak_at_the_exhale_cue_and_hr_direction(self):
        # The second shape is the real one: a quick exhale, then the chest rests.
        for period, inhale, exhale in ((10.0, 4.0, None), (5.0, 2.0, None), (10.0, 4.0, 2.4), (5.0, 2.0, 1.8)):
            polar, cycles = asymmetric_breathing_session(period, inhale, exhale_s=exhale)
            check = inhale_exhale_check(polar, ORIGIN, cycles, period, inhale)
            label = (period, exhale)
            self.assertGreaterEqual(check["cycles"], 15, label)
            self.assertAlmostEqual(check["expected_inhale_fraction"], 0.4)
            self.assertAlmostEqual(check["acc_peak_after_exhale_cue_s"], 0.0, delta=0.05 * period + 0.2, msg=label)
            self.assertAlmostEqual(check["acc_peak_s"], inhale, delta=0.05 * period + 0.2, msg=label)
            self.assertTrue(check["hr_picks_inhale_direction"], label)
            self.assertGreater(check["hr_swing_bpm"], 2.0)

    def test_too_few_cycles(self):
        polar, cycles = asymmetric_breathing_session(10.0, 4.0)
        self.assertEqual(inhale_exhale_check(polar, ORIGIN, cycles[:2], 10.0, 4.0)["cycles"], 0)


class H10UnclipTest(unittest.TestCase):
    def test_contact_link_and_gap(self):
        ecg_t = np.arange(0.0, 600.0, 1 / 130.0)
        ecg_t = ecg_t[(ecg_t < 450.0) | (ecg_t >= 490.0)]
        hr_t = np.arange(0.0, 600.0, 1.0)
        polar = SimpleNamespace(
            ecg=pd.DataFrame({"time": ORIGIN + ecg_t, "uv": 0.0}),
            hr=pd.DataFrame({"time": ORIGIN + hr_t, "hr_bpm": 60, "contact": [not 445 <= t < 495 for t in hr_t]}),
        )
        events = [
            {"event": "connected", "wall": ORIGIN + 3.0},
            {"event": "disconnected", "wall": ORIGIN + 452.0},
            {"event": "connected", "wall": ORIGIN + 489.0},
        ]
        segments = [
            {"name": "h10_off", "start_s": 440.0, "end_s": 470.0},
            {"name": "h10_back", "start_s": 470.0, "end_s": 600.0},
        ]
        h10 = _h10_unclip(polar, events, ORIGIN, segments)
        self.assertEqual((h10["disconnects"], h10["reconnects"]), (1, 1))
        self.assertEqual((h10["contact_lost_s"], h10["contact_back_s"]), (445.0, 495.0))
        self.assertAlmostEqual(h10["ecg_gap"]["seconds"], 40.0, delta=0.05)
        self.assertAlmostEqual(h10["ecg_rate_last_minute_hz"], 130.0, delta=1.0)
        self.assertFalse(h10["ecg_gap"]["until_end"])
        self.assertFalse(_h10_unclip(polar, events, ORIGIN, segments[1:])["available"])

    def test_ecg_that_never_comes_back_is_a_gap_to_the_end(self):
        ecg_t = np.arange(0.0, 452.0, 1 / 130.0)
        polar = SimpleNamespace(ecg=pd.DataFrame({"time": ORIGIN + ecg_t, "uv": 0.0}), hr=None)
        segments = [
            {"name": "h10_off", "start_s": 440.0, "end_s": 470.0},
            {"name": "h10_back", "start_s": 470.0, "end_s": 600.0},
        ]
        h10 = _h10_unclip(polar, [], ORIGIN, segments)
        self.assertTrue(h10["ecg_gap"]["until_end"])
        self.assertAlmostEqual(h10["ecg_gap"]["seconds"], 148.0, delta=0.05)
        self.assertEqual(h10["ecg_rate_last_minute_hz"], 0.0)
        page = render_calibration_report({"h10": h10, "segments": [], "pairs": {}})
        self.assertIn("did not come back before the end", page)
        self.assertNotIn("kept streaming", page)


# A time-compressed copy of the protocol with the same step names.
SHORT = (
    Segment("settle", "Settle", 0, 20, "s"),
    Segment("relaxed_1", "Relaxed", 20, 60, "r1", "relaxed"),
    Segment("jaw", "Jaw", 60, 100, "j", "jaw"),
    Segment("relaxed_2", "Relaxed", 100, 140, "r2", "relaxed"),
    Segment("forehead", "Forehead", 140, 180, "f", "forehead"),
    Segment("relaxed_3", "Relaxed", 180, 220, "r3", "relaxed"),
    Segment("clench", "Clench", 220, 260, "c", "clench", Pace(3.0, ((0.0, "сжать"), (1.0, "отпустить")), lead_s=4.0)),
    Segment("relaxed_4", "Relaxed", 260, 300, "r4", "relaxed"),
    Segment("breath_6", "6/min", 300, 370, "b6", "breath_6", Pace(10.0, ((0.0, "вдох"), (4.0, "выдох")), 6.0, 6.0)),
    Segment("breath_12", "12/min", 370, 440, "b12", "breath_12", Pace(5.0, ((0.0, "вдох"), (2.0, "выдох")), 5.0, 12.0)),
    Segment("h10_off", "Unclip", 440, 470, "off"),
    Segment("h10_back", "Back", 470, 510, "back"),
)


class CalibrationReportTest(unittest.TestCase):
    def test_tension_shows_on_its_own_channels_and_breathing_on_the_pace(self):
        from muse_tmr.data.polar_session import load_polar_session
        from tests.polar_synthetic import write_raw_session

        rng = np.random.default_rng(21)
        pieces = []
        for segment in SHORT:
            seconds = segment.end_s - segment.start_s
            signal = condition_signal("busy", seconds, rng)
            emg = {"jaw": (("TP9", "TP10"), 15.0), "forehead": (("AF7", "AF8"), 15.0), "clench": (("TP9", "TP10"), 30.0)}
            if segment.name in emg:
                channels, amplitude = emg[segment.name]
                for channel in channels:
                    signal[channel] = signal[channel] + amplitude * band_noise(signal[channel].size, (30.0, 95.0), rng)
            pieces.append(signal)
        pieces.append(condition_signal("busy", 10.0, rng))
        frames = frames_from_segments(pieces, start_time=ORIGIN)

        def breathing(t):
            return 6.0 if 300 <= t < 370 else 12.0 if 370 <= t < 440 else 10.0

        cues = [
            {"kind": cue.kind, "segment": cue.segment, "cycle_start": cue.cycle_start, "elapsed_s": cue.planned_s}
            for cue in schedule(SHORT)
        ]
        segments = segments_from_cues(cues, SHORT)
        events = [{"event": "disconnected", "wall": ORIGIN + 445.0}, {"event": "connected", "wall": ORIGIN + 480.0}]
        with tempfile.TemporaryDirectory() as tmp:
            write_raw_session(tmp, 525.0, rng, wall0=ORIGIN, acc_breaths_per_min=breathing)
            polar = load_polar_session(Path(tmp))
            report = asyncio.run(
                build_calibration_report_from_frames(aiter_frames(frames), segments, polar=polar, polar_events=events)
            )
            paths = report.write(Path(tmp) / "out")
            self.assertTrue((Path(tmp) / "out" / "jaw" / "report.html").is_file())
            page = paths["report"].read_text(encoding="utf-8")

        pairs = report.summary["pairs"]
        self.assertEqual(sorted(pairs), ["breathing", "clench", "forehead", "jaw"])
        jaw = pairs["jaw"]["variants"]["all"]
        forehead = pairs["forehead"]["variants"]["all"]
        self.assertEqual(jaw["epochs"], {"jaw": 3, "relaxed": 6})
        self.assertGreater(jaw["emg_55_95_db"]["temporal"], 5.0)
        self.assertGreater(jaw["emg_55_95_db"]["temporal"] - jaw["emg_55_95_db"]["frontal"], 4.0)
        self.assertGreater(forehead["emg_55_95_db"]["frontal"], 5.0)
        self.assertGreater(forehead["emg_55_95_db"]["frontal"] - forehead["emg_55_95_db"]["temporal"], 4.0)
        self.assertGreater(pairs["clench"]["variants"]["all"]["emg_55_95_db"]["temporal"], jaw["emg_55_95_db"]["temporal"])
        self.assertTrue(math.isfinite(jaw["lzc_all"]) and math.isfinite(jaw["aperiodic_exponent_2_40_all"]))

        breathing_rows = {item["segment"]: item for item in report.summary["breathing"]}
        self.assertAlmostEqual(breathing_rows["breath_6"]["estimates_bpm"]["acc_spectral"], 6.0, delta=1.0)
        self.assertAlmostEqual(breathing_rows["breath_12"]["estimates_bpm"]["acc_spectral"], 12.0, delta=1.0)
        self.assertGreaterEqual(breathing_rows["breath_6"]["inhale_exhale"]["cycles"], 3)
        h10 = report.summary["h10"]
        self.assertEqual((h10["disconnects"], h10["reconnects"]), (1, 1))
        self.assertGreaterEqual(len(report.summary["timeline"]), 50)

        self.assertIn("Muscle tension", page)
        self.assertIn("Breathing against a known pace", page)
        self.assertIn('href="jaw/report.html"', page)
        self.assertIn("12 of 12 steps", page)

    def test_page_survives_a_run_without_polar(self):
        summary = {
            "recording": "/x/20261009_090000_calibration",
            "segments": [{"name": "settle", "completed": True}],
            "pairs": {"jaw": {"error": "no epochs inside the blocks"}},
            "timeline": [],
            "breathing": [],
            "h10": {"available": False, "error": "no polar/ folder in this recording"},
            "stop_reason": "interrupted",
        }
        page = render_calibration_report(summary)
        self.assertIn("stopped: interrupted", page)
        self.assertIn("no polar/ folder", page)


class PairBlocksTest(unittest.TestCase):
    def test_actual_times_are_used(self):
        cues = [
            {"kind": "segment", "segment": "relaxed_1", "elapsed_s": 61.2},
            {"kind": "segment", "segment": "jaw", "elapsed_s": 181.0},
            {"kind": "segment", "segment": "relaxed_2", "elapsed_s": 300.4},
            {"kind": "segment", "segment": "forehead", "elapsed_s": 421.0},
        ]
        pairs = pair_blocks(segments_from_cues(cues))
        self.assertEqual(list(pairs), ["jaw"])
        blocks = pairs["jaw"].blocks
        # Rounded to the second so the 10 s epoch grid does not lose an epoch per boundary.
        self.assertEqual((blocks[1].start_s, blocks[1].end_s), (181.0, 300.0))
        self.assertEqual((blocks[0].start_s, blocks[0].end_s), (62.0, 181.0))
        self.assertEqual((blocks[2].start_s, blocks[2].end_s), (300.0, 419.0))


if __name__ == "__main__":
    unittest.main()
