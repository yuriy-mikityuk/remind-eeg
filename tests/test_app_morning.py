import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from muse_tmr.app import AppConfig, create_local_app_server, morning
from muse_tmr.audio import CueLibrary, CueMetadata, MockAudioBackend, VolumeCalibration, save_volume_calibration
from muse_tmr.protocol import NightPuzzleSession, PuzzleCatalog, PuzzleCueAssignment, PuzzleTask
from muse_tmr.reports import cue_recall_order, load_cue_recall_test, load_dream_report, load_morning_retest

PUZZLES = ("p1", "p2", "p3", "p4")


def write_night(
    protocol_dir,
    session_file="night-001_puzzles.json",
    session_id="night-001",
    catalog_file="puzzle_catalog.json",
    assignment=True,
    cued=("p2", "p3"),
):
    protocol_dir.mkdir(parents=True, exist_ok=True)
    PuzzleCatalog(
        puzzles=tuple(PuzzleTask(puzzle_id=puzzle, prompt=f"Prompt {puzzle}", solution=f"Answer {puzzle}") for puzzle in PUZZLES)
    ).save(protocol_dir / catalog_file)
    NightPuzzleSession(session_id=session_id, puzzle_ids=PUZZLES, puzzle_count=4).save(protocol_dir / session_file)
    if assignment:
        PuzzleCueAssignment(
            session_id=session_id,
            cued_puzzle_ids=tuple(cued),
            uncued_puzzle_ids=tuple(puzzle for puzzle in PUZZLES if puzzle not in cued),
            seed=7,
        ).save(protocol_dir / f"{session_id}_assignment.json")


def write_sounds(data_dir, puzzles=PUZZLES):
    """A cue library in data/cues (a generated tone per puzzle) and a calibration."""
    CueLibrary(
        cues=tuple(
            CueMetadata(puzzle, "generated_tone", 0.2, protocol="puzzle", frequency_hz=300.0 + 100 * index, volume_hint=0.1)
            for index, puzzle in enumerate(puzzles)
        )
    ).save(data_dir / "cues" / "starter.json")
    save_volume_calibration(VolumeCalibration("Speaker", 0.01, 0.02, 0.04), data_dir / "calibration" / "volume_calibration.json")


def recall_body(answers, **overrides):
    body = {"night": "night-001", "answers": {str(trial): answer for trial, answer in answers.items()}}
    body.update(overrides)
    return body


def dream_body(**overrides):
    body = {
        "night": "night-001",
        "lucid": "no",
        "cues_heard": "yes",
        "confidence": 0.6,
        "dream_text": "A staircase that kept folding.",
        "puzzle_links": {"p3": "the folding stairs", "p1": "  "},
    }
    body.update(overrides)
    return body


def retest_body(**overrides):
    body = {
        "night": "night-001",
        "results": [
            {"puzzle_id": puzzle, "response": f"try {puzzle}", "solved": puzzle == "p2", "duration_seconds": 40 + index, "confidence": 0.5}
            for index, puzzle in enumerate(PUZZLES)
        ],
    }
    body.update(overrides)
    return body


class TestMorningNights(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.protocol = Path(self._tmp.name) / "data" / "protocol"
        self.reports = Path(self._tmp.name) / "data" / "reports"

    def tearDown(self):
        self._tmp.cleanup()

    def test_finds_a_night_whatever_its_files_are_called(self):
        write_night(self.protocol)
        write_night(self.protocol, "pilot5_smoke_session.json", "pilot5-smoke", "pilot5_smoke_catalog.json")
        os.utime(self.protocol / "night-001_puzzles.json", (1, 1))  # the older of the two

        nights = morning.find_nights(self.protocol)
        self.assertEqual([night.name for night in nights], ["pilot5_smoke", "night-001"])
        smoke, runbook = nights
        self.assertEqual(smoke.catalog_path.name, "pilot5_smoke_catalog.json")
        self.assertEqual(runbook.catalog_path.name, "puzzle_catalog.json")
        self.assertEqual(smoke.assignment.session_id, "pilot5-smoke")
        self.assertEqual(runbook.assignment.session_id, "night-001")

    def test_a_catalog_must_hold_every_puzzle_and_an_assignment_its_session(self):
        write_night(self.protocol, assignment=False)
        PuzzleCatalog(puzzles=(PuzzleTask(puzzle_id="p1", prompt="Only one", solution="x"),)).save(self.protocol / "night-001_catalog.json")
        PuzzleCueAssignment(session_id="another-night", cued_puzzle_ids=("p1",), uncued_puzzle_ids=("p2",), seed=1).save(
            self.protocol / "another_assignment.json"
        )
        (self.protocol / "notes.json").write_text("[1, 2]")

        (night,) = morning.find_nights(self.protocol)
        self.assertEqual(night.catalog_path.name, "puzzle_catalog.json")
        self.assertIsNone(night.assignment)
        payload = morning.nights_payload(self.protocol, self.reports)["nights"][0]
        self.assertFalse(payload["assignment_found"])

    def test_the_page_never_learns_which_puzzles_were_cued(self):
        write_night(self.protocol)
        write_sounds(self.protocol.parent)
        morning.save_retest(self.protocol, self.reports, retest_body())

        payload = morning.night_payload(self.protocol, self.reports, "night-001")
        self.assertEqual([puzzle["prompt"] for puzzle in payload["puzzles"]], [f"Prompt {p}" for p in PUZZLES])
        text = json.dumps(payload)
        for leak in ("cued", "cue_condition", "uncued", "Answer p"):
            self.assertNotIn(leak, text)
        # The saved file keeps the conditions for the analysis.
        saved = load_morning_retest(self.reports / "night-001_retest.json")
        self.assertEqual({r.puzzle_id: r.cue_condition for r in saved.results}, {"p1": "uncued", "p2": "cued", "p3": "cued", "p4": "uncued"})

    def test_the_sound_check_gives_the_split_away_nowhere(self):
        """Its payload, the order, number and labels of its sounds, which sound each number
        plays and what playing one answers are the same whichever puzzles were cued, and say
        nothing of the sounds."""
        seen = {}
        for split in (("p2", "p3"), ("p1", "p4"), ("p4",)):
            with tempfile.TemporaryDirectory() as tmp:
                protocol = Path(tmp) / "data" / "protocol"
                reports = Path(tmp) / "data" / "reports"
                write_night(protocol, cued=split)
                write_sounds(protocol.parent)
                recall = morning.night_payload(protocol, reports, "night-001")["cue_recall"]
                backend = MockAudioBackend()
                plays = [
                    morning.play_cue_recall(protocol, reports, {"night": "night-001", "trial": trial}, backend=backend)
                    for trial in (1, 2, 3, 4)
                ]
                night = morning.find_nights(protocol)[0]
                order = cue_recall_order(night.session)
                morning.save_cue_recall(protocol, reports, recall_body({1: order[0], 2: None, 3: "p1", 4: "p1"}))
                after = morning.night_payload(protocol, reports, "night-001")
                saved = load_cue_recall_test(reports / "night-001_cue_recall.json")
            sounds = [request.cue_id for request in backend.requests]
            seen[split] = (recall, plays, sounds, after["cue_recall"]["trials"], after["cue_recall"]["saved"]["answers"])
            # Sound N is the cue of the Nth puzzle in the session's own order.
            self.assertEqual(sounds, [night.catalog.get_puzzle(puzzle).cue_id for puzzle in order])
            # Each at its hint of 0.1, capped at the calibrated comfortable volume of 0.04.
            self.assertEqual(
                {(request.requested_volume, request.max_volume, request.effective_volume) for request in backend.requests},
                {(0.1, 0.04, 0.04)},
            )
            self.assertEqual(recall["trials"], [{"trial": 1}, {"trial": 2}, {"trial": 3}, {"trial": 4}])
            self.assertTrue(recall["available"])
            self.assertEqual(plays[0], {"trial": 1, "status": "played", "backend": "mock"})
            text = json.dumps([recall, plays, after])
            for leak in ("cued", "cue_condition", "uncued", "Answer p", "correct", "frequency", "300.0"):
                self.assertNotIn(leak, text, split)
            self.assertEqual(after["cue_recall"]["saved"]["answers"], {"1": order[0], "2": None, "3": "p1", "4": "p1"})
            self.assertTrue(after["cue_recall_saved"])
            # The file keeps the conditions, for the analysis only.
            self.assertEqual(
                {trial.puzzle_id for trial in saved.trials if trial.cue_condition == "cued"}, set(split)
            )
        first, *others = seen.values()
        for other in others:
            self.assertEqual(other, first)

    def test_a_library_missing_some_sounds_is_reported_without_naming_them(self):
        write_night(self.protocol)
        write_sounds(self.protocol.parent, puzzles=("p2", "p3"))
        recall = morning.night_payload(self.protocol, self.reports, "night-001")["cue_recall"]
        self.assertFalse(recall["available"])
        self.assertEqual(len(recall["trials"]), 4)
        self.assertIn("2 of the night's 4 puzzles", recall["unavailable_reason"])
        for puzzle in PUZZLES:
            self.assertNotIn(puzzle, recall["unavailable_reason"])
        with self.assertRaises(morning.CueRecallUnavailable):
            morning.play_cue_recall(self.protocol, self.reports, {"night": "night-001", "trial": 1}, backend=MockAudioBackend())

    def test_the_sound_check_needs_a_calibration(self):
        write_night(self.protocol)
        write_sounds(self.protocol.parent)
        (self.protocol.parent / "calibration" / "volume_calibration.json").unlink()
        recall = morning.night_payload(self.protocol, self.reports, "night-001")["cue_recall"]
        self.assertFalse(recall["available"])
        self.assertIn("calibrate-volume", recall["unavailable_reason"])


class TestMorningSaving(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.protocol = Path(self._tmp.name) / "data" / "protocol"
        self.reports = Path(self._tmp.name) / "data" / "reports"
        write_night(self.protocol)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_dream_report_goes_where_the_runbook_puts_it(self):
        result = morning.save_dream_report(self.protocol, self.reports, dream_body())
        self.assertEqual(result["saved"], "night-001_dream_report.json")

        report = load_dream_report(self.reports / "night-001_dream_report.json")
        self.assertEqual((report.session_id, report.lucid, report.cues_heard, report.confidence), ("night-001", False, True, 0.6))
        self.assertEqual(report.incorporated_puzzle_ids, ("p3",))
        saved = morning.night_payload(self.protocol, self.reports, "night-001")["dream_report"]
        self.assertEqual(saved["puzzle_links"], {"p3": "the folding stairs"})

    def test_saving_again_needs_replace(self):
        morning.save_dream_report(self.protocol, self.reports, dream_body())
        with self.assertRaises(morning.MorningFileExists):
            morning.save_dream_report(self.protocol, self.reports, dream_body(dream_text="Changed."))
        morning.save_dream_report(self.protocol, self.reports, dream_body(dream_text="Changed.", replace=True))
        self.assertEqual(load_dream_report(self.reports / "night-001_dream_report.json").dream_text, "Changed.")

    def test_the_retest_needs_every_puzzle_and_valid_answers(self):
        with self.assertRaises(ValueError):
            morning.save_retest(self.protocol, self.reports, retest_body(results=retest_body()["results"][:3]))
        with self.assertRaises(ValueError):
            bad = retest_body()
            bad["results"][0]["confidence"] = 7
            morning.save_retest(self.protocol, self.reports, bad)
        with self.assertRaises(ValueError):
            morning.save_dream_report(self.protocol, self.reports, dream_body(lucid="maybe"))
        self.assertFalse(self.reports.exists())

        morning.save_retest(self.protocol, self.reports, retest_body())
        retest = load_morning_retest(self.reports / "night-001_retest.json")
        self.assertEqual(retest.solved_count, 1)
        self.assertEqual([r.duration_seconds for r in retest.results], [40, 41, 42, 43])

    def test_a_saved_retest_comes_back_with_its_notes(self):
        body = retest_body()
        body["results"][2]["notes"] = "knew it at once"
        morning.save_retest(self.protocol, self.reports, body)
        saved = morning.night_payload(self.protocol, self.reports, "night-001")["retest"]["results"]
        self.assertEqual(saved["p3"]["notes"], "knew it at once")
        self.assertEqual(saved["p1"]["notes"], "")

    def test_the_retest_needs_the_nights_assignment(self):
        (self.protocol / "night-001_assignment.json").unlink()
        with self.assertRaises(ValueError) as raised:
            morning.save_retest(self.protocol, self.reports, retest_body())
        self.assertIn("no cue assignment", str(raised.exception))


class TestMorningEndpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        data = Path(self._tmp.name) / "data"
        write_night(data / "protocol")
        self.server = create_local_app_server(AppConfig(port=0, source="mock"), recordings_base=data / "recordings")
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.app_state.shutdown()
        self.server.server_close()
        self._tmp.cleanup()

    def request(self, path, body=None):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def test_the_sound_check_plays_through_the_resolved_backend(self):
        write_sounds(Path(self._tmp.name) / "data")
        self.server.app_state.config = AppConfig(port=0, source="mock", cue_backend="mock")
        status, payload = self.request("/api/morning/cue-recall/play", {"night": "night-001", "trial": 2})
        self.assertEqual((status, payload), (200, {"trial": 2, "status": "played", "backend": "mock"}))
        self.assertEqual(self.request("/api/morning/cue-recall/play", {"night": "night-001", "trial": 9})[0], 400)
        self.assertEqual(self.request("/api/morning/cue-recall/play", {"night": "nope", "trial": 1})[0], 404)

        self.server.app_state.config = AppConfig(port=0, source="mock", cue_backend="system")
        with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None):
            status, payload = self.request("/api/morning/cue-recall/play", {"night": "night-001", "trial": 1})
            self.assertEqual(status, 409)
            self.assertIn("--cue-backend system needs afplay", payload["error"])
            self.assertIn("--allow-dry-run", payload["error"])
            self.server.app_state.config = AppConfig(port=0, source="mock", cue_backend="system", allow_dry_run=True)
            status, payload = self.request("/api/morning/cue-recall/play", {"night": "night-001", "trial": 1})
            self.assertEqual((status, payload["backend"]), (200, "dry-run"))

        answers = {1: "p1", 2: "p2", 3: None, 4: "p4"}
        self.assertEqual(self.request("/api/morning/cue-recall", recall_body(answers))[0], 200)
        self.assertEqual(self.request("/api/morning/cue-recall", recall_body(answers))[0], 409)
        self.assertEqual(self.request("/api/morning/cue-recall", recall_body({1: "p1"}, replace=True))[0], 400)
        status, payload = self.request("/api/morning/night?name=night-001")
        self.assertEqual(payload["cue_recall"]["saved"]["answers"], {"1": "p1", "2": "p2", "3": None, "4": "p4"})

    def test_lists_shows_and_saves_a_night(self):
        status, payload = self.request("/api/morning/nights")
        self.assertEqual((status, [night["name"] for night in payload["nights"]]), (200, ["night-001"]))
        status, payload = self.request("/api/morning/night?name=night-001")
        self.assertEqual((status, payload["puzzle_count"], payload["dream_report"]), (200, 4, None))
        self.assertEqual(self.request("/api/morning/night?name=nope")[0], 404)

        self.assertEqual(self.request("/api/morning/dream-report", dream_body())[0], 200)
        status, payload = self.request("/api/morning/dream-report", dream_body())
        self.assertEqual((status, payload["exists"]), (409, True))
        self.assertEqual(self.request("/api/morning/retest", retest_body(night="nope"))[0], 404)
        self.assertEqual(self.request("/api/morning/retest", {"night": "night-001", "results": [{"puzzle_id": "p1"}]})[0], 400)
        self.assertEqual(self.request("/api/morning/retest", retest_body())[0], 200)

        status, payload = self.request("/api/morning/nights")
        night = payload["nights"][0]
        self.assertTrue(night["dream_report_saved"] and night["retest_saved"])


class TestSoundCheckFlags(unittest.TestCase):
    def test_the_app_command_hands_its_backend_flags_to_the_server(self):
        from muse_tmr.cli.main import main as cli_main

        with patch("muse_tmr.app.run_local_app", return_value=0) as run:
            self.assertEqual(cli_main(["app", "--source", "mock", "--cue-backend", "mock", "--allow-dry-run"]), 0)
            config = run.call_args.args[0]
            self.assertEqual((config.cue_backend, config.allow_dry_run), ("mock", True))
            self.assertEqual(cli_main(["app", "--source", "mock"]), 0)
            config = run.call_args.args[0]
            self.assertEqual((config.cue_backend, config.allow_dry_run), ("system", False))


if __name__ == "__main__":
    unittest.main()
