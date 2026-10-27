"""The morning sound check: which puzzle does this sound belong to? (#231)"""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from muse_tmr.audio import (
    AudioCuePlayer,
    CueLibrary,
    CueMetadata,
    MockAudioBackend,
    VolumeCalibration,
    save_volume_calibration,
)
from muse_tmr.cli import main as cli_main
from muse_tmr.protocol import NightPuzzleSession, PuzzleCatalog, PuzzleCueAssignment, PuzzleTask
from muse_tmr.reports import (
    CueRecallTest,
    CueRecallUnavailable,
    build_cue_recall_test,
    cue_recall_order,
    load_cue_recall_test,
    plan_cue_recall,
)
from muse_tmr.reports.cue_recall import cue_recall_playback_config, play_cue_recall_trial

PUZZLES = ("p1", "p2", "p3", "p4")


def fixture(cued=("p2", "p3"), volume_hint=0.10):
    catalog = PuzzleCatalog(
        puzzles=tuple(PuzzleTask(puzzle, f"Prompt {puzzle}", f"Answer {puzzle}", cue_id=f"cue-{puzzle}") for puzzle in PUZZLES)
    )
    session = NightPuzzleSession(session_id="night-001", puzzle_ids=PUZZLES, puzzle_count=4)
    assignment = PuzzleCueAssignment(
        session_id="night-001",
        cued_puzzle_ids=tuple(cued),
        uncued_puzzle_ids=tuple(puzzle for puzzle in PUZZLES if puzzle not in cued),
        seed=7,
    )
    library = CueLibrary(
        cues=tuple(
            CueMetadata(
                f"cue-{puzzle}",
                "generated_tone",
                0.5,
                protocol="puzzle",
                frequency_hz=300.0 + 100 * index,
                volume_hint=volume_hint,
            )
            for index, puzzle in enumerate(PUZZLES)
        )
    )
    return catalog, session, assignment, library


class OrderTest(unittest.TestCase):
    def test_the_order_depends_on_the_session_alone(self):
        _, session, _, _ = fixture()
        order = cue_recall_order(session)
        self.assertEqual(sorted(order), sorted(PUZZLES))
        self.assertEqual(order, cue_recall_order(session))
        # The same session under another split plays its sounds in the same order.
        catalog, _, assignment, library = fixture(cued=("p1", "p4"))
        plans = plan_cue_recall(session, catalog, library)
        self.assertEqual(tuple(plan.puzzle_id for plan in plans), order)
        self.assertEqual([plan.trial for plan in plans], [1, 2, 3, 4])
        other = replace(session, session_id="night-002")
        self.assertEqual(sorted(cue_recall_order(other)), sorted(PUZZLES))

    def test_a_missing_sound_is_counted_never_named(self):
        catalog, session, _, library = fixture()
        only_two = CueLibrary(cues=library.cues[1:3])
        with self.assertRaises(CueRecallUnavailable) as raised:
            plan_cue_recall(session, catalog, only_two)
        message = str(raised.exception)
        self.assertIn("2 of the night's 4 puzzles", message)
        for puzzle in PUZZLES:
            self.assertNotIn(puzzle, message)
        sound_file = CueLibrary(cues=(replace(library.cues[0], cue_type="sound", frequency_hz=None),) + library.cues[1:])
        with self.assertRaises(CueRecallUnavailable):
            plan_cue_recall(session, catalog, sound_file)


class PlaybackTest(unittest.TestCase):
    def test_sounds_play_through_the_player_under_the_calibrated_cap(self):
        catalog, session, _, library = fixture(volume_hint=0.10)
        backend = MockAudioBackend()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audio.jsonl"
            player = AudioCuePlayer(
                cue_recall_playback_config(VolumeCalibration("Speaker", 0.01, 0.02, 0.04), log_path=log_path),
                backend=backend,
            )
            plan = plan_cue_recall(session, catalog, library)[0]
            played = play_cue_recall_trial(player, plan)
            logged = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(played["status"], "played")
        self.assertEqual(played["backend_name"], "mock")
        self.assertEqual((played["requested_volume"], played["capped_volume"], played["effective_volume"]), (0.10, 0.04, 0.04))
        self.assertEqual(backend.requests[0].effective_volume, 0.04)
        self.assertEqual(logged[0]["cue_id"], plan.cue.cue_id)
        # A calibration above the hard cap is held at 0.20.
        loud = cue_recall_playback_config(VolumeCalibration("Speaker", 0.1, 0.2, 0.6))
        self.assertEqual(loud.max_volume, 0.20)
        self.assertEqual(loud.cue_volume(0.5).effective_volume, 0.20)


class RecordTest(unittest.TestCase):
    def test_answers_become_trials_with_their_conditions(self):
        catalog, session, assignment, _ = fixture(cued=("p2", "p3"))
        order = cue_recall_order(session)
        answers = {index: puzzle for index, puzzle in enumerate(order, start=1)}
        answers[1] = None  # did not know
        test = build_cue_recall_test(session, catalog, assignment, answers, play_counts={1: 2})
        self.assertEqual(test.trial_count, 4)
        self.assertEqual((test.answered_count, test.correct_count, test.recall_rate), (3, 3, 0.75))
        conditions = {trial.puzzle_id: trial.cue_condition for trial in test.trials}
        self.assertEqual(conditions, {"p1": "uncued", "p2": "cued", "p3": "cued", "p4": "uncued"})
        first = order[0]
        by_condition = test.by_condition()
        missed = "cued" if first in ("p2", "p3") else "uncued"
        self.assertEqual(by_condition[missed]["correct_count"], 1)
        self.assertEqual(test.trials[0].play_count, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = test.save(Path(tmp) / "recall.json")
            self.assertEqual(load_cue_recall_test(path).to_dict(), test.to_dict())

    def test_every_sound_needs_an_answer_from_this_night(self):
        catalog, session, assignment, _ = fixture()
        with self.assertRaises(ValueError):
            build_cue_recall_test(session, catalog, assignment, {1: "p1", 2: "p2", 3: "p3"})
        with self.assertRaises(ValueError):
            build_cue_recall_test(session, catalog, assignment, {1: "p1", 2: "p2", 3: "p3", 4: "p9"})
        with self.assertRaises(ValueError):
            CueRecallTest(session_id="night-001", trials=(), order_seed=1).validate_against_session(session)


class CliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        catalog, session, assignment, library = fixture()
        self.paths = {}
        for name, record in (("catalog", catalog), ("session", session), ("assignment", assignment), ("cues", library)):
            self.paths[name] = record.save(self.root / f"{name}.json")
        self.paths["calibration"] = save_volume_calibration(
            VolumeCalibration("Speaker", 0.01, 0.02, 0.04), self.root / "calibration.json"
        )
        self.output = self.root / "night-001_cue_recall.json"
        self.order = cue_recall_order(session)

    def tearDown(self):
        self._tmp.cleanup()

    def argv(self, *extra):
        return [
            "record-cue-recall", str(self.paths["session"]),
            "--catalog", str(self.paths["catalog"]),
            "--assignment", str(self.paths["assignment"]),
            "--cue-library", str(self.paths["cues"]),
            "--calibration", str(self.paths["calibration"]),
            "--output", str(self.output),
            *extra,
        ]

    def test_the_check_plays_each_sound_and_saves_the_answers(self):
        position = {puzzle: str(index) for index, puzzle in enumerate(PUZZLES, start=1)}
        # Sound 1 again first, then its right puzzle; sound 2 unknown; 3 and 4 right.
        replies = ["r", position[self.order[0]], "", "7", position[self.order[2]], position[self.order[3]]]
        stdout = io.StringIO()
        # --backend mock reaches the player as it is, with nothing patched in between.
        with patch("builtins.input", side_effect=replies), contextlib.redirect_stdout(stdout):
            self.assertEqual(cli_main.main(self.argv("--backend", "mock")), 0)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual([trial["response_puzzle_id"] for trial in saved["trials"]],
                         [self.order[0], None, self.order[2], self.order[3]])
        self.assertEqual([trial["play_count"] for trial in saved["trials"]], [2, 1, 1, 1])
        self.assertEqual(saved["correct_count"], 3)
        self.assertTrue(all(trial["cue_condition"] in ("cued", "uncued") for trial in saved["trials"]))
        self.assertEqual((saved["metadata"]["requested_backend"], saved["metadata"]["resolved_backend"]), ("mock", "mock"))
        audio_log = self.root / "night-001_cue_recall_audio.jsonl"
        lines = [json.loads(line) for line in audio_log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 5)
        self.assertEqual({line["backend_name"] for line in lines}, {"mock"})
        self.assertEqual({line["status"] for line in lines}, {"played"})
        self.assertTrue(all(line["max_volume"] == 0.04 and line["effective_volume"] <= 0.04 for line in lines))
        printed = stdout.getvalue()
        # Nothing printed before the retest says which were cued or right.
        for leak in ("cued", "correct", "recall_rate"):
            self.assertNotIn(leak, printed)
        self.assertIn("answered=3", printed)

    def test_real_audio_without_afplay_is_refused_before_any_input(self):
        stderr = io.StringIO()
        with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None), \
                patch("muse_tmr.audio.load_cue_library") as load_cue_library, \
                contextlib.redirect_stderr(stderr):
            self.assertEqual(cli_main.main(self.argv("--backend", "system")), 2)
        self.assertIn("needs afplay", stderr.getvalue())
        load_cue_library.assert_not_called()
        self.assertFalse(self.output.exists())

        with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None), \
                patch("builtins.input", side_effect=["1", "2", "3", "4"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main.main(self.argv("--backend", "system", "--allow-dry-run")), 0)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(saved["metadata"]["resolved_backend"], "dry-run")
        self.assertEqual(saved["trials"][0]["plays"][0]["status"], "played")

    def test_input_that_ends_early_saves_nothing(self):
        with patch("builtins.input", side_effect=["1", EOFError()]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli_main.main(self.argv("--backend", "mock")), 1)
        self.assertFalse(self.output.exists())

    def test_a_library_without_every_sound_is_refused(self):
        catalog, _, _, library = fixture()
        CueLibrary(cues=library.cues[:3]).save(self.paths["cues"])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(cli_main.main(self.argv("--backend", "mock")), 2)
        self.assertIn("1 of the night's 4 puzzles", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
