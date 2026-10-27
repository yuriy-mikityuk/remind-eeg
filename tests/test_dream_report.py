import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from muse_tmr.cli.main import main
from muse_tmr.protocol import NightPuzzleSession, PuzzleCatalog, PuzzleTask
from muse_tmr.reports import (
    CUE_PERCEPTIONS,
    DreamPuzzleIncorporation,
    DreamReport,
    build_dream_report,
    load_dream_report,
    parse_eye_signal_attempt_times,
)

# A report as schema 1 wrote it, before #228.
SCHEMA_1_REPORT = {
    "schema_version": 1,
    "report_id": "night-001-dream-report",
    "session_id": "night-001",
    "reported_at_utc": "2026-10-01T06:30:00+00:00",
    "lucid": False,
    "cues_heard": True,
    "confidence": 0.6,
    "dream_text": "A staircase that kept folding.",
    "puzzle_incorporation_count": 1,
    "puzzle_incorporations": [
        {"puzzle_id": "p2", "cue_id": "cue-p2", "incorporated": True, "dream_content": "stairs", "confidence": 1.0, "notes": ""}
    ],
    "notes": "",
    "metadata": {"session_puzzle_count": 2, "linked_puzzle_count": 1},
}


class TestDreamReport(unittest.TestCase):
    def test_report_round_trips_json_and_counts_puzzle_links(self):
        report = DreamReport(
            report_id="report-001",
            session_id="night-001",
            lucid=True,
            cues_heard=False,
            confidence=0.75,
            dream_text="I saw a blue key on a desk.",
            puzzle_incorporations=(
                DreamPuzzleIncorporation(
                    puzzle_id="p1",
                    cue_id="cue-p1",
                    dream_content="blue key on a desk",
                    confidence=0.8,
                ),
            ),
            reported_at_utc="2026-05-13T00:00:00+00:00",
        )

        loaded = DreamReport.from_dict(report.to_dict())

        self.assertEqual(loaded.report_id, "report-001")
        self.assertTrue(loaded.lucid)
        self.assertFalse(loaded.cues_heard)
        self.assertEqual(loaded.incorporated_puzzle_ids, ("p1",))
        self.assertEqual(loaded.puzzle_incorporation_count, 1)

    def test_build_report_links_only_session_puzzles_and_uses_catalog_cue_ids(self):
        session = NightPuzzleSession(
            session_id="night-001",
            puzzle_ids=("p1", "p2"),
            puzzle_count=2,
        )
        catalog = PuzzleCatalog(
            puzzles=(
                PuzzleTask("p1", "Puzzle 1", "Answer 1", cue_id="cue-p1"),
                PuzzleTask("p2", "Puzzle 2", "Answer 2", cue_id="cue-p2"),
            )
        )

        report = build_dream_report(
            session,
            catalog=catalog,
            lucid=False,
            cues_heard=True,
            confidence=0.6,
            dream_text="There was a locked room.",
            puzzle_incorporation_text={"p2": "the second puzzle answer appeared"},
        )

        self.assertEqual(report.session_id, "night-001")
        self.assertEqual(report.puzzle_incorporations[0].puzzle_id, "p2")
        self.assertEqual(report.puzzle_incorporations[0].cue_id, "cue-p2")

    def test_build_report_rejects_unknown_puzzle_links(self):
        session = NightPuzzleSession(
            session_id="night-001",
            puzzle_ids=("p1",),
            puzzle_count=1,
        )

        with self.assertRaises(ValueError):
            build_dream_report(
                session,
                lucid=False,
                cues_heard=False,
                confidence=0.5,
                dream_text="No recall.",
                puzzle_incorporation_text={"p2": "not in session"},
            )

    def test_report_rejects_duplicate_puzzle_links(self):
        with self.assertRaises(ValueError):
            DreamReport(
                session_id="night-001",
                lucid=False,
                cues_heard=False,
                confidence=0.5,
                dream_text="Fragmented recall.",
                puzzle_incorporations=(
                    DreamPuzzleIncorporation("p1", "first"),
                    DreamPuzzleIncorporation("p1", "second"),
                ),
            )

    def test_cli_records_dream_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_path = tmp_path / "session.json"
            catalog_path = tmp_path / "catalog.json"
            report_path = tmp_path / "dream_report.json"
            NightPuzzleSession(
                session_id="night-001",
                puzzle_ids=("p1", "p2"),
                puzzle_count=2,
            ).save(session_path)
            PuzzleCatalog(
                puzzles=(
                    PuzzleTask("p1", "Puzzle 1", "Answer 1", cue_id="cue-p1"),
                    PuzzleTask("p2", "Puzzle 2", "Answer 2", cue_id="cue-p2"),
                )
            ).save(catalog_path)

            with redirect_stdout(io.StringIO()):
                code = main([
                    "record-dream-report",
                    str(session_path),
                    "--catalog",
                    str(catalog_path),
                    "--output",
                    str(report_path),
                    "--lucid",
                    "yes",
                    "--cues-heard",
                    "no",
                    "--confidence",
                    "0.7",
                    "--dream-text",
                    "I found Answer 1 in a notebook.",
                    "--puzzle-link",
                    "p1=Answer 1 appeared in a notebook",
                ])

            report = load_dream_report(report_path)

        self.assertEqual(code, 0)
        self.assertTrue(report.lucid)
        self.assertFalse(report.cues_heard)
        self.assertEqual(report.confidence, 0.7)
        self.assertEqual(report.puzzle_incorporations[0].cue_id, "cue-p1")

    def test_a_schema_1_report_loads_as_it_was(self):
        for cues_heard, perception in ((True, "unsure"), (False, "not_perceived")):
            with self.subTest(cues_heard=cues_heard):
                report = DreamReport.from_dict({**SCHEMA_1_REPORT, "cues_heard": cues_heard})
                self.assertEqual(report.schema_version, 1)
                self.assertEqual(report.cues_heard, cues_heard)
                self.assertEqual(report.cue_perception, perception)
                self.assertEqual(report.woke_from_sound, "unknown")
                self.assertEqual(report.eye_signal_attempt_times, ())
                self.assertEqual(report.incorporated_puzzle_ids, ("p2",))
                self.assertEqual(report.reported_at_utc, SCHEMA_1_REPORT["reported_at_utc"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps(SCHEMA_1_REPORT), encoding="utf-8")
            self.assertTrue(load_dream_report(path).cues_heard)

    def test_schema_2_round_trips_perception_awakening_and_eye_signals(self):
        session = NightPuzzleSession(session_id="night-001", puzzle_ids=("p1",), puzzle_count=1)
        report = build_dream_report(
            session,
            lucid=True,
            confidence=0.8,
            dream_text="I looked at my hands.",
            cue_perception="in_dream",
            woke_from_sound="no",
            eye_signal_attempt_times=("04:20", "2026-10-27T05:10:00+00:00"),
        )
        payload = report.to_dict()
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(
            (payload["cues_heard"], payload["cue_perception"], payload["woke_from_sound"]), (True, "in_dream", "no")
        )
        self.assertEqual(payload["eye_signal_attempt_times"], ["04:20", "2026-10-27T05:10:00+00:00"])
        self.assertEqual(DreamReport.from_dict(payload), report)
        self.assertEqual(CUE_PERCEPTIONS, ("not_perceived", "in_dream", "awake", "unsure", "unknown"))

    def test_cues_heard_and_perception_must_agree(self):
        base = dict(session_id="night-001", lucid=False, confidence=0.5, dream_text="")
        self.assertTrue(DreamReport(cues_heard=True, cue_perception="awake", **base).cues_heard)
        DreamReport(cues_heard=False, cue_perception="unknown", **base)
        DreamReport(cues_heard=True, cue_perception="unknown", **base)
        for cues_heard, perception in ((False, "in_dream"), (False, "unsure"), (True, "not_perceived")):
            with self.assertRaises(ValueError):
                DreamReport(cues_heard=cues_heard, cue_perception=perception, **base)
        for field, value in (("cue_perception", "faint"), ("woke_from_sound", "maybe")):
            with self.assertRaises(ValueError):
                DreamReport(cues_heard=False, **{field: value}, **base)
        session = NightPuzzleSession(session_id="night-001", puzzle_ids=("p1",), puzzle_count=1)
        with self.assertRaises(ValueError):
            build_dream_report(session, lucid=False, confidence=0.5, dream_text="")
        unknown = build_dream_report(session, lucid=False, confidence=0.5, dream_text="", cue_perception="unknown")
        self.assertFalse(unknown.cues_heard)

    def test_eye_signal_times_are_clock_or_iso_times(self):
        self.assertEqual(parse_eye_signal_attempt_times(" 04:20, 5:45:10;23:59 "), ("04:20", "5:45:10", "23:59"))
        self.assertEqual(parse_eye_signal_attempt_times(""), ())
        base = dict(session_id="night-001", lucid=False, cues_heard=False, confidence=0.5, dream_text="")
        self.assertEqual(
            DreamReport(eye_signal_attempt_times=("2026-10-27 04:20",), **base).eye_signal_attempt_times,
            ("2026-10-27 04:20",),
        )
        for bad in ("24:00", "4:2", "after lunch", "2026-13-40T04:20"):
            with self.assertRaises(ValueError, msg=bad):
                DreamReport(eye_signal_attempt_times=(bad,), **base)

    def test_cli_records_perception_without_cues_heard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_path = tmp_path / "session.json"
            NightPuzzleSession(session_id="night-001", puzzle_ids=("p1",), puzzle_count=1).save(session_path)
            base = [
                "record-dream-report", str(session_path), "--lucid", "no", "--confidence", "0.4", "--dream-text", "x",
            ]
            with redirect_stdout(io.StringIO()):
                code = main(base + [
                    "--output", str(tmp_path / "report.json"), "--cue-perception", "awake",
                    "--woke-from-sound", "yes", "--eye-signal-attempt", "04:20", "--eye-signal-attempt", "05:01",
                ])
            report = load_dream_report(tmp_path / "report.json")
            errors = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(errors):
                neither = main(base + ["--output", str(tmp_path / "none.json")])
                contradiction = main(base + [
                    "--output", str(tmp_path / "bad.json"), "--cues-heard", "no", "--cue-perception", "in_dream",
                ])

        self.assertEqual(code, 0)
        self.assertEqual((report.cue_perception, report.cues_heard, report.woke_from_sound), ("awake", True, "yes"))
        self.assertEqual(report.eye_signal_attempt_times, ("04:20", "05:01"))
        self.assertEqual((neither, contradiction), (2, 2))
        self.assertIn("give --cue-perception, --cues-heard or both", errors.getvalue())

    def test_saved_json_uses_stable_schema_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report = DreamReport(
                session_id="night-001",
                lucid=False,
                cues_heard=False,
                confidence=0.5,
                dream_text="No recall.",
            )
            report.save(report_path)

            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["session_id"], "night-001")
        self.assertEqual(payload["puzzle_incorporation_count"], 0)
        self.assertIn("puzzle_incorporations", payload)


if __name__ == "__main__":
    unittest.main()
