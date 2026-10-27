"""Blind incorporation coding (#228): the packet a rater gets, the codes and the
agreement of two raters. Synthetic text only."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from muse_tmr.cli import main as cli_main
from muse_tmr.protocol import NIGHT_ARMS, NightPuzzleSession, PuzzleCatalog, PuzzleCueAssignment, PuzzleTask
from muse_tmr.reports import (
    CODING_CATEGORIES,
    BlindCodingPacket,
    DreamPuzzleIncorporation,
    DreamReport,
    IncorporationCoding,
    IncorporationCodingSheet,
    build_incorporation_coding,
    coding_agreement,
    cohens_kappa,
    export_blind_coding_packet,
    load_incorporation_coding,
)
from muse_tmr.reports.incorporation_coding import parse_coding_categories, rater_incorporation

PUZZLES = tuple(f"p{index}" for index in range(1, 9))
# What a rater must not get: a key walk over the parsed JSON, as for the Morning tab.
FORBIDDEN_KEYS = {
    "arm",
    "sham",
    "cue_condition",
    "cued_puzzle_ids",
    "uncued_puzzle_ids",
    "scheduled_puzzle_ids",
    "commitment",
    "commitment_sha256",
    "cue_id",
    "cues_heard",
    "cue_perception",
    "woke_from_sound",
    "puzzle_incorporations",
    "assignment",
    "seed",
}
# Two raters of the 8-puzzle synthetic packet below, per category.
RATER_A = {
    "sound_cue": (1, 0, 0, 0, 1, 0, 0, 0),
    "direct_conceptual": (1, 1, 0, 0, 1, 0, 0, 0),
    "indirect_conceptual": (0, 0, 1, 0, 0, 0, 1, 0),
}
RATER_B = {
    "sound_cue": (1, 0, 0, 0, 0, 0, 0, 0),
    "direct_conceptual": (1, 0, 0, 0, 1, 0, 1, 0),
    "indirect_conceptual": (0, 0, 1, 0, 0, 0, 1, 0),
}


def leaks(payload, where=()):
    found = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in FORBIDDEN_KEYS:
                found.append(("key", where + (key,)))
            found.extend(leaks(value, where + (key,)))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(leaks(value, where + (index,)))
    elif isinstance(payload, str) and payload in NIGHT_ARMS:
        found.append(("value", where))
    return found


def night():
    session = NightPuzzleSession(session_id="night-003", puzzle_ids=PUZZLES, puzzle_count=len(PUZZLES))
    catalog = PuzzleCatalog(
        puzzles=tuple(PuzzleTask(puzzle, f"Prompt {puzzle}", f"Answer {puzzle}", cue_id=f"cue-{puzzle}") for puzzle in PUZZLES)
    )
    dream = DreamReport(
        session_id="night-003",
        lucid=True,
        cues_heard=True,
        cue_perception="in_dream",
        woke_from_sound="no",
        confidence=0.7,
        dream_text="A synthetic dream: a sham door, an alarm, a warm cue card.",
        puzzle_incorporations=(DreamPuzzleIncorporation("p2", "the card", cue_id="cue-p2"),),
        notes="heard the tone twice",
    )
    return session, catalog, dream


def codes_of(table):
    return {
        puzzle: tuple(category for category in CODING_CATEGORIES if table[category][index])
        for index, puzzle in enumerate(PUZZLES)
    }


class TestPacket(unittest.TestCase):
    def test_the_packet_holds_the_dream_and_the_puzzles_and_nothing_about_cues(self):
        session, catalog, dream = night()
        packet = export_blind_coding_packet(session, dream, catalog, generated_at_utc="2026-11-04T08:00:00+00:00")
        payload = json.loads(json.dumps(packet.to_dict()))
        self.assertEqual(leaks(payload), [])
        self.assertEqual(leaks({"arm": 1, "x": ["no_cue"], "y": "alarm"}), [("key", ("arm",)), ("value", ("x", 0))])
        self.assertEqual([puzzle["blind_index"] for puzzle in payload["puzzles"]], list(range(1, 9)))
        self.assertEqual([puzzle["puzzle_id"] for puzzle in payload["puzzles"]], list(PUZZLES))
        self.assertEqual(payload["dream_text"], dream.dream_text)
        self.assertEqual(sorted(payload["puzzles"][0]), ["blind_index", "prompt", "puzzle_id"])
        text = json.dumps(payload) + packet.to_markdown()
        for leak in ("cued", "cue-p", "Answer", "heard the tone", "the card"):
            self.assertNotIn(leak, text)

    def test_the_packet_is_the_same_whatever_was_cued(self):
        """It takes no assignment and no schedule; the order is the session's."""
        session, catalog, dream = night()
        first = export_blind_coding_packet(session, dream, catalog, generated_at_utc="t").to_dict()
        for cued in (PUZZLES[:4], PUZZLES[4:], PUZZLES[::2]):
            PuzzleCueAssignment(
                session_id="night-003",
                cued_puzzle_ids=cued,
                uncued_puzzle_ids=tuple(p for p in PUZZLES if p not in cued),
                seed=1,
            )
            self.assertEqual(export_blind_coding_packet(session, dream, catalog, generated_at_utc="t").to_dict(), first)

    def test_round_trip_and_a_dream_of_another_night(self):
        session, catalog, dream = night()
        packet = export_blind_coding_packet(session, dream, catalog)
        self.assertEqual(BlindCodingPacket.from_dict(packet.to_dict()), packet)
        other = DreamReport(session_id="night-004", lucid=False, cues_heard=False, confidence=0.5, dream_text="x")
        with self.assertRaises(ValueError):
            export_blind_coding_packet(session, other, catalog)


class TestCodes(unittest.TestCase):
    def test_every_puzzle_is_coded_and_categories_are_checked(self):
        session, catalog, dream = night()
        packet = export_blind_coding_packet(session, dream, catalog)
        sheet = build_incorporation_coding(packet, "rater-a", codes_of(RATER_A), notes={"p1": "the door"})
        self.assertEqual(sheet.coding_for("p1").categories, ("sound_cue", "direct_conceptual"))
        self.assertFalse(sheet.coding_for("p4").incorporated)
        self.assertEqual(sheet.coding_for("p1").notes, "the door")
        self.assertEqual(IncorporationCodingSheet.from_dict(sheet.to_dict()), sheet)
        sheet.validate_against_session(session)
        partial = dict(codes_of(RATER_A))
        partial.pop("p8")
        with self.assertRaises(ValueError):
            build_incorporation_coding(packet, "rater-a", partial)
        with self.assertRaises(ValueError):
            parse_coding_categories("sound_cue,dream_sound")
        self.assertEqual(parse_coding_categories("none"), ())
        self.assertEqual(parse_coding_categories("indirect_conceptual, sound_cue"), ("indirect_conceptual", "sound_cue"))
        with self.assertRaises(ValueError):
            IncorporationCodingSheet(session_id="night-003", rater_id="a", codings=(IncorporationCoding("b", "p1"),))

    def test_more_than_half_of_the_raters_make_a_puzzle_incorporated(self):
        session, catalog, dream = night()
        packet = export_blind_coding_packet(session, dream, catalog)
        a = build_incorporation_coding(packet, "rater-a", codes_of(RATER_A))
        b = build_incorporation_coding(packet, "rater-b", codes_of(RATER_B))
        both = rater_incorporation((a, b), session)
        self.assertTrue(both["p1"]["incorporated"])
        self.assertFalse(both["p2"]["incorporated"])  # only rater-a
        self.assertTrue(both["p2"]["split"])
        self.assertEqual(both["p1"]["sound_cue"], 1.0)
        self.assertEqual(both["p5"]["sound_cue"], 0.5)
        alone = rater_incorporation((a,), session)
        self.assertTrue(alone["p2"]["incorporated"])
        c = build_incorporation_coding(packet, "rater-c", codes_of(RATER_A))
        self.assertTrue(rater_incorporation((a, b, c), session)["p2"]["incorporated"])


class TestAgreement(unittest.TestCase):
    def test_kappa_by_hand(self):
        first = (1, 1, 0, 0, 1, 0, 0, 0)
        second = (1, 0, 0, 0, 1, 0, 1, 0)
        # Observed 6/8; chance 3/8 * 3/8 + 5/8 * 5/8 = 34/64; kappa (48 - 34) / (64 - 34) = 7/15.
        self.assertAlmostEqual(cohens_kappa(first, second), 7 / 15)
        self.assertEqual(cohens_kappa((1, 0, 1), (1, 0, 1)), 1.0)
        self.assertIsNone(cohens_kappa((0, 0, 0), (0, 0, 0)))
        self.assertIsNone(cohens_kappa((), ()))

    def test_two_raters_of_a_synthetic_packet(self):
        session, catalog, dream = night()
        packet = export_blind_coding_packet(session, dream, catalog)
        agreement = coding_agreement(
            (
                build_incorporation_coding(packet, "rater-a", codes_of(RATER_A)),
                build_incorporation_coding(packet, "rater-b", codes_of(RATER_B)),
            )
        )
        stats = agreement.categories
        self.assertEqual(agreement.rater_ids, ("rater-a", "rater-b"))
        self.assertEqual(agreement.item_count, 8)
        self.assertAlmostEqual(stats["direct_conceptual"]["kappa"], 7 / 15)
        self.assertEqual(stats["direct_conceptual"]["percent_agreement"], 0.75)
        self.assertAlmostEqual(stats["sound_cue"]["kappa"], 0.6)
        self.assertEqual(stats["indirect_conceptual"]["kappa"], 1.0)
        self.assertAlmostEqual(stats["any"]["kappa"], 0.75)
        self.assertEqual(stats["any"]["yes_counts"], {"rater-a": 5, "rater-b": 4})

    def test_agreement_needs_two_raters_and_lists_unmatched_puzzles(self):
        session, catalog, dream = night()
        packet = export_blind_coding_packet(session, dream, catalog)
        a = build_incorporation_coding(packet, "rater-a", codes_of(RATER_A))
        with self.assertRaises(ValueError):
            coding_agreement((a,))
        b = build_incorporation_coding(packet, "rater-b", codes_of(RATER_B))
        trimmed = IncorporationCodingSheet(session_id=b.session_id, rater_id="rater-b", codings=b.codings[:6])
        agreement = coding_agreement((a, trimmed))
        self.assertEqual(agreement.item_count, 6)
        self.assertEqual([item["puzzle_id"] for item in agreement.unmatched], ["p7", "p8"])


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_cli(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli_main.main([str(item) for item in argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_export_record_and_compare(self):
        session, catalog, dream = night()
        session.save(self.tmp / "night-003_puzzles.json")
        catalog.save(self.tmp / "catalog.json")
        dream.save(self.tmp / "night-003_dream_report.json")
        packet_path = self.tmp / "night-003_blind_coding_packet.json"
        code, out, _ = self.run_cli(
            "export-blind-coding", self.tmp / "night-003_puzzles.json",
            "--dream-report", self.tmp / "night-003_dream_report.json",
            "--catalog", self.tmp / "catalog.json",
            "--output", packet_path,
            "--markdown-output", self.tmp / "packet.md",
        )
        self.assertEqual(code, 0, out)
        self.assertEqual(leaks(json.loads(packet_path.read_text(encoding="utf-8"))), [])
        self.assertIn("1. `p1`: Prompt p1", (self.tmp / "packet.md").read_text(encoding="utf-8"))
        for rater, table in (("rater-a", RATER_A), ("rater-b", RATER_B)):
            flags = []
            for puzzle, categories in codes_of(table).items():
                flags += ["--code", f"{puzzle}={','.join(categories) or 'none'}"]
            code, out, err = self.run_cli(
                "record-incorporation-coding", packet_path, "--rater-id", rater, *flags,
                "--output", self.tmp / f"night-003_coding_{rater}.json",
            )
            self.assertEqual(code, 0, err)
        self.assertEqual(load_incorporation_coding(self.tmp / "night-003_coding_rater-a.json").rater_id, "rater-a")
        code, out, _ = self.run_cli(
            "compare-incorporation-codings",
            self.tmp / "night-003_coding_rater-a.json",
            self.tmp / "night-003_coding_rater-b.json",
            "--output", self.tmp / "agreement.json",
        )
        self.assertEqual(code, 0)
        self.assertIn("direct_conceptual: agreement=0.750 kappa=0.467", out)
        payload = json.loads((self.tmp / "agreement.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(payload["categories"]["sound_cue"]["kappa"], 0.6)

    def test_a_puzzle_left_uncoded_exits_2(self):
        session, catalog, dream = night()
        packet = export_blind_coding_packet(session, dream, catalog)
        packet.save(self.tmp / "packet.json")
        code, _, err = self.run_cli(
            "record-incorporation-coding", self.tmp / "packet.json", "--rater-id", "a",
            "--code", "p1=sound_cue", "--output", self.tmp / "coding.json",
        )
        self.assertEqual(code, 2)
        self.assertIn("missing", err)
        self.assertFalse((self.tmp / "coding.json").exists())


if __name__ == "__main__":
    unittest.main()
