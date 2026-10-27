"""Sealed night schedules for N-of-1 series of cue, sham and no-cue nights (#228)."""

import ast
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest import mock

from muse_tmr.audio import DEFAULT_SHAM_CUE_ID
from muse_tmr.cli import main as cli_main
from muse_tmr.protocol import (
    BLOCK_ARMS,
    CUEING_ARMS,
    NIGHT_ARMS,
    NightArmContext,
    NightSchedule,
    SealMismatch,
    WashoutNight,
    generate_night_schedule,
    load_night_schedule,
    load_sealed_commitment,
    night_arm_from_schedule,
    sealed_path_for,
    unseal_night_schedule,
    write_night_schedule,
)
from muse_tmr.protocol.night_schedule import canonical_json
from muse_tmr.protocol.protocol_presets import canonical_json as preset_canonical_json

SRC = Path(__file__).resolve().parents[1] / "src" / "muse_tmr"
SALT = "0123456789abcdef" * 2


def _arms_by_block(schedule):
    blocks = {}
    for night in schedule.nights:
        if night.block is not None:
            blocks.setdefault(night.block, []).append(night.arm)
    return blocks


class TestGenerateNightSchedule(unittest.TestCase):
    def test_one_seed_gives_one_order_and_a_new_salt(self):
        first = generate_night_schedule(11, 4, 1, schedule_id="series-a")
        second = generate_night_schedule(11, 4, 1, schedule_id="series-a")
        other = generate_night_schedule(12, 4, 1, schedule_id="series-a")

        self.assertEqual([n.arm for n in first.nights], [n.arm for n in second.nights])
        self.assertNotEqual(first.salt, second.salt)
        self.assertEqual(len(first.salt), 32)
        orders = {tuple(n.arm for n in generate_night_schedule(seed, 4, 1, schedule_id="s").nights) for seed in range(20)}
        self.assertGreater(len(orders), 1)
        self.assertIn(tuple(n.arm for n in other.nights), orders)

    def test_every_block_holds_each_arm_once_with_washout_nights_between(self):
        schedule = generate_night_schedule(3, 2, 1, schedule_id="series-a", salt=SALT)

        self.assertEqual(schedule.night_ids, tuple(f"night-00{index}" for index in range(1, 8)))
        self.assertEqual([n.block for n in schedule.nights], [1, 1, 1, None, 2, 2, 2])
        self.assertEqual(schedule.nights[3].arm, "washout")
        for arms in _arms_by_block(schedule).values():
            self.assertEqual(sorted(arms), sorted(BLOCK_ARMS))
        no_washout = generate_night_schedule(3, 2, schedule_id="series-b")
        self.assertEqual(len(no_washout.nights), 6)
        self.assertNotIn("washout", [n.arm for n in no_washout.nights])

    def test_within_three_blocks_each_arm_takes_each_place_once(self):
        for seed in range(25):
            schedule = generate_night_schedule(seed, 6, 1, schedule_id="series-a")
            blocks = _arms_by_block(schedule)
            for group in ((1, 2, 3), (4, 5, 6)):
                for place in range(3):
                    self.assertEqual(
                        sorted(blocks[block][place] for block in group), sorted(BLOCK_ARMS), (seed, group, place)
                    )
            # Two blocks are two rows of one square: no arm in the same place twice.
            two = _arms_by_block(generate_night_schedule(seed, 2, schedule_id="series-b"))
            self.assertTrue(all(two[1][place] != two[2][place] for place in range(3)), seed)

    def test_the_first_place_is_spread_over_the_arms(self):
        firsts = Counter(
            _arms_by_block(generate_night_schedule(seed, 1, schedule_id="s"))[1][0] for seed in range(300)
        )
        self.assertEqual(set(firsts), set(BLOCK_ARMS))
        self.assertGreater(min(firsts.values()), 60)

    def test_night_ids_can_be_given(self):
        ids = [f"n{index}" for index in range(7)]
        schedule = generate_night_schedule(1, 2, 1, schedule_id="s", night_ids=ids)
        self.assertEqual(schedule.night_ids, tuple(ids))
        with self.assertRaises(ValueError):
            generate_night_schedule(1, 2, 1, schedule_id="s", night_ids=ids[:6])
        with self.assertRaises(ValueError):
            generate_night_schedule(1, 0)
        with self.assertRaises(ValueError):
            generate_night_schedule(1, 2, -1)

    def test_a_schedule_that_is_not_blocks_of_each_arm_is_refused(self):
        schedule = generate_night_schedule(5, 2, 1, schedule_id="s", salt=SALT)
        payload = schedule.to_dict()
        payload["nights"][0]["arm"], payload["nights"][1]["arm"] = "cue", "cue"
        payload["nights"][2]["arm"] = "cue"
        with self.assertRaises(ValueError):
            NightSchedule.from_dict(payload)
        with self.assertRaises(ValueError):
            replace(schedule, salt="short")
        with self.assertRaises(ValueError):
            replace(schedule, nights=schedule.nights[:3] + schedule.nights[4:])
        with self.assertRaises(ValueError):
            replace(schedule, nights=(schedule.nights[0],) * 7)

    def test_the_arms(self):
        self.assertEqual(NIGHT_ARMS, ("cue", "sham", "no_cue", "washout"))
        self.assertEqual(CUEING_ARMS, BLOCK_ARMS)
        self.assertEqual(cli_main.NIGHT_ARM_CHOICES, CUEING_ARMS)


class TestSealing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.schedule = generate_night_schedule(9, 2, 1, schedule_id="series-a", salt=SALT)
        self.path, self.sealed = write_night_schedule(self.schedule, self.root / "series-a.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_commitment_is_the_sha256_of_canonical_json_and_salt(self):
        committed = self.schedule.committed_payload()
        text = json.dumps(committed, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + SALT
        self.assertEqual(self.schedule.commitment_sha256(), hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertEqual(canonical_json(committed), preset_canonical_json(committed))
        self.assertNotIn("salt", committed)
        self.assertNotIn("unsealed_at_utc", committed)

    def test_the_companion_holds_only_the_commitment(self):
        self.assertEqual(self.sealed, sealed_path_for(self.path))
        self.assertEqual(self.sealed.name, "series-a.sealed.json")
        companion = json.loads(self.sealed.read_text(encoding="utf-8"))
        self.assertEqual(companion, {"commitment_sha256": self.schedule.commitment_sha256()})
        self.assertEqual(load_night_schedule(self.path), self.schedule)

    def test_a_schedule_is_never_replaced(self):
        with self.assertRaises(FileExistsError):
            write_night_schedule(generate_night_schedule(10, 2, 1, schedule_id="series-a"), self.path)
        self.assertEqual(load_night_schedule(self.path), self.schedule)

    def test_any_change_breaks_the_seal(self):
        swapped = list(self.schedule.nights)
        swapped[0], swapped[1] = replace(swapped[0], arm=swapped[1].arm), replace(swapped[1], arm=swapped[0].arm)
        changes = (
            replace(self.schedule, nights=tuple(swapped)),
            replace(self.schedule, salt="f" * 32),
            replace(self.schedule, seed=10),
            replace(self.schedule, sham_cue_id="another_sound"),
        )
        for changed in changes:
            self.assertNotEqual(changed.commitment_sha256(), self.schedule.commitment_sha256())
        self.assertEqual(
            replace(self.schedule, unsealed_at_utc="2026-11-01T08:00:00+00:00").commitment_sha256(),
            self.schedule.commitment_sha256(),
        )

    def test_unsealing_checks_the_seal_and_records_the_first_time(self):
        unsealed = unseal_night_schedule(self.path, now="2026-11-01T08:00:00+00:00")
        again = unseal_night_schedule(self.path, now="2026-11-02T08:00:00+00:00")

        self.assertEqual(unsealed.unsealed_at_utc, "2026-11-01T08:00:00+00:00")
        self.assertEqual(again.unsealed_at_utc, "2026-11-01T08:00:00+00:00")
        self.assertEqual(load_night_schedule(self.path).unsealed_at_utc, "2026-11-01T08:00:00+00:00")
        self.assertFalse(load_night_schedule(self.path).sealed)
        self.assertEqual(load_sealed_commitment(self.sealed), unsealed.commitment_sha256())

    def test_a_changed_schedule_is_not_unsealed(self):
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        first, second = payload["nights"][0], payload["nights"][1]
        first["arm"], second["arm"] = second["arm"], first["arm"]
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        before = self.path.read_text(encoding="utf-8")

        with self.assertRaises(SealMismatch):
            unseal_night_schedule(self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        with self.assertRaises(SealMismatch):
            night_arm_from_schedule(self.path, "night-001")

    def test_a_night_runs_the_arm_its_schedule_gives_it(self):
        for night in self.schedule.nights:
            if night.arm == "washout":
                with self.assertRaises(WashoutNight):
                    night_arm_from_schedule(self.path, night.night_id)
                continue
            context = night_arm_from_schedule(self.path, night.night_id)
            self.assertEqual(context.arm, night.arm)
            self.assertEqual(context.sham_cue_id, DEFAULT_SHAM_CUE_ID)
            self.assertEqual(context.schedule_commitment_sha256, self.schedule.commitment_sha256())
            self.assertTrue(context.from_schedule)
            self.assertEqual(context.schedule_check(), {"commitment_matches": True, "arm_matches": True})
            self.assertEqual(
                context.summary_fields(),
                {
                    "arm": night.arm,
                    "night_id": night.night_id,
                    "schedule_commitment_sha256": self.schedule.commitment_sha256(),
                },
            )
        with self.assertRaises(KeyError):
            night_arm_from_schedule(self.path, "night-099")
        other = night_arm_from_schedule(self.path, "night-001", sham_cue_id="quiet_chime")
        self.assertEqual(other.sham_cue_id, "quiet_chime")

    def test_a_schedule_changed_during_the_night_fails_the_check(self):
        context = night_arm_from_schedule(self.path, "night-001")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["seed"] = 99
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(context.schedule_check(), {"commitment_matches": False, "arm_matches": True})
        self.sealed.unlink()
        check = context.schedule_check()
        self.assertFalse(check["commitment_matches"])
        self.assertIn("error", check)


class TestNightArmContext(unittest.TestCase):
    def test_a_cueing_night_runs_one_of_three_arms(self):
        with self.assertRaises(ValueError):
            NightArmContext(arm="washout")
        with self.assertRaises(ValueError):
            NightArmContext(arm="placebo")
        with self.assertRaises(ValueError):
            NightArmContext(arm="sham")
        with self.assertRaises(ValueError):
            NightArmContext(arm="cue", schedule_path="s.json", night_id="n1")
        open_label = NightArmContext(arm="sham", sham_cue_id="sham_soft_tone")
        self.assertFalse(open_label.from_schedule)
        self.assertTrue(open_label.plays_sound)
        self.assertFalse(NightArmContext(arm="no_cue").plays_sound)
        self.assertEqual(
            open_label.summary_fields(), {"arm": "sham", "night_id": None, "schedule_commitment_sha256": None}
        )
        self.assertFalse(open_label.schedule_check()["commitment_matches"])


class TestScheduleCommands(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.path = self.root / "series.json"

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_planning_prints_neither_an_arm_nor_the_seed(self):
        code, out, err = self.run_cli(
            "plan-night-schedule", "--blocks", "2", "--washout-nights", "1", "--seed", "424242",
            "--schedule-id", "series-x", "--output", str(self.path),
        )
        schedule = load_night_schedule(self.path)

        self.assertEqual((code, err), (0, ""))
        self.assertIn(f"commitment_sha256={schedule.commitment_sha256()}", out)
        self.assertIn("night-004(washout)", out)
        self.assertNotIn("424242", out)
        for night in schedule.nights:
            if night.arm != "washout":
                self.assertNotIn(f"{night.night_id}({night.arm})", out)
        nights_line = next(line for line in out.splitlines() if line.startswith("nights="))
        self.assertEqual(nights_line.count("("), 1)
        self.assertEqual(schedule.seed, 424242)
        self.assertTrue(sealed_path_for(self.path).is_file())

    def test_without_a_seed_one_is_drawn_and_kept_in_the_file(self):
        with mock.patch("secrets.randbits", return_value=3141592653) as draw:
            code, out, _ = self.run_cli("plan-night-schedule", "--blocks", "1", "--output", str(self.path))
        schedule = load_night_schedule(self.path)
        self.assertEqual(code, 0)
        draw.assert_called_once_with(32)
        self.assertEqual(schedule.seed, 3141592653)
        self.assertNotIn("3141592653", out)
        self.assertEqual(self.run_cli("plan-night-schedule", "--blocks", "1", "--output", str(self.path))[0], 2)

    def test_show_hides_the_arms_until_unsealed_and_unseal_records_it(self):
        self.run_cli("plan-night-schedule", "--blocks", "2", "--washout-nights", "1", "--seed", "5", "--output", str(self.path))
        schedule = load_night_schedule(self.path)

        code, out, _ = self.run_cli("show-night-schedule", str(self.path), "--json")
        shown = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(shown["seal"], "matches")
        self.assertIsNone(shown["unsealed_at_utc"])
        self.assertEqual(
            [night["arm"] for night in shown["nights"]],
            ["washout" if night.arm == "washout" else "sealed" for night in schedule.nights],
        )
        code, text, _ = self.run_cli("show-night-schedule", str(self.path))
        self.assertEqual(code, 0)
        night_lines = [line for line in text.splitlines() if "night-" in line and "block" in line]
        self.assertEqual(len(night_lines), 7)
        for line in night_lines:
            self.assertTrue(line.endswith(("sealed", "washout")), line)
        self.assertTrue(load_night_schedule(self.path).sealed)

        code, out, _ = self.run_cli("unseal-night-schedule", str(self.path), "--json")
        unsealed = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual([night["arm"] for night in unsealed["nights"]], [night.arm for night in schedule.nights])
        self.assertIsNotNone(unsealed["unsealed_at_utc"])
        self.assertEqual(load_night_schedule(self.path).unsealed_at_utc, unsealed["unsealed_at_utc"])
        code, out, _ = self.run_cli("show-night-schedule", str(self.path), "--json")
        self.assertEqual([night["arm"] for night in json.loads(out)["nights"]], [night.arm for night in schedule.nights])

    def test_a_changed_schedule_fails_show_and_unseal(self):
        self.run_cli("plan-night-schedule", "--blocks", "1", "--seed", "5", "--output", str(self.path))
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["created_at_utc"] = "2020-01-01T00:00:00+00:00"
        self.path.write_text(json.dumps(payload), encoding="utf-8")

        code, out, _ = self.run_cli("show-night-schedule", str(self.path), "--json")
        self.assertEqual((code, json.loads(out)["seal"]), (1, "does_not_match"))
        code, _, err = self.run_cli("unseal-night-schedule", str(self.path))
        self.assertEqual(code, 1)
        self.assertIn("does not match the commitment", err)
        self.assertTrue(load_night_schedule(self.path).sealed)


class TestLayering(unittest.TestCase):
    def test_no_model_or_feature_module_imports_audio(self):
        offenders = []
        for folder in ("models", "features"):
            for path in sorted((SRC / folder).rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        names = [module] + [f"{module}.{alias.name}" for alias in node.names]
                    if any(name == "muse_tmr.audio" or name.startswith("muse_tmr.audio.") for name in names):
                        offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_the_night_schedule_imports_neither_the_scheduler_nor_audio_playback(self):
        tree = ast.parse((SRC / "protocol" / "night_schedule.py").read_text(encoding="utf-8"))
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertNotIn("muse_tmr.protocol.tmr_scheduler", modules)
        self.assertNotIn("muse_tmr.audio.audio_player", modules)


if __name__ == "__main__":
    unittest.main()
