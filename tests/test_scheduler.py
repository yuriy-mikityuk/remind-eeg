import unittest
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Tuple

from muse_tmr.audio import AudioPlaybackConfig, CueLibrary, CueMetadata, VolumeCalibration
from muse_tmr.models import (
    RemConfidence,
    RemGateConfig,
    RemGateDecision,
    RemPrediction,
    StableRemGate,
)
from muse_tmr.protocol import (
    ArousalGuardDecision,
    PuzzleCatalog,
    PuzzleCueAssignment,
    PuzzleTask,
    TlrBlockConfig,
    default_tlr_cue,
    plan_tlr_block,
)
from muse_tmr.protocol.tmr_scheduler import (
    CueDecision,
    TmrCueScheduler,
    TmrSchedulerConfig,
    TmrSchedulerEvent,
    arousal_guard_decision,
    calibrated_cue_decision,
    ensure_sham_untrained,
    load_tmr_scheduler_events,
    tlr_cue_ids_of,
    validate_sham_cue,
)


class TestSchedulerContracts(unittest.TestCase):
    def test_cue_decision_records_block_reason(self):
        decision = CueDecision(should_play=False, reason_codes=("not_rem",))

        self.assertFalse(decision.should_play)
        self.assertEqual(decision.reason_codes, ("not_rem",))

    def test_arousal_guard_decision_blocks_when_reasons_exist(self):
        decision = arousal_guard_decision(("motion_arousal_proxy", "motion_arousal_proxy"))

        self.assertFalse(decision.should_play)
        self.assertEqual(decision.reason_codes, ("motion_arousal_proxy",))

    def test_arousal_guard_decision_allows_when_clear(self):
        decision = arousal_guard_decision(())

        self.assertTrue(decision.should_play)
        self.assertEqual(decision.reason_codes, ())

    def test_calibrated_cue_decision_uses_calibrated_max_volume(self):
        calibration = VolumeCalibration(
            device_name="Bedroom Headphones",
            detectable_volume=0.02,
            identifiable_volume=0.04,
            comfortable_volume=0.08,
        )

        decision = calibrated_cue_decision((), calibration=calibration, fallback_max_volume=0.20)

        self.assertTrue(decision.should_play)
        self.assertEqual(decision.max_volume, 0.08)
        self.assertEqual(decision.calibration_device_name, "Bedroom Headphones")

    def test_calibrated_cue_decision_blocks_without_calibration(self):
        decision = calibrated_cue_decision((), calibration=None, fallback_max_volume=0.20)

        self.assertFalse(decision.should_play)
        self.assertEqual(decision.max_volume, 0.20)
        self.assertEqual(decision.reason_codes, ("volume_calibration_missing",))

    def test_calibrated_cue_decision_keeps_arousal_block_reasons(self):
        calibration = VolumeCalibration("Bedroom", 0.02, 0.04, 0.08)

        decision = calibrated_cue_decision(
            ("motion_arousal_proxy", "motion_arousal_proxy"),
            calibration=calibration,
            fallback_max_volume=0.20,
        )

        self.assertFalse(decision.should_play)
        self.assertEqual(decision.max_volume, 0.08)
        self.assertEqual(decision.reason_codes, ("motion_arousal_proxy",))


class TestTmrCueScheduler(unittest.TestCase):
    def test_scheduler_uses_only_cued_puzzle_ids(self):
        scheduler = _scheduler(tlr_block_plan=None)

        event = scheduler.update(_gate_open(), timestamp_seconds=0.0)[0]

        self.assertEqual(event.event_type, "play")
        self.assertEqual(event.protocol, "puzzle")
        self.assertEqual(event.puzzle_id, "p1")
        self.assertEqual(scheduler.scheduled_puzzle_ids, ("p1", "p3"))
        self.assertNotIn("p2", scheduler.scheduled_puzzle_ids)

    def test_tlr_block_runs_before_first_puzzle_cue(self):
        tlr_plan = plan_tlr_block(
            default_tlr_cue(),
            config=TlrBlockConfig(
                repetitions=2,
                interval_seconds=5.0,
                post_block_pause_seconds=7.0,
            ),
        )
        scheduler = _scheduler(tlr_block_plan=tlr_plan)

        first_tlr = scheduler.update(_gate_open(), timestamp_seconds=100.0)[0]
        second_tlr = scheduler.update(_gate_open(), timestamp_seconds=105.0)[0]
        early = scheduler.update(_gate_open(), timestamp_seconds=110.0)[0]
        puzzle = scheduler.update(_gate_open(), timestamp_seconds=113.0)[0]

        self.assertEqual(first_tlr.protocol, "tlr")
        self.assertEqual(first_tlr.timestamp_seconds, 100.0)
        self.assertIn("rem_gate_open", first_tlr.reason_codes)
        self.assertIn("tlr_block", first_tlr.reason_codes)
        self.assertEqual(second_tlr.protocol, "tlr")
        self.assertEqual(second_tlr.timestamp_seconds, 105.0)
        self.assertIn("rem_gate_open", second_tlr.reason_codes)
        self.assertIn("tlr_block", second_tlr.reason_codes)
        self.assertEqual(early.event_type, "skip")
        self.assertIn("cue_interval_active", early.reason_codes)
        self.assertEqual(puzzle.event_type, "play")
        self.assertEqual(puzzle.protocol, "puzzle")
        self.assertEqual(puzzle.puzzle_id, "p1")

    def test_closed_gate_cancels_remaining_tlr_block_events(self):
        tlr_plan = plan_tlr_block(
            default_tlr_cue(),
            config=TlrBlockConfig(
                repetitions=2,
                interval_seconds=5.0,
                post_block_pause_seconds=7.0,
            ),
        )
        scheduler = _scheduler(tlr_block_plan=tlr_plan)

        first_tlr = scheduler.update(_gate_open(), timestamp_seconds=100.0)[0]
        pause = scheduler.update(_gate_closed("below_exit_threshold"), timestamp_seconds=102.0)[0]

        self.assertEqual(first_tlr.event_type, "play")
        self.assertEqual(first_tlr.protocol, "tlr")
        self.assertEqual(pause.event_type, "pause")
        self.assertEqual(
            [event.protocol for event in scheduler.events if event.event_type == "play"],
            ["tlr"],
        )

    def test_interval_max_per_block_and_cooldown_are_enforced(self):
        scheduler = _scheduler(
            tlr_block_plan=None,
            config=TmrSchedulerConfig(
                puzzle_cue_interval_seconds=30.0,
                cooldown_seconds=60.0,
                max_puzzle_cues_per_block=1,
                enable_tlr_block=False,
            ),
        )

        first = scheduler.update(_gate_open(), timestamp_seconds=0.0)[0]
        interval_skip = scheduler.update(_gate_open(), timestamp_seconds=10.0)[0]
        max_skip = scheduler.update(_gate_open(), timestamp_seconds=30.0)[0]
        cooldown_skip = scheduler.update(_gate_open(), timestamp_seconds=40.0)[0]

        self.assertEqual(first.event_type, "play")
        self.assertIn("cue_interval_active", interval_skip.reason_codes)
        self.assertIn("max_puzzle_cues_per_block_reached", max_skip.reason_codes)
        self.assertIn("scheduler_cooldown_active", cooldown_skip.reason_codes)

    def test_closed_gate_pauses_active_block_and_logs_cooldown(self):
        scheduler = _scheduler(tlr_block_plan=None)

        scheduler.update(_gate_open(), timestamp_seconds=0.0)
        pause = scheduler.update(_gate_closed("below_exit_threshold"), timestamp_seconds=5.0)[0]
        cooldown = scheduler.update(_gate_open(), timestamp_seconds=10.0)[0]

        self.assertEqual(pause.event_type, "pause")
        self.assertIn("rem_gate_closed", pause.reason_codes)
        self.assertIn("below_exit_threshold", pause.reason_codes)
        self.assertEqual(cooldown.event_type, "skip")
        self.assertIn("scheduler_cooldown_active", cooldown.reason_codes)

    def test_guard_reasons_skip_without_playing(self):
        scheduler = _scheduler(tlr_block_plan=None)

        event = scheduler.update(
            _gate_open(),
            timestamp_seconds=0.0,
            reason_codes=("arousal_guard_blocked",),
        )[0]

        self.assertEqual(event.event_type, "skip")
        self.assertIn("arousal_guard_blocked", event.reason_codes)
        self.assertEqual(len([item for item in scheduler.events if item.event_type == "play"]), 0)

    def test_lower_volume_guard_adds_its_multiplier_and_keeps_the_hint(self):
        scheduler = _scheduler(tlr_block_plan=None)

        event = scheduler.update(
            _gate_open(),
            timestamp_seconds=0.0,
            guard_decision=_lower_volume(),
        )[0]

        self.assertEqual(event.event_type, "play")
        self.assertEqual(event.metadata["arousal_guard_action"], "lower_volume")
        self.assertEqual(event.metadata["arousal_guard_reason_codes"], ["alpha_arousal_proxy_mild"])
        self.assertEqual(event.metadata["volume_multiplier"], 0.5)
        # The player applies the multiplier after its cap (#255), so the hint stays the cue's.
        self.assertEqual(event.metadata["volume_hint"], 0.10)
        self.assertNotIn("original_volume_hint", event.metadata)
        # Without the player's config the scheduler cannot say what will play.
        self.assertNotIn("effective_volume", event.metadata)

    def test_play_events_log_the_volume_the_player_plays(self):
        tlr_plan = plan_tlr_block(
            default_tlr_cue(),
            config=TlrBlockConfig(repetitions=2, interval_seconds=5.0, post_block_pause_seconds=0.0),
        )
        scheduler = _scheduler(
            tlr_block_plan=tlr_plan,
            playback_config=AudioPlaybackConfig(max_volume=0.06, default_volume=0.02),
        )

        tlr_allowed = scheduler.update(_gate_open(), timestamp_seconds=0.0)[0]
        tlr_lowered = scheduler.update(
            _gate_open(), timestamp_seconds=5.0, guard_decision=_lower_volume()
        )[0]
        puzzle_lowered = scheduler.update(
            _gate_open(), timestamp_seconds=6.0, guard_decision=_lower_volume()
        )[0]
        puzzle_allowed = scheduler.update(_gate_open(), timestamp_seconds=16.0)[0]

        def volumes(event):
            return tuple(
                event.metadata[key]
                for key in ("requested_volume", "capped_volume", "volume_multiplier", "effective_volume")
            )

        # The TLR hint 0.05 sits below the 0.06 cap, the puzzle hint 0.10 above it.
        self.assertEqual((tlr_allowed.protocol, tlr_lowered.protocol), ("tlr", "tlr"))
        self.assertEqual(volumes(tlr_allowed), (0.05, 0.05, 1.0, 0.05))
        self.assertEqual(volumes(tlr_lowered), (0.05, 0.05, 0.5, 0.025))
        self.assertEqual((puzzle_lowered.puzzle_id, puzzle_allowed.puzzle_id), ("p1", "p3"))
        self.assertEqual(volumes(puzzle_lowered), (0.10, 0.06, 0.5, 0.03))
        self.assertEqual(volumes(puzzle_allowed), (0.10, 0.06, 1.0, 0.06))
        self.assertNotIn("arousal_guard_action", puzzle_allowed.metadata)
        self.assertEqual(puzzle_lowered.metadata["volume_hint"], 0.10)

    def test_pause_guard_pauses_cueing_and_starts_cooldown(self):
        scheduler = _scheduler(tlr_block_plan=None)
        guard_decision = ArousalGuardDecision(
            action="pause",
            timestamp_seconds=0.0,
            reason_codes=("motion_arousal_proxy",),
            pause_seconds=45.0,
        )

        pause = scheduler.update(
            _gate_open(),
            timestamp_seconds=0.0,
            guard_decision=guard_decision,
        )[0]
        cooldown = scheduler.update(_gate_open(), timestamp_seconds=10.0)[0]
        recovered = scheduler.update(_gate_open(), timestamp_seconds=46.0)[0]

        self.assertEqual(pause.event_type, "pause")
        self.assertIn("arousal_guard_pause", pause.reason_codes)
        self.assertIn("motion_arousal_proxy", pause.reason_codes)
        self.assertEqual(pause.metadata["cooldown_until_seconds"], 45.0)
        self.assertEqual(cooldown.event_type, "skip")
        self.assertIn("scheduler_cooldown_active", cooldown.reason_codes)
        self.assertNotIn("scheduler_stopped", cooldown.reason_codes)
        self.assertEqual(recovered.event_type, "play")
        self.assertNotIn("scheduler_stopped", recovered.reason_codes)

    def test_stop_guard_logs_stop_and_blocks_future_updates(self):
        scheduler = _scheduler(tlr_block_plan=None)
        guard_decision = ArousalGuardDecision(
            action="stop",
            timestamp_seconds=0.0,
            reason_codes=("repeated_arousal_guard_pause",),
            volume_multiplier=0.0,
        )

        stop = scheduler.update(
            _gate_open(),
            timestamp_seconds=0.0,
            guard_decision=guard_decision,
        )[0]
        after = scheduler.update(_gate_open(), timestamp_seconds=5.0)[0]

        self.assertEqual(stop.event_type, "stop")
        self.assertIn("arousal_guard_stop", stop.reason_codes)
        self.assertIn("repeated_arousal_guard_pause", stop.reason_codes)
        self.assertEqual(after.event_type, "skip")
        self.assertIn("scheduler_stopped", after.reason_codes)

    def test_stop_logs_stop_and_blocks_future_updates(self):
        scheduler = _scheduler(tlr_block_plan=None)

        stop = scheduler.stop(timestamp_seconds=12.0, reason_codes=("manual_stop",))
        after = scheduler.update(_gate_open(), timestamp_seconds=13.0)[0]

        self.assertEqual(stop.event_type, "stop")
        self.assertIn("manual_stop", stop.reason_codes)
        self.assertEqual(after.event_type, "skip")
        self.assertIn("scheduler_stopped", after.reason_codes)

    def test_scheduler_writes_and_loads_jsonl_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "scheduler.jsonl"
            scheduler = _scheduler(tlr_block_plan=None, event_log_path=log_path)
            scheduler.update(_gate_closed(), timestamp_seconds=0.0)
            scheduler.update(_gate_open(), timestamp_seconds=60.0)
            scheduler.stop(timestamp_seconds=90.0)

            events = load_tmr_scheduler_events(log_path)

        self.assertEqual([event.event_type for event in events], ["skip", "play", "stop"])


class TestSchedulerArms(unittest.TestCase):
    """cue, sham and no_cue (#228): the arm changes what a cue becomes, never when."""

    def _night(self, arm, *, log_path=None):
        tlr_plan = plan_tlr_block(
            default_tlr_cue(),
            config=TlrBlockConfig(repetitions=2, interval_seconds=5.0, post_block_pause_seconds=0.0),
        )
        scheduler = _scheduler(
            tlr_block_plan=tlr_plan,
            config=TmrSchedulerConfig(
                puzzle_cue_interval_seconds=10.0,
                cooldown_seconds=30.0,
                max_puzzle_cues_per_block=2,
                enable_tlr_block=True,
                arm=arm,
                sham_cue_id="sham_soft_tone" if arm == "sham" else None,
            ),
            playback_config=AudioPlaybackConfig(max_volume=0.06, default_volume=0.02),
            event_log_path=log_path,
        )
        events = []
        for timestamp, gate, guard in (
            (0.0, _gate_closed("not_rem"), None),
            (100.0, _gate_open(), None),
            (105.0, _gate_open(), _lower_volume()),
            (106.0, _gate_open(), None),
            (110.0, _gate_open(), None),
            (116.0, _gate_open(), None),
            (130.0, _gate_closed("below_exit_threshold"), None),
            (140.0, _gate_open(), None),
            (200.0, _gate_open(), None),
        ):
            events.extend(scheduler.update(gate, timestamp_seconds=timestamp, guard_decision=guard))
        return scheduler, events

    def test_sham_changes_only_the_cue_id_and_adds_its_reason(self):
        _, cue = self._night("cue")
        _, sham = self._night("sham")

        plays = [event for event in cue if event.event_type == "play"]
        # Two TLR cues (one lowered), two puzzle cues, then a new REM block's first TLR cue.
        self.assertEqual([(event.protocol, event.timestamp_seconds) for event in plays], [
            ("tlr", 100.0), ("tlr", 105.0), ("puzzle", 106.0), ("puzzle", 116.0), ("tlr", 200.0),
        ])
        self.assertEqual(len(cue), len(sham))
        for original, stand_in in zip(cue, sham):
            self.assertEqual(stand_in.arm, "sham")
            self.assertEqual(original.arm, "cue")
            if original.event_type != "play":
                self.assertEqual(replace(stand_in, arm="cue"), original)
                continue
            self.assertEqual(stand_in.cue_id, "sham_soft_tone")
            self.assertEqual(stand_in.reason_codes, original.reason_codes + ("sham_arm",))
            self.assertEqual(
                replace(stand_in, arm="cue", cue_id=original.cue_id, reason_codes=original.reason_codes), original
            )
        # Length and volume stay those of the cue each sham play stands in for.
        self.assertEqual(
            [event.metadata for event in sham if event.event_type == "play"], [event.metadata for event in plays]
        )

    def test_no_cue_logs_a_virtual_play_where_the_cue_arm_plays(self):
        _, cue = self._night("cue")
        _, quiet = self._night("no_cue")

        self.assertEqual([event for event in quiet if event.event_type == "play"], [])
        self.assertEqual(len(cue), len(quiet))
        for original, virtual in zip(cue, quiet):
            self.assertEqual(virtual.arm, "no_cue")
            if original.event_type != "play":
                self.assertEqual(replace(virtual, arm="cue"), original)
                self.assertFalse(virtual.virtual_play)
                continue
            self.assertTrue(virtual.virtual_play)
            self.assertEqual(virtual.event_type, "skip")
            self.assertEqual(virtual.reason_codes, original.reason_codes + ("no_cue_arm",))
            self.assertEqual(
                (virtual.timestamp_seconds, virtual.cue_id, virtual.protocol, virtual.puzzle_id),
                (original.timestamp_seconds, original.cue_id, original.protocol, original.puzzle_id),
            )
            self.assertEqual(dict(virtual.metadata), {**dict(original.metadata), "virtual_play": True})

    def test_uncued_puzzles_stay_unschedulable_in_every_arm(self):
        for arm in ("cue", "sham", "no_cue"):
            with self.subTest(arm=arm):
                scheduler, events = self._night(arm)
                self.assertEqual(scheduler.scheduled_puzzle_ids, ("p1", "p3"))
                self.assertEqual({event.puzzle_id for event in events if event.puzzle_id}, {"p1", "p3"})
                for uncued in ("p2", "p4"):
                    with self.assertRaises(ValueError):
                        scheduler.assignment.ensure_schedulable(uncued)

    def test_a_no_cue_night_never_records_a_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "scheduler.jsonl"
            scheduler, _ = self._night("no_cue", log_path=log_path)
            before = log_path.read_text(encoding="utf-8")
            with self.assertRaises(RuntimeError):
                scheduler._record(TmrSchedulerEvent("play", 300.0, cue_id="cue-p1", protocol="puzzle", puzzle_id="p1"))
            self.assertEqual(log_path.read_text(encoding="utf-8"), before)
            loaded = load_tmr_scheduler_events(log_path)
        self.assertEqual({event.arm for event in loaded}, {"no_cue"})
        self.assertNotIn("play", {event.event_type for event in loaded})

    def test_the_log_leaves_the_cue_arm_out(self):
        _, cue = self._night("cue")
        _, sham = self._night("sham")
        self.assertTrue(all("arm" not in event.to_dict() for event in cue))
        self.assertTrue(all(event.to_dict()["arm"] == "sham" for event in sham))
        old = {"event_type": "play", "timestamp_seconds": 1.0, "cue_id": "cue-p1", "reason_codes": [], "metadata": {}}
        self.assertEqual(TmrSchedulerEvent.from_dict(old).arm, "cue")
        self.assertEqual([TmrSchedulerEvent.from_dict(event.to_dict()) for event in sham], list(sham))
        with self.assertRaises(ValueError):
            TmrSchedulerEvent("skip", 0.0, arm="washout")

    def test_the_arm_needs_a_runnable_config(self):
        with self.assertRaises(ValueError):
            TmrSchedulerConfig(arm="washout").validate()
        with self.assertRaises(ValueError):
            TmrSchedulerConfig(arm="sham").validate()
        TmrSchedulerConfig(arm="sham", sham_cue_id="sham_soft_tone").validate()
        with self.assertRaises(ValueError):
            _scheduler(config=TmrSchedulerConfig(arm="sham", sham_cue_id="cue-p2"))
        with self.assertRaises(KeyError):
            _scheduler(config=TmrSchedulerConfig(arm="sham", sham_cue_id="not-in-the-library"))

    def test_a_sham_cue_must_be_untrained(self):
        scheduler = _scheduler()
        catalog, library = scheduler.catalog, scheduler.cue_library
        tlr_plan = plan_tlr_block(default_tlr_cue())
        ensure_sham_untrained("sham_soft_tone", catalog=catalog, cue_library=library, tlr_cue_ids=tlr_cue_ids_of(tlr_plan))
        validate_sham_cue("sham_soft_tone", catalog=catalog, cue_library=library)
        self.assertEqual(tlr_cue_ids_of(tlr_plan), ("tlr_soft_tone",))
        self.assertEqual(tlr_cue_ids_of(None), ())
        refused = {
            "cue-p1": "the cue of puzzle p1",
            # An uncued puzzle's cue was trained before sleep too.
            "cue-p4": "the cue of puzzle p4",
            "tlr_soft_tone": "a TLR cue",
        }
        for cue_id, what in refused.items():
            with self.assertRaises(ValueError) as raised:
                ensure_sham_untrained(cue_id, catalog=catalog, tlr_cue_ids=tlr_cue_ids_of(tlr_plan))
            self.assertIn(what, str(raised.exception))
        with self.assertRaises(ValueError):
            ensure_sham_untrained("tlr_soft_tone", cue_library=library)
        lookalike = CueLibrary(cues=library.cues + (
            CueMetadata("echo", "generated_tone", 1.0, protocol="sham", frequency_hz=528.0),
        ))
        with self.assertRaises(ValueError) as raised:
            ensure_sham_untrained("echo", catalog=catalog, cue_library=lookalike)
        self.assertIn("sounds like cue-p1", str(raised.exception))
        generic = CueLibrary(cues=library.cues + (
            CueMetadata("plain", "generated_tone", 1.0, protocol="generic", frequency_hz=880.0),
        ))
        with self.assertRaises(ValueError):
            validate_sham_cue("plain", catalog=catalog, cue_library=generic)
        with self.assertRaises(ValueError):
            ensure_sham_untrained("")


class TestSchedulerReplayValidation(unittest.TestCase):
    def test_simulated_replay_does_not_play_without_stable_rem(self):
        events, decisions, scheduler = _run_scheduler_replay(
            [
                _tick(0.0, 0.20),
                _tick(30.0, 0.68),
                _tick(60.0, 0.40),
                _tick(90.0, 0.69),
            ]
        )

        _assert_puzzle_plays_are_eligible(self, events, decisions, scheduler)
        self.assertTrue(all(not decision.gate_open for decision in decisions))
        self.assertEqual([event.event_type for event in events], ["skip", "skip", "skip", "skip"])
        self.assertEqual(_puzzle_play_ids(events), ())

    def test_simulated_replay_plays_only_after_stable_rem(self):
        events, decisions, scheduler = _run_scheduler_replay(
            [
                _tick(0.0, 0.82),
                _tick(30.0, 0.84),
                _tick(60.0, 0.86),
            ]
        )

        _assert_puzzle_plays_are_eligible(self, events, decisions, scheduler)
        self.assertEqual([decision.gate_open for decision in decisions], [False, True, True])
        self.assertEqual(_puzzle_play_ids(events), ("p1", "p3"))
        self.assertEqual(scheduler.scheduled_puzzle_ids, ("p1", "p3"))
        self.assertTrue(set(_puzzle_play_ids(events)).issubset(scheduler.scheduled_puzzle_ids))

    def test_simulated_replay_motion_pause_blocks_cueing(self):
        events, decisions, scheduler = _run_scheduler_replay(
            [
                _tick(0.0, 0.82),
                _tick(
                    30.0,
                    0.84,
                    guard_decision=ArousalGuardDecision(
                        action="pause",
                        timestamp_seconds=30.0,
                        reason_codes=("motion_arousal_proxy",),
                        pause_seconds=45.0,
                    ),
                ),
                _tick(60.0, 0.86),
            ]
        )

        _assert_puzzle_plays_are_eligible(self, events, decisions, scheduler)
        self.assertTrue(decisions[1].gate_open)
        self.assertEqual(events[1].event_type, "pause")
        self.assertIn("arousal_guard_pause", events[1].reason_codes)
        self.assertIn("motion_arousal_proxy", events[1].reason_codes)
        self.assertEqual(events[2].event_type, "skip")
        self.assertIn("scheduler_cooldown_active", events[2].reason_codes)
        self.assertEqual(_puzzle_play_ids(events), ())

    def test_simulated_replay_rem_end_stops_active_block_before_next_cue(self):
        events, decisions, scheduler = _run_scheduler_replay(
            [
                _tick(0.0, 0.82),
                _tick(30.0, 0.84),
                _tick(60.0, 0.20),
                _tick(90.0, 0.86),
                _tick(120.0, 0.88),
            ]
        )

        _assert_puzzle_plays_are_eligible(self, events, decisions, scheduler)
        self.assertTrue(decisions[1].gate_open)
        self.assertFalse(decisions[2].gate_open)
        self.assertIn("below_exit_threshold", decisions[2].reason_codes)
        self.assertEqual(events[2].event_type, "pause")
        self.assertIn("rem_gate_closed", events[2].reason_codes)
        self.assertIn("below_exit_threshold", events[2].reason_codes)
        self.assertEqual(events[3].event_type, "skip")
        self.assertEqual(_puzzle_play_ids(events), ("p1", "p3"))

    def test_simulated_replay_never_plays_uncued_puzzles_across_blocks(self):
        events, decisions, scheduler = _run_scheduler_replay(
            [
                _tick(0.0, 0.82),
                _tick(30.0, 0.84),
                _tick(60.0, 0.86),
                _tick(90.0, 0.20),
                _tick(120.0, 0.88),
                _tick(150.0, 0.89),
                _tick(180.0, 0.90),
            ]
        )

        _assert_puzzle_plays_are_eligible(self, events, decisions, scheduler)
        uncued_puzzle_ids = {"p2", "p4"}
        self.assertEqual(scheduler.scheduled_puzzle_ids, ("p1", "p3"))
        self.assertFalse(uncued_puzzle_ids.intersection(_puzzle_play_ids(events)))
        self.assertTrue(set(_puzzle_play_ids(events)).issubset(scheduler.scheduled_puzzle_ids))


@dataclass(frozen=True)
class _ReplayTick:
    timestamp_seconds: float
    probability: float
    reason_codes: Tuple[str, ...] = ()
    guard_decision: Optional[ArousalGuardDecision] = None
    duration_seconds: float = 30.0


def _scheduler(config=None, tlr_block_plan=None, event_log_path=None, playback_config=None):
    catalog = PuzzleCatalog(
        puzzles=(
            PuzzleTask("p1", "Puzzle 1", "Answer 1", cue_id="cue-p1"),
            PuzzleTask("p2", "Puzzle 2", "Answer 2", cue_id="cue-p2"),
            PuzzleTask("p3", "Puzzle 3", "Answer 3", cue_id="cue-p3"),
            PuzzleTask("p4", "Puzzle 4", "Answer 4", cue_id="cue-p4"),
        )
    )
    assignment = PuzzleCueAssignment(
        session_id="night-001",
        cued_puzzle_ids=("p1", "p3"),
        uncued_puzzle_ids=("p2", "p4"),
        seed=17,
    )
    cue_library = CueLibrary(
        cues=(
            CueMetadata(
                cue_id="cue-p1",
                cue_type="generated_tone",
                protocol="puzzle",
                duration_seconds=1.0,
                frequency_hz=528.0,
                volume_hint=0.10,
            ),
            CueMetadata(
                cue_id="cue-p2",
                cue_type="generated_tone",
                protocol="puzzle",
                duration_seconds=1.0,
                frequency_hz=530.0,
                volume_hint=0.10,
            ),
            CueMetadata(
                cue_id="cue-p3",
                cue_type="generated_tone",
                protocol="puzzle",
                duration_seconds=1.0,
                frequency_hz=532.0,
                volume_hint=0.10,
            ),
            CueMetadata(
                cue_id="cue-p4",
                cue_type="generated_tone",
                protocol="puzzle",
                duration_seconds=1.0,
                frequency_hz=534.0,
                volume_hint=0.10,
            ),
            default_tlr_cue(),
            CueMetadata(
                cue_id="sham_soft_tone",
                cue_type="generated_tone",
                protocol="sham",
                duration_seconds=1.0,
                frequency_hz=330.0,
                volume_hint=0.05,
                tags=("sham", "untrained"),
            ),
        )
    )
    return TmrCueScheduler(
        assignment=assignment,
        catalog=catalog,
        cue_library=cue_library,
        config=config or TmrSchedulerConfig(
            puzzle_cue_interval_seconds=10.0,
            cooldown_seconds=30.0,
            max_puzzle_cues_per_block=2,
            enable_tlr_block=tlr_block_plan is not None,
        ),
        tlr_block_plan=tlr_block_plan,
        event_log_path=event_log_path,
        playback_config=playback_config,
    )


def _lower_volume():
    return ArousalGuardDecision(
        action="lower_volume",
        timestamp_seconds=0.0,
        reason_codes=("alpha_arousal_proxy_mild",),
        volume_multiplier=0.5,
    )


def _run_scheduler_replay(
    ticks: Iterable[_ReplayTick],
) -> Tuple[Tuple[TmrSchedulerEvent, ...], Tuple[RemGateDecision, ...], TmrCueScheduler]:
    gate = StableRemGate(
        RemGateConfig(
            enter_threshold=0.70,
            exit_threshold=0.45,
            min_stable_seconds=60.0,
            epoch_seconds=30.0,
            cooldown_seconds=0.0,
        )
    )
    scheduler = _scheduler(
        tlr_block_plan=None,
        config=TmrSchedulerConfig(
            puzzle_cue_interval_seconds=30.0,
            cooldown_seconds=30.0,
            max_puzzle_cues_per_block=4,
            enable_tlr_block=False,
        ),
    )
    decisions = []
    events = []
    for tick in ticks:
        decision = gate.update(
            RemPrediction(
                probability=tick.probability,
                reason_codes=tick.reason_codes,
                source="simulated_replay",
            ),
            duration_seconds=tick.duration_seconds,
        )
        decisions.append(decision)
        events.extend(
            scheduler.update(
                decision,
                timestamp_seconds=tick.timestamp_seconds,
                guard_decision=tick.guard_decision,
            )
        )
    return tuple(events), tuple(decisions), scheduler


def _tick(
    timestamp_seconds: float,
    probability: float,
    *reason_codes: str,
    guard_decision: Optional[ArousalGuardDecision] = None,
) -> _ReplayTick:
    return _ReplayTick(
        timestamp_seconds=timestamp_seconds,
        probability=probability,
        reason_codes=tuple(reason_codes),
        guard_decision=guard_decision,
    )


def _puzzle_play_ids(events: Iterable[TmrSchedulerEvent]) -> Tuple[str, ...]:
    return tuple(
        event.puzzle_id
        for event in events
        if event.event_type == "play" and event.protocol == "puzzle" and event.puzzle_id is not None
    )


def _assert_puzzle_plays_are_eligible(test_case, events, decisions, scheduler):
    for event, decision in zip(events, decisions):
        if event.event_type == "play" and event.protocol == "puzzle":
            test_case.assertTrue(decision.gate_open)
            test_case.assertIn(event.puzzle_id, scheduler.scheduled_puzzle_ids)


def _gate_open(*reason_codes):
    return RemGateDecision(
        gate_open=True,
        state="open",
        confidence=RemConfidence(
            probability=0.8,
            confidence=0.8,
            active_threshold=0.7,
            is_rem_like=True,
            source="test",
        ),
        stable_seconds=90.0,
        cooldown_remaining_seconds=0.0,
        reason_codes=reason_codes,
    )


def _gate_closed(*reason_codes):
    return RemGateDecision(
        gate_open=False,
        state="closed",
        confidence=RemConfidence(
            probability=0.3,
            confidence=0.3,
            active_threshold=0.7,
            is_rem_like=False,
            source="test",
        ),
        stable_seconds=0.0,
        cooldown_remaining_seconds=0.0,
        reason_codes=reason_codes,
    )


if __name__ == "__main__":
    unittest.main()
