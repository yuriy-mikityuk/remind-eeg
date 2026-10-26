import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from muse_tmr.audio import (
    AudioCuePlayer,
    AudioPlaybackConfig,
    CueMetadata,
    MockAudioBackend,
    load_cue_library,
)
from muse_tmr.cli.main import build_parser, main
from muse_tmr.protocol import (
    DEFAULT_TLR_CUE_ID,
    TlrBlockConfig,
    TlrCueConfig,
    TlrTrainingConfig,
    default_tlr_cue,
    default_tlr_cue_library,
    load_tlr_block_plan,
    load_tlr_training_session,
    plan_tlr_block,
    train_tlr_cue,
)


class TestTlrProtocol(unittest.TestCase):
    def test_default_tlr_cue_is_generated_tlr_metadata(self):
        cue = default_tlr_cue()

        self.assertEqual(cue.cue_id, DEFAULT_TLR_CUE_ID)
        self.assertEqual(cue.cue_type, "generated_tone")
        self.assertEqual(cue.protocol, "tlr")
        self.assertEqual(cue.frequency_hz, 396.0)
        self.assertEqual(cue.volume_hint, 0.05)

    def test_default_tlr_cue_library_validates(self):
        library = default_tlr_cue_library(TlrCueConfig(cue_id="tlr_custom", frequency_hz=432.0))

        report = library.validate()

        self.assertTrue(report.is_valid)
        self.assertEqual(library.library_id, "tlr_default")
        self.assertEqual(library.cues[0].cue_id, "tlr_custom")

    def test_training_repeats_generated_tlr_cue_and_logs_events(self):
        cue = default_tlr_cue(TlrCueConfig(duration_seconds=0.01, volume_hint=0.06))
        backend = MockAudioBackend()
        player = AudioCuePlayer(
            AudioPlaybackConfig(max_volume=0.20, default_volume=0.06),
            backend=backend,
        )

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "tlr_training.jsonl"
            session = train_tlr_cue(
                cue,
                player,
                config=TlrTrainingConfig(repetitions=3, interval_seconds=1.5),
                session_id="night-001-pre-sleep",
                event_log_path=log_path,
            )
            logged_events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(session.session_id, "night-001-pre-sleep")
        self.assertEqual(session.event_count, 3)
        self.assertEqual(len(backend.requests), 3)
        self.assertEqual([event.repetition_index for event in session.events], [1, 2, 3])
        self.assertEqual([event.scheduled_offset_seconds for event in session.events], [0.0, 1.5, 3.0])
        self.assertEqual(len(logged_events), 3)
        self.assertEqual(logged_events[0]["event_type"], "tlr_training_cue")

    def test_training_plays_the_hint_or_the_player_default_under_the_cap(self):
        cases = (
            # volume_hint, requested, effective under a 0.06 cap with a 0.02 default
            (0.10, 0.10, 0.06),
            (0.05, 0.05, 0.05),
            (None, 0.02, 0.02),
            # A hint of 0 is a hint, as in Pilot 4 and 5; it used to fall back to the default.
            (0.0, 0.0, 0.0),
        )
        for volume_hint, requested, effective in cases:
            with self.subTest(volume_hint=volume_hint):
                cue = CueMetadata(
                    cue_id="tlr_custom",
                    cue_type="generated_tone",
                    protocol="tlr",
                    duration_seconds=0.01,
                    frequency_hz=396.0,
                    volume_hint=volume_hint,
                )
                backend = MockAudioBackend()
                player = AudioCuePlayer(
                    AudioPlaybackConfig(max_volume=0.06, default_volume=0.02),
                    backend=backend,
                )

                event = train_tlr_cue(cue, player, config=TlrTrainingConfig(repetitions=1)).events[0]

                self.assertEqual((event.requested_volume, event.effective_volume), (requested, effective))
                self.assertEqual(backend.requests[0].volume_multiplier, 1.0)

    def test_training_rejects_non_tlr_or_non_generated_cues(self):
        player = AudioCuePlayer(backend=MockAudioBackend())
        puzzle_cue = CueMetadata(
            cue_id="puzzle",
            cue_type="generated_tone",
            protocol="puzzle",
            duration_seconds=0.01,
            frequency_hz=440.0,
        )
        sound_tlr = CueMetadata(
            cue_id="tlr_sound",
            cue_type="sound",
            protocol="tlr",
            duration_seconds=0.01,
            path="private/tlr.wav",
        )

        with self.assertRaises(ValueError):
            train_tlr_cue(puzzle_cue, player)
        with self.assertRaises(ValueError):
            train_tlr_cue(sound_tlr, player)

    def test_tlr_block_plan_configures_rem_block_before_puzzle_cues(self):
        cue = default_tlr_cue(TlrCueConfig(duration_seconds=1.0))

        plan = plan_tlr_block(
            cue,
            config=TlrBlockConfig(
                repetitions=3,
                interval_seconds=8.0,
                post_block_pause_seconds=10.0,
            ),
        )

        self.assertEqual(len(plan.events), 3)
        self.assertEqual([event.offset_seconds for event in plan.events], [0.0, 8.0, 16.0])
        self.assertEqual(plan.total_duration_seconds, 27.0)
        self.assertEqual(plan.puzzle_cue_start_offset_seconds, 27.0)

    def test_disabled_tlr_block_has_no_events(self):
        plan = plan_tlr_block(default_tlr_cue(), config=TlrBlockConfig(enabled=False, repetitions=0))

        self.assertEqual(plan.events, ())
        self.assertEqual(plan.total_duration_seconds, 0.0)

    def test_training_and_block_plan_round_trip_json(self):
        cue = default_tlr_cue(TlrCueConfig(duration_seconds=0.01))
        player = AudioCuePlayer(backend=MockAudioBackend())

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            training_path = tmp_path / "training.json"
            block_path = tmp_path / "block.json"
            train_tlr_cue(
                cue,
                player,
                config=TlrTrainingConfig(repetitions=1),
            ).save(training_path)
            plan_tlr_block(cue, config=TlrBlockConfig(repetitions=1)).save(block_path)

            training = load_tlr_training_session(training_path)
            block = load_tlr_block_plan(block_path)

        self.assertEqual(training.event_count, 1)
        self.assertEqual(block.events[0].cue_id, cue.cue_id)


class TestTlrProtocolCli(unittest.TestCase):
    def test_tlr_commands_parse_paths_and_options(self):
        create_args = build_parser().parse_args([
            "create-tlr-cue",
            "--output",
            "data/cues/tlr.json",
            "--frequency-hz",
            "432",
        ])
        train_args = build_parser().parse_args([
            "train-tlr-cue",
            "data/cues/tlr.json",
            "--output",
            "data/protocol/tlr_training.json",
            "--event-log",
            "data/protocol/tlr_training.jsonl",
            "--repetitions",
            "3",
            "--backend",
            "dry-run",
        ])
        block_args = build_parser().parse_args([
            "plan-tlr-block",
            "data/cues/tlr.json",
            "--output",
            "data/protocol/tlr_block.json",
            "--repetitions",
            "2",
        ])

        self.assertEqual(create_args.command, "create-tlr-cue")
        self.assertEqual(create_args.frequency_hz, 432.0)
        self.assertEqual(train_args.command, "train-tlr-cue")
        self.assertEqual(train_args.repetitions, 3)
        self.assertEqual(block_args.command, "plan-tlr-block")
        self.assertEqual(block_args.repetitions, 2)

    def test_cli_generates_trains_and_plans_tlr(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            library_path = tmp_path / "tlr_library.json"
            training_path = tmp_path / "tlr_training.json"
            event_log_path = tmp_path / "tlr_events.jsonl"
            block_path = tmp_path / "tlr_block.json"

            with redirect_stdout(io.StringIO()) as create_output:
                create_code = main([
                    "create-tlr-cue",
                    "--output",
                    str(library_path),
                    "--duration-seconds",
                    "0.01",
                ])
            library = load_cue_library(library_path)

            with redirect_stdout(io.StringIO()) as train_output:
                train_code = main([
                    "train-tlr-cue",
                    str(library_path),
                    "--output",
                    str(training_path),
                    "--event-log",
                    str(event_log_path),
                    "--backend",
                    "dry-run",
                    "--repetitions",
                    "2",
                    "--interval-seconds",
                    "1",
                ])
            training = load_tlr_training_session(training_path)
            event_lines = event_log_path.read_text(encoding="utf-8").splitlines()

            with redirect_stdout(io.StringIO()) as block_output:
                block_code = main([
                    "plan-tlr-block",
                    str(library_path),
                    "--output",
                    str(block_path),
                    "--repetitions",
                    "2",
                    "--interval-seconds",
                    "5",
                    "--post-block-pause-seconds",
                    "7",
                ])
            block = load_tlr_block_plan(block_path)

        self.assertEqual(create_code, 0)
        self.assertEqual(train_code, 0)
        self.assertEqual(block_code, 0)
        self.assertIn("TLR cue created", create_output.getvalue())
        self.assertIn("TLR training complete", train_output.getvalue())
        self.assertIn("TLR block planned", block_output.getvalue())
        self.assertEqual(library.by_id(DEFAULT_TLR_CUE_ID).protocol, "tlr")
        self.assertEqual(training.event_count, 2)
        self.assertEqual(len(event_lines), 2)
        self.assertEqual(len(block.events), 2)
        self.assertEqual(block.puzzle_cue_start_offset_seconds, 12.01)


if __name__ == "__main__":
    unittest.main()
