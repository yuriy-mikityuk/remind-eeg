import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from muse_tmr import __version__
from muse_tmr.cli import main as cli_main
from muse_tmr.cli.main import _default_recording_dir, _resolve_output_dir, build_parser
from muse_tmr.sources.amused_source import AmusedSource
from muse_tmr.sources.base_source import MuseSourceMetadata


class FailingStreamSource:
    def __init__(self):
        self.stopped = False

    async def connect(self):
        return MuseSourceMetadata(
            source_name="fake",
            device_name="Muse Test",
            device_id="test-address",
            capabilities={"eeg": True},
        )

    async def stream(self):
        raise RuntimeError("boom")
        yield

    async def stop(self):
        self.stopped = True

    def diagnostics(self):
        return {"decoder": {"decode_errors": 0}, "packet_count": 0}


class TestCli(unittest.TestCase):
    def test_stream_command_parses_amused_source(self):
        args = build_parser().parse_args([
            "stream",
            "--source",
            "amused",
            "--duration-seconds",
            "3600",
            "--debug-stats",
        ])

        self.assertEqual(args.command, "stream")
        self.assertEqual(args.source, "amused")
        self.assertEqual(args.duration_seconds, 3600)
        self.assertTrue(args.debug_stats)

    def test_app_command_parses_local_defaults_and_mock_source(self):
        args = build_parser().parse_args([
            "app",
            "--source",
            "mock",
            "--mock-scenario",
            "all_good",
        ])

        self.assertEqual(args.command, "app")
        self.assertEqual(args.source, "mock")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)
        self.assertEqual(args.mock_scenario, "all_good")
        self.assertEqual(args.contact_stability_seconds, 5.0)

    def test_app_command_parses_amused_address_and_explicit_host(self):
        args = build_parser().parse_args([
            "app",
            "--source",
            "amused",
            "--address",
            "AA-BB",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
        ])

        self.assertEqual(args.command, "app")
        self.assertEqual(args.source, "amused")
        self.assertEqual(args.address, "AA-BB")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)

    def test_stream_command_parses_openmuse_lsl_source(self):
        args = build_parser().parse_args([
            "stream",
            "--source",
            "openmuse",
            "--duration-seconds",
            "5",
            "--require-lsl-stream",
            "eeg",
            "--require-lsl-stream",
            "imu",
            "--lsl-resolve-timeout",
            "2.5",
            "--openmuse-eeg-stream",
            "Muse_EEG",
            "--openmuse-imu-stream",
            "Muse_ACCGYRO",
        ])

        self.assertEqual(args.command, "stream")
        self.assertEqual(args.source, "openmuse")
        self.assertEqual(args.duration_seconds, 5)
        self.assertEqual(args.require_lsl_stream, ["eeg", "imu"])
        self.assertEqual(args.lsl_resolve_timeout, 2.5)
        self.assertEqual(args.openmuse_eeg_stream, "Muse_EEG")
        self.assertEqual(args.openmuse_imu_stream, "Muse_ACCGYRO")

    def test_stream_command_parses_brainflow_source(self):
        args = build_parser().parse_args([
            "stream",
            "--source",
            "brainflow",
            "--address",
            "AA:BB",
            "--duration-seconds",
            "5",
            "--brainflow-preset",
            "p1041",
            "--brainflow-serial-number",
            "Muse-Test",
            "--brainflow-no-low-latency",
            "--brainflow-poll-interval",
            "0.1",
            "--brainflow-chunk-samples",
            "128",
            "--brainflow-connect-timeout",
            "12",
            "--brainflow-stream-start-timeout",
            "7",
            "--brainflow-stop-timeout",
            "5",
            "--brainflow-session-cooldown",
            "0.25",
        ])

        self.assertEqual(args.command, "stream")
        self.assertEqual(args.source, "brainflow")
        self.assertEqual(args.address, "AA:BB")
        self.assertEqual(args.duration_seconds, 5)
        self.assertEqual(args.brainflow_preset, "p1041")
        self.assertEqual(args.brainflow_serial_number, "Muse-Test")
        self.assertTrue(args.brainflow_no_low_latency)
        self.assertEqual(args.brainflow_poll_interval, 0.1)
        self.assertEqual(args.brainflow_chunk_samples, 128)
        self.assertEqual(args.brainflow_connect_timeout, 12.0)
        self.assertEqual(args.brainflow_stream_start_timeout, 7.0)
        self.assertEqual(args.brainflow_stop_timeout, 5.0)
        self.assertEqual(args.brainflow_session_cooldown, 0.25)

    def test_diagnose_blink_artifacts_command_parses_closed_eyes_phase(self):
        args = build_parser().parse_args([
            "diagnose-blink-artifacts",
            "--source",
            "brainflow",
            "--output",
            "data/reports/brainflow_blink.json",
            "--eyes-open-baseline-seconds",
            "45",
            "--blink-seconds",
            "20",
            "--eyes-closed-baseline-seconds",
            "45",
            "--non-interactive",
        ])

        self.assertEqual(args.command, "diagnose-blink-artifacts")
        self.assertEqual(args.source, "brainflow")
        self.assertEqual(args.output, Path("data/reports/brainflow_blink.json"))
        self.assertEqual(args.eyes_open_baseline_seconds, 45.0)
        self.assertEqual(args.blink_seconds, 20.0)
        self.assertEqual(args.eyes_closed_baseline_seconds, 45.0)
        self.assertTrue(args.non_interactive)

    def test_compare_source_diagnostics_command_parses_report_table_options(self):
        args = build_parser().parse_args([
            "compare-source-diagnostics",
            "data/reports/brainflow_blink.json",
            "data/reports/amused_blink.json",
            "--output",
            "data/reports/source_comparison.md",
            "--format",
            "markdown",
        ])

        self.assertEqual(args.command, "compare-source-diagnostics")
        self.assertEqual(args.reports, [
            Path("data/reports/brainflow_blink.json"),
            Path("data/reports/amused_blink.json"),
        ])
        self.assertEqual(args.output, Path("data/reports/source_comparison.md"))
        self.assertEqual(args.format, "markdown")

    def test_inspect_blink_channels_command_parses_report_table_options(self):
        args = build_parser().parse_args([
            "inspect-blink-channels",
            "data/reports/brainflow_blink.json",
            "data/reports/amused_blink.json",
            "--phase",
            "blink",
            "--phase",
            "eyes_closed_baseline",
            "--output",
            "data/reports/blink_channels.md",
            "--format",
            "markdown",
        ])

        self.assertEqual(args.command, "inspect-blink-channels")
        self.assertEqual(args.reports, [
            Path("data/reports/brainflow_blink.json"),
            Path("data/reports/amused_blink.json"),
        ])
        self.assertEqual(args.phases, ["blink", "eyes_closed_baseline"])
        self.assertEqual(args.output, Path("data/reports/blink_channels.md"))
        self.assertEqual(args.format, "markdown")

    def test_stream_command_parses_sdk_stub_source(self):
        args = build_parser().parse_args([
            "stream",
            "--source",
            "sdk",
            "--sdk-path",
            "/tmp/local-muse-sdk",
            "--duration-seconds",
            "5",
        ])

        self.assertEqual(args.command, "stream")
        self.assertEqual(args.source, "sdk")
        self.assertEqual(args.sdk_path, Path("/tmp/local-muse-sdk"))
        self.assertEqual(args.duration_seconds, 5)

    def test_record_command_parses_overnight_duration(self):
        args = build_parser().parse_args([
            "record",
            "--source",
            "amused",
            "--duration-hours",
            "8",
        ])

        self.assertEqual(args.command, "record")
        self.assertEqual(args.duration_hours, 8.0)

    def test_replay_command_parses_time_range(self):
        args = build_parser().parse_args([
            "replay",
            "data/recordings/session",
            "--speed",
            "10",
            "--start-seconds",
            "30",
            "--end-seconds",
            "90",
        ])

        self.assertEqual(args.command, "replay")
        self.assertEqual(args.input, Path("data/recordings/session"))
        self.assertEqual(args.speed, 10.0)
        self.assertEqual(args.start_seconds, 30.0)
        self.assertEqual(args.end_seconds, 90.0)

    def test_annotate_template_command_parses_output_and_label(self):
        args = build_parser().parse_args([
            "annotate-template",
            "data/recordings/session",
            "--output",
            "data/annotations/session.csv",
            "--label",
            "unknown",
            "--epoch-seconds",
            "30",
            "--stride-seconds",
            "10",
        ])

        self.assertEqual(args.command, "annotate-template")
        self.assertEqual(args.input, Path("data/recordings/session"))
        self.assertEqual(args.output, Path("data/annotations/session.csv"))
        self.assertEqual(args.label, "unknown")
        self.assertEqual(args.epoch_seconds, 30.0)
        self.assertEqual(args.stride_seconds, 10.0)

    def test_train_rem_classifier_command_parses_model_options(self):
        args = build_parser().parse_args([
            "train-rem-classifier",
            "data/annotations/session.csv",
            "--output",
            "data/models/personal_rem_model.json",
            "--feature",
            "p_rem",
            "--feature",
            "feature_score_stillness",
            "--epochs",
            "500",
            "--threshold",
            "0.6",
        ])

        self.assertEqual(args.command, "train-rem-classifier")
        self.assertEqual(args.annotations, Path("data/annotations/session.csv"))
        self.assertEqual(args.output, Path("data/models/personal_rem_model.json"))
        self.assertEqual(args.features, ["p_rem", "feature_score_stillness"])
        self.assertEqual(args.epochs, 500)
        self.assertEqual(args.threshold, 0.6)

    def test_cue_library_commands_parse_paths_and_filters(self):
        create_args = build_parser().parse_args([
            "create-cue-library",
            "--output",
            "data/cues/starter.json",
        ])
        validate_args = build_parser().parse_args([
            "validate-cue-library",
            "data/cues/starter.json",
            "--skip-file-check",
        ])
        list_args = build_parser().parse_args([
            "list-cues",
            "data/cues/starter.json",
            "--protocol",
            "puzzle",
            "--tag",
            "generated",
        ])

        self.assertEqual(create_args.command, "create-cue-library")
        self.assertEqual(create_args.output, Path("data/cues/starter.json"))
        self.assertEqual(validate_args.command, "validate-cue-library")
        self.assertTrue(validate_args.skip_file_check)
        self.assertEqual(list_args.command, "list-cues")
        self.assertEqual(list_args.protocol, "puzzle")
        self.assertEqual(list_args.tag, "generated")

    def test_pilot2_calibration_validation_command_parses_inputs(self):
        args = build_parser().parse_args([
            "validate-pilot2-calibration",
            "data/calibration/volume_calibration.json",
            "--playback-log",
            "data/calibration/volume_calibration_test.jsonl",
            "--device-name",
            "Sleep Headphones",
            "--output",
            "data/reports/pilot2_audio_calibration_validation.json",
            "--hard-max-volume",
            "0.2",
        ])

        self.assertEqual(args.command, "validate-pilot2-calibration")
        self.assertEqual(args.calibration, Path("data/calibration/volume_calibration.json"))
        self.assertEqual(args.playback_log, Path("data/calibration/volume_calibration_test.jsonl"))
        self.assertEqual(args.device_name, "Sleep Headphones")
        self.assertEqual(args.output, Path("data/reports/pilot2_audio_calibration_validation.json"))
        self.assertEqual(args.hard_max_volume, 0.2)

    def test_pilot3_replay_simulation_command_parses_inputs(self):
        args = build_parser().parse_args([
            "simulate-replay-cues",
            "data/recordings/night-001",
            "--catalog",
            "data/protocol/catalog.json",
            "--session",
            "data/protocol/night-001_puzzles.json",
            "--assignment",
            "data/protocol/night-001_assignment.json",
            "--cue-library",
            "data/cues/starter.json",
            "--output",
            "data/reports/pilot3_replay_cue_plan.json",
            "--scheduler-events-output",
            "data/reports/pilot3_scheduler_events.jsonl",
            "--start-seconds",
            "30",
            "--end-seconds",
            "600",
            "--min-stable-seconds",
            "30",
            "--disable-arousal-guard",
        ])

        self.assertEqual(args.command, "simulate-replay-cues")
        self.assertEqual(args.input, Path("data/recordings/night-001"))
        self.assertEqual(args.catalog, Path("data/protocol/catalog.json"))
        self.assertEqual(args.session, Path("data/protocol/night-001_puzzles.json"))
        self.assertEqual(args.assignment, Path("data/protocol/night-001_assignment.json"))
        self.assertEqual(args.cue_library, Path("data/cues/starter.json"))
        self.assertEqual(args.output, Path("data/reports/pilot3_replay_cue_plan.json"))
        self.assertEqual(args.scheduler_events_output, Path("data/reports/pilot3_scheduler_events.jsonl"))
        self.assertEqual(args.start_seconds, 30.0)
        self.assertEqual(args.end_seconds, 600.0)
        self.assertEqual(args.min_stable_seconds, 30.0)
        self.assertTrue(args.disable_arousal_guard)

    def test_synthetic_night_commands_parse_their_defaults(self):
        args = build_parser().parse_args(["demo"])
        self.assertEqual(args.command, "demo")
        self.assertEqual(
            (args.scenario, args.hours, args.seed, args.output, args.no_open, args.baseline),
            ("clean_night", 2.0, 0, None, False, None),
        )

        args = build_parser().parse_args([
            "demo", "--scenario", "fragmented_rem", "--hours", "8", "--seed", "3",
            "--output", "data/synthetic/demo", "--no-open",
            "--baseline", "tests/fixtures/synthetic/agreement_baseline.json",
        ])
        self.assertEqual((args.scenario, args.hours, args.seed), ("fragmented_rem", 8.0, 3))
        self.assertEqual(args.output, Path("data/synthetic/demo"))
        self.assertTrue(args.no_open)
        self.assertEqual(args.baseline, Path("tests/fixtures/synthetic/agreement_baseline.json"))

        args = build_parser().parse_args(["synthesize-night", "--list"])
        self.assertEqual(args.command, "synthesize-night")
        self.assertTrue(args.list)
        self.assertEqual((args.scenario, args.seed, args.hours, args.output), (None, 0, None, None))

        args = build_parser().parse_args([
            "synthesize-night", "--scenario", "no_rem_nap", "--seed", "4", "--hours", "0.5",
            "--output", "data/synthetic/nap",
        ])
        self.assertEqual((args.scenario, args.seed, args.hours), ("no_rem_nap", 4, 0.5))
        self.assertEqual(args.output, Path("data/synthetic/nap"))
        self.assertFalse(args.list)

    def test_pilot4_cueing_command_parses_safety_inputs(self):
        args = build_parser().parse_args([
            "run-pilot4-cueing",
            "--source",
            "amused",
            "--address",
            "AA-BB",
            "--duration-hours",
            "2",
            "--output-dir",
            "data/recordings/pilot4-night",
            "--catalog",
            "data/protocol/catalog.json",
            "--session",
            "data/protocol/session.json",
            "--assignment",
            "data/protocol/assignment.json",
            "--cue-library",
            "data/cues/starter.json",
            "--calibration",
            "data/calibration/volume_calibration.json",
            "--device-name",
            "Sleep Headphones",
            "--backend",
            "system",
            "--default-volume",
            "0.02",
            "--hard-max-volume",
            "0.2",
            "--emergency-stop-file",
            "data/recordings/pilot4-night/STOP_AUDIO",
        ])

        self.assertEqual(args.command, "run-pilot4-cueing")
        self.assertEqual(args.source, "amused")
        self.assertEqual(args.address, "AA-BB")
        self.assertEqual(args.duration_hours, 2.0)
        self.assertEqual(args.output_dir, Path("data/recordings/pilot4-night"))
        self.assertEqual(args.calibration, Path("data/calibration/volume_calibration.json"))
        self.assertEqual(args.device_name, "Sleep Headphones")
        self.assertEqual(args.backend, "system")
        self.assertEqual(args.default_volume, 0.02)
        self.assertEqual(args.emergency_stop_file, Path("data/recordings/pilot4-night/STOP_AUDIO"))

    def test_pilot5_full_night_command_requires_tlr_block(self):
        args = build_parser().parse_args([
            "run-pilot5-full-night",
            "--source",
            "amused",
            "--address",
            "AA-BB",
            "--duration-hours",
            "8",
            "--output-dir",
            "data/recordings/pilot5-night",
            "--catalog",
            "data/protocol/catalog.json",
            "--session",
            "data/protocol/session.json",
            "--assignment",
            "data/protocol/assignment.json",
            "--cue-library",
            "data/cues/starter.json",
            "--tlr-block",
            "data/protocol/tlr_block.json",
            "--calibration",
            "data/calibration/volume_calibration.json",
            "--device-name",
            "Sleep Headphones",
            "--backend",
            "system",
        ])

        self.assertEqual(args.command, "run-pilot5-full-night")
        self.assertEqual(args.source, "amused")
        self.assertEqual(args.address, "AA-BB")
        self.assertEqual(args.duration_hours, 8.0)
        self.assertEqual(args.output_dir, Path("data/recordings/pilot5-night"))
        self.assertEqual(args.tlr_block, Path("data/protocol/tlr_block.json"))
        self.assertEqual(args.calibration, Path("data/calibration/volume_calibration.json"))
        self.assertEqual(args.device_name, "Sleep Headphones")
        self.assertEqual(args.backend, "system")

    def test_pilot4_awakening_log_command_parses_marker(self):
        args = build_parser().parse_args([
            "log-pilot4-awakening",
            "data/recordings/pilot4-night/awakening_events.jsonl",
            "--event-type",
            "awakening",
            "--notes",
            "woke briefly after tone",
            "--timestamp-utc",
            "2026-05-13T22:00:00+00:00",
        ])

        self.assertEqual(args.command, "log-pilot4-awakening")
        self.assertEqual(args.output, Path("data/recordings/pilot4-night/awakening_events.jsonl"))
        self.assertEqual(args.event_type, "awakening")
        self.assertEqual(args.notes, "woke briefly after tone")
        self.assertEqual(args.timestamp_utc, "2026-05-13T22:00:00+00:00")

    def test_pilot4_awakening_log_takes_its_time_and_attribution(self):
        args = build_parser().parse_args([
            "log-pilot4-awakening",
            "awakening_events.jsonl",
            "--timestamp-seconds",
            "1250.5",
            "--attributed-to-cue",
            "unsure",
        ])
        self.assertEqual((args.timestamp_seconds, args.attributed_to_cue), (1250.5, "unsure"))
        args = build_parser().parse_args(["log-pilot4-awakening", "awakening_events.jsonl"])
        self.assertEqual((args.timestamp_seconds, args.attributed_to_cue), (None, None))
        for flags in (["--timestamp-seconds", "-1"], ["--attributed-to-cue", "maybe"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(["log-pilot4-awakening", "awakening_events.jsonl", *flags])

    def test_score_cue_arousal_command_parses_its_inputs(self):
        args = build_parser().parse_args(["score-cue-arousal", "data/recordings/pilot4-night"])
        self.assertEqual(args.command, "score-cue-arousal")
        self.assertEqual(args.recording_dir, Path("data/recordings/pilot4-night"))
        for name in (
            "pilot3_report", "summary", "scheduler_events", "audio_log", "gate_log",
            "awakening_events", "replay_input", "preset", "output_dir",
        ):
            self.assertIsNone(getattr(args, name), name)
        args = build_parser().parse_args([
            "score-cue-arousal", "data/recordings/night-001",
            "--pilot3-report", "data/reports/pilot3.json",
            "--summary", "s.json",
            "--scheduler-events", "e.jsonl",
            "--audio-log", "a.jsonl",
            "--gate-log", "g.jsonl",
            "--awakening-events", "w.jsonl",
            "--replay-input", "raw_amused.bin",
            "--preset", "p21",
            "--output-dir", "out",
        ])
        self.assertEqual(args.pilot3_report, Path("data/reports/pilot3.json"))
        self.assertEqual((args.gate_log, args.audio_log), (Path("g.jsonl"), Path("a.jsonl")))
        self.assertEqual((args.preset, args.output_dir), ("p21", Path("out")))

    def test_simulate_replay_cues_takes_a_positive_injection_interval(self):
        required = [
            "simulate-replay-cues", "night", "--catalog", "c.json", "--session", "s.json",
            "--assignment", "a.json", "--cue-library", "l.json", "--output", "o.json",
        ]
        self.assertIsNone(build_parser().parse_args(required).inject_cues_every_seconds)
        args = build_parser().parse_args(required + ["--inject-cues-every-seconds", "180"])
        self.assertEqual(args.inject_cues_every_seconds, 180.0)
        for value in ("0", "-30", "soon"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(required + ["--inject-cues-every-seconds", value])

    def test_tlr_protocol_commands_parse_paths(self):
        create_args = build_parser().parse_args([
            "create-tlr-cue",
            "--output",
            "data/cues/tlr.json",
        ])
        train_args = build_parser().parse_args([
            "train-tlr-cue",
            "data/cues/tlr.json",
            "--output",
            "data/protocol/tlr_training.json",
            "--event-log",
            "data/protocol/tlr_training.jsonl",
        ])
        block_args = build_parser().parse_args([
            "plan-tlr-block",
            "data/cues/tlr.json",
            "--output",
            "data/protocol/tlr_block.json",
            "--disabled",
        ])

        self.assertEqual(create_args.command, "create-tlr-cue")
        self.assertEqual(create_args.output, Path("data/cues/tlr.json"))
        self.assertEqual(train_args.command, "train-tlr-cue")
        self.assertEqual(train_args.backend, "dry-run")
        self.assertEqual(block_args.command, "plan-tlr-block")
        self.assertTrue(block_args.disabled)

    def test_puzzle_protocol_commands_parse_paths(self):
        import_args = build_parser().parse_args([
            "import-puzzles",
            "puzzles.csv",
            "--output",
            "data/protocol/catalog.json",
        ])
        attempt_args = build_parser().parse_args([
            "record-puzzle-attempt",
            "data/protocol/catalog.json",
            "--puzzle-id",
            "p1",
            "--response",
            "answer",
            "--duration-seconds",
            "30",
            "--solved",
        ])
        association_args = build_parser().parse_args([
            "record-association-check",
            "data/protocol/session.json",
            "--catalog",
            "data/protocol/catalog.json",
            "--puzzle-id",
            "p1",
            "--response",
            "answer",
        ])
        assignment_args = build_parser().parse_args([
            "assign-puzzle-cues",
            "data/protocol/session.json",
            "--output",
            "data/protocol/assignment.json",
            "--seed",
            "17",
        ])

        self.assertEqual(import_args.command, "import-puzzles")
        self.assertEqual(import_args.output, Path("data/protocol/catalog.json"))
        self.assertEqual(attempt_args.command, "record-puzzle-attempt")
        self.assertTrue(attempt_args.solved)
        self.assertEqual(association_args.command, "record-association-check")
        self.assertEqual(association_args.catalog, Path("data/protocol/catalog.json"))
        self.assertEqual(assignment_args.command, "assign-puzzle-cues")
        self.assertEqual(assignment_args.seed, 17)

    def test_night_arm_flags_parse_on_the_three_runners(self):
        base = {
            "simulate-replay-cues": ["recording", "--output", "r.json"],
            "run-pilot4-cueing": ["--output-dir", "night", "--calibration", "c.json"],
            "run-pilot5-full-night": ["--output-dir", "night", "--calibration", "c.json", "--tlr-block", "t.json"],
        }
        inputs = ["--catalog", "c.json", "--session", "s.json", "--assignment", "a.json", "--cue-library", "l.json"]
        for command, extra in base.items():
            with self.subTest(command=command):
                plain = build_parser().parse_args([command, *extra, *inputs])
                self.assertEqual(
                    (plain.arm, plain.night_schedule, plain.night_id, plain.sham_cue_id), (None, None, None, None)
                )
                armed = build_parser().parse_args(
                    [command, *extra, *inputs, "--arm", "sham", "--night-id", "n1", "--sham-cue-id", "chime"]
                )
                self.assertEqual((armed.arm, armed.night_id, armed.sham_cue_id), ("sham", "n1", "chime"))
                sealed = build_parser().parse_args(
                    [command, *extra, *inputs, "--night-schedule", "series.json", "--night-id", "night-002"]
                )
                self.assertEqual((sealed.night_schedule, sealed.night_id), (Path("series.json"), "night-002"))
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    build_parser().parse_args([command, *extra, *inputs, "--arm", "washout"])
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    build_parser().parse_args(
                        [command, *extra, *inputs, "--arm", "cue", "--night-schedule", "series.json"]
                    )

    def test_night_schedule_commands_parse(self):
        plan = build_parser().parse_args(["plan-night-schedule", "--blocks", "2"])
        self.assertEqual(
            (plan.blocks, plan.washout_nights, plan.seed, plan.schedule_id, plan.night_prefix, plan.sham_cue_id, plan.output),
            (2, 0, None, None, "night-", "sham_soft_tone", None),
        )
        show = build_parser().parse_args(["show-night-schedule", "series.json", "--json"])
        unseal = build_parser().parse_args(["unseal-night-schedule", "series.json", "--sealed", "s.sealed.json"])
        self.assertEqual((show.schedule, show.json, show.sealed), (Path("series.json"), True, None))
        self.assertEqual((unseal.command, unseal.sealed), ("unseal-night-schedule", Path("s.sealed.json")))
        dream = build_parser().parse_args([
            "record-dream-report", "s.json", "--output", "d.json", "--lucid", "no", "--confidence", "0.5",
            "--dream-text", "", "--cue-perception", "in_dream", "--eye-signal-attempt", "04:20",
        ])
        self.assertEqual(
            (dream.cues_heard, dream.cue_perception, dream.woke_from_sound, dream.eye_signal_attempt),
            (None, "in_dream", "unknown", ["04:20"]),
        )
        retest = build_parser().parse_args([
            "record-puzzle-retest", "s.json", "--catalog", "c.json", "--assignment", "a.json", "--output", "r.json",
        ])
        self.assertEqual(retest.wave, "morning")

    def test_morning_report_command_parses_dream_report_fields(self):
        args = build_parser().parse_args([
            "record-dream-report",
            "data/protocol/session.json",
            "--catalog",
            "data/protocol/catalog.json",
            "--output",
            "data/reports/night-001_dream_report.json",
            "--lucid",
            "yes",
            "--cues-heard",
            "no",
            "--confidence",
            "0.7",
            "--dream-text",
            "I saw the puzzle answer.",
            "--puzzle-link",
            "p1=the answer appeared",
            "--puzzle-link",
            "p2=the shape was in the dream",
        ])

        self.assertEqual(args.command, "record-dream-report")
        self.assertEqual(args.session, Path("data/protocol/session.json"))
        self.assertEqual(args.catalog, Path("data/protocol/catalog.json"))
        self.assertEqual(args.output, Path("data/reports/night-001_dream_report.json"))
        self.assertEqual(args.lucid, "yes")
        self.assertEqual(args.cues_heard, "no")
        self.assertEqual(args.confidence, 0.7)
        self.assertEqual(args.puzzle_link, [
            "p1=the answer appeared",
            "p2=the shape was in the dream",
        ])

    def test_morning_retest_command_parses_blind_results(self):
        args = build_parser().parse_args([
            "record-puzzle-retest",
            "data/protocol/session.json",
            "--catalog",
            "data/protocol/catalog.json",
            "--assignment",
            "data/protocol/assignment.json",
            "--output",
            "data/reports/night-001_retest.json",
            "--result",
            "p1=answer",
            "--result",
            "p2=wrong",
            "--solved",
            "p1",
            "--duration",
            "p1=12",
            "--duration",
            "p2=24",
            "--confidence",
            "p1=0.9",
            "--confidence",
            "p2=0.3",
        ])

        self.assertEqual(args.command, "record-puzzle-retest")
        self.assertEqual(args.session, Path("data/protocol/session.json"))
        self.assertEqual(args.catalog, Path("data/protocol/catalog.json"))
        self.assertEqual(args.assignment, Path("data/protocol/assignment.json"))
        self.assertEqual(args.output, Path("data/reports/night-001_retest.json"))
        self.assertEqual(args.result, ["p1=answer", "p2=wrong"])
        self.assertEqual(args.solved, ["p1"])
        self.assertEqual(args.duration, ["p1=12", "p2=24"])
        self.assertEqual(args.confidence, ["p1=0.9", "p2=0.3"])

    def test_cued_uncued_analysis_command_parses_report_inputs(self):
        args = build_parser().parse_args([
            "analyze-cued-uncued",
            "data/protocol/session.json",
            "--assignment",
            "data/protocol/assignment.json",
            "--retest",
            "data/reports/retest.json",
            "--retest",
            "data/reports/retest_day7.json",
            "--coding",
            "data/reports/coding_rater-a.json",
            "--dream-report",
            "data/reports/dream.json",
            "--scheduler-events",
            "data/reports/scheduler.jsonl",
            "--output",
            "data/reports/analysis.json",
            "--markdown-output",
            "data/reports/analysis.md",
            "--analysis-id",
            "analysis-001",
            "--min-group-size",
            "3",
        ])

        self.assertEqual(args.command, "analyze-cued-uncued")
        self.assertEqual(args.session, Path("data/protocol/session.json"))
        self.assertEqual(args.assignment, Path("data/protocol/assignment.json"))
        # One --retest per wave since schema 3 (#228).
        self.assertEqual(args.retest, [Path("data/reports/retest.json"), Path("data/reports/retest_day7.json")])
        self.assertEqual(args.coding, [Path("data/reports/coding_rater-a.json")])
        self.assertEqual(args.dream_report, Path("data/reports/dream.json"))
        self.assertEqual(args.scheduler_events, Path("data/reports/scheduler.jsonl"))
        self.assertEqual(args.output, Path("data/reports/analysis.json"))
        self.assertEqual(args.markdown_output, Path("data/reports/analysis.md"))
        self.assertEqual(args.analysis_id, "analysis-001")
        self.assertEqual(args.min_group_size, 3)

    def test_amused_source_import_does_not_cycle(self):
        self.assertEqual(AmusedSource.strategy, "forked-source")

    def test_build_brainflow_source_does_not_require_brainflow_dependency(self):
        args = build_parser().parse_args([
            "stream",
            "--source",
            "brainflow",
            "--duration-seconds",
            "5",
        ])

        source = cli_main._build_source(args, duration_seconds=5)

        self.assertEqual(source.source_name, "brainflow")
        self.assertEqual(source.strategy, "optional-brainflow")
        self.assertEqual(source.config.connect_timeout_seconds, 20.0)
        self.assertEqual(source.config.session_cooldown_seconds, 2.0)

    def test_default_recording_dir_avoids_root_when_cwd_is_unusable(self):
        with patch("muse_tmr.cli.main.Path.cwd", return_value=Path("/")):
            output_dir = _default_recording_dir()

        self.assertTrue(output_dir.is_absolute())
        self.assertEqual(output_dir.parent.name, "recordings")
        self.assertEqual(output_dir.parent.parent.name, "data")
        self.assertNotEqual(output_dir.parent.parent.parent, Path("/"))

    def test_relative_output_dir_avoids_root_when_cwd_is_unusable(self):
        with patch("muse_tmr.cli.main.Path.cwd", return_value=Path("/")):
            output_dir = _resolve_output_dir(Path("data/recordings/smoke"))

        self.assertTrue(output_dir.is_absolute())
        self.assertTrue(output_dir.as_posix().endswith("data/recordings/smoke"))
        self.assertNotEqual(output_dir.parent.parent.parent, Path(output_dir.anchor))


class TestCliStreamRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostic_phase_collection_does_not_cancel_slow_stream(self):
        frame = object()
        queue = asyncio.Queue()

        async def slow_stream():
            await asyncio.sleep(0.05)
            yield frame

        stream_task = asyncio.create_task(
            cli_main._pump_diagnostic_stream(slow_stream(), queue)
        )

        frames = await cli_main._collect_diagnostic_phase_frames(
            queue,
            stream_task,
            duration_seconds=0.2,
        )

        self.assertEqual(frames, (frame,))
        self.assertTrue(stream_task.done())

    async def test_stream_debug_stats_prints_on_failure(self):
        args = build_parser().parse_args([
            "stream",
            "--source",
            "amused",
            "--duration-seconds",
            "1",
            "--debug-stats",
        ])
        source = FailingStreamSource()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("muse_tmr.cli.main._build_source", return_value=source):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = await cli_main._stream(args)

        self.assertEqual(code, 1)
        self.assertTrue(source.stopped)
        self.assertIn("stream diagnostics=", stdout.getvalue())
        self.assertIn("\"packet_count\": 0", stdout.getvalue())
        self.assertIn("stream failed error=boom", stderr.getvalue())


class TestResolveGateThresholds(unittest.TestCase):
    """MUST-FIX #1: preset-aware threshold resolution must never crash the gate.

    A partially-overridden pair (only one of --enter/--exit given) must not leave
    exit > enter, which would make RemGateConfig.validate() raise mid-pilot.
    """

    def _resolve(self, preset=None, enter=None, exit_=None):
        import argparse

        args = argparse.Namespace(
            preset=preset, enter_threshold=enter, exit_threshold=exit_
        )
        with contextlib.redirect_stderr(io.StringIO()):
            return cli_main._resolve_gate_thresholds(args)

    def _assert_valid_pair(self, enter, exit_):
        from muse_tmr.models import RemGateConfig

        self.assertLessEqual(exit_, enter)
        # Must build a config that validates -- the whole point of the clamp.
        RemGateConfig(enter_threshold=enter, exit_threshold=exit_).validate()

    def test_both_omitted_uses_preset_pair(self):
        self.assertEqual(self._resolve(preset="p21"), (0.90, 0.85))
        self.assertEqual(self._resolve(preset="p1034"), (0.80, 0.70))
        self.assertEqual(self._resolve(preset=None), (0.80, 0.70))

    def test_both_supplied_are_used_verbatim(self):
        self.assertEqual(self._resolve(preset="p21", enter=0.75, exit_=0.60), (0.75, 0.60))

    def test_partial_enter_override_below_preset_exit_does_not_crash(self):
        # enter=0.60 alone on p21 leaves preset exit=0.85 > 0.60 -> must clamp.
        enter, exit_ = self._resolve(preset="p21", enter=0.60)
        self.assertEqual(enter, 0.60)
        self._assert_valid_pair(enter, exit_)

    def test_partial_exit_override_keeps_preset_enter(self):
        enter, exit_ = self._resolve(preset="p21", exit_=0.50)
        self.assertEqual((enter, exit_), (0.90, 0.50))
        self._assert_valid_pair(enter, exit_)

    def test_both_supplied_but_inverted_is_clamped(self):
        enter, exit_ = self._resolve(preset="p21", enter=0.60, exit_=0.80)
        self.assertEqual(enter, 0.60)
        self._assert_valid_pair(enter, exit_)

    def test_partial_override_emits_warning_on_clamp(self):
        import argparse

        args = argparse.Namespace(preset="p21", enter_threshold=0.60, exit_threshold=None)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cli_main._resolve_gate_thresholds(args)
        self.assertIn("clamping exit", stderr.getvalue())


class TestPilot3OpticsWiring(unittest.IsolatedAsyncioTestCase):
    """MUST-FIX #2: pilot3 must pass optics_capable through to the gate config.

    Without it, a p21 (EEG-only) replay reproduces as a silent zero-cue night
    because the missing-PPG reason code pins confidence at the 0.55 cap.
    """

    async def _capture_gate_config(self, preset, *extra_args):
        captured = {}

        async def fake_simulate(*args, gate_config, arousal_guard_config, **kwargs):
            captured["gate_config"] = gate_config
            captured["arousal_guard_config"] = arousal_guard_config
            report = MagicMock()
            report.passed = True
            report.metrics = {}
            report.audio_playback_executed = False
            report.save.return_value = Path("report.json")
            return report

        args = build_parser().parse_args([
            "simulate-replay-cues", "rec_dir",
            "--catalog", "catalog.json",
            "--session", "session.json",
            "--assignment", "assignment.json",
            "--cue-library", "cues.json",
            "--output", "out.json",
            "--preset", preset,
            *extra_args,
        ])
        with patch("muse_tmr.validation.simulate_replay_cue_plan", new=fake_simulate), \
                patch("muse_tmr.audio.load_cue_library", return_value=MagicMock()), \
                patch("muse_tmr.protocol.load_puzzle_catalog", return_value=MagicMock()), \
                patch("muse_tmr.protocol.load_night_puzzle_session", return_value=MagicMock()), \
                patch("muse_tmr.protocol.load_puzzle_cue_assignment", return_value=MagicMock()), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = await cli_main._simulate_replay_cues(args)
        self.assertEqual(rc, 0)
        self.guard_config = captured["arousal_guard_config"]
        return captured["gate_config"]

    async def test_eeg_only_preset_is_wired_optics_incapable(self):
        gate_config = await self._capture_gate_config("p21")
        self.assertFalse(gate_config.optics_capable)
        self.assertFalse(self.guard_config.optics_capable)
        self.assertEqual(
            (gate_config.enter_threshold, gate_config.exit_threshold), (0.90, 0.85)
        )

    async def test_optics_preset_is_wired_optics_capable(self):
        gate_config = await self._capture_gate_config("p1034")
        self.assertTrue(gate_config.optics_capable)
        self.assertTrue(self.guard_config.optics_capable)
        self.assertEqual(
            (gate_config.enter_threshold, gate_config.exit_threshold), (0.80, 0.70)
        )

    async def test_hr_rule_gates_reach_the_guard(self):
        await self._capture_gate_config("p1034")
        self.assertEqual(self.guard_config.min_ppg_confidence_for_hr_rule, 0.0)
        self.assertEqual(self.guard_config.min_hr_series_quality_for_hr_rule, 0.0)
        await self._capture_gate_config(
            "p1034", "--guard-min-ppg-confidence", "0.6", "--guard-min-hr-series-quality", "0.8"
        )
        self.assertEqual(self.guard_config.min_ppg_confidence_for_hr_rule, 0.6)
        self.assertEqual(self.guard_config.min_hr_series_quality_for_hr_rule, 0.8)
        self.assertTrue(self.guard_config.optics_capable)


class TestPilot4OpticsWiring(unittest.IsolatedAsyncioTestCase):
    """run-pilot4-cueing builds the gate and the guard from one preset and keeps --allow-dry-run."""

    async def _capture_config(self, preset, *extra_args):
        captured = {}

        async def fake_run(source, *, config, **kwargs):
            captured["config"] = config
            summary = MagicMock()
            summary.passed = True
            return summary

        args = build_parser().parse_args([
            "run-pilot4-cueing",
            "--output-dir", "out",
            "--catalog", "catalog.json",
            "--session", "session.json",
            "--assignment", "assignment.json",
            "--cue-library", "cues.json",
            "--calibration", "calibration.json",
            "--duration-seconds", "60",
            "--allow-short",
            "--preset", preset,
            *extra_args,
        ])
        with patch("muse_tmr.validation.run_pilot4_cueing_night", new=fake_run), \
                patch("muse_tmr.audio.load_volume_calibrations", return_value=MagicMock()), \
                patch("muse_tmr.audio.load_cue_library", return_value=MagicMock()), \
                patch("muse_tmr.protocol.load_puzzle_catalog", return_value=MagicMock()), \
                patch("muse_tmr.protocol.load_night_puzzle_session", return_value=MagicMock()), \
                patch("muse_tmr.protocol.load_puzzle_cue_assignment", return_value=MagicMock()), \
                patch.object(cli_main, "_build_source", return_value=MagicMock()), \
                patch.object(cli_main, "_resolve_output_dir", return_value=Path("out")), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = await cli_main._run_pilot4_cueing(args)
        self.assertEqual(rc, 0)
        return captured["config"]

    async def test_eeg_only_preset_is_wired_optics_incapable(self):
        config = await self._capture_config("p21")
        self.assertFalse(config.gate_config.optics_capable)
        self.assertFalse(config.arousal_guard_config.optics_capable)

    async def test_optics_preset_is_wired_optics_capable(self):
        config = await self._capture_config("p1034")
        self.assertTrue(config.gate_config.optics_capable)
        self.assertTrue(config.arousal_guard_config.optics_capable)

    async def test_allow_dry_run_reaches_the_night_config(self):
        self.assertFalse((await self._capture_config("p1034")).allow_dry_run_fallback)
        config = await self._capture_config("p1034", "--backend", "system", "--allow-dry-run")
        self.assertEqual(config.audio_backend_name, "system")
        self.assertTrue(config.allow_dry_run_fallback)

    async def test_hr_rule_gates_reach_the_night_guard(self):
        guard = (await self._capture_config("p1034")).arousal_guard_config
        self.assertEqual(
            (guard.min_ppg_confidence_for_hr_rule, guard.min_hr_series_quality_for_hr_rule), (0.0, 0.0)
        )
        config = await self._capture_config(
            "p1034", "--guard-min-ppg-confidence", "0.4", "--guard-min-hr-series-quality", "0.7"
        )
        guard = config.arousal_guard_config
        self.assertEqual(
            (guard.min_ppg_confidence_for_hr_rule, guard.min_hr_series_quality_for_hr_rule), (0.4, 0.7)
        )

    def test_pilot5_takes_the_hr_rule_gates_and_refuses_values_outside_0_1(self):
        required = [
            "run-pilot5-full-night", "--output-dir", "out", "--catalog", "c.json", "--session", "s.json",
            "--assignment", "a.json", "--cue-library", "l.json", "--tlr-block", "t.json",
            "--calibration", "v.json",
        ]
        args = build_parser().parse_args(required + ["--guard-min-ppg-confidence", "0.5"])
        self.assertEqual(args.guard_min_ppg_confidence, 0.5)
        self.assertEqual(args.guard_min_hr_series_quality, 0.0)
        for value in ("1.5", "-0.1", "high"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(required + ["--guard-min-ppg-confidence", value])


class TestAudioPreflightCli(unittest.TestCase):
    """The commands that can play a cue exit with code 2 when real audio would be silent."""

    def test_audio_commands_take_allow_dry_run(self):
        required = {
            "play-test-cue": [],
            "train-tlr-cue": ["cues.json", "--output", "training.json", "--event-log", "training.jsonl"],
            "run-pilot4-cueing": [
                "--output-dir", "out", "--catalog", "catalog.json", "--session", "session.json",
                "--assignment", "assignment.json", "--cue-library", "cues.json",
                "--calibration", "calibration.json",
            ],
        }
        required["run-pilot5-full-night"] = required["run-pilot4-cueing"] + ["--tlr-block", "block.json"]
        required["record-cue-recall"] = [
            "session.json", "--catalog", "catalog.json", "--assignment", "assignment.json",
            "--cue-library", "cues.json", "--calibration", "calibration.json", "--output", "recall.json",
        ]
        self.assertEqual(set(required), set(cli_main.AUDIO_COMMANDS))
        parser = build_parser()
        for command, argv in required.items():
            with self.subTest(command=command):
                self.assertFalse(parser.parse_args([command, *argv]).allow_dry_run)
                self.assertTrue(parser.parse_args([command, *argv, "--allow-dry-run"]).allow_dry_run)

    def test_play_test_cue_refuses_real_audio_without_afplay(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "playback.jsonl"
            for backend in ("system", "afplay"):
                with self.subTest(backend=backend):
                    stderr = io.StringIO()
                    with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None), \
                            contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                        rc = cli_main.main([
                            "play-test-cue", "--backend", backend,
                            "--duration-seconds", "0.01", "--log-path", str(log_path),
                        ])
                    self.assertEqual(rc, 2)
                    self.assertIn("needs afplay", stderr.getvalue())
                    self.assertIn("--allow-dry-run", stderr.getvalue())
            self.assertFalse(log_path.exists())

    def test_play_test_cue_runs_silently_with_allow_dry_run(self):
        stdout = io.StringIO()
        with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None), \
                contextlib.redirect_stdout(stdout):
            rc = cli_main.main([
                "play-test-cue", "--backend", "system", "--allow-dry-run", "--duration-seconds", "0.01",
            ])
        self.assertEqual(rc, 0)
        self.assertIn("status=played", stdout.getvalue())
        self.assertIn("backend=dry-run requested_backend=system", stdout.getvalue())

    def test_training_and_nights_refuse_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.json")
            night_inputs = [
                "--catalog", missing, "--session", missing, "--assignment", missing,
                "--cue-library", missing, "--calibration", missing,
                "--duration-seconds", "60", "--allow-short",
            ]
            commands = {
                "train-tlr-cue": [
                    "train-tlr-cue", missing,
                    "--output", str(Path(tmp) / "training.json"),
                    "--event-log", str(Path(tmp) / "training.jsonl"),
                ],
                "run-pilot4-cueing": [
                    "run-pilot4-cueing", "--output-dir", str(Path(tmp) / "pilot4"), *night_inputs,
                ],
                "run-pilot5-full-night": [
                    "run-pilot5-full-night", "--output-dir", str(Path(tmp) / "pilot5"), *night_inputs,
                    "--tlr-block", missing,
                ],
                "record-cue-recall": [
                    "record-cue-recall", missing, "--catalog", missing, "--assignment", missing,
                    "--cue-library", missing, "--calibration", missing,
                    "--output", str(Path(tmp) / "recall.json"),
                ],
            }
            with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None), \
                    patch("muse_tmr.audio.load_cue_library") as load_cue_library, \
                    patch("muse_tmr.audio.load_volume_calibrations") as load_calibrations, \
                    patch.object(cli_main, "_build_source") as build_source, \
                    contextlib.redirect_stderr(io.StringIO()):
                for command, argv in commands.items():
                    with self.subTest(command=command):
                        self.assertEqual(cli_main.main([*argv, "--backend", "system"]), 2)
            load_cue_library.assert_not_called()
            load_calibrations.assert_not_called()
            build_source.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])


class TestBanner(unittest.TestCase):
    """`muse-tmr status` and the end of `muse-tmr --help` say what REMind is (#232)."""

    BANNER = f"REMind {__version__}: n = 1 research software, not a medical device"

    def test_status_prints_the_banner(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cli_main.main(["status"]), 0)
        self.assertEqual(stdout.getvalue(), self.BANNER + "\n")

    def test_help_ends_with_the_banner(self):
        for argv in (["--help"], []):
            with self.subTest(argv=argv):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    try:
                        rc = cli_main.main(argv)
                    except SystemExit as exc:
                        rc = exc.code
                self.assertEqual(rc, 0)
                # argparse wraps the help to the terminal's width, so compare without line breaks.
                self.assertTrue(" ".join(stdout.getvalue().split()).endswith(self.BANNER))


class TestDataExchangeCommands(unittest.TestCase):
    """export-edf, export-bids, import and annotate-template --from-hypnogram (#230)."""

    def test_export_commands_parse_their_options(self):
        args = build_parser().parse_args([
            "export-edf", "data/recordings/n1", "--output", "out/n1.edf", "--hypnogram", "a.tsv",
            "--hypnogram", "b.edf", "--night-schedule", "series.json", "--gap-threshold-seconds", "1.5", "--json",
        ])
        self.assertEqual((args.command, args.input, args.output), ("export-edf", Path("data/recordings/n1"), Path("out/n1.edf")))
        self.assertEqual(args.hypnogram, [Path("a.tsv"), Path("b.edf")])
        self.assertEqual((args.night_schedule, args.gap_threshold_seconds, args.json), (Path("series.json"), 1.5, True))

        args = build_parser().parse_args(["export-bids", "data/recordings/n1"])
        self.assertEqual((args.subject, args.session, args.task, args.output), ("01", None, "sleep", None))
        self.assertEqual((args.hypnogram, args.gap_threshold_seconds, args.power_line_frequency), ([], 0.5, None))
        args = build_parser().parse_args([
            "export-bids", "n1", "--output", "bids", "--subject", "p07", "--session", "night2", "--power-line-frequency", "50",
        ])
        self.assertEqual((args.subject, args.session, args.power_line_frequency), ("p07", "night2", 50.0))

    def test_import_and_from_hypnogram_parse_their_options(self):
        args = build_parser().parse_args([
            "import", "night.csv", "--format", "mindmonitor-csv", "--eeg-scale", "0.5", "--eeg-offset", "-800",
            "--start-time", "2026-01-14T23:00:00+00:00", "--output", "data/recordings/imported/n",
        ])
        self.assertEqual((args.import_format, args.eeg_scale, args.eeg_offset), ("mindmonitor-csv", 0.5, -800.0))
        self.assertEqual((args.start_time, args.source_version, args.channel_map), ("2026-01-14T23:00:00+00:00", None, None))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["import", "night.xdf", "--format", "xdf"])
        args = build_parser().parse_args(["import", "a.edf", "--format", "edf", "--channel-map", "boas-headband"])
        self.assertEqual(args.channel_map, "boas-headband")

        args = build_parser().parse_args([
            "annotate-template", "n1", "--output", "rows.csv", "--from-hypnogram", "scored.tsv", "--hypnogram-column", "stage_ai",
        ])
        self.assertEqual((args.from_hypnogram, args.hypnogram_column), (Path("scored.tsv"), "stage_ai"))
        self.assertIsNone(build_parser().parse_args(["annotate-template", "n1", "--output", "r.csv"]).from_hypnogram)

    def test_exports_refuse_a_night_from_a_sealed_schedule(self):
        from muse_tmr.protocol import generate_night_schedule, night_arm_from_schedule, unseal_night_schedule, write_night_schedule

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schedule_path = root / "series-a.json"
            write_night_schedule(generate_night_schedule(7, 1, schedule_id="series-a"), schedule_path)
            arm = night_arm_from_schedule(schedule_path, "night-001")
            night = root / "night"
            night.mkdir()
            (night / "metadata.json").write_text(json.dumps({"started_at": "2026-01-14T23:00:00+00:00",
                                                             "night_arm": arm.to_dict()}))
            for command in ("export-edf", "export-bids"):
                for extra in ([], ["--night-schedule", str(schedule_path)]):
                    with self.subTest(command=command, extra=extra):
                        stderr = io.StringIO()
                        with contextlib.redirect_stderr(stderr):
                            code = cli_main.main([command, str(night), "--output", str(root / "out"), *extra])
                        self.assertEqual(code, 2)
                        self.assertIn("unseal-night-schedule", stderr.getvalue())
                        self.assertNotIn(arm.arm, stderr.getvalue().replace("unseal", ""))
            self.assertFalse((root / "out").exists())

            # Once unsealed the night passes the check; without the extra it stops at the hint.
            unseal_night_schedule(schedule_path)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), patch("muse_tmr.data.edf_io.edf_available", return_value=False):
                code = cli_main.main(["export-edf", str(night), "--night-schedule", str(schedule_path)])
            self.assertEqual(code, 2)
            self.assertIn('pip install "remind-eeg[edf]"', stderr.getvalue())
            self.assertNotIn("unseal", stderr.getvalue())

    def test_import_refuses_what_it_cannot_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "mm.csv"
            source.write_text("TimeStamp,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10\n2026-01-14 23:00:00.000,1,2,3,4\n")
            for extra, message in (([], "--eeg-scale"), (["--eeg-scale", "1", "--channel-map", "boas-headband"], "edf only")):
                with self.subTest(extra=extra):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr):
                        code = cli_main.main(["import", str(source), "--format", "mindmonitor-csv",
                                              "--output", str(Path(tmp) / "out"), *extra])
                    self.assertEqual(code, 2)
                    self.assertIn(message, stderr.getvalue())


class TestAnnotateFromHypnogram(unittest.TestCase):
    """Labels from a hypnogram (#230), on a labelled synthetic night: synthetic input, not sleep."""

    @classmethod
    def setUpClass(cls):
        from tests.exchange_synthetic import labelled_night

        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.night = labelled_night(cls.root / "night")
        cls.expected = ["wake"] * 4 + ["nrem"] * 4 + ["probable_rem"] * 4 + ["nrem"] * 2

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _rows(self, hypnogram, *extra):
        from muse_tmr.annotations import load_rem_annotations

        output = self.root / f"rows_{len(list(self.root.glob('rows_*')))}.csv"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli_main.main(["annotate-template", str(self.night.recording_dir), "--output", str(output),
                                  "--from-hypnogram", str(hypnogram), *extra])
        self.assertEqual(code, 0)
        return load_rem_annotations(output), output, stdout.getvalue()

    def test_each_epoch_gets_the_stage_at_its_middle(self):
        rows, output, printed = self._rows(self.night.recording_dir / "hypnogram.csv")

        self.assertEqual([row.label for row in rows], self.expected)
        self.assertEqual({row.prediction_source for row in rows}, {"hypnogram:hypnogram.csv"})
        self.assertIn("labelled_from=hypnogram.csv wake=4 nrem=6 probable_rem=4 unknown=0", printed)

        model = self.root / "model.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main.main(["train-rem-classifier", str(output), "--output", str(model)]), 0)
        self.assertTrue(model.exists())

    def test_a_part_of_the_night_keeps_the_whole_nights_clock(self):
        rows, _, _ = self._rows(self.night.recording_dir / "hypnogram.csv", "--start-seconds", "150", "--end-seconds", "330")

        self.assertEqual([row.label for row in rows], self.expected[5:5 + len(rows)])
        self.assertGreaterEqual(len(rows), 5)

    def test_a_bids_scoring_file_and_an_unscored_stretch(self):
        scored = self.root / "scored_events.tsv"
        scored.write_text("onset\tduration\tstage\tlabel\n0\t60\tSleep stage W\twake\n60\t60\tSleep stage R\tprobable_rem\n")

        rows, _, _ = self._rows(scored)

        self.assertEqual([row.label for row in rows], ["wake", "wake", "probable_rem", "probable_rem"] + ["unknown"] * 10)


if __name__ == "__main__":
    unittest.main()
