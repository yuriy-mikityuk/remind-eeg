"""CLI entry point for the Muse REM-TMR project."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from muse_tmr import __version__

# Commands that can play a cue; main() checks their --backend before anything else runs.
AUDIO_COMMANDS = (
    "play-test-cue",
    "train-tlr-cue",
    "run-pilot4-cueing",
    "run-pilot5-full-night",
    "record-cue-recall",
)
# Commands that take --protocol-preset; main() refuses an unknown or non-runnable one first.
PROTOCOL_PRESET_COMMANDS = (
    "simulate-replay-cues",
    "run-pilot4-cueing",
    "run-pilot5-full-night",
    "plan-tlr-block",
    "create-tlr-cue",
    "create-cue-library",
    "generate-puzzle-session",
    "assign-puzzle-cues",
    "prereg-template",
)
PROTOCOL_PRESET_DEFAULT_HELP = "Default: the protocol preset's."
# The arms a cueing night runs: muse_tmr.protocol.night_schedule.CUEING_ARMS (#228).
NIGHT_ARM_CHOICES = ("cue", "sham", "no_cue")
# The protocol roles of a cue: muse_tmr.audio.cue_library.CUE_PROTOCOLS.
CUE_PROTOCOL_CHOICES = ("puzzle", "tlr", "test", "generic", "sham")
# What a runner prints in place of its cue counts on a night with an arm.
BLINDED_NOTE = (
    "blinded night: plays, counts and volumes are in the summary, which names the arm; "
    "in a sealed series leave it closed until unseal-night-schedule has run"
)
ALLOW_DRY_RUN_HELP = "Run silently on dry-run when afplay is missing, instead of exiting with code 2."
# `muse-tmr status` prints it and `muse-tmr --help` ends with it (#232).
BANNER = f"REMind {__version__}: n = 1 research software, not a medical device"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="muse-tmr",
        description="Muse S Athena REM-TMR/TLR research tooling.",
        epilog=BANNER,
    )
    parser.add_argument("--version", action="version", version=f"muse-tmr {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("status", help="Print the version and what REMind is.")

    app_parser = subparsers.add_parser("app", help="Run the REMind app, a local web page.")
    app_parser.add_argument("--source", choices=("mock", "amused"), default="mock")
    app_parser.add_argument("--host", default="127.0.0.1")
    app_parser.add_argument("--port", type=int, default=8765)
    app_parser.add_argument("--address", help="Muse BLE address. If omitted, discovery is used.")
    app_parser.add_argument("--name-filter", default="Muse")
    app_parser.add_argument("--preset", default="p1034")
    app_parser.add_argument(
        "--mock-scenario",
        default="mixed_fair_good",
        help="Mock contact scenario for local UI development.",
    )
    app_parser.add_argument("--mock-interval-seconds", type=float, default=1.0)
    app_parser.add_argument("--contact-stability-seconds", type=float, default=5.0)
    app_parser.add_argument(
        "--auto-update",
        action="store_true",
        help="Fast-forward a clean main checkout to origin/main and restart when idle.",
    )
    app_parser.add_argument(
        "--cue-backend",
        choices=("system", "afplay", "dry-run", "mock"),
        default="system",
        help="Playback backend of the Morning tab's sound check. system and afplay need afplay; "
        "without it a sound is refused unless --allow-dry-run is given.",
    )
    app_parser.add_argument(
        "--allow-dry-run",
        action="store_true",
        help="Run the sound check silently when afplay is missing.",
    )

    discover_parser = subparsers.add_parser("discover", help="Discover Muse devices.")
    discover_parser.add_argument(
        "--source",
        choices=("amused", "brainflow", "openmuse", "sdk"),
        default="amused",
    )
    discover_parser.add_argument("--name-filter", default="Muse")
    _add_brainflow_args(discover_parser)
    _add_openmuse_lsl_args(discover_parser)
    _add_muse_sdk_args(discover_parser)

    stream_parser = subparsers.add_parser("stream", help="Stream Muse frames from a source.")
    stream_parser.add_argument(
        "--source",
        choices=("amused", "brainflow", "openmuse", "sdk"),
        default="amused",
    )
    stream_parser.add_argument("--address", help="Muse BLE address. If omitted, discovery is used.")
    stream_parser.add_argument("--name-filter", default="Muse")
    stream_parser.add_argument("--preset", default="p1034")
    stream_parser.add_argument("--duration-seconds", type=int, default=3600)
    stream_parser.add_argument(
        "--debug-stats",
        action="store_true",
        help="Print source/decoder diagnostics after the stream completes.",
    )
    stream_parser.add_argument("--quiet", action="store_true")
    _add_brainflow_args(stream_parser)
    _add_openmuse_lsl_args(stream_parser)
    _add_muse_sdk_args(stream_parser)

    diagnostic_parser = subparsers.add_parser(
        "diagnose-blink-artifacts",
        help="Run a live blink/artifact diagnostic protocol and write a JSON metric report.",
    )
    diagnostic_parser.add_argument(
        "--source",
        choices=("amused", "brainflow"),
        default="brainflow",
    )
    diagnostic_parser.add_argument("--address", help="Muse BLE address. If omitted, discovery is used.")
    diagnostic_parser.add_argument("--name-filter", default="Muse")
    diagnostic_parser.add_argument("--preset", default="p1034")
    diagnostic_parser.add_argument(
        "--output",
        type=Path,
        help="Output report .json path. Defaults under data/reports/.",
    )
    diagnostic_parser.add_argument("--settle-seconds", type=float, default=30.0)
    diagnostic_parser.add_argument("--eyes-open-baseline-seconds", type=float, default=45.0)
    diagnostic_parser.add_argument("--blink-seconds", type=float, default=20.0)
    diagnostic_parser.add_argument("--recovery-open-seconds", type=float, default=30.0)
    diagnostic_parser.add_argument("--jaw-clench-seconds", type=float, default=20.0)
    diagnostic_parser.add_argument("--head-movement-seconds", type=float, default=20.0)
    diagnostic_parser.add_argument("--eyes-closed-baseline-seconds", type=float, default=45.0)
    diagnostic_parser.add_argument("--sample-rate-hz", type=float, default=256.0)
    diagnostic_parser.add_argument("--center-window-seconds", type=float, default=5.0)
    diagnostic_parser.add_argument("--highpass-cutoff-hz", type=float, default=0.5)
    diagnostic_parser.add_argument("--window-seconds", type=float, default=1.0)
    diagnostic_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Start each phase immediately instead of waiting for Enter.",
    )
    diagnostic_parser.add_argument("--quiet", action="store_true")
    _add_brainflow_args(diagnostic_parser)

    compare_diagnostics_parser = subparsers.add_parser(
        "compare-source-diagnostics",
        help="Compare blink/artifact diagnostic JSON reports as a source quality table.",
    )
    compare_diagnostics_parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Diagnostic report JSON paths from diagnose-blink-artifacts.",
    )
    compare_diagnostics_parser.add_argument(
        "--output",
        type=Path,
        help="Optional comparison output path (.md, .csv, or .json).",
    )
    compare_diagnostics_parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format when --output is set. The console output is markdown.",
    )

    inspect_channels_parser = subparsers.add_parser(
        "inspect-blink-channels",
        help="Inspect per-channel blink/artifact metrics from diagnostic JSON reports.",
    )
    inspect_channels_parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Diagnostic report JSON paths from diagnose-blink-artifacts.",
    )
    inspect_channels_parser.add_argument(
        "--phase",
        action="append",
        dest="phases",
        help="Phase to include. Repeat to include multiple phases. Defaults to all phases.",
    )
    inspect_channels_parser.add_argument(
        "--output",
        type=Path,
        help="Optional inspection output path (.md, .csv, or .json).",
    )
    inspect_channels_parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format when --output is set. The console output is markdown.",
    )

    replay_parser = subparsers.add_parser("replay", help="Replay a recorded Muse session.")
    replay_parser.add_argument("input", type=Path, help="Recording directory or raw_amused.bin path.")
    replay_parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Replay speed multiplier. 1.0 is real time; 0.0 replays as fast as possible.",
    )
    replay_parser.add_argument("--start-seconds", type=float, help="Relative replay start offset.")
    replay_parser.add_argument("--end-seconds", type=float, help="Relative replay end offset.")

    synthesize_parser = subparsers.add_parser(
        "synthesize-night",
        help="Write a labelled synthetic night in the Athena TAG format (synthetic input, not sleep).",
    )
    synthesize_parser.add_argument("--scenario", help="Scenario name; --list shows them.")
    synthesize_parser.add_argument("--list", action="store_true", help="List the scenarios and exit.")
    synthesize_parser.add_argument("--seed", type=int, default=0)
    synthesize_parser.add_argument("--hours", type=float, help="Length in hours. Defaults to the scenario's.")
    synthesize_parser.add_argument("--output", type=Path, help="Recording directory to write.")

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run Pilot 1, Pilot 3 with mock audio and the REM report on a synthetic night; no headband.",
    )
    demo_parser.add_argument("--scenario", default="clean_night")
    demo_parser.add_argument("--hours", type=float, default=2.0)
    demo_parser.add_argument("--seed", type=int, default=0)
    demo_parser.add_argument(
        "--output", type=Path, help="Output directory. Defaults to data/synthetic/demo_<timestamp>/."
    )
    demo_parser.add_argument("--no-open", action="store_true", help="Do not open the report in a browser.")
    demo_parser.add_argument(
        "--baseline",
        type=Path,
        help="Agreement baseline .json; exit 1 when this run differs from it (the CI end_to_end job).",
    )

    annotate_parser = subparsers.add_parser(
        "annotate-template",
        help="Generate an editable REM annotation template from replayed epochs.",
    )
    annotate_parser.add_argument("input", type=Path, help="Recording directory or raw_amused.bin path.")
    annotate_parser.add_argument("--output", type=Path, required=True, help="Output .csv or .json path.")
    annotate_parser.add_argument("--epoch-seconds", type=float, default=30.0)
    annotate_parser.add_argument("--stride-seconds", type=float, default=30.0)
    annotate_parser.add_argument("--start-seconds", type=float, help="Relative replay start offset.")
    annotate_parser.add_argument("--end-seconds", type=float, help="Relative replay end offset.")
    annotate_parser.add_argument(
        "--label",
        choices=("wake", "nrem", "probable_rem", "unknown"),
        default="unknown",
        help="Initial label for generated rows. Use unknown for manual labeling templates.",
    )

    train_parser = subparsers.add_parser(
        "train-rem-classifier",
        help="Train a personal REM classifier from labeled annotation rows.",
    )
    train_parser.add_argument("annotations", type=Path, help="Input annotation .csv or .json path.")
    train_parser.add_argument("--output", type=Path, required=True, help="Output model .json path.")
    train_parser.add_argument(
        "--feature",
        action="append",
        dest="features",
        help="Feature column to use. Repeat to override the default feature set.",
    )
    train_parser.add_argument("--min-training-rows", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=1200)
    train_parser.add_argument("--learning-rate", type=float, default=0.05)
    train_parser.add_argument("--l2-penalty", type=float, default=0.01)
    train_parser.add_argument("--threshold", type=float, default=0.5)

    play_parser = subparsers.add_parser("play-test-cue", help="Play a low-volume test cue.")
    play_parser.add_argument("--frequency-hz", type=float, default=440.0)
    play_parser.add_argument("--duration-seconds", type=float, default=1.0)
    play_parser.add_argument("--volume", type=float, default=0.05)
    play_parser.add_argument("--max-volume", type=float, default=0.20)
    play_parser.add_argument("--fade-in-seconds", type=float, default=0.25)
    play_parser.add_argument("--fade-out-seconds", type=float, default=0.25)
    play_parser.add_argument("--device-name")
    play_parser.add_argument("--log-path", type=Path)
    play_parser.add_argument(
        "--calibration",
        type=Path,
        help="Volume calibration metadata .json. Uses --device-name when provided, else the latest record.",
    )
    play_parser.add_argument(
        "--backend",
        choices=("system", "afplay", "dry-run", "mock"),
        default="system",
        help="Playback backend. system and afplay need afplay and exit with code 2 without it.",
    )
    play_parser.add_argument("--allow-dry-run", action="store_true", help=ALLOW_DRY_RUN_HELP)
    play_parser.add_argument(
        "--emergency-stop",
        action="store_true",
        help="Trigger and log emergency stop instead of playing a cue.",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate-volume",
        help="Save pre-sleep volume calibration metadata for one playback device.",
    )
    calibrate_parser.add_argument("--device-name", required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True, help="Output calibration .json path.")
    calibrate_parser.add_argument("--detectable-volume", type=float, required=True)
    calibrate_parser.add_argument("--identifiable-volume", type=float, required=True)
    calibrate_parser.add_argument("--comfortable-volume", type=float, required=True)
    calibrate_parser.add_argument("--cue-id", default="test-cue")
    calibrate_parser.add_argument(
        "--backend",
        choices=("system", "afplay", "dry-run", "mock"),
        default="dry-run",
    )
    calibrate_parser.add_argument("--notes", default="")
    calibrate_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the calibration file instead of appending/updating one device record.",
    )

    create_cues_parser = subparsers.add_parser(
        "create-cue-library",
        help="Create a starter cue metadata library.",
    )
    create_cues_parser.add_argument("--output", type=Path, required=True, help="Output cue library .json path.")
    create_cues_parser.add_argument(
        "--protocol-preset",
        default=None,
        metavar="NAME",
        help="Put this protocol preset's TLR tone, the cue create-tlr-cue --protocol-preset makes, in place of "
        "the starter tlr_soft_tone, so that training, the TLR plan and the night use one library.",
    )

    validate_cues_parser = subparsers.add_parser(
        "validate-cue-library",
        help="Validate cue metadata and pre-session file availability.",
    )
    validate_cues_parser.add_argument("input", type=Path, help="Input cue library .json path.")
    validate_cues_parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Validate metadata only; do not fail on missing sound cue files.",
    )

    pilot1_parser = subparsers.add_parser(
        "validate-pilot1-recording",
        help="Validate an M8 Pilot 1 no-audio overnight recording summary.",
    )
    pilot1_parser.add_argument("input", type=Path, help="Recording directory or summary.json path.")
    pilot1_parser.add_argument("--output", type=Path, help="Optional output validation report .json path.")
    pilot1_parser.add_argument("--min-duration-hours", type=float, default=6.0)
    pilot1_parser.add_argument(
        "--required-modality",
        action="append",
        dest="required_modalities",
        help="Required nonzero modality in summary.modality_counts. Repeat to override defaults.",
    )
    pilot1_parser.add_argument("--max-downtime-fraction", type=float, default=0.05)

    pilot2_parser = subparsers.add_parser(
        "validate-pilot2-calibration",
        help="Validate an M8 Pilot 2 audio calibration and cap-probe log.",
    )
    pilot2_parser.add_argument("calibration", type=Path, help="Volume calibration .json path.")
    pilot2_parser.add_argument(
        "--playback-log",
        type=Path,
        help="Dry-run play-test-cue JSONL log proving the calibration cap was used.",
    )
    pilot2_parser.add_argument("--device-name", help="Selected calibration device. Defaults to latest.")
    pilot2_parser.add_argument("--output", type=Path, help="Optional output validation report .json path.")
    pilot2_parser.add_argument("--hard-max-volume", type=float, default=0.20)

    pilot3_parser = subparsers.add_parser(
        "simulate-replay-cues",
        help="Generate an M8 Pilot 3 replay-only cue plan with mocked audio.",
    )
    pilot3_parser.add_argument("input", type=Path, help="Recording directory or raw_amused.bin path.")
    pilot3_parser.add_argument("--catalog", type=Path, required=True, help="Puzzle catalog .json path.")
    pilot3_parser.add_argument("--session", type=Path, required=True, help="Night puzzle session .json path.")
    pilot3_parser.add_argument("--assignment", type=Path, required=True, help="Cued/uncued assignment .json path.")
    pilot3_parser.add_argument("--cue-library", type=Path, required=True, help="Cue metadata library .json path.")
    pilot3_parser.add_argument("--output", type=Path, required=True, help="Output simulation report .json path.")
    pilot3_parser.add_argument("--scheduler-events-output", type=Path, help="Optional scheduler events .jsonl path.")
    pilot3_parser.add_argument("--start-seconds", type=float, help="Relative replay start offset.")
    pilot3_parser.add_argument("--end-seconds", type=float, help="Relative replay end offset.")
    pilot3_parser.add_argument(
        "--preset",
        default=None,
        help="Recording preset (e.g. p21, p1034). Sets preset-aware gate thresholds "
        "and whether absent PPG caps confidence. Unknown/None assumes optics.",
    )
    pilot3_parser.add_argument("--epoch-seconds", type=float, default=30.0)
    pilot3_parser.add_argument("--stride-seconds", type=float, default=30.0)
    pilot3_parser.add_argument("--enter-threshold", type=float, default=None,
                               help="Override gate enter threshold (default: preset-aware).")
    pilot3_parser.add_argument("--exit-threshold", type=float, default=None,
                               help="Override gate exit threshold (default: preset-aware).")
    _add_night_protocol_args(pilot3_parser)
    _add_night_arm_args(pilot3_parser)
    pilot3_parser.add_argument("--disable-arousal-guard", action="store_true")
    pilot3_parser.add_argument(
        "--inject-cues-every-seconds",
        type=_positive_float,
        default=None,
        metavar="N",
        help="Also add a synthetic_cue play event to gate-open epochs every N seconds (mock "
        "audio, outside the scheduler) to check where score-cue-arousal puts its windows.",
    )
    _add_guard_hr_rule_args(pilot3_parser)

    pilot4_parser = subparsers.add_parser(
        "run-pilot4-cueing",
        help="Run an M8 Pilot 4 low-volume REM-gated cueing night.",
    )
    pilot4_parser.add_argument(
        "--source",
        choices=("amused", "brainflow", "openmuse", "sdk"),
        default="amused",
    )
    pilot4_parser.add_argument("--address", help="Muse BLE address. If omitted, discovery is used.")
    pilot4_parser.add_argument("--name-filter", default="Muse")
    pilot4_parser.add_argument("--preset", default="p1034")
    pilot4_parser.add_argument("--duration-hours", type=float, default=8.0)
    pilot4_parser.add_argument("--duration-seconds", type=float)
    pilot4_parser.add_argument("--output-dir", type=Path, required=True, help="Pilot output directory.")
    pilot4_parser.add_argument("--catalog", type=Path, required=True, help="Puzzle catalog .json path.")
    pilot4_parser.add_argument("--session", type=Path, required=True, help="Night puzzle session .json path.")
    pilot4_parser.add_argument("--assignment", type=Path, required=True, help="Cued/uncued assignment .json path.")
    pilot4_parser.add_argument("--cue-library", type=Path, required=True, help="Cue metadata library .json path.")
    pilot4_parser.add_argument("--calibration", type=Path, required=True, help="Volume calibration .json path.")
    pilot4_parser.add_argument("--device-name", help="Selected calibration device. Defaults to latest.")
    pilot4_parser.add_argument(
        "--backend",
        choices=("system", "afplay", "dry-run", "mock"),
        default="dry-run",
        help="Playback backend. Use system for the actual Pilot 4 sleep run.",
    )
    pilot4_parser.add_argument("--allow-dry-run", action="store_true", help=ALLOW_DRY_RUN_HELP)
    pilot4_parser.add_argument("--hard-max-volume", type=float, default=0.20)
    pilot4_parser.add_argument("--default-volume", type=float, default=0.02)
    pilot4_parser.add_argument("--fade-in-seconds", type=float, default=0.25)
    pilot4_parser.add_argument("--fade-out-seconds", type=float, default=0.25)
    pilot4_parser.add_argument("--emergency-stop-file", type=Path)
    pilot4_parser.add_argument("--epoch-seconds", type=float, default=30.0)
    pilot4_parser.add_argument("--stride-seconds", type=float, default=30.0)
    pilot4_parser.add_argument("--enter-threshold", type=float, default=None,
                               help="Override gate enter threshold (default: preset-aware).")
    pilot4_parser.add_argument("--exit-threshold", type=float, default=None,
                               help="Override gate exit threshold (default: preset-aware).")
    _add_night_protocol_args(pilot4_parser)
    _add_night_arm_args(pilot4_parser)
    pilot4_parser.add_argument("--allow-short", action="store_true", help="Allow short smoke-test cueing runs.")
    pilot4_parser.add_argument("--quiet", action="store_true")
    _add_guard_hr_rule_args(pilot4_parser)
    _add_brainflow_args(pilot4_parser)
    _add_openmuse_lsl_args(pilot4_parser)
    _add_muse_sdk_args(pilot4_parser)

    pilot5_parser = subparsers.add_parser(
        "run-pilot5-full-night",
        help="Run an M8 Pilot 5 full night with TLR block plus puzzle cues.",
    )
    pilot5_parser.add_argument(
        "--source",
        choices=("amused", "brainflow", "openmuse", "sdk"),
        default="amused",
    )
    pilot5_parser.add_argument("--address", help="Muse BLE address. If omitted, discovery is used.")
    pilot5_parser.add_argument("--name-filter", default="Muse")
    pilot5_parser.add_argument("--preset", default="p1034")
    pilot5_parser.add_argument("--duration-hours", type=float, default=8.0)
    pilot5_parser.add_argument("--duration-seconds", type=float)
    pilot5_parser.add_argument("--output-dir", type=Path, required=True, help="Pilot output directory.")
    pilot5_parser.add_argument("--catalog", type=Path, required=True, help="Puzzle catalog .json path.")
    pilot5_parser.add_argument("--session", type=Path, required=True, help="Night puzzle session .json path.")
    pilot5_parser.add_argument("--assignment", type=Path, required=True, help="Cued/uncued assignment .json path.")
    pilot5_parser.add_argument("--cue-library", type=Path, required=True, help="Cue metadata library .json path.")
    pilot5_parser.add_argument("--tlr-block", type=Path, required=True, help="TLR block plan .json path.")
    pilot5_parser.add_argument("--calibration", type=Path, required=True, help="Volume calibration .json path.")
    pilot5_parser.add_argument("--device-name", help="Selected calibration device. Defaults to latest.")
    pilot5_parser.add_argument(
        "--backend",
        choices=("system", "afplay", "dry-run", "mock"),
        default="dry-run",
        help="Playback backend. Use system for the actual Pilot 5 sleep run.",
    )
    pilot5_parser.add_argument("--allow-dry-run", action="store_true", help=ALLOW_DRY_RUN_HELP)
    pilot5_parser.add_argument("--hard-max-volume", type=float, default=0.20)
    pilot5_parser.add_argument("--default-volume", type=float, default=0.02)
    pilot5_parser.add_argument("--fade-in-seconds", type=float, default=0.25)
    pilot5_parser.add_argument("--fade-out-seconds", type=float, default=0.25)
    pilot5_parser.add_argument("--emergency-stop-file", type=Path)
    pilot5_parser.add_argument("--epoch-seconds", type=float, default=30.0)
    pilot5_parser.add_argument("--stride-seconds", type=float, default=30.0)
    pilot5_parser.add_argument("--enter-threshold", type=float, default=None,
                               help="Override gate enter threshold (default: preset-aware).")
    pilot5_parser.add_argument("--exit-threshold", type=float, default=None,
                               help="Override gate exit threshold (default: preset-aware).")
    _add_night_protocol_args(pilot5_parser)
    _add_night_arm_args(pilot5_parser)
    pilot5_parser.add_argument("--allow-short", action="store_true", help="Allow short smoke-test cueing runs.")
    pilot5_parser.add_argument("--quiet", action="store_true")
    _add_guard_hr_rule_args(pilot5_parser)
    _add_brainflow_args(pilot5_parser)
    _add_openmuse_lsl_args(pilot5_parser)
    _add_muse_sdk_args(pilot5_parser)

    awakening_parser = subparsers.add_parser(
        "log-pilot4-awakening",
        help="Append a manual awakening marker to a Pilot 4 awakening_events.jsonl file.",
    )
    awakening_parser.add_argument("output", type=Path, help="awakening_events.jsonl path.")
    awakening_parser.add_argument("--event-type", default="awakening")
    awakening_parser.add_argument("--notes", default="")
    awakening_parser.add_argument(
        "--timestamp-utc",
        default="",
        help="When it happened, as an ISO time; without an offset it is read as UTC. Default: now.",
    )
    awakening_parser.add_argument(
        "--timestamp-seconds",
        type=_non_negative_float,
        default=None,
        help="When it happened in seconds from the night's first epoch, the time base of "
        "scheduler_events.jsonl. Without it score-cue-arousal uses --timestamp-utc.",
    )
    awakening_parser.add_argument(
        "--attributed-to-cue",
        choices=("yes", "no", "unsure"),
        default=None,
        help="Whether you think a cue woke you.",
    )

    cue_arousal_parser = subparsers.add_parser(
        "score-cue-arousal",
        help="Score cue-locked arousal proxies against paired quiet controls for a cueing night.",
    )
    cue_arousal_parser.add_argument(
        "recording_dir",
        type=Path,
        help="Pilot 4/5 output directory, or with --pilot3-report the recording it replayed.",
    )
    cue_arousal_parser.add_argument(
        "--pilot3-report",
        type=Path,
        help="Score this Pilot 3 report's planned and injected cues instead of a night's logs.",
    )
    cue_arousal_parser.add_argument("--summary", type=Path, help="pilot4_summary.json or pilot5_summary.json path.")
    cue_arousal_parser.add_argument("--scheduler-events", type=Path, help="scheduler_events.jsonl path.")
    cue_arousal_parser.add_argument("--audio-log", type=Path, help="audio_playback.jsonl path.")
    cue_arousal_parser.add_argument("--gate-log", type=Path, help="rem_gate_decisions.jsonl path.")
    cue_arousal_parser.add_argument("--awakening-events", type=Path, help="awakening_events.jsonl path.")
    cue_arousal_parser.add_argument(
        "--replay-input",
        type=Path,
        help="Raw file or recording to cut the windows from. Defaults to the recording directory.",
    )
    cue_arousal_parser.add_argument(
        "--preset",
        default=None,
        help="Recording preset (p21, p1034); says whether optics are expected. Defaults to "
        "the preset in the recording's metadata.json.",
    )
    cue_arousal_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where cue_arousal_events.jsonl and cue_arousal_report.json/.md go. Defaults to "
        "the recording directory, or the Pilot 3 report's directory.",
    )

    fidelity_parser = subparsers.add_parser(
        "protocol-fidelity",
        help="Compare a cueing night's scheduler, audio and gate logs with its protocol preset.",
    )
    fidelity_parser.add_argument(
        "recording_dir",
        type=Path,
        help="Pilot 4/5 output directory, or with --pilot3-report the recording it replayed.",
    )
    fidelity_parser.add_argument("--pilot3-report", type=Path, help="Read this Pilot 3 report's plan and epochs instead.")
    fidelity_parser.add_argument(
        "--protocol-preset",
        default=None,
        metavar="NAME",
        help="Compare with this preset instead of the one the night recorded.",
    )
    fidelity_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where protocol_fidelity.json and .md go. Defaults to the recording directory, or the "
        "Pilot 3 report's directory.",
    )

    list_cues_parser = subparsers.add_parser("list-cues", help="List cues from a cue metadata library.")
    list_cues_parser.add_argument("input", type=Path, help="Input cue library .json path.")
    list_cues_parser.add_argument("--protocol", choices=CUE_PROTOCOL_CHOICES)
    list_cues_parser.add_argument("--tag")

    presets_parser = subparsers.add_parser(
        "protocol-presets",
        help="List the protocol presets, or print one with its citation, parameters and hash.",
    )
    presets_parser.add_argument("name", nargs="?", help="A preset name; prints it as JSON.")
    presets_parser.add_argument("--json", action="store_true", help="Print the list as JSON.")

    create_tlr_parser = subparsers.add_parser(
        "create-tlr-cue",
        help="Create a default generated TLR cue library.",
    )
    create_tlr_parser.add_argument("--output", type=Path, required=True, help="Output cue library .json path.")
    create_tlr_parser.add_argument("--cue-id", default="tlr_soft_tone")
    create_tlr_parser.add_argument("--frequency-hz", type=float, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    create_tlr_parser.add_argument("--duration-seconds", type=float, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    create_tlr_parser.add_argument("--volume-hint", type=float, default=0.05)
    _add_protocol_preset_arg(create_tlr_parser)

    train_tlr_parser = subparsers.add_parser(
        "train-tlr-cue",
        help="Run pre-sleep TLR cue training and write structured events.",
    )
    train_tlr_parser.add_argument("cue_library", type=Path, help="Input TLR cue library .json path.")
    train_tlr_parser.add_argument("--cue-id", default="tlr_soft_tone")
    train_tlr_parser.add_argument("--output", type=Path, required=True, help="Output training summary .json path.")
    train_tlr_parser.add_argument("--event-log", type=Path, required=True, help="Output training events .jsonl path.")
    train_tlr_parser.add_argument("--session-id", default="tlr-training")
    train_tlr_parser.add_argument("--repetitions", type=int, default=3)
    train_tlr_parser.add_argument("--interval-seconds", type=float, default=2.0)
    train_tlr_parser.add_argument("--volume", type=float)
    train_tlr_parser.add_argument("--max-volume", type=float, default=0.20)
    train_tlr_parser.add_argument("--device-name")
    train_tlr_parser.add_argument(
        "--backend",
        choices=("system", "afplay", "dry-run", "mock"),
        default="dry-run",
    )
    train_tlr_parser.add_argument("--allow-dry-run", action="store_true", help=ALLOW_DRY_RUN_HELP)

    plan_tlr_parser = subparsers.add_parser(
        "plan-tlr-block",
        help="Plan a configurable TLR block before REM-gated puzzle cues.",
    )
    plan_tlr_parser.add_argument("cue_library", type=Path, help="Input TLR cue library .json path.")
    plan_tlr_parser.add_argument("--cue-id", default="tlr_soft_tone")
    plan_tlr_parser.add_argument("--output", type=Path, required=True, help="Output TLR block plan .json path.")
    plan_tlr_parser.add_argument("--repetitions", type=int, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    plan_tlr_parser.add_argument("--interval-seconds", type=float, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    plan_tlr_parser.add_argument("--post-block-pause-seconds", type=float, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    plan_tlr_parser.add_argument("--disabled", action="store_true")
    plan_tlr_parser.add_argument(
        "--training",
        type=Path,
        default=None,
        help="The summary train-tlr-cue wrote: refuse to plan when it played another tone than this library's "
        "cue, and record the training in the plan.",
    )
    _add_protocol_preset_arg(plan_tlr_parser)

    import_puzzles_parser = subparsers.add_parser(
        "import-puzzles",
        help="Import puzzle tasks from .csv or .json into a versioned puzzle catalog.",
    )
    import_puzzles_parser.add_argument("input", type=Path, help="Input .csv or .json puzzle task file.")
    import_puzzles_parser.add_argument("--output", type=Path, required=True, help="Output catalog .json path.")

    generate_puzzle_session_parser = subparsers.add_parser(
        "generate-puzzle-session",
        help="Generate a pre-sleep night puzzle session from eligible unsolved tasks.",
    )
    generate_puzzle_session_parser.add_argument("catalog", type=Path, help="Input puzzle catalog .json path.")
    generate_puzzle_session_parser.add_argument("--output", type=Path, required=True, help="Output session .json path.")
    generate_puzzle_session_parser.add_argument("--session-id", required=True)
    generate_puzzle_session_parser.add_argument("--count", type=int, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    generate_puzzle_session_parser.add_argument("--seed", type=int)
    generate_puzzle_session_parser.add_argument(
        "--include-known",
        action="store_true",
        help="Allow known-but-unsolved tasks in the generated session.",
    )
    _add_protocol_preset_arg(generate_puzzle_session_parser)

    attempt_parser = subparsers.add_parser(
        "record-puzzle-attempt",
        help="Append a timed pre-sleep puzzle attempt to a puzzle catalog.",
    )
    attempt_parser.add_argument("catalog", type=Path, help="Input puzzle catalog .json path.")
    attempt_parser.add_argument("--output", type=Path, help="Output catalog .json path. Defaults to input.")
    attempt_parser.add_argument("--puzzle-id", required=True)
    attempt_parser.add_argument("--response", required=True)
    attempt_parser.add_argument("--duration-seconds", type=float, required=True)
    attempt_parser.add_argument("--solved", action="store_true")
    attempt_parser.add_argument("--known-after", action="store_true")
    attempt_parser.add_argument("--notes", default="")

    association_parser = subparsers.add_parser(
        "record-association-check",
        help="Append a cue-to-puzzle association result to a generated puzzle session.",
    )
    association_parser.add_argument("session", type=Path, help="Input night puzzle session .json path.")
    association_parser.add_argument("--catalog", type=Path, required=True, help="Puzzle catalog .json path.")
    association_parser.add_argument("--output", type=Path, help="Output session .json path. Defaults to input.")
    association_parser.add_argument("--puzzle-id", required=True)
    association_parser.add_argument("--response", required=True)
    association_parser.add_argument("--notes", default="")

    dream_report_parser = subparsers.add_parser(
        "record-dream-report",
        help="Capture a structured morning dream report for a night puzzle session.",
    )
    dream_report_parser.add_argument("session", type=Path, help="Input night puzzle session .json path.")
    dream_report_parser.add_argument("--output", type=Path, required=True, help="Output dream report .json path.")
    dream_report_parser.add_argument("--lucid", choices=("yes", "no"), required=True)
    dream_report_parser.add_argument(
        "--cues-heard",
        choices=("yes", "no"),
        help="Heard any sound in the night. Give it, --cue-perception or both.",
    )
    dream_report_parser.add_argument(
        "--cue-perception",
        choices=("not_perceived", "in_dream", "awake", "unsure", "unknown"),
        help="Where a sound was heard: not at all, in a dream, awake, heard but unsure where, or unknown.",
    )
    dream_report_parser.add_argument(
        "--woke-from-sound",
        choices=("yes", "no", "unknown"),
        default="unknown",
        help="Whether a sound woke you (default unknown).",
    )
    dream_report_parser.add_argument(
        "--eye-signal-attempt",
        action="append",
        default=[],
        metavar="TIME",
        help="When you think you made the eye signal (LRLR), HH:MM or an ISO time. Repeat for each. "
        "A marking only: nothing detects the signal.",
    )
    dream_report_parser.add_argument("--confidence", type=float, required=True, help="Self-report confidence 0.0-1.0.")
    dream_report_parser.add_argument("--dream-text", required=True, help="Free-text dream recall.")
    dream_report_parser.add_argument(
        "--puzzle-link",
        action="append",
        default=[],
        metavar="PUZZLE_ID=TEXT",
        help="Link a session puzzle to dream content. Repeat for multiple puzzles.",
    )
    dream_report_parser.add_argument("--catalog", type=Path, help="Optional puzzle catalog for cue IDs.")
    dream_report_parser.add_argument("--report-id", default="")
    dream_report_parser.add_argument("--notes", default="")

    cue_recall_parser = subparsers.add_parser(
        "record-cue-recall",
        help="Play each puzzle's sound and record which puzzle you think it belongs to, before the retest.",
    )
    cue_recall_parser.add_argument("session", type=Path, help="Input night puzzle session .json path.")
    cue_recall_parser.add_argument("--catalog", type=Path, required=True, help="Puzzle catalog .json path.")
    cue_recall_parser.add_argument("--assignment", type=Path, required=True, help="Cued/uncued assignment .json path.")
    cue_recall_parser.add_argument("--cue-library", type=Path, required=True, help="Cue metadata library .json path.")
    cue_recall_parser.add_argument("--calibration", type=Path, required=True, help="Volume calibration .json path.")
    cue_recall_parser.add_argument("--device-name", help="Selected calibration device. Defaults to latest.")
    cue_recall_parser.add_argument("--output", type=Path, required=True, help="Output cue recall .json path.")
    cue_recall_parser.add_argument(
        "--audio-log",
        type=Path,
        help="Playback log .jsonl. Defaults to the output path with _audio.jsonl.",
    )
    cue_recall_parser.add_argument(
        "--backend",
        choices=("system", "afplay", "dry-run", "mock"),
        default="system",
        help="Playback backend. system and afplay need afplay and exit with code 2 without it.",
    )
    cue_recall_parser.add_argument("--allow-dry-run", action="store_true", help=ALLOW_DRY_RUN_HELP)
    cue_recall_parser.add_argument("--hard-max-volume", type=float, default=0.20)
    cue_recall_parser.add_argument("--default-volume", type=float, default=0.02)
    cue_recall_parser.add_argument("--notes", default="")

    retest_parser = subparsers.add_parser(
        "record-puzzle-retest",
        help="Capture morning puzzle retest results for a night puzzle session.",
    )
    retest_parser.add_argument("session", type=Path, help="Input night puzzle session .json path.")
    retest_parser.add_argument("--catalog", type=Path, required=True, help="Puzzle catalog .json path.")
    retest_parser.add_argument("--assignment", type=Path, required=True, help="Cued/uncued assignment .json path.")
    retest_parser.add_argument("--output", type=Path, required=True, help="Output morning retest .json path.")
    retest_parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="PUZZLE_ID=RESPONSE",
        help="Retest response. Repeat once per session puzzle; empty response is allowed.",
    )
    retest_parser.add_argument(
        "--solved",
        action="append",
        default=[],
        metavar="PUZZLE_ID",
        help="Mark a retest puzzle as solved. Unlisted result puzzles are saved as unsolved.",
    )
    retest_parser.add_argument(
        "--duration",
        action="append",
        default=[],
        metavar="PUZZLE_ID=SECONDS",
        help="Retest duration in seconds. Repeat once per result puzzle.",
    )
    retest_parser.add_argument(
        "--confidence",
        action="append",
        default=[],
        metavar="PUZZLE_ID=0.0-1.0",
        help="Self-report confidence. Repeat once per result puzzle.",
    )
    retest_parser.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="PUZZLE_ID=TEXT",
        help="Optional per-puzzle retest note.",
    )
    retest_parser.add_argument("--retest-id", default="")
    retest_parser.add_argument(
        "--wave",
        choices=("morning", "day7", "day10"),
        default="morning",
        help="The retest of the night's puzzles this is: the morning after, a week or ten days later.",
    )
    retest_parser.add_argument("--notes", default="")

    analysis_parser = subparsers.add_parser(
        "analyze-cued-uncued",
        help="Generate a cued-vs-uncued morning analysis report.",
    )
    analysis_parser.add_argument("session", type=Path, help="Input night puzzle session .json path.")
    analysis_parser.add_argument("--assignment", type=Path, required=True, help="Cued/uncued assignment .json path.")
    analysis_parser.add_argument(
        "--retest",
        type=Path,
        action="append",
        required=True,
        help="Puzzle retest .json path. Repeat for the delayed waves (day7, day10), one file per wave.",
    )
    analysis_parser.add_argument(
        "--coding",
        type=Path,
        action="append",
        default=[],
        help="A rater's incorporation coding sheet of this night (record-incorporation-coding). Repeat per "
        "rater; rater codes decide incorporation in place of the dream report's own links.",
    )
    analysis_parser.add_argument("--output", type=Path, required=True, help="Output analysis report .json path.")
    analysis_parser.add_argument("--dream-report", type=Path, help="Optional morning dream report .json path.")
    analysis_parser.add_argument("--scheduler-events", type=Path, help="Optional TMR scheduler events .jsonl path.")
    analysis_parser.add_argument("--markdown-output", type=Path, help="Optional human-readable .md report path.")
    analysis_parser.add_argument("--analysis-id", default="")
    analysis_parser.add_argument("--min-group-size", type=int, default=5)
    analysis_parser.add_argument(
        "--cue-recall",
        type=Path,
        help="Morning sound-to-puzzle recall test .json (record-cue-recall or the app's Morning tab).",
    )
    analysis_parser.add_argument(
        "--protocol-metadata",
        type=Path,
        help="A .json with the night's protocol block: its metadata.json, Pilot 4/5 summary or Pilot 3 "
        "report. Without it the assignment's block is used when it has one.",
    )

    assignment_parser = subparsers.add_parser(
        "assign-puzzle-cues",
        help="Randomize a night puzzle session into cued and uncued groups.",
    )
    assignment_parser.add_argument("session", type=Path, help="Input night puzzle session .json path.")
    assignment_parser.add_argument("--output", type=Path, required=True, help="Output cue assignment .json path.")
    assignment_parser.add_argument("--seed", type=int, required=True)
    assignment_parser.add_argument(
        "--cued-count",
        type=int,
        help="Override the number of cued puzzles. Defaults to the protocol preset's, which is half "
        "of the session tasks unless the preset says otherwise.",
    )
    _add_protocol_preset_arg(assignment_parser)

    plan_schedule_parser = subparsers.add_parser(
        "plan-night-schedule",
        help="Seal an N-of-1 series: blocks of a cue, a sham and a no-cue night in a counterbalanced order.",
    )
    plan_schedule_parser.add_argument("--blocks", type=int, required=True, help="Blocks of one night per arm.")
    plan_schedule_parser.add_argument(
        "--washout-nights", type=int, default=0, help="Washout nights between two blocks (default 0)."
    )
    plan_schedule_parser.add_argument(
        "--seed",
        type=int,
        help="Seed of the order. Default: drawn at random and kept only in the schedule file.",
    )
    plan_schedule_parser.add_argument("--schedule-id", help="Default: series-<UTC time>.")
    plan_schedule_parser.add_argument("--night-prefix", default="night-", help="Night IDs are <prefix>001, <prefix>002, ...")
    plan_schedule_parser.add_argument(
        "--sham-cue-id",
        default="sham_soft_tone",
        help="The untrained cue sham nights play (default sham_soft_tone, in the starter cue library).",
    )
    plan_schedule_parser.add_argument(
        "--output",
        type=Path,
        help="Schedule .json path. Default: data/protocol/night_schedules/<schedule-id>.json; the sealed "
        "companion goes next to it as <name>.sealed.json.",
    )

    show_schedule_parser = subparsers.add_parser(
        "show-night-schedule",
        help="Show a night schedule's nights and check it against its seal; arms only once unsealed.",
    )
    show_schedule_parser.add_argument("schedule", type=Path, help="Night schedule .json path.")
    show_schedule_parser.add_argument("--sealed", type=Path, help="Sealed companion. Default: <name>.sealed.json next to it.")
    show_schedule_parser.add_argument("--json", action="store_true", help="Print it as JSON.")

    unseal_schedule_parser = subparsers.add_parser(
        "unseal-night-schedule",
        help="Check a night schedule against its seal, record when it was unsealed and show the arms.",
    )
    unseal_schedule_parser.add_argument("schedule", type=Path, help="Night schedule .json path.")
    unseal_schedule_parser.add_argument("--sealed", type=Path, help="Sealed companion. Default: <name>.sealed.json next to it.")
    unseal_schedule_parser.add_argument("--json", action="store_true", help="Print it as JSON.")

    prereg_parser = subparsers.add_parser(
        "prereg-template",
        help="Write an OSF-style preregistration of a sealed series: its design and commitment, no arm, seed or salt.",
    )
    prereg_parser.add_argument("schedule", type=Path, help="Night schedule .json path.")
    prereg_parser.add_argument("--sealed", type=Path, help="Sealed companion. Default: <name>.sealed.json next to it.")
    prereg_parser.add_argument(
        "--protocol-metadata",
        type=Path,
        help="A .json with the series' protocol block, such as a night's puzzle session or assignment. "
        "Without it the block of --protocol-preset.",
    )
    _add_protocol_preset_arg(prereg_parser)
    prereg_parser.add_argument("--output", type=Path, help="Markdown path. Default: print it.")

    blind_coding_parser = subparsers.add_parser(
        "export-blind-coding",
        help="Write a night's dream text and puzzles for a rater, with nothing about cues, the arm or the assignment.",
    )
    blind_coding_parser.add_argument("session", type=Path, help="Input night puzzle session .json path.")
    blind_coding_parser.add_argument("--dream-report", type=Path, required=True, help="The night's dream report .json.")
    blind_coding_parser.add_argument("--catalog", type=Path, required=True, help="Puzzle catalog .json path.")
    blind_coding_parser.add_argument("--output", type=Path, required=True, help="Packet .json path.")
    blind_coding_parser.add_argument("--markdown-output", type=Path, help="The packet as Markdown, for a person to read.")

    record_coding_parser = subparsers.add_parser(
        "record-incorporation-coding",
        help="Record a rater's incorporation codes for a blind coding packet.",
    )
    record_coding_parser.add_argument("packet", type=Path, help="Packet .json from export-blind-coding.")
    record_coding_parser.add_argument("--rater-id", required=True, help="Who coded it, such as rater-a.")
    record_coding_parser.add_argument(
        "--code",
        action="append",
        default=[],
        metavar="PUZZLE_ID=CATEGORIES",
        help="A puzzle's categories, comma-separated: sound_cue, direct_conceptual, indirect_conceptual, or none. "
        "Repeat once per packet puzzle.",
    )
    record_coding_parser.add_argument(
        "--note", action="append", default=[], metavar="PUZZLE_ID=TEXT", help="Optional note on a puzzle's code."
    )
    record_coding_parser.add_argument("--notes", default="")
    record_coding_parser.add_argument("--output", type=Path, required=True, help="Coding sheet .json path.")

    compare_codings_parser = subparsers.add_parser(
        "compare-incorporation-codings",
        help="Percent agreement and Cohen's kappa per category between two raters' coding sheets.",
    )
    compare_codings_parser.add_argument(
        "codings", type=Path, nargs="+", help="Coding sheets of two raters, one or more nights each."
    )
    compare_codings_parser.add_argument("--output", type=Path, help="Agreement .json path.")

    nof1_parser = subparsers.add_parser(
        "analyze-nof1",
        help="Describe an unsealed N-of-1 series by arm, with permutation intervals and a SCRIBE/CENT checklist.",
    )
    nof1_parser.add_argument("schedule", type=Path, help="Night schedule .json path, unsealed.")
    nof1_parser.add_argument("--sealed", type=Path, help="Sealed companion. Default: <name>.sealed.json next to it.")
    nof1_parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("data/protocol"),
        help="Where the nights' puzzle sessions and assignments are (default data/protocol).",
    )
    nof1_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/reports"),
        help="Where the dream reports, retests, sound checks and coding sheets are (default data/reports).",
    )
    nof1_parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("data/recordings"),
        help="Searched for the series' pilot4_summary.json and pilot5_summary.json (default data/recordings).",
    )
    nof1_parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        default=[],
        help="A night's Pilot 4/5 summary or Pilot 3 report; wins over one found for the same night. Repeat.",
    )
    nof1_parser.add_argument(
        "--coding", type=Path, action="append", default=[], help="A coding sheet outside --reports-dir. Repeat."
    )
    nof1_parser.add_argument("--seed", type=int, default=0, help="Seed of the drawn arm orders when not all are used (default 0).")
    nof1_parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Use every arm order the schedule could have drawn up to this many, else draw this many (default 10000).",
    )
    nof1_parser.add_argument("--min-nights-per-arm", type=int, default=5)
    nof1_parser.add_argument("--primary-outcome", default=None, help="The preregistered outcome, marked primary.")
    nof1_parser.add_argument(
        "--primary-contrast",
        choices=("cue-no_cue", "cue-sham", "sham-no_cue"),
        default=None,
        help="The preregistered contrast (default cue-no_cue with --primary-outcome).",
    )
    nof1_parser.add_argument("--output", type=Path, required=True, help="Output .json path.")
    nof1_parser.add_argument("--markdown-output", type=Path, help="Output .md path.")

    meditation_plan_parser = subparsers.add_parser(
        "meditation-plan",
        help="Write a counterbalanced A/B meditation blocks.json template.",
    )
    meditation_plan_parser.add_argument(
        "--conditions", required=True, help="Two comma-separated condition labels, e.g. focus,open."
    )
    meditation_plan_parser.add_argument("--blocks", type=int, default=4, help="4, 8 or 12 (whole ABBA groups).")
    meditation_plan_parser.add_argument("--block-minutes", type=float, default=8.0)
    meditation_plan_parser.add_argument("--settle-seconds", type=float, default=60.0)
    meditation_plan_parser.add_argument("--seed", type=int, required=True)
    meditation_plan_parser.add_argument(
        "--output",
        type=Path,
        help="Output blocks .json path. Defaults to data/protocol/meditation/blocks_<timestamp>.json.",
    )

    meditation_analysis_parser = subparsers.add_parser(
        "analyze-meditation",
        help="Complexity, 1/f, DFA and EMG analysis of one meditation recording.",
    )
    meditation_analysis_parser.add_argument("recording_dir", type=Path)
    meditation_analysis_parser.add_argument("--blocks", type=Path, required=True, help="blocks.json for this recording.")
    meditation_analysis_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to data/reports/meditation/<recording name>/.",
    )
    meditation_analysis_parser.add_argument("--epoch-seconds", type=float, default=10.0)
    meditation_analysis_parser.add_argument("--trim-block-start", type=float, default=30.0)
    meditation_analysis_parser.add_argument(
        "--emg-indicator",
        choices=("auto", "emg_power_55_95", "emg_power_30_45"),
        default="auto",
    )
    meditation_analysis_parser.add_argument("--no-lyapunov", action="store_true")
    meditation_analysis_parser.add_argument(
        "--no-polar",
        action="store_true",
        help="Ignore a polar/ folder (Polar H10 breathing and HRV) in the recording.",
    )

    meditation_aggregate_parser = subparsers.add_parser(
        "aggregate-meditation",
        help="Cross-session table of meditation A-B differences (inference unit: session).",
    )
    meditation_aggregate_parser.add_argument("summaries", type=Path, nargs="+", help="summary.json files.")
    meditation_aggregate_parser.add_argument("--output", type=Path, required=True, help="Output .json path.")
    meditation_aggregate_parser.add_argument("--min-sessions", type=int, default=8)
    meditation_aggregate_parser.add_argument("--seed", type=int, default=0)

    calibration_run_parser = subparsers.add_parser(
        "calibration-run",
        help="Voice-guided ~23 min Muse + Polar H10 recording for the EMG and breathing checks.",
    )
    calibration_run_parser.add_argument(
        "--output-dir", type=Path, help="Defaults to data/recordings/session/<timestamp>_calibration/."
    )
    calibration_run_parser.add_argument("--address", help="Muse BLE address. If omitted, discovery is used.")
    calibration_run_parser.add_argument("--polar-address", help="Polar H10 BLE address. If omitted, name discovery is used.")
    calibration_run_parser.add_argument(
        "--preset",
        choices=("p1034", "p21"),
        default="p21",
        help="Headband preset for the calibration recording. p21 has no optics, whose 64 Hz line "
        "gets into the EEG on p1034 (default: p21).",
    )
    calibration_run_parser.add_argument(
        "--battery-pull",
        action="store_true",
        help="In the H10 minute, also take the H10 battery out for ~10 s (forces a reconnect and a clock reset).",
    )
    calibration_run_parser.add_argument(
        "--line-check",
        action="store_true",
        help="First record 2 min each of p1034, p21, p1034, p21 to see whether the 64 Hz line comes from the optics.",
    )
    _add_calibration_voice_args(calibration_run_parser)

    calibration_guide_parser = subparsers.add_parser(
        "calibration-guide",
        help="Speak the calibration protocol along a recording that is starting (used by the app).",
    )
    calibration_guide_parser.add_argument("recording_dir", type=Path)
    calibration_guide_parser.add_argument(
        "--recorder-pid", type=int, help="Stop early when this process exits."
    )
    calibration_guide_parser.add_argument(
        "--battery-pull", action="store_true", help="In the H10 minute, also take the H10 battery out for ~10 s."
    )
    _add_calibration_voice_args(calibration_guide_parser)

    calibration_report_parser = subparsers.add_parser(
        "calibration-report",
        help="EMG, breathing and H10 report for one calibration-run recording.",
    )
    calibration_report_parser.add_argument("recording_dir", type=Path)
    calibration_report_parser.add_argument(
        "--output-dir", type=Path, help="Defaults to data/reports/calibration/<recording name>/."
    )

    record_parser = subparsers.add_parser("record", help="Record an overnight Muse session.")
    record_parser.add_argument(
        "--source",
        choices=("amused", "brainflow", "openmuse", "sdk"),
        default="amused",
    )
    record_parser.add_argument("--address", help="Muse BLE address. If omitted, discovery is used.")
    record_parser.add_argument("--name-filter", default="Muse")
    record_parser.add_argument("--preset", default="p1034")
    record_parser.add_argument("--duration-hours", type=float, default=8.0)
    record_parser.add_argument("--duration-seconds", type=float)
    record_parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Recording directory. Relative paths resolve under the current working "
            "directory, or under the project checkout when launched via macOS Python.app."
        ),
    )
    record_parser.add_argument("--allow-short", action="store_true", help="Allow short smoke-test recordings.")
    record_parser.add_argument(
        "--duration-from-first-frame",
        action="store_true",
        help="Count the duration from the first Muse frame, not from the start (timed plans).",
    )
    record_parser.add_argument(
        "--no-data-timeout-seconds",
        type=float,
        default=30.0,
        help="Seconds without a frame before a reconnect attempt is triggered.",
    )
    record_parser.add_argument(
        "--max-reconnect-attempts",
        type=int,
        default=5,
        help="Cumulative reconnect attempts allowed for the whole recording before giving up.",
    )
    record_parser.add_argument("--quiet", action="store_true")
    record_parser.add_argument(
        "--with-polar",
        action="store_true",
        help="Also record a Polar H10 in a separate record-polar process (its failures never stop the Muse).",
    )
    record_parser.add_argument("--polar-address", help="Polar H10 BLE address. If omitted, name discovery is used.")

    polar_parser = subparsers.add_parser(
        "record-polar",
        help="Record a Polar H10 (HR/RR, ECG, chest ACC) into <output-dir>/polar/.",
    )
    polar_parser.add_argument("--output-dir", type=Path, required=True, help="Session directory, shared with the Muse recording.")
    polar_duration = polar_parser.add_mutually_exclusive_group(required=True)
    polar_duration.add_argument("--duration-seconds", type=float)
    polar_duration.add_argument("--duration-hours", type=float)
    polar_parser.add_argument("--address", help="Polar H10 BLE address. If omitted, name discovery is used.")
    polar_parser.add_argument("--name-filter", default="Polar H10")
    polar_parser.add_argument("--no-ecg", action="store_true")
    polar_parser.add_argument("--no-acc", action="store_true")
    polar_parser.add_argument("--acc-rate", type=int, default=50, choices=(25, 50, 100, 200))
    polar_parser.add_argument("--acc-range", type=int, default=2, choices=(2, 4, 8))
    polar_parser.add_argument("--max-reconnect-attempts", type=int, default=1000)

    decode_polar_parser = subparsers.add_parser(
        "decode-polar",
        help="Rebuild decoded Polar files from <session_dir>/polar/raw_notifications.jsonl.",
    )
    decode_polar_parser.add_argument("session_dir", type=Path)

    optics_agreement_parser = subparsers.add_parser(
        "analyze-optics-agreement",
        help="Agreement of the optics heart rate with the Polar H10 ECG of a p1034 recording.",
    )
    optics_agreement_parser.add_argument("recording_dir", type=Path)
    optics_agreement_parser.add_argument("--epoch-seconds", type=float, default=30.0)
    optics_agreement_parser.add_argument(
        "--output-dir", type=Path, help="Defaults to data/reports/optics_agreement/<recording name>/."
    )
    _add_brainflow_args(record_parser)
    _add_openmuse_lsl_args(record_parser)
    _add_muse_sdk_args(record_parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in PROTOCOL_PRESET_COMMANDS:
        # An unknown preset, or one that documents a protocol REMind cannot run, stops
        # the command before it reads or writes anything.
        from muse_tmr.protocol import resolve_protocol_preset

        try:
            resolve_protocol_preset(args.protocol_preset)
        except (KeyError, ValueError) as exc:
            print(exc.args[0] if exc.args else str(exc), file=sys.stderr)
            return 2
    if args.command in AUDIO_COMMANDS:
        # Refuse before any input is read, any file is written or a headband is touched.
        # Pilot 4/5 resolve again inside run_pilot4_cueing_night, which records both names.
        from muse_tmr.audio import AudioBackendFallbackError, resolve_audio_backend

        try:
            args.audio_resolution = resolve_audio_backend(
                args.backend, allow_dry_run_fallback=args.allow_dry_run
            )
        except AudioBackendFallbackError as exc:
            print(f"audio preflight failed: {exc}", file=sys.stderr)
            return 2

    if args.command == "status":
        print(BANNER)
        return 0
    if args.command == "app":
        return _run_app(args)
    if args.command == "discover":
        return asyncio.run(_discover(args))
    if args.command == "stream":
        return asyncio.run(_stream(args))
    if args.command == "diagnose-blink-artifacts":
        return asyncio.run(_diagnose_blink_artifacts(args))
    if args.command == "compare-source-diagnostics":
        return _compare_source_diagnostics(args)
    if args.command == "inspect-blink-channels":
        return _inspect_blink_channels(args)
    if args.command == "replay":
        return asyncio.run(_replay(args))
    if args.command == "synthesize-night":
        return _synthesize_night(args)
    if args.command == "demo":
        return _demo(args)
    if args.command == "annotate-template":
        return asyncio.run(_annotate_template(args))
    if args.command == "train-rem-classifier":
        return _train_rem_classifier(args)
    if args.command == "play-test-cue":
        return _play_test_cue(args)
    if args.command == "calibrate-volume":
        return _calibrate_volume(args)
    if args.command == "create-cue-library":
        return _create_cue_library(args)
    if args.command == "validate-cue-library":
        return _validate_cue_library(args)
    if args.command == "validate-pilot1-recording":
        return _validate_pilot1_recording(args)
    if args.command == "validate-pilot2-calibration":
        return _validate_pilot2_calibration(args)
    if args.command == "simulate-replay-cues":
        return asyncio.run(_simulate_replay_cues(args))
    if args.command == "run-pilot4-cueing":
        return asyncio.run(_run_pilot4_cueing(args))
    if args.command == "run-pilot5-full-night":
        return asyncio.run(_run_pilot5_full_night(args))
    if args.command == "log-pilot4-awakening":
        return _log_pilot4_awakening(args)
    if args.command == "score-cue-arousal":
        return asyncio.run(_score_cue_arousal(args))
    if args.command == "protocol-fidelity":
        return _protocol_fidelity(args)
    if args.command == "protocol-presets":
        return _protocol_presets(args)
    if args.command == "list-cues":
        return _list_cues(args)
    if args.command == "create-tlr-cue":
        return _create_tlr_cue(args)
    if args.command == "train-tlr-cue":
        return _train_tlr_cue(args)
    if args.command == "plan-tlr-block":
        return _plan_tlr_block(args)
    if args.command == "import-puzzles":
        return _import_puzzles(args)
    if args.command == "generate-puzzle-session":
        return _generate_puzzle_session(args)
    if args.command == "record-puzzle-attempt":
        return _record_puzzle_attempt(args)
    if args.command == "record-association-check":
        return _record_association_check(args)
    if args.command == "record-dream-report":
        return _record_dream_report(args)
    if args.command == "record-cue-recall":
        return _record_cue_recall(args)
    if args.command == "record-puzzle-retest":
        return _record_puzzle_retest(args)
    if args.command == "analyze-cued-uncued":
        return _analyze_cued_uncued(args)
    if args.command == "assign-puzzle-cues":
        return _assign_puzzle_cues(args)
    if args.command == "plan-night-schedule":
        return _plan_night_schedule(args)
    if args.command == "show-night-schedule":
        return _show_night_schedule(args)
    if args.command == "unseal-night-schedule":
        return _unseal_night_schedule(args)
    if args.command == "prereg-template":
        return _prereg_template(args)
    if args.command == "export-blind-coding":
        return _export_blind_coding(args)
    if args.command == "record-incorporation-coding":
        return _record_incorporation_coding(args)
    if args.command == "compare-incorporation-codings":
        return _compare_incorporation_codings(args)
    if args.command == "analyze-nof1":
        return _analyze_nof1(args)
    if args.command == "meditation-plan":
        return _meditation_plan(args)
    if args.command == "analyze-meditation":
        return asyncio.run(_analyze_meditation(args))
    if args.command == "aggregate-meditation":
        return _aggregate_meditation(args)
    if args.command == "calibration-run":
        return _calibration_run(args)
    if args.command == "calibration-guide":
        return _calibration_guide(args)
    if args.command == "calibration-report":
        return asyncio.run(_calibration_report(args))
    if args.command == "record":
        return asyncio.run(_record(args))
    if args.command == "record-polar":
        return asyncio.run(_record_polar(args))
    if args.command == "decode-polar":
        return _decode_polar(args)
    if args.command == "analyze-optics-agreement":
        return asyncio.run(_analyze_optics_agreement(args))

    parser.print_help()
    return 0


def _run_app(args: argparse.Namespace) -> int:
    from muse_tmr.app import AppConfig, run_local_app

    return run_local_app(
        AppConfig(
            host=args.host,
            port=args.port,
            source=args.source,
            address=args.address,
            name_filter=args.name_filter,
            preset=args.preset,
            mock_scenario=args.mock_scenario,
            mock_interval_seconds=args.mock_interval_seconds,
            gate_stability_seconds=args.contact_stability_seconds,
            auto_update=args.auto_update,
            cue_backend=args.cue_backend,
            allow_dry_run=args.allow_dry_run,
        )
    )


async def _discover(args: argparse.Namespace) -> int:
    source = _build_source(args, duration_seconds=0)
    devices = await source.discover()
    for device in devices:
        print(f"{device.name}\t{device.address}\trssi={device.rssi}")
    return 0 if devices else 1


async def _stream(args: argparse.Namespace) -> int:
    source = _build_source(args, duration_seconds=args.duration_seconds)
    frame_count = 0
    modality_counts = {}
    metadata = None
    try:
        metadata = await source.connect()
        async for frame in source.stream():
            frame_count += 1
            for modality in frame.modalities():
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
    except Exception as exc:
        if args.debug_stats:
            _print_stream_diagnostics(source)
        print(f"stream failed error={exc}", file=sys.stderr)
        return 1
    finally:
        await source.stop()

    print(
        f"stream complete source={metadata.source_name} "
        f"device={metadata.device_name} frames={frame_count} modalities={modality_counts}"
    )
    if args.debug_stats:
        _print_stream_diagnostics(source)
    return 0


def _print_stream_diagnostics(source) -> None:
    diagnostics = source.diagnostics() if hasattr(source, "diagnostics") else {}
    print(f"stream diagnostics={json.dumps(diagnostics, sort_keys=True)}")


async def _diagnose_blink_artifacts(args: argparse.Namespace) -> int:
    from muse_tmr.features import (
        ArtifactDiagnosticConfig,
        analyze_blink_artifact_phases,
        default_blink_artifact_phases,
    )

    phases = default_blink_artifact_phases(
        settle_seconds=args.settle_seconds,
        eyes_open_baseline_seconds=args.eyes_open_baseline_seconds,
        blink_seconds=args.blink_seconds,
        recovery_open_seconds=args.recovery_open_seconds,
        jaw_clench_seconds=args.jaw_clench_seconds,
        head_movement_seconds=args.head_movement_seconds,
        eyes_closed_baseline_seconds=args.eyes_closed_baseline_seconds,
    )
    config = ArtifactDiagnosticConfig(
        sample_rate_hz=args.sample_rate_hz,
        center_window_seconds=args.center_window_seconds,
        highpass_cutoff_hz=args.highpass_cutoff_hz,
        window_seconds=args.window_seconds,
    )
    output_path = (
        _resolve_output_path(args.output)
        if args.output is not None
        else _default_blink_diagnostic_output(args.source)
    )

    source = _build_source(args, duration_seconds=0)
    phase_frames = {}
    metadata = None
    stream_queue = asyncio.Queue()
    stream_task = None
    try:
        metadata = await source.connect()
        stream_task = asyncio.create_task(
            _pump_diagnostic_stream(source.stream(), stream_queue)
        )
        for phase in phases:
            if not args.quiet:
                print(
                    f"PHASE_READY phase={phase.name} "
                    f"duration={phase.duration_seconds:.1f}s "
                    f"instruction={phase.instruction}"
                )
            if not args.non_interactive:
                await asyncio.to_thread(input, "Press Enter to start this phase...")
            if not args.quiet:
                print(f"PHASE_START phase={phase.name}")
            _drain_diagnostic_queue(stream_queue)
            frames = await _collect_diagnostic_phase_frames(
                stream_queue,
                stream_task,
                phase.duration_seconds,
            )
            phase_frames[phase.name] = frames
            if not args.quiet:
                print(
                    f"PHASE_DONE phase={phase.name} "
                    f"frames={len(frames)} eeg_frames={_eeg_frame_count(frames)}"
                )
    finally:
        if stream_task is not None:
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
        await source.stop()

    report = analyze_blink_artifact_phases(
        phase_frames,
        source=metadata.source_name if metadata is not None else args.source,
        phases=phases,
        config=config,
        source_metadata=_source_metadata_to_dict(metadata),
        source_diagnostics=_source_diagnostics(source),
        session_summary=_diagnostic_session_summary(output_path.stem, phase_frames),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "blink artifact diagnostic complete "
        f"source={report.source} "
        f"blink_detected={report.blink_summary.get('detected')} "
        f"closed_eyes_present={report.closed_eyes_summary.get('present')} "
        f"output={output_path}"
    )
    return 0


def _compare_source_diagnostics(args: argparse.Namespace) -> int:
    from muse_tmr.reports import (
        compare_source_diagnostic_reports,
        format_source_diagnostic_markdown,
        save_source_diagnostic_comparison,
    )

    report_paths = tuple(_resolve_output_path(path) for path in args.reports)
    rows = compare_source_diagnostic_reports(report_paths)
    print(format_source_diagnostic_markdown(rows))
    if args.output is not None:
        output_path = save_source_diagnostic_comparison(
            rows,
            _resolve_output_path(args.output),
            output_format=args.format,
        )
        print(f"source diagnostic comparison saved output={output_path}")
    return 0


def _inspect_blink_channels(args: argparse.Namespace) -> int:
    from muse_tmr.reports import (
        format_blink_channel_inspection_markdown,
        inspect_blink_channel_reports,
        save_blink_channel_inspection,
    )

    report_paths = tuple(_resolve_output_path(path) for path in args.reports)
    rows = inspect_blink_channel_reports(report_paths, phases=args.phases)
    print(format_blink_channel_inspection_markdown(rows))
    if args.output is not None:
        output_path = save_blink_channel_inspection(
            rows,
            _resolve_output_path(args.output),
            output_format=args.format,
        )
        print(f"blink channel inspection saved output={output_path}")
    return 0


async def _pump_diagnostic_stream(stream, queue: asyncio.Queue) -> None:
    async for frame in stream:
        await queue.put(frame)


async def _collect_diagnostic_phase_frames(
    queue: asyncio.Queue,
    stream_task: asyncio.Task,
    duration_seconds: float,
) -> tuple:
    frames = []
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            frame = await asyncio.wait_for(queue.get(), timeout=min(0.5, max(0.05, remaining)))
        except asyncio.TimeoutError:
            if stream_task.done():
                exc = stream_task.exception()
                if exc is not None:
                    raise exc
                break
            continue
        frames.append(frame)
    return tuple(frames)


def _drain_diagnostic_queue(queue: asyncio.Queue) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


def _eeg_frame_count(frames: Sequence[object]) -> int:
    return sum(1 for frame in frames if getattr(frame, "eeg", None) is not None)


def _source_metadata_to_dict(metadata) -> Mapping[str, object]:
    if metadata is None:
        return {}
    return {
        "source_name": metadata.source_name,
        "device_name": metadata.device_name,
        "device_id": metadata.device_id,
        "capabilities": dict(metadata.capabilities),
        "metadata": dict(metadata.metadata or {}),
    }


def _source_diagnostics(source) -> Mapping[str, object]:
    diagnostics = source.diagnostics() if hasattr(source, "diagnostics") else {}
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def _diagnostic_session_summary(
    session_id: str,
    phase_frames: Mapping[str, Sequence[object]],
) -> Mapping[str, object]:
    phase_summaries = {}
    total_modality_counts = {}
    total_eeg_sample_counts = {}
    total_frame_count = 0
    for phase_name, frames in phase_frames.items():
        modality_counts = {}
        eeg_sample_counts = {}
        for frame in frames:
            total_frame_count += 1
            for modality in frame.modalities():
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
                total_modality_counts[modality] = total_modality_counts.get(modality, 0) + 1
            if frame.eeg is not None:
                for channel, values in frame.eeg.channels_uv.items():
                    count = len(values)
                    eeg_sample_counts[channel] = eeg_sample_counts.get(channel, 0) + count
                    total_eeg_sample_counts[channel] = (
                        total_eeg_sample_counts.get(channel, 0) + count
                    )
        phase_summaries[phase_name] = {
            "frame_count": len(frames),
            "modality_counts": modality_counts,
            "eeg_sample_counts": eeg_sample_counts,
        }
    return {
        "session_id": session_id,
        "total_frame_count": total_frame_count,
        "total_modality_counts": total_modality_counts,
        "total_eeg_sample_counts": total_eeg_sample_counts,
        "phases": phase_summaries,
    }


async def _replay(args: argparse.Namespace) -> int:
    from muse_tmr.data.replay import ReplayConfig, ReplaySession

    session = ReplaySession(
        ReplayConfig(
            input_path=args.input,
            speed=args.speed,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
        )
    )
    metadata = await session.connect()
    frame_count = 0
    modality_counts = {}
    try:
        async for frame in session.stream():
            frame_count += 1
            for modality in frame.modalities():
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
    finally:
        await session.stop()

    print(
        f"replay complete source={metadata.source_name} "
        f"input={session.raw_path} frames={frame_count} modalities={modality_counts}"
    )
    return 0


def _meditation_plan(args: argparse.Namespace) -> int:
    from muse_tmr.reports.meditation_analysis import (
        build_meditation_plan,
        check_drift_cancelling,
        write_meditation_blocks,
    )

    try:
        check_drift_cancelling(args.blocks)
    except ValueError as exc:
        print(f"meditation-plan: {exc}", file=sys.stderr)
        return 2
    plan = build_meditation_plan(
        [condition for condition in args.conditions.split(",")],
        blocks=args.blocks,
        block_minutes=args.block_minutes,
        settle_seconds=args.settle_seconds,
        seed=args.seed,
    )
    if args.output is not None:
        output = _resolve_output_path(args.output)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = _default_path_base() / "data" / "protocol" / "meditation" / f"blocks_{timestamp}.json"
    write_meditation_blocks(plan, output)
    print(f"meditation plan written: {output}")
    print(f"order: {' '.join(plan.order)}")
    for block in plan.blocks:
        print(f"  block {block.index}: {block.condition:<12} {block.start_s / 60:6.1f} - {block.end_s / 60:6.1f} min")
    print("Fill in depth (stability), sensory_fading and drowsiness (0-10) after each block.")
    return 0


async def _analyze_meditation(args: argparse.Namespace) -> int:
    from muse_tmr.features.complexity_features import ComplexityConfig
    from muse_tmr.reports.meditation_analysis import (
        MeditationAnalysisConfig,
        analyze_meditation_recording,
        load_meditation_blocks,
    )

    recording_dir = args.recording_dir.expanduser().resolve()
    blocks = load_meditation_blocks(args.blocks.expanduser())
    config = MeditationAnalysisConfig(
        epoch_seconds=args.epoch_seconds,
        trim_block_start_seconds=args.trim_block_start,
        emg_indicator=args.emg_indicator,
        complexity=ComplexityConfig(lyapunov_enabled=not args.no_lyapunov),
    )
    output_dir = (
        _resolve_output_dir(args.output_dir)
        if args.output_dir is not None
        else _default_path_base() / "data" / "reports" / "meditation" / recording_dir.name
    )
    started = time.monotonic()
    analysis = await analyze_meditation_recording(recording_dir, blocks, config, use_polar=not args.no_polar)
    paths = analysis.write(output_dir)
    summary = analysis.summary
    primary = next(
        item for item in summary["contrasts"] if item["primary"] and item["variant"] == "clean"
    )
    emg = summary["emg"]
    print(f"meditation analysis written: {paths['summary']} ({time.monotonic() - started:.1f} s)")
    print(f"report: {paths['report']}")
    print(
        f"epochs in blocks={summary['counts']['epochs_in_blocks']} clean={summary['counts']['clean_epochs']}"
    )
    print(f"primary {summary['contrast']} lzc_all (clean epochs): {primary['difference']:+.4f}")
    print(
        f"EMG indicator {emg['indicator']}: confounded={emg['emg_confounded']} "
        f"(log10 diff clean={emg['condition_difference']['clean']['difference_log10']:+.3f})"
    )
    cardio = summary["cardio"]
    if cardio.get("available"):
        difference = cardio["breathing_difference_bpm"]
        if difference is not None and difference == difference:
            print(
                f"Polar H10 breathing {summary['contrast']}: {difference:+.1f} /min, "
                f"confounded={cardio['breathing_confounded']}"
            )
        else:
            print(f"Polar H10 breathing not compared: unreliable in block(s) {cardio['breathing_unreliable_blocks']}")
    elif cardio.get("error"):
        print(f"Polar H10 data not used: {cardio['error']}")
    return 0


def _aggregate_meditation(args: argparse.Namespace) -> int:
    from muse_tmr.reports.meditation_analysis import aggregate_meditation_summaries, json_safe

    summaries = [json.loads(path.expanduser().read_text(encoding="utf-8")) for path in args.summaries]
    try:
        result = aggregate_meditation_summaries(
            summaries,
            labels=[str(path) for path in args.summaries],
            min_sessions_for_inference=args.min_sessions,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"aggregate-meditation: {exc}", file=sys.stderr)
        return 2
    output = _resolve_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    for row in result["rows"]:
        if row["primary"] and row["kind"] == "raw":
            extra = ""
            if "p_sign_flip" in row:
                extra = f" p={row['p_sign_flip']:.4f} ci95=[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]"
            print(
                f"primary {result['contrast']} lzc_all ({row['variant']}): "
                f"mean {row['mean_difference']:+.4f} over {row['n_sessions']} sessions{extra}"
            )
    print(f"aggregate written: {output} ({len(result['rows'])} rows, all but the primary are exploratory)")
    return 0


def _add_calibration_voice_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--language", choices=("en", "ru"), default="en", help="Language of the spoken steps (default: en).")
    parser.add_argument("--voice", help="macOS say voice (default: Samantha for en, Milena for ru).")
    parser.add_argument("--rate", type=int, help="Speech rate in words per minute.")
    parser.add_argument("--silent", action="store_true", help="Log the cues without speaking them.")


def _calibration_speaker(args: argparse.Namespace):
    from muse_tmr.protocol.calibration import VOICES, SaySpeaker, SilentSpeaker, available_voices

    if args.silent:
        return SilentSpeaker()
    voices = available_voices()
    if not voices:
        raise SystemExit("macOS `say` is not available here; use --silent to run without a voice.")
    voice = args.voice or VOICES[args.language]
    if voice not in voices:
        note = ""
        if args.language == "ru":
            russian = ", ".join(sorted({name for name in voices if name in ("Milena", "Yuri", "Katya")})) or "none installed"
            note = f" (Russian voices here: {russian})"
        print(f"voice {voice} is not installed, using the system voice{note}")
        voice = None
    return SaySpeaker(voice=voice, rate=args.rate)


def _calibration_record_command(
    output_dir: Path,
    preset: str,
    seconds: float,
    *,
    with_polar: bool,
    address: Optional[str] = None,
    polar_address: Optional[str] = None,
) -> List[str]:
    import shutil

    command = [
        sys.executable, "-m", "muse_tmr.cli.main", "record",
        "--source", "amused",
        "--preset", preset,
        "--duration-seconds", f"{seconds:g}",
        "--duration-from-first-frame",
        "--no-data-timeout-seconds", "45",
        "--max-reconnect-attempts", "1000",
        "--output-dir", str(output_dir),
        "--quiet",
        "--allow-short",
    ]  # fmt: skip
    if with_polar:
        command.append("--with-polar")
    if address:
        command += ["--address", address]
    if with_polar and polar_address:
        command += ["--polar-address", polar_address]
    if shutil.which("caffeinate"):
        # Keep the Mac awake for the whole run, on battery too.
        command = ["caffeinate", "-i", "-s", *command]
    return command


def _start_recorder(command: List[str], output_dir: Path):
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "record.log").open("ab") as log_file:
        # Own process group: Ctrl-C reaches only this command, which then stops
        # the recorder (caffeinate + python) with one clean SIGINT.
        return subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )


def _stop_recorder(recorder, stop: bool, wait_seconds: float) -> None:
    import subprocess

    if stop and recorder.poll() is None:
        try:
            os.killpg(recorder.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
    # The Muse summary comes first, then the recorder stops the H10 streams.
    deadline = time.monotonic() + wait_seconds
    while recorder.poll() is None and time.monotonic() < deadline:
        try:
            recorder.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            print("still stopping the recorder (it stops the H10 streams first)...")


def _run_line_check(args: argparse.Namespace, output_dir: Path, speaker) -> Dict[str, object]:
    from muse_tmr.protocol.calibration import (
        CALIBRATION_DIRNAME,
        LINE_CHECK_FILENAME,
        LINE_CHECK_PRESETS,
        LINE_CHECK_SECONDS,
        run_line_check,
    )
    from muse_tmr.reports.line_check import VERDICT_TEXT, measure_recording, summarize
    from muse_tmr.reports.meditation_analysis import json_safe

    def record(directory: Path, preset: str) -> int:
        command = _calibration_record_command(
            directory, preset, LINE_CHECK_SECONDS, with_polar=False, address=args.address
        )
        recorder = _start_recorder(command, directory)
        try:
            recorder.wait()
        except KeyboardInterrupt:
            _stop_recorder(recorder, stop=True, wait_seconds=60.0)
            raise
        return int(recorder.returncode)

    directories = [
        output_dir.parent / f"{output_dir.name}_line{number}_{preset}"
        for number, preset in enumerate(LINE_CHECK_PRESETS, start=1)
    ]
    print(f"line check: {len(directories)} x {LINE_CHECK_SECONDS / 60:.0f} min, presets {', '.join(LINE_CHECK_PRESETS)}")
    results = run_line_check(directories, speaker, record, language=args.language, log=print)
    segments = []
    for result in results:
        channels: Mapping[str, object] = {}
        if result["returncode"] == 0:
            try:
                channels = asyncio.run(measure_recording(Path(str(result["recording"]))))
            except Exception as exc:  # a broken segment must not stop the calibration
                print(f"  segment {result['index']}: could not measure the line ({type(exc).__name__}: {exc})")
        segments.append({**result, "channels": channels})
    summary = summarize(segments)
    for row in summary["segments"]:
        levels = row.get("channels") or {}
        cells = ", ".join(
            f"{channel} {_format_float(level.get('amplitude_uv'), 1)} µV ({_format_float(level.get('height_db'), 0)} dB)"
            for channel, level in levels.items()
        )
        print(f"  {row['index']} {row['preset']}: {cells or 'no data (recorder exit code ' + str(row['returncode']) + ')'}")
    print(VERDICT_TEXT[summary["verdict"]])
    target = output_dir / CALIBRATION_DIRNAME / LINE_CHECK_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _format_float(value, digits: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if not math.isfinite(number) else f"{number:.{digits}f}"


def _calibration_run(args: argparse.Namespace) -> int:
    from muse_tmr.protocol.calibration import RECORD_SECONDS, calibration_protocol, run_guide

    speaker = _calibration_speaker(args)
    protocol = calibration_protocol(battery_pull=args.battery_pull, language=args.language)
    if args.output_dir is not None:
        output_dir = _resolve_output_dir(args.output_dir)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _default_path_base() / "data" / "recordings" / "session" / f"{stamp}_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"calibration run: {output_dir} (preset {args.preset})")
    print("Put on the Muse and the H10, sit, eyes closed. If the REMind app is connected to the Muse, press Disconnect there.")
    if args.line_check:
        _run_line_check(args, output_dir, speaker)
    print(f"{protocol[-1].end_s / 60:.0f} min from the first Muse frame:")
    for segment in protocol:
        print(f"  {segment.start_s / 60:4.1f}-{segment.end_s / 60:4.1f} min  {segment.label}")
    print("Connecting to the Muse and the H10.")
    command = _calibration_record_command(
        output_dir,
        args.preset,
        RECORD_SECONDS,
        with_polar=True,
        address=args.address,
        polar_address=args.polar_address,
    )
    recorder = _start_recorder(command, output_dir)
    result = None
    try:
        result = run_guide(
            output_dir,
            speaker,
            protocol=protocol,
            language=args.language,
            recorder_alive=lambda: recorder.poll() is None,
            log=print,
        )
    finally:
        stop = result is None or result.stop_reason != "completed"
        _stop_recorder(recorder, stop=stop, wait_seconds=RECORD_SECONDS + 120.0)
    print(f"guide: {result.stop_reason if result else 'interrupted'}, recorder exit code {recorder.returncode}")
    if result and result.blocks_files:
        print("blocks files: " + ", ".join(sorted(result.blocks_files)))
    print(f"next: muse-tmr calibration-report {output_dir}")
    ok = result is not None and result.stop_reason == "completed" and recorder.returncode == 0
    return 0 if ok else 1


def _calibration_guide(args: argparse.Namespace) -> int:
    from muse_tmr.protocol.calibration import calibration_protocol, run_guide

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    # The app may terminate the guide; finish the cue log and blocks files first.
    signal.signal(signal.SIGTERM, interrupt)
    pid = args.recorder_pid

    def recorder_alive() -> bool:
        if pid is None:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    recording_dir = args.recording_dir.expanduser().resolve()
    protocol = calibration_protocol(battery_pull=args.battery_pull, language=args.language)
    result = run_guide(
        recording_dir,
        _calibration_speaker(args),
        protocol=protocol,
        language=args.language,
        recorder_alive=recorder_alive,
        log=lambda message: print(message, flush=True),
    )
    print(f"guide finished: {result.stop_reason}, {result.cues_spoken} cues, pairs: {', '.join(sorted(result.blocks_files)) or 'none'}")
    return 0 if result.stop_reason in ("completed", "recording_ended") else 1


async def _calibration_report(args: argparse.Namespace) -> int:
    from muse_tmr.reports.calibration_report import build_calibration_report

    recording_dir = args.recording_dir.expanduser().resolve()
    output_dir = (
        _resolve_output_dir(args.output_dir)
        if args.output_dir is not None
        else _default_path_base() / "data" / "reports" / "calibration" / recording_dir.name
    )
    started = time.monotonic()
    report = await build_calibration_report(recording_dir)
    paths = report.write(output_dir)
    print(f"calibration report written: {paths['report']} ({time.monotonic() - started:.1f} s)")
    for name, pair in report.summary["pairs"].items():
        if "error" in pair:
            print(f"  {name}: {pair['error']}")
            continue
        item = pair["variants"]["all"]
        print(
            f"  {name:9s} 55-95 dB AF {item['emg_55_95_db']['frontal']:+.1f} TP {item['emg_55_95_db']['temporal']:+.1f}"
            f" | 30-45 dB AF {item['emg_30_45_db']['frontal']:+.1f} TP {item['emg_30_45_db']['temporal']:+.1f}"
            f" | dLZC {item['lzc_all']:+.3f} d1/f(2-40) {item['aperiodic_exponent_2_40_all']:+.2f}"
        )
    for item in report.summary["breathing"]:
        estimates = item["estimates_bpm"]
        print(
            f"  {item['segment']}: paced {item['known_rate_bpm']:.0f}/min, ACC spectral {estimates['acc_spectral']:.1f},"
            f" breath-by-breath {estimates['acc_breath']:.1f}, ECG-derived {estimates['edr']:.1f}"
        )
    return 0


def _cancel_on_stop_signals() -> None:
    """Make SIGINT and SIGTERM cancel the running task, however we were launched.

    A process started in the background by a non-interactive shell (``cmd &``)
    inherits SIGINT as ignored, and Python then installs no KeyboardInterrupt
    handler, so neither Ctrl-C, the app's Stop nor the --with-polar parent
    could stop a recorder cleanly. The recorders treat the cancel as a normal
    user stop and close their streams.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is None:
        return
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(stop_signal, task.cancel)
        except (NotImplementedError, RuntimeError, ValueError):
            pass  # no signal support here (Windows, not the main thread)


async def _record(args: argparse.Namespace) -> int:
    from muse_tmr.data.recorder import OvernightRecorder, RecordingConfig

    _cancel_on_stop_signals()

    duration_seconds = (
        args.duration_seconds
        if args.duration_seconds is not None
        else args.duration_hours * 3600
    )
    output_dir = _resolve_output_dir(args.output_dir) if args.output_dir else _default_recording_dir()
    source = _build_source(args, duration_seconds=0)
    companions = []
    if getattr(args, "with_polar", False):
        # The Muse run may wait up to the first-frame grace before its clock starts.
        polar_seconds = duration_seconds + (600.0 if args.duration_from_first_frame else 0.0)
        companions.append(_polar_companion(output_dir, polar_seconds, args.polar_address))
    recorder = OvernightRecorder(
        RecordingConfig(
            output_dir=output_dir,
            duration_seconds=duration_seconds,
            source_name=args.source,
            allow_short=args.allow_short,
            duration_from_first_frame=args.duration_from_first_frame,
            no_data_timeout_seconds=args.no_data_timeout_seconds,
            max_reconnect_attempts=args.max_reconnect_attempts,
        ),
        companions=companions,
    )
    summary = await recorder.record(source)
    print(f"recording complete summary={summary.summary_path}")
    return 0


def _polar_companion(output_dir: Path, duration_seconds: float, address: Optional[str]):
    from muse_tmr.data.recorder import CompanionProcess

    # A little longer than the Muse run: the parent decides the end and stops it.
    command = [
        sys.executable,
        "-m",
        "muse_tmr.cli.main",
        "record-polar",
        "--output-dir",
        str(output_dir),
        "--duration-seconds",
        str(duration_seconds + 120.0),
    ]
    if address:
        command.extend(["--address", address])
    return CompanionProcess("polar", command, Path(output_dir) / "polar" / "record-polar.log")


async def _record_polar(args: argparse.Namespace) -> int:
    from muse_tmr.data.polar_recorder import PolarRecorder, PolarRecordingConfig
    from muse_tmr.sources.polar_h10 import PolarH10Client, PolarH10Settings

    _cancel_on_stop_signals()
    if args.no_ecg and args.no_acc:
        print("note: --no-ecg and --no-acc leave only HR/RR")
    duration_seconds = args.duration_seconds if args.duration_seconds is not None else args.duration_hours * 3600
    output_dir = _resolve_output_dir(args.output_dir)
    client = PolarH10Client(
        PolarH10Settings(
            ecg=not args.no_ecg,
            acc=not args.no_acc,
            acc_rate_hz=args.acc_rate,
            acc_range_g=args.acc_range,
        ),
        address=args.address,
        name_filter=args.name_filter,
    )
    recorder = PolarRecorder(
        PolarRecordingConfig(
            output_dir=output_dir,
            duration_seconds=duration_seconds,
            max_reconnect_attempts=args.max_reconnect_attempts,
        )
    )
    summary = await recorder.record(client)
    decode = summary.get("decode") or {}
    print(
        f"polar recording complete stop_reason={summary['stop_reason']} "
        f"reconnects={summary['reconnects']} counts={json.dumps(decode.get('counts', {}), sort_keys=True)} "
        f"dir={output_dir / 'polar'}"
    )
    return 0


def _decode_polar(args: argparse.Namespace) -> int:
    from muse_tmr.data.polar_recorder import decode_polar_session

    session_dir = _resolve_output_dir(args.session_dir)
    result = decode_polar_session(session_dir)
    print(f"decoded {session_dir / 'polar'}: {json.dumps(result['counts'], sort_keys=True)}")
    if result["errors"]:
        print(f"decode errors: {json.dumps(result['errors'], sort_keys=True)}")
    return 0


async def _analyze_optics_agreement(args: argparse.Namespace) -> int:
    from muse_tmr.reports.optics_agreement import (
        OpticsAgreementConfig,
        OpticsAgreementInputError,
        analyze_optics_agreement,
    )
    from muse_tmr.testing import SYNTHETIC_NOTE

    recording_dir = _resolve_output_dir(args.recording_dir).resolve()
    output_dir = (
        _resolve_output_dir(args.output_dir)
        if args.output_dir is not None
        else _default_path_base() / "data" / "reports" / "optics_agreement" / recording_dir.name
    )
    try:
        config = OpticsAgreementConfig(epoch_seconds=args.epoch_seconds)
        config.validate()
        started = time.monotonic()
        report = await analyze_optics_agreement(recording_dir, config)
    except (OpticsAgreementInputError, FileNotFoundError, ValueError) as exc:
        print(f"analyze-optics-agreement: {exc}", file=sys.stderr)
        return 2
    paths = report.write(output_dir)
    note = f" ({SYNTHETIC_NOTE})" if report.synthetic else ""
    print(f"optics agreement written: {paths['json']} ({time.monotonic() - started:.1f} s){note}")
    print(f"report: {paths['markdown']}")
    counts = report.counts
    print(
        f"epochs={counts['epochs']} compared={counts['compared_epochs']} "
        f"no_contact_seconds={counts['no_contact_seconds']:.0f}"
    )
    for estimator in ("decoder_heart_rate", "ppg_estimated_hr_bpm", "hr_series"):
        row = report.overall(estimator)
        if row is None or not row.n:
            print(f"  {estimator}: no estimate")
            continue
        print(
            f"  {estimator}: n={row.n} coverage={row.coverage:.2f} bias={row.bias_bpm:+.1f} bpm "
            f"limits={_format_float(row.loa_low_bpm, 1)}..{_format_float(row.loa_high_bpm, 1)} "
            f"mae={row.mae_bpm:.1f}"
        )
    jumps = report.jumps[0]
    print(
        f"  sudden HR rule: fired={jumps.fired} false_jumps={jumps.false_jumps} "
        f"ecg_jumps={jumps.ecg_jumps} missed={jumps.missed}"
    )
    if report.limitations:
        print(f"limitations: {', '.join(report.limitation_codes)}")
    return 0


async def _annotate_template(args: argparse.Namespace) -> int:
    from muse_tmr.annotations import build_rem_annotation_rows, export_rem_annotations
    from muse_tmr.data.replay import ReplayConfig, ReplaySession
    from muse_tmr.features import new_rem_event_tracker
    from muse_tmr.features.epochs import EpochBuilder, EpochConfig
    from muse_tmr.models import HeuristicRemDetector

    session = ReplaySession(
        ReplayConfig(
            input_path=args.input,
            speed=0.0,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
        )
    )
    await session.connect()
    try:
        builder = EpochBuilder(
            EpochConfig(
                epoch_seconds=args.epoch_seconds,
                stride_seconds=args.stride_seconds,
            )
        )
        epochs = [epoch async for epoch in builder.build(session.stream())]
    finally:
        await session.stop()

    rows = build_rem_annotation_rows(
        epochs,
        detector=HeuristicRemDetector(),
        recording_id=str(session.recording_dir),
        label=args.label,
        rem_tracker=new_rem_event_tracker(),
    )
    output_path = export_rem_annotations(rows, _resolve_output_path(args.output))
    print(f"annotation template complete rows={len(rows)} output={output_path}")
    return 0


def _train_rem_classifier(args: argparse.Namespace) -> int:
    from muse_tmr.annotations import load_rem_annotations
    from muse_tmr.models import (
        DEFAULT_PERSONAL_REM_FEATURES,
        PersonalRemClassifierConfig,
        train_personal_rem_classifier,
    )

    annotations = load_rem_annotations(_resolve_output_path(args.annotations))
    feature_names = tuple(args.features) if args.features else DEFAULT_PERSONAL_REM_FEATURES
    config = PersonalRemClassifierConfig(
        feature_names=feature_names,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2_penalty=args.l2_penalty,
        decision_threshold=args.threshold,
        min_training_rows=args.min_training_rows,
    )
    model = train_personal_rem_classifier(annotations, config=config)
    output_path = model.save(_resolve_output_path(args.output))
    summary = model.training_summary
    metrics = summary.metrics if summary is not None else {}
    group_metrics = summary.group_holdout_metrics if summary is not None else {}
    print(
        "personal REM classifier trained "
        f"rows={summary.training_rows if summary else 0} "
        f"positive={summary.positive_rows if summary else 0} "
        f"negative={summary.negative_rows if summary else 0} "
        f"skipped_unknown={summary.skipped_unknown_rows if summary else 0} "
        f"accuracy={metrics.get('accuracy', 'nan')} "
        f"balanced_accuracy={metrics.get('balanced_accuracy', 'nan')} "
        f"brier={metrics.get('brier_score', 'nan')} "
        f"group_holdout={group_metrics.get('status', 'nan')} "
        f"output={output_path}"
    )
    return 0


def _play_test_cue(args: argparse.Namespace) -> int:
    from muse_tmr.audio import (
        AudioCuePlayer,
        AudioPlaybackConfig,
        TestCue,
        audio_config_with_calibration,
        load_volume_calibrations,
    )

    config = AudioPlaybackConfig(
        max_volume=args.max_volume,
        default_volume=args.volume,
        fade_in_seconds=args.fade_in_seconds,
        fade_out_seconds=args.fade_out_seconds,
        device_name=args.device_name,
        log_path=_resolve_output_path(args.log_path) if args.log_path else None,
    )
    if args.calibration:
        try:
            store = load_volume_calibrations(_resolve_output_path(args.calibration))
            calibration = (
                store.latest_for_device(args.device_name)
                if args.device_name
                else store.latest()
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"volume calibration error: {exc}", file=sys.stderr)
            return 1
        config = audio_config_with_calibration(config, calibration)

    player = AudioCuePlayer(config, backend=args.audio_resolution.backend)
    if args.emergency_stop:
        result = player.emergency_stop()
    else:
        result = player.play_test_cue(
            TestCue(
                cue_id="test-cue",
                frequency_hz=args.frequency_hz,
                duration_seconds=args.duration_seconds,
            ),
            volume=args.volume,
        )
    print(
        "test cue "
        f"status={result.status} "
        f"backend={result.backend_name} requested_backend={args.backend} "
        f"volume={result.effective_volume} "
        f"requested_volume={result.requested_volume} "
        f"volume_capped={result.volume_capped} "
        f"reasons={','.join(result.reason_codes)}"
    )
    return 0 if result.status in {"played", "stopped", "skipped"} else 1


def _calibrate_volume(args: argparse.Namespace) -> int:
    from muse_tmr.audio import VolumeCalibration, save_volume_calibration

    calibration = VolumeCalibration(
        device_name=args.device_name,
        detectable_volume=args.detectable_volume,
        identifiable_volume=args.identifiable_volume,
        comfortable_volume=args.comfortable_volume,
        cue_id=args.cue_id,
        backend_name=args.backend,
        notes=args.notes,
    )
    output_path = save_volume_calibration(
        calibration,
        _resolve_output_path(args.output),
        append=not args.replace,
    )
    print(
        "volume calibration saved "
        f"device={calibration.device_name} "
        f"detectable={calibration.detectable_volume} "
        f"identifiable={calibration.identifiable_volume} "
        f"comfortable={calibration.comfortable_volume} "
        f"scheduler_max={calibration.scheduler_max_volume} "
        f"output={output_path}"
    )
    return 0


def _create_cue_library(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from muse_tmr.audio import default_cue_library, export_cue_library

    library = default_cue_library()
    tlr_note = ""
    if args.protocol_preset is not None:
        from muse_tmr.protocol import default_tlr_cue

        # The cue create-tlr-cue --protocol-preset makes, in place of the starter's (#264).
        preset, overrides = _protocol_from_args(args, {})
        tlr = default_tlr_cue(preset.tlr_cue_config(overrides=overrides), protocol=preset.manifest(overrides))
        library = replace(library, cues=tuple(tlr if cue.cue_id == tlr.cue_id else cue for cue in library.cues))
        tlr_note = (
            f"tlr_cue={tlr.cue_id} tlr_frequency_hz={tlr.frequency_hz} "
            f"tlr_duration_seconds={tlr.duration_seconds} protocol={preset.name} "
        )
    output_path = export_cue_library(library, _resolve_output_path(args.output))
    print(f"cue library created {tlr_note}output={output_path}")
    return 0


def _validate_cue_library(args: argparse.Namespace) -> int:
    import json

    from muse_tmr.audio import validate_cue_library_file

    input_path = _resolve_output_path(args.input)
    report = validate_cue_library_file(
        input_path,
        check_files=not args.skip_file_check,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.is_valid else 1


def _validate_pilot1_recording(args: argparse.Namespace) -> int:
    import json

    from muse_tmr.validation import (
        DEFAULT_PILOT1_REQUIRED_MODALITIES,
        validate_pilot1_recording,
    )

    required_modalities = (
        tuple(args.required_modalities)
        if args.required_modalities
        else DEFAULT_PILOT1_REQUIRED_MODALITIES
    )
    report = validate_pilot1_recording(
        _resolve_output_path(args.input),
        min_duration_seconds=args.min_duration_hours * 3600,
        required_modalities=required_modalities,
        max_downtime_fraction=args.max_downtime_fraction,
    )
    # A pass on a synthetic night must not read as a recorded night's.
    if _says_synthetic(Path(report.summary_path).parent / "metadata.json"):
        payload = _synthetic_payload(report.to_dict())
        if args.output is not None:
            _write_synthetic_json(_resolve_output_path(args.output), payload)
    else:
        payload = report.to_dict()
        if args.output is not None:
            report.save(_resolve_output_path(args.output))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.passed else 1


def _validate_pilot2_calibration(args: argparse.Namespace) -> int:
    import json

    from muse_tmr.validation import validate_pilot2_calibration

    report = validate_pilot2_calibration(
        _resolve_output_path(args.calibration),
        device_name=args.device_name,
        playback_log_path=(
            _resolve_output_path(args.playback_log)
            if args.playback_log is not None
            else None
        ),
        hard_max_volume=args.hard_max_volume,
    )
    if args.output is not None:
        report.save(_resolve_output_path(args.output))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def _resolve_gate_thresholds(args: argparse.Namespace) -> Tuple[float, float]:
    """Resolve ``(enter, exit)`` gate thresholds, preset-aware, honoring overrides.

    Both flags omitted -> the preset's default pair (muse_tmr.presets). A flag that
    is given is used verbatim; the other falls back to the preset default. If a
    partial override would break the invariant, exit is clamped to enter so the
    gate config always validates.
    """
    from muse_tmr.presets import preset_gate_thresholds

    preset_enter, preset_exit = preset_gate_thresholds(getattr(args, "preset", None))
    enter = preset_enter if args.enter_threshold is None else args.enter_threshold
    exit_ = preset_exit if args.exit_threshold is None else args.exit_threshold
    if exit_ > enter:
        print(
            f"warning: exit threshold {exit_} > enter {enter} after preset resolution; "
            f"clamping exit to {enter}. Pass both --enter-threshold and --exit-threshold "
            "to set them explicitly.",
            file=sys.stderr,
        )
        exit_ = enter
    return enter, exit_


async def _simulate_replay_cues(args: argparse.Namespace) -> int:
    try:
        report = await _replay_cue_report(args)
    except _NightArmError as exc:
        print(f"simulate-replay-cues: {exc}", file=sys.stderr)
        return 2
    # A cue plan from a synthetic night must not read as a recorded night's.
    note = ""
    if _says_synthetic(Path(report.recording_input) / "metadata.json"):
        from muse_tmr.testing import SYNTHETIC_NOTE

        note = f" ({SYNTHETIC_NOTE})"
        output_path = _write_synthetic_json(_resolve_output_path(args.output), report.to_dict())
    else:
        output_path = report.save(_resolve_output_path(args.output))
    scheduler_events_output = None
    if args.scheduler_events_output is not None:
        scheduler_events_output = report.save_scheduler_events(
            _resolve_output_path(args.scheduler_events_output)
        )

    if report.arm is not None:
        # Never the arm, nor what would give it away: the plan and its first play.
        print(
            "pilot3 replay simulation complete "
            f"passed={report.passed} "
            f"epochs={report.metrics.get('epoch_count', 0)} "
            f"stride_seconds={report.metrics.get('stride_seconds')} "
            f"gate_open={report.metrics.get('gate_open_count', 0)} "
            f"first_gate_open_seconds={report.metrics.get('first_gate_open_seconds')} "
            f"audio_playback_executed={report.audio_playback_executed} "
            f"protocol={report.protocol.get('name')} "
            + (f"night_id={report.night_id} " if report.night_id else "")
            + f"output={output_path}{note} ({BLINDED_NOTE})"
        )
    else:
        print(
            "pilot3 replay cue simulation complete "
            f"passed={report.passed} "
            f"epochs={report.metrics.get('epoch_count', 0)} "
            f"stride_seconds={report.metrics.get('stride_seconds')} "
            f"gate_open={report.metrics.get('gate_open_count', 0)} "
            f"first_gate_open_seconds={report.metrics.get('first_gate_open_seconds')} "
            f"cue_plan={report.metrics.get('cue_plan_count', 0)} "
            f"first_play_seconds={report.metrics.get('first_play_seconds')} "
            f"injected_cues={report.metrics.get('injected_cue_count', 0)} "
            f"audio_playback_executed={report.audio_playback_executed} "
            f"protocol={report.protocol.get('name')} "
            f"output={output_path}{note}"
        )
    if scheduler_events_output is not None:
        print(f"scheduler_events={scheduler_events_output}")
    return 0 if report.passed else 1


async def _replay_cue_report(args: argparse.Namespace):
    """The Pilot 3 report for the inputs and flags of ``simulate-replay-cues``."""
    from muse_tmr.audio import load_cue_library
    from muse_tmr.features import EpochConfig
    from muse_tmr.models import RemGateConfig
    from muse_tmr.presets import preset_provides_optics
    from muse_tmr.protocol import (
        load_night_puzzle_session,
        load_puzzle_catalog,
        load_puzzle_cue_assignment,
    )
    from muse_tmr.protocol.arousal_guard import ArousalGuardConfig
    from muse_tmr.validation import simulate_replay_cue_plan

    night_arm = _night_arm_from_args(args)
    catalog = load_puzzle_catalog(_resolve_output_path(args.catalog))
    session = load_night_puzzle_session(_resolve_output_path(args.session))
    assignment = load_puzzle_cue_assignment(_resolve_output_path(args.assignment))
    cue_library = load_cue_library(_resolve_output_path(args.cue_library))
    _check_sham_cue(night_arm, catalog=catalog, cue_library=cue_library)
    preset, overrides = _protocol_from_args(args, _NIGHT_PRESET_FLAGS)
    settings = preset.runner_settings(overrides)
    enter_threshold, exit_threshold = _resolve_gate_thresholds(args)
    optics_capable = preset_provides_optics(getattr(args, "preset", None))
    return await simulate_replay_cue_plan(
        _resolve_output_path(args.input),
        catalog=catalog,
        session=session,
        assignment=assignment,
        cue_library=cue_library,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        epoch_config=EpochConfig(
            epoch_seconds=args.epoch_seconds,
            stride_seconds=args.stride_seconds,
        ),
        gate_config=RemGateConfig(
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold,
            min_stable_seconds=settings.gate_min_stable_seconds,
            epoch_seconds=args.epoch_seconds,
            cooldown_seconds=args.gate_cooldown_seconds,
            optics_capable=optics_capable,
        ),
        scheduler_config=preset.scheduler_config(enable_tlr_block=False, overrides=overrides),
        arousal_guard_config=ArousalGuardConfig(
            enabled=not args.disable_arousal_guard,
            optics_capable=optics_capable,
            **_guard_hr_rule_config(args),
        ),
        inject_cues_every_seconds=getattr(args, "inject_cues_every_seconds", None),
        protocol=_protocol_block(preset, overrides, assignment=assignment.metadata),
        night_arm=night_arm,
    )


# The flags each command takes from a protocol preset, by RunnerSettings field. Left out
# (None) a flag takes the preset's value; given, it wins and is recorded as an override.
_NIGHT_PRESET_FLAGS = {
    "min_stable_seconds": "gate_min_stable_seconds",
    "puzzle_cue_interval_seconds": "puzzle_cue_interval_seconds",
    "scheduler_cooldown_seconds": "scheduler_cooldown_seconds",
    "max_puzzle_cues_per_block": "max_puzzle_cues_per_block",
}
_TLR_BLOCK_PRESET_FLAGS = {
    "repetitions": "tlr_repetitions",
    "interval_seconds": "tlr_interval_seconds",
    "post_block_pause_seconds": "tlr_post_block_pause_seconds",
}
_TLR_CUE_PRESET_FLAGS = {
    "frequency_hz": "tlr_tone_frequency_hz",
    "duration_seconds": "tlr_tone_duration_seconds",
}


def _protocol_from_args(
    args: argparse.Namespace,
    flags: Mapping[str, str],
    *,
    extra_overrides: Optional[Mapping[str, object]] = None,
):
    """The command's protocol preset and the overrides its explicit flags make."""
    from muse_tmr.protocol import resolve_protocol_preset

    preset = resolve_protocol_preset(getattr(args, "protocol_preset", None))
    overrides: Dict[str, object] = {
        setting: getattr(args, dest)
        for dest, setting in flags.items()
        if getattr(args, dest, None) is not None
    }
    overrides.update(extra_overrides or {})
    return preset, overrides


def _protocol_block(preset, overrides: Mapping[str, object], **inputs: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """The preset block a run records, with the blocks its protocol files were made with.

    ``inputs`` maps a file (assignment, tlr_block_plan, ...) to its metadata. A file made
    with another preset is named on stderr; the run goes on and records it under
    ``inputs``.
    """
    block = preset.manifest(overrides)
    recorded: Dict[str, object] = {}
    for name, metadata in inputs.items():
        protocol = metadata.get("protocol") if isinstance(metadata, Mapping) else None
        if not isinstance(protocol, Mapping):
            continue
        recorded[name] = {key: protocol.get(key) for key in ("name", "version", "content_hash", "effective_hash")}
        if protocol.get("name") != preset.name:
            print(
                f"warning: the {name.replace('_', ' ')} was made with protocol preset {protocol.get('name')}, "
                f"this run uses {preset.name}",
                file=sys.stderr,
            )
    if recorded:
        block["inputs"] = recorded
    return block


def _synthesize_night(args: argparse.Namespace) -> int:
    from muse_tmr.testing import SCENARIOS, SYNTHETIC_NOTE, generate_night, scenario_config

    if args.list:
        print(f"synthetic scenarios ({SYNTHETIC_NOTE}):")
        for scenario in SCENARIOS.values():
            print(f"  {scenario.name:<19} {scenario.description}")
        return 0
    if args.scenario is None or args.output is None:
        print("synthesize-night: give --scenario and --output, or --list", file=sys.stderr)
        return 2
    try:
        config = scenario_config(args.scenario, seed=args.seed, hours=args.hours)
        config.validate()
    except ValueError as exc:
        print(f"synthesize-night: {exc}", file=sys.stderr)
        return 2
    output_dir = _resolve_output_dir(args.output)
    refusal = _synthetic_output_refusal(output_dir, "metadata.json")
    if refusal is not None:
        print(f"synthesize-night: {refusal}", file=sys.stderr)
        return 2
    try:
        night = generate_night(config, output_dir)
    except ValueError as exc:
        print(f"synthesize-night: {exc}", file=sys.stderr)
        return 2
    print(f"synthetic night written {_synthetic_night_fields(night)}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    """Generate a synthetic night and run Pilot 1, Pilot 3, the report and the agreement on it."""
    import webbrowser

    from muse_tmr.reports.nightly_report import build_report
    from muse_tmr.testing import SYNTHETIC_NOTE, generate_night, scenario_config
    from muse_tmr.testing.agreement import (
        gate_epochs_from_pilot3,
        load_hypnogram,
        stage_at,
        synthetic_agreement,
    )
    from muse_tmr.validation import validate_pilot1_recording

    started = time.perf_counter()
    try:
        config = scenario_config(args.scenario, seed=args.seed, hours=args.hours)
        config.validate()
        baseline = _load_demo_baseline(_resolve_output_path(args.baseline)) if args.baseline else None
    except (OSError, ValueError) as exc:
        print(f"demo: {exc}", file=sys.stderr)
        return 2
    if baseline is not None:
        mismatch = _demo_baseline_identity(baseline, config)
        if mismatch is not None:
            print(f"demo: {mismatch}", file=sys.stderr)
            return 2
    output_dir = _resolve_output_dir(args.output) if args.output is not None else _default_demo_dir()
    recording_dir = output_dir / f"{config.scenario}_seed{config.seed}"
    # Neither the night nor the reports next to it overwrite or join anything but an earlier demo.
    refusal = _synthetic_output_refusal(output_dir, "demo_summary.json") or _synthetic_output_refusal(
        recording_dir, "metadata.json"
    )
    if refusal is not None:
        print(f"demo: {refusal}", file=sys.stderr)
        return 2

    print(f"demo: {config.scenario}, seed {config.seed}, {config.hours:g} h on {config.preset}; {SYNTHETIC_NOTE}")
    try:
        night = generate_night(config, recording_dir)
    except ValueError as exc:
        print(f"demo: {exc}", file=sys.stderr)
        return 2
    print(f"generated {_synthetic_night_fields(night)}")

    required = ("eeg", "imu", "ppg") if config.has_optics else ("eeg", "imu")
    pilot1 = validate_pilot1_recording(
        recording_dir,
        min_duration_seconds=night.night.duration_seconds,
        required_modalities=required,
    )
    pilot1_path = _write_synthetic_json(output_dir / "pilot1.json", pilot1.to_dict())
    print(f"pilot1 passed={pilot1.passed} failed={','.join(pilot1.failed_criteria) or 'none'}")

    protocol = _write_demo_protocol(output_dir / "protocol", seed=config.seed)
    pilot3_path = output_dir / "pilot3_report.json"
    pilot3_args = build_parser().parse_args([
        "simulate-replay-cues", str(recording_dir),
        "--catalog", str(protocol["catalog"]),
        "--session", str(protocol["session"]),
        "--assignment", str(protocol["assignment"]),
        "--cue-library", str(protocol["cue_library"]),
        "--output", str(pilot3_path),
        "--preset", config.preset,
    ])
    pilot3_started = time.perf_counter()
    report = asyncio.run(_replay_cue_report(pilot3_args))
    pilot3_seconds = time.perf_counter() - pilot3_started
    if report.audio_backend != "mock" or report.audio_playback_executed:
        print(
            f"demo: Pilot 3 ran with audio backend {report.audio_backend!r}; the demo runs on mock audio only",
            file=sys.stderr,
        )
        return 1
    _write_synthetic_json(pilot3_path, report.to_dict())
    metrics = report.metrics
    print(
        f"pilot3 passed={report.passed} epochs={metrics['epoch_count']} gate_open={metrics['gate_open_count']} "
        f"cue_plan={metrics['cue_plan_count']} uncued_puzzle_play_count={metrics['uncued_puzzle_play_count']} "
        f"first_gate_open_seconds={metrics['first_gate_open_seconds']} "
        f"first_play_seconds={metrics['first_play_seconds']} audio_backend={report.audio_backend}"
    )

    report_started = time.perf_counter()
    report_path = output_dir / "report.html"
    build_report(recording_dir, report_path)
    report_seconds = time.perf_counter() - report_started

    hypnogram = load_hypnogram(recording_dir)
    agreement = synthetic_agreement(hypnogram, gate_epochs_from_pilot3(report))
    agreement_path = _write_synthetic_json(output_dir / "agreement.json", agreement)
    # Each play with the generated stage of the epoch that planned it, the same epoch
    # labelling as the agreement.
    plays = [
        {
            "puzzle_id": event.puzzle_id,
            "cue_id": event.cue_id,
            "decided_at_seconds": result.timestamp_seconds + result.duration_seconds,
            "generated_stage": stage_at(hypnogram, result.timestamp_seconds + result.duration_seconds / 2),
        }
        for result in report.epoch_results
        for event in result.scheduler_events
        if event.event_type == "play"
    ]
    in_rem = sum(1 for play in plays if play["generated_stage"] == "R")
    described = "; ".join(
        f"{play['puzzle_id']} decided at {play['decided_at_seconds']:g} s, generated stage {play['generated_stage']}"
        for play in plays
    )
    print(f"cued plays: {described} ({in_rem} of {len(plays)} in generated REM)" if plays else "cued plays: none")
    latencies = [period["first_gate_open_latency_s"] for period in agreement["rem_periods"]]
    print(
        f"agreement with the generated hypnogram ({SYNTHETIC_NOTE}): "
        f"kappa={_format_float(agreement['kappa'], 3)} recall={_format_float(agreement['recall'], 3)} "
        f"precision={_format_float(agreement['precision'], 3)} "
        f"gate_open_fraction={_format_float(agreement['gate_open_fraction'], 3)} "
        f"first_gate_open_latency_s={latencies}"
    )

    summary: Dict[str, object] = {
        "scenario": config.scenario,
        "seed": config.seed,
        "hours": config.hours,
        "preset": config.preset,
        "raw_sha256": night.sha256,
        "raw_bytes": night.raw_path.stat().st_size,
        "duration_seconds": night.generation["duration_seconds"],
        "stop_reason": night.generation["stop_reason"],
        "pilot1": {"passed": pilot1.passed, "failed_criteria": list(pilot1.failed_criteria)},
        "pilot3": {
            "passed": report.passed,
            "failed_criteria": list(report.failed_criteria),
            "audio_backend": report.audio_backend,
            "audio_playback_executed": report.audio_playback_executed,
            "epoch_count": metrics["epoch_count"],
            "gate_open_count": metrics["gate_open_count"],
            "cue_plan_count": metrics["cue_plan_count"],
            "uncued_puzzle_play_count": metrics["uncued_puzzle_play_count"],
            "first_gate_open_seconds": metrics["first_gate_open_seconds"],
            "first_play_seconds": metrics["first_play_seconds"],
        },
        "cued_plays": plays,
        "cued_plays_in_generated_rem": in_rem,
        "agreement": {
            "kappa": agreement["kappa"],
            "recall": agreement["recall"],
            "precision": agreement["precision"],
            "gate_open_fraction": agreement["gate_open_fraction"],
            "first_gate_open_latency_s": latencies,
        },
        "files": {
            "recording": str(recording_dir),
            "protocol": str(output_dir / "protocol"),
            "pilot1": str(pilot1_path),
            "pilot3": str(pilot3_path),
            "report": str(report_path),
            "agreement": str(agreement_path),
        },
    }
    checks = {
        "pilot1_passed": pilot1.passed,
        "pilot3_passed": report.passed,
        "no_uncued_plays": metrics["uncued_puzzle_play_count"] == 0,
        "audio_backend_mock": report.audio_backend == "mock" and not report.audio_playback_executed,
        "report_written": report_path.is_file() and report_path.stat().st_size > 0,
    }
    if baseline is not None:
        mismatches = _demo_baseline_mismatches(baseline, _json_safe(summary))
        summary["baseline"] = {"path": str(args.baseline), "mismatches": mismatches}
        checks["agreement_within_baseline"] = not mismatches
        print(
            f"baseline {args.baseline}: "
            + ("matches" if not mismatches else f"{len(mismatches)} differences: " + "; ".join(mismatches))
        )
    summary["checks"] = checks
    summary["seconds"] = {
        "generation": round(night.generation_seconds, 2),
        "pilot3": round(pilot3_seconds, 2),
        "report": round(report_seconds, 2),
        "total": round(time.perf_counter() - started, 2),
    }
    summary_path = _write_synthetic_json(output_dir / "demo_summary.json", summary)
    print(f"report={report_path}")
    print(f"summary={summary_path}")
    failed = [name for name, passed in checks.items() if not passed]
    print(f"demo {'passed' if not failed else 'failed: ' + ', '.join(failed)} in {summary['seconds']['total']:.0f} s")
    if not args.no_open:
        webbrowser.open(report_path.resolve().as_uri())
    return 0 if not failed else 1


_DEMO_TONES_HZ = (440.0, 550.0, 660.0, 770.0)


def _write_demo_protocol(protocol_dir: Path, *, seed: int) -> Dict[str, Path]:
    """Pilot 3 inputs for the demo: four synthetic puzzles with a generated tone each,
    half of them cued, written where the loaders of simulate-replay-cues read them."""
    from dataclasses import replace

    from muse_tmr.audio import CueLibrary, CueMetadata
    from muse_tmr.protocol import PuzzleCatalog, PuzzleTask, assign_cued_uncued_puzzles
    from muse_tmr.testing import SYNTHETIC_NOTE

    note = {"synthetic": True, "note": SYNTHETIC_NOTE}
    numbers = range(1, len(_DEMO_TONES_HZ) + 1)
    catalog = PuzzleCatalog(
        puzzles=tuple(
            PuzzleTask(
                puzzle_id=f"synthetic-{number}",
                prompt=f"Synthetic puzzle {number}",
                solution=f"answer {number}",
                cue_id=f"synthetic-tone-{number}",
                source="synthetic",
                tags=("synthetic",),
                metadata=note,
            )
            for number in numbers
        )
    )
    session = catalog.generate_night_session(
        session_id=f"synthetic-demo-seed{seed}",
        puzzle_count=len(_DEMO_TONES_HZ),
        selection_seed=seed,
    )
    session = replace(session, metadata={**session.metadata, **note})
    assignment = assign_cued_uncued_puzzles(session, seed=seed)
    assignment = replace(assignment, metadata={**assignment.metadata, **note})
    # Uncued puzzles keep their tones in the library, so only the assignment keeps
    # them from being scheduled.
    cue_library = CueLibrary(
        library_id="synthetic-demo",
        cues=tuple(
            CueMetadata(
                cue_id=f"synthetic-tone-{number}",
                cue_type="generated_tone",
                duration_seconds=1.0,
                protocol="puzzle",
                frequency_hz=frequency,
                volume_hint=0.05,
                tags=("puzzle", "generated", "synthetic"),
                description=f"Generated tone for the demo night, {SYNTHETIC_NOTE}.",
                metadata=note,
            )
            for number, frequency in zip(numbers, _DEMO_TONES_HZ)
        ),
    )
    return {
        "catalog": catalog.save(protocol_dir / "puzzle_catalog.json"),
        "session": session.save(protocol_dir / "night_puzzles.json"),
        "assignment": assignment.save(protocol_dir / "assignment.json"),
        "cue_library": cue_library.save(protocol_dir / "cue_library.json"),
    }


def _synthetic_night_fields(night) -> str:
    from muse_tmr.testing import SYNTHETIC_NOTE

    return (
        f"scenario={night.config.scenario} preset={night.config.preset} seed={night.config.seed} "
        f"hours={night.config.hours:g} duration_seconds={night.generation['duration_seconds']:g} "
        f"stop_reason={night.generation['stop_reason']} "
        f"size_mb={night.raw_path.stat().st_size / 1e6:.1f} "
        f"generation_seconds={night.generation_seconds:.1f} sha256={night.sha256} "
        f"output={night.recording_dir} ({SYNTHETIC_NOTE})"
    )


def _synthetic_output_refusal(directory: Path, earlier_run: str) -> Optional[str]:
    """Why a synthetic night or demo may not be written into ``directory``, or None.

    It may be new, empty but for hidden files such as .DS_Store, or hold an earlier run
    of the same command, whose ``earlier_run`` file (metadata.json of a night,
    demo_summary.json of a demo) says ``synthetic: true``. Anything else can be a
    recording, a Polar folder or protocol files, which synthetic output must neither
    overwrite nor sit next to.
    """
    if not directory.exists():
        return None
    if not directory.is_dir():
        return f"{directory} is not a directory"
    if all(path.name.startswith(".") for path in directory.iterdir()):
        return None
    if _says_synthetic(directory / earlier_run):
        return None
    return (
        f"{directory} is not empty and has no {earlier_run} that says synthetic: true; "
        "choose a new or empty directory"
    )


def _says_synthetic(path: Path) -> bool:
    """Whether the JSON file at ``path`` (a recording's metadata.json, say) holds
    ``synthetic: true`` at its top level, as everything a synthetic night writes does."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("synthetic") is True


def _default_demo_dir() -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _default_path_base() / "data" / "synthetic" / f"demo_{timestamp}"


def _load_demo_baseline(path: Path) -> Mapping[str, object]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not {"scenario", "seed", "hours", "agreement"} <= set(baseline):
        raise ValueError(f"{path} is not a demo baseline: it needs scenario, seed, hours and agreement")
    return baseline


def _demo_baseline_identity(baseline: Mapping[str, object], config) -> Optional[str]:
    """Why ``baseline`` cannot be compared with a run of ``config``, or None."""
    if (
        baseline["scenario"] == config.scenario
        and int(baseline["seed"]) == config.seed
        and math.isclose(float(baseline["hours"]), config.hours)
    ):
        return None
    return (
        f"the baseline is for {baseline['scenario']}, seed {baseline['seed']}, {baseline['hours']} h; "
        f"this run is {config.scenario}, seed {config.seed}, {config.hours:g} h"
    )


def _demo_baseline_mismatches(baseline: Mapping[str, object], summary: Mapping[str, object]) -> List[str]:
    """How a demo summary differs from a baseline: the raw file's sha256 exactly, the
    agreement within the baseline's ``tolerance``, its ``pilot3`` values exactly."""
    mismatches = []
    expected_sha256 = baseline.get("raw_sha256")
    if expected_sha256 and expected_sha256 != summary.get("raw_sha256"):
        mismatches.append(f"raw_sha256 {summary.get('raw_sha256')} != {expected_sha256}")
    tolerance = float(baseline.get("tolerance", 0.0))
    for name, expected in dict(baseline["agreement"]).items():
        observed = dict(summary.get("agreement", {})).get(name)
        if not _within_tolerance(observed, expected, tolerance):
            mismatches.append(f"agreement.{name} {observed} != {expected} (tolerance {tolerance:g})")
    for name, expected in dict(baseline.get("pilot3", {})).items():
        observed = dict(summary.get("pilot3", {})).get(name)
        if observed != expected:
            mismatches.append(f"pilot3.{name} {observed} != {expected}")
    return mismatches


def _within_tolerance(observed: object, expected: object, tolerance: float) -> bool:
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(_within_tolerance(o, e, tolerance) for o, e in zip(observed, expected))
        )
    if expected is None or observed is None:
        return expected is None and observed is None
    try:
        return abs(float(observed) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def _synthetic_payload(payload: Mapping[str, object]) -> Dict[str, object]:
    """``payload`` with top-level ``synthetic`` and ``note``, NaN as None."""
    from muse_tmr.testing import SYNTHETIC_NOTE

    data: Dict[str, object] = {"synthetic": True, "note": SYNTHETIC_NOTE}
    data.update(payload)
    return _json_safe(data)


def _write_synthetic_json(path: Path, payload: Mapping[str, object]) -> Path:
    """``payload`` as strict JSON (NaN as null), marked synthetic input, not sleep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_synthetic_payload(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


async def _run_pilot4_cueing(args: argparse.Namespace) -> int:
    return await _run_live_cueing_pilot(
        args,
        command_label="pilot4 cueing",
        pilot_id="m8_pilot4_low_volume_rem_gated_cueing",
        summary_filename="pilot4_summary.json",
        enable_tlr_block=False,
        require_tlr_block=False,
    )


async def _run_pilot5_full_night(args: argparse.Namespace) -> int:
    return await _run_live_cueing_pilot(
        args,
        command_label="pilot5 full-night",
        pilot_id="m8_pilot5_full_night_tlr_puzzle_cueing",
        summary_filename="pilot5_summary.json",
        enable_tlr_block=True,
        require_tlr_block=True,
    )


async def _run_live_cueing_pilot(
    args: argparse.Namespace,
    *,
    command_label: str,
    pilot_id: str,
    summary_filename: str,
    enable_tlr_block: bool,
    require_tlr_block: bool,
) -> int:
    from muse_tmr.audio import load_cue_library, load_volume_calibrations
    from muse_tmr.protocol import (
        load_night_puzzle_session,
        load_puzzle_catalog,
        load_puzzle_cue_assignment,
        load_tlr_block_plan,
    )
    from muse_tmr.validation import run_pilot4_cueing_night

    command = getattr(args, "command", None) or command_label
    try:
        night_arm = _night_arm_from_args(args)
    except _NightArmError as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return 2
    store = load_volume_calibrations(_resolve_output_path(args.calibration))
    calibration = (
        store.latest_for_device(args.device_name)
        if args.device_name is not None
        else store.latest()
    )
    catalog = load_puzzle_catalog(_resolve_output_path(args.catalog))
    session = load_night_puzzle_session(_resolve_output_path(args.session))
    assignment = load_puzzle_cue_assignment(_resolve_output_path(args.assignment))
    cue_library = load_cue_library(_resolve_output_path(args.cue_library))
    tlr_block_plan = (
        load_tlr_block_plan(_resolve_output_path(args.tlr_block))
        if enable_tlr_block
        else None
    )
    try:
        _check_sham_cue(
            night_arm,
            catalog=catalog,
            cue_library=cue_library,
            tlr_block_plan=tlr_block_plan,
            generated_tone_only=True,
        )
    except _NightArmError as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return 2
    tone_refusal = _tlr_tone_refusal(tlr_block_plan, cue_library)
    if tone_refusal is not None:
        print(f"{command}: {tone_refusal}", file=sys.stderr)
        return 2
    config = _live_cueing_config(
        args,
        pilot_id=pilot_id,
        summary_filename=summary_filename,
        enable_tlr_block=enable_tlr_block,
        require_tlr_block=require_tlr_block,
        assignment=assignment,
        tlr_block_plan=tlr_block_plan,
        night_arm=night_arm,
    )
    source = _build_source(args, duration_seconds=int(config.duration_seconds))
    summary = await run_pilot4_cueing_night(
        source,
        config=config,
        catalog=catalog,
        session=session,
        assignment=assignment,
        cue_library=cue_library,
        calibration=calibration,
        tlr_block_plan=tlr_block_plan,
    )
    if night_arm is not None:
        # Never the arm, nor what would give it away: the plays, their volumes, the
        # first play and the scorer's status.
        print(
            f"{command_label} complete "
            f"passed={summary.passed} "
            f"epochs={summary.epoch_count} "
            f"stride_seconds={summary.stride_seconds} "
            f"first_gate_open_seconds={summary.first_gate_open_seconds} "
            f"requested_backend={summary.requested_audio_backend_name} "
            f"resolved_backend={summary.resolved_audio_backend_name} "
            f"emergency_stop={summary.emergency_stop_path} "
            f"protocol={config.protocol.get('name')} "
            + (f"night_id={summary.night_id} " if summary.night_id else "")
            + f"summary={summary.summary_path} ({BLINDED_NOTE})"
        )
        return 0 if summary.passed else 1
    print(
        f"{command_label} complete "
        f"passed={summary.passed} "
        f"epochs={summary.epoch_count} "
        f"stride_seconds={summary.stride_seconds} "
        f"first_gate_open_seconds={summary.first_gate_open_seconds} "
        f"first_play_seconds={summary.first_play_seconds} "
        f"scheduler_play={summary.cue_play_count} "
        f"tlr_play={summary.tlr_cue_play_count} "
        f"puzzle_play={summary.puzzle_cue_play_count} "
        f"audio_status={dict(summary.audio_status_counts)} "
        f"requested_backend={summary.requested_audio_backend_name} "
        f"resolved_backend={summary.resolved_audio_backend_name} "
        f"max_effective_volume={summary.max_effective_volume} "
        f"emergency_stop={summary.emergency_stop_path} "
        f"cue_arousal={summary.cue_arousal.get('status')} "
        f"protocol={config.protocol.get('name')} "
        f"summary={summary.summary_path}"
    )
    return 0 if summary.passed else 1


def _live_cueing_config(
    args: argparse.Namespace,
    *,
    pilot_id: str,
    summary_filename: str,
    enable_tlr_block: bool,
    require_tlr_block: bool,
    assignment=None,
    tlr_block_plan=None,
    night_arm=None,
):
    """The Pilot 4/5 config for ``args``: the protocol preset's gate and scheduler
    settings, explicit flags over them, the audio caps from the flags alone, and the
    night's arm, which changes what a cue becomes but none of those settings."""
    from muse_tmr.features import EpochConfig
    from muse_tmr.models import RemGateConfig
    from muse_tmr.presets import preset_provides_optics
    from muse_tmr.protocol.arousal_guard import ArousalGuardConfig
    from muse_tmr.validation import Pilot4CueingConfig

    duration_seconds = (
        args.duration_seconds
        if args.duration_seconds is not None
        else args.duration_hours * 3600
    )
    preset, overrides = _protocol_from_args(args, _NIGHT_PRESET_FLAGS)
    settings = preset.runner_settings(overrides)
    enter_threshold, exit_threshold = _resolve_gate_thresholds(args)
    optics_capable = preset_provides_optics(getattr(args, "preset", None))
    return Pilot4CueingConfig(
        output_dir=_resolve_output_dir(args.output_dir),
        duration_seconds=duration_seconds,
        source_name=args.source,
        allow_short=args.allow_short,
        hard_max_volume=args.hard_max_volume,
        default_volume=args.default_volume,
        fade_in_seconds=args.fade_in_seconds,
        fade_out_seconds=args.fade_out_seconds,
        audio_backend_name=args.backend,
        allow_dry_run_fallback=args.allow_dry_run,
        emergency_stop_path=(
            _resolve_output_path(args.emergency_stop_file)
            if args.emergency_stop_file is not None
            else None
        ),
        epoch_config=EpochConfig(
            epoch_seconds=args.epoch_seconds,
            stride_seconds=args.stride_seconds,
        ),
        gate_config=RemGateConfig(
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold,
            min_stable_seconds=settings.gate_min_stable_seconds,
            epoch_seconds=args.epoch_seconds,
            cooldown_seconds=args.gate_cooldown_seconds,
            optics_capable=optics_capable,
        ),
        scheduler_config=preset.scheduler_config(enable_tlr_block=enable_tlr_block, overrides=overrides),
        arousal_guard_config=ArousalGuardConfig(
            optics_capable=optics_capable,
            **_guard_hr_rule_config(args),
        ),
        pilot_id=pilot_id,
        summary_filename=summary_filename,
        require_tlr_block=require_tlr_block,
        protocol=_protocol_block(
            preset,
            overrides,
            assignment=assignment.metadata if assignment is not None else None,
            tlr_block_plan=tlr_block_plan.metadata if tlr_block_plan is not None else None,
        ),
        night_arm=night_arm,
    )


def _log_pilot4_awakening(args: argparse.Namespace) -> int:
    from muse_tmr.validation import AwakeningEvent, append_awakening_event

    output_path = append_awakening_event(
        _resolve_output_path(args.output),
        AwakeningEvent(
            event_type=args.event_type,
            timestamp_utc=args.timestamp_utc,
            notes=args.notes,
            timestamp_seconds=getattr(args, "timestamp_seconds", None),
            attributed_to_cue=getattr(args, "attributed_to_cue", None),
        ),
    )
    print(f"awakening event logged output={output_path}")
    return 0


async def _score_cue_arousal(args: argparse.Namespace) -> int:
    """Score a night's or a Pilot 3 report's cues and write the three outputs."""
    from muse_tmr.presets import preset_provides_optics
    from muse_tmr.reports.cue_arousal import (
        SYNTHETIC_NOTE,
        find_night_summary,
        score_cue_arousal,
        write_summary_block,
    )

    recording_dir = _resolve_output_path(args.recording_dir)
    pilot3_report = _optional_path(args.pilot3_report)
    try:
        report = await score_cue_arousal(
            recording_dir,
            pilot3_report=pilot3_report,
            summary_path=_optional_path(args.summary),
            scheduler_events_path=_optional_path(args.scheduler_events),
            audio_log_path=_optional_path(args.audio_log),
            gate_log_path=_optional_path(args.gate_log),
            awakening_events_path=_optional_path(args.awakening_events),
            replay_input=_optional_path(args.replay_input),
            optics_capable=preset_provides_optics(args.preset) if args.preset else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"score-cue-arousal: {exc}", file=sys.stderr)
        return 2
    if args.output_dir is not None:
        output_dir = _resolve_output_dir(args.output_dir)
    elif pilot3_report is not None:
        output_dir = pilot3_report.parent
    else:
        output_dir = recording_dir
    paths = report.save_outputs(output_dir)
    summary_note = ""
    if pilot3_report is None:
        # Pilot 4/5 put the block into the summary when the night ended; scored again, for
        # example after awakening markers were added, the block follows the new report.
        # It keeps that report's paths, so only a report written next to it replaces it.
        summary_file = find_night_summary(recording_dir, _optional_path(args.summary))
        if (
            summary_file is not None
            and summary_file.parent.resolve() == Path(output_dir).resolve()
            and write_summary_block(summary_file, report.summary_block(paths))
        ):
            summary_note = f" summary={summary_file}"
    outcomes = report.outcomes
    per_cue = dict(outcomes.get("arousal_proxy_per_cue", {}) or {})
    per_control = dict(outcomes.get("arousal_proxy_per_control", {}) or {})
    note = f" ({SYNTHETIC_NOTE})" if report.synthetic else ""
    print(
        "cue arousal proxies scored "
        f"cues={outcomes.get('played_cue_count', 0)} "
        f"controls={outcomes.get('control_count', 0)} "
        f"any_per_cue={_format_float(per_cue.get('any'), 3)} "
        f"any_per_control={_format_float(per_control.get('any'), 3)} "
        f"limitations={','.join(report.limitation_codes)} "
        f"report={paths['report_path']}{summary_note}{note}"
    )
    return 0


def _optional_path(value: Optional[Path]) -> Optional[Path]:
    return None if value is None else _resolve_output_path(value)


def _protocol_presets(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import get_protocol_preset, list_protocol_presets

    if args.name:
        try:
            preset = get_protocol_preset(args.name)
        except KeyError as exc:
            print(f"protocol-presets: {exc.args[0]}", file=sys.stderr)
            return 2
        payload = {
            "content_hash": preset.content_hash(),
            "manifest": preset.manifest(),
            "preset": preset.to_dict(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    rows = [
        {
            "name": preset.name,
            "version": preset.version,
            "runnable": preset.runnable,
            "content_hash": preset.content_hash(),
            "source_checks": sorted({citation.source_check for citation in preset.citations}),
            "title": preset.title,
        }
        for preset in list_protocol_presets()
    ]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    for row in rows:
        print(
            f"{row['name']}\tv{row['version']}\t"
            f"{'runnable' if row['runnable'] else 'documentation only'}\t"
            f"{row['content_hash'][:12]}\t{','.join(row['source_checks']) or 'no citation'}\t{row['title']}"
        )
    return 0


def _protocol_fidelity(args: argparse.Namespace) -> int:
    from muse_tmr.reports.protocol_fidelity import SYNTHETIC_NOTE, build_protocol_fidelity

    recording_dir = _resolve_output_dir(args.recording_dir)
    pilot3_report = _resolve_output_path(args.pilot3_report) if args.pilot3_report is not None else None
    if pilot3_report is not None and not pilot3_report.is_file():
        print(f"protocol-fidelity: no Pilot 3 report at {pilot3_report}", file=sys.stderr)
        return 2
    if pilot3_report is None and not recording_dir.is_dir():
        print(f"protocol-fidelity: no recording directory at {recording_dir}", file=sys.stderr)
        return 2
    try:
        report = build_protocol_fidelity(
            recording_dir,
            pilot3_report=pilot3_report,
            preset_name=args.protocol_preset,
        )
    except KeyError as exc:
        print(f"protocol-fidelity: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    if args.output_dir is not None:
        output_dir = _resolve_output_dir(args.output_dir)
    else:
        output_dir = pilot3_report.parent if pilot3_report is not None else recording_dir
    paths = report.save_outputs(output_dir)
    statuses = " ".join(f"{name}={status}" for name, status in report.statuses().items())
    note = f" ({SYNTHETIC_NOTE})" if report.synthetic else ""
    print(
        "protocol fidelity "
        f"preset={report.preset['name']} "
        f"rem_periods={report.summary['rem_periods']} "
        f"plays={dict(report.summary['plays'])} "
        f"{statuses} "
        f"limitations={','.join(report.limitations) or 'none'} "
        f"report={paths['report']}{note}"
    )
    return 0


class _InputEnded(Exception):
    """stdin closed before every sound of the check had an answer."""


def _record_cue_recall(args: argparse.Namespace) -> int:
    from muse_tmr.audio import AudioCuePlayer, load_cue_library, load_volume_calibrations
    from muse_tmr.protocol import (
        load_night_puzzle_session,
        load_puzzle_catalog,
        load_puzzle_cue_assignment,
    )
    from muse_tmr.reports.cue_recall import (
        CueRecallUnavailable,
        build_cue_recall_test,
        cue_recall_playback_config,
        play_cue_recall_trial,
        plan_cue_recall,
    )

    session = load_night_puzzle_session(_resolve_output_path(args.session))
    catalog = load_puzzle_catalog(_resolve_output_path(args.catalog))
    assignment = load_puzzle_cue_assignment(_resolve_output_path(args.assignment))
    assignment.validate_against_session(session)
    store = load_volume_calibrations(_resolve_output_path(args.calibration))
    calibration = store.latest_for_device(args.device_name) if args.device_name else store.latest()
    try:
        plans = plan_cue_recall(session, catalog, load_cue_library(_resolve_output_path(args.cue_library)))
    except CueRecallUnavailable as exc:
        print(f"record-cue-recall: {exc}", file=sys.stderr)
        return 2
    output_path = _resolve_output_path(args.output)
    audio_log = (
        _resolve_output_path(args.audio_log)
        if args.audio_log is not None
        else output_path.with_name(f"{output_path.stem}_audio.jsonl")
    )
    player = AudioCuePlayer(
        cue_recall_playback_config(
            calibration,
            hard_max_volume=args.hard_max_volume,
            default_volume=args.default_volume,
            log_path=audio_log,
        ),
        backend=args.audio_resolution.backend,
    )
    options = len(session.puzzle_ids)
    print("Which puzzle does this sound belong to? Last night's puzzles:")
    for position, puzzle_id in enumerate(session.puzzle_ids, start=1):
        print(f"  {position}. {catalog.get_puzzle(puzzle_id).prompt}")
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    answers: Dict[int, Optional[str]] = {}
    plays: Dict[int, List[Mapping[str, object]]] = {}
    try:
        for plan in plans:
            plays[plan.trial] = []
            while True:
                plays[plan.trial].append(play_cue_recall_trial(player, plan))
                answer = _ask_cue_recall(plan.trial, len(plans), options)
                if answer != "replay":
                    break
            answers[plan.trial] = None if answer is None else session.puzzle_ids[int(answer) - 1]
    except _InputEnded:
        print("record-cue-recall: input ended before every sound had an answer; nothing saved", file=sys.stderr)
        return 1
    test = build_cue_recall_test(
        session,
        catalog,
        assignment,
        answers,
        plays=plays,
        started_at_utc=started_at,
        notes=args.notes,
        metadata={
            "audio_log": audio_log.name,
            "requested_backend": args.backend,
            "resolved_backend": player.backend.name,
        },
    )
    test.save(output_path)
    # The answers' correctness and the conditions stay in the file: the retest comes next.
    print(
        "cue recall saved "
        f"session={test.session_id} "
        f"sounds={test.trial_count} "
        f"answered={test.answered_count} "
        f"backend={player.backend.name} "
        f"output={output_path}"
    )
    return 0


def _ask_cue_recall(trial: int, total: int, options: int):
    """A puzzle number, None for "do not know", or "replay"."""
    prompt = f"Sound {trial} of {total}: puzzle number 1-{options}, r to play it again, Enter if you do not know: "
    while True:
        try:
            text = input(prompt).strip().lower()
        except EOFError:
            raise _InputEnded() from None
        if not text:
            return None
        if text in ("r", "replay"):
            return "replay"
        if text.isdigit() and 1 <= int(text) <= options:
            return int(text)
        print(f"Type a number from 1 to {options}, r, or nothing.")


def _list_cues(args: argparse.Namespace) -> int:
    from muse_tmr.audio import load_cue_library

    library = load_cue_library(_resolve_output_path(args.input))
    cues = library.filter(protocol=args.protocol, tag=args.tag)
    for cue in cues:
        print(
            f"{cue.cue_id}\t{cue.cue_type}\t{cue.protocol}\t"
            f"{cue.duration_seconds}s\ttags={','.join(cue.tags)}"
        )
    return 0


def _create_tlr_cue(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import TlrCueConfig, default_tlr_cue_library

    preset, overrides = _protocol_from_args(args, _TLR_CUE_PRESET_FLAGS)
    tone = preset.tlr_cue_config(cue_id=args.cue_id, overrides=overrides)
    library = default_tlr_cue_library(
        TlrCueConfig(
            cue_id=tone.cue_id,
            frequency_hz=tone.frequency_hz,
            duration_seconds=tone.duration_seconds,
            # The volume hint is this flag's, never the preset's.
            volume_hint=args.volume_hint,
            description=tone.description,
        ),
        protocol=preset.manifest(overrides),
    )
    output_path = library.save(_resolve_output_path(args.output))
    cue = library.cues[0]
    print(
        "TLR cue created "
        f"cue={cue.cue_id} "
        f"frequency_hz={cue.frequency_hz} "
        f"duration_seconds={cue.duration_seconds} "
        f"volume_hint={cue.volume_hint} "
        f"protocol={preset.name} "
        f"output={output_path}"
    )
    return 0


def _train_tlr_cue(args: argparse.Namespace) -> int:
    from muse_tmr.audio import AudioCuePlayer, AudioPlaybackConfig, load_cue_library
    from muse_tmr.protocol import TlrTrainingConfig, train_tlr_cue

    library = load_cue_library(_resolve_output_path(args.cue_library))
    cue = library.by_id(args.cue_id)
    default_volume = args.volume if args.volume is not None else (cue.volume_hint or 0.05)
    player = AudioCuePlayer(
        AudioPlaybackConfig(
            max_volume=args.max_volume,
            default_volume=default_volume,
            device_name=args.device_name,
        ),
        backend=args.audio_resolution.backend,
    )
    session = train_tlr_cue(
        cue,
        player,
        config=TlrTrainingConfig(
            repetitions=args.repetitions,
            interval_seconds=args.interval_seconds,
            volume=args.volume,
            backend_name=args.backend,
        ),
        session_id=args.session_id,
        event_log_path=_resolve_output_path(args.event_log),
        library_id=library.library_id,
    )
    output_path = session.save(_resolve_output_path(args.output))
    print(
        "TLR training complete "
        f"session={session.session_id} "
        f"cue={session.cue_id} "
        f"events={session.event_count} "
        f"output={output_path} "
        f"event_log={_resolve_output_path(args.event_log)}"
    )
    return 0


def _plan_tlr_block(args: argparse.Namespace) -> int:
    from muse_tmr.audio import load_cue_library
    from muse_tmr.protocol import load_tlr_training_session, plan_tlr_block, training_tone_mismatch

    library = load_cue_library(_resolve_output_path(args.cue_library))
    cue = library.by_id(args.cue_id)
    training = None
    if args.training is not None:
        # The plan is the night's record of the tone; it must be the one trained (#264).
        training = load_tlr_training_session(_resolve_output_path(args.training))
        mismatch = training_tone_mismatch(training, cue)
        if mismatch is not None:
            print(f"plan-tlr-block: {mismatch}", file=sys.stderr)
            return 2
    preset, overrides = _protocol_from_args(
        args,
        _TLR_BLOCK_PRESET_FLAGS,
        extra_overrides={"tlr_block_enabled": False} if args.disabled else None,
    )
    metadata: Dict[str, object] = {"protocol": _protocol_block(preset, overrides, tlr_cue=cue.metadata)}
    if args.protocol_preset is not None:
        tone_warning = _preset_tone_warning(cue, preset, overrides)
        if tone_warning is not None:
            print(tone_warning, file=sys.stderr)
    if training is not None:
        metadata["training"] = {"session_id": training.session_id, "completed_at_utc": training.completed_at_utc}
    plan = plan_tlr_block(
        cue,
        config=preset.tlr_block_config(overrides),
        metadata=metadata,
        library_id=library.library_id,
    )
    output_path = plan.save(_resolve_output_path(args.output))
    print(
        "TLR block planned "
        f"cue={plan.cue_id} "
        f"frequency_hz={cue.frequency_hz} "
        f"duration_seconds={cue.duration_seconds} "
        f"events={len(plan.events)} "
        f"puzzle_start_offset={plan.puzzle_cue_start_offset_seconds} "
        f"protocol={preset.name} "
        + (f"training={training.session_id} " if training is not None else "")
        + f"output={output_path}"
    )
    return 0


def _preset_tone_warning(cue, preset, overrides: Mapping[str, object]) -> Optional[str]:
    """A warning when ``cue`` records no protocol preset and is not the preset's TLR tone,
    as the starter library's tlr_soft_tone is without create-cue-library --protocol-preset
    (#264). A cue made with another preset is named by _protocol_block()."""
    from muse_tmr.protocol import TlrTone

    if isinstance(cue.metadata, Mapping) and isinstance(cue.metadata.get("protocol"), Mapping):
        return None
    config = preset.tlr_cue_config(cue_id=cue.cue_id, overrides=overrides)
    tone = TlrTone(cue_id=cue.cue_id, frequency_hz=config.frequency_hz, duration_seconds=config.duration_seconds)
    if not tone.differs_from(cue):
        return None
    return (
        f"warning: the tlr cue {cue.cue_id} is {TlrTone.of_cue(cue).describe()} and records no protocol preset, "
        f"{preset.name}'s TLR tone is {tone.describe()}; create-cue-library --protocol-preset {preset.name} makes "
        "a library with it"
    )


def _import_puzzles(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import import_puzzle_file

    catalog = import_puzzle_file(_resolve_output_path(args.input))
    output_path = catalog.save(_resolve_output_path(args.output))
    print(f"puzzle catalog imported puzzles={catalog.puzzle_count} output={output_path}")
    return 0


def _generate_puzzle_session(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from muse_tmr.protocol import load_puzzle_catalog

    preset, overrides = _protocol_from_args(args, {"count": "puzzle_count"})
    puzzle_count, _cued = preset.puzzle_counts(overrides)
    catalog = load_puzzle_catalog(_resolve_output_path(args.catalog))
    session = catalog.generate_night_session(
        session_id=args.session_id,
        puzzle_count=puzzle_count,
        selection_seed=args.seed,
        include_known=args.include_known,
    )
    session = replace(session, metadata={**session.metadata, "protocol": preset.manifest(overrides)})
    output_path = session.save(_resolve_output_path(args.output))
    print(
        "puzzle session generated "
        f"session={session.session_id} "
        f"puzzles={len(session.puzzle_ids)} "
        f"eligible={session.metadata.get('eligible_count')} "
        f"protocol={preset.name} "
        f"output={output_path}"
    )
    return 0


def _record_puzzle_attempt(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import PuzzleAttempt, load_puzzle_catalog

    input_path = _resolve_output_path(args.catalog)
    output_path = _resolve_output_path(args.output) if args.output else input_path
    catalog = load_puzzle_catalog(input_path)
    attempt = PuzzleAttempt(
        puzzle_id=args.puzzle_id,
        response=args.response,
        duration_seconds=args.duration_seconds,
        solved=args.solved,
        known_after=args.known_after,
        notes=args.notes,
    )
    updated = catalog.with_attempt(attempt)
    updated.save(output_path)
    print(
        "puzzle attempt recorded "
        f"puzzle={attempt.puzzle_id} "
        f"duration_seconds={attempt.duration_seconds} "
        f"solved={attempt.solved} "
        f"known_after={attempt.known_after} "
        f"output={output_path}"
    )
    return 0


def _record_association_check(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import load_night_puzzle_session, load_puzzle_catalog

    input_path = _resolve_output_path(args.session)
    output_path = _resolve_output_path(args.output) if args.output else input_path
    catalog = load_puzzle_catalog(_resolve_output_path(args.catalog))
    session = load_night_puzzle_session(input_path)
    result = catalog.check_association(
        args.puzzle_id,
        args.response,
        notes=args.notes,
    )
    updated = session.with_association_result(result)
    updated.save(output_path)
    print(
        "association check recorded "
        f"session={updated.session_id} "
        f"puzzle={result.puzzle_id} "
        f"matched={result.matched} "
        f"output={output_path}"
    )
    return 0


def _record_dream_report(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import load_night_puzzle_session, load_puzzle_catalog
    from muse_tmr.reports import build_dream_report

    session = load_night_puzzle_session(_resolve_output_path(args.session))
    catalog = (
        load_puzzle_catalog(_resolve_output_path(args.catalog))
        if args.catalog is not None
        else None
    )
    if args.cues_heard is None and args.cue_perception is None:
        print("record-dream-report: give --cue-perception, --cues-heard or both", file=sys.stderr)
        return 2
    try:
        report = build_dream_report(
            session,
            catalog=catalog,
            report_id=args.report_id,
            lucid=_yes_no(args.lucid),
            cues_heard=_yes_no(args.cues_heard) if args.cues_heard is not None else None,
            cue_perception=args.cue_perception,
            woke_from_sound=args.woke_from_sound,
            eye_signal_attempt_times=tuple(args.eye_signal_attempt),
            confidence=args.confidence,
            dream_text=args.dream_text,
            puzzle_incorporation_text=_parse_puzzle_links(args.puzzle_link),
            notes=args.notes,
        )
    except ValueError as exc:
        print(f"record-dream-report: {exc}", file=sys.stderr)
        return 2
    output_path = report.save(_resolve_output_path(args.output))
    print(
        "dream report recorded "
        f"session={report.session_id} "
        f"lucid={report.lucid} "
        f"cues_heard={report.cues_heard} "
        f"cue_perception={report.cue_perception} "
        f"woke_from_sound={report.woke_from_sound} "
        f"eye_signal_attempts={len(report.eye_signal_attempt_times)} "
        f"linked_puzzles={report.puzzle_incorporation_count} "
        f"output={output_path}"
    )
    return 0


def _record_puzzle_retest(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import (
        load_night_puzzle_session,
        load_puzzle_catalog,
        load_puzzle_cue_assignment,
    )
    from muse_tmr.reports import MorningRetestResult, build_morning_retest

    session = load_night_puzzle_session(_resolve_output_path(args.session))
    catalog = load_puzzle_catalog(_resolve_output_path(args.catalog))
    assignment = load_puzzle_cue_assignment(_resolve_output_path(args.assignment))
    responses = _parse_key_values(args.result, "--result", allow_empty_value=True)
    durations = _parse_float_key_values(args.duration, "--duration")
    confidences = _parse_float_key_values(args.confidence, "--confidence")
    notes = _parse_key_values(args.note, "--note", allow_empty_value=True)

    result_ids = set(responses)
    if not result_ids:
        raise ValueError("record-puzzle-retest requires at least one --result")
    solved_ids = set(args.solved)
    extra_solved = tuple(sorted(solved_ids - result_ids))
    if extra_solved:
        raise ValueError(f"--solved references puzzles without --result: {extra_solved}")
    _require_matching_keys("--duration", durations, result_ids)
    _require_matching_keys("--confidence", confidences, result_ids)

    results = tuple(
        MorningRetestResult(
            puzzle_id=puzzle_id,
            response=response,
            solved=puzzle_id in solved_ids,
            duration_seconds=durations[puzzle_id],
            confidence=confidences[puzzle_id],
            notes=notes.get(puzzle_id, ""),
        )
        for puzzle_id, response in responses.items()
    )
    retest = build_morning_retest(
        session,
        results,
        catalog=catalog,
        assignment=assignment,
        retest_id=args.retest_id,
        notes=args.notes,
        retest_wave=args.wave,
    )
    output_path = retest.save(_resolve_output_path(args.output))
    print(
        "morning puzzle retest recorded "
        f"session={retest.session_id} "
        f"wave={retest.retest_wave} "
        f"results={len(retest.results)} "
        f"solved={retest.solved_count} "
        f"unsolved={retest.unsolved_count} "
        f"output={output_path}"
    )
    return 0


def _plan_night_schedule(args: argparse.Namespace) -> int:
    """Write a sealed series and its companion; print neither an arm nor the seed."""
    import secrets

    from muse_tmr.protocol import generate_night_schedule, write_night_schedule

    seed = args.seed if args.seed is not None else secrets.randbits(32)
    try:
        schedule = generate_night_schedule(
            seed,
            args.blocks,
            args.washout_nights,
            schedule_id=args.schedule_id,
            night_prefix=args.night_prefix,
            sham_cue_id=args.sham_cue_id,
        )
        output_path = (
            _resolve_output_path(args.output)
            if args.output is not None
            else _default_path_base() / "data" / "protocol" / "night_schedules" / f"{schedule.schedule_id}.json"
        )
        schedule_path, sealed_path = write_night_schedule(schedule, output_path)
    except (OSError, ValueError) as exc:
        print(f"plan-night-schedule: {exc}", file=sys.stderr)
        return 2
    nights = ",".join(
        f"{night.night_id}(washout)" if night.arm == "washout" else night.night_id for night in schedule.nights
    )
    print(
        "night schedule sealed "
        f"schedule_id={schedule.schedule_id} "
        f"blocks={schedule.blocks} "
        f"washout_nights={schedule.washout_nights} "
        f"sham_cue_id={schedule.sham_cue_id} "
        f"commitment_sha256={schedule.commitment_sha256()} "
        f"schedule={schedule_path} "
        f"sealed={sealed_path}"
    )
    print(f"nights={nights}")
    print("The arms stay sealed until unseal-night-schedule; do not open the schedule file before then.")
    return 0


def _show_night_schedule(args: argparse.Namespace) -> int:
    return _print_night_schedule(args, unseal=False)


def _unseal_night_schedule(args: argparse.Namespace) -> int:
    return _print_night_schedule(args, unseal=True)


def _print_night_schedule(args: argparse.Namespace, *, unseal: bool) -> int:
    """show-night-schedule and unseal-night-schedule. The arms show only once unsealed;
    a washout night's place follows from the blocks, so it always shows."""
    from muse_tmr.protocol import (
        SealMismatch,
        load_night_schedule,
        load_sealed_commitment,
        sealed_path_for,
        unseal_night_schedule,
    )

    command = "unseal-night-schedule" if unseal else "show-night-schedule"
    schedule_path = _resolve_output_path(args.schedule)
    sealed_path = _resolve_output_path(args.sealed) if args.sealed is not None else sealed_path_for(schedule_path)
    try:
        if unseal:
            schedule = unseal_night_schedule(schedule_path, sealed_path=sealed_path)
            seal = "matches"
        else:
            schedule = load_night_schedule(schedule_path)
            try:
                seal = "matches" if schedule.commitment_sha256() == load_sealed_commitment(sealed_path) else "does_not_match"
            except FileNotFoundError:
                seal = "companion_missing"
    except SealMismatch as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError) as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return 2
    shown = not schedule.sealed
    nights = [
        {
            "night_id": night.night_id,
            "position": night.position,
            "block": night.block,
            "arm": night.arm if shown or night.arm == "washout" else "sealed",
        }
        for night in schedule.nights
    ]
    payload = {
        "schedule_id": schedule.schedule_id,
        "created_at_utc": schedule.created_at_utc,
        "blocks": schedule.blocks,
        "washout_nights": schedule.washout_nights,
        "sham_cue_id": schedule.sham_cue_id,
        "commitment_sha256": schedule.commitment_sha256(),
        "seal": seal,
        "unsealed_at_utc": schedule.unsealed_at_utc,
        "nights": nights,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        state = f"unsealed at {schedule.unsealed_at_utc}" if shown else "sealed"
        print(
            f"night schedule {schedule.schedule_id}: {len(nights)} nights, {schedule.blocks} blocks, "
            f"{schedule.washout_nights} washout nights between blocks, sham cue {schedule.sham_cue_id}; {state}"
        )
        print(f"commitment_sha256={payload['commitment_sha256']} seal={seal} ({sealed_path})")
        for night in nights:
            block = "-" if night["block"] is None else night["block"]
            print(f"{night['position']:>3}  {night['night_id']}  block {block}  {night['arm']}")
    if seal != "matches":
        return 1
    return 0


def _checked_schedule(command: str, schedule_arg: Path, sealed_arg: Optional[Path]):
    """The schedule, after it was checked against its sealed companion; or an exit code
    with the reason on stderr: 1 when it changed after it was sealed, 2 otherwise."""
    from muse_tmr.protocol import SealMismatch, load_night_schedule, sealed_path_for, verify_seal

    schedule_path = _resolve_output_path(schedule_arg)
    sealed_path = _resolve_output_path(sealed_arg) if sealed_arg is not None else sealed_path_for(schedule_path)
    try:
        schedule = load_night_schedule(schedule_path)
        verify_seal(schedule, sealed_path)
    except SealMismatch as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return None, 1
    except (OSError, KeyError, ValueError) as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return None, 2
    return schedule, 0


def _prereg_template(args: argparse.Namespace) -> int:
    """Print or write the preregistration; it holds no arm, no seed and no salt."""
    from muse_tmr.protocol import prereg_template, protocol_block_of
    from muse_tmr.reports.nof1_analysis import (
        DEFAULT_PERMUTATIONS,
        DEFAULT_SEED,
        INTERVAL_LEVEL,
        OUTCOMES,
        outside_possible,
        schedule_design_orders,
    )

    schedule, code = _checked_schedule("prereg-template", args.schedule, args.sealed)
    if schedule is None:
        return code
    if args.protocol_metadata is not None:
        metadata_path = _resolve_output_path(args.protocol_metadata)
        try:
            protocol = protocol_block_of(json.loads(metadata_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"prereg-template: {exc}", file=sys.stderr)
            return 2
        if protocol is None:
            print(f"prereg-template: no protocol block in {metadata_path}", file=sys.stderr)
            return 2
        source = f"`{metadata_path.name}`"
    else:
        preset, overrides = _protocol_from_args(args, {})
        protocol = preset.manifest(overrides)
        source = (
            f"`--protocol-preset {preset.name}`"
            if args.protocol_preset is not None
            else f"`{preset.name}`, the default preset of every runner"
        )
    orders = schedule_design_orders(schedule.blocks, schedule.arms)
    text = prereg_template(
        schedule,
        protocol=protocol,
        protocol_source=source,
        analysis={
            "seed": DEFAULT_SEED,
            "permutations": DEFAULT_PERMUTATIONS,
            "interval": INTERVAL_LEVEL,
            "outcomes": list(OUTCOMES),
            "possible_orders": orders,
            "outside_possible": outside_possible(min(orders, DEFAULT_PERMUTATIONS)),
        },
        software_version=__version__,
    )
    if not schedule.sealed:
        print(
            f"prereg-template: {schedule.schedule_id} was unsealed at {schedule.unsealed_at_utc}; a template written "
            "now is not a preregistration",
            file=sys.stderr,
        )
    if args.output is None:
        sys.stdout.write(text)
        return 0
    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(
        "preregistration template written "
        f"schedule_id={schedule.schedule_id} "
        f"commitment_sha256={schedule.commitment_sha256()} "
        f"protocol={protocol.get('name')} "
        f"output={output_path}"
    )
    return 0


def _export_blind_coding(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import load_night_puzzle_session, load_puzzle_catalog
    from muse_tmr.reports import export_blind_coding_packet, load_dream_report

    try:
        packet = export_blind_coding_packet(
            load_night_puzzle_session(_resolve_output_path(args.session)),
            load_dream_report(_resolve_output_path(args.dream_report)),
            load_puzzle_catalog(_resolve_output_path(args.catalog)),
        )
    except (KeyError, ValueError) as exc:
        print(f"export-blind-coding: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    output_path = packet.save(_resolve_output_path(args.output))
    markdown_path = (
        packet.save_markdown(_resolve_output_path(args.markdown_output)) if args.markdown_output is not None else None
    )
    print(
        "blind coding packet written "
        f"session={packet.session_id} "
        f"puzzles={len(packet.puzzles)} "
        f"output={output_path}"
        + (f" markdown={markdown_path}" if markdown_path is not None else "")
    )
    return 0


def _record_incorporation_coding(args: argparse.Namespace) -> int:
    from muse_tmr.reports import build_incorporation_coding, load_blind_coding_packet
    from muse_tmr.reports.incorporation_coding import parse_coding_categories

    try:
        packet = load_blind_coding_packet(_resolve_output_path(args.packet))
        codes = {
            puzzle_id: parse_coding_categories(text)
            for puzzle_id, text in _parse_key_values(args.code, "--code", allow_empty_value=False).items()
        }
        sheet = build_incorporation_coding(
            packet,
            args.rater_id,
            codes,
            notes=_parse_key_values(args.note, "--note", allow_empty_value=True),
            sheet_notes=args.notes,
        )
    except (KeyError, ValueError) as exc:
        print(f"record-incorporation-coding: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    output_path = sheet.save(_resolve_output_path(args.output))
    print(
        "incorporation coding recorded "
        f"session={sheet.session_id} "
        f"rater={sheet.rater_id} "
        f"puzzles={len(sheet.codings)} "
        f"incorporated={sum(1 for coding in sheet.codings if coding.incorporated)} "
        f"output={output_path}"
    )
    return 0


def _compare_incorporation_codings(args: argparse.Namespace) -> int:
    from muse_tmr.reports import coding_agreement, load_incorporation_coding

    try:
        agreement = coding_agreement(load_incorporation_coding(_resolve_output_path(path)) for path in args.codings)
    except (KeyError, ValueError) as exc:
        print(f"compare-incorporation-codings: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    first, second = agreement.rater_ids
    print(f"incorporation coding agreement raters={first},{second} puzzles={agreement.item_count}")
    for category, stats in agreement.categories.items():
        print(
            f"  {category}: agreement={_format_float(stats.get('percent_agreement'), 3)} "
            f"kappa={_format_float(stats.get('kappa'), 3)}"
            + (f" ({stats['kappa_note']})" if stats.get("kappa_note") else "")
        )
    if agreement.unmatched:
        print(f"  coded by one rater only, left out: {len(agreement.unmatched)}")
    if args.output is not None:
        print(f"output={agreement.save(_resolve_output_path(args.output))}")
    return 0


def _analyze_nof1(args: argparse.Namespace) -> int:
    from muse_tmr.reports import ScheduleStillSealed, build_nof1_analysis, load_series_inputs
    from muse_tmr.reports.nof1_analysis import OUTCOMES, SYNTHETIC_NOTE

    if args.primary_outcome is not None and args.primary_outcome not in OUTCOMES:
        print(f"analyze-nof1: --primary-outcome must be one of: {', '.join(OUTCOMES)}", file=sys.stderr)
        return 2
    schedule, code = _checked_schedule("analyze-nof1", args.schedule, args.sealed)
    if schedule is None:
        return code
    if schedule.sealed:
        print(
            f"analyze-nof1: night schedule {schedule.schedule_id} is still sealed; run "
            f"`muse-tmr unseal-night-schedule {args.schedule}` after the last delayed retest of the series, "
            "then analyze-nof1",
            file=sys.stderr,
        )
        return 2
    try:
        nights = load_series_inputs(
            schedule,
            protocol_dir=_resolve_output_dir(args.protocol_dir),
            reports_dir=_resolve_output_dir(args.reports_dir),
            recordings_dir=_resolve_output_dir(args.recordings_dir),
            summary_paths=tuple(_resolve_output_path(path) for path in args.summary),
            coding_paths=tuple(_resolve_output_path(path) for path in args.coding),
        )
        report = build_nof1_analysis(
            schedule,
            nights,
            seed=args.seed,
            permutations=args.permutations,
            min_nights_per_arm=args.min_nights_per_arm,
            primary_outcome=args.primary_outcome,
            primary_contrast=args.primary_contrast,
        )
    except ScheduleStillSealed as exc:
        print(f"analyze-nof1: {exc}", file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as exc:
        print(f"analyze-nof1: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    output_path = report.save(_resolve_output_path(args.output))
    markdown_path = (
        report.save_markdown(_resolve_output_path(args.markdown_output)) if args.markdown_output is not None else None
    )
    with_data = sum(1 for night in report.nights if night.get("has_data"))
    print(
        "nof1 analysis complete "
        f"schedule={schedule.schedule_id} "
        f"nights={with_data}/{len(report.nights)} "
        f"orders={report.settings['possible_orders']} ({report.settings['method']}) "
        f"limitations={','.join(report.limitation_codes)} "
        f"output={output_path}"
        + (f" markdown={markdown_path}" if markdown_path is not None else "")
        + (f" ({SYNTHETIC_NOTE})" if report.synthetic else "")
    )
    return 0


def _analyze_cued_uncued(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import (
        load_night_puzzle_session,
        load_puzzle_cue_assignment,
        load_tmr_scheduler_events,
    )
    from muse_tmr.protocol import protocol_block_of
    from muse_tmr.reports import (
        build_cued_uncued_analysis,
        load_cue_recall_test,
        load_dream_report,
        load_incorporation_coding,
        load_morning_retest,
    )

    session = load_night_puzzle_session(_resolve_output_path(args.session))
    assignment = load_puzzle_cue_assignment(_resolve_output_path(args.assignment))
    retests = tuple(load_morning_retest(_resolve_output_path(path)) for path in args.retest)
    codings = tuple(load_incorporation_coding(_resolve_output_path(path)) for path in args.coding)
    dream_report = (
        load_dream_report(_resolve_output_path(args.dream_report))
        if args.dream_report is not None
        else None
    )
    scheduler_events = (
        load_tmr_scheduler_events(_resolve_output_path(args.scheduler_events))
        if args.scheduler_events is not None
        else ()
    )
    cue_recall = (
        load_cue_recall_test(_resolve_output_path(args.cue_recall))
        if args.cue_recall is not None
        else None
    )
    protocol, protocol_source = None, None
    if args.protocol_metadata is not None:
        metadata_path = _resolve_output_path(args.protocol_metadata)
        protocol = protocol_block_of(json.loads(metadata_path.read_text(encoding="utf-8")))
        if protocol is None:
            print(f"analyze-cued-uncued: no protocol block in {metadata_path}", file=sys.stderr)
            return 2
        protocol_source = metadata_path.name
    elif protocol_block_of(assignment.to_dict()) is not None:
        protocol, protocol_source = protocol_block_of(assignment.to_dict()), "assignment"
    try:
        report = build_cued_uncued_analysis(
            session,
            assignment,
            retests,
            dream_report=dream_report,
            scheduler_events=scheduler_events,
            cue_recall=cue_recall,
            incorporation_codings=codings,
            protocol=protocol,
            protocol_source=protocol_source,
            analysis_id=args.analysis_id,
            min_group_size=args.min_group_size,
        )
    except ValueError as exc:
        print(f"analyze-cued-uncued: {exc}", file=sys.stderr)
        return 2
    output_path = report.save(_resolve_output_path(args.output))
    markdown_path = None
    if args.markdown_output is not None:
        markdown_path = report.save_markdown(_resolve_output_path(args.markdown_output))

    print(
        "cued-vs-uncued analysis complete "
        f"session={report.session_id} "
        f"rows={len(report.rows)} "
        f"waves={','.join(report.wave_metrics)} "
        f"incorporation={report.incorporation.get('source')} "
        f"limitations={','.join(report.limitation_codes)} "
        f"output={output_path}"
        + (f" markdown={markdown_path}" if markdown_path is not None else "")
    )
    return 0


def _assign_puzzle_cues(args: argparse.Namespace) -> int:
    from muse_tmr.protocol import assign_cued_uncued_puzzles, load_night_puzzle_session

    session = load_night_puzzle_session(_resolve_output_path(args.session))
    preset, overrides = _protocol_from_args(args, {"cued_count": "cued_puzzle_count"})
    _count, cued_count = preset.puzzle_counts(overrides)
    assignment = assign_cued_uncued_puzzles(
        session,
        seed=args.seed,
        cued_count=cued_count,
        protocol=_protocol_block(preset, overrides, session=session.metadata),
    )
    output_path = assignment.save(_resolve_output_path(args.output))
    print(
        "puzzle cue assignment generated "
        f"session={assignment.session_id} "
        f"seed={assignment.seed} "
        f"cued={len(assignment.cued_puzzle_ids)} "
        f"uncued={len(assignment.uncued_puzzle_ids)} "
        f"scheduled={len(assignment.scheduled_puzzle_ids)} "
        f"protocol={preset.name} "
        f"output={output_path}"
    )
    return 0


def _parse_puzzle_links(values: Sequence[str]) -> dict:
    return _parse_key_values(values, "--puzzle-link", allow_empty_value=False)


def _parse_key_values(
    values: Sequence[str],
    option_name: str,
    *,
    allow_empty_value: bool = False,
) -> dict:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} must use PUZZLE_ID=VALUE")
        puzzle_id, text = value.split("=", 1)
        puzzle_id = puzzle_id.strip()
        text = text.strip()
        if not puzzle_id or (not text and not allow_empty_value):
            raise ValueError(f"{option_name} must include non-empty puzzle ID and value")
        if puzzle_id in parsed:
            raise ValueError(f"duplicate {option_name} for puzzle_id={puzzle_id}")
        parsed[puzzle_id] = text
    return parsed


def _parse_float_key_values(values: Sequence[str], option_name: str) -> dict:
    parsed = _parse_key_values(values, option_name, allow_empty_value=False)
    return {key: float(value) for key, value in parsed.items()}


def _require_matching_keys(option_name: str, values: Mapping[str, object], expected_keys: set) -> None:
    missing = tuple(sorted(expected_keys - set(values)))
    extra = tuple(sorted(set(values) - expected_keys))
    if missing or extra:
        raise ValueError(f"{option_name} keys must match --result keys: missing={missing} extra={extra}")


def _yes_no(value: str) -> bool:
    return value == "yes"


def _build_source(args: argparse.Namespace, duration_seconds: int):
    if getattr(args, "source", "amused") == "sdk":
        from muse_tmr.sources.muse_sdk_source_stub import MuseSdkSourceConfig, MuseSdkSourceStub

        return MuseSdkSourceStub(
            MuseSdkSourceConfig(
                sdk_path=getattr(args, "sdk_path", None),
                metadata={"duration_seconds": str(duration_seconds)},
            )
        )

    if getattr(args, "source", "amused") == "openmuse":
        from muse_tmr.sources.openmuse_lsl_source import OpenMuseLslConfig, OpenMuseLslSource

        return OpenMuseLslSource(
            OpenMuseLslConfig(
                stream_names=_openmuse_stream_names(args),
                required_modalities=tuple(getattr(args, "require_lsl_stream", ()) or ()),
                resolve_timeout_seconds=getattr(args, "lsl_resolve_timeout", 5.0),
                pull_timeout_seconds=getattr(args, "lsl_pull_timeout", 0.0),
                poll_interval_seconds=getattr(args, "lsl_poll_interval", 0.01),
                duration_seconds=duration_seconds,
            )
        )

    if getattr(args, "source", "amused") == "brainflow":
        from muse_tmr.sources.brainflow_source import BrainFlowSource, BrainFlowSourceConfig

        return BrainFlowSource(
            BrainFlowSourceConfig(
                board_name=getattr(args, "brainflow_board", "MUSE_S_ATHENA_BOARD"),
                preset=getattr(args, "brainflow_preset", "p1041"),
                low_latency=not getattr(args, "brainflow_no_low_latency", False),
                address=getattr(args, "address", None),
                serial_number=getattr(args, "brainflow_serial_number", None),
                name_filter=getattr(args, "name_filter", "Muse"),
                duration_seconds=duration_seconds,
                poll_interval_seconds=getattr(args, "brainflow_poll_interval", 0.05),
                max_chunk_samples=getattr(args, "brainflow_chunk_samples", 256),
                connect_timeout_seconds=getattr(args, "brainflow_connect_timeout", 20.0),
                stream_start_timeout_seconds=getattr(
                    args,
                    "brainflow_stream_start_timeout",
                    10.0,
                ),
                stop_timeout_seconds=getattr(args, "brainflow_stop_timeout", 10.0),
                session_cooldown_seconds=getattr(args, "brainflow_session_cooldown", 2.0),
            )
        )

    from muse_tmr.sources.amused_source import AmusedSource

    return AmusedSource(
        address=getattr(args, "address", None),
        name_filter=getattr(args, "name_filter", "Muse"),
        preset=getattr(args, "preset", "p1034"),
        duration_seconds=duration_seconds,
        verbose=not getattr(args, "quiet", False),
    )


def _add_protocol_preset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protocol-preset",
        default=None,
        metavar="NAME",
        help="Protocol preset the protocol settings come from (muse-tmr protocol-presets lists them); "
        "default remind_default_v1. A flag given explicitly wins and is recorded as an override.",
    )


def _add_night_protocol_args(parser: argparse.ArgumentParser) -> None:
    """The gate and scheduler flags of a night or a replay, and --protocol-preset.

    The four protocol flags default to None: left out, they take the preset's value.
    """
    parser.add_argument("--min-stable-seconds", type=float, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    parser.add_argument("--gate-cooldown-seconds", type=float, default=120.0)
    parser.add_argument("--puzzle-cue-interval-seconds", type=float, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    parser.add_argument("--scheduler-cooldown-seconds", type=float, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    parser.add_argument("--max-puzzle-cues-per-block", type=int, default=None, help=PROTOCOL_PRESET_DEFAULT_HELP)
    _add_protocol_preset_arg(parser)


def _add_night_arm_args(parser: argparse.ArgumentParser) -> None:
    """The arm flags of a cueing night or a replay (#228). The command never prints the arm."""
    arm = parser.add_mutually_exclusive_group()
    arm.add_argument(
        "--arm",
        choices=NIGHT_ARM_CHOICES,
        default=None,
        help="Run the night as this arm, open label: cue plays the cues, sham an untrained sound in "
        "their place, no_cue nothing. Default: as before arms, the cues.",
    )
    arm.add_argument(
        "--night-schedule",
        type=Path,
        default=None,
        help="A sealed schedule from plan-night-schedule: the night's arm comes from it after its "
        "commitment is checked, and is never printed.",
    )
    parser.add_argument("--night-id", default=None, help="The night's ID; required with --night-schedule.")
    parser.add_argument(
        "--sham-cue-id",
        default=None,
        help="The untrained cue of the sham arm. Default: the schedule's, else sham_soft_tone.",
    )


class _NightArmError(Exception):
    """The arm flags cannot run. The message never says which arm the night has."""


def _night_arm_from_args(args: argparse.Namespace):
    """The night's NightArmContext from --arm or --night-schedule, or None without them."""
    from muse_tmr.audio import DEFAULT_SHAM_CUE_ID
    from muse_tmr.protocol import NightArmContext, night_arm_from_schedule

    schedule_path = getattr(args, "night_schedule", None)
    arm = getattr(args, "arm", None)
    night_id = getattr(args, "night_id", None)
    sham_cue_id = getattr(args, "sham_cue_id", None)
    try:
        if schedule_path is not None:
            if not night_id:
                raise _NightArmError("--night-schedule needs --night-id")
            return night_arm_from_schedule(
                _resolve_output_path(schedule_path), night_id, sham_cue_id=sham_cue_id
            )
        if arm is not None:
            return NightArmContext(
                arm=arm,
                night_id=night_id,
                sham_cue_id=sham_cue_id or (DEFAULT_SHAM_CUE_ID if arm == "sham" else None),
            )
    except KeyError as exc:
        raise _NightArmError(exc.args[0] if exc.args else str(exc)) from None
    except (OSError, ValueError) as exc:
        raise _NightArmError(str(exc)) from None
    if night_id is not None or sham_cue_id is not None:
        raise _NightArmError("--night-id and --sham-cue-id go with --arm or --night-schedule")
    return None


def _tlr_tone_refusal(tlr_block_plan, cue_library) -> Optional[str]:
    """Why the night's cue library would not play the TLR tone its plan was made with, or
    None (#264). Checked in every arm before anything runs, as the sham cue is."""
    from muse_tmr.protocol import tlr_tone_mismatch

    if tlr_block_plan is None:
        return None
    try:
        return tlr_tone_mismatch(tlr_block_plan, cue_library)
    except KeyError as exc:
        return f"the TLR block plan's cue is not in the night's cue library ({exc.args[0] if exc.args else exc})"


def _check_sham_cue(night_arm, *, catalog, cue_library, tlr_block_plan=None, generated_tone_only=False) -> None:
    """The sham cue of a night that names one, checked before anything runs, in every arm."""
    from muse_tmr.protocol import tlr_cue_ids_of, validate_sham_cue

    if night_arm is None or not night_arm.sham_cue_id:
        return
    try:
        validate_sham_cue(
            night_arm.sham_cue_id,
            catalog=catalog,
            cue_library=cue_library,
            tlr_cue_ids=tlr_cue_ids_of(tlr_block_plan),
        )
        sham = cue_library.by_id(night_arm.sham_cue_id)
    except KeyError as exc:
        raise _NightArmError(f"sham cue: {exc.args[0] if exc.args else exc}") from None
    except ValueError as exc:
        raise _NightArmError(str(exc)) from None
    if generated_tone_only and (sham.cue_type != "generated_tone" or sham.frequency_hz is None):
        raise _NightArmError(f"sham cue {sham.cue_id} must be a generated_tone with frequency_hz")


def _add_guard_hr_rule_args(parser: argparse.ArgumentParser) -> None:
    """The arousal guard's quality gates for its HR rules; 0 gates nothing (docs/safety.md)."""
    parser.add_argument(
        "--guard-min-ppg-confidence",
        type=_unit_fraction,
        default=0.0,
        help="Skip the guard's HR rules on epochs whose ppg_confidence is below this (default 0: never).",
    )
    parser.add_argument(
        "--guard-min-hr-series-quality",
        type=_unit_fraction,
        default=0.0,
        help="Skip the guard's HR rules on epochs whose heart-rate series has a median quality "
        "below this (default 0: never).",
    )


def _guard_hr_rule_config(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "min_ppg_confidence_for_hr_rule": args.guard_min_ppg_confidence,
        "min_hr_series_quality_for_hr_rule": args.guard_min_hr_series_quality,
    }


def _unit_fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{text} is not between 0 and 1")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"{text} is not a positive number")
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not (math.isfinite(value) and value >= 0):
        raise argparse.ArgumentTypeError(f"{text} is not a number of zero or more")
    return value


def _add_brainflow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--brainflow-board", default="MUSE_S_ATHENA_BOARD")
    parser.add_argument("--brainflow-preset", default="p1041")
    parser.add_argument("--brainflow-serial-number")
    parser.add_argument(
        "--brainflow-no-low-latency",
        action="store_true",
        help="Disable BrainFlow's low_latency option for Muse S Athena.",
    )
    parser.add_argument("--brainflow-poll-interval", type=float, default=0.05)
    parser.add_argument("--brainflow-chunk-samples", type=int, default=256)
    parser.add_argument("--brainflow-connect-timeout", type=float, default=20.0)
    parser.add_argument("--brainflow-stream-start-timeout", type=float, default=10.0)
    parser.add_argument("--brainflow-stop-timeout", type=float, default=10.0)
    parser.add_argument("--brainflow-session-cooldown", type=float, default=2.0)


def _add_openmuse_lsl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lsl-resolve-timeout", type=float, default=5.0)
    parser.add_argument("--lsl-pull-timeout", type=float, default=0.0)
    parser.add_argument("--lsl-poll-interval", type=float, default=0.01)
    parser.add_argument(
        "--require-lsl-stream",
        action="append",
        choices=("eeg", "imu", "ppg", "heart_rate", "battery"),
        default=[],
        help="Require an OpenMuse LSL modality before connect succeeds. Repeat as needed.",
    )
    parser.add_argument("--openmuse-eeg-stream", default="Muse_EEG")
    parser.add_argument("--openmuse-imu-stream", default="Muse_ACCGYRO")
    parser.add_argument("--openmuse-ppg-stream", default="Muse_PPG")
    parser.add_argument("--openmuse-heart-rate-stream", default="Muse_HR")
    parser.add_argument("--openmuse-battery-stream", default="Muse_BATT")


def _add_muse_sdk_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sdk-path",
        type=Path,
        help="Local official Muse SDK path for future adapter work. Never commit this path or its contents.",
    )


def _openmuse_stream_names(args: argparse.Namespace) -> Mapping[str, tuple]:
    return {
        "eeg": (args.openmuse_eeg_stream, "Muse-EEG"),
        "imu": (args.openmuse_imu_stream, "Muse-ACCGYRO"),
        "ppg": (args.openmuse_ppg_stream, "Muse-PPG"),
        "heart_rate": (args.openmuse_heart_rate_stream, "Muse_HEART", "Muse_HEART_RATE"),
        "battery": (args.openmuse_battery_stream, "Muse_BATTERY", "Muse-Telemetry"),
    }


def _default_blink_diagnostic_output(source: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _default_path_base() / "data" / "reports" / f"blink_artifacts_{source}_{timestamp}.json"


def _default_recording_dir() -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _default_path_base() / "data" / "recordings" / timestamp


def _resolve_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser()
    if output_dir.is_absolute():
        return output_dir
    return _default_path_base() / output_dir


def _resolve_output_path(output_path: Path) -> Path:
    output_path = output_path.expanduser()
    if output_path.is_absolute():
        return output_path
    return _default_path_base() / output_path


def _default_path_base() -> Path:
    cwd = Path.cwd()
    if _is_writable_non_root(cwd):
        return cwd.resolve()

    project_root = _find_project_root(Path(__file__).resolve())
    if project_root is not None:
        return project_root

    return Path.home().resolve()


def _is_writable_non_root(path: Path) -> bool:
    if path.parent == path:
        return False
    return os.access(path, os.W_OK)


def _find_project_root(start: Path) -> Optional[Path]:
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "muse_tmr").exists():
            return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())
