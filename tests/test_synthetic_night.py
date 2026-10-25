"""Synthetic nights in the Athena TAG format (#233). Synthetic input, not sleep."""

import asyncio
import contextlib
import csv
import io
import json
import tempfile
import time
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest import mock

import numpy as np

import muse_athena_protocol as proto
from muse_raw_stream import MuseRawStream
from muse_realtime_decoder import MuseRealtimeDecoder

from muse_tmr.audio import load_cue_library
from muse_tmr.cli import main as cli_module
from muse_tmr.cli.main import (
    _demo_baseline_identity,
    _demo_baseline_mismatches,
    _load_demo_baseline,
    _write_demo_protocol,
)
from muse_tmr.cli.main import main as cli_main
from muse_tmr.data.recorder import RecordingSummary
from muse_tmr.data.replay import ReplayConfig, ReplaySession
from muse_tmr.features import EpochBuilder, EpochConfig, extract_eeg_features
from muse_tmr.protocol import load_puzzle_catalog, load_puzzle_cue_assignment
from muse_tmr.testing import (
    SYNTHETIC_NOTE,
    SyntheticNightConfig,
    generate_night,
    heartbeat_onsets,
    scenario_config,
    scenario_names,
    script_night,
)
from muse_tmr.testing import tag_encoder as enc
from muse_tmr.testing.agreement import (
    GateEpoch,
    HypnogramEpoch,
    reference_rem_runs,
    stage_at,
    synthetic_agreement,
)
from muse_tmr.testing.scenarios import BATTERY_DEATH_SECONDS
from muse_tmr.testing.synthetic import OPTICS_RATE_HZ, beat_volume

BASELINE_PATH = Path(__file__).resolve().parent / "fixtures" / "synthetic" / "agreement_baseline.json"

SCENARIOS = (
    "clean_night",
    "fragmented_rem",
    "no_rem_nap",
    "battery_death_2h29",
    "optics_line",
    "bad_contact_af7",
    "movement_heavy",
    "short_allow_short",
)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _decode_all(raw_path):
    """Every packet through the decoder that live streaming and replay use."""
    stream = MuseRawStream(str(raw_path))
    stream.open_read()
    decoder = MuseRealtimeDecoder()
    try:
        for packet in stream.read_packets():
            decoder.decode(packet.data, packet.timestamp)
    finally:
        stream.close()
    return decoder


def _decode_optics(raw_path):
    """LO_NIR as decoded, in order, and (seconds, bpm) of each heart rate the decoder
    derives from it."""
    stream = MuseRawStream(str(raw_path))
    stream.open_read()
    decoder = MuseRealtimeDecoder()
    light, heart_rate = [], []
    try:
        for packet in stream.read_packets():
            decoded = decoder.decode(packet.data, packet.timestamp)
            if decoded.ppg:
                light.extend(decoded.ppg["LO_NIR"])
            if decoded.heart_rate is not None:
                seconds = (packet.timestamp - stream.session_start).total_seconds()
                heart_rate.append((seconds, decoded.heart_rate))
    finally:
        stream.close()
    return np.asarray(light), np.asarray(heart_rate)


def _band_rms(values, low_hz, high_hz, fs=OPTICS_RATE_HZ):
    spectrum = np.fft.rfft(values - np.mean(values))
    freqs = np.fft.rfftfreq(values.size, 1.0 / fs)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sqrt(2.0 * np.sum(np.abs(spectrum[band]) ** 2)) / values.size)


def _beat_average(light, onsets, step=0.002):
    """Light averaged over every beat from its onset, linearly interpolated on a 2 ms
    grid as long as the median beat, and scaled to 0 at its trough and 1 at its peak."""
    times = np.arange(light.size) / OPTICS_RATE_HZ
    window = float(np.median(np.diff(onsets)))
    grid = np.arange(0.0, window, step)
    starts = onsets[onsets + window < times[-1]]
    average = np.mean([np.interp(start + grid, times, light) for start in starts], axis=0)
    return grid, (average - average.min()) / (average.max() - average.min())


async def _replay(recording_dir):
    session = ReplaySession(ReplayConfig(input_path=recording_dir))
    await session.connect()
    return [frame async for frame in session.stream()]


async def _epochs(recording_dir):
    session = ReplaySession(ReplayConfig(input_path=recording_dir))
    await session.connect()
    return [epoch async for epoch in EpochBuilder(EpochConfig()).build(session.stream())]


class TagEncoderRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_eeg_microvolts_within_one_count(self):
        values = self.rng.uniform(-720.0, 720.0, size=(500, 4, 4))
        for row, original in zip(enc.encode_eeg_uv(values), values):
            decoded = proto.decode_eeg(bytes(row), 4)
            self.assertLessEqual(float(np.max(np.abs(decoded - original))), proto.EEG_SCALE_UV_PER_COUNT)

    def test_eeg_counts_are_exact_in_both_modes(self):
        for shape, channels in (((300, 4, 4), 4), ((300, 2, 8), 8)):
            counts = self.rng.integers(0, proto.EEG_COUNTS, size=shape)
            for row, original in zip(enc.encode_eeg_counts(counts), counts):
                np.testing.assert_array_equal(proto.decode_eeg_raw_counts(bytes(row), channels), original)

    def test_eeg_past_the_rails_clips(self):
        counts = enc.eeg_uv_to_counts(np.array([-5000.0, 0.0, 5000.0]))
        self.assertEqual(counts.tolist(), [0, proto.EEG_MID_COUNT, proto.EEG_MAX_COUNT])

    def test_accgyro_within_one_count(self):
        accel = self.rng.uniform(-1.9, 1.9, size=(300, 3, 3))
        gyro = self.rng.uniform(-240.0, 240.0, size=(300, 3, 3))
        for row, a, g in zip(enc.encode_accgyro(accel, gyro), accel, gyro):
            decoded = proto.decode_accgyro(bytes(row))
            self.assertLessEqual(float(np.max(np.abs(decoded[:, :3] - a))), proto.ACC_SCALE)
            self.assertLessEqual(float(np.max(np.abs(decoded[:, 3:] - g))), abs(proto.GYRO_SCALE))

    def test_optics_within_one_count_in_every_mode(self):
        for channels, shape in ((4, (3, 4)), (8, (2, 8)), (16, (1, 16))):
            values = self.rng.uniform(0.0, 31.9, size=(200,) + shape)
            data = enc.encode_optics(values)
            self.assertEqual(data.shape[1], {4: 30, 8: 40, 16: 40}[channels])
            for row, original in zip(data, values):
                decoded = proto.decode_optics(bytes(row), channels)
                self.assertLessEqual(float(np.max(np.abs(decoded - original))), proto.OPTICS_SCALE)

    def test_battery_within_one_count(self):
        percent = self.rng.uniform(0.0, 100.0, size=50)
        for row, original in zip(enc.encode_battery(percent), percent):
            decoded = proto.decode_battery(bytes(row), proto.TAG_BATTERY_2)
            self.assertLessEqual(abs(decoded["battery_percent"] - original), 1 / 256)

    def test_notification_reads_back_as_assembled(self):
        uv = self.rng.uniform(-100.0, 100.0, size=(3, 4, 4))
        eeg = enc.subpacket_records(proto.TAG_EEG_4CH, np.array([254, 255, 256]), enc.encode_eeg_uv(uv))
        accel = np.zeros((1, 3, 3))
        accel[..., 2] = 1.0
        imu = enc.subpacket_records(proto.TAG_ACCGYRO, np.array([7]), enc.encode_accgyro(accel, np.zeros((1, 3, 3))))
        payload = enc.assemble_notification([bytes(eeg[0]), bytes(imu[0]), bytes(eeg[1]), bytes(eeg[2])], 300)

        self.assertEqual(payload[0], len(payload))
        self.assertEqual(payload[1], 300 % 256)
        parsed = proto.parse_payload(payload)
        self.assertEqual([sub["counter"] for sub in parsed["EEG"]], [254, 255, 0])
        self.assertEqual([sub["counter"] for sub in parsed["ACCGYRO"]], [7])
        inspection = proto.inspect_payload(payload)
        self.assertFalse(inspection["truncated"])
        self.assertEqual(inspection["bytes_consumed"], len(payload))
        np.testing.assert_allclose(parsed["EEG"][1]["data"], uv[1], atol=proto.EEG_SCALE_UV_PER_COUNT)

    def test_notification_longer_than_244_bytes_is_refused(self):
        record = bytes(enc.subpacket_records(proto.TAG_EEG_4CH, np.zeros(1), np.zeros((1, 28)))[0])
        with self.assertRaises(ValueError):
            enc.assemble_notification([record] * 8)


class ScriptNightTest(unittest.TestCase):
    def test_same_seed_same_hypnogram(self):
        self.assertEqual(script_night(4), script_night(4))
        self.assertNotEqual(script_night(4).stages, script_night(5).stages)

    def test_eight_hours_has_four_or_five_rem_periods(self):
        for seed in range(6):
            night = script_night(seed)
            self.assertEqual(len(night.stages), 960)
            self.assertIn(len(night.rem_runs()), (4, 5))
            counts = night.stage_counts()
            self.assertGreater(counts["N3"], 0)
            self.assertEqual(night.stages[0], "W")

    def test_nap_without_rem(self):
        night = script_night(1, hours=1.0, rem_periods=0)
        self.assertNotIn("R", night.stages)
        self.assertGreater(night.stage_counts()["N3"], 0)

    def test_interruptions_split_rem_periods(self):
        plain = script_night(2)
        broken = script_night(2, rem_interruptions=3)
        self.assertGreater(len(broken.rem_runs()), 2 * len(plain.rem_runs()))

    def test_arousals_stay_inside_the_night(self):
        night = script_night(3, arousals_per_hour=30.0)
        self.assertGreater(len(night.arousals), 50)
        for arousal in night.arousals:
            self.assertGreaterEqual(arousal.onset_seconds, 0.0)
            self.assertLessEqual(arousal.onset_seconds + arousal.duration_seconds, night.duration_seconds)


class ScenarioTest(unittest.TestCase):
    def test_the_eight_scenarios(self):
        self.assertEqual(scenario_names(), SCENARIOS)
        for name in SCENARIOS:
            config = scenario_config(name, seed=1, hours=0.25)
            config.validate()
            self.assertEqual((config.scenario, config.seed, config.hours), (name, 1, 0.25))
        self.assertEqual(scenario_config("clean_night").preset, "p21")
        self.assertEqual(scenario_config("no_rem_nap").rem_periods, 0)
        self.assertEqual(scenario_config("short_allow_short").hours, 10 / 60)

    def test_unknown_scenario(self):
        with self.assertRaises(ValueError):
            scenario_config("full_moon")

    def test_unsupported_preset(self):
        with self.assertRaises(ValueError):
            SyntheticNightConfig(preset="p1041").validate()

    def test_imu_bundles(self):
        # Since #244 the decoder keeps every ACCGYRO subpacket, so a slow notification
        # rate that bundles several is fine.
        SyntheticNightConfig(notifications_per_second=10.0, imu_bundle_shares=(1.0,)).validate()
        for shares in ((), (0.5, -0.1), (0.0, 0.0)):
            with self.assertRaises(ValueError):
                SyntheticNightConfig(imu_bundle_shares=shares).validate()


class GeneratedRecordingTest(unittest.TestCase):
    """short_allow_short with more BLE gaps than usual, so every counted path runs."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        config = replace(scenario_config("short_allow_short", seed=7), ble_gaps_per_hour=60.0)
        cls.night = generate_night(config, cls.root / "night")
        cls.generation = json.loads((cls.night.recording_dir / "generation.json").read_text(encoding="utf-8"))
        cls.decoder = _decode_all(cls.night.raw_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_layout(self):
        names = sorted(path.name for path in self.night.recording_dir.iterdir())
        self.assertEqual(names, [
            "arousals.csv", "events.jsonl", "generation.json", "hypnogram.csv",
            "metadata.json", "raw_amused.bin", "rem_events.csv", "summary.json",
        ])
        summary = json.loads((self.night.recording_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue({field.name for field in fields(RecordingSummary)} <= set(summary))
        self.assertEqual(summary["stop_reason"], "duration_complete")
        self.assertEqual(summary["duration_seconds"], 600.0)
        self.assertEqual(summary["raw_packet_count"], self.generation["ble"]["frames"])
        metadata = json.loads((self.night.recording_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["source"]["source_name"], "synthetic")
        self.assertEqual(metadata["source"]["metadata"]["preset"], "p1034")
        self.assertIs(metadata["synthetic"], True)
        self.assertEqual(len(_read_csv(self.night.recording_dir / "hypnogram.csv")), 20)

    def test_every_artefact_says_synthetic(self):
        for name in ("metadata.json", "summary.json", "generation.json", "events.jsonl"):
            self.assertIn(SYNTHETIC_NOTE, (self.night.recording_dir / name).read_text(encoding="utf-8"), name)

    def test_generation_lists_every_file_but_the_summary(self):
        files = self.generation["files"]
        self.assertEqual(sorted(files), [
            "arousals.csv", "events.jsonl", "hypnogram.csv", "metadata.json", "raw_amused.bin", "rem_events.csv",
        ])
        self.assertEqual(files["raw_amused.bin"]["bytes"], self.night.raw_path.stat().st_size)

    def test_bundling_follows_the_tag_format(self):
        sizes = []
        imu_per_notification = []
        stream = MuseRawStream(str(self.night.raw_path))
        stream.open_read()
        try:
            for packet in stream.read_packets():
                inspection = proto.inspect_payload(packet.data)
                self.assertEqual(packet.data[0], len(packet.data))
                self.assertFalse(inspection["truncated"])
                self.assertEqual(inspection["unknown_tags"], [])
                imu_per_notification.append(inspection["tags"].count(proto.TAG_ACCGYRO))
                sizes.append(len(packet.data))
        finally:
            stream.close()
        self.assertLessEqual(max(sizes), enc.MAX_NOTIFICATION_BYTES)
        self.assertGreater(len(set(sizes)), 5)
        # Like the committed capture, where 49, 32 and 2 of the 83 notifications with
        # any ACCGYRO carry one, two and three.
        counts = [imu_per_notification.count(n) for n in (1, 2, 3)]
        self.assertEqual(max(imu_per_notification), 3)
        shares = [count / sum(counts) for count in counts]
        for share, expected in zip(shares, self.night.config.imu_bundle_shares):
            self.assertAlmostEqual(share, expected, delta=0.03)

    def test_decoder_stats_equal_the_scripted_repeats_and_losses(self):
        ble = self.generation["ble"]
        subpackets = ble["subpackets"]
        self.assertTrue(ble["gaps"])
        self.assertGreater(subpackets["EEG"]["lost_in_gaps"], 0)
        self.assertGreater(ble["repeated_notifications"], 0)
        stats = self.decoder.get_stats()
        expected_lost = {
            name: counts["lost"] + counts["lost_in_gaps"]
            for name, counts in subpackets.items()
            if counts["lost"] + counts["lost_in_gaps"]
        }
        expected_repeated = {name: counts["repeated"] for name, counts in subpackets.items() if counts["repeated"]}
        self.assertEqual(stats["lost_subpackets"], expected_lost)
        self.assertEqual(stats["repeated_subpackets"], expected_repeated)
        self.assertEqual(stats["packets_decoded"], ble["frames"])
        self.assertEqual(stats["decode_errors"], 0)

    def test_replay_counts_match_generation_minus_losses(self):
        frames = asyncio.run(_replay(self.night.recording_dir))
        ble = self.generation["ble"]
        subpackets = ble["subpackets"]
        self.assertEqual(len(frames), ble["frames"])
        for name, counts in subpackets.items():
            self.assertEqual(counts["delivered"], counts["produced"] - counts["lost"] - counts["lost_in_gaps"], name)
        rows = {
            "eeg": sum(len(frame.eeg.channels_uv["AF7"]) for frame in frames if frame.eeg),
            "imu": sum(len(frame.imu.accelerometer_g) for frame in frames if frame.imu),
            "ppg": sum(len(frame.ppg.channels["LI_NIR"]) for frame in frames if frame.ppg),
            "battery": sum(1 for frame in frames if frame.battery),
        }
        self.assertEqual(rows, {
            "eeg": subpackets["EEG"]["rows"],
            "imu": subpackets["ACCGYRO"]["rows"],
            "ppg": subpackets["OPTICS"]["rows"],
            "battery": subpackets["BATTERY"]["rows"],
        })
        modality_frames = {key: sum(1 for frame in frames if getattr(frame, key)) for key in ble["frames_with"]}
        self.assertEqual(modality_frames, ble["frames_with"])
        missing = sum(missing for frame in frames if frame.eeg for _, missing in frame.eeg.gaps)
        self.assertEqual(missing, 4 * (subpackets["EEG"]["lost"] + subpackets["EEG"]["lost_in_gaps"]))
        self.assertGreater(subpackets["ACCGYRO"]["lost_in_gaps"], 0)
        missing = sum(missing for frame in frames if frame.imu for _, missing in frame.imu.gaps)
        self.assertEqual(missing, 3 * (subpackets["ACCGYRO"]["lost"] + subpackets["ACCGYRO"]["lost_in_gaps"]))


class ReproducibilityTest(unittest.TestCase):
    # One minute of short_allow_short with seed 0. A change to the generator changes it:
    # update it in the same commit, so the bytes of a seed never move unnoticed.
    PINNED_SHA256 = "7e6a92b4ba9e658b48b433a44cd5572ab33745d04f026db17d67f40d4dc7ab78"

    def test_pinned_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = generate_night(output_dir=Path(tmp) / "pinned", scenario="short_allow_short", hours=1 / 60)
        self.assertEqual(night.sha256, self.PINNED_SHA256)

    def test_same_seed_same_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = generate_night(output_dir=Path(tmp) / "a", scenario="short_allow_short", seed=11, hours=0.05)
            second = generate_night(output_dir=Path(tmp) / "b", scenario="short_allow_short", seed=11, hours=0.05)
            other = generate_night(output_dir=Path(tmp) / "c", scenario="short_allow_short", seed=12, hours=0.05)
            self.assertEqual(first.raw_path.read_bytes(), second.raw_path.read_bytes())
            self.assertEqual(first.generation["files"], second.generation["files"])
            self.assertEqual(
                (Path(tmp) / "a" / "generation.json").read_bytes(),
                (Path(tmp) / "b" / "generation.json").read_bytes(),
            )
            self.assertNotEqual(first.sha256, other.sha256)

    def test_short_allow_short_is_small_and_quick(self):
        # The issue hoped for under 1 MB; 10 min at the device's rates cannot be: the EEG
        # payload alone is 64 subpackets/s x 28 bytes x 600 s = 1.08 MB (2.9 MB in all).
        with tempfile.TemporaryDirectory() as tmp:
            started = time.perf_counter()
            night = generate_night(output_dir=Path(tmp) / "short", scenario="short_allow_short")
            elapsed = time.perf_counter() - started
            self.assertLess(night.raw_path.stat().st_size, 3.5e6)
            self.assertLess(elapsed, 2.0)


class StageSignalTest(unittest.TestCase):
    """A fixed hypnogram: 4 min of N3, then 4 min of REM."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        config = SyntheticNightConfig(seed=5, hours=16 * 30 / 3600, preset="p21", stages=("N3",) * 8 + ("R",) * 8)
        cls.night = generate_night(config, Path(cls._tmp.name) / "night")
        cls.epochs = asyncio.run(_epochs(cls.night.recording_dir))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_n3_delta_above_rem_delta(self):
        rows = [extract_eeg_features(epoch) for epoch in self.epochs[:16]]
        n3 = [row.band_powers["delta"] for row in rows[:8]]
        rem = [row.band_powers["delta"] for row in rows[9:16]]
        self.assertGreater(min(n3), max(rem))
        self.assertGreater(
            np.median([row.relative_band_powers["delta"] for row in rows[:8]]),
            np.median([row.relative_band_powers["delta"] for row in rows[9:16]]),
        )

    def test_deflections_only_in_rem_epochs(self):
        stages = {int(row["epoch"]): row["stage"] for row in _read_csv(self.night.recording_dir / "hypnogram.csv")}
        events = _read_csv(self.night.recording_dir / "rem_events.csv")
        self.assertGreaterEqual(len(events), 4)
        for event in events:
            onset, seconds = float(event["onset_s"]), float(event["duration_s"])
            self.assertEqual(stages[int(onset // 30)], "R")
            self.assertEqual(int((onset + seconds) // 30), int(onset // 30))
            self.assertTrue(0.3 <= seconds <= 1.5 and 20.0 <= float(event["amplitude_uv"]) <= 60.0)


class OpticsPulseTest(unittest.TestCase):
    """The optics pulse against what the maintainer's awake p1034 recordings show with
    LO_NIR averaged on Polar H10 R-peaks (local measurements, the recordings are not
    published; the sources are in docs/numbers.md). Synthetic input, not sleep."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        # Without movements, optics losses and BLE gaps the decoded LO_NIR is every sample
        # in order, and it holds only the pulse, the wander and the noise.
        still = replace(
            scenario_config("short_allow_short"), movement_share=0.0, optics_loss_fraction=0.0, ble_gaps_per_hour=0.0
        )
        cls.still = generate_night(still, root / "still")
        cls.light, _ = _decode_optics(cls.still.raw_path)
        cls.nap = generate_night(output_dir=root / "nap", scenario="no_rem_nap", hours=0.25)
        _, cls.heart_rate = _decode_optics(cls.nap.raw_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_one_beat_from_foot_to_foot(self):
        phase = np.linspace(0.0, 1.0, 1001)
        volume = beat_volume(phase)
        self.assertAlmostEqual(float(volume.min()), 0.0, delta=1e-4)
        self.assertAlmostEqual(float(volume.max()), 1.0, delta=1e-4)
        self.assertAlmostEqual(float(phase[np.argmin(volume)]), 0.157, delta=0.002)
        # Continuous from one beat into the next.
        self.assertAlmostEqual(float(beat_volume(np.array([1.0 - 1e-9]))[0]), float(volume[0]), places=6)
        np.testing.assert_allclose(beat_volume(phase + 3.0), volume, atol=1e-9)

    def test_beat_shape(self):
        grid, beat = _beat_average(self.light, heartbeat_onsets(self.still))
        step = grid[1] - grid[0]
        below_half = beat < 0.5
        # A sharp light maximum at the pulse foot, the steepest fall after it, one broad trough.
        self.assertTrue(0.11 <= grid[np.argmax(beat)] <= 0.20, grid[np.argmax(beat)])
        self.assertTrue(0.22 <= grid[np.argmin(np.diff(beat))] + step / 2 <= 0.30, grid[np.argmin(np.diff(beat))])
        self.assertEqual(int(np.sum(np.diff(below_half.astype(int)) == 1)), 1)
        self.assertTrue(0.45 <= np.sum(below_half) * step <= 0.64, np.sum(below_half) * step)
        self.assertTrue(0.04 <= np.mean(beat >= 0.9) <= 0.06, np.mean(beat >= 0.9))

    def test_pulse_noise_and_wander_levels(self):
        dc = float(np.mean(self.light))
        cardiac = _band_rms(self.light, 0.7, 3.0) / dc
        noise = _band_rms(self.light, 8.0, 30.0) / dc
        self.assertTrue(0.0010 <= cardiac <= 0.0029, cardiac)
        self.assertTrue(0.00007 <= noise <= 0.00012, noise)
        self.assertTrue(10.0 <= cardiac / noise <= 33.0, cardiac / noise)
        wander = _band_rms(self.light, 0.1, 0.5) / dc
        self.assertAlmostEqual(wander / cardiac, self.still.config.optics_wander_ratio, delta=0.05)

    def test_decoder_heart_rate_follows_the_script(self):
        onsets = heartbeat_onsets(self.nap)
        beat_times, scripted = onsets[1:], 60.0 / np.diff(onsets)
        # A movement buries the pulse in the light, as on a headband; the decoder holds 5 s.
        moving = [
            (float(row["onset_s"]), float(row["onset_s"]) + float(row["duration_s"]) + 5.0)
            for row in _read_csv(self.nap.recording_dir / "arousals.csv")
            if row["movement"] == "1"
        ]
        checked = 0
        for start in np.arange(0.0, float(self.nap.generation["duration_seconds"]), 30.0):
            end = start + 30.0
            if any(first < end and start < last for first, last in moving):
                continue
            decoded = self.heart_rate[(self.heart_rate[:, 0] >= start) & (self.heart_rate[:, 0] < end), 1]
            expected = scripted[(beat_times >= start) & (beat_times < end)]
            self.assertLessEqual(abs(float(np.median(decoded)) - float(np.median(expected))), 5.0, start)
            checked += 1
        self.assertGreaterEqual(checked, 25)

    def test_no_onsets_without_optics(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = generate_night(output_dir=Path(tmp) / "p21", scenario="clean_night", hours=1 / 60)
        self.assertEqual(heartbeat_onsets(night).size, 0)


class Pilot1Test(unittest.TestCase):
    def _validate(self, recording_dir, *args):
        output = recording_dir / "pilot1.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = cli_main(["validate-pilot1-recording", str(recording_dir), "--output", str(output), *args])
        return code, json.loads(output.read_text(encoding="utf-8"))

    def test_half_hour_clean_night_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = generate_night(output_dir=Path(tmp) / "clean", scenario="clean_night", hours=0.5)
            code, report = self._validate(
                night.recording_dir,
                "--min-duration-hours", "0.5",
                "--required-modality", "eeg",
                "--required-modality", "imu",
            )
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])

    def test_battery_death_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = generate_night(output_dir=Path(tmp) / "dead", scenario="battery_death_2h29")
            code, report = self._validate(night.recording_dir)
        self.assertEqual(night.generation["stop_reason"], "source_ended")
        self.assertAlmostEqual(night.generation["duration_seconds"], BATTERY_DEATH_SECONDS, places=6)
        self.assertEqual(code, 1)
        self.assertIn("stop_reason_duration_complete", report["failed_criteria"])
        self.assertIn("duration_at_least_minimum", report["failed_criteria"])


class AgreementTest(unittest.TestCase):
    def setUp(self):
        stages = ["N2"] * 4 + ["R"] * 4 + ["N2"] * 2 + ["R"] * 2
        self.hypnogram = [
            HypnogramEpoch(index, 30.0 * index, 30.0 * (index + 1), stage) for index, stage in enumerate(stages)
        ]

    def test_rem_runs_from_the_hypnogram(self):
        self.assertEqual(reference_rem_runs(self.hypnogram), [(120.0, 240.0), (300.0, 360.0)])

    def test_scores_the_gate_against_the_hypnogram(self):
        # Open on epochs 5-7 (REM) and 9 (N2, decided at 300 s, as the second REM run starts).
        opened = {5, 6, 7, 9}
        epochs = [GateEpoch(30.0 * index, 30.0, index in opened, 0.9 if index in opened else 0.2) for index in range(12)]
        result = synthetic_agreement(self.hypnogram, epochs)

        self.assertIs(result["synthetic"], True)
        self.assertEqual(result["note"], SYNTHETIC_NOTE)
        self.assertEqual(result["confusion"], {"tp": 3, "fp": 1, "fn": 3, "tn": 5})
        self.assertEqual(result["gate_open_fraction"], 4 / 12)
        self.assertEqual([period["first_gate_open_latency_s"] for period in result["rem_periods"]], [60.0, None])
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["precision"], 0.75)
        # 15 of 36 REM/non-REM pairs ranked right, 18 tied.
        self.assertAlmostEqual(result["p_rem_roc_auc"], (15 + 18 / 2) / 36)

    def test_stage_at(self):
        self.assertEqual(stage_at(self.hypnogram, 0.0), "N2")
        self.assertEqual(stage_at(self.hypnogram, 135.0), "R")
        self.assertEqual(stage_at(self.hypnogram, 240.0), "N2")
        self.assertIsNone(stage_at(self.hypnogram, 360.0))


def _run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


class SynthesizeNightCommandTest(unittest.TestCase):
    def test_list_names_every_scenario(self):
        code, out, _ = _run_cli("synthesize-night", "--list")
        self.assertEqual(code, 0)
        self.assertIn(SYNTHETIC_NOTE, out)
        for name in SCENARIOS:
            self.assertIn(name, out)

    def test_writes_what_generate_night_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "cli"
            args = ("synthesize-night", "--scenario", "short_allow_short", "--seed", "3", "--hours", "0.02")
            code, out, _ = _run_cli(*args, "--output", str(target))
            direct = generate_night(output_dir=Path(tmp) / "direct", scenario="short_allow_short", seed=3, hours=0.02)
            self.assertEqual(code, 0)
            self.assertIn(SYNTHETIC_NOTE, out)
            self.assertIn(f"sha256={direct.sha256}", out)
            self.assertEqual((target / "raw_amused.bin").read_bytes(), direct.raw_path.read_bytes())
            # A synthetic night may be written again over itself.
            self.assertEqual(_run_cli(*args, "--output", str(target))[0], 0)

    def test_refuses_a_directory_holding_a_real_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            recording = Path(tmp) / "night"
            recording.mkdir()
            (recording / "raw_amused.bin").write_bytes(b"recorded")
            (recording / "metadata.json").write_text(json.dumps({"source": {"source_name": "amused"}}), encoding="utf-8")
            code, _, err = _run_cli("synthesize-night", "--scenario", "short_allow_short", "--output", str(recording))
            self.assertEqual(code, 2)
            self.assertIn("choose a new or empty directory", err)
            self.assertEqual((recording / "raw_amused.bin").read_bytes(), b"recorded")

    def test_usage_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "never"
            self.assertEqual(_run_cli("synthesize-night", "--output", str(target))[0], 2)
            self.assertEqual(_run_cli("synthesize-night", "--scenario", "clean_night")[0], 2)
            code, _, err = _run_cli("synthesize-night", "--scenario", "full_moon", "--output", str(target))
            self.assertEqual(code, 2)
            self.assertIn("unknown scenario", err)
            self.assertFalse(target.exists())


class ValidatorsOnASyntheticNightTest(unittest.TestCase):
    """validate-pilot1-recording and simulate-replay-cues run by hand on a synthetic night."""

    def _validate(self, recording_dir, protocol, output_dir):
        pilot1_path, pilot3_path = output_dir / "pilot1.json", output_dir / "pilot3.json"
        pilot1 = _run_cli(
            "validate-pilot1-recording", str(recording_dir), "--min-duration-hours", "0.05",
            "--required-modality", "eeg", "--required-modality", "imu", "--output", str(pilot1_path),
        )
        pilot3 = _run_cli(
            "simulate-replay-cues", str(recording_dir),
            "--catalog", str(protocol["catalog"]), "--session", str(protocol["session"]),
            "--assignment", str(protocol["assignment"]), "--cue-library", str(protocol["cue_library"]),
            "--preset", "p21", "--output", str(pilot3_path),
        )
        self.assertEqual((pilot1[0], pilot3[0]), (0, 0), pilot1[2] + pilot3[2])
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in (pilot1_path, pilot3_path)]
        return json.loads(pilot1[1]), pilot3[1], reports

    def test_their_reports_say_synthetic_only_when_the_night_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = generate_night(output_dir=Path(tmp) / "night", scenario="clean_night", hours=0.05)
            protocol = _write_demo_protocol(Path(tmp) / "protocol", seed=0)
            pilot1_stdout, pilot3_stdout, reports = self._validate(night.recording_dir, protocol, Path(tmp) / "a")
            for payload in (pilot1_stdout, *reports):
                self.assertIs(payload["synthetic"], True)
                self.assertEqual(payload["note"], SYNTHETIC_NOTE)
            self.assertTrue(pilot3_stdout.rstrip().endswith(f"({SYNTHETIC_NOTE})"), pilot3_stdout)

            # The same night with a metadata.json that does not say synthetic, as a recording's.
            metadata_path = night.recording_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("synthetic", "note", "scenario", "seed"):
                metadata.pop(key)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            pilot1_stdout, pilot3_stdout, reports = self._validate(night.recording_dir, protocol, Path(tmp) / "b")
            for payload in (pilot1_stdout, *reports):
                self.assertNotIn("synthetic", payload)
                self.assertNotIn("note", payload)
            self.assertNotIn(SYNTHETIC_NOTE, pilot3_stdout)


class DemoCommandTest(unittest.TestCase):
    """The demo on a quarter hour of clean_night: wake, then sleep onset, no REM yet."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls._tmp.name) / "demo"
        with mock.patch("webbrowser.open") as browser:
            cls.code, cls.stdout, cls.stderr = _run_cli("demo", "--hours", "0.25", "--output", str(cls.output))
        cls.browser = browser
        cls.summary = json.loads((cls.output / "demo_summary.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_passes_every_check(self):
        self.assertEqual(self.code, 0, self.stderr)
        self.assertEqual(set(self.summary["checks"].values()), {True})
        self.assertTrue(self.summary["pilot1"]["passed"])
        pilot3 = self.summary["pilot3"]
        self.assertTrue(pilot3["passed"])
        self.assertEqual(pilot3["uncued_puzzle_play_count"], 0)
        self.assertEqual(pilot3["audio_backend"], "mock")
        self.assertFalse(pilot3["audio_playback_executed"])
        self.assertEqual((self.summary["scenario"], self.summary["seed"], self.summary["preset"]), ("clean_night", 0, "p21"))

    def test_writes_the_recording_next_to_its_reports(self):
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), [
            "agreement.json", "clean_night_seed0", "demo_summary.json", "pilot1.json",
            "pilot3_report.json", "protocol", "report.html",
        ])
        self.assertEqual(sorted(path.name for path in (self.output / "protocol").iterdir()), [
            "assignment.json", "cue_library.json", "night_puzzles.json", "puzzle_catalog.json",
        ])
        generation = json.loads((self.output / "clean_night_seed0" / "generation.json").read_text(encoding="utf-8"))
        self.assertEqual(self.summary["raw_sha256"], generation["files"]["raw_amused.bin"]["sha256"])

    def test_every_artefact_says_synthetic(self):
        self.assertIn(SYNTHETIC_NOTE, self.stdout)
        for path in sorted(self.output.glob("*.json")) + sorted((self.output / "protocol").glob("*.json")):
            payload = path.read_text(encoding="utf-8")
            self.assertIn(SYNTHETIC_NOTE, payload, path.name)
            self.assertNotIn("NaN", payload, path.name)
        page = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn(SYNTHETIC_NOTE, page)
        self.assertIn("<strong>Synthetic input, not sleep.</strong>", page)

    def test_cued_plays_come_at_sleep_onset_not_in_rem(self):
        # The p21 gate opens on the N1 epochs at sleep onset, so the cued plays of
        # clean_night are not evidence about REM gating.
        assignment = load_puzzle_cue_assignment(self.output / "protocol" / "assignment.json")
        plays = self.summary["cued_plays"]
        self.assertGreaterEqual(len(plays), 1)
        self.assertEqual(self.summary["pilot3"]["cue_plan_count"], len(plays))
        self.assertEqual(self.summary["cued_plays_in_generated_rem"], 0)
        for play in plays:
            self.assertIn(play["puzzle_id"], assignment.cued_puzzle_ids)
            self.assertEqual(play["generated_stage"], "N1")
        self.assertIn("in generated REM", self.stdout)

    def test_uncued_puzzles_keep_their_tones_but_never_play(self):
        assignment = load_puzzle_cue_assignment(self.output / "protocol" / "assignment.json")
        catalog = load_puzzle_catalog(self.output / "protocol" / "puzzle_catalog.json")
        library = load_cue_library(self.output / "protocol" / "cue_library.json")
        self.assertEqual(len(assignment.cued_puzzle_ids), 2)
        self.assertEqual(len(assignment.uncued_puzzle_ids), 2)
        for puzzle_id in assignment.uncued_puzzle_ids:
            self.assertEqual(library.by_id(catalog.get_puzzle(puzzle_id).cue_id).cue_type, "generated_tone")
        pilot3 = json.loads((self.output / "pilot3_report.json").read_text(encoding="utf-8"))
        self.assertEqual(pilot3["audio_backend"], "mock")
        played = {event["puzzle_id"] for event in pilot3["scheduler_events"] if event["event_type"] == "play"}
        self.assertTrue(played)
        self.assertTrue(played <= set(assignment.cued_puzzle_ids))

    def test_opens_the_report(self):
        self.browser.assert_called_once_with((self.output / "report.html").resolve().as_uri())

    def test_baseline_comparison(self):
        baseline = {
            "scenario": "clean_night",
            "seed": 0,
            "hours": 0.25,
            "raw_sha256": self.summary["raw_sha256"],
            "tolerance": 0.01,
            "agreement": dict(self.summary["agreement"]),
            "pilot3": {key: self.summary["pilot3"][key] for key in ("cue_plan_count", "uncued_puzzle_play_count")},
        }
        self.assertEqual(_demo_baseline_mismatches(baseline, self.summary), [])

        changed = json.loads(json.dumps(baseline))
        changed["agreement"]["gate_open_fraction"] += 0.005
        self.assertEqual(_demo_baseline_mismatches(changed, self.summary), [])
        changed["agreement"]["gate_open_fraction"] += 0.01
        changed["agreement"]["first_gate_open_latency_s"] = [60.0]
        changed["raw_sha256"] = "0" * 64
        changed["pilot3"]["cue_plan_count"] += 1
        mismatches = _demo_baseline_mismatches(changed, self.summary)
        self.assertEqual(len(mismatches), 4, mismatches)

    def test_a_baseline_for_another_night_stops_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps({
                "scenario": "clean_night", "seed": 0, "hours": 2.0, "agreement": {},
            }), encoding="utf-8")
            target = Path(tmp) / "demo"
            code, _, err = _run_cli(
                "demo", "--hours", "0.25", "--no-open", "--output", str(target), "--baseline", str(baseline_path),
            )
        self.assertEqual(code, 2)
        self.assertIn("the baseline is for clean_night, seed 0, 2.0 h", err)
        self.assertFalse(target.exists())

    def test_refuses_the_folder_of_a_real_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            recording = Path(tmp) / "night"
            recording.mkdir()
            (recording / "raw_amused.bin").write_bytes(b"recorded")
            (recording / "metadata.json").write_text(json.dumps({"source": {"source_name": "amused"}}), encoding="utf-8")
            code, _, err = _run_cli("demo", "--hours", "0.25", "--no-open", "--output", str(recording))
            self.assertEqual(code, 2)
            self.assertIn("choose a new or empty directory", err)
            self.assertEqual(sorted(path.name for path in recording.iterdir()), ["metadata.json", "raw_amused.bin"])


class DemoBranchTest(unittest.TestCase):
    """The demo's other ways out, each on three minutes of a night."""

    def _demo(self, output, *argv):
        code, out, err = _run_cli("demo", "--hours", "0.05", "--no-open", "--output", str(output), *argv)
        summary_path = output / "demo_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        return code, out, err, summary

    def test_a_failed_check_exits_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps({
                "scenario": "clean_night", "seed": 0, "hours": 0.05, "agreement": {},
                "pilot3": {"cue_plan_count": 99},
            }), encoding="utf-8")
            code, out, _, summary = self._demo(Path(tmp) / "demo", "--baseline", str(baseline_path))
        self.assertEqual(code, 1)
        self.assertIs(summary["checks"]["agreement_within_baseline"], False)
        self.assertEqual(summary["baseline"]["mismatches"], ["pilot3.cue_plan_count 0 != 99"])
        self.assertIn("demo failed: agreement_within_baseline", out)

    def test_stops_when_pilot3_did_not_run_on_mock_audio(self):
        # simulate_cue_plan_from_epochs always reports mock audio, so only a patched
        # report reaches this stop.
        real_report = cli_module._replay_cue_report

        async def sounddevice_report(args):
            return replace(await real_report(args), audio_backend="sounddevice")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo"
            with mock.patch("muse_tmr.cli.main._replay_cue_report", sounddevice_report):
                code, _, err, summary = self._demo(output)
            self.assertFalse((output / "pilot3_report.json").exists())
        self.assertEqual(code, 1)
        self.assertIn("'sounddevice'; the demo runs on mock audio only", err)
        self.assertIsNone(summary)

    def test_no_open_and_the_default_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("muse_tmr.cli.main._default_path_base", return_value=Path(tmp)), \
                    mock.patch("webbrowser.open") as browser:
                code, _, err = _run_cli("demo", "--hours", "0.05", "--no-open")
            demos = list((Path(tmp) / "data" / "synthetic").iterdir())
            self.assertEqual(code, 0, err)
            browser.assert_not_called()
            self.assertEqual(len(demos), 1)
            self.assertRegex(demos[0].name, r"^demo_\d{8}_\d{6}$")
            self.assertTrue((demos[0] / "clean_night_seed0" / "raw_amused.bin").is_file())
            self.assertTrue((demos[0] / "demo_summary.json").is_file())

    def test_an_optics_preset_needs_ppg_in_pilot1(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err, summary = self._demo(Path(tmp) / "demo", "--scenario", "optics_line")
            pilot1 = json.loads((Path(tmp) / "demo" / "pilot1.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertEqual(summary["preset"], "p1034")
        self.assertEqual(sorted(pilot1["coverage_targets"]), ["eeg", "imu", "ppg"])
        self.assertTrue(pilot1["coverage_targets"]["ppg"]["passed"])


class OutputFolderTest(unittest.TestCase):
    """synthesize-night and demo write into a new or empty folder or over an earlier run of their own."""

    SYNTHESIZE = ("synthesize-night", "--scenario", "short_allow_short", "--hours", "0.02")
    DEMO = ("demo", "--hours", "0.05", "--no-open")

    def test_refuse_a_polar_folder(self):
        # record-polar's <session>/polar/ has no raw_amused.bin, but a metadata.json,
        # summary.json and events.jsonl that a night would overwrite.
        with tempfile.TemporaryDirectory() as tmp:
            polar = Path(tmp) / "session" / "polar"
            polar.mkdir(parents=True)
            files = {
                "metadata.json": json.dumps({"source": "polar_h10"}),
                "summary.json": json.dumps({"duration_seconds": 28800.0}),
                "events.jsonl": json.dumps({"event": "recording_started"}) + "\n",
                "raw_notifications.jsonl": json.dumps({"kind": "ecg"}) + "\n",
            }
            for name, text in files.items():
                (polar / name).write_text(text, encoding="utf-8")
            for argv in (self.SYNTHESIZE, self.DEMO):
                code, _, err = _run_cli(*argv, "--output", str(polar))
                self.assertEqual(code, 2, argv[0])
                self.assertIn("choose a new or empty directory", err)
            self.assertEqual({path.name: path.read_text(encoding="utf-8") for path in polar.iterdir()}, files)

    def test_refuse_other_files_and_take_hidden_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            # demo --output data would otherwise replace data/protocol/puzzle_catalog.json.
            data = Path(tmp) / "data"
            (data / "protocol").mkdir(parents=True)
            catalog = data / "protocol" / "puzzle_catalog.json"
            catalog.write_text(json.dumps({"puzzles": []}), encoding="utf-8")
            for argv in (self.SYNTHESIZE, self.DEMO):
                self.assertEqual(_run_cli(*argv, "--output", str(data))[0], 2, argv[0])
            self.assertEqual(catalog.read_text(encoding="utf-8"), json.dumps({"puzzles": []}))
            self.assertEqual([path.name for path in data.iterdir()], ["protocol"])

            for argv in (self.SYNTHESIZE, self.DEMO):
                opened = Path(tmp) / argv[0]
                opened.mkdir()
                (opened / ".DS_Store").write_bytes(b"\0")
                code, _, err = _run_cli(*argv, "--output", str(opened))
                self.assertEqual(code, 0, err)

    def test_demo_runs_again_over_an_earlier_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo"
            for _ in range(2):
                code, _, err = _run_cli(*self.DEMO, "--output", str(output))
                self.assertEqual(code, 0, err)

    def test_demo_refuses_a_recording_where_its_night_goes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo"
            recording = output / "clean_night_seed0"
            recording.mkdir(parents=True)
            (output / "demo_summary.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
            (recording / "raw_amused.bin").write_bytes(b"recorded")
            (recording / "metadata.json").write_text(json.dumps({"source": {"source_name": "amused"}}), encoding="utf-8")
            code, _, err = _run_cli(*self.DEMO, "--output", str(output))
            self.assertEqual(code, 2)
            self.assertIn(f"{recording} is not empty", err)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["clean_night_seed0", "demo_summary.json"])
            self.assertEqual(sorted(path.name for path in recording.iterdir()), ["metadata.json", "raw_amused.bin"])


class AgreementBaselineFixtureTest(unittest.TestCase):
    """The baseline the CI end_to_end job compares `muse-tmr demo --hours 2` with."""

    def test_fits_the_ci_run(self):
        baseline = _load_demo_baseline(BASELINE_PATH)
        self.assertIs(baseline["synthetic"], True)
        self.assertEqual(baseline["note"], SYNTHETIC_NOTE)
        self.assertIsNone(_demo_baseline_identity(baseline, scenario_config("clean_night", hours=2.0)))
        self.assertEqual(len(baseline["raw_sha256"]), 64)
        self.assertEqual(
            set(baseline["agreement"]),
            {"kappa", "recall", "precision", "gate_open_fraction", "first_gate_open_latency_s"},
        )
        self.assertEqual(baseline["pilot3"]["uncued_puzzle_play_count"], 0)
        self.assertGreaterEqual(baseline["pilot3"]["cue_plan_count"], 1)


if __name__ == "__main__":
    unittest.main()
