"""Continuous arrays from MuseFrames (#230): muse_tmr.data.continuous.

Frames are built by hand, or replayed from a labelled synthetic night (synthetic input,
not sleep) generated into a temporary directory.
"""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

import numpy as np

from muse_tmr.data.continuous import (
    DROPOUT,
    LOST_SAMPLES,
    ContinuousBuilder,
    ContinuousConfig,
    reconstruct_continuous,
    reconstruct_recording,
)
from muse_tmr.data.sample_types import EEGSample, IMUSample, MuseFrame, PPGSample
from tests.exchange_synthetic import frames as replay_frames
from tests.exchange_synthetic import labelled_night

REPO_ROOT = Path(__file__).resolve().parents[1]
T0 = 1_768_431_600.0
RATE = 256.0
CHANNELS = ("TP9", "AF7", "AF8", "TP10")
NEW_DATA_MODULES = (
    "muse_tmr.data.continuous",
    "muse_tmr.data.night_events",
    "muse_tmr.data.edf_io",
    "muse_tmr.data.bids_export",
    "muse_tmr.data.mne_export",
    "muse_tmr.data.import_recording",
    "muse_tmr.data.replay",
    "muse_tmr.data.boas",
)
FORBIDDEN_PACKAGES = ("muse_tmr.audio", "muse_tmr.protocol", "muse_tmr.models")


def eeg_frame(ts, start, width=12, gaps=(), channels=CHANNELS):
    """A frame whose sample k on channel c is start + k + 1000 * c, so order shows."""
    samples = {name: [float(start + k + 1000 * c) for k in range(width)] for c, name in enumerate(channels)}
    return MuseFrame(timestamp=ts, eeg=EEGSample(ts, samples, gaps=gaps))


def steady(count, *, width=12, t=T0, start=0):
    return [eeg_frame(t + i * width / RATE, start + i * width, width) for i in range(count)]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_collect(frames):
    """_collect() of scripts/validate_rem_detection.py as it was before #230, kept here
    to show the refactor changed nothing the reference stager gets."""
    names = ("AF7", "AF8", "TP9", "TP10")
    channels = {name: array("f") for name in names}
    state = {"t0": None, "t_last": None, "n": 0, "prev_ts": None, "prev_width": 0}
    for frame in frames:
        eeg = getattr(frame, "eeg", None)
        if eeg is None:
            continue
        per_channel = eeg.channels_uv or {}
        reference = per_channel.get("AF7")
        if not reference:
            continue
        ts = frame.timestamp
        width = len(reference)
        if state["t0"] is None:
            state["t0"] = ts
        elif ts - state["prev_ts"] > 0.5:
            n_fill = int(round((ts - state["prev_ts"]) * 256.0)) - int(state["prev_width"])
            if n_fill > 0:
                zeros = [0.0] * n_fill
                for name in names:
                    channels[name].extend(zeros)
                state["n"] += n_fill
        state["prev_ts"] = ts
        state["prev_width"] = width
        state["t_last"] = ts
        state["n"] += width
        for name in names:
            samples = per_channel.get(name) or []
            if len(samples) == width:
                channels[name].extend(samples)
            else:
                channels[name].extend(list(samples)[:width] + [0.0] * max(0, width - len(samples)))
    n = int(state["n"])
    return {
        "channels": {name: np.frombuffer(buf, dtype=np.float32).copy() for name, buf in channels.items()},
        "sfreq": float(n / (state["t_last"] - state["t0"])),
        "t0": float(state["t0"]),
        "n_samples": n,
    }


class TestReconstructContinuous(unittest.TestCase):
    def test_steady_frames_lie_end_to_end(self):
        frames = steady(100)
        recording = reconstruct_continuous(frames)

        eeg = recording.eeg
        self.assertEqual(eeg.channel_names, CHANNELS)
        self.assertEqual(eeg.channels["AF7"].dtype, np.float32)
        np.testing.assert_array_equal(eeg.channels["TP9"], np.arange(1200, dtype=np.float32))
        np.testing.assert_array_equal(eeg.channels["AF7"], np.arange(1200, dtype=np.float32) + 1000)
        self.assertEqual(eeg.n_samples, 1200)
        self.assertEqual(recording.t0, T0)
        self.assertEqual(len(recording.segments), 1)
        self.assertEqual(recording.gap_markers, ())
        self.assertAlmostEqual(recording.measured_rate_hz, 1200 / (99 * 12 / RATE))

    def test_zero_fill_puts_a_dropout_on_host_time(self):
        frames = steady(10) + [eeg_frame(T0 + 9 * 12 / RATE + 1.0, 120)]
        recording = reconstruct_continuous(frames)

        (marker,) = recording.gap_markers
        self.assertEqual(marker.kind, DROPOUT)
        self.assertEqual((marker.index, marker.samples, marker.filled), (120, 244, True))
        self.assertAlmostEqual(marker.seconds, 1.0 - 12 / RATE)
        values = recording.eeg.channels["TP9"]
        self.assertEqual(values.size, 120 + 244 + 12)
        self.assertTrue(np.all(values[120:364] == 0))
        self.assertEqual(values[364], 120.0)
        self.assertEqual(len(recording.segments), 1)

    def test_discontinuous_starts_a_segment_and_keeps_the_rate(self):
        frames = steady(10) + [eeg_frame(T0 + 9 * 12 / RATE + 1.0, 120)]
        filled = reconstruct_continuous(frames)
        joined = reconstruct_continuous(frames, ContinuousConfig(gap_policy="discontinuous"))

        self.assertEqual(joined.eeg.n_samples, 132)
        self.assertEqual([(s.start_index, s.stop_index) for s in joined.segments], [(0, 120), (120, 132)])
        (marker,) = joined.gap_markers
        self.assertFalse(marker.filled)
        self.assertEqual(marker.samples, 244)
        self.assertAlmostEqual(joined.measured_rate_hz, filled.measured_rate_hz)

    def test_lost_samples_are_zeros_at_their_place_or_joined(self):
        frames = steady(2) + [eeg_frame(T0 + 2 * 12 / RATE, 24, gaps=((4, 8),))]
        filled = reconstruct_continuous(frames)
        joined = reconstruct_continuous(frames, ContinuousConfig(fill_lost_samples=False))

        values = filled.eeg.channels["TP9"]
        np.testing.assert_array_equal(values[24:28], [24, 25, 26, 27])
        self.assertTrue(np.all(values[28:36] == 0))
        np.testing.assert_array_equal(values[36:44], np.arange(28, 36))
        (marker,) = filled.gap_markers
        self.assertEqual((marker.kind, marker.index, marker.samples, marker.filled), (LOST_SAMPLES, 28, 8, True))
        np.testing.assert_array_equal(joined.eeg.channels["TP9"], np.arange(36, dtype=np.float32))
        self.assertFalse(joined.gap_markers[0].filled)

    def test_a_dropout_takes_the_counter_count_when_it_agrees_with_host_time(self):
        after = T0 + 9 * 12 / RATE + 1.0  # host time says 244 samples went missing

        def rebuilt(gaps, **config):
            frames = steady(10) + [eeg_frame(after, 120, gaps=gaps)]
            return reconstruct_continuous(frames, ContinuousConfig(**config)).gap_markers

        self.assertEqual(rebuilt(((0, 248),))[0].samples, 248)
        # A counter that wrapped while the stream was away is off by whole wraps.
        self.assertEqual(rebuilt(((0, 248 + 1024),))[0].samples, 244)
        # The REM check reads the night as it always has: host time only.
        self.assertEqual(rebuilt(((0, 248),), fill_lost_samples=False)[0].samples, 244)
        # The counter's skip is the dropout's, not a second loss.
        self.assertEqual([marker.kind for marker in rebuilt(((0, 248),))], [DROPOUT])

    def test_frames_need_the_reference_channel_and_others_are_padded_or_cut(self):
        no_af7 = MuseFrame(timestamp=T0 + 1, eeg=EEGSample(T0 + 1, {"TP9": [9.0] * 12}))
        short = MuseFrame(timestamp=T0 + 2, eeg=EEGSample(T0 + 2, {"AF7": [1.0] * 12, "AF8": [2.0] * 5}))
        long = MuseFrame(timestamp=T0 + 3, eeg=EEGSample(T0 + 3, {"AF7": [1.0] * 4, "AF8": [3.0] * 9}))
        config = ContinuousConfig(channels=("AF7", "AF8"), reference_channel="AF7", gap_threshold_seconds=5.0)

        recording = reconstruct_continuous([no_af7, short, long], config)

        self.assertEqual(recording.eeg.t0, T0 + 2)
        np.testing.assert_array_equal(recording.eeg.channels["AF8"], [2.0] * 5 + [0.0] * 7 + [3.0] * 4)
        self.assertEqual(recording.t0, T0 + 1)  # the first frame of any kind
        self.assertEqual(recording.offset_samples("eeg"), 256)

    def test_imu_and_optics_streams_go_on_their_own_rates(self):
        frames = []
        for i in range(8):
            ts = T0 + i * 0.25
            imu = IMUSample(ts, accelerometer_g=[{"x": i, "y": 0.0, "z": 1.0}] * 13, gyroscope_dps=[(1.0, 2.0, 3.0)] * 13)
            ppg = PPGSample(ts, {"LO_NIR": [float(i)] * 16})
            frames.append(MuseFrame(timestamp=ts, eeg=eeg_frame(ts, i * 64, 64).eeg, imu=imu, ppg=ppg))

        recording = reconstruct_continuous(frames, ContinuousConfig(streams=("eeg", "accel", "gyro", "optics")))

        self.assertEqual(set(recording.streams), {"eeg", "accel", "gyro", "optics"})
        self.assertEqual(recording.streams["accel"].n_samples, 104)
        self.assertEqual(recording.streams["accel"].nominal_rate_hz, 52.0)
        np.testing.assert_array_equal(recording.streams["gyro"].channels["GYRO_Z"], [3.0] * 104)
        self.assertEqual(recording.streams["optics"].channel_names, ("LO_NIR",))
        self.assertEqual(recording.streams["optics"].nominal_rate_hz, 64.0)

    def test_time_map_goes_both_ways(self):
        frames = steady(10) + [eeg_frame(T0 + 9 * 12 / RATE + 1.0, 120)] + steady(5, t=T0 + 9 * 12 / RATE + 1.0 + 12 / RATE, start=132)
        recording = reconstruct_continuous(frames)

        for host in (T0 - 0.5, T0, T0 + 0.2, T0 + 1.4, T0 + 5.0):
            self.assertAlmostEqual(float(recording.unix_time(recording.seconds(host))), host, places=9)
        # The frame after the dropout starts at the sample after the zeros.
        self.assertAlmostEqual(float(recording.seconds(T0 + 9 * 12 / RATE + 1.0)), 364 / RATE, places=9)

    def test_no_eeg_and_bad_configs_are_errors(self):
        with self.assertRaises(ValueError):
            reconstruct_continuous([MuseFrame(timestamp=T0, imu=IMUSample(T0, accelerometer_g=[(0, 0, 1)]))])
        for bad in (
            {"gap_policy": "edf_d"},
            {"gap_threshold_seconds": 0},
            {"nominal_rate_hz": -1},
            {"streams": ("eeg", "emg")},
            {"channels": ("AF7",), "reference_channel": "TP9"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ContinuousBuilder(ContinuousConfig(**bad))


class TestSyntheticNightReconstruction(unittest.TestCase):
    """A labelled night with BLE gaps, lost subpackets and repeats (synthetic input, not sleep)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.night = labelled_night(Path(cls._tmp.name) / "night", ble_gaps_per_hour=30.0)
        cls.frames = replay_frames(cls.night.recording_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_sample_the_device_made_has_its_place(self):
        config = ContinuousConfig(streams=("eeg", "accel", "gyro", "optics"))
        recording = asyncio.run(reconstruct_recording(self.night.recording_dir, config))
        ble = self.night.generation["ble"]
        self.assertGreaterEqual(len(ble["gaps"]), 1)

        # Lost subpackets are zeros at their place and each dropout takes the counter's
        # count, so the arrays hold exactly the samples the generator made.
        for stream, tag, rows in (("eeg", "EEG", 4), ("accel", "ACCGYRO", 3), ("optics", "OPTICS", 2)):
            with self.subTest(stream=stream):
                subpackets = ble["subpackets"][tag]
                built = recording.streams[stream]
                self.assertEqual(built.n_samples, subpackets["produced"] * rows)
                self.assertEqual(built.received_samples, subpackets["delivered"] * rows)
        self.assertEqual(len(recording.eeg.dropouts()), len(ble["gaps"]))
        self.assertTrue(all(marker.filled for marker in recording.eeg.dropouts()))

    def test_the_rem_check_gets_what_it_always_got(self):
        vrd = _load_script("validate_rem_detection")
        legacy = legacy_collect(self.frames)

        collected = vrd.collect(self.night.recording_dir, 30.0)

        self.assertEqual(collected["n_samples"], legacy["n_samples"])
        self.assertEqual(collected["sfreq"], legacy["sfreq"])
        self.assertEqual(collected["t0"], legacy["t0"])
        for name, values in legacy["channels"].items():
            np.testing.assert_array_equal(collected["channels"][name], values)
        self.assertGreater(legacy["n_samples"], 0)


class TestDataLayering(unittest.TestCase):
    """No module under muse_tmr.data reaches the audio, protocol or model packages (#230),
    the BOAS loader of #226 among them."""

    def test_each_new_data_module_imports_none_of_them(self):
        code = (
            "import importlib, json, sys\n"
            "importlib.import_module(sys.argv[1])\n"
            "print(json.dumps(sorted(name for name in sys.modules if name.startswith('muse_tmr.'))))\n"
        )
        for module in NEW_DATA_MODULES:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-c", code, module],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=REPO_ROOT,
                    env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                )
                loaded = json.loads(result.stdout)
                for package in FORBIDDEN_PACKAGES:
                    self.assertFalse(
                        [name for name in loaded if name == package or name.startswith(package + ".")],
                        f"{module} pulls in {package}",
                    )


if __name__ == "__main__":
    unittest.main()
