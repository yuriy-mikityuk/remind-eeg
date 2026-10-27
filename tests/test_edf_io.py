"""EDF+ export and import (#230): muse_tmr.data.edf_io, through the optional edfio.

Every EDF here is written in-test into a temporary directory, from a labelled synthetic
night (synthetic input, not sleep) or from made-up signals shaped like a BOAS headband
file (OpenNeuro ds005555); none is committed. Skipped without the ``[edf]`` extra.
"""

import asyncio
import datetime as dt
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muse_tmr.ble import muse_athena_protocol as proto
from muse_tmr.data.continuous import ContinuousConfig, reconstruct_continuous, reconstruct_recording
from muse_tmr.data.edf_io import (
    BOAS_HEADBAND_CHANNEL_MAP,
    GAP_TEXT,
    default_channel_map,
    edf_available,
    edf_channels,
    edf_header,
    export_config,
    export_edf,
    fit_header_number,
    read_edf,
    read_edf_hypnogram,
    resolve_channel_map,
    write_edf_plus,
)
from muse_tmr.data.night_events import SealedNightError, read_hypnogram
from muse_tmr.data.sample_types import EEGSample, MuseFrame
from tests.exchange_synthetic import labelled_night

ONE_COUNT_UV = proto.EEG_SCALE_UV_PER_COUNT
SKIP_REASON = 'needs the [edf] extra: pip install "remind-eeg[edf]"'

if edf_available():
    import edfio


def write_boas_like_edf(path, *, seconds=60, with_imu=True, eeg_unit="uV", seed=0):
    """An EDF laid out like a BOAS headband file: HB_1/HB_2 EEG, HB_IMU_1 to 6 and
    HB_PULSE, all at 256 Hz, plain EDF with an anonymised start date. Made-up values."""
    rng = np.random.default_rng(seed)
    n = int(seconds * 256)
    scale = {"uV": 1.0, "mV": 1e-3}[eeg_unit]
    eeg = {name: rng.normal(0.0, 20.0, n) for name in ("HB_1", "HB_2")}
    signals = [
        edfio.EdfSignal(eeg[name] * scale, 256, label=name, physical_dimension=eeg_unit,
                        physical_range=(-500 * scale, 500 * scale))
        for name in ("HB_1", "HB_2")
    ]
    imu = {}
    if with_imu:
        for index in range(1, 7):
            unit = "g" if index <= 3 else "deg/s"
            values = rng.normal(0.0, 0.05 if index <= 3 else 2.0, n) + (1.0 if index == 3 else 0.0)
            imu[f"HB_IMU_{index}"] = values
            limit = 4.0 if index <= 3 else 250.0
            signals.append(edfio.EdfSignal(values, 256, label=f"HB_IMU_{index}", physical_dimension=unit,
                                           physical_range=(-limit, limit)))
        pulse = 100.0 + 5.0 * np.sin(2 * np.pi * 1.1 * np.arange(n) / 256)
        imu["HB_PULSE"] = pulse
        signals.append(edfio.EdfSignal(pulse, 256, label="HB_PULSE", physical_dimension="nA",
                                       physical_range=(0.0, 200.0)))
    edfio.Edf(signals, starttime=dt.time(22, 30, 0)).write(path)
    return eeg, imu


@unittest.skipUnless(edf_available(), SKIP_REASON)
class TestSyntheticRoundTrip(unittest.TestCase):
    """Export a labelled night with BLE gaps and read it back (synthetic input, not sleep)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.night = labelled_night(root / "night", ble_gaps_per_hour=30.0)
        cls.edf_path = root / "night.edf"
        cls.result = export_edf(cls.night.recording_dir, cls.edf_path)
        cls.recording = asyncio.run(reconstruct_recording(cls.night.recording_dir, export_config()))
        cls.edf = edfio.read_edf(cls.edf_path, lazy_load_data=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_eeg_comes_back_within_a_count(self):
        eeg = self.recording.eeg
        offset = self.recording.offset_samples("eeg")
        for name, values in eeg.channels.items():
            with self.subTest(channel=name):
                data = self.edf.get_signal(name).data
                self.assertEqual(data.size, self.result.records * 256)
                back = data[offset:offset + values.size]
                self.assertLess(float(np.max(np.abs(back - values.astype(np.float64)))), ONE_COUNT_UV)
                self.assertLess(float(np.max(np.abs(data[offset + values.size:]), initial=0.0)), ONE_COUNT_UV)
        signal = self.edf.get_signal("AF7")
        self.assertEqual((signal.digital_min, signal.digital_max), (-8192, 8191))
        self.assertEqual(signal.physical_dimension, "uV")
        self.assertIn("vendor-stated, unverified", signal.transducer_type)
        eeg_infos = [info for info in self.result.signals if info.stream == "eeg"]
        self.assertTrue(all(info.lossless for info in eeg_infos))

    def test_imu_keeps_its_counts_and_optics_its_resolution(self):
        for stream in ("accel", "gyro", "optics"):
            built = self.recording.streams[stream]
            offset = self.recording.offset_samples(stream)
            infos = {info.label: info for info in self.result.signals if info.stream == stream}
            for name, values in built.channels.items():
                with self.subTest(channel=name):
                    data = self.edf.get_signal(name).data[offset:offset + values.size]
                    error = float(np.max(np.abs(data - values.astype(np.float64))))
                    self.assertLessEqual(error, infos[name].resolution)
                    self.assertEqual(infos[name].lossless, stream != "optics")

    def test_header_says_what_the_night_is(self):
        self.assertEqual(self.edf.reserved, "EDF+C")
        self.assertIn("synthetic input, not sleep", self.edf.local_patient_identification)
        self.assertIn("measured_rate=", self.edf.local_recording_identification)
        self.assertIn("gaps=zero_fill", self.edf.local_recording_identification)
        start = self.edf.startdatetime.replace(tzinfo=dt.timezone.utc).timestamp()
        self.assertAlmostEqual(start, self.recording.t0, places=5)
        self.assertTrue(edf_header(self.edf_path)["synthetic"])
        self.assertEqual(self.result.to_dict()["note"], "synthetic input, not sleep")

    def test_annotations_come_back_at_their_onsets(self):
        written = sorted(self.result.annotation_onsets)
        read = sorted((a.onset, a.duration or 0.0, a.text) for a in self.edf.annotations)
        self.assertEqual([text for _, _, text in written], [text for _, _, text in read])
        onset_error = max(abs(a[0] - b[0]) for a, b in zip(written, read))
        self.assertLess(onset_error, 0.001)
        texts = [text for _, _, text in read]
        self.assertIn("synthetic input, not sleep", texts)
        dropouts = self.recording.eeg.dropouts()
        self.assertGreaterEqual(len(dropouts), 1)
        gaps = [(onset, length) for onset, length, text in read if text == GAP_TEXT]
        for marker in dropouts:
            self.assertIn((round(marker.index / 256, 6), round(marker.samples / 256, 6)), gaps)

    def test_stages_land_on_their_epochs(self):
        stages = [(a.onset, a.duration, a.text) for a in self.edf.annotations if a.text.startswith("Sleep stage")]
        expected = [label for label in read_hypnogram(self.night.recording_dir / "hypnogram.csv")]
        self.assertEqual(len(stages), len(expected))
        for (onset, duration, text), interval in zip(stages, expected):
            self.assertAlmostEqual(onset, interval.onset_s, places=6)
            self.assertAlmostEqual(duration, interval.duration_s, places=6)
        self.assertEqual(
            [text for _, _, text in stages],
            ["Sleep stage W"] * 4 + ["Sleep stage N"] * 4 + ["Sleep stage R"] * 4 + ["Sleep stage N"] * 2,
        )
        self.assertEqual([i.label for i in read_edf_hypnogram(self.edf_path)],
                         ["wake"] * 4 + ["nrem"] * 4 + ["probable_rem"] * 4 + ["nrem"] * 2)

    def test_read_edf_gives_the_night_back(self):
        frames = list(read_edf(self.edf_path))
        again = reconstruct_continuous(frames, export_config())

        self.assertEqual(again.eeg.n_samples, self.recording.eeg.n_samples)
        for name, values in self.recording.eeg.channels.items():
            self.assertLess(float(np.max(np.abs(again.eeg.channels[name] - values))), ONE_COUNT_UV)
        # The dropouts the export filled are gaps again, in the same places.
        self.assertEqual(
            [(marker.index, marker.samples) for marker in again.eeg.dropouts()],
            [(marker.index, marker.samples) for marker in self.recording.eeg.dropouts()],
        )
        self.assertTrue(all(frame.imu is not None and frame.ppg is not None for frame in frames[:10]))

    def test_a_hypnogram_in_recording_seconds_lands_where_it_says(self):
        scored = Path(self._tmp.name) / "scored.tsv"
        scored.write_text("onset\tduration\tstage\n15\t30\tN3\n45\t30\tR\n")
        out = Path(self._tmp.name) / "scored.edf"

        result = export_edf(self.night.recording_dir, out, hypnograms=[scored])

        extra = [(onset, length, text) for onset, length, text in result.annotation_onsets
                 if text.startswith("Sleep stage") and onset in (15.0, 45.0)]
        self.assertEqual(extra, [(15.0, 30.0, "Sleep stage N"), (45.0, 30.0, "Sleep stage R")])


@unittest.skipUnless(edf_available(), SKIP_REASON)
class TestThirdPartyEdf(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_boas_headband_file_reads_through_its_channel_map(self):
        path = self.root / "sub-3_task-Sleep_acq-headband_eeg.edf"
        eeg, imu = write_boas_like_edf(path)

        frames = list(read_edf(path, channel_map=resolve_channel_map("boas-headband")))

        self.assertEqual(len(frames), 240)
        self.assertEqual(set(frames[0].eeg.channels_uv), {"AF7", "AF8"})
        start = dt.datetime(1985, 1, 1, 22, 30, tzinfo=dt.timezone.utc).timestamp()
        self.assertEqual(frames[0].timestamp, start)
        self.assertAlmostEqual(frames[1].timestamp - frames[0].timestamp, 0.25)
        af7 = np.concatenate([frame.eeg.channels_uv["AF7"] for frame in frames])
        self.assertLess(float(np.max(np.abs(af7 - eeg["HB_1"]))), 1000 / 65535 + 1e-9)
        accel_z = np.array([row["z"] for frame in frames for row in frame.imu.accelerometer_g])
        self.assertLess(float(np.max(np.abs(accel_z - imu["HB_IMU_3"]))), 8 / 65535 + 1e-9)
        self.assertEqual(len(frames[0].imu.gyroscope_dps), 64)
        self.assertEqual(set(frames[0].ppg.channels), {"HB_PULSE"})
        self.assertEqual({channel.target for channel in edf_channels(path, BOAS_HEADBAND_CHANNEL_MAP)},
                         {"eeg", "accel", "gyro", "ppg"})

    def test_rates_resamples_a_stream_and_a_night_without_imu_still_reads(self):
        path = self.root / "with_imu.edf"
        write_boas_like_edf(path)
        bare = self.root / "eeg_only.edf"
        write_boas_like_edf(bare, with_imu=False)
        boas = resolve_channel_map("boas-headband")

        frames = list(read_edf(path, channel_map=boas, rates={"accel": 52.0, "gyro": 52.0, "ppg": 64.0}))
        eeg_only = list(read_edf(bare, channel_map=boas))

        self.assertEqual(len(frames[1].imu.accelerometer_g), 13)
        self.assertEqual(len(frames[1].ppg.channels["HB_PULSE"]), 16)
        self.assertEqual(len(frames[1].eeg.channels_uv["AF7"]), 64)
        self.assertTrue(all(frame.imu is None and frame.ppg is None for frame in eeg_only))

    def test_eeg_units_are_converted_or_must_be_given(self):
        path = self.root / "mv.edf"
        eeg, _ = write_boas_like_edf(path, eeg_unit="mV", with_imu=False)
        boas = resolve_channel_map("boas-headband")

        af7 = np.concatenate([f.eeg.channels_uv["AF7"] for f in read_edf(path, channel_map=boas)])
        self.assertLess(float(np.max(np.abs(af7 - eeg["HB_1"]))), 0.02)

        odd = self.root / "odd.edf"
        edfio.Edf([edfio.EdfSignal(np.zeros(256), 256, label="HB_1", physical_dimension="counts",
                                   physical_range=(-1, 1))]).write(odd)
        with self.assertRaises(ValueError):
            list(read_edf(odd, channel_map=boas))
        frames = list(read_edf(odd, channel_map=boas, eeg_unit_scale=0.5))
        self.assertEqual(len(frames), 4)

    def test_edf_plus_d_and_unmapped_files_are_refused(self):
        path = self.root / "plain.edf"
        write_boas_like_edf(path, with_imu=False)
        with self.assertRaises(ValueError):
            list(read_edf(path))  # no label of an export, and no map

        exported = self.root / "c.edf"
        frames = [MuseFrame(timestamp=1.0e9 + i * 12 / 256, eeg=EEGSample(1.0e9 + i * 12 / 256, {"AF7": [0.0] * 12}))
                  for i in range(100)]
        write_edf_plus(reconstruct_continuous(frames), exported)
        raw = bytearray(exported.read_bytes())
        self.assertEqual(bytes(raw[192:197]), b"EDF+C")
        raw[192:197] = b"EDF+D"
        discontinuous = self.root / "d.edf"
        discontinuous.write_bytes(bytes(raw))
        with self.assertRaises(ValueError) as caught:
            list(read_edf(discontinuous))
        self.assertIn("EDF+D", str(caught.exception))


@unittest.skipUnless(edf_available(), SKIP_REASON)
class TestExportRules(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_sealed_night_is_refused_before_anything_is_read(self):
        night = self.root / "sealed"
        night.mkdir()
        (night / "metadata.json").write_text(json.dumps({
            "started_at": "2026-01-14T23:00:00+00:00",
            "night_arm": {"arm": "no_cue", "night_id": "n2", "sealed_schedule": True, "schedule_id": "s1",
                          "schedule_commitment_sha256": "ab" * 32, "schedule_unsealed_at_utc": None},
        }))
        with self.assertRaises(SealedNightError):
            export_edf(night, self.root / "sealed.edf")
        self.assertFalse((self.root / "sealed.edf").exists())

    def test_edf_plus_d_is_not_written(self):
        frames = [MuseFrame(timestamp=1.0e9 + i * 0.05, eeg=EEGSample(1.0e9 + i * 0.05, {"AF7": [1.0] * 12}))
                  for i in range(20)]
        recording = reconstruct_continuous(frames, ContinuousConfig(gap_policy="discontinuous"))
        with self.assertRaises(ValueError):
            write_edf_plus(recording, self.root / "x.edf")

    def test_a_signal_off_the_count_grid_is_quantised_over_its_range(self):
        frames = [MuseFrame(timestamp=1.0e9 + i * 12 / 256,
                            eeg=EEGSample(1.0e9 + i * 12 / 256, {"AF7": [0.001 * (i * 12 + k) for k in range(12)]}))
                  for i in range(50)]
        result = write_edf_plus(reconstruct_continuous(frames), self.root / "q.edf")
        info, = result.signals
        self.assertFalse(info.lossless)
        data = edfio.read_edf(self.root / "q.edf", lazy_load_data=False).get_signal("AF7").data[:600]
        self.assertLess(float(np.max(np.abs(data - 0.001 * np.arange(600)))), info.resolution)


class TestHelpers(unittest.TestCase):
    def test_header_numbers_fit_eight_characters(self):
        self.assertEqual(fit_header_number(-8192 * ONE_COUNT_UV), -725.044)
        self.assertEqual(fit_header_number(8191 * ONE_COUNT_UV), 724.9557)
        self.assertEqual(fit_header_number(2.0000014, "floor"), 2.000001)
        self.assertEqual(fit_header_number(-32768 * proto.ACC_SCALE), -2.0)
        self.assertEqual(fit_header_number(1.99993896, "ceil"), 1.999939)
        self.assertEqual(fit_header_number(0.0), 0.0)
        with self.assertRaises(ValueError):
            fit_header_number(math.inf)

    def test_channel_maps(self):
        self.assertEqual(
            default_channel_map(["TP9", "ACC_X", "GYRO_Z", "LO_NIR", "opt3", "EDF Annotations", "HB_1"]),
            {"TP9": "eeg:TP9", "ACC_X": "accel:x", "GYRO_Z": "gyro:z", "LO_NIR": "ppg:LO_NIR", "opt3": "ppg:opt3"},
        )
        self.assertEqual(resolve_channel_map("HB_1=AF7, HB_IMU_1=accel:x,X=skip"),
                         {"HB_1": "AF7", "HB_IMU_1": "accel:x", "X": "skip"})
        self.assertEqual(resolve_channel_map("boas-headband"), BOAS_HEADBAND_CHANNEL_MAP)
        self.assertIsNone(resolve_channel_map(None))
        for bad in ("HB_1", "HB_1=emg:x", ","):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_channel_map(bad)


if __name__ == "__main__":
    unittest.main()
