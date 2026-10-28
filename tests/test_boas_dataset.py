"""The BOAS loader, muse_tmr.data.boas (#226), on BOAS-shaped layouts written in-test.

Nothing of OpenNeuro ds005555 is read or copied: tests.boas_synthetic writes the layout
its README and sidecars describe, with synthetic or made-up signals (synthetic input,
not sleep). The headband EDF tests need the [edf] extra and are skipped without it.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

from muse_tmr.data.boas import (
    BOAS_DATASET_VERSION,
    BOAS_DOI,
    DETECTOR_MODALITIES,
    BoasHeadbandSession,
    boas_channel_map,
    boas_night,
    dataset_doi,
    dataset_version,
    is_boas_night,
    is_synthetic_dataset,
    iter_boas_nights,
    load_boas_hypnogram,
    stage_counts,
    stage_name,
)
from muse_tmr.data.edf_io import edf_available
from tests.boas_synthetic import NightSpec, write_layout

SKIP_REASON = 'needs the [edf] extra: pip install "remind-eeg[edf]"'
STAGES = ("W", "N1", "N2", "N3", "R", "R", "N2", "W")


def _events(path):
    return path.read_text(encoding="utf-8").splitlines()


class TemporaryLayout(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "ds"
        self.nights = (
            NightSpec("sub-2", "7", stages=STAGES, unscored=(7,), stage_ai_changes={5: -2}),
            NightSpec("sub-10", "7", stages=STAGES),
            NightSpec("sub-3", "8", stages=STAGES),
        )
        write_layout(self.root, self.nights, edf=False)

    def tearDown(self):
        self._tmp.cleanup()


class DatasetTest(TemporaryLayout):
    def test_nights_come_in_recording_order_with_their_participant(self):
        nights = list(iter_boas_nights(self.root))

        self.assertEqual([night.recording_id for night in nights], ["sub-2", "sub-3", "sub-10"])
        self.assertEqual([night.pid for night in nights], ["7", "8", "7"])
        self.assertEqual(nights[0].night_dir, self.root / "sub-2")
        self.assertEqual(nights[0].headband_edf.name, "sub-2_task-Sleep_acq-headband_eeg.edf")
        self.assertEqual(nights[0].missing_files(), ("sub-2_task-Sleep_acq-headband_eeg.edf",))
        self.assertEqual(nights[0].headband_channel_names(), ("HB_1", "HB_2"))
        self.assertFalse(nights[0].has_imu)
        self.assertFalse(nights[0].has_pulse)

    def test_the_dataset_description_gives_version_doi_and_the_synthetic_flag(self):
        self.assertEqual(dataset_version(self.root), BOAS_DATASET_VERSION)
        self.assertEqual(dataset_doi(self.root), BOAS_DOI)
        self.assertTrue(is_synthetic_dataset(self.root))
        self.assertIsNone(dataset_version(self.root / "sub-2"))

    def test_a_subject_directory_participants_tsv_does_not_list_is_refused(self):
        (self.root / "sub-99" / "eeg").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "sub-99"):
            list(iter_boas_nights(self.root))

    def test_without_participants_tsv_nothing_is_listed(self):
        (self.root / "participants.tsv").unlink()
        with self.assertRaises(FileNotFoundError):
            list(iter_boas_nights(self.root))

    def test_a_night_is_found_from_its_directory_its_eeg_directory_or_its_edf(self):
        edf = self.root / "sub-3" / "eeg" / "sub-3_task-Sleep_acq-headband_eeg.edf"
        self.assertFalse(is_boas_night(self.root / "sub-3"))
        edf.write_bytes(b"")
        for path in (self.root / "sub-3", self.root / "sub-3" / "eeg", edf):
            with self.subTest(path=path.name):
                night = boas_night(path)
                self.assertEqual((night.recording_id, night.pid), ("sub-3", "8"))
                self.assertTrue(is_boas_night(path))
        with self.assertRaises(ValueError):
            boas_night(self.root)
        self.assertFalse(is_boas_night(self.root))


class HypnogramTest(TemporaryLayout):
    def test_stage_hum_maps_to_the_rem_annotation_labels(self):
        labels = load_boas_hypnogram(self.root / "sub-2")

        self.assertEqual(len(labels), len(STAGES))
        self.assertEqual(
            [row.label for row in labels],
            ["wake", "nrem", "nrem", "nrem", "probable_rem", "probable_rem", "nrem", "unknown"],
        )
        self.assertEqual([row.stage_hum for row in labels], [0, 1, 2, 3, 4, 4, 2, 8])
        self.assertEqual([row.onset_s for row in labels], [30.0 * k for k in range(len(STAGES))])
        # stage_ai of the headband file is kept as reference_stage_ai, the PSG's beside it.
        self.assertEqual(labels[5].reference_stage_ai, -2)
        self.assertEqual(labels[5].psg_stage_ai, 4)
        self.assertEqual(labels[7].end_s, 240.0)
        self.assertEqual(stage_counts(labels), {"W": 1, "N1": 1, "N2": 2, "N3": 1, "R": 2, "disconnection": 1})
        self.assertEqual([stage_name(code) for code in (0, 4, 8, -2, None)], ["W", "R", "disconnection", "artefact", "n/a"])

    def test_a_missing_stage_is_unknown(self):
        path = self.root / "sub-3" / "eeg" / "sub-3_task-Sleep_acq-psg_events.tsv"
        lines = _events(path)
        fields = lines[2].split("\t")
        fields[5] = "n/a"
        lines[2] = "\t".join(fields)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        labels = load_boas_hypnogram(self.root / "sub-3")

        self.assertIsNone(labels[1].stage_hum)
        self.assertEqual(labels[1].label, "unknown")

    def test_headband_and_psg_epochs_must_be_the_same(self):
        path = self.root / "sub-3" / "eeg" / "sub-3_task-Sleep_acq-headband_events.tsv"
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines()
        fields = lines[3].split("\t")
        fields[0] = "61"
        path.write_text("\n".join(lines[:3] + ["\t".join(fields)] + lines[4:]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "headband events"):
            load_boas_hypnogram(self.root / "sub-3")

        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "epochs"):
            load_boas_hypnogram(self.root / "sub-3")

    def test_epochs_must_start_at_zero_and_follow_one_another(self):
        path = self.root / "sub-3" / "eeg" / "sub-3_task-Sleep_acq-psg_events.tsv"
        lines = _events(path)
        del lines[3]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.root / "sub-3" / "eeg" / "sub-3_task-Sleep_acq-headband_events.tsv").unlink()
        with self.assertRaisesRegex(ValueError, "follow one another"):
            load_boas_hypnogram(self.root / "sub-3")

    def test_epochs_must_be_30_s(self):
        for name in ("psg", "headband"):
            path = self.root / "sub-3" / "eeg" / f"sub-3_task-Sleep_acq-{name}_events.tsv"
            lines = _events(path)
            last = lines[-1].split("\t")
            last[1] = "20"
            path.write_text("\n".join(lines[:-1] + ["\t".join(last)]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lasts 20.0 s, not 30 s"):
            load_boas_hypnogram(self.root / "sub-3")

    def test_the_psg_events_need_stage_hum(self):
        path = self.root / "sub-3" / "eeg" / "sub-3_task-Sleep_acq-psg_events.tsv"
        path.write_text(path.read_text(encoding="utf-8").replace("stage_hum", "stage_x"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stage_hum"):
            load_boas_hypnogram(self.root / "sub-3")


class ChannelMapTest(unittest.TestCase):
    def test_modalities_pick_the_channels(self):
        self.assertEqual(boas_channel_map(DETECTOR_MODALITIES), {"HB_1": "eeg:AF7", "HB_2": "eeg:AF8"})
        everything = boas_channel_map()
        self.assertEqual(everything["HB_PULSE"], "ppg:HB_PULSE")
        self.assertEqual(everything["HB_IMU_4"], "gyro:x")
        self.assertEqual(len(everything), 9)
        with self.assertRaises(ValueError):
            boas_channel_map(("imu",))
        with self.assertRaises(ValueError):
            boas_channel_map(("eeg", "optics"))


@unittest.skipUnless(edf_available(), SKIP_REASON)
class HeadbandSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_edf_io import write_boas_like_edf

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name) / "ds"
        specs = (NightSpec("sub-1", "1", stages=STAGES, with_imu=True), NightSpec("sub-2", "2", stages=STAGES))
        write_layout(root, specs, edf=False)
        for spec in specs:
            write_boas_like_edf(
                root / spec.recording_id / "eeg" / f"{spec.recording_id}_task-Sleep_acq-headband_eeg.edf",
                seconds=12, with_imu=spec.with_imu,
            )
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _frames(self, recording_id, **kwargs):
        session = BoasHeadbandSession(self.root / recording_id, **kwargs)

        async def collect():
            metadata = await session.connect()
            return metadata, [frame async for frame in session.stream()]

        return session, *asyncio.run(collect())

    def test_frames_carry_af7_af8_imu_and_the_pulse(self):
        session, metadata, frames = self._frames("sub-1")

        self.assertEqual(len(frames), 48)  # 12 s in frames of 0.25 s
        self.assertTrue(all(frame.source == "boas" for frame in frames))
        self.assertEqual(set(frames[0].eeg.channels_uv), {"AF7", "AF8"})
        self.assertEqual(len(frames[0].eeg.channels_uv["AF7"]), 64)
        self.assertEqual(sum(len(frame.imu.accelerometer_g) for frame in frames), 12 * 52)
        self.assertEqual(set(frames[0].ppg.channels), {"HB_PULSE"})
        self.assertEqual(sum(len(frame.ppg.channels["HB_PULSE"]) for frame in frames), 12 * 64)
        self.assertEqual(metadata.source_name, "boas")
        self.assertEqual(metadata.device_id, "sub-1")
        self.assertTrue(metadata.capabilities["imu"] and metadata.capabilities["ppg"])
        self.assertFalse(metadata.capabilities["raw_packets"])
        device = asyncio.run(session.discover())[0]
        self.assertEqual(device.metadata["dataset"], "ds005555")

    def test_the_detector_modalities_are_the_eeg_alone(self):
        _, metadata, frames = self._frames("sub-1", modalities=DETECTOR_MODALITIES)

        self.assertTrue(all(frame.imu is None and frame.ppg is None for frame in frames))
        self.assertFalse(metadata.capabilities["imu"])
        self.assertEqual(frames[0].timestamp, frames[0].eeg.timestamp)

    def test_a_night_without_imu_gives_eeg_frames(self):
        _, metadata, frames = self._frames("sub-2")

        self.assertTrue(all(frame.imu is None and frame.ppg is None for frame in frames))
        self.assertFalse(metadata.capabilities["imu"])

    def test_stop_ends_the_stream(self):
        session = BoasHeadbandSession(self.root / "sub-1")

        async def first_two():
            frames = []
            async for frame in session.stream():
                frames.append(frame)
                if len(frames) == 2:
                    await session.stop()
            return frames

        self.assertEqual(len(asyncio.run(first_two())), 2)

    def test_a_missing_edf_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            write_layout(root, (NightSpec("sub-5", "5", stages=STAGES),), edf=False)
            session = BoasHeadbandSession(root / "sub-5" / "eeg")
            with self.assertRaisesRegex(FileNotFoundError, "sub-5_task-Sleep_acq-headband_eeg.edf"):
                asyncio.run(session.connect())


if __name__ == "__main__":
    unittest.main()
