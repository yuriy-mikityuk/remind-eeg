"""EEG-BIDS export (#230): muse_tmr.data.bids_export.

The dataset is written in-test from a labelled synthetic night (synthetic input, not
sleep) into a temporary directory. CI runs bids-validator on one in the ``bids`` job.
Skipped without the ``[edf]`` extra.
"""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from muse_tmr.data.bids_export import EEG_REFERENCE, export_bids
from muse_tmr.data.edf_io import edf_available, resolve_channel_map
from muse_tmr.data.import_recording import import_recording
from muse_tmr.data.night_events import SealedNightError, read_hypnogram, recorded_with_muse
from tests.exchange_synthetic import labelled_night
from tests.test_edf_io import write_boas_like_edf

SKIP_REASON = 'needs the [edf] extra: pip install "remind-eeg[edf]"'


def _tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@unittest.skipUnless(edf_available(), SKIP_REASON)
class TestBidsExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.night = labelled_night(root / "night")
        cls.root = root / "bids"
        cls.result = export_bids(cls.night.recording_dir, cls.root, subject="p01")
        cls.eeg_dir = cls.root / "sub-p01" / "ses-20260114" / "eeg"
        cls.stem = "sub-p01_ses-20260114_task-sleep"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_session_has_its_files(self):
        expected = {
            "README.md",
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
            f"sub-p01/ses-20260114/eeg/{self.stem}_eeg.edf",
            f"sub-p01/ses-20260114/eeg/{self.stem}_eeg.json",
            f"sub-p01/ses-20260114/eeg/{self.stem}_channels.tsv",
            f"sub-p01/ses-20260114/eeg/{self.stem}_events.tsv",
            f"sub-p01/ses-20260114/eeg/{self.stem}_events.json",
            f"sub-p01/ses-20260114/eeg/{self.stem}_acq-scoring_events.tsv",
            f"sub-p01/ses-20260114/eeg/{self.stem}_acq-scoring_events.json",
        }
        written = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(written, expected)
        self.assertEqual(set(self.result.files), expected)
        self.assertEqual((self.result.subject, self.result.session), ("p01", "20260114"))

    def test_sidecar_and_description_say_what_the_data_are(self):
        sidecar = json.loads((self.eeg_dir / f"{self.stem}_eeg.json").read_text())
        self.assertEqual(sidecar["EEGReference"], "Fpz (vendor-stated, unverified)")
        self.assertEqual(sidecar["EEGReference"], EEG_REFERENCE)
        for key, value in (("TaskName", "sleep"), ("SamplingFrequency", 256.0), ("PowerLineFrequency", "n/a"),
                           ("SoftwareFilters", "n/a"), ("EEGChannelCount", 4), ("MISCChannelCount", 8)):
            self.assertEqual(sidecar[key], value)
        self.assertIn("1450 uV full scale is vendor-stated and unverified", sidecar["EEGFullScale"])
        self.assertIn("synthetic input, not sleep", sidecar["TaskDescription"])
        self.assertEqual(sidecar["Notes"], ["synthetic input, not sleep"])
        self.assertGreater(sidecar["MeasuredSamplingFrequency"], 255.0)

        description = json.loads((self.root / "dataset_description.json").read_text())
        self.assertIn("synthetic input, not sleep", description["Name"])
        self.assertEqual(description["DatasetType"], "raw")
        self.assertNotIn("HEDVersion", description)
        self.assertIn("Synthetic input, not sleep", (self.root / "README.md").read_text())
        self.assertEqual(_tsv(self.root / "participants.tsv"), [{"participant_id": "sub-p01"}])

    def test_channels_carry_types_and_units(self):
        rows = {row["name"]: row for row in _tsv(self.eeg_dir / f"{self.stem}_channels.tsv")}
        self.assertEqual({rows[name]["type"] for name in ("TP9", "AF7", "AF8", "TP10")}, {"EEG"})
        self.assertEqual((rows["AF7"]["units"], rows["AF7"]["sampling_frequency"]), ("uV", "256"))
        self.assertEqual((rows["ACC_X"]["type"], rows["ACC_X"]["units"]), ("ACCEL", "g"))
        self.assertEqual((rows["GYRO_Z"]["type"], rows["GYRO_Z"]["units"]), ("GYRO", "deg/s"))
        self.assertEqual((rows["LO_NIR"]["type"], rows["LO_NIR"]["units"]), ("MISC", "n/a"))

    def test_events_and_stages_go_to_two_files(self):
        events = _tsv(self.eeg_dir / f"{self.stem}_events.tsv")
        stages = _tsv(self.eeg_dir / f"{self.stem}_acq-scoring_events.tsv")

        self.assertFalse([row for row in events if row["description"].startswith("Sleep stage")])
        self.assertIn("note", {row["trial_type"] for row in events})
        self.assertEqual([row["stage"] for row in stages],
                         ["Sleep stage W"] * 4 + ["Sleep stage N"] * 4 + ["Sleep stage R"] * 4 + ["Sleep stage N"] * 2)
        self.assertEqual([float(row["onset"]) for row in stages], [30.0 * i for i in range(14)])
        self.assertEqual([i.label for i in read_hypnogram(self.eeg_dir / f"{self.stem}_acq-scoring_events.tsv")],
                         [row["label"] for row in stages])
        levels = json.loads((self.eeg_dir / f"{self.stem}_acq-scoring_events.json").read_text())["stage"]["Levels"]
        self.assertEqual(set(levels), {"Sleep stage W", "Sleep stage N", "Sleep stage R", "Sleep stage ?"})

    def test_a_second_night_joins_the_dataset_and_a_taken_session_is_refused(self):
        root = Path(self._tmp.name) / "two"
        shutil.copytree(self.root, root)
        export_bids(self.night.recording_dir, root, subject="p02")
        self.assertEqual([row["participant_id"] for row in _tsv(root / "participants.tsv")], ["sub-p01", "sub-p02"])
        with self.assertRaises(ValueError):
            export_bids(self.night.recording_dir, root, subject="p02")
        with self.assertRaises(ValueError):
            export_bids(self.night.recording_dir, root, subject="p-03")

    def test_synthetic_and_recorded_nights_never_share_a_root(self):
        root = Path(self._tmp.name) / "recorded"
        root.mkdir()
        (root / "dataset_description.json").write_text(json.dumps({"Name": "REMind nights", "BIDSVersion": "1.10.0"}))
        with self.assertRaises(ValueError):
            export_bids(self.night.recording_dir, root)

    def test_a_sealed_night_writes_nothing(self):
        night = Path(self._tmp.name) / "sealed"
        night.mkdir()
        (night / "metadata.json").write_text(json.dumps({
            "night_arm": {"arm": "cue", "night_id": "n1", "sealed_schedule": True, "schedule_id": "s1",
                          "schedule_commitment_sha256": "ab" * 32, "schedule_unsealed_at_utc": None},
        }))
        root = Path(self._tmp.name) / "sealed_root"
        with self.assertRaises(SealedNightError):
            export_bids(night, root)
        self.assertFalse(root.exists())


@unittest.skipUnless(edf_available(), SKIP_REASON)
class TestAnImportedEdf(unittest.TestCase):
    """A BOAS-shaped headband EDF (made-up values) imported and exported again: the import
    does not record the device, so the export names none, where the real file's sidecar
    says Bitbrain IKON Sleep against the left mastoid."""

    @classmethod
    def setUpClass(cls):
        import edfio

        cls.edfio = edfio
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        source = root / "sub-3_task-Sleep_acq-headband_eeg.edf"
        write_boas_like_edf(source)
        cls.imported = root / "imported"
        import_recording(source, "edf", cls.imported, channel_map=resolve_channel_map("boas-headband"))
        cls.root = root / "bids"
        export_bids(cls.imported, cls.root, subject="b03", session="boas")
        cls.eeg_dir = cls.root / "sub-b03" / "ses-boas" / "eeg"
        cls.stem = "sub-b03_ses-boas_task-sleep"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_sidecar_names_no_device(self):
        sidecar = json.loads((self.eeg_dir / f"{self.stem}_eeg.json").read_text())

        for key in ("EEGReference", "Manufacturer", "ManufacturersModelName", "CapManufacturer",
                    "CapManufacturersModelName", "EEGPlacementScheme"):
            self.assertEqual(sidecar[key], "n/a", key)
        self.assertNotIn("EEGFullScale", sidecar)
        self.assertIn("imported: edf", sidecar["TaskDescription"])
        self.assertEqual((sidecar["EEGChannelCount"], sidecar["MISCChannelCount"]), (2, 1))

    def test_channels_and_edf_name_no_device(self):
        rows = _tsv(self.eeg_dir / f"{self.stem}_channels.tsv")
        self.assertEqual([row["name"] for row in rows][:2], ["AF7", "AF8"])
        self.assertFalse([row for row in rows if "Muse" in row["description"] or "headband" in row["description"]])
        pulse, = [row for row in rows if row["name"] == "HB_PULSE"]
        self.assertEqual((pulse["type"], pulse["units"]), ("MISC", "n/a"))

        edf = self.edfio.read_edf(self.eeg_dir / f"{self.stem}_eeg.edf", lazy_load_data=False)
        self.assertNotIn("Muse", edf.local_recording_identification)
        self.assertEqual(edf.recording.equipment_code, "X")
        self.assertEqual({signal.transducer_type for signal in edf.signals}, {""})

    def test_only_a_night_imported_from_an_edf_loses_the_headband(self):
        cases = (({}, True), ({"synthetic": True}, True), ({"import": {"format": "muselsl-csv"}}, True),
                 ({"import": {"format": "decoded-frames"}}, True), ({"import": {"format": "edf"}}, False))
        for metadata, expected in cases:
            with self.subTest(metadata=metadata):
                self.assertEqual(recorded_with_muse(metadata), expected)


if __name__ == "__main__":
    unittest.main()
