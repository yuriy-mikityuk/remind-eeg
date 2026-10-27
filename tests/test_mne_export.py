"""A night as an MNE Raw (#230): muse_tmr.data.mne_export.

On a labelled synthetic night (synthetic input, not sleep) generated into a temporary
directory. Skipped without MNE-Python, which REMind never installs.
"""

import asyncio
import datetime as dt
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muse_tmr.data.continuous import ContinuousConfig, reconstruct_recording
from muse_tmr.data.edf_io import EXPORT_STREAMS, export_config, place_annotations
from muse_tmr.data.mne_export import to_raw
from muse_tmr.data.night_events import collect_night_annotations
from tests.exchange_synthetic import labelled_night

HAVE_MNE = importlib.util.find_spec("mne") is not None


@unittest.skipUnless(HAVE_MNE, "needs MNE-Python: pip install mne")
class TestToRaw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.night = labelled_night(Path(cls._tmp.name) / "night", ble_gaps_per_hour=30.0)
        cls.raw = to_raw(cls.night.recording_dir)
        cls.recording = asyncio.run(reconstruct_recording(cls.night.recording_dir, export_config()))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_eeg_is_in_volts_and_the_rest_is_misc(self):
        raw = self.raw
        self.assertEqual(raw.info["sfreq"], 256.0)
        types = dict(zip(raw.ch_names, raw.get_channel_types()))
        self.assertEqual({types[name] for name in ("TP9", "AF7", "AF8", "TP10")}, {"eeg"})
        self.assertEqual({types[name] for name in ("ACC_X", "GYRO_Z", "LO_NIR")}, {"misc"})
        eeg = self.recording.eeg
        offset = self.recording.offset_samples("eeg")
        data = raw.get_data(picks=["AF7"])[0]
        np.testing.assert_allclose(data[offset:offset + eeg.n_samples], eeg.channels["AF7"] * 1e-6, atol=1e-12)
        self.assertEqual(raw.info["meas_date"], dt.datetime.fromtimestamp(self.recording.t0, tz=dt.timezone.utc))

    def test_annotations_are_the_edf_exports(self):
        annotations = collect_night_annotations(
            self.night.recording_dir, t0=self.recording.t0, to_unix=self.recording.unix_time
        )
        placed = place_annotations(
            self.recording,
            annotations,
            notes=("synthetic input, not sleep",),
            duration=self.raw.n_times / 256.0,
        )
        # MNE keeps its annotations sorted by onset; ties may come in another order.
        got = sorted((round(float(onset), 6), text) for onset, text in zip(self.raw.annotations.onset, self.raw.annotations.description))
        self.assertEqual(got, sorted((round(item.onset_s, 6), item.text) for item in placed))
        self.assertIn("BAD_ACQ_SKIP", set(self.raw.annotations.description))
        self.assertIn("Sleep stage R", set(self.raw.annotations.description))

    def test_a_discontinuous_night_marks_its_joins(self):
        config = ContinuousConfig(gap_policy="discontinuous", streams=EXPORT_STREAMS)
        raw = to_raw(self.night.recording_dir, config=config)

        boundaries = [onset for onset, text in zip(raw.annotations.onset, raw.annotations.description)
                      if text == "BAD boundary"]
        self.assertEqual(len(boundaries), len(self.recording.eeg.dropouts()))
        self.assertNotIn("BAD_ACQ_SKIP", [text for onset, text in zip(raw.annotations.onset, raw.annotations.description)
                                          if 1.0 < onset < raw.times[-1] - 1.0])
        self.assertLess(raw.n_times, self.raw.n_times)


if __name__ == "__main__":
    unittest.main()
