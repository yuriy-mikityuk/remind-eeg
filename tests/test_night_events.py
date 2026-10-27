"""A night's logs as timed annotations (#230): muse_tmr.data.night_events.

The logs are written with the classes the Pilot 4/5 runners use, into a temporary
directory, and one night by the Pilot 4 runner itself on made-up frames with mock audio;
every value is made up.
"""

import datetime as dt
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from muse_tmr.audio import MockAudioBackend, VolumeCalibration
from muse_tmr.audio.audio_player import CuePlaybackResult
from muse_tmr.data.night_events import (
    EDF_STAGE_TEXTS,
    LABEL_TO_EDF_STAGE,
    REM_LABELS,
    SealedNightError,
    blinding_refusal,
    collect_night_annotations,
    edf_stage_text,
    notes_for,
    read_hypnogram,
    stage_label,
)
from muse_tmr.protocol import TlrBlockConfig, default_tlr_cue, plan_tlr_block
from muse_tmr.protocol.arousal_guard import ArousalGuardDecision
from muse_tmr.protocol.tmr_scheduler import TmrSchedulerEvent
from muse_tmr.validation import run_pilot4_cueing_night
from muse_tmr.validation.gate_log import (
    GateDecisionRecord,
    emitting_record,
    gate_open_runs,
    load_rem_gate_decisions,
    write_rem_gate_decisions,
)
from muse_tmr.validation.pilot4 import AwakeningEvent
from tests.test_pilot4_cueing import FakeMuseSource, pilot4_config, rem_frames
from tests.test_pilot4_cueing import protocol_fixture as pilot4_protocol

STARTED = dt.datetime(2026, 1, 14, 23, 0, tzinfo=dt.timezone.utc)
FIRST_EPOCH = STARTED.timestamp() + 2.5
PRIVATE = ("AirPods", "EMERGENCY_STOP", "dreamt", "AA:BB:CC", "/Users/")


def _iso(seconds_after_start):
    return (STARTED + dt.timedelta(seconds=seconds_after_start)).isoformat()


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_cueing_night(directory, *, night_arm=None, gate_log=True):
    """What a Pilot 4 night leaves, cut down to the logs the collector reads.

    As the runner does, each epoch's gate, guard and scheduler rows carry the epoch's
    start, and the cue plays at its end, when the epoch is decided: the play logged at
    60 s sounds at 90 s, a little after the gate log's decision of that epoch.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"started_at": _iso(0), "pilot_id": "m8_pilot4_low_volume_cueing", "source": {"source_name": "amused"}}
    if night_arm is not None:
        metadata["night_arm"] = night_arm
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (directory / "pilot4_summary.json").write_text(json.dumps({"first_epoch_start": FIRST_EPOCH}), encoding="utf-8")
    _jsonl(
        directory / "events.jsonl",
        [
            {"event": "pilot_started", "timestamp_utc": _iso(0), "pilot_id": "m8_pilot4_low_volume_cueing",
             "source": "amused", "emergency_stop_path": "/Users/me/EMERGENCY_STOP"},
            {"event": "audio_playback_result", "timestamp_seconds": 60.0, "scheduler_event": {}, "playback_result": {}},
            {"event": "emergency_stop_triggered", "timestamp_utc": _iso(200),
             "emergency_stop_path": "/Users/me/EMERGENCY_STOP", "playback_result": {"device_name": "AirPods"}},
            {"event": "pilot_stopped", "timestamp_utc": _iso(240), "pilot_id": "m8_pilot4_low_volume_cueing",
             "reason": "duration_complete"},
        ],
    )
    scheduler = [
        TmrSchedulerEvent("skip", 0.0, reason_codes=("rem_gate_closed", "low_rem_probability")),
        TmrSchedulerEvent("skip", 30.0, reason_codes=("rem_gate_closed", "low_rem_probability")),
        TmrSchedulerEvent("play", 60.0, cue_id="cue-a", protocol="puzzle", puzzle_id="p1",
                          reason_codes=("rem_gate_open", "cued_puzzle")),
        TmrSchedulerEvent("pause", 90.0, reason_codes=("arousal_guard_pause", "motion_arousal_proxy")),
        TmrSchedulerEvent("pause", 120.0, reason_codes=("arousal_guard_pause", "motion_arousal_proxy")),
        TmrSchedulerEvent("skip", 150.0, reason_codes=("rem_gate_closed",)),
        TmrSchedulerEvent("stop", 180.0, reason_codes=("emergency_stop",)),
    ]
    _jsonl(directory / "scheduler_events.jsonl", [event.to_dict() for event in scheduler])
    if gate_log:
        opens = (False, False, True, True, True, False, True)
        records = [
            GateDecisionRecord(index=i, timestamp_seconds=30.0 * i, duration_seconds=30.0, gate_open=state)
            for i, state in enumerate(opens)
        ]
        write_rem_gate_decisions(records, directory / "rem_gate_decisions.jsonl")
    guard = [
        ArousalGuardDecision("allow", 0.0),
        ArousalGuardDecision("allow", 30.0),
        ArousalGuardDecision("lower_volume", 60.0, reason_codes=("alpha_arousal_proxy",), volume_multiplier=0.5),
        ArousalGuardDecision("pause", 90.0, reason_codes=("motion_arousal_proxy",), pause_seconds=60.0),
        ArousalGuardDecision("pause", 120.0, reason_codes=("motion_arousal_proxy",), pause_seconds=60.0),
        ArousalGuardDecision("allow", 150.0),
        ArousalGuardDecision("allow", 180.0),
    ]
    _jsonl(directory / "arousal_guard_events.jsonl", [decision.to_dict() for decision in guard])
    playback = CuePlaybackResult(
        cue_id="cue-a", status="played", backend_name="dry-run", requested_volume=0.1, effective_volume=0.05,
        max_volume=0.2, volume_capped=False, fade_in_seconds=0.25, fade_out_seconds=0.25,
        device_name="Someone's AirPods", started_at=_iso(92.6), ended_at=_iso(93.6),
    )
    _jsonl(directory / "audio_playback.jsonl", [playback.to_dict()])
    awakening = AwakeningEvent(timestamp_utc=_iso(150), notes="dreamt of the sea", timestamp_seconds=147.5,
                               attributed_to_cue="unsure")
    _jsonl(directory / "awakening_events.jsonl", [awakening.to_dict()])
    return directory


class TestStages(unittest.TestCase):
    def test_stage_texts_and_codes_map_to_the_four_labels(self):
        cases = {
            "W": "wake", "wake": "wake", "Sleep stage W": "wake", 0: "wake", "0": "wake",
            "N1": "nrem", "N2": "nrem", "N3": "nrem", "N": "nrem", "S4": "nrem", "Sleep stage 4": "nrem",
            "Sleep stage N2": "nrem", 1: "nrem", 2: "nrem", 3: "nrem",
            "R": "probable_rem", "REM": "probable_rem", "Sleep stage R": "probable_rem", 4: "probable_rem",
            "probable_rem": "probable_rem",
            "?": "unknown", "Sleep stage ?": "unknown", "Movement time": "unknown", 8: "unknown", -2: "unknown",
            "": "unknown", None: "unknown", float("nan"): "unknown", 2.5: "unknown",
        }
        for stage, label in cases.items():
            with self.subTest(stage=stage):
                self.assertEqual(stage_label(stage), label)

    def test_the_lossy_map_writes_no_n1_n2_or_n3(self):
        self.assertEqual(
            LABEL_TO_EDF_STAGE,
            {"wake": "Sleep stage W", "nrem": "Sleep stage N", "probable_rem": "Sleep stage R",
             "unknown": "Sleep stage ?"},
        )
        self.assertEqual(set(LABEL_TO_EDF_STAGE), set(REM_LABELS))
        self.assertFalse([text for text in EDF_STAGE_TEXTS if text[-1] in "123"])
        for label in REM_LABELS:
            self.assertEqual(stage_label(edf_stage_text(label)), label)
        with self.assertRaises(ValueError):
            edf_stage_text("N2")


class TestReadHypnogram(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_synthetic_nights_hypnogram_csv(self):
        path = self.root / "hypnogram.csv"
        path.write_text("epoch,start_s,end_s,stage\n0,0.000,30.000,W\n1,30.000,60.000,N3\n2,60.000,75.500,R\n")

        intervals = read_hypnogram(path)

        self.assertEqual([(i.onset_s, i.duration_s, i.stage, i.label) for i in intervals],
                         [(0.0, 30.0, "W", "wake"), (30.0, 30.0, "N3", "nrem"), (60.0, 15.5, "R", "probable_rem")])
        self.assertFalse(any(interval.absolute for interval in intervals))

    def test_a_boas_events_file_prefers_the_human_scoring(self):
        path = self.root / "sub-3_task-Sleep_acq-psg_events.tsv"
        path.write_text(
            "onset\tduration\tbegsample\tendsample\toffset\tstage_hum\tstage_ai\n"
            "0\t30\t1\t7680\t0\t0\t0\n30\t30\t7681\t15360\t0\t2\t4\n60\t30\t15361\t23040\t0\t4\t4\n"
            "90\t30\t23041\t30720\t0\t8\t-2\n"
        )

        intervals = read_hypnogram(path)
        ai = read_hypnogram(path, column="stage_ai")

        self.assertEqual([i.label for i in intervals], ["wake", "nrem", "probable_rem", "unknown"])
        self.assertEqual([i.label for i in ai], ["wake", "probable_rem", "probable_rem", "unknown"])
        self.assertEqual(intervals[2].onset_s, 60.0)

    def test_annotate_template_rows_are_host_times(self):
        path = self.root / "rows.csv"
        path.write_text("epoch_index,start_time,end_time,duration_seconds,label\n0,100.0,130.0,30,wake\n1,130.0,160.0,30,probable_rem\n")
        json_path = self.root / "rows.json"
        json_path.write_text(json.dumps([{"start_time": 100.0, "end_time": 130.0, "label": "nrem"}]))

        intervals = read_hypnogram(path)

        self.assertTrue(all(interval.absolute for interval in intervals))
        self.assertEqual([(i.onset_s, i.label) for i in intervals], [(100.0, "wake"), (130.0, "probable_rem")])
        self.assertEqual([i.label for i in read_hypnogram(json_path)], ["nrem"])

    def test_files_without_stages_or_times_are_refused(self):
        no_stage = self.root / "a.tsv"
        no_stage.write_text("onset\tduration\tvalue_x\n0\t30\t1\n")
        no_time = self.root / "b.csv"
        no_time.write_text("stage\nW\n")
        for path in (no_stage, no_time, self.root / "c.txt"):
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                read_hypnogram(path)


class TestCollectNightAnnotations(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.night = write_cueing_night(Path(self._tmp.name) / "night")

    def tearDown(self):
        self._tmp.cleanup()

    def _by_kind(self, annotations):
        result = {}
        for item in annotations:
            result.setdefault(item.kind, []).append(item)
        return result

    def test_the_cueing_logs_become_timed_annotations(self):
        annotations = collect_night_annotations(self.night)
        kinds = self._by_kind(annotations)

        # Onsets count from the first epoch's start without an export's t0, and a
        # scheduler or guard row goes in where its epoch was decided, 30 s after its start.
        skip, = [a for a in kinds["scheduler_skip"] if a.time_unix == FIRST_EPOCH + 30.0]
        self.assertEqual(skip.text, "scheduler skip reasons=rem_gate_closed,low_rem_probability n=2")
        self.assertEqual((skip.onset_s, skip.duration_s), (30.0, 60.0))
        play, = kinds["scheduler_play"]
        self.assertEqual(play.text, "scheduler play cue=cue-a protocol=puzzle puzzle=p1 reasons=rem_gate_open,cued_puzzle")
        self.assertEqual((play.onset_s, play.duration_s), (90.0, 0.0))
        self.assertEqual([(a.onset_s, a.text) for a in kinds["scheduler_stop"]], [(210.0, "scheduler stop reasons=emergency_stop")])
        self.assertEqual([a.onset_s for a in kinds["scheduler_pause"]], [120.0, 150.0])

        guard = kinds["arousal_guard"]
        self.assertEqual([(a.onset_s, a.duration_s, a.text) for a in guard], [
            (90.0, 30.0, "arousal_guard lower_volume reasons=alpha_arousal_proxy volume_multiplier=0.5"),
            (120.0, 60.0, "arousal_guard pause reasons=motion_arousal_proxy pause_seconds=60 n=2"),
        ])
        audio, = kinds["audio"]
        self.assertEqual(audio.text, "audio played cue=cue-a volume=0.050 backend=dry-run")
        self.assertAlmostEqual(audio.time_unix, STARTED.timestamp() + 92.6)
        self.assertAlmostEqual(audio.duration_s, 1.0)
        # The play is where the sound began, inside the gate-open run that let it through.
        self.assertLess(audio.time_unix - play.time_unix, 1.0)
        self.assertGreaterEqual(audio.time_unix, play.time_unix)
        run = kinds["rem_gate_open"][0]
        self.assertTrue(run.onset_s <= play.onset_s < run.onset_s + run.duration_s)
        awakening, = kinds["awakening"]
        self.assertEqual((awakening.onset_s, awakening.text), (147.5, "awakening attributed_to_cue=unsure"))

        recording = [a.text for a in kinds["recording"]]
        self.assertEqual(recording, [
            "pilot_started pilot_id=m8_pilot4_low_volume_cueing",
            "emergency_stop_triggered",
            "pilot_stopped reason=duration_complete pilot_id=m8_pilot4_low_volume_cueing",
        ])
        self.assertEqual([a.onset_s for a in annotations], sorted(a.onset_s for a in annotations))

    def test_gate_runs_are_the_runs_of_the_gate_log(self):
        annotations = collect_night_annotations(self.night)
        runs = [a for a in annotations if a.kind == "rem_gate_open"]

        records = [
            GateDecisionRecord.from_dict(json.loads(line))
            for line in (self.night / "rem_gate_decisions.jsonl").read_text().splitlines()
        ]
        expected = gate_open_runs(records)
        self.assertEqual(
            [(a.onset_s, a.duration_s) for a in runs],
            [(run.start_seconds, run.duration_seconds) for run in expected],
        )
        self.assertEqual([(a.onset_s, a.duration_s) for a in runs], [(90.0, 90.0), (210.0, 0.0)])

    def test_without_a_gate_log_the_scheduler_codes_give_the_runs(self):
        night = write_cueing_night(Path(self._tmp.name) / "old", gate_log=False)

        annotations = collect_night_annotations(night)
        runs = [a for a in annotations if a.kind == "rem_gate_open"]

        # rem_gate_open is only on the play logged at 60 s; the skip logged at 150 s closes
        # it. Each was decided an epoch of 30 s later, and so is the play.
        self.assertEqual([(a.onset_s, a.duration_s, a.source) for a in runs], [(90.0, 90.0, "scheduler_events.jsonl")])
        self.assertEqual([a.onset_s for a in annotations if a.kind == "scheduler_play"], [90.0])

    def test_nothing_private_reaches_a_text(self):
        texts = " ".join(item.text for item in collect_night_annotations(self.night))
        for private in PRIVATE:
            self.assertNotIn(private, texts)

    def test_recorder_events_count_from_started_at(self):
        night = Path(self._tmp.name) / "recorded"
        night.mkdir()
        (night / "metadata.json").write_text(json.dumps({"started_at": _iso(0)}))
        _jsonl(night / "events.jsonl", [
            {"event": "recording_started", "timestamp": 5000.0, "details": {"source": "amused"}},
            {"event": "stream_error", "timestamp": 5010.25, "details": {"error": "AA:BB:CC lost"}},
            {"event": "reconnect_attempt", "timestamp": 5011.0, "details": {"attempt": 1, "backoff_seconds": 1.0}},
            {"event": "recording_stopped", "timestamp": 5600.0, "details": {"reason": "duration_complete"}},
        ])

        annotations = collect_night_annotations(night)

        self.assertEqual([(a.onset_s, a.text) for a in annotations], [
            (0.0, "recording_started"),
            (10.25, "stream_error"),
            (11.0, "reconnect_attempt attempt=1 backoff_seconds=1.0"),
            (600.0, "recording_stopped reason=duration_complete"),
        ])

    def test_hypnograms_another_recording_holds_are_not_read(self):
        (self.night / "hypnogram.csv").write_text("epoch,start_s,end_s,stage\n0,0,30,W\n")
        self.assertFalse([a for a in collect_night_annotations(self.night) if a.kind == "stage"])

        extra = Path(self._tmp.name) / "scored.tsv"
        extra.write_text("onset\tduration\tstage\n0\t30\tN2\n30\t30\tR\n")
        stages = [a for a in collect_night_annotations(self.night, hypnograms=[extra]) if a.kind == "stage"]
        self.assertEqual([(a.onset_s, a.text) for a in stages], [(0.0, "Sleep stage N"), (30.0, "Sleep stage R")])

    def test_notes_name_a_synthetic_or_imported_night(self):
        self.assertEqual(notes_for({"synthetic": True}), ("synthetic input, not sleep",))
        banner = "imported: muselsl-csv, scale 1; report and simulation only"
        self.assertEqual(notes_for({"import": {"banner": banner}}), (banner,))
        self.assertEqual(notes_for({}), ())


class TestARunnersNight(unittest.IsolatedAsyncioTestCase):
    """A night the Pilot 4 runner writes, with mock audio and a TLR block: each scheduler
    play goes in where the gate log says its epoch was decided, which is where the runner
    played it, inside the gate-open run that let it through."""

    async def test_plays_land_where_their_epoch_was_decided(self):
        catalog, session, assignment, cue_library = pilot4_protocol(include_tlr=True)
        # Two TLR cues 8 s apart: the second is logged at its planned time, between two
        # epoch starts, and plays when the next epoch is decided.
        plan = plan_tlr_block(
            default_tlr_cue(),
            config=TlrBlockConfig(repetitions=2, interval_seconds=8.0, post_block_pause_seconds=0.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            night = Path(tmp)
            summary = await run_pilot4_cueing_night(
                FakeMuseSource(rem_frames(6)),
                config=replace(
                    pilot4_config(night, duration_seconds=60.0, enable_tlr_block=True), score_cue_arousal=False
                ),
                catalog=catalog,
                session=session,
                assignment=assignment,
                cue_library=cue_library,
                calibration=VolumeCalibration("Sleep Headphones", 0.01, 0.02, 0.04),
                backend=MockAudioBackend(),
                tlr_block_plan=plan,
            )
            annotations = collect_night_annotations(night)
            records = load_rem_gate_decisions(night / "rem_gate_decisions.jsonl")
            logged = [json.loads(line) for line in (night / "scheduler_events.jsonl").read_text().splitlines()]

        logged_plays = [row for row in logged if row["event_type"] == "play"]
        plays = [a for a in annotations if a.kind == "scheduler_play"]
        self.assertEqual([row["timestamp_seconds"] for row in logged_plays], [30.0, 38.0, 90.0, 120.0])
        self.assertEqual(
            [a.onset_s for a in plays],
            [emitting_record(records, row["timestamp_seconds"]).decision_seconds for row in logged_plays],
        )
        self.assertEqual([a.onset_s for a in plays], [60.0, 90.0, 120.0, 150.0])
        self.assertEqual(plays[0].onset_s, summary.first_play_seconds)
        run, = [a for a in annotations if a.kind == "rem_gate_open"]
        self.assertEqual(run.onset_s, summary.first_gate_open_seconds)
        for play in plays:
            self.assertTrue(run.onset_s <= play.onset_s <= run.onset_s + run.duration_s, play)


class TestBlinding(unittest.TestCase):
    ARM = {
        "arm": "sham",
        "night_id": "n03",
        "sham_cue_id": "sham-tone",
        "sealed_schedule": True,
        "schedule_id": "series-a",
        "schedule_commitment_sha256": "ab" * 32,
        "schedule_unsealed_at_utc": None,
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.night = write_cueing_night(Path(self._tmp.name) / "night", night_arm=dict(self.ARM))

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_night_from_a_sealed_schedule_is_refused(self):
        with self.assertRaises(SealedNightError) as caught:
            collect_night_annotations(self.night)
        self.assertIn("unseal-night-schedule", str(caught.exception))
        self.assertNotIn("sham", str(caught.exception).replace("sham_cue", ""))

    def test_the_schedule_must_be_the_nights_and_unsealed(self):
        metadata = {"night_arm": dict(self.ARM)}
        unsealed = {"schedule_id": "series-a", "commitment_sha256": "ab" * 32, "unsealed_at_utc": "2026-02-01T08:00:00+00:00"}
        self.assertIsNone(blinding_refusal(metadata, unsealed))
        self.assertIn("still sealed", blinding_refusal(metadata, {**unsealed, "unsealed_at_utc": None}))
        self.assertIn("series-b", blinding_refusal(metadata, {**unsealed, "schedule_id": "series-b"}))
        self.assertIn("commitment", blinding_refusal(metadata, {**unsealed, "commitment_sha256": "cd" * 32}))

        annotations = collect_night_annotations(self.night, night_schedule=unsealed)
        self.assertTrue(annotations)

    def test_open_label_and_unarmed_nights_export(self):
        open_label = {**self.ARM, "sealed_schedule": False, "schedule_id": None}
        self.assertIsNone(blinding_refusal({"night_arm": open_label}))
        self.assertIsNone(blinding_refusal({}))


if __name__ == "__main__":
    unittest.main()
