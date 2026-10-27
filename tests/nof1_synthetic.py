"""A hand-made N-of-1 series for the tests of #228: synthetic input, not sleep.

``write_series()`` writes, under a temporary root and never under the repository's
``data/``, what a series of cue, sham and no-cue nights leaves: a sealed schedule (then
unsealed), each night's puzzle session, assignment, dream report, retests (morning,
day7, day10) and a Pilot 5 summary with its awakening markers and arousal proxies per
cue. Every value is drawn from ``random.Random`` with a seed, so a series is the same
on every run; nothing comes from a recording or a real night. ``planted`` adds an arm
difference on cue nights: more lucid reports and cued puzzles solved more often than
uncued ones.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from muse_tmr.protocol import (
    NightPuzzleSession,
    NightSchedule,
    PuzzleCatalog,
    PuzzleTask,
    assign_cued_uncued_puzzles,
    generate_night_schedule,
    get_protocol_preset,
    unseal_night_schedule,
    write_night_schedule,
)
from muse_tmr.reports import (
    MorningRetestResult,
    build_dream_report,
    build_incorporation_coding,
    build_morning_retest,
    export_blind_coding_packet,
)

SYNTHETIC_NOTE = "synthetic input, not sleep"
SALT = "5a17" * 8
PUZZLES_PER_NIGHT = 4
FIRST_NIGHT = dt.datetime(2026, 11, 2, 23, 0, tzinfo=dt.timezone.utc)
SCHEDULE_CREATED = "2026-11-01T12:00:00+00:00"
AROUSAL_FLAGS = ("alpha", "beta", "emg", "imu", "hr", "any")


@dataclass(frozen=True)
class SyntheticSeries:
    root: Path
    schedule_path: Path
    schedule: NightSchedule
    protocol_dir: Path
    reports_dir: Path
    recordings_dir: Path
    catalog: PuzzleCatalog


def write_series(
    root: Path,
    *,
    blocks: int = 4,
    washout_nights: int = 0,
    schedule_seed: int = 2028,
    value_seed: int = 0,
    planted: bool = False,
    unseal: bool = True,
    unsealed_at: Optional[str] = None,
    delayed_retests: bool = True,
    arousal_scored: bool = True,
    raters: Sequence[str] = (),
    skip_nights: Sequence[str] = (),
) -> SyntheticSeries:
    root = Path(root)
    protocol_dir = root / "protocol"
    reports_dir = root / "reports"
    recordings_dir = root / "recordings"
    for directory in (protocol_dir, reports_dir, recordings_dir):
        directory.mkdir(parents=True, exist_ok=True)
    schedule = generate_night_schedule(
        schedule_seed,
        blocks,
        washout_nights,
        schedule_id="synthetic-series",
        salt=SALT,
        created_at_utc=SCHEDULE_CREATED,
    )
    schedule_path, _sealed = write_night_schedule(schedule, protocol_dir / "night_schedules" / "synthetic-series.json")
    block_nights = [night for night in schedule.nights if night.arm != "washout"]
    catalog = PuzzleCatalog(
        puzzles=tuple(
            PuzzleTask(
                f"{night.night_id}-p{index}",
                f"Synthetic puzzle {index} of {night.night_id}",
                f"answer {index}",
                cue_id=f"{night.night_id}-sound-{index}",
            )
            for night in block_nights
            for index in range(1, PUZZLES_PER_NIGHT + 1)
        )
    )
    catalog.save(protocol_dir / "puzzle_catalog.json")
    protocol = get_protocol_preset("remind_default_v1").manifest()
    last_form = FIRST_NIGHT
    for night in block_nights:
        if night.night_id in skip_nights:
            continue
        rng = random.Random(f"{value_seed}-{night.night_id}")
        started = FIRST_NIGHT + dt.timedelta(days=night.position - 1)
        morning = started + dt.timedelta(hours=8)
        session = NightPuzzleSession(
            session_id=night.night_id,
            puzzle_ids=tuple(f"{night.night_id}-p{index}" for index in range(1, PUZZLES_PER_NIGHT + 1)),
            puzzle_count=PUZZLES_PER_NIGHT,
        )
        session.save(protocol_dir / f"{night.night_id}_puzzles.json")
        assignment = assign_cued_uncued_puzzles(session, seed=night.position)
        assignment.save(protocol_dir / f"{night.night_id}_assignment.json")
        effect = planted and night.arm == "cue"
        lucid = rng.random() < (0.9 if effect else 0.25)
        if night.arm == "no_cue" and rng.random() < 0.8:
            perception = "not_perceived"
        else:
            perception = rng.choice(("in_dream", "unsure", "not_perceived"))
        woke = rng.choice(("no", "no", "no", "yes")) if night.arm != "no_cue" else "no"
        dream = build_dream_report(
            session,
            lucid=lucid,
            confidence=0.5,
            dream_text=f"Synthetic dream text for {night.night_id}, {SYNTHETIC_NOTE}.",
            cue_perception=perception,
            woke_from_sound=woke,
            puzzle_incorporation_text={session.puzzle_ids[0]: "a synthetic fragment"} if rng.random() < 0.5 else None,
            reported_at_utc=morning.isoformat(),
        )
        dream.save(reports_dir / f"{night.night_id}_dream_report.json")
        waves = (("morning", 0.5), ("day7", 7), ("day10", 10)) if delayed_retests else (("morning", 0.5),)
        for wave, days in waves:
            saved = morning + dt.timedelta(days=days if wave != "morning" else 0, minutes=30)
            last_form = max(last_form, saved)
            results = []
            for puzzle_id in session.puzzle_ids:
                cued = assignment.is_cued(puzzle_id)
                chance = (0.9 if cued else 0.2) if effect else 0.45
                results.append(
                    MorningRetestResult(
                        puzzle_id=puzzle_id,
                        response="synthetic",
                        solved=rng.random() < chance,
                        duration_seconds=60.0,
                        confidence=0.5,
                    )
                )
            retest = build_morning_retest(
                session,
                results,
                catalog=catalog,
                assignment=assignment,
                retest_wave=wave,
                reported_at_utc=saved.isoformat(),
            )
            suffix = "" if wave == "morning" else f"_{wave}"
            retest.save(reports_dir / f"{night.night_id}_retest{suffix}.json")
        for rater in raters:
            packet = export_blind_coding_packet(session, dream, catalog)
            codes = {}
            for puzzle_id in packet.puzzle_ids:
                chance = 0.6 if effect and assignment.is_cued(puzzle_id) else 0.2
                codes[puzzle_id] = ("direct_conceptual",) if rng.random() < chance else ()
            build_incorporation_coding(packet, rater, codes, coded_at_utc=morning.isoformat()).save(
                reports_dir / f"{night.night_id}_coding_{rater}.json"
            )
        _write_night(
            recordings_dir / night.night_id, night.night_id, night.arm, schedule, started, protocol, rng, arousal_scored
        )
    if unseal:
        stamp = unsealed_at or (last_form + dt.timedelta(days=1)).isoformat()
        schedule = unseal_night_schedule(schedule_path, now=stamp)
    return SyntheticSeries(
        root=root,
        schedule_path=schedule_path,
        schedule=schedule,
        protocol_dir=protocol_dir,
        reports_dir=reports_dir,
        recordings_dir=recordings_dir,
        catalog=catalog,
    )


def _write_night(
    night_dir: Path,
    night_id: str,
    arm: str,
    schedule: NightSchedule,
    started: dt.datetime,
    protocol: Dict[str, object],
    rng: random.Random,
    arousal_scored: bool,
) -> None:
    night_dir.mkdir(parents=True, exist_ok=True)
    plays = 0 if arm == "no_cue" else 4 + rng.randrange(3)
    rates: Dict[str, Optional[float]] = {
        flag: (None if arm == "no_cue" else round(rng.uniform(0.0, 0.4), 3)) for flag in AROUSAL_FLAGS
    }
    summary = {
        "synthetic": True,
        "note": SYNTHETIC_NOTE,
        "schema_version": 4,
        "pilot_id": "m8_pilot5_full_night_tlr_puzzle_cueing",
        "night_id": night_id,
        "arm": arm,
        "schedule_commitment_sha256": schedule.commitment_sha256(),
        "started_at": started.isoformat(),
        "ended_at": (started + dt.timedelta(hours=7)).isoformat(),
        "passed": True,
        "failed_criteria": [],
        "cue_play_count": plays,
        "puzzle_cue_play_count": max(0, plays - 3),
        "tlr_cue_play_count": min(plays, 3),
        "virtual_play_count": 5 if arm == "no_cue" else 0,
        "protocol": protocol,
        "cue_arousal": (
            {"status": "scored", "synthetic": True, "note": SYNTHETIC_NOTE, "arousal_proxy_per_cue": rates}
            if arousal_scored
            else {"status": "failed", "error": "synthetic: not scored"}
        ),
    }
    (night_dir / "pilot5_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markers = [
        json.dumps({"event_type": "awakening", "timestamp_utc": (started + dt.timedelta(hours=2 + index)).isoformat()})
        for index in range(rng.randrange(3))
    ]
    (night_dir / "awakening_events.jsonl").write_text("".join(line + "\n" for line in markers), encoding="utf-8")


def arms_by_night(schedule: NightSchedule) -> Dict[str, str]:
    return {night.night_id: night.arm for night in schedule.nights}


def block_orders(schedule: NightSchedule) -> Tuple[Tuple[str, ...], ...]:
    blocks: Dict[int, list] = {}
    for night in schedule.nights:
        if night.block is not None:
            blocks.setdefault(night.block, []).append(night.arm)
    return tuple(tuple(blocks[number]) for number in sorted(blocks))
