"""Per-arm description of an N-of-1 series of cue, sham and no-cue nights (#228).

``build_nof1_analysis()`` reads a series once its schedule is unsealed: the arm of every
night from the schedule, what each night logged from its Pilot 4/5 summary or Pilot 3
report, the arousal proxies per cue of ``score-cue-arousal`` (``arousal_proxy_per_cue``),
and the morning forms: lucidity, cue perception and sound awakenings from the dream
report (schema 2), incorporation from rater codes or else the sleeper's links, and the
cued-minus-uncued solve rate of each retest wave from ``analysis.py``. It defines no
schema of its own for any of them.

For each outcome it gives the mean per arm, with and without the series' first night,
and for each pair of arms the difference of the means with a permutation interval: the
arm labels are moved within their blocks the way ``plan-night-schedule`` draws them
(within every group of three blocks the orders are distinct rows of one Latin square),
over every order it could have drawn when there are at most ``permutations`` of them,
else over ``permutations`` orders drawn with ``random.Random(seed)``. The interval is
what is left of those differences once the lowest and the highest 2.5 % are cut: what
the difference looks like when the labels carry no information. With two or three
blocks the schedule has 12 orders, and nothing can fall outside. It also gives the lag-1
autocorrelation of each outcome once the arm means are taken out, and flags the first
night.

All of it is a single-case description in pure Python, deterministic for a seed. No
p-value is computed, and nothing here measures whether cueing works: at n = 1 the
experimenter is the participant, a sham still makes sound, consecutive nights can carry
over, and every arm comparison inherits the REM detector's open issues (#116, #167).
The Markdown report ends with a SCRIBE 2016 / CENT 2015 checklist that marks each item
``auto``, ``partial`` or ``manual``.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from muse_tmr import __version__
from muse_tmr.protocol import (
    NightPuzzleSession,
    NightSchedule,
    PuzzleCueAssignment,
    ScheduledNight,
    protocol_block_of,
)
from muse_tmr.reports.analysis import (
    AnalysisLimitation,
    build_cued_uncued_analysis,
    incorporation_by_puzzle,
)
from muse_tmr.reports.cue_recall import CueRecallTest
from muse_tmr.reports.dream_report import HEARD_PERCEPTIONS, DreamReport
from muse_tmr.reports.incorporation_coding import CODING_KIND, IncorporationCodingSheet
from muse_tmr.reports.morning_retest import RETEST_WAVES, MorningRetest

NOF1_ANALYSIS_SCHEMA_VERSION = 1
REPORT_KIND = "nof1_analysis"
SYNTHETIC_NOTE = "synthetic input, not sleep"
DEFAULT_SEED = 0
DEFAULT_PERMUTATIONS = 10000
DEFAULT_MIN_NIGHTS_PER_ARM = 5
INTERVAL_LEVEL = 0.95
# First arm minus second arm.
CONTRASTS = ("cue-no_cue", "cue-sham", "sham-no_cue")
DELAYED_WAVES = tuple(wave for wave in RETEST_WAVES if wave != "morning")
SUMMARY_FILENAMES = ("pilot4_summary.json", "pilot5_summary.json")
AROUSAL_REPORT_FILENAME = "cue_arousal_report.json"
AWAKENING_EVENTS_FILENAME = "awakening_events.jsonl"
PLAYING_ARMS = ("cue", "sham")
_ROUND = 12
# One night of the analysis as _night_row() builds it.
Row = Dict[str, Any]


def _wave_outcome(wave: str) -> str:
    return f"cued_minus_uncued_solve_rate_{wave}"


OUTCOMES: Dict[str, str] = {
    "lucid": "1 when the dream report says lucid, else 0.",
    "cue_perceived": (
        "1 when the dream report says a sound was heard (cue_perception in_dream, awake or unsure), 0 for "
        "not_perceived; unknown is missing."
    ),
    "woke_from_sound": "1 when the dream report says a sound woke the sleeper, 0 for no; unknown is missing.",
    "awakening_markers": "Awakening markers in the night's awakening_events.jsonl (log-pilot4-awakening).",
    "incorporation_rate": (
        "Share of the night's puzzles incorporated: by rater codes when the night has them, else by the "
        "sleeper's own links."
    ),
}
OUTCOMES.update(
    {
        _wave_outcome(wave): (
            f"Solve rate of the night's cued puzzles minus its uncued ones at the {wave} retest. On a sham or "
            "no-cue night the cued puzzles' sounds did not play."
        )
        for wave in RETEST_WAVES
    }
)

# SCRIBE 2016 items (Tate et al. 2016) with the CENT 2015 items (Vohra et al. 2015) on the
# same topic, by CENT's numbers (CONSORT 2010's, plus 4c), then three CENT items SCRIBE
# has no item for, and what this report fills. Status: auto (the report holds it),
# partial (the report holds part of it), manual (the author writes it).
REPORTING_CHECKLIST: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("1", "1a", "Title names the single-case design", "partial",
     "The report's title names an N-of-1 series; the paper's title is yours."),
    ("2", "1b", "Abstract", "manual", "Yours."),
    ("3", "2a", "Scientific background", "manual", "Yours; docs/references.md lists the papers REMind leans on."),
    ("4", "2b", "Aims and hypotheses", "manual", "Yours, as preregistered with prereg-template."),
    ("5", "3a", "Design: arms, phases and their sequence", "auto",
     "Series: blocks, arms, washout nights and every night's arm in order."),
    ("6", "3b, 6b", "Changes after the start", "partial",
     "Nights whose protocol effective hash differs from the first night's; the reasons are yours."),
    ("7", "", "Planned replication", "manual", "Yours."),
    ("8", "8a, 8b, 9", "Randomisation and concealment", "auto",
     "How the order was drawn (seed, Latin-square rows), the commitment and the seal check."),
    ("9", "11a, 11b", "Blinding", "partial",
     "What stayed sealed and until when, and unblinded_early; who was blind and how alike the sounds were are "
     "yours."),
    ("10", "4a", "Selection criteria", "manual", "Yours."),
    ("11", "15", "Participant characteristics", "manual", "Yours."),
    ("12", "4b", "Setting", "manual", "Yours."),
    ("13", "4c", "Ethics approval and consent", "manual", "Yours."),
    ("14", "6a", "Measures", "partial",
     "Each outcome's definition and source; rater agreement comes from compare-incorporation-codings; "
     "validity is yours."),
    ("15", "", "Equipment and materials", "partial",
     "REMind's version, the protocol preset and its hashes, the sham cue; the headband and speaker are yours."),
    ("16", "5", "Interventions as given", "partial",
     "Plays and virtual plays per night and arm; the protocol is in docs/protocol_presets.md."),
    ("17", "", "Procedural fidelity", "partial",
     "Each night's pilot criteria and arm_mismatch; protocol-fidelity reports are not read here."),
    ("18", "12a, 12b", "Analysis methods", "auto",
     "Per-arm means, permutation intervals, lag-1 autocorrelation and the means without the first night."),
    ("19", "13a, 13b, 14a, 14b", "Sequence completed and losses", "partial",
     "Nights planned and analysed per arm, missing forms and each night's start; why the series ended is "
     "yours."),
    ("20", "16, 17a, 17b", "Results per period and estimates", "partial",
     "Every night's values, the per-arm means and each difference with its interval; no relative effect for "
     "the yes/no outcomes."),
    ("21", "19", "Adverse events", "partial",
     "Awakening markers, sound awakenings and arousal proxies per cue; other harms are yours."),
    ("22", "22", "Interpretation", "manual", "Yours."),
    ("23", "20", "Limitations", "partial", "The limitation codes; the discussion is yours."),
    ("24", "21", "Applicability", "manual", "Yours."),
    ("25", "23, 24", "Protocol and registration", "partial",
     "The commitment a preregistration carries; where it was filed is yours."),
    ("26", "25", "Funding", "manual", "Yours."),
    ("", "7a, 7b", "Number of periods and stopping rule", "manual", "Yours, as preregistered."),
    ("", "10", "Who drew the sequence and who assigned the nights", "partial",
     "plan-night-schedule drew the order and each runner read its night's arm from the schedule; who ran "
     "them is yours."),
    ("", "18", "Carry-over, period effects and serial correlation", "auto",
     "carry_over_possible, the first night's flag and the means without it, lag-1 autocorrelation."),
)
CHECKLIST_STATUSES = ("auto", "partial", "manual")


class ScheduleStillSealed(ValueError):
    """The series' schedule has not been unsealed: no arm may be analysed yet."""


@dataclass(frozen=True)
class Nof1NightInputs:
    """What one night of a series left: its summary and its morning files."""

    night_id: str
    summary: Mapping[str, object] = field(default_factory=dict)
    summary_path: Optional[str] = None
    session: Optional[NightPuzzleSession] = None
    assignment: Optional[PuzzleCueAssignment] = None
    dream_report: Optional[DreamReport] = None
    retests: Tuple[MorningRetest, ...] = ()
    cue_recall: Optional[CueRecallTest] = None
    codings: Tuple[IncorporationCodingSheet, ...] = ()
    # Initiative 3's rates per flag (alpha, beta, emg, imu, hr, any), and where they came from.
    arousal_proxy_per_cue: Optional[Mapping[str, object]] = None
    arousal_source: Optional[str] = None
    awakening_count: Optional[int] = None
    synthetic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", dict(self.summary or {}))
        object.__setattr__(self, "retests", tuple(self.retests))
        object.__setattr__(self, "codings", tuple(self.codings))

    @property
    def retest_waves(self) -> Tuple[str, ...]:
        return tuple(wave for wave in RETEST_WAVES if any(item.retest_wave == wave for item in self.retests))

    @property
    def has_data(self) -> bool:
        return bool(self.summary) or self.dream_report is not None or bool(self.retests)


@dataclass(frozen=True)
class Nof1AnalysisReport:
    schedule: Mapping[str, object]
    settings: Mapping[str, object]
    nights: Tuple[Mapping[str, object], ...]
    descriptives: Mapping[str, object]
    differences: Tuple[Mapping[str, object], ...]
    autocorrelation: Mapping[str, object]
    cue_counts: Mapping[str, object]
    arousal: Mapping[str, object]
    limitations: Tuple[AnalysisLimitation, ...]
    synthetic: bool = False
    analysis_id: str = ""
    generated_at_utc: str = ""
    schema_version: int = NOF1_ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "nights", tuple(dict(item) for item in self.nights))
        object.__setattr__(self, "differences", tuple(dict(item) for item in self.differences))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if not self.analysis_id:
            object.__setattr__(self, "analysis_id", f"{self.schedule.get('schedule_id', 'series')}-nof1-analysis")
        if not self.generated_at_utc:
            object.__setattr__(self, "generated_at_utc", _utc_now())

    @property
    def limitation_codes(self) -> Tuple[str, ...]:
        return tuple(item.code for item in self.limitations)

    def difference(self, outcome: str, contrast: str) -> Mapping[str, object]:
        for item in self.differences:
            if item["outcome"] == outcome and item["contrast"] == contrast:
                return item
        raise KeyError(f"no difference for {outcome} {contrast}")

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        if self.synthetic:
            payload.update({"synthetic": True, "note": SYNTHETIC_NOTE})
        payload.update(
            {
                "schema_version": self.schema_version,
                "report": REPORT_KIND,
                "analysis_id": self.analysis_id,
                "generated_at_utc": self.generated_at_utc,
                "schedule": dict(self.schedule),
                "settings": dict(self.settings),
                "outcomes": dict(OUTCOMES),
                "nights": [dict(item) for item in self.nights],
                "descriptives": dict(self.descriptives),
                "differences": [dict(item) for item in self.differences],
                "autocorrelation": dict(self.autocorrelation),
                "cue_counts": dict(self.cue_counts),
                "arousal": dict(self.arousal),
                "limitations": [item.to_dict() for item in self.limitations],
                "reporting_checklist": checklist_rows(),
                "reporting_checklist_counts": checklist_counts(),
                "interpretation": "descriptive_single_case",
            }
        )
        return payload

    def to_markdown(self) -> str:
        schedule = self.schedule
        settings = self.settings
        arms = list(schedule.get("arms", ()) or ())
        lines = [f"# N-of-1 series {schedule.get('schedule_id')}: description by arm", ""]
        if self.synthetic:
            lines += [f"**{SYNTHETIC_NOTE.capitalize()}.** Nothing here is about sleep.", ""]
        lines += [
            "A single-case description of one participant's series. No p-value is computed, and a difference "
            "outside its interval describes this series; it is not evidence that cueing works.",
            "",
            "## Series",
            "",
            f"- Schedule `{schedule.get('schedule_id')}`: {schedule.get('blocks')} blocks of {', '.join(arms)}, "
            f"{schedule.get('washout_nights')} washout night(s) between blocks, {schedule.get('night_count')} nights.",
            f"- Order drawn with `random.Random({schedule.get('seed')})` on {str(schedule.get('created_at_utc'))[:10]}; "
            f"commitment `{schedule.get('commitment_sha256')}`, seal {schedule.get('seal')}, unsealed at "
            f"{schedule.get('unsealed_at_utc')}.",
            f"- Sham sound: `{schedule.get('sham_cue_id')}`. Protocol effective hashes: "
            f"{_join(schedule.get('protocol_effective_hashes')) or 'none recorded'}.",
            f"- Permutation intervals: {settings.get('reference_set')} orders, {settings.get('possible_orders')} "
            f"possible, {settings.get('method')} ({settings.get('relabelings')} used), seed {settings.get('seed')}, "
            f"central {float(settings.get('interval', INTERVAL_LEVEL)) * 100:g} %. "
            + (
                "With this few orders nothing can fall outside an interval."
                if not settings.get("outside_possible")
                else "Lowest and highest 2.5 % of the differences cut."
            ),
            f"- Primary: {settings.get('primary_outcome') or 'none named'}"
            + (f", {settings.get('primary_contrast')}" if settings.get("primary_outcome") else "")
            + f". REMind {settings.get('software_version')}.",
            "",
            "## Nights",
            "",
            "| # | Night | Block | Arm | Lucid | Heard | Woke | Awakenings | Incorporation | Delta morning "
            "| Delta day7 | Delta day10 | Plays | Virtual |",
            "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for night in self.nights:
            values = dict(night.get("values", {}) or {})
            first = " (first)" if night.get("first_night") else ""
            lines.append(
                f"| {night['position']} | {night['night_id']}{first} | {night['block']} | {night['arm']} | "
                + " | ".join(
                    _number(values.get(name))
                    for name in (
                        "lucid",
                        "cue_perceived",
                        "woke_from_sound",
                        "awakening_markers",
                        "incorporation_rate",
                        _wave_outcome("morning"),
                        _wave_outcome("day7"),
                        _wave_outcome("day10"),
                    )
                )
                + f" | {_number(night.get('plays'))} | {_number(night.get('virtual_plays'))} |"
            )
        lines += ["", "## Means by arm", "", "| Outcome | " + " | ".join(arms) + " | Without the first night |",
                  "| --- | " + " | ".join("---:" for _ in arms) + " | --- |"]
        for outcome, by_arm in self.descriptives.items():
            stats = {arm: dict(by_arm.get(arm, {}) or {}) for arm in arms}
            cells = [f"{_number(stats[arm].get('mean'))} ({stats[arm].get('nights', 0)})" for arm in arms]
            without = ", ".join(f"{arm} {_number(stats[arm].get('mean_without_first_night'))}" for arm in arms)
            lines.append(f"| {outcome} | " + " | ".join(cells) + f" | {without} |")
        lines += ["", "## Differences between arms", "",
                  "| Outcome | Contrast | Difference | Interval | Outside | Relabelings | Role |",
                  "| --- | --- | ---: | --- | --- | ---: | --- |"]
        for item in self.differences:
            interval = item.get("interval")
            interval_text = f"{_number(interval[0])} to {_number(interval[1])}" if interval else "n/a"
            outside = item.get("observed_outside_interval")
            lines.append(
                f"| {item['outcome']} | {item['contrast']} | {_number(item.get('difference'))} | {interval_text} | "
                f"{'n/a' if outside is None else ('yes' if outside else 'no')} | {item.get('relabelings', 0)} | "
                f"{'primary' if item.get('primary') else 'exploratory'} |"
            )
        lines += ["", "## Serial dependence", "", "| Outcome | Lag-1 autocorrelation | Pairs |", "| --- | ---: | ---: |"]
        for outcome, item in self.autocorrelation.items():
            lines.append(f"| {outcome} | {_number(item.get('lag1'))} | {item.get('pairs', 0)} |")
        lines += [
            "",
            "## Plays and arousal proxies by arm",
            "",
            "| Arm | Nights | Plays | Virtual plays | Any arousal flag per cue |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for arm in arms:
            counts = dict(self.cue_counts.get(arm, {}) or {})
            rates = dict(dict(self.arousal.get(arm, {}) or {}).get("mean_rate", {}) or {})
            lines.append(
                f"| {arm} | {counts.get('nights_with_summary', 0)} | {_number(counts.get('plays'))} | "
                f"{_number(counts.get('virtual_plays'))} | {_number(rates.get('any'))} |"
            )
        lines += ["", "## Limitations", ""]
        lines.extend(f"- `{item.code}`: {item.message}" for item in self.limitations)
        counts = checklist_counts()
        lines += [
            "",
            "## Reporting checklist: SCRIBE 2016 and CENT 2015",
            "",
            f"{counts['auto']} items auto, {counts['partial']} partial, {counts['manual']} manual "
            "(Tate et al. 2016; Vohra et al. 2015).",
            "",
            "| SCRIBE | CENT | Item | Status | Here |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in checklist_rows():
            lines.append(
                f"| {row['scribe'] or '-'} | {row['cent'] or '-'} | {row['item']} | {row['status']} | {row['filled_by']} |"
            )
        return "\n".join(lines) + "\n"

    def save(self, output_path: Path) -> Path:
        return _write_text(output_path, json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")

    def save_markdown(self, output_path: Path) -> Path:
        return _write_text(output_path, self.to_markdown())


def build_nof1_analysis(
    schedule: NightSchedule,
    nights: Iterable[Nof1NightInputs],
    *,
    seal: str = "not_checked",
    seed: int = DEFAULT_SEED,
    permutations: int = DEFAULT_PERMUTATIONS,
    min_nights_per_arm: int = DEFAULT_MIN_NIGHTS_PER_ARM,
    primary_outcome: Optional[str] = None,
    primary_contrast: Optional[str] = None,
    analysis_id: str = "",
    generated_at_utc: str = "",
) -> Nof1AnalysisReport:
    """The per-arm description of an unsealed series. ``nights`` are the inputs of its
    block nights (``load_series_inputs()``); a night left out counts as one without
    data. ScheduleStillSealed for a schedule ``unseal-night-schedule`` has not unsealed.
    ``seal`` is what the report says of the schedule against its sealed companion:
    ``matches`` once the caller has checked it with ``verify_seal()``, as
    ``analyze-nof1`` does, else ``not_checked``."""
    if schedule.sealed:
        raise ScheduleStillSealed(
            f"night schedule {schedule.schedule_id} is still sealed: run `muse-tmr unseal-night-schedule` on it "
            "after the last delayed retest of the series, then analyze-nof1"
        )
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if min_nights_per_arm <= 0:
        raise ValueError("min_nights_per_arm must be positive")
    if primary_outcome is not None and primary_outcome not in OUTCOMES:
        raise ValueError(f"primary_outcome must be one of: {', '.join(OUTCOMES)}")
    if primary_contrast is not None and primary_contrast not in CONTRASTS:
        raise ValueError(f"primary_contrast must be one of: {', '.join(CONTRASTS)}")
    if primary_outcome is not None and primary_contrast is None:
        primary_contrast = CONTRASTS[0]

    block_nights = [night for night in schedule.nights if night.arm != "washout"]
    known = {night.night_id for night in block_nights}
    by_night: Dict[str, Nof1NightInputs] = {}
    for item in nights:
        if item.night_id not in known:
            raise ValueError(f"night {item.night_id} is not a block night of schedule {schedule.schedule_id}")
        if item.night_id in by_night:
            raise ValueError(f"night {item.night_id} is given twice")
        by_night[item.night_id] = item
    first_night_id = block_nights[0].night_id if block_nights else None
    rows = [
        _night_row(night, by_night.get(night.night_id) or Nof1NightInputs(night.night_id), night.night_id == first_night_id)
        for night in block_nights
    ]
    synthetic = any(row["synthetic"] for row in rows)
    analysed = [row for row in rows if row["has_data"]]
    arms = tuple(schedule.arms)

    design = _design(schedule)
    relabelings, method = _relabelings(design, seed=seed, permutations=permutations)
    differences = _differences(rows, design, relabelings, arms, primary_outcome, primary_contrast)
    can_exclude = outside_possible(len(relabelings))

    report_schedule = {
        "schedule_id": schedule.schedule_id,
        "created_at_utc": schedule.created_at_utc,
        "seed": schedule.seed,
        "blocks": schedule.blocks,
        "washout_nights": schedule.washout_nights,
        "arms": list(arms),
        "sham_cue_id": schedule.sham_cue_id,
        "night_count": len(schedule.nights),
        "commitment_sha256": schedule.commitment_sha256(),
        "seal": seal,
        "unsealed_at_utc": schedule.unsealed_at_utc,
        "protocol_effective_hashes": sorted(
            {str(row["protocol_effective_hash"]) for row in rows if row.get("protocol_effective_hash")}
        ),
    }
    settings = {
        "seed": seed,
        "permutations": permutations,
        "interval": INTERVAL_LEVEL,
        "min_nights_per_arm": min_nights_per_arm,
        "primary_outcome": primary_outcome,
        "primary_contrast": primary_contrast,
        "reference_set": design.reference_set,
        "possible_orders": design.size,
        "method": method,
        "relabelings": len(relabelings),
        "outside_possible": can_exclude,
        "software_version": __version__,
    }
    return Nof1AnalysisReport(
        schedule=report_schedule,
        settings=settings,
        nights=tuple(_public_row(row) for row in rows),
        descriptives=_descriptives(rows, arms),
        differences=tuple(differences),
        autocorrelation=_autocorrelation(rows),
        cue_counts=_cue_counts(rows, arms),
        arousal=_arousal(rows, arms),
        limitations=_limitations(
            schedule,
            rows,
            analysed,
            arms=arms,
            min_nights_per_arm=min_nights_per_arm,
            synthetic=synthetic,
            possible_orders=design.size,
            can_exclude=can_exclude,
            outcome_count=len(OUTCOMES),
        ),
        synthetic=synthetic,
        analysis_id=analysis_id,
        generated_at_utc=generated_at_utc,
    )


def load_series_inputs(
    schedule: NightSchedule,
    *,
    protocol_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    recordings_dir: Optional[Path] = None,
    summary_paths: Sequence[Path] = (),
    coding_paths: Sequence[Path] = (),
) -> Tuple[Nof1NightInputs, ...]:
    """Every block night's inputs, found by the night ID the runbook names them by.

    Puzzle sessions and assignments are the files in ``protocol_dir`` whose
    ``session_id`` is a night ID; dream reports, retests (one per wave), sound checks
    and coding sheets the files in ``reports_dir`` with such a ``session_id``; night
    summaries the ``pilot4_summary.json``/``pilot5_summary.json`` under
    ``recordings_dir`` whose ``night_id`` and ``schedule_commitment_sha256`` are this
    series', and ``summary_paths`` (Pilot 4/5 summaries or Pilot 3 reports), which must
    be. Two files of one kind for one night raise, so a night is never picked by chance.
    """
    night_ids = [night.night_id for night in schedule.nights if night.arm != "washout"]
    wanted = set(night_ids)
    commitment = schedule.commitment_sha256()
    found: Dict[str, Dict[str, object]] = {night_id: {"retests": {}, "codings": {}} for night_id in night_ids}
    paths: Dict[Tuple[str, str], Path] = {}

    def put(night_id: str, kind: str, value: object, path: Path) -> None:
        key = (night_id, kind)
        if key in paths and paths[key].resolve() != path.resolve():
            raise ValueError(f"two {kind.replace('_', ' ')} files for {night_id}: {paths[key]} and {path}")
        paths[key] = path
        found[night_id][kind] = value

    for path, payload in _json_files(protocol_dir):
        night_id = payload.get("session_id")
        if night_id not in wanted:
            continue
        if "puzzle_ids" in payload:
            put(str(night_id), "session", NightPuzzleSession.from_dict(payload), path)
        elif "cued_puzzle_ids" in payload:
            put(str(night_id), "assignment", PuzzleCueAssignment.from_dict(payload), path)

    coding_files = []
    for path, payload in _json_files(reports_dir):
        night_id = payload.get("session_id")
        if night_id not in wanted:
            continue
        night_id = str(night_id)
        if payload.get("kind") == CODING_KIND:
            coding_files.append((path, payload))
        elif payload.get("kind") is not None:
            continue
        elif "dream_text" in payload and "lucid" in payload:
            put(night_id, "dream_report", DreamReport.from_dict(payload), path)
        elif "results" in payload and "retest_id" in payload:
            retest = MorningRetest.from_dict(payload)
            put(night_id, f"retest_{retest.retest_wave}", retest, path)
            found[night_id]["retests"][retest.retest_wave] = retest
        elif "trials" in payload and "order_seed" in payload:
            put(night_id, "cue_recall", CueRecallTest.from_dict(payload), path)
    explicit_codings = []
    for path in coding_paths:
        path = Path(path).expanduser()
        sheet = IncorporationCodingSheet.load(path)
        if sheet.session_id not in wanted:
            raise ValueError(f"{path} codes {sheet.session_id}, which is not a block night of {schedule.schedule_id}")
        explicit_codings.append((path, sheet.to_dict()))
    for path, payload in coding_files + explicit_codings:
        night_id = payload.get("session_id")
        if night_id not in wanted:
            continue
        sheet = IncorporationCodingSheet.from_dict(payload)
        codings = found[str(night_id)]["codings"]  # type: Dict[str, Tuple[Path, IncorporationCodingSheet]]
        earlier = codings.get(sheet.rater_id)
        if earlier is not None and earlier[0].resolve() != path.resolve():
            raise ValueError(f"two coding sheets of {sheet.rater_id} for {night_id}: {earlier[0]} and {path}")
        codings[sheet.rater_id] = (path, sheet)

    explicit = []
    for path in summary_paths:
        path = Path(path).expanduser()
        payload = _read_json(path)
        if payload.get("night_id") not in wanted or payload.get("schedule_commitment_sha256") != commitment:
            raise ValueError(
                f"{path} is not a night of schedule {schedule.schedule_id}: night_id={payload.get('night_id')!r}, "
                "or its schedule_commitment_sha256 is another series'"
            )
        explicit.append((path, payload))
    named = {str(payload["night_id"]) for _, payload in explicit}
    for path, payload in explicit:
        put(str(payload["night_id"]), "summary", (path, payload), path)
    if recordings_dir is not None and Path(recordings_dir).expanduser().is_dir():
        for path in sorted(Path(recordings_dir).expanduser().rglob("*.json")):
            if path.name not in SUMMARY_FILENAMES:
                continue
            payload = _read_json_or_none(path)
            if not payload or payload.get("night_id") not in wanted or payload.get("night_id") in named:
                continue
            if payload.get("schedule_commitment_sha256") != commitment:
                continue
            put(str(payload["night_id"]), "summary", (path, payload), path)

    inputs = []
    for night_id in night_ids:
        item = found[night_id]
        summary_path, summary = item.get("summary") or (None, {})
        night_dir = Path(summary_path).parent if summary_path is not None else None
        arousal, arousal_source, arousal_synthetic = _arousal_of(summary, night_dir)
        inputs.append(
            Nof1NightInputs(
                night_id=night_id,
                summary=summary,
                summary_path=str(summary_path) if summary_path is not None else None,
                session=item.get("session"),
                assignment=item.get("assignment"),
                dream_report=item.get("dream_report"),
                retests=tuple(item["retests"].values()),
                cue_recall=item.get("cue_recall"),
                codings=tuple(sheet for _, sheet in item["codings"].values()),
                arousal_proxy_per_cue=arousal,
                arousal_source=arousal_source,
                awakening_count=_awakening_count(night_dir),
                synthetic=bool(summary.get("synthetic") is True or arousal_synthetic or _metadata_synthetic(night_dir)),
            )
        )
    return tuple(inputs)


def checklist_rows() -> List[Dict[str, str]]:
    return [
        {"scribe": scribe, "cent": cent, "item": item, "status": status, "filled_by": how}
        for scribe, cent, item, status, how in REPORTING_CHECKLIST
    ]


def checklist_counts() -> Dict[str, int]:
    return {status: sum(1 for row in REPORTING_CHECKLIST if row[3] == status) for status in CHECKLIST_STATUSES}


def schedule_design_orders(blocks: int, arms: Sequence[str]) -> int:
    """How many arm orders ``plan-night-schedule`` can draw for ``blocks`` blocks."""
    return _product(len(_latin_group_options(tuple(arms), size)) for size in _group_sizes(blocks, len(arms)))


def outside_possible(relabelings: int) -> bool:
    """Whether a difference can fall outside an interval over ``relabelings`` orders:
    under 40 nothing is cut from either end."""
    return _interval_cut(relabelings) >= 1


# --------------------------------------------------------------------------- #
# Nights
# --------------------------------------------------------------------------- #


def _night_row(night: ScheduledNight, inputs: Nof1NightInputs, first: bool) -> Row:
    summary = dict(inputs.summary)
    values: Dict[str, Optional[float]] = {name: None for name in OUTCOMES}
    report = inputs.dream_report
    if report is not None:
        values["lucid"] = 1.0 if report.lucid else 0.0
        if report.cue_perception != "unknown":
            values["cue_perceived"] = 1.0 if report.cue_perception in HEARD_PERCEPTIONS else 0.0
        if report.woke_from_sound != "unknown":
            values["woke_from_sound"] = 1.0 if report.woke_from_sound == "yes" else 0.0
    if inputs.awakening_count is not None:
        values["awakening_markers"] = float(inputs.awakening_count)
    incorporation_source = None
    if inputs.session is not None and (report is not None or inputs.codings):
        try:
            if report is not None:
                report.validate_against_session(inputs.session)
            per_puzzle, summary_block = incorporation_by_puzzle(
                inputs.session, dream_report=report, incorporation_codings=inputs.codings
            )
        except ValueError as exc:
            raise ValueError(f"night {night.night_id}: {exc}") from None
        incorporation_source = summary_block["source"]
        values["incorporation_rate"] = sum(1 for item in per_puzzle.values() if item["incorporated"]) / len(per_puzzle)
    if inputs.session is not None and inputs.assignment is not None and inputs.retests:
        try:
            analysis = build_cued_uncued_analysis(
                inputs.session,
                inputs.assignment,
                inputs.retests,
                dream_report=report,
                incorporation_codings=inputs.codings,
                min_group_size=1,
            )
        except ValueError as exc:
            raise ValueError(f"night {night.night_id}: {exc}") from None
        for wave, block in analysis.wave_metrics.items():
            delta = dict(block.get("effect_summary", {}) or {}).get("solve_rate_difference_cued_minus_uncued")
            values[_wave_outcome(wave)] = None if delta is None else float(delta)
    pilot3 = summary.get("pilot_id") == "m8_pilot3_replay_cue_simulation"
    metrics = dict(summary.get("metrics", {}) or {})
    plays = summary.get("cue_play_count", metrics.get("cue_plan_count")) if summary else None
    virtual = summary.get("virtual_play_count", metrics.get("virtual_play_count")) if summary else None
    protocol = protocol_block_of(summary) if summary else None
    return {
        "night_id": night.night_id,
        "position": night.position,
        "block": night.block,
        "arm": night.arm,
        "first_night": first,
        "has_data": inputs.has_data,
        "synthetic": inputs.synthetic,
        "summary_arm": summary.get("arm") if summary else None,
        "summary_kind": ("pilot3_report" if pilot3 else "pilot_summary") if summary else None,
        "started_at": summary.get("started_at") if summary else None,
        "summary_passed": summary.get("passed") if summary else None,
        "failed_criteria": list(summary.get("failed_criteria", ()) or ()) if summary else [],
        "protocol_effective_hash": protocol.get("effective_hash") if protocol else None,
        "plays": None if plays is None else int(plays),
        "virtual_plays": None if virtual is None else int(virtual),
        "inputs": {
            "summary": inputs.summary_path,
            "session": inputs.session is not None,
            "assignment": inputs.assignment is not None,
            "dream_report": report is not None,
            "retest_waves": list(inputs.retest_waves),
            "cue_recall": inputs.cue_recall is not None,
            "rater_ids": sorted(sheet.rater_id for sheet in inputs.codings),
            "arousal_scoring": inputs.arousal_source,
            "awakening_markers": inputs.awakening_count is not None,
        },
        "incorporation_source": incorporation_source,
        "arousal_proxy_per_cue": dict(inputs.arousal_proxy_per_cue) if inputs.arousal_proxy_per_cue else None,
        "values": values,
        "_form_times": _form_times(inputs),
    }


def _public_row(row: Row) -> Dict[str, object]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _form_times(inputs: Nof1NightInputs) -> List[Tuple[str, str]]:
    """(form, saved at) of every morning form and retest of the night."""
    times = []
    if inputs.dream_report is not None:
        times.append(("dream_report", inputs.dream_report.reported_at_utc))
    if inputs.cue_recall is not None:
        times.append(("cue_recall", inputs.cue_recall.completed_at_utc))
    for retest in inputs.retests:
        times.append((f"retest_{retest.retest_wave}", retest.reported_at_utc))
    return times


# --------------------------------------------------------------------------- #
# Descriptives, permutation intervals, autocorrelation
# --------------------------------------------------------------------------- #


def _descriptives(rows: Sequence[Row], arms: Sequence[str]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for outcome in OUTCOMES:
        by_arm: Dict[str, object] = {}
        for arm in arms:
            values = [row["values"][outcome] for row in rows if row["arm"] == arm]
            values = [value for value in values if value is not None]
            later = [
                row["values"][outcome]
                for row in rows
                if row["arm"] == arm and not row["first_night"] and row["values"][outcome] is not None
            ]
            by_arm[arm] = {
                "nights": len(values),
                "mean": _mean(values),
                "nights_without_first_night": len(later),
                "mean_without_first_night": _mean(later),
            }
        result[outcome] = by_arm
    return result


@dataclass(frozen=True)
class _Design:
    """The block nights by block and place, and the orders a relabeling draws from."""

    night_ids: Tuple[Tuple[str, ...], ...]
    observed: Tuple[Tuple[str, ...], ...]
    groups: Tuple[Tuple[Tuple[Tuple[str, ...], ...], ...], ...]
    reference_set: str
    size: int


def _design(schedule: NightSchedule) -> _Design:
    arms = tuple(schedule.arms)
    blocks: Dict[int, List[ScheduledNight]] = {}
    for night in schedule.nights:
        if night.block is not None:
            blocks.setdefault(night.block, []).append(night)
    ordered = [sorted(blocks[number], key=lambda night: night.position) for number in sorted(blocks)]
    night_ids = tuple(tuple(night.night_id for night in block) for block in ordered)
    observed = tuple(tuple(night.arm for night in block) for block in ordered)
    groups: List[Tuple[Tuple[Tuple[str, ...], ...], ...]] = []
    in_design = True
    start = 0
    for size in _group_sizes(len(ordered), len(arms)):
        options = _latin_group_options(arms, size)
        groups.append(options)
        in_design = in_design and observed[start : start + size] in set(options)
        start += size
    if in_design:
        reference_set = "schedule_design"
    else:
        # A schedule plan-night-schedule could not have drawn: every order of every block.
        reference_set = "within_block"
        everything = tuple((order,) for order in itertools.permutations(arms))
        groups = [everything for _ in ordered]
    return _Design(
        night_ids=night_ids,
        observed=observed,
        groups=tuple(groups),
        reference_set=reference_set,
        size=_product(len(options) for options in groups),
    )


def _group_sizes(blocks: int, arm_count: int) -> List[int]:
    sizes = []
    while sum(sizes) < blocks:
        sizes.append(min(arm_count, blocks - sum(sizes)))
    return sizes


def _latin_group_options(arms: Tuple[str, ...], size: int) -> Tuple[Tuple[Tuple[str, ...], ...], ...]:
    """Every tuple of ``size`` block orders ``night_schedule._block_orders`` can give a
    group: distinct rows, in any order, of one cyclic Latin square of the arms."""
    options: List[Tuple[Tuple[str, ...], ...]] = []
    for rest in itertools.permutations(arms[1:]):
        base = (arms[0],) + tuple(rest)
        rows = [base[shift:] + base[:shift] for shift in range(len(base))]
        options.extend(tuple(selection) for selection in itertools.permutations(rows, size))
    return tuple(options)


def _relabelings(design: _Design, *, seed: int, permutations: int) -> Tuple[List[Tuple[Tuple[str, ...], ...]], str]:
    if design.size <= permutations:
        return (
            [tuple(order for option in combo for order in option) for combo in itertools.product(*design.groups)],
            "exact",
        )
    rng = random.Random(seed)
    return (
        [tuple(order for options in design.groups for order in rng.choice(options)) for _ in range(permutations)],
        "monte_carlo",
    )


def _differences(
    rows: Sequence[Row],
    design: _Design,
    relabelings: Sequence[Tuple[Tuple[str, ...], ...]],
    arms: Sequence[str],
    primary_outcome: Optional[str],
    primary_contrast: Optional[str],
) -> List[Dict[str, object]]:
    by_night = {row["night_id"]: row for row in rows}
    contrasts = [contrast for contrast in CONTRASTS if all(arm in arms for arm in contrast.split("-"))]
    results = []
    for outcome in OUTCOMES:
        values = tuple(
            tuple(by_night[night_id]["values"][outcome] for night_id in block)
            for block in design.night_ids
        )
        for contrast in contrasts:
            first, second = contrast.split("-")
            observed = _contrast(values, design.observed, first, second)
            permuted = [
                value
                for value in (_contrast(values, orders, first, second) for orders in relabelings)
                if value is not None
            ]
            interval = _interval(permuted) if permuted and observed is not None else None
            results.append(
                {
                    "outcome": outcome,
                    "contrast": contrast,
                    "difference": observed,
                    "nights": {
                        first: sum(1 for row in rows if row["arm"] == first and row["values"][outcome] is not None),
                        second: sum(1 for row in rows if row["arm"] == second and row["values"][outcome] is not None),
                    },
                    "interval": list(interval) if interval else None,
                    "observed_outside_interval": (
                        None if interval is None else bool(observed < interval[0] or observed > interval[1])
                    ),
                    "relabelings": len(permuted) if observed is not None else 0,
                    "primary": outcome == primary_outcome and contrast == primary_contrast,
                }
            )
    return results


def _contrast(
    values: Tuple[Tuple[Optional[float], ...], ...],
    orders: Tuple[Tuple[str, ...], ...],
    first: str,
    second: str,
) -> Optional[float]:
    """Mean of ``first`` minus mean of ``second`` under the arm ``orders``; None when
    either arm has no value."""
    sums = {first: 0.0, second: 0.0}
    counts = {first: 0, second: 0}
    for block_values, order in zip(values, orders):
        for value, arm in zip(block_values, order):
            if value is not None and arm in sums:
                sums[arm] += value
                counts[arm] += 1
    if not counts[first] or not counts[second]:
        return None
    return round(sums[first] / counts[first] - sums[second] / counts[second], _ROUND)


def _interval_cut(count: int) -> int:
    """How many differences each end of the interval leaves out."""
    return int(math.floor(count * (1.0 - INTERVAL_LEVEL) / 2.0 + 1e-9))


def _interval(values: Sequence[float]) -> Tuple[float, float]:
    """The range of ``values`` once the lowest and the highest 2.5 % are cut; with fewer
    than 40 values nothing is cut."""
    ordered = sorted(values)
    cut = _interval_cut(len(ordered))
    return ordered[cut], ordered[len(ordered) - 1 - cut]


def _autocorrelation(rows: Sequence[Row]) -> Dict[str, object]:
    """Lag-1 autocorrelation of each outcome over consecutive nights once the arm means are
    taken out; a washout night or a night without a value breaks the pair."""
    result: Dict[str, object] = {}
    for outcome in OUTCOMES:
        present = [
            (int(row["position"]), str(row["arm"]), float(row["values"][outcome]))
            for row in rows
            if row["values"][outcome] is not None
        ]
        means = {
            arm: _mean([value for _, other, value in present if other == arm])
            for arm in {arm for _, arm, _ in present}
        }
        residuals = {position: value - float(means[arm]) for position, arm, value in present}
        pairs = [(residuals[position], residuals[position + 1]) for position in residuals if position + 1 in residuals]
        lag1 = None
        if len(residuals) >= 3 and pairs:
            center = sum(residuals.values()) / len(residuals)
            denominator = sum((value - center) ** 2 for value in residuals.values())
            if denominator > 1e-12:
                lag1 = round(
                    sum((left - center) * (right - center) for left, right in pairs) / denominator, _ROUND
                )
        result[outcome] = {"lag1": lag1, "pairs": len(pairs), "nights": len(residuals)}
    return result


def _cue_counts(rows: Sequence[Row], arms: Sequence[str]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for arm in arms:
        with_summary = [row for row in rows if row["arm"] == arm and row["plays"] is not None]
        plays = [int(row["plays"]) for row in with_summary]
        virtual = [int(row["virtual_plays"]) for row in with_summary if row["virtual_plays"] is not None]
        result[arm] = {
            "nights_with_summary": len(with_summary),
            "plays": sum(plays) if with_summary else None,
            "plays_per_night": _mean([float(value) for value in plays]),
            "virtual_plays": sum(virtual) if virtual else (0 if with_summary else None),
        }
    return result


def _arousal(rows: Sequence[Row], arms: Sequence[str]) -> Dict[str, object]:
    """Initiative 3's arousal proxies per cue, averaged over each arm's scored nights.
    Only nights that played sound have them, so this stays descriptive."""
    result: Dict[str, object] = {}
    for arm in arms:
        scored = [dict(row["arousal_proxy_per_cue"]) for row in rows if row["arm"] == arm and row["arousal_proxy_per_cue"]]
        flags = sorted({flag for item in scored for flag in item})
        result[arm] = {
            "nights_scored": len(scored),
            "mean_rate": {
                flag: _mean([float(item[flag]) for item in scored if item.get(flag) is not None]) for flag in flags
            },
        }
    return result


# --------------------------------------------------------------------------- #
# Limitations
# --------------------------------------------------------------------------- #


def _limitations(
    schedule: NightSchedule,
    rows: Sequence[Row],
    analysed: Sequence[Row],
    *,
    arms: Sequence[str],
    min_nights_per_arm: int,
    synthetic: bool,
    possible_orders: int,
    can_exclude: bool,
    outcome_count: int,
) -> Tuple[AnalysisLimitation, ...]:
    items: List[AnalysisLimitation] = []
    if synthetic:
        items.append(
            AnalysisLimitation(
                code="synthetic_input",
                severity="info",
                message="At least one night is synthetic input, not sleep: nothing here is about sleep.",
            )
        )
    items.append(
        AnalysisLimitation(
            code="descriptive_only",
            severity="info",
            message=(
                "A single-case description: no p-value is computed, and an interval is the spread of the "
                f"differences over the arm orders the schedule could have drawn. With {outcome_count} outcomes and "
                "three contrasts some differences may fall outside their interval by chance; only a primary one "
                "named in advance means more than that, and none is evidence that cueing works."
            ),
        )
    )
    items.append(
        AnalysisLimitation(
            code="single_subject",
            severity="info",
            message=(
                "One participant, who is also the experimenter and cannot be blind; the series says nothing about "
                "anyone else."
            ),
        )
    )
    counts = {arm: sum(1 for row in analysed if row["arm"] == arm) for arm in arms}
    if not counts.get("sham"):
        items.append(
            AnalysisLimitation(
                code="no_sham_nights",
                severity="warning",
                message="No sham night has data, so a cue night cannot be told from any night with sound.",
            )
        )
    if not counts.get("no_cue"):
        items.append(
            AnalysisLimitation(
                code="no_no_cue_nights",
                severity="warning",
                message="No no-cue night has data, so there is no night without sound to compare with.",
            )
        )
    if len(set(counts.values())) > 1:
        items.append(
            AnalysisLimitation(
                code="unbalanced_arms",
                severity="warning",
                message="The arms have different numbers of nights with data.",
                metadata={"nights_with_data": counts},
            )
        )
    small = {arm: count for arm, count in counts.items() if count < min_nights_per_arm}
    if small:
        items.append(
            AnalysisLimitation(
                code="small_n",
                severity="warning",
                message=(
                    f"Fewer than {min_nights_per_arm} nights with data in some arms. The schedule's "
                    f"{schedule.blocks} blocks allow {possible_orders} arm orders"
                    + ("; that is too few for any difference to fall outside its interval." if not can_exclude else ".")
                ),
                metadata={
                    "nights_with_data": small,
                    "min_nights_per_arm": min_nights_per_arm,
                    "possible_orders": possible_orders,
                    "outside_possible": can_exclude,
                },
            )
        )
    missing: Dict[str, List[str]] = {}
    for row in rows:
        inputs = dict(row["inputs"])
        gaps = [
            name
            for name, present in (
                ("summary", inputs["summary"]),
                ("dream_report", inputs["dream_report"]),
                ("morning_retest", "morning" in inputs["retest_waves"]),
            )
            if not present
        ]
        if gaps:
            missing[str(row["night_id"])] = gaps
    if missing:
        items.append(
            AnalysisLimitation(
                code="missing_night_data",
                severity="warning",
                message="Some nights lack their summary, dream report or morning retest; their values are missing.",
                metadata={"missing": missing},
            )
        )
    mismatched = {
        str(row["night_id"]): {"scheduled": row["arm"], "summary": row["summary_arm"]}
        for row in rows
        if row["summary_arm"] is not None and row["summary_arm"] != row["arm"]
    }
    if mismatched:
        items.append(
            AnalysisLimitation(
                code="arm_mismatch",
                severity="error",
                message=(
                    "A night's summary names another arm than the sealed schedule; the analysis keeps the "
                    "schedule's arm, as randomised."
                ),
                metadata={"nights": mismatched},
            )
        )
    unblinded = _unblinded(schedule, rows)
    if unblinded:
        items.append(
            AnalysisLimitation(
                code="unblinded_early",
                severity="warning",
                message=(
                    "The schedule was unsealed before the series' last morning form or delayed retest was saved, "
                    "or before a night started: what came after was not blind."
                ),
                metadata=unblinded,
            )
        )
    sham_rows = [row for row in analysed if row["arm"] == "sham"]
    if sham_rows:
        items.append(
            AnalysisLimitation(
                code="sham_detectable",
                severity="warning",
                message=(
                    "A sham night still makes sound, and the sleeper may tell the untrained sound from a trained "
                    "cue, in the night or from the sound check."
                ),
                metadata={
                    "sham_nights": len(sham_rows),
                    "sham_nights_sound_heard": sum(1 for row in sham_rows if row["values"]["cue_perceived"] == 1.0),
                    "sham_nights_woke_from_sound": sum(
                        1 for row in sham_rows if row["values"]["woke_from_sound"] == 1.0
                    ),
                },
            )
        )
    positions = {int(row["position"]): row for row in analysed}
    after_playing = [
        str(row["night_id"])
        for position, row in sorted(positions.items())
        if position - 1 in positions and positions[position - 1]["arm"] in PLAYING_ARMS
    ]
    if any(position + 1 in positions for position in positions):
        items.append(
            AnalysisLimitation(
                code="carry_over_possible",
                severity="warning",
                message=(
                    f"Nights follow each other inside a block, with {schedule.washout_nights} washout night(s) "
                    "only between blocks, so one night's arm may carry into the next."
                ),
                metadata={"washout_nights": schedule.washout_nights, "nights_after_a_playing_night": after_playing},
            )
        )
    first = next((row for row in rows if row["first_night"]), None)
    if first is not None and first["has_data"]:
        items.append(
            AnalysisLimitation(
                code="first_night_effect_possible",
                severity="info",
                message="The series' first night may differ from the rest; the means without it are given next to each.",
                metadata={"night_id": first["night_id"], "arm": first["arm"]},
            )
        )
    delayed_missing = {
        str(row["night_id"]): [wave for wave in DELAYED_WAVES if wave not in dict(row["inputs"])["retest_waves"]]
        for row in analysed
    }
    delayed_missing = {night: waves for night, waves in delayed_missing.items() if waves}
    if delayed_missing:
        items.append(
            AnalysisLimitation(
                code="missing_delayed_retest",
                severity="warning",
                message=(
                    "Some nights lack a retest a week or ten days later; cueing effects have shown up only at a "
                    "delayed retest in some studies. A missing wave is not a zero."
                ),
                metadata={"missing_waves": delayed_missing},
            )
        )
    unscored = [
        str(row["night_id"])
        for row in analysed
        if row["arm"] in PLAYING_ARMS and not row["arousal_proxy_per_cue"]
    ]
    if unscored:
        items.append(
            AnalysisLimitation(
                code="missing_arousal_scoring",
                severity="warning",
                message="Some nights that played sound have no arousal proxies per cue (score-cue-arousal).",
                metadata={"night_ids": unscored},
            )
        )
    items.append(
        AnalysisLimitation(
            code="inherits_detector_limits",
            severity="info",
            message=(
                "Every arm comparison inherits the REM gate's open issues: the detector's weights (#116) and the "
                "first cue's latency after REM onset (#167)."
            ),
        )
    )
    return tuple(items)


def _unblinded(schedule: NightSchedule, rows: Sequence[Row]) -> Dict[str, object]:
    unsealed = _parse_time(schedule.unsealed_at_utc)
    if unsealed is None:
        return {}
    later_forms = []
    last: Optional[dt.datetime] = None
    for row in rows:
        for form, saved in row["_form_times"]:
            when = _parse_time(saved)
            if when is None:
                continue
            last = when if last is None or when > last else last
            if when > unsealed:
                later_forms.append({"night_id": row["night_id"], "form": form, "saved_at_utc": saved})
    later_nights = [
        {"night_id": row["night_id"], "started_at": row["started_at"]}
        for row in rows
        if _parse_time(row["started_at"]) is not None and _parse_time(row["started_at"]) > unsealed
    ]
    if not later_forms and not later_nights:
        return {}
    return {
        "unsealed_at_utc": schedule.unsealed_at_utc,
        "last_form_at_utc": last.isoformat() if last is not None else None,
        "forms_after_unsealing": later_forms,
        "nights_after_unsealing": later_nights,
    }


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def _arousal_of(
    summary: Mapping[str, object],
    night_dir: Optional[Path],
) -> Tuple[Optional[Mapping[str, object]], Optional[str], bool]:
    """``arousal_proxy_per_cue`` from the summary's scored ``cue_arousal`` block, else from
    the ``cue_arousal_report.json`` next to it; and whether that report is synthetic."""
    block = summary.get("cue_arousal") if summary else None
    if isinstance(block, Mapping) and block.get("status") == "scored":
        rates = block.get("arousal_proxy_per_cue")
        if isinstance(rates, Mapping):
            return dict(rates), "summary", block.get("synthetic") is True
    payload = _read_json_or_none(night_dir / AROUSAL_REPORT_FILENAME) if night_dir is not None else None
    outcomes = payload.get("outcomes") if payload else None
    if payload and isinstance(outcomes, Mapping) and isinstance(outcomes.get("arousal_proxy_per_cue"), Mapping):
        return dict(outcomes["arousal_proxy_per_cue"]), AROUSAL_REPORT_FILENAME, payload.get("synthetic") is True
    return None, None, False


def _awakening_count(night_dir: Optional[Path]) -> Optional[int]:
    """Awakening markers in the night's ``awakening_events.jsonl``; None without the file."""
    if night_dir is None:
        return None
    path = night_dir / AWAKENING_EVENTS_FILENAME
    if not path.is_file():
        return None
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping) and payload.get("event_type", "awakening") == "awakening":
            count += 1
    return count


def _metadata_synthetic(night_dir: Optional[Path]) -> bool:
    if night_dir is None:
        return False
    payload = _read_json_or_none(night_dir / "metadata.json")
    return bool(payload) and payload.get("synthetic") is True


def _json_files(directory: Optional[Path]) -> List[Tuple[Path, Mapping[str, object]]]:
    if directory is None:
        return []
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        return []
    files = []
    for path in sorted(directory.glob("*.json")):
        payload = _read_json_or_none(path)
        if payload:
            files.append((path, payload))
    return files


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} holds no JSON object")
    return payload


def _read_json_or_none(path: Path) -> Optional[Mapping[str, object]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _write_text(output_path: Path, text: str) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _parse_time(value: object) -> Optional[dt.datetime]:
    """An ISO time as UTC; without an offset it is read as UTC. None when it is not one."""
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.timezone.utc)


def _mean(values: Sequence[float]) -> Optional[float]:
    values = [float(value) for value in values]
    if not values:
        return None
    return round(sum(values) / len(values), _ROUND)


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _join(values: object) -> str:
    return ", ".join(f"`{value}`" for value in (values or ()))


def _number(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.2f}"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
