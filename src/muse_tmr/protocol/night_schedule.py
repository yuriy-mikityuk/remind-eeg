"""Sealed night schedules for an N-of-1 series of cue, sham and no-cue nights (#228).

A series is ``blocks`` blocks with one night of each arm, in a counterbalanced order, and
``washout_nights`` washout nights between two blocks:

- ``cue``: the night plays its cues as planned, as every night did before arms existed.
- ``sham``: an untrained sound plays wherever a cue would, at the same times, lengths
  and volumes.
- ``no_cue``: nothing plays; each play the scheduler would make is logged as a ``skip``
  with ``metadata.virtual_play``.
- ``washout``: a night between blocks without a cueing run. Its place follows from
  ``blocks`` and ``washout_nights``, so it is not sealed.

``generate_night_schedule()`` draws the order with ``random.Random(seed)`` and the salt
with ``secrets``, so one seed gives one order. Within each group of three blocks every
arm takes every place in the block once (the rows of a Latin square), so no arm is
always the first night of its block.

The schedule, with its seed and salt, stays in the gitignored ``data/protocol/``. Its
sealed companion holds only ``commitment_sha256``: the sha256 of the schedule's
canonical JSON (``committed_payload()``: sorted keys, no spaces) followed by the salt. It
can be shared before the series without giving an arm away, and ``unseal()`` checks the
schedule against it before it records ``unsealed_at_utc``.

The runners read a night's arm with ``arm_for()`` and never print it, and the app's
Morning tab never reads a schedule. A night's summary, logs and reports do name it, so
the runbook (docs/nof1_protocol.md) keeps them closed until the series is unsealed:
nothing hides the arms from whoever opens those files, and at n = 1 the experimenter
cannot be blind.

``prereg_template()`` renders a preregistration of the series from its design, its
commitment and the protocol block, with no arm, no seed and no salt, so it can be filed
before the first night.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import re
import secrets
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from muse_tmr.audio.cue_library import DEFAULT_SHAM_CUE_ID

NIGHT_SCHEDULE_SCHEMA_VERSION = 1
NIGHT_ARMS = ("cue", "sham", "no_cue", "washout")
# The arms of a block, which are also the arms a cueing night can run.
BLOCK_ARMS = ("cue", "sham", "no_cue")
CUEING_ARMS = BLOCK_ARMS
# The arms in which the player makes a sound.
PLAYING_ARMS = ("cue", "sham")
SEALED_SUFFIX = ".sealed.json"
DEFAULT_NIGHT_PREFIX = "night-"
_SALT_BYTES = 16
_HEX = re.compile(r"^[0-9a-f]+$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SealMismatch(ValueError):
    """A schedule does not match the commitment in its sealed companion."""


class WashoutNight(ValueError):
    """The night is a washout night of its schedule: no cueing night is planned."""


@dataclass(frozen=True)
class ScheduledNight:
    """One night of a series: its place, its block (None for washout) and its arm."""

    night_id: str
    position: int
    arm: str
    block: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "night_id", _identifier(self.night_id, "night_id"))
        if self.position <= 0:
            raise ValueError("position must be positive")
        if self.arm not in NIGHT_ARMS:
            raise ValueError(f"arm must be one of: {', '.join(NIGHT_ARMS)}")
        if self.arm == "washout":
            if self.block is not None:
                raise ValueError("a washout night belongs to no block")
        elif self.block is None or self.block <= 0:
            raise ValueError("a block night needs a positive block number")

    def to_dict(self) -> Dict[str, object]:
        return {"night_id": self.night_id, "position": self.position, "block": self.block, "arm": self.arm}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ScheduledNight":
        block = payload.get("block")
        return cls(
            night_id=str(payload["night_id"]),
            position=int(payload["position"]),
            arm=str(payload["arm"]),
            block=None if block is None else int(block),
        )


@dataclass(frozen=True)
class NightSchedule:
    """The arm of every night of a series, the seed that drew them and the salt."""

    schedule_id: str
    seed: int
    blocks: int
    washout_nights: int
    nights: Tuple[ScheduledNight, ...]
    salt: str
    sham_cue_id: str = DEFAULT_SHAM_CUE_ID
    arms: Tuple[str, ...] = BLOCK_ARMS
    created_at_utc: str = ""
    unsealed_at_utc: Optional[str] = None
    schema_version: int = NIGHT_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule_id", _identifier(self.schedule_id, "schedule_id"))
        object.__setattr__(self, "sham_cue_id", _identifier(self.sham_cue_id, "sham_cue_id"))
        object.__setattr__(self, "nights", tuple(self.nights))
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "salt", str(self.salt).strip().lower())
        if not self.created_at_utc:
            object.__setattr__(self, "created_at_utc", _utc_now())
        if self.blocks <= 0:
            raise ValueError("blocks must be positive")
        if self.washout_nights < 0:
            raise ValueError("washout_nights must be non-negative")
        if not self.arms or len(set(self.arms)) != len(self.arms) or not set(self.arms) <= set(BLOCK_ARMS):
            raise ValueError(f"arms must be distinct arms out of: {', '.join(BLOCK_ARMS)}")
        if len(self.salt) < 2 * _SALT_BYTES or not _HEX.match(self.salt):
            raise ValueError(f"salt must be at least {2 * _SALT_BYTES} hex characters")
        night_ids = [night.night_id for night in self.nights]
        if len(set(night_ids)) != len(night_ids):
            raise ValueError("night IDs must be unique")
        if [night.position for night in self.nights] != list(range(1, len(self.nights) + 1)):
            raise ValueError("night positions must run 1, 2, 3, ... in order")
        if [night.block for night in self.nights] != _block_layout(self.blocks, self.washout_nights, len(self.arms)):
            raise ValueError("the nights must be the blocks in order, with washout_nights washout nights between two blocks")
        for block in range(1, self.blocks + 1):
            block_arms = sorted(night.arm for night in self.nights if night.block == block)
            if block_arms != sorted(self.arms):
                raise ValueError(f"block {block} must hold each arm once: {', '.join(self.arms)}")

    @property
    def night_ids(self) -> Tuple[str, ...]:
        return tuple(night.night_id for night in self.nights)

    @property
    def sealed(self) -> bool:
        """Whether ``unseal()`` has not run on it yet."""
        return self.unsealed_at_utc is None

    def night(self, night_id: str) -> ScheduledNight:
        for night in self.nights:
            if night.night_id == night_id:
                return night
        raise KeyError(f"night {night_id!r} is not in night schedule {self.schedule_id}")

    def arm_for(self, night_id: str) -> str:
        """The arm of ``night_id``; KeyError when the schedule has no such night."""
        return self.night(night_id).arm

    def committed_payload(self) -> Dict[str, object]:
        """What the commitment covers: everything but the salt and ``unsealed_at_utc``."""
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "created_at_utc": self.created_at_utc,
            "seed": self.seed,
            "blocks": self.blocks,
            "washout_nights": self.washout_nights,
            "arms": list(self.arms),
            "sham_cue_id": self.sham_cue_id,
            "nights": [night.to_dict() for night in self.nights],
        }

    def commitment_sha256(self) -> str:
        """sha256 of the canonical JSON of ``committed_payload()`` followed by the salt."""
        text = canonical_json(self.committed_payload()) + self.salt
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def sealed_companion(self) -> Dict[str, str]:
        return {"commitment_sha256": self.commitment_sha256()}

    def to_dict(self) -> Dict[str, object]:
        payload = self.committed_payload()
        payload.update(
            {
                "salt": self.salt,
                "commitment_sha256": self.commitment_sha256(),
                "unsealed_at_utc": self.unsealed_at_utc,
            }
        )
        return payload

    def save(self, output_path: Path) -> Path:
        return _write_json(Path(output_path), self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NightSchedule":
        unsealed = payload.get("unsealed_at_utc")
        return cls(
            schema_version=int(payload.get("schema_version", NIGHT_SCHEDULE_SCHEMA_VERSION)),
            schedule_id=str(payload["schedule_id"]),
            created_at_utc=str(payload.get("created_at_utc", "")),
            seed=int(payload["seed"]),
            blocks=int(payload["blocks"]),
            washout_nights=int(payload.get("washout_nights", 0)),
            arms=tuple(str(arm) for arm in payload.get("arms", BLOCK_ARMS)),
            sham_cue_id=str(payload.get("sham_cue_id", DEFAULT_SHAM_CUE_ID)),
            nights=tuple(ScheduledNight.from_dict(item) for item in payload.get("nights", ())),
            salt=str(payload["salt"]),
            unsealed_at_utc=None if unsealed in (None, "") else str(unsealed),
        )

    @classmethod
    def load(cls, input_path: Path) -> "NightSchedule":
        return cls.from_dict(json.loads(Path(input_path).expanduser().read_text(encoding="utf-8")))


@dataclass(frozen=True)
class NightArmContext:
    """The arm one cueing night runs and where it came from.

    From ``night_arm_from_schedule()`` it carries the schedule's commitment and paths, so
    a run can check at its end that the arm still matches the sealed schedule; from
    ``--arm`` it carries the arm alone, and the night is open label.
    """

    arm: str
    night_id: Optional[str] = None
    sham_cue_id: Optional[str] = None
    schedule_id: Optional[str] = None
    schedule_commitment_sha256: Optional[str] = None
    schedule_unsealed_at_utc: Optional[str] = None
    schedule_path: Optional[str] = None
    sealed_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.arm not in CUEING_ARMS:
            raise ValueError(f"a cueing night runs one of the arms: {', '.join(CUEING_ARMS)}")
        if self.night_id is not None:
            object.__setattr__(self, "night_id", _identifier(self.night_id, "night_id"))
        if self.sham_cue_id is not None:
            object.__setattr__(self, "sham_cue_id", _identifier(self.sham_cue_id, "sham_cue_id"))
        if self.arm == "sham" and not self.sham_cue_id:
            raise ValueError("a sham night needs the cue_id of its untrained sound")
        if self.schedule_path is not None and (
            not self.night_id or not self.schedule_commitment_sha256 or not self.sealed_path
        ):
            raise ValueError("a night from a schedule needs its night_id, commitment and sealed companion")

    @property
    def from_schedule(self) -> bool:
        return self.schedule_path is not None

    @property
    def plays_sound(self) -> bool:
        return self.arm in PLAYING_ARMS

    def summary_fields(self) -> Dict[str, Optional[str]]:
        """The fields a Pilot 3 report and a Pilot 4/5 summary carry."""
        return {
            "arm": self.arm,
            "night_id": self.night_id,
            "schedule_commitment_sha256": self.schedule_commitment_sha256,
        }

    def to_dict(self) -> Dict[str, object]:
        """The ``night_arm`` block of a night's ``metadata.json``."""
        return {
            "arm": self.arm,
            "night_id": self.night_id,
            "sham_cue_id": self.sham_cue_id,
            "sealed_schedule": self.from_schedule,
            "schedule_id": self.schedule_id,
            "schedule_commitment_sha256": self.schedule_commitment_sha256,
            "schedule_unsealed_at_utc": self.schedule_unsealed_at_utc,
        }

    def schedule_check(self) -> Dict[str, object]:
        """Read the schedule and its companion again: does the sealed commitment still
        hold, and does the schedule still give this night this arm?"""
        if not self.from_schedule:
            return {"commitment_matches": False, "arm_matches": False, "error": "no night schedule"}
        try:
            schedule = load_night_schedule(Path(str(self.schedule_path)))
            sealed = load_sealed_commitment(Path(str(self.sealed_path)))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return {"commitment_matches": False, "arm_matches": False, "error": f"{type(exc).__name__}: {exc}"}
        commitment = schedule.commitment_sha256()
        try:
            arm = schedule.arm_for(str(self.night_id))
        except KeyError:
            arm = None
        return {
            "commitment_matches": commitment == sealed == self.schedule_commitment_sha256,
            "arm_matches": arm == self.arm,
        }


def generate_night_schedule(
    seed: int,
    blocks: int,
    washout_nights: int = 0,
    *,
    schedule_id: Optional[str] = None,
    night_ids: Optional[Sequence[str]] = None,
    night_prefix: str = DEFAULT_NIGHT_PREFIX,
    sham_cue_id: str = DEFAULT_SHAM_CUE_ID,
    salt: Optional[str] = None,
    created_at_utc: str = "",
) -> NightSchedule:
    """A series of ``blocks`` counterbalanced blocks of cue, sham and no-cue nights, with
    ``washout_nights`` washout nights between two blocks.

    The order comes from ``random.Random(seed)`` alone, so one seed gives one order; the
    salt comes from ``secrets`` unless given. Night IDs are ``night_ids`` or
    ``<night_prefix>001``, ``<night_prefix>002``, ... in series order.
    """
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    if washout_nights < 0:
        raise ValueError("washout_nights must be non-negative")
    rng = random.Random(seed)
    orders = _block_orders(rng, blocks, BLOCK_ARMS)
    layout = _block_layout(blocks, washout_nights, len(BLOCK_ARMS))
    if night_ids is None:
        width = max(3, len(str(len(layout))))
        night_ids = [f"{night_prefix}{index:0{width}d}" for index in range(1, len(layout) + 1)]
    night_ids = [str(night_id) for night_id in night_ids]
    if len(night_ids) != len(layout):
        raise ValueError(f"the series has {len(layout)} nights; {len(night_ids)} night IDs were given")
    places = {block: 0 for block in range(1, blocks + 1)}
    nights = []
    for position, (night_id, block) in enumerate(zip(night_ids, layout), start=1):
        if block is None:
            arm = "washout"
        else:
            arm = orders[block - 1][places[block]]
            places[block] += 1
        nights.append(ScheduledNight(night_id=night_id, position=position, arm=arm, block=block))
    if schedule_id is None:
        schedule_id = f"series-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    return NightSchedule(
        schedule_id=schedule_id,
        seed=int(seed),
        blocks=blocks,
        washout_nights=washout_nights,
        nights=tuple(nights),
        salt=salt if salt is not None else secrets.token_hex(_SALT_BYTES),
        sham_cue_id=sham_cue_id,
        created_at_utc=created_at_utc,
    )


def sealed_path_for(schedule_path: Path) -> Path:
    """``series.json`` -> ``series.sealed.json``, next to it."""
    schedule_path = Path(schedule_path)
    stem = schedule_path.name[: -len(".json")] if schedule_path.name.endswith(".json") else schedule_path.name
    return schedule_path.with_name(stem + SEALED_SUFFIX)


def write_night_schedule(
    schedule: NightSchedule,
    output_path: Path,
    *,
    sealed_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Write the schedule and its sealed companion. Refuses to replace either: a series
    whose arms could be drawn again until they suit would not be sealed."""
    output_path = Path(output_path).expanduser()
    sealed_path = Path(sealed_path).expanduser() if sealed_path is not None else sealed_path_for(output_path)
    if output_path.resolve() == sealed_path.resolve():
        raise ValueError("the schedule and its sealed companion need two different paths")
    for path in (output_path, sealed_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; a night schedule is never replaced")
    schedule.save(output_path)
    _write_json(sealed_path, schedule.sealed_companion())
    return output_path, sealed_path


def load_night_schedule(input_path: Path) -> NightSchedule:
    return NightSchedule.load(input_path)


def load_sealed_commitment(input_path: Path) -> str:
    """The ``commitment_sha256`` of a sealed companion."""
    payload = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    commitment = str(payload.get("commitment_sha256", "")) if isinstance(payload, dict) else ""
    if len(commitment) != 64 or not _HEX.match(commitment):
        raise ValueError(f"{input_path} holds no commitment_sha256")
    return commitment


def seal_matches(schedule: NightSchedule, sealed_path: Path) -> bool:
    return schedule.commitment_sha256() == load_sealed_commitment(sealed_path)


def verify_seal(schedule: NightSchedule, sealed_path: Path) -> str:
    """The commitment, or SealMismatch when the schedule no longer matches it."""
    sealed = load_sealed_commitment(sealed_path)
    if schedule.commitment_sha256() != sealed:
        raise SealMismatch(
            f"night schedule {schedule.schedule_id} does not match the commitment in {Path(sealed_path).name}: "
            "it changed after it was sealed"
        )
    return sealed


def unseal_night_schedule(
    schedule_path: Path,
    *,
    sealed_path: Optional[Path] = None,
    now: Optional[str] = None,
) -> NightSchedule:
    """Check the schedule against its sealed companion and record ``unsealed_at_utc``.

    A schedule unsealed before keeps its first time. SealMismatch when the schedule
    changed after it was sealed; then nothing is written.
    """
    schedule_path = Path(schedule_path).expanduser()
    sealed_path = Path(sealed_path).expanduser() if sealed_path is not None else sealed_path_for(schedule_path)
    schedule = load_night_schedule(schedule_path)
    verify_seal(schedule, sealed_path)
    if not schedule.sealed:
        return schedule
    unsealed = replace(schedule, unsealed_at_utc=now or _utc_now())
    unsealed.save(schedule_path)
    return unsealed


def night_arm_from_schedule(
    schedule_path: Path,
    night_id: str,
    *,
    sealed_path: Optional[Path] = None,
    sham_cue_id: Optional[str] = None,
) -> NightArmContext:
    """The arm ``night_id`` runs, after the schedule was checked against its sealed
    companion. SealMismatch when it changed, KeyError for a night it does not hold,
    WashoutNight for a washout night."""
    schedule_path = Path(schedule_path).expanduser()
    sealed_path = Path(sealed_path).expanduser() if sealed_path is not None else sealed_path_for(schedule_path)
    schedule = load_night_schedule(schedule_path)
    commitment = verify_seal(schedule, sealed_path)
    arm = schedule.arm_for(night_id)
    if arm == "washout":
        raise WashoutNight(
            f"night {night_id} is a washout night of {schedule.schedule_id}: no cueing night is planned for it"
        )
    return NightArmContext(
        arm=arm,
        night_id=night_id,
        sham_cue_id=sham_cue_id or schedule.sham_cue_id,
        schedule_id=schedule.schedule_id,
        schedule_commitment_sha256=commitment,
        schedule_unsealed_at_utc=schedule.unsealed_at_utc,
        schedule_path=str(schedule_path),
        sealed_path=str(sealed_path),
    )


def prereg_template(
    schedule: NightSchedule,
    *,
    protocol: Optional[Mapping[str, object]] = None,
    protocol_source: str = "",
    sealed_name: str = "",
    analysis: Optional[Mapping[str, object]] = None,
    software_version: str = "",
    generated_at_utc: Optional[str] = None,
) -> str:
    """An OSF-style preregistration of the series, as Markdown.

    It renders what can be filed before the first night: the design (blocks, arms,
    washout nights, the nights in order and how their arms are drawn), the sealed
    commitment and the protocol block's hashes (``ProtocolPreset.manifest()``).
    It never renders an arm, the seed or the salt: the seed alone gives the order back.
    So two schedules of one design drawn on the same day differ in the template only in
    their commitment. ``sealed_name`` is the sealed companion's file name. ``analysis``
    holds the analysis settings it states (seed, permutations, interval, outcomes);
    lines marked "To fill in" are the author's.
    """
    generated = generated_at_utc or _utc_now()
    analysis = dict(analysis or {})
    block_nights = [night for night in schedule.nights if night.arm != "washout"]
    lines = [
        f"# Preregistration: N-of-1 series {schedule.schedule_id}",
        "",
        f"Written by `muse-tmr prereg-template`{_version_text(software_version)} on {generated[:10]} from the sealed "
        "night schedule and its commitment. It gives the series' design and its seal and holds no arm, no seed "
        "and no salt, so it can be filed, on OSF for example, before the first night. Lines marked _To fill in_ "
        "are yours.",
        "",
    ]
    if not schedule.sealed:
        lines += [
            f"> This schedule was unsealed at {schedule.unsealed_at_utc}. A template written after unsealing is not "
            "a preregistration.",
            "",
        ]
    lines += [
        "## Study information",
        "",
        "- Title: _To fill in._",
        "- Research question and hypotheses: _To fill in_: which arm difference you expect, on which outcome and at "
        "which retest.",
        "- Description: one participant runs a series of nights with REMind and a Muse S Athena headband. A `cue` "
        "night plays the trained sounds when the REM gate and the arousal guard let a cue through, a `sham` night "
        "plays a sound that was never trained at the same moments, and a `no_cue` night plays nothing. This is a "
        "single-case self-experiment, not a clinical trial: it can describe what happened, not show that cueing "
        "works.",
        "",
        "## Design",
        "",
        "- Study type: an N-of-1 series, randomised in blocks, one participant.",
        f"- Arms: {', '.join(f'`{arm}`' for arm in schedule.arms)}; the sham sound is `{schedule.sham_cue_id}`, "
        "which is never trained.",
        f"- Blocks: {schedule.blocks}, each with one night of every arm; {schedule.washout_nights} washout "
        f"night{'s' if schedule.washout_nights != 1 else ''} between two blocks, without a cueing run; "
        f"{len(schedule.nights)} nights in all, {len(block_nights)} of them in blocks.",
        f"- Counterbalancing: the blocks go in groups of {len(schedule.arms)}, and the blocks of a group take "
        "distinct rows of one Latin square, so within a group no arm takes the same place twice, and in a full "
        "group every arm takes every place once.",
        "- Within a night: `assign-puzzle-cues` splits the night's puzzles into cued and uncued with the night's "
        "own seed, and an uncued puzzle is never scheduled, in any arm.",
        "- Blinding: the arms stay in the schedule file. The runners never print a night's arm and the Morning tab "
        "never reads the schedule; a night's summary, logs and reports name the arm and stay closed until "
        "`unseal-night-schedule`. At n = 1 the participant is the experimenter and cannot be blind, a sham night "
        "still makes sound, and the sleeper may tell it from a trained cue.",
        "",
        "| Position | Night | Block |",
        "| ---: | --- | --- |",
    ]
    for night in schedule.nights:
        lines.append(f"| {night.position} | {night.night_id} | {'washout' if night.block is None else night.block} |")
    lines += [
        "",
        "## Randomisation and seal",
        "",
        f"- Drawn by `plan-night-schedule` on {schedule.created_at_utc[:10]} with `random.Random(seed)`, from a "
        "seed given with `--seed` or, without it, 32 random bits from `secrets`. Which of the two: _To fill in._ "
        "The seed stays in the schedule file with a random 16-byte salt and is not given here, because the seed "
        "alone gives the order back.",
        f"- Commitment: `commitment_sha256 = {schedule.commitment_sha256()}`, the sha256 of the schedule's canonical "
        "JSON (sorted keys, no spaces) followed by the salt. The sealed companion"
        + (f" `{sealed_name}`" if sealed_name else f", `<name>{SEALED_SUFFIX}` next to the schedule,")
        + " holds nothing else.",
        "- After the series: `muse-tmr unseal-night-schedule` checks the schedule against the commitment and "
        "records `unsealed_at_utc`. Anyone given the unsealed schedule can compute the hash again.",
        "",
        "## Protocol",
        "",
        f"- {_protocol_line(protocol, protocol_source)}",
        f"- Software: REMind{_version_text(software_version, prefix=' ') or ' (version not given)'}.",
        "",
        "## Sampling plan",
        "",
        "- Participant: one. Inclusion and exclusion criteria: _To fill in._",
        f"- Planned nights: {len(schedule.nights)}, with retests the morning after, a week later (`day7`) and ten "
        "days later (`day10`).",
        "- Stopping rule: _To fill in._",
        "",
        "## Variables",
        "",
        "- Manipulated: the night's arm; within a night, cued against uncued puzzles.",
        "- Measured in the morning, before unsealing (the Morning tab, or `record-dream-report`, "
        "`record-cue-recall` and `record-puzzle-retest`): lucidity, whether and where a sound was heard "
        "(`cue_perception`), whether a sound woke the sleeper, the times of eye-signal attempts (marked by the "
        "sleeper, never detected), the sound check, and each puzzle's retest (solved, duration, confidence) the "
        "morning after, a week later and ten days later.",
        "- Coded blind: each puzzle's dream incorporation (`sound_cue`, `direct_conceptual`, "
        "`indirect_conceptual`) from the packets of `export-blind-coding`, which hold no arm and no cue condition.",
        "- Logged by each night: plays and virtual plays, awakening markers, and the arousal proxies per cue of "
        "`score-cue-arousal`, run after unsealing.",
        "",
        "## Analysis plan",
        "",
        "- `muse-tmr analyze-nof1`, after `unseal-night-schedule`: per arm, the mean of "
        f"{_outcome_list(analysis.get('outcomes'))}; for each of them the differences cue - no_cue, cue - sham "
        f"and sham - no_cue, each with its permutation interval: the central {_percent(analysis.get('interval'))} "
        "of the differences over the arm orders `plan-night-schedule` could have drawn for these blocks, every "
        f"order up to {_count(analysis.get('permutations'))} of them, else that many drawn with "
        f"`--seed {analysis.get('seed', 0)}`; the lag-1 autocorrelation of each outcome after the arm means are "
        "taken out; the first night flagged, with the means without it."
        + _orders_note(analysis.get("possible_orders"), analysis.get("outside_possible")),
        "- Primary outcome and contrast: _To fill in_ (`--primary-outcome`, `--primary-contrast`).",
        "- Inference: descriptive and single-case. No p-value is computed, and a difference outside its interval "
        "describes this series; it is not evidence that cueing works.",
        "- Missing data: a missing form or retest is a missing value, never a zero, and the report names the "
        "nights it concerns.",
        "- Exclusions: _To fill in._",
        "",
        "## Known limitations",
        "",
        "- One participant, who is also the experimenter; a sham still makes sound and may be recognised; "
        "consecutive nights can carry over into each other; the first night may differ from the rest; the REM "
        "gate runs on consumer forehead EEG, and its detector weights and first-cue latency are open issues "
        "(#116, #167), which every arm comparison inherits.",
        "- Reporting follows SCRIBE 2016 and CENT 2015: `analyze-nof1` writes their checklist with the report.",
    ]
    return "\n".join(lines) + "\n"


def _protocol_line(protocol: Optional[Mapping[str, object]], source: str) -> str:
    where = f" From {source}." if source else ""
    if not protocol:
        return "Protocol preset: unknown; give `--protocol-preset` or `--protocol-metadata`." + where
    overrides = dict(protocol.get("overrides", {}) or {})
    override_text = (
        ", ".join(f"`{name}={value}`" for name, value in sorted(overrides.items())) if overrides else "none"
    )
    registry = protocol.get("registry_version")
    return (
        f"Protocol preset `{protocol.get('name', 'unknown')}` version {protocol.get('version', 'unknown')}"
        + (f" (registry {registry})" if registry is not None else "")
        + f", content hash `{protocol.get('content_hash', 'unknown')}`, effective hash "
        f"`{protocol.get('effective_hash', 'unknown')}`, overrides: {override_text}.{where}"
    )


def _orders_note(possible_orders: object, outside_possible: object) -> str:
    if not isinstance(possible_orders, int):
        return ""
    note = f" These blocks allow {possible_orders:,} arm orders"
    if outside_possible is False:
        return note + ", too few for any difference to fall outside its interval."
    return note + "."


def _version_text(version: str, prefix: str = ", REMind ") -> str:
    return f"{prefix}{version}" if version else ""


def _outcome_list(outcomes: object) -> str:
    names = [str(name) for name in (outcomes or ())]
    return ", ".join(f"`{name}`" for name in names) if names else "each outcome"


def _percent(level: object) -> str:
    return f"{float(level) * 100:g} %" if isinstance(level, (int, float)) else "95 %"


def _count(value: object) -> str:
    return f"{int(value):,}" if isinstance(value, int) else "10,000"


def canonical_json(payload: object) -> str:
    """Sorted keys, no spaces, as ``protocol_presets.canonical_json`` writes it."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _block_orders(rng: random.Random, blocks: int, arms: Sequence[str]) -> List[Tuple[str, ...]]:
    """The arm order of each block: per group of len(arms) blocks, distinct rows of a
    random Latin square, so within a group no arm takes the same place twice."""
    orders: List[Tuple[str, ...]] = []
    while len(orders) < blocks:
        base = list(arms)
        rng.shuffle(base)
        square = [tuple(base[shift:] + base[:shift]) for shift in range(len(base))]
        rng.shuffle(square)
        orders.extend(square[: blocks - len(orders)])
    return orders


def _block_layout(blocks: int, washout_nights: int, arm_count: int) -> List[Optional[int]]:
    """Each night's block in series order, None for a washout night."""
    layout: List[Optional[int]] = []
    for block in range(1, blocks + 1):
        if block > 1:
            layout.extend([None] * washout_nights)
        layout.extend([block] * arm_count)
    return layout


def _identifier(value: object, name: str) -> str:
    text = str(value).strip()
    if not _ID.match(text):
        raise ValueError(f"{name} must be letters, digits, '.', '_' or '-', got {value!r}")
    return text


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
