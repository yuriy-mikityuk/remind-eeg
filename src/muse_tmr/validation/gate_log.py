"""The REM gate's decision per epoch, and the gate-open runs read from it (#227).

Pilot 4 and 5 write one line per epoch to ``rem_gate_decisions.jsonl``; a Pilot 3 report
keeps the same decisions in its ``epochs``. ``load_rem_gate_decisions()`` reads either
and ``gate_open_runs()`` turns them into gate-open runs. A gate-open run is what the
cue arousal scorer and the protocol fidelity report (Initiative 7) call a REM period: the
epochs in a row whose decision left the gate open. It starts when the first of them was
decided, the end of that epoch, and ends when the first decision after them closed the
gate. Defined here once so both read REM periods the same way.

Times are seconds from the start of the night's first epoch, the time base of the
scheduler's ``timestamp_seconds``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

REM_GATE_DECISIONS_FILENAME = "rem_gate_decisions.jsonl"
GATE_LOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GateDecisionRecord:
    """One epoch's gate decision.

    ``timestamp_seconds`` is the epoch's start and ``duration_seconds`` its length, as in
    a Pilot 3 report's ``epochs``; the decision exists at their sum,
    ``decision_seconds``, and holds until the next one.
    """

    index: int
    timestamp_seconds: float
    duration_seconds: float
    gate_open: bool
    state: str = ""
    p_rem: Optional[float] = None
    confidence: Optional[float] = None
    stable_seconds: float = 0.0
    cooldown_remaining_seconds: float = 0.0
    reason_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", tuple(str(code) for code in self.reason_codes))

    @property
    def decision_seconds(self) -> float:
        return self.timestamp_seconds + self.duration_seconds

    def to_dict(self) -> dict:
        return {
            "schema_version": GATE_LOG_SCHEMA_VERSION,
            "index": self.index,
            "timestamp_seconds": self.timestamp_seconds,
            "duration_seconds": self.duration_seconds,
            "decision_seconds": self.decision_seconds,
            "gate_open": self.gate_open,
            "state": self.state,
            "p_rem": self.p_rem,
            "confidence": self.confidence,
            "stable_seconds": self.stable_seconds,
            "cooldown_remaining_seconds": self.cooldown_remaining_seconds,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GateDecisionRecord":
        return cls(
            index=int(payload["index"]),
            timestamp_seconds=float(payload["timestamp_seconds"]),
            duration_seconds=float(payload["duration_seconds"]),
            gate_open=bool(payload["gate_open"]),
            state=str(payload.get("state", "")),
            p_rem=_optional_float(payload.get("p_rem")),
            confidence=_optional_float(payload.get("confidence")),
            stable_seconds=float(payload.get("stable_seconds", 0.0) or 0.0),
            cooldown_remaining_seconds=float(payload.get("cooldown_remaining_seconds", 0.0) or 0.0),
            reason_codes=tuple(str(code) for code in payload.get("reason_codes", ()) or ()),
        )

    @classmethod
    def from_pilot3_epoch(cls, epoch: Mapping[str, object]) -> "GateDecisionRecord":
        """The record of one entry of a Pilot 3 report's ``epochs``."""
        decision = dict(epoch["gate_decision"])
        confidence = dict(decision.get("confidence", {}) or {})
        return gate_decision_record(
            index=int(epoch["index"]),
            timestamp_seconds=float(epoch["timestamp_seconds"]),
            duration_seconds=float(epoch["duration_seconds"]),
            decision=decision,
            confidence=confidence,
        )


@dataclass(frozen=True)
class GateOpenRun:
    """A gate-open run, the REM period of the scorer and the fidelity report.

    ``start_seconds`` is the decision time of its first open epoch. ``end_seconds`` is
    the decision time of the first epoch after it whose decision closed the gate, or,
    when the log ends with the gate open, of its last open epoch; ``closed`` says which.
    """

    index: int
    start_seconds: float
    end_seconds: float
    closed: bool
    first_epoch_index: int
    last_epoch_index: int
    epoch_count: int

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def contains(self, seconds: float) -> bool:
        """Whether the gate was open at ``seconds`` by this run's decisions."""
        if self.closed:
            return self.start_seconds <= seconds < self.end_seconds
        return self.start_seconds <= seconds <= self.end_seconds

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "closed": self.closed,
            "first_epoch_index": self.first_epoch_index,
            "last_epoch_index": self.last_epoch_index,
            "epoch_count": self.epoch_count,
        }


def gate_decision_record(
    *,
    index: int,
    timestamp_seconds: float,
    duration_seconds: float,
    decision: object,
    confidence: Optional[Mapping[str, object]] = None,
) -> GateDecisionRecord:
    """The record of a ``RemGateDecision`` (or its ``to_dict()``) for one epoch."""
    payload = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)  # type: ignore[arg-type]
    confidence = dict(confidence if confidence is not None else payload.get("confidence", {}) or {})
    return GateDecisionRecord(
        index=int(index),
        timestamp_seconds=float(timestamp_seconds),
        duration_seconds=float(duration_seconds),
        gate_open=bool(payload["gate_open"]),
        state=str(payload.get("state", "")),
        p_rem=_optional_float(confidence.get("probability")),
        confidence=_optional_float(confidence.get("confidence")),
        stable_seconds=float(payload.get("stable_seconds", 0.0) or 0.0),
        cooldown_remaining_seconds=float(payload.get("cooldown_remaining_seconds", 0.0) or 0.0),
        reason_codes=tuple(str(code) for code in payload.get("reason_codes", ()) or ()),
    )


def write_rem_gate_decisions(
    records: Iterable[GateDecisionRecord],
    output_path: Path,
    *,
    append: bool = False,
) -> Path:
    """Write ``records`` as JSONL, one line each; ``append`` adds them to the file."""
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a" if append else "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return output_path


GateLogSource = Union[Path, str, Mapping[str, object]]


def load_rem_gate_decisions(source: GateLogSource) -> Tuple[GateDecisionRecord, ...]:
    """The gate decisions of a night, in epoch order.

    ``source`` is a Pilot 4/5 output directory (its ``rem_gate_decisions.jsonl``), that
    JSONL file, a Pilot 3 report ``.json`` or its loaded dict. A missing file gives none.
    """
    if isinstance(source, Mapping):
        return _records_from_pilot3(source)
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / REM_GATE_DECISIONS_FILENAME
    if not path.exists():
        return ()
    if path.suffix == ".jsonl":
        records = tuple(
            GateDecisionRecord.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return tuple(sorted(records, key=lambda record: (record.timestamp_seconds, record.index)))
    return _records_from_pilot3(json.loads(path.read_text(encoding="utf-8")))


def gate_open_runs(records: Sequence[GateDecisionRecord]) -> Tuple[GateOpenRun, ...]:
    """The gate-open runs of ``records`` (see ``GateOpenRun``), in time order."""
    ordered = sorted(records, key=lambda record: (record.timestamp_seconds, record.index))
    runs = []
    current = []
    for record in ordered:
        if record.gate_open:
            current.append(record)
            continue
        if current:
            runs.append(_run(len(runs), current, end_seconds=record.decision_seconds, closed=True))
            current = []
    if current:
        runs.append(_run(len(runs), current, end_seconds=current[-1].decision_seconds, closed=False))
    return tuple(runs)


def load_gate_open_runs(source: GateLogSource) -> Tuple[GateOpenRun, ...]:
    """``gate_open_runs()`` of what ``load_rem_gate_decisions(source)`` reads."""
    return gate_open_runs(load_rem_gate_decisions(source))


def _records_from_pilot3(report: Mapping[str, object]) -> Tuple[GateDecisionRecord, ...]:
    return tuple(GateDecisionRecord.from_pilot3_epoch(epoch) for epoch in report.get("epochs", ()) or ())


def _run(index: int, records: Sequence[GateDecisionRecord], *, end_seconds: float, closed: bool) -> GateOpenRun:
    return GateOpenRun(
        index=index,
        start_seconds=records[0].decision_seconds,
        end_seconds=end_seconds,
        closed=closed,
        first_epoch_index=records[0].index,
        last_epoch_index=records[-1].index,
        epoch_count=len(records),
    )


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)
