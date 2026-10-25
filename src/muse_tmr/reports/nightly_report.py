"""Generate a self-contained HTML REM-probability report for one recording.

Replays a recording directory through the same epoch/feature/REM-detector chain
as `muse-tmr annotate-template`, then renders a single-file HTML chart (line +
area, hover tooltip, accessible table fallback) instead of raw JSON/CSV. A
recording whose metadata.json says `synthetic: true` gets a "Synthetic input,
not sleep" callout at the top.

Output mirrors the recording's kind folder: a recording under
data/recordings/<kind>/<name> produces data/reports/<kind>/<name>.html for
kind in {night, session}; anything else falls back to
data/reports/nightly/<name>.html. All of data/reports/ is gitignored (see
docs/sdk_policy.md: personal sleep reports must not be committed) -- this
code is checked in, its output is not.

The report lives in the package so `muse-tmr demo` can build it from an
installed wheel; scripts/generate_nightly_report.py runs it from a checkout.

Usage:
    python scripts/generate_nightly_report.py data/recordings/night/20260707_010000
    python scripts/generate_nightly_report.py data/recordings/session/20260707_140000 --output /tmp/report.html
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from muse_tmr.annotations import build_rem_annotation_rows
from muse_tmr.reports.report_style import (
    TOKENS_CSS,
    brand_line,
    format_braces,
    local_minute_of_day,
    readable_time,
)
from muse_tmr.data.replay import ReplayConfig, ReplaySession
from muse_tmr.features.epochs import EpochBuilder, EpochConfig
from muse_tmr.models import HeuristicRemDetector
from muse_tmr.reports.night_cardio import night_cardio_section
from muse_tmr.testing.synthetic import SYNTHETIC_NOTE


async def _build_rows(recording_dir: Path, epoch_seconds: float) -> list[dict]:
    session = ReplaySession(ReplayConfig(input_path=recording_dir, speed=0.0))
    await session.connect()
    try:
        builder = EpochBuilder(EpochConfig(epoch_seconds=epoch_seconds, stride_seconds=epoch_seconds))
        epochs = [epoch async for epoch in builder.build(session.stream())]
    finally:
        await session.stop()

    rows = build_rem_annotation_rows(
        epochs,
        detector=HeuristicRemDetector(),
        recording_id=str(recording_dir),
        label="unknown",
    )
    return [row.to_dict() for row in rows]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _default_report_path(recording_dir: Path) -> Path:
    """Mirror the recording's kind folder into data/reports/<kind>/<name>.html.

    A recording stored under data/recordings/<kind>/<name> keeps its report
    alongside siblings of the same kind; older flat recordings fall back to the
    legacy data/reports/nightly/ folder.
    """
    kind = recording_dir.parent.name
    if kind in ("night", "session"):
        return Path("data/reports") / kind / f"{recording_dir.name}.html"
    return Path("data/reports/nightly") / f"{recording_dir.name}.html"


def _build_callout(mean_p: float, ppg_present: bool, reconnects: int, stop_reason: str) -> str:
    parts = []
    if mean_p > 0.55:
        parts.append(
            f"Mean p_rem for the night is {mean_p:.2f} &mdash; physiologically, REM sleep normally "
            "occupies only ~20&ndash;25% of total sleep time, concentrated in later cycles. This heuristic "
            "is an uncalibrated non-ML baseline (default thresholds, no personal training data), so a "
            "persistently high reading usually means the thresholds are running hot for this person's "
            "EEG/IMU baseline, not that REM genuinely dominated the night."
        )
    else:
        parts.append(
            f"Mean p_rem for the night is {mean_p:.2f}, within the physiologically plausible range for "
            "REM occupancy &mdash; but this is still an uncalibrated non-ML baseline, not a validated "
            "REM/NREM classifier. Treat the trend as directional, not diagnostic."
        )
    if not ppg_present:
        parts.append(
            "No PPG/heart-rate data this session (hr_variability and hr_trend contributed nothing to "
            "every epoch's score) &mdash; expected if this ran on preset p21."
        )
    if reconnects:
        parts.append(
            f"{reconnects} BLE reconnect(s) occurred during the night; any near-zero dip lining up with "
            "a reconnect event is a data gap, not a sleep-stage reading."
        )
    if stop_reason and stop_reason != "duration_complete":
        parts.append(f"Recording did not reach its full configured duration (stop_reason: {stop_reason}).")
    return " ".join(parts)


def _is_synthetic(metadata: Mapping[str, object]) -> bool:
    return metadata.get("synthetic") is True


def _synthetic_callout(metadata: Mapping[str, object]) -> str:
    """The callout of a night from muse_tmr.testing; "" for a recorded one."""
    if not _is_synthetic(metadata):
        return ""
    scenario = html.escape(str(metadata.get("scenario", "unknown")))
    seed = html.escape(str(metadata.get("seed", "?")))
    return (
        '<div class="callout"><strong>Synthetic input, not sleep.</strong> '
        f"This recording was generated by <code>muse_tmr.testing</code> (scenario {scenario}, "
        f"seed {seed}) from a scripted hypnogram, and nobody slept in it. The curve shows what "
        "the detector makes of signals written to look like sleep stages; it says nothing about "
        "how well REM is detected on a real night.</div>"
    )


# The app's tokens under the names this page uses; the night's curve is the app's indigo.
TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
<style>""" + format_braces(TOKENS_CSS) + """
.viz-root {{
  --surface-1: var(--sheet); --page: var(--paper); --text-primary: var(--ink); --text-secondary: var(--muted);
  --text-muted: var(--muted); --gridline: var(--line); --baseline: #c9c2b6; --series-1: var(--sleep);
  --series-1-fill-top: rgba(75,86,148,0.22); --series-1-fill-bot: rgba(75,86,148,0.02);
  --border: var(--line); --warn-ink: var(--warn); --warn-border: var(--warn-line);
  font-family: var(--sans); background: var(--page); color: var(--text-primary);
}}
@media (prefers-color-scheme: dark) {{
  .viz-root {{ --baseline: #4a4944; --series-1-fill-top: rgba(165,173,240,0.28); --series-1-fill-bot: rgba(165,173,240,0.02); }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
.viz-root {{ min-height: 100vh; padding: 32px 20px 60px; }}
.wrap {{ max-width: 920px; margin: 0 auto; }}
h1 {{ font-size: 26px; font-weight: 600; margin: 0 0 4px; }}
.subtitle {{ color: var(--text-secondary); font-size: 14px; margin: 0 0 24px; }}
.stat-row {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
.stat-tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; min-width: 120px; flex: 1; }}
.stat-tile .label {{ font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }}
.stat-tile .value {{ font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.callout {{ background: var(--warn-bg); border: 1px solid var(--warn-border); color: var(--warn-ink); border-radius: 10px; padding: 14px 16px; font-size: 13px; line-height: 1.5; margin-bottom: 24px; }}
.callout strong {{ font-weight: 600; }}
.chart-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 20px 20px 8px; position: relative; }}
.chart-title {{ font-size: 14px; font-weight: 600; margin: 0 0 2px; }}
.chart-sub {{ font-size: 12px; color: var(--text-muted); margin: 0 0 12px; }}
svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
.gridline {{ stroke: var(--gridline); stroke-width: 1; }}
.axis-label {{ fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
.baseline {{ stroke: var(--baseline); stroke-width: 1; }}
.area-fill {{ fill: url(#areaGrad); }}
.line-path {{ fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.crosshair-line {{ stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0; pointer-events: none; }}
.crosshair-dot {{ fill: var(--series-1); stroke: var(--surface-1); stroke-width: 2; opacity: 0; pointer-events: none; }}
.hit-layer {{ fill: transparent; cursor: crosshair; }}
.tooltip {{ position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.12); opacity: 0; transform: translate(-50%, -110%); white-space: nowrap; z-index: 5; }}
.tooltip .t-time {{ color: var(--text-muted); margin-bottom: 3px; }}
.tooltip .t-val {{ font-weight: 700; font-size: 15px; font-variant-numeric: tabular-nums; }}
.tooltip .t-val .key {{ display: inline-block; width: 10px; height: 2px; background: var(--series-1); margin-right: 6px; vertical-align: middle; border-radius: 1px; }}
.tooltip .t-rc {{ color: var(--text-muted); font-size: 10.5px; margin-top: 4px; max-width: 260px; white-space: normal; }}
details {{ margin-top: 20px; }}
summary {{ cursor: pointer; font-size: 13px; color: var(--text-secondary); padding: 6px 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--gridline); font-variant-numeric: tabular-nums; }}
th {{ color: var(--text-muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; }}
</style>
</head>
<body>
<div class="viz-root">
  <div class="wrap">
    {brand}
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
    {synthetic_callout}

    <div class="stat-row">
      <div class="stat-tile"><div class="label">Mean p_rem</div><div class="value">{mean_p:.2f}</div></div>
      <div class="stat-tile"><div class="label">Recording</div><div class="value">{duration_h:.1f}h</div></div>
      <div class="stat-tile"><div class="label">Epochs</div><div class="value">{n_epochs}</div></div>
      <div class="stat-tile"><div class="label">Reconnects</div><div class="value">{reconnects}</div></div>
    </div>

    <div class="callout"><strong>Read this with a grain of salt.</strong> {callout}</div>

    <div class="chart-card">
      <p class="chart-title">p_rem over time</p>
      <p class="chart-sub">Hover to inspect an epoch &middot; dashed line = mean</p>
      <div style="position:relative">
        <svg id="chart" viewBox="0 0 860 260" preserveAspectRatio="none">
          <defs><linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="var(--series-1-fill-top)" />
            <stop offset="100%" stop-color="var(--series-1-fill-bot)" />
          </linearGradient></defs>
          <g id="gridlines"></g>
          <path id="areaPath" class="area-fill"></path>
          <path id="linePath" class="line-path"></path>
          <line id="meanLine" class="baseline" stroke-dasharray="4 3"></line>
          <g id="xaxis"></g>
          <line id="crosshairLine" class="crosshair-line" x1="0" y1="0" x2="0" y2="230"></line>
          <circle id="crosshairDot" class="crosshair-dot" r="4"></circle>
          <rect id="hitLayer" class="hit-layer" x="0" y="0" width="860" height="230"></rect>
        </svg>
        <div id="tooltip" class="tooltip">
          <div class="t-time" id="ttTime"></div>
          <div class="t-val"><span class="key"></span><span id="ttVal"></span></div>
          <div class="t-rc" id="ttRc"></div>
        </div>
      </div>
    </div>

    <details>
      <summary>Table view &mdash; hourly averages (accessible fallback)</summary>
      <table><thead><tr><th>Hour</th><th>Mean p_rem</th><th>Epochs</th></tr></thead><tbody id="hourlyBody"></tbody></table>
    </details>
    {cardio_section}
  </div>
</div>

<script>
const DATA = {data_json};
const HOURLY = {hourly_json};
const T_MAX = {t_max};
const START_MIN = {start_min};

const svg = document.getElementById('chart');
const W = 860, H = 260, padL = 34, padR = 10, padT = 10, padB = 30;
const plotW = W - padL - padR, plotH = H - padT - padB;

function x(t) {{ return padL + (t / T_MAX) * plotW; }}
function y(p) {{ return padT + (1 - p) * plotH; }}

const gl = document.getElementById('gridlines');
[0, 0.25, 0.5, 0.75, 1].forEach(p => {{
  const ly = y(p);
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', padL); line.setAttribute('x2', W - padR);
  line.setAttribute('y1', ly); line.setAttribute('y2', ly); line.setAttribute('class', 'gridline');
  gl.appendChild(line);
  const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  label.setAttribute('x', padL - 8); label.setAttribute('y', ly + 3);
  label.setAttribute('text-anchor', 'end'); label.setAttribute('class', 'axis-label');
  label.textContent = p.toFixed(2);
  gl.appendChild(label);
}});

const xa = document.getElementById('xaxis');
const hourStep = T_MAX > 4 ? 1 : 0.5;
for (let h = 0; h <= T_MAX + 0.001; h += hourStep) {{
  const lx = x(h);
  const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  label.setAttribute('x', lx); label.setAttribute('y', H - 8);
  label.setAttribute('text-anchor', h < 0.01 ? 'start' : (h > T_MAX - 0.01 ? 'end' : 'middle'));
  label.setAttribute('class', 'axis-label');
  label.textContent = h + 'h';
  xa.appendChild(label);
}}

let areaD = 'M ' + x(DATA[0].t) + ' ' + y(0) + ' ';
let lineD = 'M ' + x(DATA[0].t) + ' ' + y(DATA[0].p) + ' ';
DATA.forEach((d, i) => {{
  const px = x(d.t), py = y(d.p);
  areaD += 'L ' + px + ' ' + py + ' ';
  if (i > 0) lineD += 'L ' + px + ' ' + py + ' ';
}});
areaD += 'L ' + x(DATA[DATA.length - 1].t) + ' ' + y(0) + ' Z';
document.getElementById('areaPath').setAttribute('d', areaD);
document.getElementById('linePath').setAttribute('d', lineD);

const meanP = DATA.reduce((s, d) => s + d.p, 0) / DATA.length;
const meanLine = document.getElementById('meanLine');
meanLine.setAttribute('x1', padL); meanLine.setAttribute('x2', W - padR);
meanLine.setAttribute('y1', y(meanP)); meanLine.setAttribute('y2', y(meanP));

const hit = document.getElementById('hitLayer');
const chLine = document.getElementById('crosshairLine');
const chDot = document.getElementById('crosshairDot');
const tooltip = document.getElementById('tooltip');
const ttTime = document.getElementById('ttTime');
const ttVal = document.getElementById('ttVal');
const ttRc = document.getElementById('ttRc');

function nearest(t) {{
  let lo = 0, hi = DATA.length - 1;
  while (lo < hi) {{ const mid = (lo + hi) >> 1; if (DATA[mid].t < t) lo = mid + 1; else hi = mid; }}
  return DATA[lo];
}}

function fmtTime(tHours) {{
  const totalMin = (START_MIN + tHours * 60) % (24 * 60);
  const hh = Math.floor(totalMin / 60), mm = Math.floor(totalMin % 60);
  return String(hh).padStart(2, '0') + ':' + String(mm).padStart(2, '0');
}}

function showAt(clientX) {{
  const rect = svg.getBoundingClientRect();
  const svgX = ((clientX - rect.left) / rect.width) * W;
  const t = Math.max(0, Math.min(T_MAX, ((svgX - padL) / plotW) * T_MAX));
  const d = nearest(t);
  const px = x(d.t), py = y(d.p);
  chLine.setAttribute('x1', px); chLine.setAttribute('x2', px);
  chLine.setAttribute('y1', padT); chLine.setAttribute('y2', padT + plotH);
  chLine.style.opacity = 1;
  chDot.setAttribute('cx', px); chDot.setAttribute('cy', py);
  chDot.style.opacity = 1;
  tooltip.style.opacity = 1;
  tooltip.style.left = ((px / W) * 100) + '%';
  tooltip.style.top = ((py / H) * 100) + '%';
  ttTime.textContent = fmtTime(d.t) + '  (t+' + d.t.toFixed(2) + 'h)';
  ttVal.textContent = 'p_rem ' + d.p.toFixed(2);
  ttRc.textContent = d.rc.split(';').slice(0, 4).join(', ');
}}
function hide() {{ chLine.style.opacity = 0; chDot.style.opacity = 0; tooltip.style.opacity = 0; }}
hit.addEventListener('pointermove', e => showAt(e.clientX));
hit.addEventListener('pointerleave', hide);

const tbody = document.getElementById('hourlyBody');
HOURLY.forEach(row => {{
  const tr = document.createElement('tr');
  const c1 = document.createElement('td'); c1.textContent = row.h;
  const c2 = document.createElement('td'); c2.textContent = row.avg.toFixed(2);
  const c3 = document.createElement('td'); c3.textContent = row.n;
  tr.appendChild(c1); tr.appendChild(c2); tr.appendChild(c3);
  tbody.appendChild(tr);
}});
</script>
</body>
</html>
"""


def build_report(recording_dir: Path, output_path: Path, epoch_seconds: float = 30.0) -> None:
    rows = asyncio.run(_build_rows(recording_dir, epoch_seconds))
    if not rows:
        raise SystemExit(f"no epochs produced from {recording_dir}")

    summary = _load_json(recording_dir / "summary.json")
    metadata = _load_json(recording_dir / "metadata.json")

    t0 = rows[0]["start_time"]
    chart_points = []
    for r in rows:
        chart_points.append(
            {
                "t": round((r["start_time"] - t0) / 3600, 4),
                "p": round(r["p_rem"], 4),
                "rc": r["reason_codes"],
            }
        )

    t_max = chart_points[-1]["t"]
    hour_step = 1 if t_max > 4 else 0.5
    hourly = []
    h = 0.0
    while h < t_max + 1e-9:
        chunk = [c for c in chart_points if h <= c["t"] < h + hour_step]
        if chunk:
            avg = sum(c["p"] for c in chunk) / len(chunk)
            label = f"{h:.1f}-{h + hour_step:.1f}h"
            hourly.append({"h": label, "avg": round(avg, 3), "n": len(chunk)})
        h += hour_step

    mean_p = sum(c["p"] for c in chart_points) / len(chart_points)
    ppg_present = any("hr_variability" in r["reason_codes"] or "hr_trend_support" in r["reason_codes"] for r in rows)
    modality_counts = summary.get("modality_counts", {})
    ppg_present = ppg_present or bool(modality_counts.get("ppg"))
    reconnects = summary.get("reconnect_attempts", 0)
    stop_reason = summary.get("stop_reason", "")
    duration_h = summary.get("duration_seconds", t_max * 3600) / 3600

    started_at = metadata.get("started_at", "")
    source_meta = metadata.get("source", {})
    source_name = source_meta.get("source_name", "unknown")
    preset = source_meta.get("metadata", {}).get("preset", "")

    started = readable_time(started_at)
    title = f"REM report · {started or recording_dir.name}"
    subtitle_bits = [recording_dir.name, source_name]
    if preset:
        subtitle_bits.append(f"preset {preset}")
    subtitle_bits.append(f"{len(rows)} epochs of {epoch_seconds:.0f} s")
    page_title = f"REM report · {recording_dir.name}"
    if _is_synthetic(metadata):
        subtitle_bits.append(SYNTHETIC_NOTE)
        page_title += " (synthetic)"
    subtitle = " · ".join(subtitle_bits)

    # The chart's clock times are local, like the heading.
    start_min = local_minute_of_day(started_at)

    callout = _build_callout(mean_p, ppg_present, reconnects, stop_reason)

    page = TEMPLATE.format(
        page_title=page_title,
        brand=brand_line("REM report"),
        title=title,
        subtitle=subtitle,
        synthetic_callout=_synthetic_callout(metadata),
        mean_p=mean_p,
        duration_h=duration_h,
        n_epochs=len(rows),
        reconnects=reconnects,
        callout=callout,
        data_json=json.dumps(chart_points, separators=(",", ":")),
        hourly_json=json.dumps(hourly, separators=(",", ":")),
        t_max=t_max,
        start_min=start_min,
        cardio_section=night_cardio_section(recording_dir, t0, rows[-1]["end_time"]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path, help="Recording directory (contains raw_amused.bin etc.)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path. Defaults to data/reports/<kind>/<name>.html (kind inferred from the recording folder).",
    )
    parser.add_argument("--epoch-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    recording_dir = args.recording_dir.resolve()
    if not recording_dir.exists():
        raise SystemExit(f"recording directory not found: {recording_dir}")

    output = args.output or _default_report_path(recording_dir)
    build_report(recording_dir, output, args.epoch_seconds)
    print(f"report written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
