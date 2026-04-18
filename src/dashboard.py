"""
VLM Orchestrator — Evaluation Dashboard

Generates a detailed HTML report comparing pipeline output against ground truth.
Shows a SVG-based timeline view with procedure steps, GT events, predicted events,
and missed events aligned on the same time axis.

Usage:
    python -m src.dashboard \
        --predicted output/events.json \
        --ground-truth data/ground_truth_sample/clip.json \
        --output output/dashboard.html \
        --tolerance 5
"""

import argparse
import json
import html as html_mod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from src.evaluator import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _esc(text: Any) -> str:
    """Escape for safe HTML embedding."""
    return html_mod.escape(str(text)) if text is not None else ""


def _esc_attr(text: Any) -> str:
    """Escape for use inside HTML attributes (also escapes quotes)."""
    return html_mod.escape(str(text), quote=True) if text is not None else ""


# ---------------------------------------------------------------------------
# Optimal greedy matching (mirrors evaluator logic)
# ---------------------------------------------------------------------------

def _min_distance_match_detailed(
    pairs: List[Tuple[int, int, float]],
) -> Dict[int, Tuple[int, float]]:
    """
    Greedy closest-first bipartite matching.
    Returns dict: pred_idx -> (gt_idx, distance).
    """
    pairs_sorted = sorted(pairs, key=lambda x: x[2])
    matched_pred: Dict[int, Tuple[int, float]] = {}
    matched_gt: set = set()
    for pi, gi, dist in pairs_sorted:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred[pi] = (gi, dist)
        matched_gt.add(gi)
    return matched_pred


def _match_events(pred_events, gt_events, gt_idles, tolerance):
    """
    Return detailed match info for every predicted and GT event.
    Uses the same optimal greedy matching strategy as the evaluator.
    """
    pred_steps = [e for e in pred_events if e.get("type") == "step_completion"]
    pred_errors = [e for e in pred_events if e.get("type") == "error_detected"]
    pred_idles = [e for e in pred_events if e.get("type") == "idle_detected"]
    gt_steps = [e for e in gt_events if e.get("type") == "step_completion"]
    gt_errors = [e for e in gt_events if e.get("type") == "error_detected"]

    results = []

    # --- Steps (match by step_id + timestamp) ---
    step_pairs = []
    for pi, p in enumerate(pred_steps):
        for gi, g in enumerate(gt_steps):
            if p.get("step_id") != g.get("step_id"):
                continue
            dist = abs(p.get("timestamp_sec", 0) - g.get("timestamp_sec", 0))
            if dist <= tolerance:
                step_pairs.append((pi, gi, dist))
    step_matches = _min_distance_match_detailed(step_pairs)
    matched_gt_steps = {gi for gi, _ in step_matches.values()}

    for pi, p in enumerate(pred_steps):
        if pi in step_matches:
            gi, delta = step_matches[pi]
            results.append({
                "source": "predicted", "event": p, "type": "step_completion",
                "match": "TP", "gt_event": gt_steps[gi], "delta": delta,
            })
        else:
            results.append({
                "source": "predicted", "event": p, "type": "step_completion",
                "match": "FP", "gt_event": None, "delta": None,
            })
    for gi, g in enumerate(gt_steps):
        if gi not in matched_gt_steps:
            results.append({
                "source": "ground_truth", "event": g, "type": "step_completion",
                "match": "FN", "gt_event": g, "delta": None,
            })

    # --- Errors (match by timestamp) ---
    error_pairs = []
    for pi, p in enumerate(pred_errors):
        for gi, g in enumerate(gt_errors):
            dist = abs(p.get("timestamp_sec", 0) - g.get("timestamp_sec", 0))
            if dist <= tolerance:
                error_pairs.append((pi, gi, dist))
    error_matches = _min_distance_match_detailed(error_pairs)
    matched_gt_errors = {gi for gi, _ in error_matches.values()}

    for pi, p in enumerate(pred_errors):
        if pi in error_matches:
            gi, delta = error_matches[pi]
            results.append({
                "source": "predicted", "event": p, "type": "error_detected",
                "match": "TP", "gt_event": gt_errors[gi], "delta": delta,
            })
        else:
            results.append({
                "source": "predicted", "event": p, "type": "error_detected",
                "match": "FP", "gt_event": None, "delta": None,
            })
    for gi, g in enumerate(gt_errors):
        if gi not in matched_gt_errors:
            results.append({
                "source": "ground_truth", "event": g, "type": "error_detected",
                "match": "FN", "gt_event": g, "delta": None,
            })

    # --- Idles (match by overlap) ---
    idle_pairs = []
    for pi, p in enumerate(pred_idles):
        t = p.get("timestamp_sec", 0)
        for gi, g in enumerate(gt_idles):
            start = g.get("start_sec", 0)
            end = g.get("end_sec", 0)
            if start <= t <= end:
                midpoint = (start + end) / 2
                dist = abs(t - midpoint)
                idle_pairs.append((pi, gi, dist))
    idle_matches = _min_distance_match_detailed(idle_pairs)
    matched_gt_idles = {gi for gi, _ in idle_matches.values()}

    for pi, p in enumerate(pred_idles):
        if pi in idle_matches:
            gi, _ = idle_matches[pi]
            results.append({
                "source": "predicted", "event": p, "type": "idle_detected",
                "match": "TP", "gt_event": gt_idles[gi], "delta": None,
            })
        else:
            results.append({
                "source": "predicted", "event": p, "type": "idle_detected",
                "match": "FP", "gt_event": None, "delta": None,
            })
    for gi, g in enumerate(gt_idles):
        if gi not in matched_gt_idles:
            results.append({
                "source": "ground_truth",
                "event": {"timestamp_sec": g["start_sec"], "type": "idle_period",
                          "duration_sec": g.get("duration_sec")},
                "type": "idle_detected", "match": "FN", "gt_event": g, "delta": None,
            })

    results.sort(key=lambda r: r["event"].get("timestamp_sec", 0))
    return results


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_STEP_COLORS = [
    "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b",
    "#ef4444", "#ec4899", "#6366f1", "#14b8a6", "#f97316",
    "#84cc16", "#a855f7", "#0ea5e9", "#22c55e", "#e11d48",
]


def _step_color(idx: int) -> str:
    return _STEP_COLORS[idx % len(_STEP_COLORS)]


def _step_color_dim(idx: int) -> str:
    """Dimmed version for the band background."""
    base = _STEP_COLORS[idx % len(_STEP_COLORS)]
    # Return with opacity via rgba
    r = int(base[1:3], 16)
    g = int(base[3:5], 16)
    b = int(base[5:7], 16)
    return f"rgba({r},{g},{b},0.35)"


def _score_color(val: float, good: float = 0.7, mid: float = 0.4) -> str:
    if val >= good:
        return "#22c55e"
    if val >= mid:
        return "#eab308"
    return "#ef4444"


def _fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# SVG Timeline builder
# ---------------------------------------------------------------------------

def _tip_json(data: Dict) -> str:
    """Encode a dict as an HTML-safe JSON string for data-tip attributes."""
    return _esc_attr(json.dumps(data, ensure_ascii=True))


def _build_timeline_svg(
    duration: float,
    proc_steps: List[Dict],
    gt_events: List[Dict],
    gt_idles: List[Dict],
    match_details: List[Dict],
) -> str:
    """Build SVG timeline with 2 lanes: Ground Truth (steps + errors) and Predictions."""
    if duration <= 0:
        duration = 1.0

    margin_left = 120
    margin_right = 30
    lane_height = 50
    lane_gap = 8
    lane_label_x = 8
    marker_radius = 8
    num_lanes = 2
    axis_height = 28
    top_pad = 10

    total_height = top_pad + num_lanes * (lane_height + lane_gap) + axis_height + 10

    # Filter to only steps + errors (no idle)
    pred_steps_errors = [
        r for r in match_details
        if r["source"] == "predicted" and r["type"] in ("step_completion", "error_detected")
    ]

    svg_parts = []

    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" id="tl-svg" '
        f'viewBox="0 0 1100 {total_height}" '
        f'style="width:100%;height:auto;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;" '
        f'preserveAspectRatio="xMinYMin meet">'
    )

    svg_parts.append("""<defs>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#000" flood-opacity="0.5"/>
        </filter>
    </defs>""")

    chart_width = 1100 - margin_left - margin_right

    def x_pos(sec: float) -> float:
        return margin_left + (sec / duration) * chart_width

    def lane_y(lane_idx: int) -> float:
        return top_pad + lane_idx * (lane_height + lane_gap)

    # Lane backgrounds
    lane_labels = ["Ground Truth", "Predictions"]
    for i, label in enumerate(lane_labels):
        y = lane_y(i)
        svg_parts.append(
            f'<rect x="{margin_left}" y="{y}" width="{chart_width}" height="{lane_height}" '
            f'rx="4" fill="#1e293b" stroke="#334155" stroke-width="0.5"/>'
        )
        svg_parts.append(
            f'<text x="{lane_label_x}" y="{y + lane_height / 2 + 4}" '
            f'fill="#94a3b8" font-size="11" font-weight="500">{_esc(label)}</text>'
        )

    # ================================================================
    # Lane 0: Ground Truth — step bands + step completion diamonds + error triangles
    # ================================================================
    ly0 = lane_y(0)
    center_y0 = ly0 + lane_height / 2

    # Step bands
    for idx, step in enumerate(proc_steps):
        s = step.get("start_sec", 0)
        e = step.get("end_sec", 0)
        sx = x_pos(s)
        ex = x_pos(e)
        w = max(ex - sx, 2)
        color = _step_color_dim(idx)
        border_color = _step_color(idx)
        sid = step.get("step_id", idx + 1)
        exp_dur = step.get("expected_duration_s", "")

        tip = _tip_json({
            "lane": "ground_truth", "type": "step",
            "step_id": sid, "description": step.get("description", ""),
            "start": _fmt_time(s), "end": _fmt_time(e),
            "duration": f"{e - s:.1f}s",
            "expected_duration": f"{exp_dur}s" if exp_dur else "",
        })

        svg_parts.append(
            f'<rect class="tip" data-tip="{tip}" x="{sx}" y="{ly0}" width="{w}" height="{lane_height}" '
            f'fill="transparent" cursor="pointer"/>'
        )
        svg_parts.append(
            f'<rect x="{sx}" y="{ly0 + 4}" width="{w}" height="{lane_height - 8}" '
            f'rx="3" fill="{color}" stroke="{border_color}" stroke-width="1.5" pointer-events="none"/>'
        )
        if w > 30:
            svg_parts.append(
                f'<text x="{sx + w / 2}" y="{ly0 + lane_height / 2 + 4}" '
                f'fill="#e2e8f0" font-size="10" font-weight="600" text-anchor="middle" pointer-events="none">'
                f'S{sid}</text>'
            )

    # GT step completion markers (blue diamonds on top of bands)
    gt_step_events = [e for e in gt_events if e.get("type") == "step_completion"]
    for evt in gt_step_events:
        t = evt.get("timestamp_sec", 0)
        cx = x_pos(t)
        sid = evt.get("step_id", "")
        tip = _tip_json({
            "lane": "ground_truth", "type": "step_completion",
            "step_id": sid, "timestamp": _fmt_time(t),
            "description": evt.get("description", ""),
        })
        svg_parts.append(
            f'<g class="tip" data-tip="{tip}" transform="translate({cx},{center_y0})" cursor="pointer">'
            f'<circle r="{marker_radius + 4}" fill="transparent"/>'
            f'<polygon points="0,-{marker_radius} {marker_radius},0 0,{marker_radius} -{marker_radius},0" '
            f'fill="#3b82f6" stroke="#93c5fd" stroke-width="1.5" filter="url(#shadow)"/>'
            f'<text y="1" fill="#fff" font-size="8" font-weight="700" text-anchor="middle" '
            f'dominant-baseline="middle">{sid}</text></g>'
        )

    # GT error markers (red triangles)
    gt_error_events = [e for e in gt_events if e.get("type") == "error_detected"]
    for evt in gt_error_events:
        t = evt.get("timestamp_sec", 0)
        cx = x_pos(t)
        tip = _tip_json({
            "lane": "ground_truth", "type": "error_detected",
            "timestamp": _fmt_time(t),
            "description": evt.get("description", ""),
            "error_type": evt.get("error_type", ""),
            "severity": evt.get("severity", ""),
            "correction": evt.get("correction", ""),
        })
        r = marker_radius + 1
        svg_parts.append(
            f'<g class="tip" data-tip="{tip}" transform="translate({cx},{center_y0})" cursor="pointer">'
            f'<circle r="{r + 4}" fill="transparent"/>'
            f'<polygon points="0,-{r} {r},{r} -{r},{r}" '
            f'fill="#ef4444" stroke="#fca5a5" stroke-width="1.5" filter="url(#shadow)"/>'
            f'<text y="2" fill="#fff" font-size="7" font-weight="700" text-anchor="middle" '
            f'dominant-baseline="middle">!</text></g>'
        )

    # ================================================================
    # Lane 1: Predictions — step completions + errors (TP/FP)
    # ================================================================
    ly1 = lane_y(1)
    center_y1 = ly1 + lane_height / 2

    for r in pred_steps_errors:
        evt = r["event"]
        t = evt.get("timestamp_sec", 0)
        cx = x_pos(t)
        etype = evt.get("type", "")
        is_tp = r["match"] == "TP"
        border = "#22c55e" if is_tp else "#f97316"
        status = r["match"]

        tip_data = {
            "lane": "predicted", "type": etype, "match": status,
            "timestamp": _fmt_time(t),
            "description": evt.get("description", ""),
            "vlm_observation": evt.get("vlm_observation", ""),
            "confidence": evt.get("confidence", ""),
            "detection_delay": f"{evt.get('detection_delay_sec', '')}",
            "source": evt.get("source", ""),
            "spoken_response": evt.get("spoken_response", ""),
            "error_type": evt.get("error_type", ""),
            "severity": evt.get("severity", ""),
            "step_id": evt.get("step_id", ""),
        }
        if r.get("gt_event"):
            gt = r["gt_event"]
            tip_data["gt_description"] = gt.get("description", "")
            gt_ts = gt.get("timestamp_sec", gt.get("start_sec"))
            if gt_ts is not None:
                tip_data["gt_timestamp"] = _fmt_time(gt_ts)
        if r.get("delta") is not None:
            tip_data["delta"] = f"{r['delta']:.1f}s"
        tip = _tip_json(tip_data)

        if etype == "step_completion":
            sid = evt.get("step_id", "")
            svg_parts.append(
                f'<g class="tip" data-tip="{tip}" transform="translate({cx},{center_y1})" cursor="pointer">'
                f'<circle r="{marker_radius + 4}" fill="transparent"/>'
                f'<polygon points="0,-{marker_radius} {marker_radius},0 0,{marker_radius} -{marker_radius},0" '
                f'fill="#3b82f6" stroke="{border}" stroke-width="2" filter="url(#shadow)"/>'
                f'<text y="1" fill="#fff" font-size="8" font-weight="700" text-anchor="middle" '
                f'dominant-baseline="middle">{sid}</text></g>'
            )
        elif etype == "error_detected":
            er = marker_radius + 1
            svg_parts.append(
                f'<g class="tip" data-tip="{tip}" transform="translate({cx},{center_y1})" cursor="pointer">'
                f'<circle r="{er + 4}" fill="transparent"/>'
                f'<polygon points="0,-{er} {er},{er} -{er},{er}" '
                f'fill="#ef4444" stroke="{border}" stroke-width="2" filter="url(#shadow)"/>'
                f'<text y="2" fill="#fff" font-size="7" font-weight="700" text-anchor="middle" '
                f'dominant-baseline="middle">!</text></g>'
            )

    # ================================================================
    # Time axis
    # ================================================================
    axis_y = lane_y(num_lanes - 1) + lane_height + lane_gap + 2
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{axis_y}" x2="{margin_left + chart_width}" y2="{axis_y}" '
        f'stroke="#475569" stroke-width="1"/>'
    )

    if duration <= 60:
        tick_interval = 5
    elif duration <= 300:
        tick_interval = 15
    elif duration <= 600:
        tick_interval = 30
    elif duration <= 1800:
        tick_interval = 60
    else:
        tick_interval = 120

    t = 0.0
    while t <= duration:
        tx = x_pos(t)
        svg_parts.append(
            f'<line x1="{tx}" y1="{axis_y}" x2="{tx}" y2="{axis_y + 5}" stroke="#475569" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{tx}" y="{axis_y + 16}" fill="#64748b" font-size="10" text-anchor="middle">'
            f'{_fmt_time(t)}</text>'
        )
        svg_parts.append(
            f'<line x1="{tx}" y1="{top_pad}" x2="{tx}" y2="{axis_y}" '
            f'stroke="#334155" stroke-width="0.5" stroke-dasharray="2,4"/>'
        )
        t += tick_interval

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(predicted_path: str, gt_path: str, tolerance: float = 5.0) -> str:
    pred = _load(predicted_path)
    gt = _load(gt_path)

    metrics = evaluate(predicted_path, gt_path, tolerance)
    m = asdict(metrics)

    pred_events = pred.get("events", [])
    gt_events = gt.get("events", [])
    gt_idles = gt.get("idle_periods", [])
    gt_proc_steps = gt.get("procedure_steps", [])

    video_name = gt.get("video_name", "Unknown")
    task_type = gt.get("task_type", "Unknown")
    duration = gt.get("total_duration_sec", 0)

    match_details = _match_events(pred_events, gt_events, gt_idles, tolerance)

    timeline_svg = _build_timeline_svg(
        duration, gt_proc_steps, gt_events, gt_idles, match_details
    )

    esc = _esc

    # Score color helper
    def sc(val, good=0.7, mid=0.4):
        return _score_color(val, good, mid)

    # Latency color (lower is better)
    def lc(val):
        if val <= 3:
            return "#22c55e"
        if val <= 6:
            return "#eab308"
        return "#ef4444"

    # --- Build event detail table rows (steps + errors only, no idle/FN) ---
    detail_rows = []
    for r in [r for r in match_details if r["type"] in ("step_completion", "error_detected") and r["source"] == "predicted"]:
        evt = r["event"]
        t = evt.get("timestamp_sec", 0)
        etype = r["type"]
        match_label = r["match"]

        match_display = {"TP": "True Positive", "FP": "False Positive", "FN": "False Negative"}[match_label]
        badge_cls = {"TP": "badge-tp", "FP": "badge-fp", "FN": "badge-fn"}[match_label]
        type_display = {"step_completion": "user step completion", "error_detected": "user error"}.get(etype, etype.replace("_", " "))
        type_cls = {
            "step_completion": "badge-step",
            "error_detected": "badge-error",
        }.get(etype, "")

        # GT side
        gt_evt = r.get("gt_event")
        gt_desc = esc(gt_evt.get("description", "")) if gt_evt and isinstance(gt_evt, dict) else ""
        gt_time = ""
        if gt_evt and isinstance(gt_evt, dict):
            gt_ts = gt_evt.get("timestamp_sec", gt_evt.get("start_sec"))
            if gt_ts is not None:
                gt_time = f"{gt_ts:.1f}s"

        # Pred side
        pred_desc = esc(evt.get("description", "")) if r["source"] == "predicted" else ""
        vlm_obs = esc(evt.get("vlm_observation", "")) if r["source"] == "predicted" else ""
        spoken = esc(evt.get("spoken_response", "")) if r["source"] == "predicted" else ""
        source = esc(evt.get("source", "")) if r["source"] == "predicted" else ""
        conf = evt.get("confidence", "")
        conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) and conf != "" else esc(str(conf)) if conf else ""
        delay = evt.get("detection_delay_sec", "")
        delay_str = f"{delay:.1f}s" if isinstance(delay, (int, float)) and delay != "" else ""
        step_id = evt.get("step_id", "")
        delta = f"{r['delta']:.1f}s" if r["delta"] is not None else ""

        detail_rows.append(f"""<tr>
<td>{t:.1f}s</td>
<td>{gt_time}</td>
<td><span class="badge {type_cls}">{esc(type_display)}</span></td>
<td><span class="badge {badge_cls}">{esc(match_display)}</span></td>
<td>{gt_desc}</td>
<td>{pred_desc}</td>
<td class="vlm-obs">{vlm_obs}</td>
<td class="vlm-obs">{spoken}</td>
<td>{esc(str(source))}</td>
<td>{conf_str}</td>
<td>{delay_str}</td>
<td>{delta}</td>
</tr>""")

    detail_table = "\n".join(detail_rows)

    # --- Procedure steps list ---
    proc_html_parts = []
    for idx, s in enumerate(gt_proc_steps):
        color = _step_color(idx)
        sid = s.get("step_id", idx + 1)
        desc = esc(s.get("description", ""))
        start = s.get("start_sec", 0)
        end = s.get("end_sec", 0)
        proc_html_parts.append(
            f'<div class="proc-step">'
            f'<span class="proc-dot" style="background:{color}"></span>'
            f'<strong>Step {sid}</strong>: {desc} '
            f'<span class="proc-time">{_fmt_time(start)} &ndash; {_fmt_time(end)}</span>'
            f'</div>'
        )
    proc_html = "\n".join(proc_html_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation: {esc(video_name)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 24px;
    line-height: 1.5;
}}
.container {{ max-width: 1300px; margin: 0 auto; }}

/* Header */
h1 {{ font-size: 22px; margin-bottom: 4px; color: #f8fafc; }}
.subtitle {{ color: #94a3b8; margin-bottom: 20px; font-size: 14px; }}

/* Score Banner */
.score-banner {{
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 12px; padding: 24px; margin-bottom: 20px;
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;
    border: 1px solid #475569;
}}
.score-item {{ text-align: center; }}
.score-val {{ font-size: 42px; font-weight: 700; line-height: 1.1; }}
.score-label {{ font-size: 12px; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}

/* F1 explainer */
.f1-explainer {{
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    padding: 10px 16px; margin-bottom: 16px; font-size: 12px;
    color: #94a3b8; line-height: 1.6;
}}
.f1-explainer strong {{ color: #e2e8f0; }}

/* Metrics Grid */
.metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px; }}
.metric-card {{
    background: #1e293b; border-radius: 8px; padding: 16px;
    border: 1px solid #334155;
}}
.metric-card h3 {{
    font-size: 11px; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 8px;
}}
.metric-val {{ font-size: 28px; font-weight: 600; }}
.metric-detail {{ font-size: 12px; color: #64748b; margin-top: 4px; }}

/* Section */
.section {{ margin-bottom: 28px; }}
.section h2 {{
    font-size: 15px; margin-bottom: 12px; color: #f8fafc;
    text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid #334155; padding-bottom: 8px;
}}

/* Timeline */
.timeline-wrap {{
    background: #0f172a; border-radius: 8px;
    border: 1px solid #334155; padding: 16px 12px;
    overflow-x: auto;
}}
.timeline-wrap svg {{ min-width: 900px; }}

/* Legend */
.legend {{
    display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px;
    font-size: 12px; color: #94a3b8;
}}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-swatch {{
    width: 14px; height: 14px; border-radius: 3px;
    flex-shrink: 0;
}}

/* Procedure steps */
.proc-list {{ display: flex; flex-wrap: wrap; gap: 8px 24px; }}
.proc-step {{
    font-size: 13px; display: flex; align-items: center; gap: 6px;
}}
.proc-dot {{
    width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
}}
.proc-time {{ color: #64748b; }}

/* Table */
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; white-space: nowrap; }}
th {{
    background: #1e293b; padding: 8px 10px; text-align: left;
    font-weight: 500; color: #94a3b8; position: sticky; top: 0;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px;
}}
td {{ padding: 7px 10px; border-bottom: 1px solid #1e293b; }}
tr:hover {{ background: rgba(30,41,59,0.5); }}

/* Badges */
.badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 10px; font-weight: 600;
}}
.badge-tp {{ background: #166534; color: #86efac; }}
.badge-fp {{ background: #7c2d12; color: #fdba74; }}
.badge-fn {{ background: #7f1d1d; color: #fca5a5; }}
.badge-step {{ background: #1e3a5f; color: #93c5fd; }}
.badge-error {{ background: #5b2130; color: #fda4af; }}
.badge-idle {{ background: #3b3520; color: #fde68a; }}

.vlm-obs {{
    color: #64748b; font-style: italic; max-width: 250px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}

/* JSON block */
details summary {{
    cursor: pointer; color: #94a3b8; font-size: 13px;
    padding: 8px 0;
}}
details summary:hover {{ color: #e2e8f0; }}
pre.json-block {{
    background: #1e293b; padding: 16px; border-radius: 8px;
    overflow-x: auto; font-size: 12px; margin-top: 8px;
    max-height: 400px; overflow-y: auto; color: #cbd5e1;
    border: 1px solid #334155;
}}

/* Footer */
.footer {{
    color: #475569; font-size: 11px; margin-top: 32px;
    padding-top: 12px; border-top: 1px solid #1e293b;
}}

/* Tooltip popover */
#tip-popover {{
    position: fixed; z-index: 1000; pointer-events: none;
    background: #1e293b; border: 1px solid #475569; border-radius: 8px;
    padding: 12px 14px; max-width: 420px; min-width: 220px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    font-size: 12px; line-height: 1.6; color: #e2e8f0;
    opacity: 0; transition: opacity 0.15s;
}}
#tip-popover.visible {{ opacity: 1; }}
#tip-popover .tip-header {{
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
    padding-bottom: 6px; border-bottom: 1px solid #334155;
}}
#tip-popover .tip-badge {{
    padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700;
    text-transform: uppercase;
}}
#tip-popover .tip-badge-tp {{ background: #166534; color: #86efac; }}
#tip-popover .tip-badge-fp {{ background: #7c2d12; color: #fdba74; }}
#tip-popover .tip-badge-fn {{ background: #7f1d1d; color: #fca5a5; }}
#tip-popover .tip-badge-gt {{ background: #1e3a5f; color: #93c5fd; }}
#tip-popover .tip-badge-proc {{ background: #3b3520; color: #fde68a; }}
#tip-popover .tip-type {{ font-weight: 600; color: #f8fafc; }}
#tip-popover .tip-time {{ color: #94a3b8; font-size: 11px; margin-left: auto; }}
#tip-popover .tip-row {{
    display: flex; gap: 6px; color: #cbd5e1;
}}
#tip-popover .tip-label {{
    color: #64748b; min-width: 80px; flex-shrink: 0; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.3px;
}}
#tip-popover .tip-val {{ word-break: break-word; white-space: pre-wrap; }}
#tip-popover .tip-val.vlm {{ color: #94a3b8; font-style: italic; }}

/* Detail panel (click to pin) */
#detail-panel {{
    background: #1e293b; border: 1px solid #475569; border-radius: 8px;
    padding: 16px; margin-top: 12px; display: none;
    font-size: 12px; line-height: 1.6; color: #e2e8f0;
}}
#detail-panel.active {{ display: block; }}
#detail-panel .dp-header {{
    display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
    padding-bottom: 8px; border-bottom: 1px solid #334155;
}}
#detail-panel .dp-close {{
    margin-left: auto; cursor: pointer; color: #64748b; font-size: 18px;
    line-height: 1; padding: 0 4px;
}}
#detail-panel .dp-close:hover {{ color: #e2e8f0; }}
#detail-panel .dp-grid {{
    display: grid; grid-template-columns: auto 1fr; gap: 4px 12px;
}}
#detail-panel .dp-label {{
    color: #64748b; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.3px; padding-top: 1px;
}}
#detail-panel .dp-val {{ color: #cbd5e1; word-break: break-word; }}
#detail-panel .dp-val.vlm {{ color: #94a3b8; font-style: italic; }}

/* Responsive */
@media (max-width: 900px) {{
    .score-banner {{ grid-template-columns: repeat(2, 1fr); }}
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>
<div class="container">

<h1>Evaluation Report: {esc(video_name)}</h1>
<p class="subtitle">{esc(task_type.title())} &middot; {duration:.0f}s &middot; Tolerance &plusmn;{tolerance:.0f}s</p>

<!-- Score Banner -->
<div class="score-banner">
    <div class="score-item">
        <div class="score-val" style="color:{sc(m['step_f1'])}">{m['step_f1']:.0%}</div>
        <div class="score-label">Step F1 Score</div>
    </div>
    <div class="score-item">
        <div class="score-val" style="color:{sc(m['error_f1'])}">{m['error_f1']:.0%}</div>
        <div class="score-label">Error F1 Score</div>
    </div>
    <div class="score-item">
        <div class="score-val" style="color:{lc(m['mean_detection_delay_sec'])}">{m['mean_detection_delay_sec']:.1f}s</div>
        <div class="score-label">Mean Latency</div>
    </div>
</div>

<div class="f1-explainer">
    <strong>F1 Score</strong> = 2 &times; (Precision &times; Recall) / (Precision + Recall) &mdash;
    the harmonic mean of Precision (what fraction of your detections were correct) and Recall (what fraction of ground truth events you detected).
    Ranges from 0% (no correct detections) to 100% (perfect). Predictions match ground truth within &plusmn;{tolerance:.0f}s.
</div>

<!-- Metrics Grid -->
<div class="metrics-grid">
    <div class="metric-card">
        <h3>Steps</h3>
        <div class="metric-val" style="color:{sc(m['step_f1'])}">{m['step_tp']}/{m['total_gt_steps']}</div>
        <div class="metric-detail">Precision={m['step_precision']:.0%} Recall={m['step_recall']:.0%} | {m['step_fp']} False Pos, {m['step_fn']} False Neg</div>
    </div>
    <div class="metric-card">
        <h3>Errors</h3>
        <div class="metric-val" style="color:{sc(m['error_f1'])}">{m['error_tp']}/{m['total_gt_errors']}</div>
        <div class="metric-detail">Precision={m['error_precision']:.0%} Recall={m['error_recall']:.0%} | {m['error_fp']} False Pos, {m['error_fn']} False Neg</div>
    </div>
    <div class="metric-card">
        <h3>Latency</h3>
        <div class="metric-val">{m['mean_detection_delay_sec']:.1f}s</div>
        <div class="metric-detail">p50={m['p50_detection_delay_sec']:.1f}s p90={m['p90_detection_delay_sec']:.1f}s max={m['max_detection_delay_sec']:.1f}s</div>
    </div>
</div>

<!-- Timeline -->
<div class="section">
<h2>Timeline</h2>
<div class="legend">
    <div class="legend-item"><svg width="14" height="14"><polygon points="7,0 14,7 7,14 0,7" fill="#3b82f6" stroke="#93c5fd" stroke-width="1"/></svg> User step completion</div>
    <div class="legend-item"><svg width="14" height="14"><polygon points="7,1 13,13 1,13" fill="#ef4444" stroke="#fca5a5" stroke-width="1"/></svg> User error</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#166534;border:2px solid #22c55e"></div> True Positive (matched GT)</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#7c2d12;border:2px solid #f97316"></div> False Positive (no GT match)</div>
</div>
<div class="timeline-wrap">
{timeline_svg}
</div>
<div id="detail-panel">
    <div class="dp-header">
        <span id="dp-badge"></span>
        <span id="dp-type" style="font-weight:600;color:#f8fafc"></span>
        <span id="dp-time" style="color:#94a3b8;font-size:11px;margin-left:auto"></span>
        <span class="dp-close" onclick="document.getElementById('detail-panel').classList.remove('active')">&times;</span>
    </div>
    <div class="dp-grid" id="dp-grid"></div>
</div>
</div>

<div id="tip-popover"></div>

<!-- Procedure Steps -->
<div class="section">
<h2>Procedure Steps</h2>
<div class="proc-list">
{proc_html}
</div>
</div>

<!-- Event Detail Table -->
<div class="section">
<h2>Event Detail</h2>
<div class="table-wrap">
<table>
<thead><tr>
<th>Time</th>
<th>GT Time</th>
<th>Type</th>
<th>Result</th>
<th>GT Description</th>
<th>Pred Description</th>
<th>VLM Observation</th>
<th>Spoken Response</th>
<th>Source</th>
<th>Conf</th>
<th>Delay</th>
<th>&Delta; to GT</th>
</tr></thead>
<tbody>
{detail_table}
</tbody>
</table>
</div>
</div>

<!-- Raw JSON -->
<div class="section">
<h2>Raw Data</h2>
<details>
<summary>Predicted JSON</summary>
<pre class="json-block">{esc(json.dumps(pred, indent=2))}</pre>
</details>
<details>
<summary>Ground Truth JSON</summary>
<pre class="json-block">{esc(json.dumps(gt, indent=2))}</pre>
</details>
</div>

<p class="footer">Generated by VLM Orchestrator Evaluator &middot; Alcor Labs</p>
</div>

<script>
(function() {{
    const pop = document.getElementById('tip-popover');
    const panel = document.getElementById('detail-panel');
    const dpBadge = document.getElementById('dp-badge');
    const dpType = document.getElementById('dp-type');
    const dpTime = document.getElementById('dp-time');
    const dpGrid = document.getElementById('dp-grid');

    // --- Field display config ---
    const FIELDS = [
        ['description', 'Description'],
        ['gt_description', 'GT Expected'],
        ['gt_timestamp', 'GT Time'],
        ['step_id', 'Step ID'],
        ['start', 'Start'],
        ['end', 'End'],
        ['duration', 'Duration'],
        ['expected_duration', 'Expected Dur.'],
        ['match', 'Match Result'],
        ['delta', 'Delta to GT'],
        ['confidence', 'Confidence'],
        ['detection_delay', 'Detect Delay'],
        ['source', 'Source'],
        ['error_type', 'Error Type'],
        ['severity', 'Severity'],
        ['vlm_observation', 'VLM Output', true],
        ['spoken_response', 'Spoken Resp.'],
        ['correction', 'Correction'],
    ];

    function badgeFor(d) {{
        const lane = d.lane || '';
        const match = d.match || '';
        if (match === 'TP') return '<span class="tip-badge tip-badge-tp">True Positive</span>';
        if (match === 'FP') return '<span class="tip-badge tip-badge-fp">False Positive</span>';
        if (match === 'FN') return '<span class="tip-badge tip-badge-fn">False Negative</span>';
        if (lane === 'ground_truth') return '<span class="tip-badge tip-badge-gt">GT</span>';
        if (lane === 'procedure') return '<span class="tip-badge tip-badge-proc">Procedure</span>';
        return '';
    }}

    function typeLabel(d) {{
        var t = (d.type || '').replace(/_/g, ' ');
        if (t === 'step completion' || t === 'error detected') t = 'user ' + t;
        return t;
    }}

    function timeLabel(d) {{
        if (d.timestamp) return d.timestamp;
        if (d.start && d.end) return d.start + ' – ' + d.end;
        return '';
    }}

    function buildRows(d, asHtml) {{
        let html = '';
        FIELDS.forEach(function(f) {{
            const key = f[0], label = f[1], isVlm = f[2] || false;
            let val = d[key];
            if (val === undefined || val === null || val === '' || val === 0 && key !== 'step_id') return;
            val = String(val);
            if (asHtml) {{
                html += '<div class="dp-label">' + label + '</div>';
                html += '<div class="dp-val' + (isVlm ? ' vlm' : '') + '">' + esc(val) + '</div>';
            }} else {{
                html += '<div class="tip-row"><span class="tip-label">' + label + '</span>';
                html += '<span class="tip-val' + (isVlm ? ' vlm' : '') + '">' + esc(truncate(val, 120)) + '</span></div>';
            }}
        }});
        return html;
    }}

    function esc(s) {{
        const el = document.createElement('span');
        el.textContent = s;
        return el.innerHTML;
    }}

    function truncate(s, n) {{
        return s.length > n ? s.slice(0, n) + '…' : s;
    }}

    // --- Hover tooltip ---
    document.querySelectorAll('.tip').forEach(function(el) {{
        el.addEventListener('mouseenter', function(e) {{
            const d = JSON.parse(el.getAttribute('data-tip'));
            let html = '<div class="tip-header">';
            html += badgeFor(d);
            html += '<span class="tip-type">' + esc(typeLabel(d)) + '</span>';
            html += '<span class="tip-time">' + esc(timeLabel(d)) + '</span>';
            html += '</div>';
            html += buildRows(d, false);
            pop.innerHTML = html;
            pop.classList.add('visible');
        }});
        el.addEventListener('mousemove', function(e) {{
            const pw = pop.offsetWidth, ph = pop.offsetHeight;
            let x = e.clientX + 14, y = e.clientY + 14;
            if (x + pw > window.innerWidth - 8) x = e.clientX - pw - 10;
            if (y + ph > window.innerHeight - 8) y = e.clientY - ph - 10;
            pop.style.left = x + 'px';
            pop.style.top = y + 'px';
        }});
        el.addEventListener('mouseleave', function() {{
            pop.classList.remove('visible');
        }});

        // --- Click to pin detail panel ---
        el.addEventListener('click', function() {{
            const d = JSON.parse(el.getAttribute('data-tip'));
            dpBadge.innerHTML = badgeFor(d);
            dpType.textContent = typeLabel(d);
            dpTime.textContent = timeLabel(d);
            dpGrid.innerHTML = buildRows(d, true);
            panel.classList.add('active');
            panel.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        }});
    }});
}})();
</script>
</body>
</html>"""

    return html


def generate_multi_html(
    clips: List[Dict[str, str]],
    tolerance: float = 5.0,
) -> str:
    """
    Generate a multi-clip dashboard with tabs.

    Args:
        clips: list of {"name": "clip_name", "predicted": "path", "ground_truth": "path"}
        tolerance: timestamp matching tolerance in seconds
    """
    esc = _esc

    # Generate per-clip content
    tab_headers = []
    tab_bodies = []
    for i, clip in enumerate(clips):
        name = clip["name"]
        active = " active" if i == 0 else ""

        # Load GT for summary info
        gt = _load(clip["ground_truth"])
        metrics = evaluate(clip["predicted"], clip["ground_truth"], tolerance)

        tab_headers.append(
            f'<div class="tab{active}" data-tab="tab-{i}" onclick="switchTab(this)">'
            f'<div class="tab-name">{esc(name)}</div>'
            f'<div class="tab-scores">'
            f'<span style="color:{_score_color(metrics.step_f1)}">S:{metrics.step_f1:.0%}</span> '
            f'<span style="color:{_score_color(metrics.error_f1)}">E:{metrics.error_f1:.0%}</span>'
            f'</div></div>'
        )

        # Generate full single-clip HTML and extract just the <body> content
        single_html = generate_html(clip["predicted"], clip["ground_truth"], tolerance)
        # Extract content between <div class="container"> and closing </div>\n</body>
        start_marker = '<div class="container">'
        end_marker = '\n<script>'
        body_start = single_html.index(start_marker) + len(start_marker)
        body_end = single_html.index(end_marker)
        body_content = single_html[body_start:body_end]

        display = "block" if i == 0 else "none"
        tab_bodies.append(
            f'<div class="tab-panel" id="tab-{i}" style="display:{display}">'
            f'{body_content}</div>'
        )

    tabs_html = "\n".join(tab_headers)
    panels_html = "\n".join(tab_bodies)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Dashboard — {len(clips)} clips</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 0; line-height: 1.5;
}}
.multi-header {{
    background: #1e293b; border-bottom: 1px solid #334155;
    padding: 16px 24px 0;
}}
.multi-header h1 {{ font-size: 18px; margin-bottom: 12px; color: #f8fafc; }}
.tabs {{
    display: flex; gap: 4px; overflow-x: auto; padding-bottom: 0;
}}
.tab {{
    padding: 8px 16px; cursor: pointer; border-radius: 6px 6px 0 0;
    background: #0f172a; border: 1px solid #334155; border-bottom: none;
    font-size: 12px; white-space: nowrap; transition: background 0.15s;
    min-width: 120px;
}}
.tab:hover {{ background: #334155; }}
.tab.active {{ background: #0f172a; border-color: #475569; border-bottom: 2px solid #0f172a; margin-bottom: -1px; }}
.tab-name {{ font-weight: 600; color: #e2e8f0; margin-bottom: 2px; }}
.tab-scores {{ font-size: 11px; color: #94a3b8; }}
.tab-scores span {{ font-weight: 600; }}
.container {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}

/* Import all styles from single-clip dashboard */
h1 {{ font-size: 22px; margin-bottom: 4px; color: #f8fafc; }}
.subtitle {{ color: #94a3b8; margin-bottom: 20px; font-size: 14px; }}
.score-banner {{
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 12px; padding: 24px; margin-bottom: 20px;
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;
    border: 1px solid #475569;
}}
.score-item {{ text-align: center; }}
.score-val {{ font-size: 42px; font-weight: 700; line-height: 1.1; }}
.score-label {{ font-size: 12px; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
.f1-explainer {{
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    padding: 10px 16px; margin-bottom: 16px; font-size: 12px;
    color: #94a3b8; line-height: 1.6;
}}
.f1-explainer strong {{ color: #e2e8f0; }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px; }}
.metric-card {{ background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; }}
.metric-card h3 {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
.metric-val {{ font-size: 28px; font-weight: 600; }}
.metric-detail {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
.section {{ margin-bottom: 28px; }}
.section h2 {{ font-size: 15px; margin-bottom: 12px; color: #f8fafc; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
.timeline-wrap {{ background: #0f172a; border-radius: 8px; border: 1px solid #334155; padding: 16px 12px; overflow-x: auto; }}
.timeline-wrap svg {{ min-width: 900px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; font-size: 12px; color: #94a3b8; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }}
.proc-list {{ display: flex; flex-wrap: wrap; gap: 8px 24px; }}
.proc-step {{ font-size: 13px; display: flex; align-items: center; gap: 6px; }}
.proc-dot {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
.proc-time {{ color: #64748b; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; white-space: nowrap; }}
th {{ background: #1e293b; padding: 8px 10px; text-align: left; font-weight: 500; color: #94a3b8; position: sticky; top: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }}
td {{ padding: 7px 10px; border-bottom: 1px solid #1e293b; }}
tr:hover {{ background: rgba(30,41,59,0.5); }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
.badge-tp {{ background: #166534; color: #86efac; }}
.badge-fp {{ background: #7c2d12; color: #fdba74; }}
.badge-fn {{ background: #7f1d1d; color: #fca5a5; }}
.badge-step {{ background: #1e3a5f; color: #93c5fd; }}
.badge-error {{ background: #5b2130; color: #fda4af; }}
.badge-idle {{ background: #3b3520; color: #fde68a; }}
.vlm-obs {{ color: #64748b; font-style: italic; max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
details summary {{ cursor: pointer; color: #94a3b8; font-size: 13px; padding: 8px 0; }}
details summary:hover {{ color: #e2e8f0; }}
pre.json-block {{ background: #1e293b; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 12px; margin-top: 8px; max-height: 400px; overflow-y: auto; color: #cbd5e1; border: 1px solid #334155; }}
.footer {{ color: #475569; font-size: 11px; margin-top: 32px; padding-top: 12px; border-top: 1px solid #1e293b; }}
#tip-popover {{ position: fixed; z-index: 1000; pointer-events: none; background: #1e293b; border: 1px solid #475569; border-radius: 8px; padding: 12px 14px; max-width: 420px; min-width: 220px; box-shadow: 0 8px 30px rgba(0,0,0,0.5); font-size: 12px; line-height: 1.6; color: #e2e8f0; opacity: 0; transition: opacity 0.15s; }}
#tip-popover.visible {{ opacity: 1; }}
#tip-popover .tip-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; padding-bottom: 6px; border-bottom: 1px solid #334155; }}
#tip-popover .tip-badge {{ padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase; }}
#tip-popover .tip-badge-tp {{ background: #166534; color: #86efac; }}
#tip-popover .tip-badge-fp {{ background: #7c2d12; color: #fdba74; }}
#tip-popover .tip-badge-fn {{ background: #7f1d1d; color: #fca5a5; }}
#tip-popover .tip-badge-gt {{ background: #1e3a5f; color: #93c5fd; }}
#tip-popover .tip-badge-proc {{ background: #3b3520; color: #fde68a; }}
#tip-popover .tip-type {{ font-weight: 600; color: #f8fafc; }}
#tip-popover .tip-time {{ color: #94a3b8; font-size: 11px; margin-left: auto; }}
#tip-popover .tip-row {{ display: flex; gap: 6px; color: #cbd5e1; }}
#tip-popover .tip-label {{ color: #64748b; min-width: 80px; flex-shrink: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }}
#tip-popover .tip-val {{ word-break: break-word; white-space: pre-wrap; }}
#tip-popover .tip-val.vlm {{ color: #94a3b8; font-style: italic; }}
#detail-panel {{ background: #1e293b; border: 1px solid #475569; border-radius: 8px; padding: 16px; margin-top: 12px; display: none; font-size: 12px; line-height: 1.6; color: #e2e8f0; }}
#detail-panel.active {{ display: block; }}
#detail-panel .dp-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #334155; }}
#detail-panel .dp-close {{ margin-left: auto; cursor: pointer; color: #64748b; font-size: 18px; line-height: 1; padding: 0 4px; }}
#detail-panel .dp-close:hover {{ color: #e2e8f0; }}
#detail-panel .dp-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; }}
#detail-panel .dp-label {{ color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; padding-top: 1px; }}
#detail-panel .dp-val {{ color: #cbd5e1; word-break: break-word; }}
#detail-panel .dp-val.vlm {{ color: #94a3b8; font-style: italic; }}
@media (max-width: 900px) {{ .score-banner {{ grid-template-columns: repeat(2, 1fr); }} .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>

<div class="multi-header">
<h1>Evaluation Dashboard &mdash; {len(clips)} clips</h1>
<div class="tabs">
{tabs_html}
</div>
</div>

<div class="container">
{panels_html}
</div>

<div id="tip-popover"></div>

<script>
function switchTab(el) {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
    el.classList.add('active');
    document.getElementById(el.getAttribute('data-tab')).style.display = 'block';
    // Close any open detail panel
    document.querySelectorAll('#detail-panel').forEach(p => p.classList.remove('active'));
}}

// Tooltip + detail panel logic
(function() {{
    const pop = document.getElementById('tip-popover');
    const FIELDS = [
        ['description', 'Description'],
        ['gt_description', 'GT Expected'],
        ['gt_timestamp', 'GT Time'],
        ['step_id', 'Step ID'],
        ['start', 'Start'], ['end', 'End'],
        ['duration', 'Duration'], ['expected_duration', 'Expected Dur.'],
        ['match', 'Match Result'], ['delta', 'Delta to GT'],
        ['confidence', 'Confidence'], ['detection_delay', 'Detect Delay'],
        ['source', 'Source'], ['error_type', 'Error Type'], ['severity', 'Severity'],
        ['vlm_observation', 'VLM Output', true],
        ['spoken_response', 'Spoken Resp.'], ['correction', 'Correction'],
    ];
    function badgeFor(d) {{
        var m = d.match || '', l = d.lane || '';
        if (m === 'TP') return '<span class="tip-badge tip-badge-tp">True Positive</span>';
        if (m === 'FP') return '<span class="tip-badge tip-badge-fp">False Positive</span>';
        if (m === 'FN') return '<span class="tip-badge tip-badge-fn">False Negative</span>';
        if (l === 'ground_truth') return '<span class="tip-badge tip-badge-gt">GT</span>';
        if (l === 'procedure') return '<span class="tip-badge tip-badge-proc">Procedure</span>';
        return '';
    }}
    function typeLabel(d) {{
        var t = (d.type || '').replace(/_/g, ' ');
        if (t === 'step completion' || t === 'error detected') t = 'user ' + t;
        return t;
    }}
    function timeLabel(d) {{ return d.timestamp || (d.start && d.end ? d.start + ' – ' + d.end : ''); }}
    function esc(s) {{ var el = document.createElement('span'); el.textContent = s; return el.innerHTML; }}
    function truncate(s, n) {{ return s.length > n ? s.slice(0, n) + '…' : s; }}
    function buildRows(d, full) {{
        var h = '';
        FIELDS.forEach(function(f) {{
            var k = f[0], lbl = f[1], vlm = f[2] || false, v = d[k];
            if (v === undefined || v === null || v === '' || (v === 0 && k !== 'step_id')) return;
            v = String(v);
            if (full) {{ h += '<div class="dp-label">' + lbl + '</div><div class="dp-val' + (vlm ? ' vlm' : '') + '">' + esc(v) + '</div>'; }}
            else {{ h += '<div class="tip-row"><span class="tip-label">' + lbl + '</span><span class="tip-val' + (vlm ? ' vlm' : '') + '">' + esc(truncate(v, 120)) + '</span></div>'; }}
        }});
        return h;
    }}
    document.querySelectorAll('.tip').forEach(function(el) {{
        el.addEventListener('mouseenter', function(e) {{
            var d = JSON.parse(el.getAttribute('data-tip'));
            pop.innerHTML = '<div class="tip-header">' + badgeFor(d) + '<span class="tip-type">' + esc(typeLabel(d)) + '</span><span class="tip-time">' + esc(timeLabel(d)) + '</span></div>' + buildRows(d, false);
            pop.classList.add('visible');
        }});
        el.addEventListener('mousemove', function(e) {{
            var pw = pop.offsetWidth, ph = pop.offsetHeight, x = e.clientX + 14, y = e.clientY + 14;
            if (x + pw > window.innerWidth - 8) x = e.clientX - pw - 10;
            if (y + ph > window.innerHeight - 8) y = e.clientY - ph - 10;
            pop.style.left = x + 'px'; pop.style.top = y + 'px';
        }});
        el.addEventListener('mouseleave', function() {{ pop.classList.remove('visible'); }});
        el.addEventListener('click', function() {{
            var d = JSON.parse(el.getAttribute('data-tip'));
            var panel = el.closest('.tab-panel').querySelector('#detail-panel') || document.getElementById('detail-panel');
            if (panel) {{
                panel.querySelector('#dp-badge').innerHTML = badgeFor(d);
                panel.querySelector('#dp-type').textContent = typeLabel(d);
                panel.querySelector('#dp-time').textContent = timeLabel(d);
                panel.querySelector('#dp-grid').innerHTML = buildRows(d, true);
                panel.classList.add('active');
                panel.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
            }}
        }});
    }});
}})();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation dashboard")
    parser.add_argument("--predicted", help="Candidate output JSON (single clip)")
    parser.add_argument("--ground-truth", help="Ground truth JSON (single clip)")
    parser.add_argument("--multi", nargs="+", help="Multiple clip pairs as pred:gt (e.g., out1.json:gt1.json out2.json:gt2.json)")
    parser.add_argument("--output", default="output/dashboard.html", help="Output HTML path")
    parser.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.multi:
        clips = []
        for pair in args.multi:
            pred, gt = pair.split(":")
            name = Path(pred).stem
            clips.append({"name": name, "predicted": pred, "ground_truth": gt})
        html = generate_multi_html(clips, args.tolerance)
    elif args.predicted and args.ground_truth:
        html = generate_html(args.predicted, args.ground_truth, args.tolerance)
    else:
        parser.error("Provide --predicted and --ground-truth, or --multi")

    with open(args.output, "w") as f:
        f.write(html)
    print(f"Dashboard saved to: {args.output}")


if __name__ == "__main__":
    main()
