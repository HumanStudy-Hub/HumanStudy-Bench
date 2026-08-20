"""Deterministic comparison charts for a playground run.

These are the charts the playground falls back to when the charting agent does
not produce a valid set, and they are also the reference the agent is asked to
improve on. Everything here is derived from `output/analysis.json` alone.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

SIGNIFICANT = "#0e7490"
NOT_SIGNIFICANT = "#a5e5ea"
HUMAN = "#334155"
AGENT = "#0e7490"


def _stars(p_value: Any, significant: Any) -> str:
    if isinstance(p_value, (int, float)):
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return ""
    return "*" if significant is True else ""


def _axis_bounds(values: List[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    low, high = min(values), max(values)
    padding = max(0.15, (high - low) * 0.1)
    return low - padding, high + padding


def effect_scatter(analysis: Dict[str, Any]) -> Dict[str, Any] | None:
    """Agent effect size against the published human effect size."""
    points = [test for test in analysis["tests"] if test["human_effect"] is not None and test["agent_effect"] is not None]
    if not points:
        return None
    significant = [test for test in points if test["agent_significant"]]
    other = [test for test in points if not test["agent_significant"]]
    low, high = _axis_bounds([test["human_effect"] for test in points] + [test["agent_effect"] for test in points])

    def trace(rows: List[Dict[str, Any]], name: str, colour: str) -> Dict[str, Any]:
        return {
            "type": "scatter",
            "mode": "markers",
            "name": name,
            "x": [row["human_effect"] for row in rows],
            "y": [row["agent_effect"] for row in rows],
            "text": [row["label"] for row in rows],
            "customdata": [[_stars(row.get("human_p"), row.get("human_significant")), _stars(row.get("agent_p"), row.get("agent_significant"))] for row in rows],
            "hovertemplate": "%{text}<br>Human d=%{x:.2f}%{customdata[0]}<br>Agent d=%{y:.2f}%{customdata[1]}<extra></extra>",
            "marker": {"size": 11, "color": colour, "line": {"color": "white", "width": 1}},
        }

    data = [trace(rows, name, colour) for rows, name, colour in (
        (significant, "Agent effect significant", SIGNIFICANT),
        (other, "Not significant", NOT_SIGNIFICANT),
    ) if rows]
    data.append({
        "type": "scatter",
        "mode": "lines",
        "name": "Human = agent",
        "x": [low, high],
        "y": [low, high],
        "line": {"color": "#94a3b8", "width": 1.5, "dash": "dash"},
        "hoverinfo": "skip",
    })
    return {
        "id": "effect-scatter",
        "title": "Agent effect size vs. published human effect size",
        "description": "Each point is one statistical test. Hover for significance: * p<.05, ** p<.01, *** p<.001.",
        "plotly": {
            "data": data,
            "layout": {
                "xaxis": {"title": {"text": "Human effect size (d)"}, "range": [low, high], "zeroline": True},
                "yaxis": {"title": {"text": "Agent effect size (d)"}, "range": [low, high], "scaleanchor": "x", "zeroline": True},
                "legend": {"orientation": "h", "y": -0.2},
                "margin": {"t": 20, "r": 16, "b": 60, "l": 60},
            },
        },
    }


def effect_bars(analysis: Dict[str, Any]) -> Dict[str, Any] | None:
    """Human and agent effect sizes side by side, per test."""
    points = [test for test in analysis["tests"] if test["human_effect"] is not None or test["agent_effect"] is not None][:20]
    if not points:
        return None
    labels = [test["label"] for test in points]
    return {
        "id": "effect-bars",
        "title": "Effect size by test",
        "description": "Human and agent effects by test. * p<.05, ** p<.01, *** p<.001.",
        "plotly": {
            "data": [
                {"type": "bar", "name": "Human", "x": labels, "y": [test["human_effect"] for test in points], "text": [_stars(test.get("human_p"), test.get("human_significant")) for test in points], "textposition": "outside", "marker": {"color": HUMAN}},
                {"type": "bar", "name": "Agent", "x": labels, "y": [test["agent_effect"] for test in points], "text": [_stars(test.get("agent_p"), test.get("agent_significant")) for test in points], "textposition": "outside", "marker": {"color": AGENT}},
            ],
            "layout": {
                "barmode": "group",
                "xaxis": {"tickangle": -35, "automargin": True},
                "yaxis": {"title": {"text": "Effect size (d)"}, "zeroline": True},
                "legend": {"orientation": "h", "y": -0.25},
                "margin": {"t": 20, "r": 16, "b": 80, "l": 60},
            },
        },
    }


def replication_breakdown(analysis: Dict[str, Any]) -> Dict[str, Any] | None:
    """How many published findings the agent reproduced, missed, or reversed."""
    scored = [test for test in analysis["tests"] if test["replicated"] is not None or test["direction_match"] is not None]
    if not scored:
        return None
    reproduced = sum(1 for test in scored if test["replicated"])
    wrong_direction = sum(1 for test in scored if not test["replicated"] and test["direction_match"] is False)
    missed = len(scored) - reproduced - wrong_direction
    return {
        "id": "replication-breakdown",
        "title": "What the agent did with each published finding",
        "description": "Reproduced means the agent found the same effect, in the same direction, at the same significance level as the paper. Wrong direction means the agent showed the opposite effect.",
        "plotly": {
            "data": [{
                "type": "bar",
                "orientation": "h",
                "x": [reproduced, missed, wrong_direction],
                "y": ["Reproduced", "No significant effect", "Wrong direction"],
                "marker": {"color": [SIGNIFICANT, "#cbd5e1", "#f59e0b"]},
                "hovertemplate": "%{y}: %{x} tests<extra></extra>",
            }],
            "layout": {
                "xaxis": {"title": {"text": "Statistical tests"}, "dtick": 1},
                "yaxis": {"automargin": True},
                "showlegend": False,
                "margin": {"t": 20, "r": 16, "b": 45, "l": 20},
            },
        },
    }


def _macro_block(analysis: Dict[str, Any]) -> Dict[str, Any] | None:
    """The fixed headline numbers every report carries, prefering the buffer
    summary when present and falling back to the benchmark summary."""
    summary = analysis.get("summary") if isinstance(analysis.get("summary"), dict) else {}
    buffer = analysis.get("bufferSummary") if isinstance(analysis.get("bufferSummary"), dict) else {}
    if buffer:
        rows: List[Dict[str, str]] = []
        sessions = buffer.get("sessions")
        if isinstance(sessions, int):
            rows.append({"label": "Sessions", "value": str(sessions), "note": "completed sessions"})
        compliance = buffer.get("formatCompliance")
        if isinstance(compliance, (int, float)):
            rows.append({"label": "Format compliance", "value": f"{compliance:.0f}%", "note": "model replies parsed cleanly"})
        fallback = buffer.get("fallbackRate")
        if isinstance(fallback, (int, float)):
            rows.append({"label": "Fallback rate", "value": f"{fallback:.0f}%", "note": "replies that fell back to a default action"})
        coverage = buffer.get("coverage")
        if isinstance(coverage, dict) and coverage:
            rows.append({"label": "Coverage", "value": ", ".join(f"{k} × {v}" for k, v in sorted(coverage.items())), "note": "sessions per condition"})
        headline = buffer.get("headline")
        return {"headline": (headline if isinstance(headline, (int, float)) else None), "rows": rows}

    replication = summary.get("replicationRate")
    if isinstance(replication, (int, float)):
        scored = summary.get("scoredTests")
        replicated = summary.get("replicatedTests")
        return {
            "headline": round(replication * 100, 1),
            "rows": [
                {"label": "Strict replication", "value": f"{replicated}/{scored}" if isinstance(scored, int) else "-", "note": "same direction and significance"},
                {"label": "Direction matched", "value": _pct(summary.get("directionMatchRate")), "note": "effect pointed the same way"},
                {"label": "Mean effect gap", "value": _num(summary.get("meanAbsoluteEffectGap")), "note": "average distance from human effect"},
            ],
        }
    return None


def _pct(value: Any) -> str:
    return f"{value * 100:.0f}%" if isinstance(value, (int, float)) else "-"


def _num(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "-"


def _table_block(analysis: Dict[str, Any]) -> Dict[str, Any] | None:
    """A detailed results table: buffer metrics when present, else statistical tests."""
    metrics = analysis.get("metrics")
    if isinstance(metrics, list) and metrics:
        columns = ["Condition", "Metric", "Value"]
        rows = [[str(row.get("arm", "")), str(row.get("metric", "")), _num(row.get("value"))] for row in metrics if isinstance(row, dict)]
        return {"columns": columns, "rows": rows, "note": "One row per numeric evaluator metric."}
    tests = analysis.get("tests")
    if isinstance(tests, list) and tests:
        columns = ["Test", "Human d", "Agent d", "Result"]
        rows = []
        for row in tests:
            if not isinstance(row, dict):
                continue
            result = "replicated" if row.get("replicated") else ("wrong direction" if row.get("direction_match") is False else "not scored")
            rows.append([str(row.get("label", "")), _num(row.get("human_effect")), _num(row.get("agent_effect")), result])
        return {"columns": columns, "rows": rows, "note": "One row per statistical test."}
    return None


def build_charts(analysis: Dict[str, Any]) -> Dict[str, Any]:
    charts = [chart for chart in (effect_scatter(analysis), effect_bars(analysis), replication_breakdown(analysis)) if chart]
    document: Dict[str, Any] = {"charts": charts, "source": "default"}
    macro = _macro_block(analysis)
    if macro is not None:
        document["macro"] = macro
    table = _table_block(analysis)
    if table is not None:
        document["table"] = table
    # The deterministic fallback has no free-text agent reasoning; the reading
    # string (present for both study kinds) is the close equivalent.
    reading = analysis.get("reading")
    if isinstance(reading, str):
        document["reading"] = reading
    return document


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write the default playground charts for a run.")
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_charts(analysis), indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
