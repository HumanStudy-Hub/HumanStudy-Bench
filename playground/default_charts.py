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


def build_charts(analysis: Dict[str, Any]) -> Dict[str, Any]:
    charts = [chart for chart in (effect_scatter(analysis), effect_bars(analysis), replication_breakdown(analysis)) if chart]
    return {"charts": charts, "source": "default"}


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
