#!/usr/bin/env python3
"""Check that agent-written charts are safe, plottable Plotly specifications.

The charting agent writes `output/charts.json`. That file is rendered in the
browser, so it is validated here before it ever leaves the runner: only plain
JSON data is allowed, trace types are restricted to an allowlist, and every
numeric series must actually contain numbers. A file that fails validation is
replaced by the deterministic charts.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, List

ALLOWED_TYPES = {"scatter", "scattergl", "bar", "box", "violin", "histogram", "heatmap", "line"}
MAX_CHARTS = 6
MAX_POINTS = 5000
MAX_DEPTH = 12
# Plotly renders these as raw markup, which would let generated text inject HTML.
FORBIDDEN_KEYS = {"meta", "customdatasrc", "xsrc", "ysrc", "zsrc", "textsrc", "idssrc"}


class ChartError(Exception):
    pass


def _check_value(value: Any, path: str, depth: int = 0) -> int:
    """Walk a JSON value, rejecting anything that is not plain data."""
    if depth > MAX_DEPTH:
        raise ChartError(f"{path}: nested too deeply")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > 4000:
            raise ChartError(f"{path}: string is too long")
        return 1
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ChartError(f"{path}: not a finite number")
        return 1
    if isinstance(value, list):
        if len(value) > MAX_POINTS:
            raise ChartError(f"{path}: more than {MAX_POINTS} values")
        return sum(_check_value(item, f"{path}[{index}]", depth + 1) for index, item in enumerate(value))
    if isinstance(value, dict):
        count = 0
        for key, item in value.items():
            if not isinstance(key, str):
                raise ChartError(f"{path}: non-string key")
            if key.lower() in FORBIDDEN_KEYS:
                raise ChartError(f"{path}.{key}: field is not allowed")
            count += _check_value(item, f"{path}.{key}", depth + 1)
        return count
    raise ChartError(f"{path}: unsupported value of type {type(value).__name__}")


def validate(document: Any) -> List[str]:
    if not isinstance(document, dict):
        raise ChartError("charts.json must contain an object")
    charts = document.get("charts")
    if not isinstance(charts, list) or not charts:
        raise ChartError("charts.json must contain a non-empty 'charts' list")
    if len(charts) > MAX_CHARTS:
        raise ChartError(f"charts.json has more than {MAX_CHARTS} charts")

    ids: List[str] = []
    for index, chart in enumerate(charts):
        where = f"charts[{index}]"
        if not isinstance(chart, dict):
            raise ChartError(f"{where}: each chart must be an object")
        for field in ("id", "title", "description"):
            if not isinstance(chart.get(field), str) or not chart[field].strip():
                raise ChartError(f"{where}.{field}: required non-empty string")
        if chart["id"] in ids:
            raise ChartError(f"{where}.id: duplicate chart id {chart['id']}")
        ids.append(chart["id"])

        plotly = chart.get("plotly")
        if not isinstance(plotly, dict):
            raise ChartError(f"{where}.plotly: required object")
        traces = plotly.get("data")
        if not isinstance(traces, list) or not traces:
            raise ChartError(f"{where}.plotly.data: required non-empty list")
        for trace_index, trace in enumerate(traces):
            if not isinstance(trace, dict):
                raise ChartError(f"{where}.plotly.data[{trace_index}]: must be an object")
            trace_type = trace.get("type")
            if trace_type is not None and trace_type not in ALLOWED_TYPES:
                raise ChartError(f"{where}.plotly.data[{trace_index}].type: '{trace_type}' is not an allowed chart type")
            if not any(key in trace for key in ("x", "y", "z", "values", "labels")):
                raise ChartError(f"{where}.plotly.data[{trace_index}]: has no data series")
        if "layout" in plotly and not isinstance(plotly["layout"], dict):
            raise ChartError(f"{where}.plotly.layout: must be an object")
        _check_value(plotly, f"{where}.plotly")

    interpretation = document.get("interpretation")
    if interpretation is not None and (not isinstance(interpretation, str) or len(interpretation) > 8000):
        raise ChartError("interpretation must be a string of at most 8000 characters")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("charts", type=Path)
    args = parser.parse_args()
    try:
        document = json.loads(args.charts.read_text())
        ids = validate(document)
    except (OSError, json.JSONDecodeError) as error:
        print(f"charts.json could not be read: {error}")
        raise SystemExit(1)
    except ChartError as error:
        print(f"charts.json is not valid: {error}")
        raise SystemExit(1)
    print(f"charts.json is valid: {', '.join(ids)}")


if __name__ == "__main__":
    main()
