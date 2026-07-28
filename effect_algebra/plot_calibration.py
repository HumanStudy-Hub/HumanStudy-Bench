"""Human proportion against model proportion, as a self-contained SVG.

This is the figure the calibration framing rests on: one point per scenario,
the diagonal is perfect calibration, and the distance from the diagonal is the
MAE being reported. Written as raw SVG rather than through a plotting library so
it renders identically on Colab, in CI, and in a paper build with no extra
dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


WIDTH = 460
HEIGHT = 460
MARGIN = 58
SERIES_COLOURS = (
    "#2f6fdb",
    "#d1495b",
    "#2a9d8f",
    "#e09f3e",
    "#7c4dff",
)


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _points(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for row in rows:
        human = row.get("human_probability_by_code")
        if not isinstance(human, dict):
            continue
        points.append(
            {
                "human": float(human["X"]),
                "model": float(row["probability_by_code"]["X"]),
                "label": str(
                    row.get("scenario_id") or row.get("id") or ""
                ),
                "condition": str(row.get("authority_condition") or row.get("bucket") or ""),
            }
        )
    return points


def _to_canvas(value: float, *, axis: str) -> float:
    span = (WIDTH if axis == "x" else HEIGHT) - 2 * MARGIN
    if axis == "x":
        return MARGIN + value * span
    return HEIGHT - MARGIN - value * span


def render_svg(
    series: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    title: str,
    noise_floor: Optional[float] = None,
) -> str:
    parts: List[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
        'viewBox="0 0 {} {}" font-family="Helvetica,Arial,sans-serif">'.format(
            WIDTH, HEIGHT, WIDTH, HEIGHT
        ),
        '<rect width="{}" height="{}" fill="white"/>'.format(WIDTH, HEIGHT),
    ]

    # Axes and gridlines at every 0.25.
    for step in range(5):
        value = step / 4.0
        x = _to_canvas(value, axis="x")
        y = _to_canvas(value, axis="y")
        parts.append(
            '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="#e6e6e6" stroke-width="1"/>'.format(
                x, MARGIN, x, HEIGHT - MARGIN
            )
        )
        parts.append(
            '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="#e6e6e6" stroke-width="1"/>'.format(
                MARGIN, y, WIDTH - MARGIN, y
            )
        )
        parts.append(
            '<text x="{:.1f}" y="{:.1f}" font-size="10" fill="#555" '
            'text-anchor="middle">{:.2f}</text>'.format(
                x, HEIGHT - MARGIN + 16, value
            )
        )
        parts.append(
            '<text x="{:.1f}" y="{:.1f}" font-size="10" fill="#555" '
            'text-anchor="end">{:.2f}</text>'.format(MARGIN - 8, y + 3, value)
        )

    if noise_floor:
        # Band the width of the sampling noise floor: a perfect model still
        # lands inside it, so points there are not evidence of miscalibration.
        span = (WIDTH - 2 * MARGIN) * noise_floor
        parts.append(
            '<polygon points="{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}" '
            'fill="#2f6fdb" opacity="0.08"/>'.format(
                MARGIN,
                HEIGHT - MARGIN - span,
                WIDTH - MARGIN,
                MARGIN - span,
                WIDTH - MARGIN,
                MARGIN + span,
                MARGIN,
                HEIGHT - MARGIN + span,
            )
        )

    parts.append(
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333" '
        'stroke-width="1.2" stroke-dasharray="5 4"/>'.format(
            MARGIN, HEIGHT - MARGIN, WIDTH - MARGIN, MARGIN
        )
    )
    parts.append(
        '<rect x="{}" y="{}" width="{}" height="{}" fill="none" '
        'stroke="#333" stroke-width="1"/>'.format(
            MARGIN, MARGIN, WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN
        )
    )

    for index, (name, rows) in enumerate(sorted(series.items())):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        for point in _points(rows):
            parts.append(
                '<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="{}" '
                'opacity="0.75"><title>{}</title></circle>'.format(
                    _to_canvas(point["human"], axis="x"),
                    _to_canvas(point["model"], axis="y"),
                    colour,
                    _escape(
                        "{} {} human={:.3f} model={:.3f}".format(
                            name,
                            point["label"],
                            point["human"],
                            point["model"],
                        )
                    ),
                )
            )
        legend_y = MARGIN + 14 + index * 16
        parts.append(
            '<circle cx="{}" cy="{}" r="4" fill="{}"/>'.format(
                WIDTH - MARGIN - 110, legend_y - 4, colour
            )
        )
        parts.append(
            '<text x="{}" y="{}" font-size="11" fill="#333">{}</text>'.format(
                WIDTH - MARGIN - 100, legend_y, _escape(name)
            )
        )

    parts.append(
        '<text x="{}" y="26" font-size="14" fill="#111" '
        'text-anchor="middle">{}</text>'.format(WIDTH / 2, _escape(title))
    )
    parts.append(
        '<text x="{}" y="{}" font-size="12" fill="#333" '
        'text-anchor="middle">human P(X)</text>'.format(
            WIDTH / 2, HEIGHT - 12
        )
    )
    parts.append(
        '<text x="16" y="{}" font-size="12" fill="#333" text-anchor="middle" '
        'transform="rotate(-90 16 {})">model P(X)</text>'.format(
            HEIGHT / 2, HEIGHT / 2
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _series_argument(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result must be LABEL=/path/to/result.json")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        type=_series_argument,
        required=True,
        help="Repeatable LABEL=/path/to/evaluate_choices_output.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Calibration against human choice proportions")
    parser.add_argument(
        "--noise-floor",
        type=float,
        help="Half-width of the sampling-noise band to shade, e.g. 0.035 for C.",
    )
    parser.add_argument("--csv", type=Path, help="Also write the raw point data.")
    args = parser.parse_args()

    series: Dict[str, Sequence[Mapping[str, Any]]] = {}
    for label, path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        series[label] = payload["rows"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_svg(series, title=args.title, noise_floor=args.noise_floor),
        encoding="utf-8",
    )
    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["series", "label", "condition", "human_p_x", "model_p_x"])
            for label, rows in sorted(series.items()):
                for point in _points(rows):
                    writer.writerow(
                        [
                            label,
                            point["label"],
                            point["condition"],
                            "{:.6f}".format(point["human"]),
                            "{:.6f}".format(point["model"]),
                        ]
                    )
    print(str(args.output))


if __name__ == "__main__":
    main()
