#!/usr/bin/env python3
"""Generate deterministic visual numerosity stimuli for study_018.

The public LIONESS archive references external animal sprite URLs that are no
longer available. These drawings preserve the published species, exact counts,
clutter, overlap, and deterministic round-level placement without claiming to
be the original artwork.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw


REFERENCE_SCREEN = (1920, 1080)
INTERIOR = (REFERENCE_SCREEN[0] // 2, REFERENCE_SCREEN[1] // 2)
SOURCE_PERIOD = 1
ICON_SIZE = 50
OVERLAP_RATIO = 0.6
BORDER = 50
CANVAS = (INTERIOR[0] + BORDER, INTERIOR[1] + BORDER)
BASE_ICON_SIZE = 38
BACKGROUND = (204, 204, 255)
INK = (28, 31, 34)
ACCENT = (201, 70, 57)


def _ant(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    cy = y + ICON_SIZE // 2
    for offset, radius in ((9, 4), (18, 5), (28, 6)):
        draw.ellipse(
            (x + offset - radius, cy - radius, x + offset + radius, cy + radius),
            fill=INK,
        )
    for offset in (14, 21, 27):
        draw.line((x + offset, cy - 2, x + offset - 7, cy - 10), fill=INK, width=2)
        draw.line((x + offset, cy + 2, x + offset - 7, cy + 10), fill=INK, width=2)
    draw.line((x + 5, cy - 2, x, cy - 9), fill=INK, width=1)
    draw.line((x + 5, cy + 2, x, cy + 9), fill=INK, width=1)


def _bee(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.ellipse((x + 5, y + 11, x + 32, y + 28), fill=(238, 183, 43), outline=INK)
    for offset in (12, 19, 26):
        draw.line((x + offset, y + 12, x + offset, y + 27), fill=INK, width=2)
    draw.ellipse((x + 9, y + 3, x + 19, y + 15), fill=(222, 237, 244), outline=INK)
    draw.ellipse((x + 19, y + 3, x + 29, y + 15), fill=(222, 237, 244), outline=INK)


def _flamingo(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    pink = (222, 98, 135)
    draw.ellipse((x + 8, y + 9, x + 29, y + 27), fill=pink, outline=INK)
    draw.arc((x + 19, y + 1, x + 35, y + 18), 110, 280, fill=pink, width=3)
    draw.ellipse((x + 29, y + 1, x + 35, y + 7), fill=pink, outline=INK)
    draw.line((x + 15, y + 25, x + 13, y + 37), fill=INK, width=2)
    draw.line((x + 23, y + 25, x + 26, y + 37), fill=INK, width=2)


def _crane(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.ellipse((x + 5, y + 11, x + 28, y + 28), fill=(220, 223, 218), outline=INK)
    draw.line((x + 24, y + 13, x + 31, y + 3), fill=INK, width=3)
    draw.ellipse((x + 28, y, x + 35, y + 7), fill=(220, 223, 218), outline=INK)
    draw.polygon([(x + 35, y + 3), (x + 38, y + 5), (x + 35, y + 6)], fill=ACCENT)
    draw.line((x + 14, y + 26, x + 12, y + 37), fill=INK, width=2)
    draw.line((x + 22, y + 26, x + 25, y + 37), fill=INK, width=2)


def _cricket(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    green = (75, 126, 69)
    draw.ellipse((x + 8, y + 11, x + 29, y + 25), fill=green, outline=INK)
    draw.ellipse((x + 25, y + 9, x + 34, y + 18), fill=green, outline=INK)
    draw.line((x + 12, y + 22, x + 2, y + 34), fill=INK, width=2)
    draw.line((x + 20, y + 23, x + 29, y + 36), fill=INK, width=2)
    draw.line((x + 30, y + 11, x + 37, y + 2), fill=INK, width=1)
    draw.line((x + 32, y + 12, x + 38, y + 7), fill=INK, width=1)


DRAWERS: dict[str, Callable[[ImageDraw.ImageDraw, int, int], None]] = {
    "ant": _ant,
    "bee": _bee,
    "flamingo": _flamingo,
    "crane": _crane,
    "cricket": _cricket,
}


def _icon(species: str) -> Image.Image:
    tile = Image.new("RGBA", (BASE_ICON_SIZE, BASE_ICON_SIZE), (0, 0, 0, 0))
    DRAWERS[species](ImageDraw.Draw(tile), 0, 0)
    return tile.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)


def _original_positions(*, count: int) -> list[tuple[int, int]]:
    """Reproduce the seeded placement formula in stage32741.php."""

    seed = SOURCE_PERIOD
    random_values: list[int] = []
    for _ in range(200):
        value = math.sin(seed) * INTERIOR[0]
        seed += 1
        fraction = value - math.floor(value)
        random_values.append(math.floor(fraction * INTERIOR[0] + 0.5))

    unique_values = list(dict.fromkeys(random_values))
    if len(unique_values) < count:
        raise ValueError("published placement sequence has too few unique positions")
    # JavaScript Array.sort() without a comparator sorts values as strings.
    selected = sorted(unique_values[:count], key=str)

    horizontal_positions = INTERIOR[0] / (ICON_SIZE * OVERLAP_RATIO)
    vertical_positions = INTERIOR[1] / (ICON_SIZE * OVERLAP_RATIO)
    horizontal_step = INTERIOR[0] / horizontal_positions
    return [
        (
            round((value % horizontal_positions) * horizontal_step),
            round(math.floor(value / horizontal_positions) * vertical_positions),
        )
        for value in selected
    ]


def generate_image(*, species: str, count: int, output: Path) -> None:
    if species not in DRAWERS:
        raise ValueError(f"unsupported species: {species}")
    image = Image.new("RGB", CANVAS, BACKGROUND)
    icon = _icon(species)
    for x, y in _original_positions(count=count):
        image.paste(icon, (x, y), icon)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lookup",
        type=Path,
        default=Path(__file__).parents[1] / "materials" / "peer_lookup.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    lookup = json.loads(args.lookup.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "fidelity": (
            "Deterministic regenerated artwork preserving published species, exact "
            "animal counts, display parameters, and source-period placement formula. "
            "The original sprite URLs referenced by the OSF experiment software are "
            "no longer available."
        ),
        "canvas": list(CANVAS),
        "reference_screen": list(REFERENCE_SCREEN),
        "source_period_seed": SOURCE_PERIOD,
        "background_rgb": list(BACKGROUND),
        "icon_size": ICON_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "border": BORDER,
        "items": [],
    }

    for round_data in lookup["main_task"]["rounds"]:
        round_number = int(round_data["round"])
        species = str(round_data["species"])
        filename = f"round_{round_number:02d}_{species}.png"
        generate_image(
            species=species,
            count=int(round_data["true_count"]),
            output=args.output_dir / filename,
        )
        manifest["items"].append(
            {
                "round": round_number,
                "species": species,
                "file": filename,
                "true_count_source": "peer_lookup.json main_task.rounds",
            }
        )

    manifest_path = args.output_dir.parent / "materials" / "stimulus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest['items'])} stimuli and {manifest_path}")


if __name__ == "__main__":
    main()
