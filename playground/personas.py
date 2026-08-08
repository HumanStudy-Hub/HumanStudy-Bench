"""Persona groups: who the agents in a playground run are.

A group describes a *population*, not a fixed cast. Each segment carries a share
of the participants and the ranges its members are drawn from, so one saved
group fits a 20-session run and a 600-session run alike, and works across
studies whose designs need different numbers of participants.

Sampling is deterministic for a given seed, so a saved group plus a run seed
reproduces exactly the same participants.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

SCHEMA_VERSION = 1
MAX_SEGMENTS = 12
MAX_TEXT = 600
MIN_AGE = 10
MAX_AGE = 110


class PersonaError(ValueError):
    """A persona group cannot be used as written."""


def _text(value: Any, field: str, limit: int = MAX_TEXT) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PersonaError(f"{field} must be text")
    cleaned = value.strip()
    if len(cleaned) > limit:
        raise PersonaError(f"{field} is longer than {limit} characters")
    return cleaned or None


def _age(value: Any, field: str) -> Optional[Dict[str, int]]:
    """Accept a fixed age or a range, and normalise both to {min, max}."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise PersonaError(f"{field} must be a number or a range")
    if isinstance(value, (int, float)):
        low = high = int(value)
    elif isinstance(value, Mapping):
        if "value" in value:
            low = high = int(value["value"])
        else:
            try:
                low, high = int(value["min"]), int(value["max"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PersonaError(f"{field} needs either a value or a min and max") from exc
    else:
        raise PersonaError(f"{field} must be a number or a range")
    if low > high:
        low, high = high, low
    if low < MIN_AGE or high > MAX_AGE:
        raise PersonaError(f"{field} must fall between {MIN_AGE} and {MAX_AGE}")
    return {"min": low, "max": high}


def _weights(value: Any, field: str) -> Optional[Dict[str, float]]:
    """Accept a single value or a weighted mix, and normalise both to weights."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return {cleaned: 1.0} if cleaned else None
    if not isinstance(value, Mapping) or not value:
        raise PersonaError(f"{field} must be a value or a set of weights")
    weights: Dict[str, float] = {}
    for key, weight in value.items():
        if not isinstance(key, str) or not key.strip():
            raise PersonaError(f"{field} has an unnamed option")
        try:
            number = float(weight)
        except (TypeError, ValueError) as exc:
            raise PersonaError(f"{field}.{key} must be a number") from exc
        if number < 0:
            raise PersonaError(f"{field}.{key} cannot be negative")
        if number > 0:
            weights[key.strip()] = number
    if not weights:
        raise PersonaError(f"{field} has no option with a share above zero")
    total = sum(weights.values())
    return {key: weight / total for key, weight in weights.items()}


def normalise_group(raw: Any) -> Dict[str, Any]:
    """Validate a persona group and return it in canonical form."""
    if not isinstance(raw, Mapping):
        raise PersonaError("A persona group must be an object")
    segments_raw = raw.get("segments")
    if not isinstance(segments_raw, list) or not segments_raw:
        raise PersonaError("A persona group needs at least one segment")
    if len(segments_raw) > MAX_SEGMENTS:
        raise PersonaError(f"A persona group cannot have more than {MAX_SEGMENTS} segments")

    segments: List[Dict[str, Any]] = []
    used_ids = set()
    for index, entry in enumerate(segments_raw):
        if not isinstance(entry, Mapping):
            raise PersonaError(f"Segment {index + 1} must be an object")
        where = f"segments[{index}]"
        label = _text(entry.get("label"), f"{where}.label", 120) or f"Group {index + 1}"
        identifier = _text(entry.get("id"), f"{where}.id", 60) or label.lower().replace(" ", "_")
        if identifier in used_ids:
            raise PersonaError(f"Two segments share the id {identifier}")
        used_ids.add(identifier)
        try:
            share = float(entry.get("share", 1))
        except (TypeError, ValueError) as exc:
            raise PersonaError(f"{where}.share must be a number") from exc
        if share <= 0:
            raise PersonaError(f"{where}.share must be greater than zero")
        segments.append({
            "id": identifier,
            "label": label,
            "share": share,
            "age": _age(entry.get("age"), f"{where}.age"),
            "gender": _weights(entry.get("gender"), f"{where}.gender"),
            "education": _text(entry.get("education"), f"{where}.education", 160),
            "background": _text(entry.get("background"), f"{where}.background"),
            "persona": _text(entry.get("persona"), f"{where}.persona"),
        })

    total_share = sum(segment["share"] for segment in segments)
    for segment in segments:
        segment["share"] = segment["share"] / total_share

    return {
        "schemaVersion": SCHEMA_VERSION,
        "name": _text(raw.get("name"), "name", 120) or "Untitled persona group",
        "description": _text(raw.get("description"), "description"),
        "studyId": _text(raw.get("studyId"), "studyId", 60),
        "contributor": _text(raw.get("contributor"), "contributor", 80),
        "segments": segments,
    }


def _counts(segments: List[Dict[str, Any]], total: int) -> List[int]:
    """Split `total` participants across segments by share.

    Uses largest remainder, so a 30/70 split of 10 participants is 3 and 7 rather
    than whatever repeated rounding happens to produce. Every segment keeps at
    least one participant when there are enough to go around, because a segment
    that never appears is a silently ignored part of the researcher's design.
    """
    if total <= 0:
        return [0] * len(segments)
    exact = [segment["share"] * total for segment in segments]
    counts = [int(value) for value in exact]
    if len(segments) <= total:
        counts = [max(1, count) for count in counts]
    while sum(counts) > total:
        # Take back from the largest segment first.
        counts[counts.index(max(counts))] -= 1
    remainders = sorted(range(len(segments)), key=lambda index: exact[index] - int(exact[index]), reverse=True)
    position = 0
    while sum(counts) < total:
        counts[remainders[position % len(remainders)]] += 1
        position += 1
    return counts


def sample_profiles(group: Mapping[str, Any], total: int, seed: int = 42, defaults: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Draw `total` participant profiles from a persona group."""
    normalised = normalise_group(group)
    rng = random.Random(seed)
    counts = _counts(normalised["segments"], total)
    profiles: List[Dict[str, Any]] = []

    for segment, count in zip(normalised["segments"], counts):
        for _ in range(count):
            profile: Dict[str, Any] = dict(defaults or {})
            profile["persona_segment"] = segment["id"]
            profile["persona_label"] = segment["label"]
            if segment["age"]:
                profile["age"] = rng.randint(segment["age"]["min"], segment["age"]["max"])
            if segment["gender"]:
                options = list(segment["gender"].items())
                profile["gender"] = rng.choices([name for name, _ in options], weights=[weight for _, weight in options])[0]
            for field in ("education", "background", "persona"):
                if segment[field]:
                    profile[field] = segment[field]
            profiles.append(profile)

    # Interleave the segments so a truncated or partially failed run still covers
    # the whole population rather than only its first group.
    rng.shuffle(profiles)
    for index, profile in enumerate(profiles):
        profile["participant_id"] = index
    return profiles


def describe_mix(group: Mapping[str, Any], total: int) -> List[Dict[str, Any]]:
    """How many participants each segment gets, for a preview before running."""
    normalised = normalise_group(group)
    counts = _counts(normalised["segments"], total)
    return [
        {"id": segment["id"], "label": segment["label"], "count": count, "share": count / total if total else 0}
        for segment, count in zip(normalised["segments"], counts)
    ]


def load_group(path: Path) -> Dict[str, Any]:
    try:
        return normalise_group(json.loads(Path(path).read_text()))
    except json.JSONDecodeError as exc:
        raise PersonaError(f"{path} is not valid JSON: {exc}") from exc
