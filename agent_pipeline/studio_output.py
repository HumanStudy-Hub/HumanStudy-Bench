"""Validate Studio outputs before publishing a completed job."""

from __future__ import annotations

import json
from pathlib import Path


try:
    from agent_pipeline.human_program import validate_human_program, canonical_model
except ModuleNotFoundError:
    from human_program import validate_human_program, canonical_model

KINDS = {"background", "hypothesis", "design", "participants", "material", "procedure", "record", "variable", "analysis", "result"}
STATUSES = {"reported", "implementation", "unresolved"}
SEVERITIES = {"blocking", "decision", "check"}


def studio_mode(request: dict) -> str:
    mode = request.get("mode", "build")
    if mode not in ("discuss", "build"):
        raise ValueError("Studio mode must be 'discuss' or 'build'")
    return mode


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{label} must be a{' nonempty' if nonempty else ''} string")
    return value


def _array(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _choice(value: object, choices: set[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} is invalid")
    return value


def _evidence(value: object, label: str) -> None:
    item = _object(value, label)
    page = item.get("page")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError(f"{label}.page must be a positive PDF page number")
    _string(item.get("quote"), f"{label}.quote")
    if "sourceId" in item:
        _string(item["sourceId"], f"{label}.sourceId")
    for index, rect in enumerate(_array(item.get("rects"), f"{label}.rects")):
        rect = _object(rect, f"{label}.rects[{index}]")
        for key in ("x", "y", "w", "h"):
            number = rect.get(key)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 100:
                raise ValueError(f"{label}.rects[{index}].{key} must be a page percentage")


def validate_study_model(value: object) -> dict:
    """Check the portable StudySchema shape used by the Studio frontend."""
    model = _object(value, "model")
    if "schemaVersion" in model or "program" in model:
        validate_human_program(canonical_model(model))
        return model
    for key in ("id", "title"):
        _string(model.get(key), f"model.{key}", nonempty=True)
    source = _object(model.get("source"), "model.source")
    for key in ("title", "authors", "filename"):
        _string(source.get(key), f"model.source.{key}")
    entity_ids: set[str] = set()
    for index, raw in enumerate(_array(model.get("entities"), "model.entities")):
        label = f"model.entities[{index}]"
        entity = _object(raw, label)
        entity_id = _string(entity.get("id"), f"{label}.id", nonempty=True)
        if entity_id in entity_ids:
            raise ValueError(f"duplicate entity id: {entity_id}")
        entity_ids.add(entity_id)
        _choice(entity.get("kind"), KINDS, f"{label}.kind")
        for key in ("title", "subtitle", "description"):
            _string(entity.get(key), f"{label}.{key}")
        _evidence(entity.get("evidence"), f"{label}.evidence")
        for key in ("x", "y", "w", "h"):
            number = entity.get(key)
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"{label}.{key} must be a number")
        for field_index, raw_field in enumerate(_array(entity.get("fields"), f"{label}.fields")):
            field = _object(raw_field, f"{label}.fields[{field_index}]")
            _string(field.get("name"), f"{label}.fields[{field_index}].name")
            _string(field.get("value"), f"{label}.fields[{field_index}].value")
            if "status" in field:
                _choice(field["status"], STATUSES, f"{label}.fields[{field_index}].status")
    for index, raw in enumerate(_array(model.get("relations"), "model.relations")):
        label = f"model.relations[{index}]"
        relation = _object(raw, label)
        for key in ("from", "to"):
            endpoint = _string(relation.get(key), f"{label}.{key}", nonempty=True)
            if endpoint not in entity_ids:
                raise ValueError(f"{label}.{key} references an unknown entity")
        _string(relation.get("label"), f"{label}.label")
    for index, raw in enumerate(_array(model.get("procedure"), "model.procedure")):
        label = f"model.procedure[{index}]"
        step = _object(raw, label)
        for key in ("id", "name", "input", "actor", "output"):
            _string(step.get(key), f"{label}.{key}")
        _evidence(step.get("evidence"), f"{label}.evidence")
    for index, raw in enumerate(_array(model.get("variables"), "model.variables")):
        label = f"model.variables[{index}]"
        variable = _object(raw, label)
        for key in ("id", "name", "role", "type", "unit", "producedBy", "usedBy", "definition"):
            _string(variable.get(key), f"{label}.{key}")
        _choice(variable.get("status"), STATUSES, f"{label}.status")
        if _string(variable.get("entity"), f"{label}.entity") not in entity_ids:
            raise ValueError(f"{label}.entity references an unknown entity")
    if "reviewIssues" in model:
        for index, raw in enumerate(_array(model["reviewIssues"], "model.reviewIssues")):
            label = f"model.reviewIssues[{index}]"
            issue = _object(raw, label)
            for key in ("id", "title", "reason", "impact", "suggestedAction"):
                _string(issue.get(key), f"{label}.{key}")
            _choice(issue.get("severity"), SEVERITIES, f"{label}.severity")
            if "entity" in issue and _string(issue["entity"], f"{label}.entity") not in entity_ids:
                raise ValueError(f"{label}.entity references an unknown entity")
            for key in ("study", "field", "sourcePointer"):
                if key in issue:
                    _string(issue[key], f"{label}.{key}")
            if "evidence" in issue:
                _evidence(issue["evidence"], f"{label}.evidence")
    return model


def validate_discussion(job: Path, request_id: str) -> tuple[bool, str]:
    try:
        output_path = job / "studio-turn.json"
        if output_path.stat().st_size > 5_000_000:
            raise ValueError("studio-turn.json exceeds the 5 MB output limit")
        output = _object(json.loads(output_path.read_text()), "studio-turn.json")
        if output.get("requestId") != request_id:
            raise ValueError("studio-turn.json requestId does not match this request")
        _string(output.get("reply"), "studio-turn.json.reply", nonempty=True)
        if "summary" in output:
            _string(output["summary"], "studio-turn.json.summary")
        if "model" in output:
            validate_study_model(output["model"])
        return True, "Valid Studio discussion turn"
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return False, str(exc)


def validate_build_sidecars(package: Path, accepted_model: object | None = None) -> tuple[bool, str]:
    roots = [path for path in package.iterdir() if path.is_dir()] if package.exists() else []
    if len(roots) != 1:
        return False, "package must contain exactly one paper folder"
    root = roots[0]
    try:
        model = validate_study_model(json.loads((root / "studio-model.json").read_text()))
        if accepted_model is not None and canonical_model(model) != canonical_model(validate_study_model(accepted_model)):
            raise ValueError("studio-model.json must exactly match the accepted document.model")
        overview_path = root / "study.json"
        if canonical_model(model).get("schemaVersion") == 2 and not overview_path.is_file():
            raise ValueError("v2 requires canonical study.json.program")
        if overview_path.is_file():
            overview = json.loads(overview_path.read_text())
            if canonical_model(model).get("schemaVersion") == 2 and "program" not in overview:
                raise ValueError("v2 requires canonical study.json.program")
            if "program" in overview and validate_human_program(overview["program"]) != canonical_model(model):
                raise ValueError("study.json.program and studio-model.json disagree")
        _string((root / "studio-reply.md").read_text(), "studio-reply.md", nonempty=True)
        return True, "Valid Studio build sidecars"
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return False, str(exc)
