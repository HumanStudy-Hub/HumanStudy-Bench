#!/usr/bin/env python3
"""
Adapter for the Ross, Greene, & House (1977) "False Consensus Effect" package.

Runs fully offline. Presents one (study, condition) cell of materials to a
participant (human or agent), collects a structured response, validates it,
and emits a single JSON response record. Use --smoke-test to verify the whole
pipeline (materials loading, prompt construction, validation) with synthetic
responses and no external input.

Typical use by an agent-participant:
    python3 adapter.py --study study1 --condition supermarket --print-prompt
    # agent reads the printed prompt, forms its answer as JSON, then:
    python3 adapter.py --study study1 --condition supermarket \
        --response-file my_response.json --out record.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent

TASK_PATH = HERE / "task.json"
MATERIALS_PATH = PACKAGE_ROOT / "materials" / "materials.json"

STUDY2_EXCLUDED_CATEGORY_1 = "Expects removal of Nixon from office"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_trait_anchors(trait, template):
    return {
        "+50": template["anchor_plus50"].format(trait=trait),
        "0": template["anchor_zero"].format(trait=trait),
        "-50": template["anchor_minus50"].format(trait=trait),
    }


def get_condition(materials, study_id, condition_id):
    if study_id in ("study1", "study3"):
        conditions = materials[study_id]["conditions"]
        for c in conditions:
            if c["condition_id"] == condition_id:
                return c
        raise KeyError(f"unknown condition_id {condition_id!r} for {study_id}")
    if study_id == "study2":
        if condition_id not in ("self_first", "peer_first"):
            raise KeyError(f"unknown condition_id {condition_id!r} for study2")
        return {"condition_id": condition_id}
    if study_id == "study4":
        if condition_id != "generic":
            raise KeyError(f"unknown condition_id {condition_id!r} for study4")
        return {"condition_id": "generic"}
    raise KeyError(f"unknown study_id {study_id!r}")


def build_prompt(study_id, condition_id, materials):
    """Return a dict of participant-facing text/questions for this cell.
    No hidden instructions are injected; every field here traces to materials.json."""
    template = materials["trait_scale_template"]

    if study_id == "study1":
        cond = get_condition(materials, study_id, condition_id)
        traits = {t: render_trait_anchors(t, template) for t in cond["traits"]}
        return {
            "story_text": cond["story_text"],
            "consensus_question": cond["consensus_question"],
            "own_choice_options": cond["own_choice_options"],
            "trait_scales": traits,
            "instructions": (
                "First state which option you personally would choose. "
                "Then estimate the percentage of your peers who would choose each option "
                "(the two percentages must sum to 100). Then rate the 'typical person' who "
                "would choose each option on each listed trait, from -50 to +50, using the "
                "anchors given."
            ),
        }

    if study_id == "study2":
        items = [
            it for it in materials["study2"]["items"]
            if it["category_1"] != STUDY2_EXCLUDED_CATEGORY_1
        ]
        order = condition_id if condition_id in ("self_first", "peer_first") else "self_first"
        return {
            "items": [
                {"index": i, "category_1": it["category_1"], "category_2": it["category_2"], "domain": it["domain"]}
                for i, it in enumerate(items)
            ],
            "order": order,
            "instructions": (
                "For each item, first indicate which category applies to you personally, then "
                "estimate the percentage of 'college students in general' who fall into category_1 "
                "(order reversed if order == 'peer_first')."
            ),
            "faithfulness_caveat": (
                "The source paper does not reproduce full item wording, only these short category "
                "labels (evidence_label='reported'/'missing' — see materials.json study2.full_item_wording_status). "
                "Treat each item as 'are you best described by category_1 or category_2?'."
            ),
        }

    if study_id == "study3":
        cond = get_condition(materials, study_id, condition_id)
        traits = {t: render_trait_anchors(t, template) for t in cond["traits"]}
        return {
            "story_text": cond["story_text"],
            "consensus_question": cond["consensus_question"],
            "own_choice_options": cond["own_choice_options"],
            "trait_scales": traits,
            "instructions": (
                "First state which option you personally would choose. Then estimate the "
                "percentage of your peers who would choose each option (summing to 100). Then "
                "rate the typical person who would choose each option on each listed trait, "
                "-50 to +50."
            ),
        }

    if study_id == "study4":
        s4 = materials["study4"]
        versions = list(s4["target_person_materials"])
        random.shuffle(versions)
        agreed_version, refused_version = versions[0], versions[1]
        traits = {t: render_trait_anchors(t, template) for t in s4["traits"]["value"]}
        return {
            "likes_dislikes_intro": s4["likes_dislikes_intro_task"]["text"],
            "cover_story_script": s4["cover_story_script"]["text"],
            "own_choice_options": s4["own_choice_options"],
            "consensus_question": s4["consensus_question"],
            "target_persons": {
                "agreed_to_wear_sign": {"likes": agreed_version["likes"], "dislikes": agreed_version["dislikes"], "version": agreed_version["version"]},
                "refused_to_wear_sign": {"likes": refused_version["likes"], "dislikes": refused_version["dislikes"], "version": refused_version["version"]},
            },
            "trait_scales": traits,
            "instructions": (
                "State which choice you personally would make. Estimate the percentage of "
                "participants like you who would agree vs. refuse (summing to 100). Then rate "
                "each of the two target persons described above on each listed trait, -50 to +50."
            ),
            "faithfulness_caveat": (
                "This is a text-only reconstruction of Study 4's real-choice manipulation "
                "(see materials.json study4.design_note and task.json study4.blocking_reason). "
                "It is NOT a live behavioral replication: no real 30-minute campus walk, no "
                "live experimenter, and no genuine deception/debriefing occur here. Also, the "
                "8-trait list is an assumption carried over from Study 3 "
                "(evidence_label='missing', materials.json study4.traits)."
            ),
        }

    raise KeyError(f"unknown study_id {study_id!r}")


def validate_response(study_id, condition_id, response, materials):
    """Return (ok: bool, errors: list[str])."""
    errors = []
    if not isinstance(response, dict):
        return False, ["response must be a JSON object"]

    if study_id in ("study1", "study3", "study4"):
        options = None
        if study_id == "study1":
            options = get_condition(materials, study_id, condition_id)["own_choice_options"]
        elif study_id == "study3":
            options = get_condition(materials, study_id, condition_id)["own_choice_options"]
        else:
            options = materials["study4"]["own_choice_options"]

        own_choice = response.get("own_choice")
        if own_choice not in options:
            errors.append(f"own_choice must be one of {options}, got {own_choice!r}")

        estimate = response.get("consensus_estimate")
        if not isinstance(estimate, dict) or len(estimate) != 2:
            errors.append("consensus_estimate must be an object with exactly the two option percentages")
        else:
            keys_ok = set(estimate.keys()) == set(options)
            if not keys_ok:
                errors.append(f"consensus_estimate keys must be exactly {options}")
            else:
                try:
                    total = sum(float(v) for v in estimate.values())
                    if abs(total - 100.0) > 0.5:
                        errors.append(f"consensus_estimate percentages must sum to 100, got {total}")
                except (TypeError, ValueError):
                    errors.append("consensus_estimate values must be numeric")

        trait_ratings = response.get("trait_ratings")
        if not isinstance(trait_ratings, dict):
            errors.append("trait_ratings must be an object")
        else:
            for trait, ratings in trait_ratings.items():
                if not isinstance(ratings, dict):
                    errors.append(f"trait_ratings[{trait}] must be an object")
                    continue
                for target, val in ratings.items():
                    try:
                        fv = float(val)
                        if fv < -50 or fv > 50:
                            errors.append(f"trait_ratings[{trait}][{target}]={fv} out of [-50,50]")
                    except (TypeError, ValueError):
                        errors.append(f"trait_ratings[{trait}][{target}] must be numeric")

    elif study_id == "study2":
        self_category = response.get("self_category")
        peer_estimate_pct = response.get("peer_estimate_pct")
        if not isinstance(self_category, list) or not self_category:
            errors.append("self_category must be a non-empty list")
        if not isinstance(peer_estimate_pct, list) or not peer_estimate_pct:
            errors.append("peer_estimate_pct must be a non-empty list")
        if isinstance(peer_estimate_pct, list):
            for row in peer_estimate_pct:
                pct = row.get("pct_category_1") if isinstance(row, dict) else None
                try:
                    fv = float(pct)
                    if fv < 0 or fv > 100:
                        errors.append(f"pct_category_1={fv} out of [0,100]")
                except (TypeError, ValueError):
                    errors.append(f"peer_estimate_pct row missing numeric pct_category_1: {row!r}")
    else:
        errors.append(f"unknown study_id {study_id!r}")

    return (len(errors) == 0), errors


def synthetic_response(study_id, condition_id, materials):
    """Deterministic canned response used only by --smoke-test, never presented as real data."""
    if study_id in ("study1", "study3"):
        cond = get_condition(materials, study_id, condition_id)
        opts = cond["own_choice_options"]
        return {
            "own_choice": opts[0],
            "consensus_estimate": {opts[0]: 60.0, opts[1]: 40.0},
            "trait_ratings": {t: {opts[0]: 5.0, opts[1]: -5.0} for t in cond["traits"]},
        }
    if study_id == "study4":
        opts = materials["study4"]["own_choice_options"]
        traits = materials["study4"]["traits"]["value"]
        return {
            "own_choice": opts[0],
            "consensus_estimate": {opts[0]: 55.0, opts[1]: 45.0},
            "trait_ratings": {t: {"agreed_to_wear_sign": 3.0, "refused_to_wear_sign": -3.0} for t in traits},
        }
    if study_id == "study2":
        items = [it for it in materials["study2"]["items"] if it["category_1"] != STUDY2_EXCLUDED_CATEGORY_1]
        return {
            "self_category": [{"item_index": i, "category_chosen": "category_1"} for i in range(len(items))],
            "peer_estimate_pct": [{"item_index": i, "pct_category_1": 55.0} for i in range(len(items))],
        }
    raise KeyError(study_id)


def run_smoke_test():
    materials = load_json(MATERIALS_PATH)
    task = load_json(TASK_PATH)
    failures = []
    for study in task["studies"]:
        study_id = study["study_id"]
        for condition_id in study["conditions"]:
            try:
                prompt = build_prompt(study_id, condition_id, materials)
                if not prompt:
                    failures.append(f"{study_id}/{condition_id}: empty prompt")
                    continue
                resp = synthetic_response(study_id, condition_id, materials)
                ok, errors = validate_response(study_id, condition_id, resp, materials)
                if not ok:
                    failures.append(f"{study_id}/{condition_id}: {errors}")
                else:
                    print(f"PASS {study_id}/{condition_id}")
            except Exception as e:
                failures.append(f"{study_id}/{condition_id}: exception {e!r}")

    if failures:
        print("SMOKE TEST FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("SMOKE TEST: all study/condition cells passed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["study1", "study2", "study3", "study4"])
    parser.add_argument("--condition")
    parser.add_argument("--print-prompt", action="store_true", help="print the built prompt as JSON and exit")
    parser.add_argument("--response-file", help="path to a JSON file containing the participant's response")
    parser.add_argument("--out", help="path to write the final validated record; defaults to stdout")
    parser.add_argument("--smoke-test", action="store_true", help="run offline self-test over all study/condition cells")
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(run_smoke_test())

    if not args.study:
        parser.error("--study is required unless --smoke-test is given")

    materials = load_json(MATERIALS_PATH)
    task = load_json(TASK_PATH)

    study_def = next((s for s in task["studies"] if s["study_id"] == args.study), None)
    if study_def is None:
        parser.error(f"unknown study {args.study!r}")

    condition_id = args.condition or random.choice(study_def["conditions"])
    if condition_id not in study_def["conditions"]:
        parser.error(f"condition {condition_id!r} not valid for {args.study}; choices: {study_def['conditions']}")

    prompt = build_prompt(args.study, condition_id, materials)

    if args.print_prompt:
        print(json.dumps({"study_id": args.study, "condition_id": condition_id, "prompt": prompt}, indent=2))
        return

    if not args.response_file:
        parser.error("--response-file is required to record a response (or use --print-prompt to only view materials)")

    response = load_json(args.response_file)
    ok, errors = validate_response(args.study, condition_id, response, materials)
    if not ok:
        print(json.dumps({"status": "invalid_response", "errors": errors}, indent=2), file=sys.stderr)
        sys.exit(1)

    record = {"study_id": args.study, "condition_id": condition_id, **response}
    if args.study == "study4":
        record["faithfulness_caveat"] = prompt["faithfulness_caveat"]
    if args.study == "study2":
        record["faithfulness_caveat"] = prompt["faithfulness_caveat"]

    output = json.dumps(record, indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
