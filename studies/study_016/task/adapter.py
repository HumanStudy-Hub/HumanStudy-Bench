#!/usr/bin/env python3
"""
Runnable adapter for the CoMAP human-study package.

This module never calls a network API itself. It orchestrates the procedure
described in task/task.json and expects the caller (an external agent
harness) to supply an `agent_fn` callable that actually produces each
participant agent's turn. This keeps the adapter's own execution fully
offline, so `--smoke-test` can run with no credentials and no network access.

agent_fn signature:
    agent_fn(role: str, system_prompt: str, messages: list[dict]) -> str
        role            e.g. "agent_a", "agent_b", "comap_global_agent"
        system_prompt   the fixed instructions for this role (verbatim from
                         materials.json for scripted AI roles; a
                         researcher-authored persona brief for participant
                         roles)
        messages        list of {"role": "user"|"assistant", "content": str}
        returns         the agent's next message as plain text

Scripted AI roles (comap_global_agent, comap_refine_agent, comap_split_agent)
must be called with the verbatim system prompts in materials.json -- do not
substitute a different prompt for these roles; that would silently break the
fidelity of the replayed manipulation.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TASK_JSON = PACKAGE_ROOT / "task" / "task.json"
MATERIALS_JSON = PACKAGE_ROOT / "materials" / "materials.json"

REFINE_ACTION_RE = re.compile(r"ACTION:\s*REFINE\s+node_id=(\S+)", re.IGNORECASE)
SPLIT_ACTION_RE = re.compile(r"ACTION:\s*SPLIT\s+node_id=(\S+)", re.IGNORECASE)


class BlockedError(Exception):
    """Raised when a faithful run is impossible without a researcher decision."""

    def __init__(self, missing_field, reason):
        self.missing_field = missing_field
        self.reason = reason
        super().__init__(f"blocked on '{missing_field}': {reason}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_task():
    return load_json(TASK_JSON)


def load_materials():
    return load_json(MATERIALS_JSON)


def get_study(task, study_id):
    for study in task["studies"]:
        if study["id"] == study_id:
            return study
    raise KeyError(f"no such study in task.json: {study_id}")


# ---------------------------------------------------------------------------
# Condition / tool wiring
# ---------------------------------------------------------------------------

def tool_for_block(condition_order, block_number):
    """condition_order 'A' = Baseline first; 'B' = CoMAP first."""
    if condition_order not in ("A", "B"):
        raise ValueError("condition_order must be 'A' or 'B'")
    if block_number == 1:
        return "baseline" if condition_order == "A" else "comap"
    if block_number == 2:
        return "comap" if condition_order == "A" else "baseline"
    raise ValueError("block_number must be 1 or 2")


def task_for_block(task_order, block_number):
    """task_order 1 = Task A first; 2 = Task B first."""
    if task_order not in (1, 2):
        raise ValueError("task_order must be 1 or 2")
    if block_number == 1:
        return "task_a" if task_order == 1 else "task_b"
    if block_number == 2:
        return "task_b" if task_order == 1 else "task_a"
    raise ValueError("block_number must be 1 or 2")


# ---------------------------------------------------------------------------
# Local (Refine/Split) agent routing -- see task.json local_agent_invocation_protocol
# ---------------------------------------------------------------------------

def route_local_agent_actions(participant_text, materials, agent_fn, node_registry):
    """
    Scans a CoMAP-condition participant turn for ACTION: REFINE/SPLIT lines,
    invokes the corresponding scripted agent verbatim, and returns a list of
    {action, node_id, result} records to attach to the transcript.
    """
    records = []
    for match in REFINE_ACTION_RE.finditer(participant_text):
        node_id = match.group(1)
        node = node_registry.get(node_id)
        prompt = materials["comap_ai_agent_prompts"]["refine_agent"]["system_prompt"]
        user_msg = json.dumps({"node_id": node_id, "current_content": node})
        result = agent_fn("comap_refine_agent", prompt, [{"role": "user", "content": user_msg}])
        records.append({"action": "refine", "node_id": node_id, "result": result})
    for match in SPLIT_ACTION_RE.finditer(participant_text):
        node_id = match.group(1)
        node = node_registry.get(node_id)
        prompt = materials["comap_ai_agent_prompts"]["split_agent"]["system_prompt"]
        user_msg = json.dumps({"old_node_id": node_id, "current_content": node})
        result = agent_fn("comap_split_agent", prompt, [{"role": "user", "content": user_msg}])
        records.append({"action": "split", "node_id": node_id, "result": result})
    return records


# ---------------------------------------------------------------------------
# Single-agent steps
# ---------------------------------------------------------------------------

def run_background_survey(role, agent_fn, materials):
    subscale_responses = {}
    for subscale in materials["background_survey"]["subscales"]:
        answers = []
        for item in subscale["items"]:
            prompt = (
                "You are answering a background survey item on a 1-7 scale "
                "(1=Strongly Disagree, 7=Strongly Agree). Respond with ONLY the integer.\n"
                f"Item: {item}"
            )
            reply = agent_fn(role, "You are a study participant honestly self-reporting your background.", [
                {"role": "user", "content": prompt}
            ])
            answers.append({"item": item, "response_raw": reply})
        subscale_responses[subscale["id"]] = answers
    return subscale_responses


def run_design_task(role, agent_fn, materials, tool, task_key, block_number):
    """
    Runs one participant's private design-task block. Returns a transcript
    dict. For 'baseline', raises BlockedError unless materials contains an
    unblocked baseline system prompt (see --baseline-system-prompt-file).
    """
    task_prompt = materials["design_tasks"][task_key]["prompt"]
    task_title = materials["design_tasks"][task_key]["title"]

    if tool == "baseline":
        baseline_prompt = materials.get("baseline_condition_override_system_prompt")
        if not baseline_prompt:
            raise BlockedError(
                "baseline_system_prompt",
                "materials.json 'baseline_condition' has no verbatim system prompt; "
                "supply one via --baseline-system-prompt-file to unblock this arm.",
            )
        history = [{"role": "user", "content": f"Design task ({task_title}): {task_prompt}"}]
        reply = agent_fn(role, baseline_prompt, history)
        return {
            "tool": "baseline",
            "task": task_key,
            "block": block_number,
            "transcript": history + [{"role": "assistant", "content": reply}],
            "design_artifact": {"format": "freeform_document", "content": reply},
        }

    if tool == "comap":
        global_prompt = materials["comap_ai_agent_prompts"]["global_agent"]["system_prompt"]
        participant_opening = agent_fn(
            role,
            (
                "You are a study participant using the CoMAP graph canvas tool to design a "
                f"PBL unit. Your task ({task_title}): {task_prompt} "
                "Address the Global Agent with your first request. "
                "If you want to invoke the Refine or Split agent on a node you have already "
                "created, include a line 'ACTION: REFINE node_id=<id>' or "
                "'ACTION: SPLIT node_id=<id>'."
            ),
            [],
        )
        global_reply = agent_fn("comap_global_agent", global_prompt, [
            {"role": "user", "content": participant_opening}
        ])
        node_registry = {}
        actions = _extract_add_actions(global_reply)
        for act in actions:
            node_registry[act.get("card_id") or act.get("id") or str(len(node_registry) + 1)] = act
        local_actions = route_local_agent_actions(participant_opening, materials, agent_fn, node_registry)
        return {
            "tool": "comap",
            "task": task_key,
            "block": block_number,
            "transcript": [
                {"role": "user", "content": participant_opening},
                {"role": "assistant", "content": global_reply},
            ],
            "design_artifact": {"format": "structured_graph", "nodes": node_registry},
            "local_agent_invocations": local_actions,
        }

    raise ValueError(f"unknown tool: {tool}")


def _extract_add_actions(global_agent_reply):
    """Best-effort extraction of JSON action blocks from a Global Agent reply."""
    actions = []
    for match in re.finditer(r"\{.*?\"actions\"\s*:\s*\[.*?\]\s*\}", global_agent_reply, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            actions.extend(parsed.get("actions", []))
        except json.JSONDecodeError:
            continue
    return actions


def run_sharing(presenter_role, listener_role, design_artifact, agent_fn):
    presentation = agent_fn(
        presenter_role,
        "You are presenting your finished PBL design to a partner in a brief mock-teaching format.",
        [{"role": "user", "content": json.dumps(design_artifact)}],
    )
    listener_summary = agent_fn(
        listener_role,
        "Your partner just presented their PBL design to you in a mock-teaching format. "
        "Briefly summarize what you understood.",
        [{"role": "user", "content": presentation}],
    )
    return {"presentation": presentation, "listener_summary": listener_summary}


def run_post_task_questionnaires(role, agent_fn, materials, understanding_target_present):
    responses = {}
    scale_key = "human_ai_interaction_experience"
    for subscale in materials["post_task_questionnaires"][scale_key]["subscales"]:
        answers = []
        for item in subscale["items"]:
            reply = agent_fn(role, "Answer on a 1-7 scale (1=Strongly Disagree, 7=Strongly Agree). Respond with ONLY the integer.", [
                {"role": "user", "content": item}
            ])
            answers.append({"item": item, "response_raw": reply})
        responses[subscale["id"]] = answers

    expr_items = materials["post_task_questionnaires"]["perceived_expression"]["items"]
    responses["perceived_expression"] = [
        {"item": item, "response_raw": agent_fn(role, "Answer on a 1-7 scale (1=Strongly Disagree, 7=Strongly Agree). Respond with ONLY the integer.", [{"role": "user", "content": item}])}
        for item in expr_items
    ]

    if understanding_target_present:
        und_items = materials["post_task_questionnaires"]["perceived_understanding"]["items"]
        responses["perceived_understanding"] = [
            {"item": item, "response_raw": agent_fn(role, "Answer on a 1-7 scale (1=Strongly Disagree, 7=Strongly Agree), about your partner's design you just saw. Respond with ONLY the integer.", [{"role": "user", "content": item}])}
            for item in und_items
        ]
    return responses


# ---------------------------------------------------------------------------
# Dyad-level session
# ---------------------------------------------------------------------------

def run_dyad_session(dyad_id, agent_a_order, agent_a_task_order, agent_b_order, agent_b_task_order,
                      agent_fn, materials, baseline_system_prompt=None):
    if baseline_system_prompt:
        materials = dict(materials)
        materials["baseline_condition_override_system_prompt"] = baseline_system_prompt

    transcript = {"dyad_id": dyad_id, "status": "ok", "agents": {}}
    agents = {
        "agent_a": {"order": agent_a_order, "task_order": agent_a_task_order},
        "agent_b": {"order": agent_b_order, "task_order": agent_b_task_order},
    }

    for role in ("agent_a", "agent_b"):
        transcript["agents"][role] = {"background_survey": run_background_survey(role, agent_fn, materials)}

    block_designs = {"agent_a": {}, "agent_b": {}}
    blocked_reason = None

    for block_number in (1, 2):
        for role in ("agent_a", "agent_b"):
            cfg = agents[role]
            tool = tool_for_block(cfg["order"], block_number)
            task_key = task_for_block(cfg["task_order"], block_number)
            try:
                result = run_design_task(role, agent_fn, materials, tool, task_key, block_number)
                block_designs[role][block_number] = result
            except BlockedError as e:
                blocked_reason = {"missing_field": e.missing_field, "reason": e.reason, "role": role, "block": block_number}
                block_designs[role][block_number] = {"status": "blocked", **blocked_reason}

        sharing = None
        a_design = block_designs["agent_a"][block_number]
        b_design = block_designs["agent_b"][block_number]
        if a_design.get("status") != "blocked" and b_design.get("status") != "blocked":
            sharing_a_to_b = run_sharing("agent_a", "agent_b", a_design["design_artifact"], agent_fn)
            sharing_b_to_a = run_sharing("agent_b", "agent_a", b_design["design_artifact"], agent_fn)
            sharing = {"a_presents_to_b": sharing_a_to_b, "b_presents_to_a": sharing_b_to_a}

        for role in ("agent_a", "agent_b"):
            design = block_designs[role][block_number]
            key = f"block{block_number}"
            transcript["agents"][role][key] = {"design_task": design}
            if design.get("status") == "blocked":
                transcript["agents"][role][key]["questionnaires"] = {"status": "blocked", **blocked_reason}
                transcript["status"] = "blocked"
            else:
                transcript["agents"][role][key]["questionnaires"] = run_post_task_questionnaires(
                    role, agent_fn, materials, understanding_target_present=sharing is not None
                )
        if sharing is not None:
            transcript[f"block{block_number}_sharing"] = sharing

    return transcript


# ---------------------------------------------------------------------------
# Formative study (Study 1) -- Stage 1 only; Stage 2 is blocked (see audit)
# ---------------------------------------------------------------------------

def run_formative_session(participant_id, agent_fn, materials):
    scripts = materials["interview_scripts"]["formative_study"]["stage_1_semi_structured_interview"]
    role = f"formative_participant_{participant_id}"
    transcript = {"participant_id": participant_id, "stage_1": {}, "stage_2": {"status": "blocked", "missing_field": "formative_stage2_script"}}
    for section, questions in scripts.items():
        answers = []
        for q in questions:
            reply = agent_fn(role, "You are a PBL practitioner being interviewed about your design practice and AI use.", [
                {"role": "user", "content": q}
            ])
            answers.append({"question": q, "answer": reply})
        transcript["stage_1"][section] = answers
    return transcript


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _stub_agent_fn(role, system_prompt, messages):
    """
    Deterministic, offline, no-network stand-in used only by --smoke-test.
    Clearly labeled placeholder text -- never presented as paper content.
    """
    last_user = messages[-1]["content"] if messages else ""
    if re.search(r"1-7 scale", system_prompt) or "1-7 scale" in last_user:
        return "5"
    if role == "comap_global_agent":
        return json.dumps({
            "actions": [
                {"option": "add", "type": "Activity", "title": "[SMOKE-TEST PLACEHOLDER] Sample Activity",
                 "description": "[SMOKE-TEST PLACEHOLDER] generated by the stub agent, not from the paper."}
            ]
        })
    if role in ("comap_refine_agent",):
        return json.dumps({"new_node": {"id": "1", "title": "[SMOKE-TEST PLACEHOLDER]", "description": "...", "tag": "Activity"}})
    if role in ("comap_split_agent",):
        return json.dumps({"old_node_id": "1", "new_nodes": [{"title": "[SMOKE-TEST PLACEHOLDER] part 1", "description": "...", "tag": "Activity"}]})
    return f"[SMOKE-TEST PLACEHOLDER response from {role}]"


def _run_smoke_test():
    materials = load_materials()
    task = load_task()
    assert get_study(task, "study_1_formative")
    assert get_study(task, "study_2_main")

    print("[1/3] Running one formative-study participant (Stage 1 only)...")
    formative_result = run_formative_session(1, _stub_agent_fn, materials)
    assert formative_result["stage_1"]
    assert formative_result["stage_2"]["status"] == "blocked"
    print("      ok -- stage 1 produced answers, stage 2 correctly reports blocked.")

    print("[2/3] Running one dyad through Block 1/Block 2 in the CoMAP-only path...")
    # Force both agents to be CoMAP-first-and-second is impossible (crossover
    # requires one Baseline block); instead verify the CoMAP-condition code
    # path directly without going through the Baseline-blocking dyad runner.
    materials_copy = dict(materials)
    comap_result = run_design_task("agent_a", _stub_agent_fn, materials_copy, tool="comap", task_key="task_a", block_number=1)
    assert comap_result["tool"] == "comap"
    assert comap_result["design_artifact"]["nodes"]
    print("      ok -- CoMAP design task produced a structured design artifact.")

    print("[3/3] Verifying the Baseline arm is correctly blocked without a supplied system prompt...")
    blocked = False
    try:
        run_design_task("agent_a", _stub_agent_fn, materials, tool="baseline", task_key="task_a", block_number=1)
    except BlockedError as e:
        blocked = e.missing_field == "baseline_system_prompt"
    assert blocked, "expected a BlockedError for the Baseline arm with no system prompt supplied"
    print("      ok -- Baseline arm reports a structured blocked state instead of fabricating a prompt.")

    print("\nSMOKE TEST PASSED. No network access was used. All output above is placeholder text from a stub agent, not paper content.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="CoMAP human-study package adapter")
    parser.add_argument("--smoke-test", action="store_true", help="run an offline self-test with a stub agent")
    parser.add_argument("--study", choices=["study_1_formative", "study_2_main"], default="study_2_main")
    parser.add_argument("--dyad-id", type=int, default=1)
    parser.add_argument("--agent-a-order", choices=["A", "B"], default="A")
    parser.add_argument("--agent-a-task-order", type=int, choices=[1, 2], default=1)
    parser.add_argument("--agent-b-order", choices=["A", "B"], default="B")
    parser.add_argument("--agent-b-task-order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--baseline-system-prompt-file", type=str, default=None,
                         help="path to a researcher-supplied Baseline system prompt (required to unblock the Baseline arm)")
    parser.add_argument("--agent-fn-module", type=str, default=None,
                         help="'module:function' providing agent_fn(role, system_prompt, messages) -> str; "
                              "required for a real (non-smoke-test) run")
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(_run_smoke_test())

    if not args.agent_fn_module:
        print("ERROR: --agent-fn-module is required for a real run (or use --smoke-test).", file=sys.stderr)
        sys.exit(2)

    module_name, func_name = args.agent_fn_module.split(":")
    import importlib
    module = importlib.import_module(module_name)
    agent_fn = getattr(module, func_name)

    materials = load_materials()
    baseline_prompt = None
    if args.baseline_system_prompt_file:
        baseline_prompt = Path(args.baseline_system_prompt_file).read_text(encoding="utf-8")

    if args.study == "study_1_formative":
        result = run_formative_session(args.dyad_id, agent_fn, materials)
    else:
        result = run_dyad_session(
            args.dyad_id, args.agent_a_order, args.agent_a_task_order,
            args.agent_b_order, args.agent_b_task_order,
            agent_fn, materials, baseline_system_prompt=baseline_prompt,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
