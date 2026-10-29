"""MCP server (stdio) for one HumanStudy program. Standard library only, Python 3.8+.

Each tool calls study_core.py, which reads program.json and the replay package of this skill; the
server adds no study facts and holds no model keys. The connected agent supplies AI participants'
replies itself (plan_participants -> its own model, one fresh context per prompt -> score_replies).

    claude mcp add <name> -- python3 /path/to/server.py
"""
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The protocol owns the real stdout. Everything else that writes to file descriptor 1 (package code,
# subprocesses, C extensions) goes to stderr, so it can never corrupt the JSON stream.
OUT = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
os.dup2(2, 1)
sys.stdout = sys.stderr
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(sys.stderr):
    from study_core import Refusal, Study, status_of  # noqa: E402

SUPPORTED = ["2025-06-18", "2025-03-26", "2024-11-05"]
STUDY = Study()  # the skill folder this script sits in
NAME = (STUDY.program.get("id") or "humanstudy-program")[:60]


def obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


S = {"type": "string"}
TOOLS = [
    ("study_overview", "Title, authors, the studies in the paper, open issues and whether the study can be run.",
     obj({}), lambda a: STUDY.overview()),
    ("capabilities", "What this study can do (read, run, human data, author code, reproduction, profiles), each "
     "yes / partial / no with its reason. Work outside these is refused with a reason, never approximated.",
     obj({}), lambda a: STUDY.capabilities()),
    ("list_conditions", "Conditions of one study, with ids for participant_prompt.",
     obj({"study_id": S}, ["study_id"]), lambda a: STUDY.conditions(a["study_id"])),
    ("read_study", "One study in full: participants, design, conditions, verbatim materials, ordered procedure, "
     "variables, analysis, reported results and open issues, each with its page in the paper.",
     obj({"study_id": S}, ["study_id"]), lambda a: STUDY.study_markdown(a["study_id"])),
    ("read_paper_context", "Paper-level overview, background, findings and limitations.",
     obj({}), lambda a: STUDY.paper_markdown()),
    ("get_item", "One program node or step with every source quote it rests on.",
     obj({"id": S}, ["id"]), lambda a: STUDY.node(a["id"])),
    ("search_study", "Find text in the program fields and in the quoted source passages.",
     obj({"query": S, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, ["query"]),
     lambda a: STUDY.search(a["query"], a.get("limit", 20))),
    ("list_open_issues", "What the paper does not state or what still needs a decision. Never fill these in silently.",
     obj({"study_id": S}), lambda a: STUDY.open_issues(a.get("study_id"))),
    ("participant_prompt", "Exactly what one AI participant reads in a condition, from the replay package. "
     "Give it to a fresh model context with no other instructions, then pass the reply to parse_reply.",
     obj({"study_id": S, "condition_id": S}, ["study_id", "condition_id"]),
     lambda a: STUDY.participant_prompt(a["study_id"], a["condition_id"])),
    ("parse_reply", "Turn one participant's reply into the response format the paper used.",
     obj({"study_id": S, "condition_id": S, "reply": S}, ["study_id", "condition_id", "reply"]),
     lambda a: STUDY.parse_reply(a["study_id"], a["condition_id"], a["reply"])),
    ("plan_participants", "Every prompt the AI participants will read (from the replay package, n per condition), in "
     "order: one entry per model call, usually one per AI participant. Answer each in a fresh model context that "
     "sees only that prompt, "
     "then send the replies, in the same order and with the same n, arms and seed, to score_replies. Never "
     "answer the prompts yourself in this conversation: you know the hypothesis.",
     obj({"n": {"type": "integer", "minimum": 1, "maximum": 500}, "arms": {"type": "array", "items": S},
          "seed": {"type": "integer"}}),
     lambda a: STUDY.plan(a.get("n", 5), a.get("arms"), a.get("seed", 20261015))),
    ("score_replies", "Rebuild the sessions from the participants' replies and run the paper's own analysis. "
     "Returns the evaluation and the sessions.",
     obj({"replies": {"type": "array", "items": S}, "n": {"type": "integer", "minimum": 1, "maximum": 500},
          "arms": {"type": "array", "items": S}, "seed": {"type": "integer"}}, ["replies"]),
     lambda a: STUDY.replay(a["replies"], a.get("n", 5), a.get("arms"), a.get("seed", 20261015))),
    ("score_sessions", "Run the paper's own analysis on a list of sessions (each with its condition and arm) "
     "and return the test, statistic and effect size, or what is missing.",
     obj({"sessions": {"type": "array", "items": {"type": "object"}}}, ["sessions"]),
     lambda a: STUDY.evaluate(a["sessions"])),
]
BY_NAME = {t[0]: t for t in TOOLS}


TYPES = {"string": str, "integer": int, "array": list, "object": dict}


def check_args(schema, args):
    """Reject missing or wrongly typed arguments before any study code runs."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    for key in schema.get("required", []):
        if key not in args:
            return "missing argument %r" % key
    for key, value in args.items():
        spec = schema["properties"].get(key)
        if spec is None:
            return "unknown argument %r" % key
        want = TYPES.get(spec.get("type"))
        if want and (not isinstance(value, want) or (want is int and isinstance(value, bool))):
            return "argument %r must be %s" % (key, spec["type"])
        if want is int and not spec.get("minimum", value) <= value <= spec.get("maximum", value):
            return "argument %r is out of range" % key
        item = TYPES.get(spec.get("items", {}).get("type"))
        if want is list and item and not all(isinstance(v, item) for v in value):
            return "every item of %r must be %s" % (key, spec["items"]["type"])
    return None


def ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def err(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def tool_result(text, is_error=False):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle(msg):
    """One JSON-RPC message -> its response, or None for notifications and client responses."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return err(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
    method, mid = msg.get("method"), msg.get("id")
    if not isinstance(method, str):
        return None if "result" in msg or "error" in msg else err(mid, -32600, "invalid request")
    if "id" not in msg:
        return None  # notification
    params = msg.get("params") if msg.get("params") is not None else {}
    if not isinstance(params, dict):
        return err(mid, -32602, "params must be an object")
    if method == "initialize":
        asked = params.get("protocolVersion")
        return ok(mid, {"protocolVersion": asked if asked in SUPPORTED else SUPPORTED[0],
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": NAME, "version": "1"},
                        "instructions": "Tools for the study '%s'. Answers come only from the program and its "
                                        "sources. AI participants' answers are a simulation, not data from people."
                                        % STUDY.program.get("title")})
    if method == "ping":
        return ok(mid, {})
    if method == "tools/list":
        return ok(mid, {"tools": [{"name": n, "description": d, "inputSchema": s} for n, d, s, _ in TOOLS]})
    if method == "tools/call":
        tool = BY_NAME.get(params.get("name"))
        if not tool:
            return err(mid, -32602, "unknown tool %r" % params.get("name"))
        args = params.get("arguments") if params.get("arguments") is not None else {}
        problem = check_args(tool[2], args)
        if problem:
            return ok(mid, tool_result("error: " + problem, True))
        try:
            with contextlib.redirect_stdout(sys.stderr):
                out = tool[3](args)
            if not isinstance(out, str):  # text answers (study, paper) are status ok
                if isinstance(out, dict) and out.get("status") in ("ok", "partial"):
                    pass
                elif isinstance(out, dict) and "status" not in out:
                    out = dict({"status": status_of(out)}, **out)
                else:  # lists, and dicts with a status of their own (an evaluation's not_ready)
                    out = {"status": status_of(out), "result": out}
            return ok(mid, tool_result(out if isinstance(out, str) else
                                       json.dumps(out, ensure_ascii=False, indent=2, default=str)))
        except Refusal as exc:  # the study cannot do this: say why and what to do instead
            return ok(mid, tool_result(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), True))
        except Exception as exc:  # report to the agent instead of dropping the connection
            detail = exc.args[0] if isinstance(exc, (KeyError, RuntimeError)) and exc.args else repr(exc)
            print(traceback.format_exc(), file=sys.stderr)
            return ok(mid, tool_result("error: %s" % detail, True))
    return err(mid, -32601, "method not found: %s" % method)


def send(message):
    OUT.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
    OUT.flush()


def main():
    for raw in sys.stdin.buffer:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            if raw.strip():
                send(err(None, -32700, "parse error"))
            continue
        if isinstance(msg, list):
            answers = [a for a in (handle(m) for m in msg) if a is not None]
            if not msg:
                send(err(None, -32600, "empty batch"))
            elif answers:
                send(answers)
        else:
            answer = handle(msg)
            if answer is not None:
                send(answer)


if __name__ == "__main__":
    main()
