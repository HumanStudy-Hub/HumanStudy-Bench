"""Read and run one HumanStudy program from any agent.

Every answer comes from program.json and the replay package next to it; nothing here adds study
facts. Standard library only, Python 3.8+, so the folder works outside the HumanStudy app.

    python study_core.py overview | studies | study <id> | node <id> | search <text> | issues
    python study_core.py prompt <study> <condition>    exactly what one AI participant reads
    python study_core.py score sessions.jsonl          the paper's own analysis on those sessions
    python study_core.py plan --n 5 > plan.json        every participant prompt, for your own model to answer
    python study_core.py replay replies.json --n 5     score those answers (same n, arms and seed)
    python study_core.py run --cli "claude -p --model haiku" [--n 5]   one new CLI process per participant
    python study_core.py run --model <id> [--n 5] [--arm A]   needs OPENROUTER_API_KEY or OPENAI_API_KEY
    python study_core.py capabilities                  what this study can do, each answer with its reason

Work the study cannot do is refused, never approximated: the CLI prints
{"status": "refused", "reason", "capability", "next_steps"} and exits 3; over MCP the same object is the
tool result. Answers that run only part of the paper carry status "partial".
"""
import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

PAPER_KINDS = ["overview", "background", "finding", "limitation"]
STUDY_KINDS = ["hypothesis", "participants", "design", "condition", "material", "procedure",
               "variable", "analysis", "result", "simulation"]
KIND_TITLES = {"hypothesis": "Hypotheses", "participants": "Participants", "design": "Design",
               "condition": "Conditions", "material": "Materials", "procedure": "Procedure",
               "variable": "Variables", "analysis": "Analysis", "result": "Reported results",
               "simulation": "Simulation scope", "overview": "Overview", "background": "Background",
               "finding": "Findings", "limitation": "Limitations"}
MISSING_TEXT = "Not stated in the paper."
SIMULATION_NOTE = "AI participants: a simulation, not observations of people."


class Refusal(RuntimeError):
    """Work this study cannot do. Carries why, which capability it depends on, and what to do instead."""

    def __init__(self, reason, capability, next_steps=()):
        RuntimeError.__init__(self, reason)
        self.reason, self.capability, self.next_steps = reason, capability, list(next_steps)

    def to_dict(self):
        return {"status": "refused", "reason": self.reason, "capability": self.capability,
                "next_steps": self.next_steps}


def has_not_ready(value):
    """True when an evaluation leaves any part not_ready (so the answer is partial)."""
    if isinstance(value, dict):
        return value.get("status") == "not_ready" or "not_ready" in value or any(has_not_ready(v) for v in value.values())
    if isinstance(value, list):
        return any(has_not_ready(v) for v in value)
    return value == "not_ready"


def status_of(value):
    return "partial" if has_not_ready(value) else "ok"


def find_root(start=None):
    """The folder holding program.json: this file's folder or one of its parents."""
    for base in [Path(start).resolve()] if start else [HERE, *HERE.parents]:
        if (base / "program.json").is_file():
            return base
    raise FileNotFoundError("program.json not found next to study_core.py or in a parent folder")


def text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
        return "; ".join(str(v) for v in value)
    return json.dumps(value, ensure_ascii=False)


class Study:
    def __init__(self, root=None):
        self.root = find_root(root)
        self.program = json.loads((self.root / "program.json").read_text(encoding="utf-8"))
        p = self.program
        self.studies = p.get("studies", [])
        self.nodes = p.get("nodes", [])
        self.steps = p.get("steps", [])
        self.evidence = {e["id"]: e for e in p.get("evidence", []) if "id" in e}
        self.issues = p.get("issues", [])
        self.node_by_id = {n["id"]: n for n in self.nodes if "id" in n}
        packages = sorted(d for d in (self.root / "package").glob("*") if (d / "task/adapter.py").is_file()) \
            if (self.root / "package").is_dir() else []
        self.package = packages[0] if len(packages) == 1 else None
        self._modules = {}
        try:
            self.caps = json.loads((self.root / "capabilities.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.caps = None

    # ---- reading -------------------------------------------------------------------------------

    def cite(self, evidence_ids):
        out = []
        for eid in evidence_ids or []:
            e = self.evidence.get(eid)
            if not e:
                continue
            loc = e.get("locator", {})
            if loc.get("page"):
                ref = "p. %s" % loc["page"]
            else:
                ref = e.get("path") or e.get("url") or eid
                if loc.get("sheet") or loc.get("cell"):
                    ref += " " + "!".join(x for x in [loc.get("sheet"), loc.get("cell")] if x)
            if ref not in out:
                out.append(ref)
        return ", ".join(out)

    def readiness(self):
        blocking = [i["id"] for i in self.issues if i.get("severity") == "blocking"]
        if self.package and (self.package / "study.json").is_file():
            try:
                r = json.loads((self.package / "study.json").read_text(encoding="utf-8")).get("readiness")
                if isinstance(r, dict):
                    if blocking and r.get("status") != "blocked":  # program changed after the package
                        return {"status": "blocked", "blocking": blocking}
                    return r
            except ValueError:
                pass
        return {"status": "blocked" if blocking or not self.package else "unknown", "blocking": blocking,
                "note": None if self.package else "no replay package was built"}

    def overview(self):
        src = self.program.get("source", {})
        counts = {}
        for i in self.issues:
            counts[i.get("severity", "check")] = counts.get(i.get("severity", "check"), 0) + 1
        return {"id": self.program.get("id"), "title": self.program.get("title"),
                "authors": src.get("authors"), "venue": src.get("venue"), "year": src.get("year"),
                "doi": src.get("doi"),
                "studies": [{"id": s.get("id"), "title": s.get("title"), "type": s.get("type")} for s in self.studies],
                "openIssues": counts, "readiness": self.readiness(),
                "capabilities": self.capabilities(),
                "package": self.package.name if self.package else None}

    def capabilities(self):
        """capabilities.json as built; without one, only what readiness alone shows."""
        if isinstance(self.caps, dict):
            return self.caps
        ready = self.readiness()
        run = {"status": "yes" if ready.get("status") == "ready" else "no" if ready.get("status") == "blocked" else "unknown",
               "reason": ready.get("note") or ("blocked by: " + ", ".join(ready.get("blocking") or []) if ready.get("status") == "blocked" else None)}
        return {"schemaVersion": 1, "read": {"status": "yes", "reason": None}, "run": run,
                "note": "No capabilities.json in this export; human data, author code and reproduction are unknown."}

    def study_nodes(self, study_id):
        return [n for n in self.nodes if study_id in (n.get("studyIds") or [])
                and n.get("kind") in STUDY_KINDS]

    def conditions(self, study_id):
        return [{"id": n["id"], "title": n.get("title"), "summary": n.get("summary")}
                for n in self.study_nodes(study_id) if n.get("kind") == "condition"]

    def field_lines(self, node):
        lines = []
        for f in node.get("fields", []):
            cite = self.cite(f.get("evidenceIds"))
            tail = " (%s)" % cite if cite else ""
            if f.get("state") == "missing":
                lines.append("- **%s**: %s%s" % (f.get("label"), MISSING_TEXT, tail))
            elif f.get("origin") == "verbatim":
                body = text(f.get("value")).strip()
                flags = ["verbatim"] + [x for x in ["to check" if f.get("state") == "check" else None,
                                                    "open decision" if f.get("state") == "decision" else None] if x]
                lines.append("- **%s** (%s%s):" % (f.get("label"), ", ".join(flags), ", " + cite if cite else ""))
                lines.extend("  > " + ln if ln.strip() else "  >" for ln in body.splitlines() or [""])
            else:
                label = f.get("label")
                notes = [x for x in [f.get("origin") if f.get("origin") not in (None, "reported") else None,
                                     "to check" if f.get("state") == "check" else None,
                                     "open decision" if f.get("state") == "decision" else None] if x]
                if notes:
                    label = "%s [%s]" % (label, ", ".join(notes))
                lines.append("- **%s**: %s%s" % (label, text(f.get("value")), tail))
                if f.get("derivation"):
                    lines.append("  - how it was derived: %s" % f["derivation"])
        return lines

    def node_markdown(self, node, level=3):
        out = ["%s %s" % ("#" * level, node.get("title") or node.get("id")), "",
               "`%s`" % node.get("id")]
        if node.get("summary"):
            out += ["", node["summary"]]
        if node.get("description") and node.get("description") != node.get("summary"):
            out += ["", node["description"]]
        conds = [self.node_by_id[c].get("title", c) if c in self.node_by_id else c for c in node.get("conditionIds", [])]
        if conds:
            out += ["", "Conditions: " + "; ".join(conds)]
        lines = self.field_lines(node)
        if lines:
            out += [""] + lines
        return "\n".join(out)

    def procedure_lines(self, study_id):
        steps = [s for s in self.steps if s.get("studyId") == study_id]
        by_id = {s["id"]: s for s in steps}
        children = {c for s in steps for c in s.get("children", [])}
        targets = {n.get("to") for s in steps for n in s.get("next", [])}
        roots = [s for s in steps if s["id"] not in children]
        starts = [s for s in roots if s["id"] not in targets] or roots
        seen, out = set(), []

        def visit(step, depth):
            if step["id"] in seen:
                out.append("%s- (back to: %s)" % ("  " * depth, step.get("title")))
                return
            seen.add(step["id"])
            kind = step.get("kind")
            label = {"branch": "Depending on condition: ", "repeat": "Repeated: ",
                     "parallel": "At the same time: "}.get(kind, "")
            cite = self.cite(step.get("evidenceIds"))
            visible = [self.node_by_id[i].get("title", i) for i in step.get("visibleIds", []) if i in self.node_by_id]
            line = "%s- %s%s" % ("  " * depth, label, step.get("title"))
            if visible:
                line += " (participant sees: %s)" % "; ".join(visible)
            if cite:
                line += " (%s)" % cite
            out.append(line)
            pad = "  " * (depth + 1)
            conds = [self.node_by_id[c].get("title", c) if c in self.node_by_id else c for c in step.get("conditionIds", [])]
            if conds:
                out.append("%s- conditions: %s" % (pad, "; ".join(conds)))
            if isinstance(step.get("rule"), dict):
                out.extend(pad + ln for ln in self.field_lines({"fields": [dict(step["rule"], label=step["rule"].get("label") or "Rule")]}))
            for cid in step.get("children", []):
                if cid in by_id:
                    visit(by_id[cid], depth + 1)
            for nxt in step.get("next", []):
                target = by_id.get(nxt.get("to"))
                if not target:
                    continue
                if nxt.get("when"):
                    out.append("%s  - when %s:" % ("  " * depth, text(nxt["when"])))
                    visit(target, depth + 2)
                else:
                    visit(target, depth)

        for s in starts:
            visit(s, 0)
        for s in steps:  # roots first, then anything only reachable through a cycle
            if s["id"] not in seen:
                visit(s, 0)
        return out

    def study_markdown(self, study_id):
        study = next((s for s in self.studies if s.get("id") == study_id), None)
        if not study:
            raise KeyError("unknown study %r; known: %s" % (study_id, ", ".join(s.get("id") for s in self.studies)))
        out = ["# %s" % study.get("title"), "", "`%s`" % study_id]
        if study.get("description"):
            out += ["", study["description"]]
        nodes = self.study_nodes(study_id)
        for kind in STUDY_KINDS:
            group = [n for n in nodes if n.get("kind") == kind]
            if kind == "procedure":
                flow = self.procedure_lines(study_id)
                if not group and not flow:
                    continue
                out += ["", "## Procedure"]
                if flow:
                    out += ["", "Order of steps:", ""] + flow
                for n in group:
                    out += ["", self.node_markdown(n)]
                continue
            if not group:
                continue
            out += ["", "## " + KIND_TITLES[kind]]
            for n in group:
                out += ["", self.node_markdown(n)]
        issues = [i for i in self.issues if i.get("studyId") == study_id]
        if issues:
            out += ["", "## Open issues", ""] + [self.issue_line(i) for i in issues]
        return "\n".join(out) + "\n"

    def paper_markdown(self):
        out = []
        for kind in PAPER_KINDS:
            for n in self.nodes:
                if n.get("kind") == kind and not n.get("studyIds"):
                    out += ["", self.node_markdown(n, 2)]
        return "\n".join(out).strip() + "\n"

    def issue_line(self, i):
        cite = self.cite(i.get("evidenceIds"))
        where = i.get("nodeId") or i.get("studyId") or "paper"
        line = "- **%s** [%s, `%s`, %s]: %s" % (i.get("title"), i.get("severity"), i.get("id"), where, i.get("reason", ""))
        if i.get("suggestedAction"):
            line += " Suggested: %s" % i["suggestedAction"]
        return line + (" (%s)" % cite if cite else "")

    def open_issues(self, study_id=None):
        return [i for i in self.issues if not study_id or i.get("studyId") == study_id]

    def node(self, node_id):
        n = self.node_by_id.get(node_id) or next((s for s in self.steps if s.get("id") == node_id), None)
        if not n:
            raise KeyError("unknown node or step %r" % node_id)
        ids = set(n.get("evidenceIds", [])) | {e for f in n.get("fields", []) for e in f.get("evidenceIds", [])}
        return {"item": n, "evidence": [self.evidence[e] for e in sorted(ids) if e in self.evidence]}

    def search(self, query, limit=20):
        limit = max(1, min(int(limit), 50))
        words = [w for w in str(query).lower().split() if w]
        if not words:
            return []
        hits = []

        def score(hay):
            hay = hay.lower()
            return sum(hay.count(w) for w in words) if all(w in hay for w in words) else 0

        for n in self.nodes:
            for f in n.get("fields", []):
                s = score("%s %s %s" % (n.get("title"), f.get("label"), text(f.get("value"))))
                if s:
                    hits.append((s, {"node": n["id"], "field": f.get("id"), "title": n.get("title"),
                                     "label": f.get("label"), "value": text(f.get("value"))[:400],
                                     "source": self.cite(f.get("evidenceIds"))}))
        for e in self.evidence.values():
            s = score(e.get("quote", ""))
            if s:
                hits.append((s, {"evidence": e["id"], "quote": e.get("quote", "")[:400], "source": self.cite([e["id"]])}))
        hits.sort(key=lambda h: -h[0])
        return [h for _, h in hits[:limit]]

    # ---- running -------------------------------------------------------------------------------

    def module(self, name):
        if not self.package:
            raise Refusal("this export has no replay package; only reading is available", "run",
                          ["read the study with study/read_study", "build the package in the HumanStudy app"])
        if name not in self._modules:
            path = self.package / ("task/adapter.py" if name == "adapter" else "evaluation/evaluation.py")
            spec = importlib.util.spec_from_file_location("hs_" + name, path)
            mod = importlib.util.module_from_spec(spec)
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))  # kept: adapters may import siblings lazily
            sys.modules[spec.name] = mod  # dataclasses and pickling look the module up here
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop(spec.name, None)
                raise
            self._modules[name] = mod
        return self._modules[name]

    def adapter_function(self, name):
        fn = getattr(self.module("adapter"), name, None)
        if not callable(fn):
            raise Refusal("the replay package's adapter has no %s()" % name, "run",
                          ["use run with a model or CLI instead, which calls run_sessions()"])
        return fn

    def participant_prompt(self, study_id, condition_id):
        self.require_ready()
        return self.adapter_function("build_prompt")(study_id, condition_id)

    def parse_reply(self, study_id, condition_id, reply):
        self.require_ready()
        return self.adapter_function("parse_reply")(study_id, condition_id, reply)

    def evaluate(self, sessions):
        self.require_ready()
        return self.module("evaluation").evaluate(sessions)

    def require_ready(self):
        ready = self.readiness()
        if ready.get("status") == "blocked":
            open_ids = ready.get("blocking") or [i["id"] for i in self.issues if i.get("severity") in ("blocking", "decision")]
            raise Refusal("the study is not ready to run AI participants; settle these first: %s"
                          % (", ".join(open_ids) if open_ids else ready.get("note") or "see list_open_issues"), "run",
                          ["list_open_issues shows what the paper does not state",
                           "a researcher settles each blocking issue in the HumanStudy app (never fill it in silently)"])

    def require_arms(self, arms):
        """Refuse arms the study declares it cannot run (capabilities.json run.excluded[].arms)."""
        excluded = ((self.caps or {}).get("run") or {}).get("excluded") or []
        for item in excluded:
            hit = sorted(set(arms or []) & set(item.get("arms") or []))
            if hit:
                raise Refusal("%s cannot be run: %s" % (", ".join(hit), item.get("reason")), "run.excluded",
                              ["run the other arms", "see capabilities for what is excluded and why"])

    def _sessions(self, llm, seed, n, arms):
        self.require_ready()
        self.require_arms(arms)
        return self.module("adapter").run_sessions(llm, seed, n or 5, arms)

    def run(self, llm, seed=20261015, n=None, arms=None):
        """Run AI participants through the package's own run_sessions with a model you provide."""
        sessions = self._sessions(llm, seed, n, arms)
        return self.result(sessions, arms)

    def plan(self, n=5, arms=None, seed=20261015):
        """Every prompt the participants will read, in order, so another agent can answer them itself.

        Answer each prompt in a fresh context, then pass the replies, in the same order and with the same
        n, arms and seed, to replay()."""
        calls = []

        def record(prompt, *args, **kw):
            calls.append({"index": len(calls), "prompt": prompt, "options": (list(args) or None, kw or None)})
            return ""
        self._sessions(record, seed, n, arms)
        for c in calls:
            if c["options"] == (None, None):
                del c["options"]
        return {"n": n, "arms": arms, "seed": seed, "prompts": calls,
                "note": "One entry per model call the package makes, in order (usually one per AI participant). "
                        "Answer each in a fresh context that sees only that prompt."}

    def replay(self, replies, n=5, arms=None, seed=20261015):
        """Score replies collected for plan(): the package rebuilds each session and runs the paper's analysis."""
        planned = self.plan(n, arms, seed)["prompts"]
        if len(replies) != len(planned):
            raise RuntimeError("expected %d replies (one per planned prompt), got %d" % (len(planned), len(replies)))
        queue = iter(enumerate(replies))

        def answer(prompt, *args, **kw):
            try:
                i, reply = next(queue)
            except StopIteration:
                raise Refusal("the package asked for more answers than planned: the study is not "
                              "deterministic for plan/replay", "run.plan", ["use run with a model or CLI instead"])
            if prompt != planned[i]["prompt"]:
                raise Refusal("prompt %d depends on earlier answers, so this study must be run turn by turn" % i,
                              "run.plan", ["use run with a model or CLI instead"])
            return reply if isinstance(reply, str) else json.dumps(reply)
        sessions = self._sessions(answer, seed, n, arms)
        left = sum(1 for _ in queue)
        if left:
            raise Refusal("%d replies were not used: the package's calls differed from the plan" % left,
                          "run.plan", ["use run with a model or CLI instead"])
        return self.result(sessions, arms)

    def result(self, sessions, arms):
        """Sessions plus the paper's analysis; partial when part of the paper was not run or analysed."""
        evaluation = self.evaluate(sessions)
        run = (self.capabilities().get("run") or {})
        skipped = [] if arms else list(run.get("excluded") or [])  # chosen arms: excluded ones were refused
        status = "partial" if skipped or has_not_ready(evaluation) else "ok"
        out = {"status": status, "note": SIMULATION_NOTE, "evaluation": evaluation, "sessions": sessions}
        if skipped:
            out["notRun"] = [{"what": x.get("what"), "reason": x.get("reason")} for x in skipped]
        return out


def http_llm(model, temperature=1.0):
    """llm(prompt) -> str over an OpenAI-compatible API. Keys are read from the environment only."""
    key = os.environ.get("OPENROUTER_API_KEY")
    base = "https://openrouter.ai/api/v1"
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY, or OPENAI_API_KEY (and OPENAI_BASE_URL), to run AI participants")

    def llm(prompt, **kw):
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        if kw.get("system") and not isinstance(prompt, list):
            messages = [{"role": "system", "content": kw["system"]}] + messages
        body = json.dumps({"model": model, "messages": messages,
                           "temperature": kw.get("temperature", temperature)}).encode()
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""
    return llm


def split_command(command, windows=None):
    """Split a command line. On Windows keep backslashes in paths but drop the quotes around an argument."""
    if not (os.name == "nt" if windows is None else windows):
        return shlex.split(command)
    out = []
    for token in shlex.split(command, posix=False):
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        out.append(token)
    return out


def cli_llm(command, timeout=600):
    """llm(prompt) -> str by starting a new agent CLI process per call (one fresh context per participant).

    The prompt goes to stdin, the reply is the process's stdout. The process runs in a new empty folder,
    so it cannot read the study, its hypotheses or other participants' answers. Uses the CLI's own login.
    """
    argv = split_command(command)
    if not argv:
        raise RuntimeError("--cli needs a command, e.g. \"claude -p --model haiku\"")

    def llm(prompt, *args, **kw):
        if isinstance(prompt, list):
            text = "\n\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
        else:
            system = kw.get("system") or (args[0] if args and isinstance(args[0], str) else None)
            text = (system + "\n\n" + prompt) if system else prompt
        with tempfile.TemporaryDirectory(prefix="participant-") as empty:
            run = subprocess.run(argv, input=text, capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=timeout, cwd=empty)
        if run.returncode != 0:
            raise RuntimeError("participant command failed (%d): %s" % (run.returncode, run.stderr.strip()[-400:]))
        return run.stdout.strip()
    return llm


def read_sessions(path):
    raw = Path(path).read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="folder with program.json (default: found from this file)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("overview")
    sub.add_parser("capabilities")
    sub.add_parser("studies")
    sub.add_parser("paper")
    sub.add_parser("study").add_argument("id")
    sub.add_parser("node").add_argument("id")
    sub.add_parser("search").add_argument("text", nargs="+")
    sub.add_parser("issues").add_argument("study", nargs="?")
    p = sub.add_parser("prompt")
    p.add_argument("study")
    p.add_argument("condition")
    sub.add_parser("score").add_argument("sessions")
    for cmd in ["plan", "replay"]:
        q = sub.add_parser(cmd)
        if cmd == "replay":
            q.add_argument("replies", help="JSON list of reply strings, in plan order")
        q.add_argument("--n", type=int, default=5)
        q.add_argument("--arm", action="append")
        q.add_argument("--seed", type=int, default=20261015)
    r = sub.add_parser("run")
    who = r.add_mutually_exclusive_group(required=True)
    who.add_argument("--cli", help='agent CLI to start once per participant, e.g. "claude -p --model haiku"')
    who.add_argument("--model", help="model id for an OpenAI-compatible API (key from the environment)")
    r.add_argument("--n", type=int)
    r.add_argument("--arm", action="append")
    r.add_argument("--seed", type=int, default=20261015)
    r.add_argument("--out", help="write sessions and evaluation to this JSON file")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    s = Study(a.root)
    show = lambda v: print(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=2, default=str))
    if a.cmd == "overview":
        show(s.overview())
    elif a.cmd == "capabilities":
        show(s.capabilities())
    elif a.cmd == "studies":
        show([dict(st, conditions=s.conditions(st["id"])) for st in s.overview()["studies"]])
    elif a.cmd == "paper":
        show(s.paper_markdown())
    elif a.cmd == "study":
        show(s.study_markdown(a.id))
    elif a.cmd == "node":
        show(s.node(a.id))
    elif a.cmd == "search":
        show(s.search(" ".join(a.text)))
    elif a.cmd == "issues":
        show("\n".join(s.issue_line(i) for i in s.open_issues(a.study)) or "No open issues.")
    elif a.cmd == "prompt":
        show(s.participant_prompt(a.study, a.condition))
    elif a.cmd == "score":
        show(s.evaluate(read_sessions(a.sessions)))
    elif a.cmd == "plan":
        show(s.plan(a.n, a.arm, a.seed))
    elif a.cmd == "replay":
        show(s.replay(json.loads(Path(a.replies).read_text(encoding="utf-8")), a.n, a.arm, a.seed)["evaluation"])
    elif a.cmd == "run":
        result = s.run(cli_llm(a.cli) if a.cli else http_llm(a.model), a.seed, a.n, a.arm)
        if a.out:
            Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        show(result["evaluation"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refusal as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
        sys.exit(3)
    except (KeyError, RuntimeError, FileNotFoundError) as exc:
        print("error: %s" % (exc.args[0] if exc.args else exc), file=sys.stderr)
        sys.exit(1)
