#!/usr/bin/env python3
"""effects.json: the executable link from a paper's hypotheses to its data.

program.json says, in the paper's words, what the study is. effects.json turns its hypotheses into
effects that a program can compute: for each effect, the outcome, the contrast, the predicted
direction, the paper's reported numbers (each one quoted) and how to compute it from sessions.
Human and AI-participant sessions are computed the same way and give the same result block, so
effects from many papers can be pooled later without an agent re-reading any paper.

    python effects_core.py check <project>                       validate; exits 1 on a problem
    python effects_core.py graph <project>                       hypothesis -> effect -> program nodes
    python effects_core.py kg <project>                          the paper's whole typed graph (schema 2)
    python effects_core.py types                                 the core vocabulary (graph-types.json)
    python effects_core.py compute <project> <sessions.json>     result blocks for every effect

Raw quantities are kept: a result block holds the cells (n, mean, sd or counts) and the test, never only
a converted effect size. Converting between metrics (d, r, log odds ratio) is left to later tools.
Standard library only, Python 3.8+. Shared by the build skill, the agent export and HumanStudy-Bench.
"""
import json
import math
import re
import sys
from pathlib import Path

SCHEMA_VERSION = 2          # new files; schema 1 files stay valid (their "other" values included)
SCHEMA_VERSIONS = (1, 2)
VOCAB_VERSION = "2.0.0"     # core vocabulary below; bump the minor version when an extension is promoted
METRICS = (
    "mean", "mean_difference", "standardized_mean_difference", "hedges_g", "proportion", "risk_difference",
    "risk_ratio", "odds_ratio", "log_odds_ratio", "correlation", "phi", "cramers_v", "chi_square",
    "regression_coefficient", "standardized_coefficient", "t", "F", "z", "bayes_factor", "eta_squared",
    "partial_eta_squared", "count", "cohens_q", "auc", "ratio", "percent_change", "other")
DESIGNS = ("between", "within", "mixed", "correlational", "other")
OUTCOME_TYPES = ("continuous", "ordinal", "binary", "categorical", "count", "other")
CONTRASTS = ("two_groups", "paired", "association", "interaction", "omnibus", "single_group", "linear", "ratio",
             "model_comparison", "other")
# Schema 2 never accepts "other" for these: a new kind is declared as an extension (see EXTENSIONS below).
NO_OTHER = ("contrast", "metric")
DIRECTIONS = ("positive", "negative", "nonzero", "none", "unspecified")
OPS = ("mean_difference", "paired_difference", "proportion_difference", "contingency", "correlation", "package")
ROLES = ("test", "robustness", "check", "descriptive")
POPULATIONS = ("human", "ai-model", "human-vs-ai-model", "ai-model-vs-ai-model")
UNTESTED = ("not-an-effect", "not-tested-in-paper", "not-computable", "out-of-scope")
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
NUMBER_FIELDS = ("value", "se", "df", "p", "n", "lower", "upper")
TOLERANCE = "a result is recovered when |computed - reported| <= max(0.05, 0.15 * |reported|)"


def load(root):
    path = Path(root) / "effects.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---- numbers in quotes -----------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def numbers_in(text):
    """Every number written in a quote: unicode minus, thousands separators and leading-zero-less
    decimals (".58") are read as numbers; "6,554" is 6554."""
    t = str(text or "").replace("−", "-").replace("–", "-").replace(" ", " ")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    out = []
    for m in _NUM.finditer(t):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def quoted(value, quotes):
    """'exact' when the number is written in one of the quotes, 'unsigned' when only its absolute value
    is (the sign comes from the wording), else None."""
    nums = [x for q in quotes for x in numbers_in(q)]
    if any(abs(x - value) < 1e-9 for x in nums):
        return "exact"
    if any(abs(abs(x) - abs(value)) < 1e-9 for x in nums):
        return "unsigned"
    return None


# ---- validation -------------------------------------------------------------------------------------

def _numbers(block):
    """(where, value) for every number in a reported block."""
    out = []
    for i, e in enumerate(block.get("estimates") or []):
        for k in ("value", "se", "df", "p"):
            if isinstance(e.get(k), (int, float)) and not isinstance(e.get(k), bool):
                out.append(("estimates[%d].%s" % (i, k), e[k]))
        for j, x in enumerate(e.get("ci") or []):
            if isinstance(x, (int, float)):
                out.append(("estimates[%d].ci[%d]" % (i, j), x))
    test = block.get("test") or {}
    for k in ("value", "p"):
        if isinstance(test.get(k), (int, float)) and not isinstance(test.get(k), bool):
            out.append(("test.%s" % k, test[k]))
    df = test.get("df")
    for j, x in enumerate(df if isinstance(df, list) else [df] if df is not None else []):
        if isinstance(x, (int, float)):
            out.append(("test.df[%d]" % j, x))
    for arm, cell in (block.get("cells") or {}).items():
        for k, x in (cell or {}).items():
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                out.append(("cells.%s.%s" % (arm, k), x))
    if isinstance(block.get("n"), (int, float)):
        out.append(("n", block["n"]))
    return out


EXT_NAME = re.compile(r"^x:[a-z0-9][a-z0-9_.-]*$")
EXT_CATEGORIES = ("contrast", "metric", "edge", "node")


def core_vocab(category):
    return {"contrast": CONTRASTS, "metric": METRICS, "edge": tuple(EDGE_TYPES), "node": NODE_KINDS}[category]


def extensions(effects):
    """{name: declaration} for the extensions this file declares, with promoted ones mapped to core."""
    return {x.get("name"): x for x in (effects or {}).get("extensions") or [] if isinstance(x, dict)}


def resolve(category, value, effects=None):
    """The core term a value stands for: itself, the core term a promoted extension became, or the
    parent an extension declares. Tools that do not know an extension read it as its parent."""
    if value in core_vocab(category):
        return value
    if value in PROMOTED.get(category, {}):
        return PROMOTED[category][value]
    decl = extensions(effects).get(value)
    if decl and decl.get("category") == category:
        return decl.get("parent")
    return None


def check(root, effects=None, program=None):
    """(problems, warnings). A problem makes the file unusable; a warning is a gap to look at."""
    root = Path(root)
    effects = effects if effects is not None else load(root)
    if effects is None:
        return ["effects.json is missing"], []
    if program is None:
        program = json.loads((root / "program.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in program.get("nodes", []) if "id" in n}
    evidence = {e["id"]: e for e in program.get("evidence", []) if "id" in e}
    studies = {s["id"] for s in program.get("studies", [])}
    problems, warnings = [], []
    P, W = problems.append, warnings.append

    if effects.get("schemaVersion") not in SCHEMA_VERSIONS:
        P("schemaVersion must be one of %s" % ", ".join(map(str, SCHEMA_VERSIONS)))
    v2 = effects.get("schemaVersion") == 2
    exts = extensions(effects)
    if not v2 and (effects.get("extensions") or effects.get("links")):
        P("extensions and links need schemaVersion 2")
    for name, x in exts.items():
        where = "extension %s" % name
        if not EXT_NAME.match(str(name or "")):
            P("%s: an extension name is x: followed by lower-case letters, digits, _ . or -" % where)
        elif x.get("category") not in EXT_CATEGORIES:
            P("%s: category must be one of %s" % (where, ", ".join(EXT_CATEGORIES)))
        elif x.get("parent") not in core_vocab(x["category"]) or x.get("parent") == "other":
            P("%s: parent must be a core %s (what tools that do not know it read it as), not other"
              % (where, x["category"]))
        if not x.get("definition"):
            P("%s: definition is required (what it is, so others can reuse it or merge it)" % where)
        if x.get("category") == "edge" and not (x.get("from") and x.get("to")):
            P("%s: an edge extension names the node kinds it connects (from, to)" % where)

    def term(where, category, value):
        """A vocabulary value: core, a promoted extension, or one this file declares."""
        if value == "other" and v2 and category in NO_OTHER:
            P("%s: 'other' is not accepted in schema 2; use a core %s or declare an extension (x:...)"
              % (where, category))
        elif resolve(category, value, effects) is None:
            P("%s: %r is not a core %s or a declared extension" % (where, value, category))
    if effects.get("programId") != program.get("id"):
        P("programId %r does not match program.json id %r" % (effects.get("programId"), program.get("id")))
    if not isinstance(effects.get("topics"), list):
        P("topics must be a list of keywords (may be empty)")

    def node_ref(where, nid, kinds=None):
        if nid not in nodes:
            P("%s: %r is not a program node" % (where, nid))
        elif kinds and nodes[nid].get("kind") not in kinds:
            P("%s: %r is a %s node, expected %s" % (where, nid, nodes[nid].get("kind"), " or ".join(kinds)))

    hyps = effects.get("hypotheses")
    if not isinstance(hyps, list):
        P("hypotheses must be a list")
        hyps = []
    hyp_ids, covered = set(), set()
    for i, h in enumerate(hyps):
        where = "hypotheses[%d]" % i
        if not ID.match(str(h.get("id", ""))) or h["id"] in hyp_ids:
            P("%s: id missing, malformed or repeated" % where)
            continue
        hyp_ids.add(h["id"])
        where = "hypothesis %s" % h["id"]
        node_ref(where + " nodeId", h.get("nodeId"), ["hypothesis"])
        covered.add(h.get("nodeId"))
        for b in h.get("backgroundIds") or []:
            node_ref(where + " backgroundIds", b)
        if not isinstance(h.get("phenomena"), list):
            P("%s: phenomena must be a list of keywords (may be empty)" % where)
        if h.get("status") not in ("tested", "untested"):
            P("%s: status must be tested or untested" % where)
        if h.get("status") == "untested" and (h.get("untestedBecause") not in UNTESTED or not h.get("reason")):
            P("%s: an untested hypothesis needs untestedBecause (%s) and a reason" % (where, ", ".join(UNTESTED)))
    for nid, n in nodes.items():
        if n.get("kind") == "hypothesis" and nid not in covered:
            P("program hypothesis %s (%s) is not in effects.json; list it as tested or untested" % (nid, n.get("title")))

    effs = effects.get("effects")
    if not isinstance(effs, list):
        P("effects must be a list")
        effs = []
    eff_ids, tested = set(), set()
    for i, e in enumerate(effs):
        if not ID.match(str(e.get("id", ""))) or e["id"] in eff_ids:
            P("effects[%d]: id missing, malformed or repeated" % i)
            continue
        eff_ids.add(e["id"])
        where = "effect %s" % e["id"]
        if not e.get("label"):
            P("%s: label is required" % where)
        if e.get("role") not in ROLES:
            P("%s: role must be one of %s" % (where, ", ".join(ROLES)))
        hs = e.get("hypothesisIds") or []
        if not hs and e.get("role") in ("test", "robustness"):
            P("%s: a %s effect must name at least one hypothesis in hypothesisIds" % (where, e.get("role")))
        for h in hs:
            if h not in hyp_ids:
                P("%s: hypothesis %r is not in hypotheses" % (where, h))
            tested.add(h)
        if e.get("studyId") not in studies:
            P("%s: studyId %r is not a program study" % (where, e.get("studyId")))
        pop = e.get("population") or {"kind": "human"}
        if pop.get("kind") not in POPULATIONS:
            P("%s: population.kind must be one of %s" % (where, ", ".join(POPULATIONS)))
        if pop.get("kind") != "human" and not pop.get("models"):
            P("%s: a population with AI models names them in population.models" % where)
        if "phenomena" in e and not isinstance(e["phenomena"], list):
            P("%s: phenomena must be a list (it replaces the hypothesis's phenomena for this effect)" % where)
        if e.get("design") not in DESIGNS:
            P("%s: design must be one of %s" % (where, ", ".join(DESIGNS)))
        out = e.get("outcome") or {}
        if out.get("type") not in OUTCOME_TYPES:
            P("%s: outcome.type must be one of %s" % (where, ", ".join(OUTCOME_TYPES)))
        if out.get("variableId"):
            node_ref(where + " outcome.variableId", out["variableId"], ["variable", "record", "measure"])
        else:
            W("%s: outcome has no variableId, so the measure is not linked into the graph" % where)
        if not out.get("higherMeans"):
            W("%s: outcome.higherMeans is empty; say what a higher value means" % where)
        con = e.get("contrast") or {}
        term(where + " contrast.kind", "contrast", con.get("kind"))
        ckind = resolve("contrast", con.get("kind"), effects)
        if ckind == "linear" and not (isinstance(con.get("weights"), dict) and con["weights"] and all(
                _num(w) is not None for w in con["weights"].values())):
            P("%s: a linear contrast needs weights {cell: number} (value = sum of weight x cell, plus constant)" % where)
        if ckind in ("ratio", "model_comparison") and not (con.get("treatment") and con.get("reference")):
            P("%s: a %s contrast needs treatment and reference (value = treatment / reference)" % (where, ckind))
        for c in con.get("conditionIds") or []:
            node_ref(where + " contrast.conditionIds", c, ["condition"])
        for c in con.get("factorIds") or []:  # the manipulated or predictor variables
            node_ref(where + " contrast.factorIds", c, ["variable", "record", "measure"])
        if ckind in ("two_groups", "paired") and not (con.get("treatment") and con.get("reference")):
            P("%s: a %s contrast needs treatment and reference (the value is treatment minus reference)"
              % (where, ckind))
        pred = e.get("prediction") or {}
        if pred.get("direction") not in DIRECTIONS:
            P("%s: prediction.direction must be one of %s" % (where, ", ".join(DIRECTIONS)))
        for key, kinds in (("analysisIds", ["analysis"]), ("resultIds", ["result", "finding"])):
            if not isinstance(e.get(key, []), list):
                P("%s: %s must be a list" % (where, key))
            elif e.get(key):
                for x in e[key]:
                    node_ref("%s %s" % (where, key), x, kinds)
            else:
                W("%s: no %s" % (where, key))
        comp = e.get("compute") or {}
        if comp.get("op") not in OPS:
            P("%s: compute.op must be one of %s" % (where, ", ".join(OPS)))
        elif comp["op"] == "package" and not comp.get("key"):
            P("%s: compute.op package needs key (the effect's key in the package evaluation)" % where)
        elif comp["op"] != "package" and not comp.get("field"):
            P("%s: compute.op %s needs field (path to the value in a session, e.g. responses.favorable)"
              % (where, comp["op"]))
        if comp.get("metric"):
            term(where + " compute.metric", "metric", comp["metric"])
        if comp.get("compareAs"):
            term(where + " compute.compareAs.metric", "metric", comp["compareAs"].get("metric"))
        if comp.get("compareAs") and not comp["compareAs"].get("reason"):
            P("%s: compute.compareAs needs a reason (why the computed and reported numbers are the same quantity)"
              % where)

        rep = e.get("reported")
        if rep is None:
            W("%s: the paper reports no number for it (reported is null)" % where)
            continue
        quotes = []
        for eid in rep.get("evidenceIds") or []:
            if eid not in evidence:
                P("%s: reported evidence %r is not in program evidence" % (where, eid))
            else:
                quotes.append(evidence[eid].get("quote") or "")
        if not rep.get("evidenceIds"):
            P("%s: reported numbers need evidenceIds" % where)
        for j, est in enumerate(rep.get("estimates") or []):
            term("%s reported.estimates[%d].metric" % (where, j), "metric", est.get("metric"))
        if rep.get("test") and rep["test"].get("statistic") not in ("t", "F", "z", "chi_square", "W", "U", "other"):
            P("%s: reported.test.statistic must be t, F, z, chi_square, W, U or other" % where)
        if not isinstance(rep.get("missing", []), list):
            P("%s: reported.missing must be a list of what the paper does not report" % where)
        for loc, value in _numbers(rep):
            q = quoted(value, quotes)
            if q is None:
                P("%s: reported %s = %s is not written in its cited quotes" % (where, loc, value))
            elif q == "unsigned":
                W("%s: reported %s = %s: only its absolute value is quoted; check the sign" % (where, loc, value))
    for h in hyps:
        if h.get("status") == "tested" and h.get("id") not in tested:
            P("hypothesis %s is tested but no effect names it" % h.get("id"))
        if h.get("status") == "untested" and h.get("id") in tested:
            P("hypothesis %s is untested but effect(s) name it" % h.get("id"))
    if not effs and not any(h.get("status") == "untested" for h in hyps):
        W("no effects")
    for i, link in enumerate(effects.get("links") or []):
        for msg in link_problems(link, effects, program):
            P("links[%d]: %s" % (i, msg))
    reach = graph(effects, program)
    for nid in reach["unlinked"]:
        W("program %s %s is not reachable from any hypothesis or effect" % (nodes[nid].get("kind"), nid))
    return problems, warnings


# ---- the graph --------------------------------------------------------------------------------------

LINKED_KINDS = ("hypothesis", "condition", "variable", "analysis", "result")


def graph(effects, program):
    """Edges from background through hypotheses and effects to the program nodes they bind, extended by
    the program's own relations and step conditions. `unlinked` lists hypothesis/condition/variable/
    analysis/result nodes that no hypothesis reaches."""
    nodes = {n["id"]: n for n in program.get("nodes", []) if "id" in n}
    edges = []
    add = lambda a, b, t: edges.append({"from": a, "to": b, "type": t})
    for t in effects.get("topics") or []:
        add("paper", "topic:" + str(t), "about")
    for h in effects.get("hypotheses") or []:
        hid = "hypothesis:" + str(h.get("id"))
        add(hid, h.get("nodeId"), "states")
        for b in h.get("backgroundIds") or []:
            add(b, hid, "motivates")
        for ph in h.get("phenomena") or []:
            add(hid, "phenomenon:" + str(ph), "instance-of")
    for e in effects.get("effects") or []:
        eid = "effect:" + str(e.get("id"))
        for h in e.get("hypothesisIds") or []:
            add("hypothesis:" + h, eid, "tested-by")
        if (e.get("outcome") or {}).get("variableId"):
            add(eid, e["outcome"]["variableId"], "measures")
        for c in (e.get("contrast") or {}).get("conditionIds") or []:
            add(eid, c, "contrasts")
        for c in (e.get("contrast") or {}).get("factorIds") or []:
            add(eid, c, "varies")
        for key, t in (("analysisIds", "analysed-by"), ("resultIds", "reported-in")):
            for x in e.get(key) or []:
                add(eid, x, t)
    adj = {}
    for x in edges:
        adj.setdefault(x["from"], set()).add(x["to"])
    for r in program.get("relations", []):  # program relations count both ways
        adj.setdefault(r.get("from"), set()).add(r.get("to"))
        adj.setdefault(r.get("to"), set()).add(r.get("from"))
    seen, todo = set(), ["hypothesis:" + str(h.get("id")) for h in effects.get("hypotheses") or []]
    todo += ["effect:" + str(e.get("id")) for e in effects.get("effects") or [] if not e.get("hypothesisIds")]
    while todo:
        x = todo.pop()
        if x in seen:
            continue
        seen.add(x)
        todo.extend(adj.get(x, ()))
    unlinked = sorted(nid for nid, n in nodes.items() if n.get("kind") in LINKED_KINDS and nid not in seen)
    return {"edges": edges, "reached": sorted(x for x in seen if x in nodes), "unlinked": unlinked}


# ---- the knowledge graph (schema 2) ------------------------------------------------------------------
#
# One typed graph per paper: the program's nodes and procedure steps, the effects (estimands), and the
# constructs they are about, joined by edges from a small core vocabulary. Program relations written with
# older names are read through RELATION_ALIASES; anything that fits no core term is reported as a gap,
# never forced into one. The vocabulary grows by extensions (x:name, declared in effects.json with a core
# parent and a definition); an extension used by several papers is promoted into the core and listed in
# PROMOTED, so files that used the old name read as the core term without being rewritten.

VARLIKE = ["variable", "record", "measure"]
NODE_KINDS = ("paper", "construct", "background", "overview", "hypothesis", "design", "participants", "condition",
              "material", "procedure", "step", "variable", "record", "measure", "analysis", "model", "result", "finding",
              "limitation", "simulation", "effect")
EDGE_TYPES = {
    # claims
    "about": (["paper"], ["construct"], "the paper is about this construct or topic"),
    "motivates": (["background"], ["hypothesis"], "background that motivates the hypothesis"),
    "instance-of": (["hypothesis", "effect"], ["construct"], "tests a case of this phenomenon or construct"),
    "predicts": (["hypothesis"], ["construct"], "the hypothesis predicts a change in this construct"),
    "tested-by": (["hypothesis"], ["effect", "analysis", "finding"], "evidence that tests the hypothesis"),
    "supports": (["result", "finding"], ["hypothesis"], "the paper reads this result as support"),
    "contradicts": (["result", "finding"], ["hypothesis"], "the paper reads this result as counter-evidence"),
    # the estimand
    "measures": (["effect"], VARLIKE, "the effect's outcome variable"),
    "contrasts": (["effect"], ["condition"], "a condition the effect compares"),
    "varies": (["effect"], VARLIKE, "a predictor the effect varies"),
    "estimated-by": (["effect"], ["analysis", "model"], "the analysis that estimates the effect"),
    "reported-in": (["effect"], ["result", "finding"], "where the paper reports the effect"),
    "moderates": (VARLIKE + ["condition"], ["effect"], "the effect differs across its levels"),
    "mediates": (VARLIKE, ["effect"], "the effect runs through it"),
    "replicates": (["effect"], ["effect"], "the same effect estimated again, in another sample or paper"),
    "compares": (["effect"], ["effect"], "the effect is defined as a difference between these effects"),
    # operationalisation and analysis
    "operationalizes": (VARLIKE, ["construct"], "the variable measures the construct"),
    "manipulates": (["condition"], ["construct"], "the condition manipulates the construct"),
    "input-of": (VARLIKE + ["condition"], ["analysis", "model"], "the analysis uses it"),
    "reports": (["analysis", "model"], ["result", "finding"], "the analysis produced this reported result"),
    "concerns": (["result", "finding"], VARLIKE + ["condition", "hypothesis"], "the reported result is about it"),
    "derived-from": (VARLIKE + ["analysis", "model"], VARLIKE + ["analysis", "model"], "computed from it"),
    # procedure (from program steps)
    "shows": (["step", "condition"], ["*"], "what a participant sees at this step or in this condition"),
    "produces": (["step"], ["*"], "what the step records"),
    "uses": (["step"], ["*"], "an input of the step"),
    "in-condition": (["step"], ["condition"], "the step belongs to this condition"),
    "part-of": (["step"], ["step"], "a sub-step"),
    "next": (["step"], ["step"], "the step that follows"),
    "performed-by": (["step"], ["*"], "who acts at this step"),
}
PROCEDURE_EDGES = ("shows", "produces", "uses", "in-condition", "part-of", "next", "performed-by")
CONTRAST_DEFS = {
    "two_groups": "treatment minus reference, two independent groups",
    "paired": "treatment minus reference, measured on the same units",
    "single_group": "one group's value, or its difference from a fixed constant",
    "association": "a correlation or slope between two variables",
    "interaction": "a difference between two contrasts (difference of differences, or across samples)",
    "omnibus": "an undirected test over more than two levels; not poolable as a signed effect",
    "linear": "sum of weight x cell over named cells, plus an optional constant (e.g. 1 vs mean of 2 others)",
    "ratio": "treatment divided by reference",
    "model_comparison": "evidence for one model of the data over another (e.g. a Bayes factor)",
    "other": "schema 1 only",
}
# Older program relation names -> core edge candidates (tried forwards, then reversed, against the kinds).
RELATION_ALIASES = {"uses": ["input-of"], "outcome": ["input-of", "concerns"], "estimates": ["input-of"],
                    "sees": ["shows"], "analysed-by": ["estimated-by"], "reported-by": ["reports"],
                    "reports": ["concerns"], "produces": ["reports"]}
# Promoted extensions: {category: {"x:old-name": "core-name"}}. Empty in 2.0.0.
PROMOTED = {"contrast": {}, "metric": {}, "edge": {}, "node": {}}


def _kind_ok(kind, allowed):
    return "*" in allowed or kind in allowed


def _norm(t):
    return re.sub(r"[\s_]+", "-", str(t or "").strip().lower())


def map_relation(rtype, from_kind, to_kind):
    """(core edge, reversed) for a program relation, or None when no core edge fits the two kinds."""
    t = _norm(rtype)
    for cand in ([t] if t in EDGE_TYPES else []) + RELATION_ALIASES.get(t, []):
        dom, rng = EDGE_TYPES[cand][0], EDGE_TYPES[cand][1]
        if _kind_ok(from_kind, dom) and _kind_ok(to_kind, rng):
            return cand, False
        if _kind_ok(to_kind, dom) and _kind_ok(from_kind, rng):
            return cand, True
    return None


def construct_id(label):
    """A construct keeps the paper's own wording: no case folding, no synonym merging. Merging labels across
    papers is a later, reviewed step (a sameAs map), never done here."""
    return "construct:" + str(label).strip()


def ref_kind(ref, effects, program):
    """The node kind a reference names, or None when it names nothing."""
    ref = str(ref or "")
    if ref.startswith("paper:"):
        inner = ref.split("/", 1)[1] if "/" in ref else ""
        return "effect" if inner.startswith("effect:") else ("external" if inner else None)
    if ref.startswith("construct:"):
        return "construct" if len(ref) > len("construct:") else None
    if ref.startswith("effect:"):
        return "effect" if any(e.get("id") == ref[7:] for e in effects.get("effects") or []) else None
    for n in program.get("nodes", []):
        if n.get("id") == ref:
            return n.get("kind")
    if any(st.get("id") == ref for st in program.get("steps", [])):
        return "step"
    return None


def link_problems(link, effects, program):
    out = []
    t = link.get("type")
    core = resolve("edge", t, effects)
    if core is None:
        return ["type %r is not a core edge or a declared edge extension" % t]
    decl = extensions(effects).get(t) if t not in EDGE_TYPES else None
    dom, rng = (decl.get("from"), decl.get("to")) if decl else EDGE_TYPES[core][:2]
    fk, tk = ref_kind(link.get("from"), effects, program), ref_kind(link.get("to"), effects, program)
    if fk is None:
        out.append("from %r names no node (program node or step id, effect:<id>, construct:<label>, "
                   "or paper:<programId>/effect:<id>)" % link.get("from"))
    if tk is None:
        out.append("to %r names no node" % link.get("to"))
    if fk and fk != "external" and not _kind_ok(fk, dom if isinstance(dom, list) else [dom]):
        out.append("%s goes from %s, not from a %s node" % (t, "/".join(dom if isinstance(dom, list) else [dom]), fk))
    if tk and tk != "external" and not _kind_ok(tk, rng if isinstance(rng, list) else [rng]):
        out.append("%s goes to %s, not to a %s node" % (t, "/".join(rng if isinstance(rng, list) else [rng]), tk))
    if link.get("origin") not in ("verbatim", "reported", "derived"):
        out.append("origin must be verbatim, reported or derived (how the paper supports the edge)")
    if link.get("origin") in ("verbatim", "reported") and not link.get("evidenceIds"):
        out.append("a %s edge needs evidenceIds" % link.get("origin"))
    ev = {e.get("id") for e in program.get("evidence", [])}
    out += ["evidence %r is not in program evidence" % x for x in link.get("evidenceIds") or [] if x not in ev]
    return out


def _label(n):
    return n.get("title") or n.get("label") or n.get("id")


def kg(effects, program):
    """The paper's typed graph: {schemaVersion, vocabulary, programId, nodes, edges, gaps, unlinked}.
    Every edge has a core type (extensions are kept with their core parent in `core`)."""
    effects = effects or {}
    nodes, edges, gaps = {}, [], []
    pid = program.get("id")

    def node(nid, kind, label, **extra):
        if nid not in nodes:
            nodes[nid] = dict({"id": nid, "kind": kind, "label": label}, **extra)

    def edge(a, b, t, source, **extra):
        e = {"from": a, "to": b, "type": t, "source": source}
        core = resolve("edge", t, effects)
        if core != t:
            e["core"] = core
        e.update({k: v for k, v in extra.items() if v})
        edges.append(e)

    node("paper", "paper", program.get("title") or pid)
    kinds = {}
    for n in program.get("nodes", []):
        kind = n.get("kind")
        kinds[n["id"]] = kind
        if kind not in NODE_KINDS:
            gaps.append({"what": "node kind", "value": kind, "id": n["id"]})
        node(n["id"], kind, _label(n), studyIds=n.get("studyIds") or None)
    for st in program.get("steps", []):
        kinds[st["id"]] = "step"
        node(st["id"], "step", st.get("title") or st["id"], studyId=st.get("studyId"))
    for st in program.get("steps", []):
        for key, t in (("visibleIds", "shows"), ("outputIds", "produces"), ("inputIds", "uses"),
                       ("conditionIds", "in-condition"), ("actorIds", "performed-by")):
            for x in st.get(key) or []:
                if x in kinds:
                    edge(st["id"], x, t, "program.steps")
        for x in st.get("children") or []:
            if x in kinds:
                edge(x, st["id"], "part-of", "program.steps")
        nxt = st.get("next")
        for x in (nxt if isinstance(nxt, list) else [nxt] if nxt else []):
            if isinstance(x, dict):
                x = x.get("to") or x.get("id")
            if x in kinds:
                edge(st["id"], x, "next", "program.steps")
    for r in program.get("relations", []):
        a, b = r.get("from"), r.get("to")
        m = map_relation(r.get("type"), kinds.get(a), kinds.get(b))
        if m is None or a not in kinds or b not in kinds:
            gaps.append({"what": "relation", "value": r.get("type"), "id": r.get("id"),
                         "fromKind": kinds.get(a), "toKind": kinds.get(b)})
            continue
        t, rev = m
        edge(b if rev else a, a if rev else b, t, "program.relations", was=r.get("type") if _norm(r.get("type")) != t else None)

    for t in effects.get("topics") or []:
        node(construct_id(t), "construct", str(t))
        edge("paper", construct_id(t), "about", "effects.topics")
    hyp_node = {}
    for h in effects.get("hypotheses") or []:
        hn = h.get("nodeId")
        hyp_node[h.get("id")] = hn
        if hn in nodes:
            nodes[hn].update({"status": h.get("status"), "untestedBecause": h.get("untestedBecause")})
        for b in h.get("backgroundIds") or []:
            if b in nodes:
                edge(b, hn, "motivates", "effects.hypotheses")
        for ph in h.get("phenomena") or []:
            node(construct_id(ph), "construct", str(ph))
            edge(hn, construct_id(ph), "instance-of", "effects.hypotheses")
    for e in effects.get("effects") or []:
        eid = "effect:" + str(e.get("id"))
        con, comp = e.get("contrast") or {}, e.get("compute") or {}
        node(eid, "effect", e.get("label") or e.get("id"), role=e.get("role"),
             contrast=con.get("kind"), contrastCore=resolve("contrast", con.get("kind"), effects),
             population=(e.get("population") or {}).get("kind", "human"), op=comp.get("op"))
        for h in e.get("hypothesisIds") or []:
            if hyp_node.get(h) in nodes:
                edge(hyp_node[h], eid, "tested-by", "effects")
        if "phenomena" in e:
            for ph in e.get("phenomena") or []:
                node(construct_id(ph), "construct", str(ph))
                edge(eid, construct_id(ph), "instance-of", "effects")
        if (e.get("outcome") or {}).get("variableId") in nodes:
            edge(eid, e["outcome"]["variableId"], "measures", "effects")
        for c in con.get("conditionIds") or []:
            if c in nodes:
                edge(eid, c, "contrasts", "effects")
        for c in con.get("factorIds") or []:
            if c in nodes:
                edge(eid, c, "varies", "effects")
        for key, t in (("analysisIds", "estimated-by"), ("resultIds", "reported-in")):
            for x in e.get(key) or []:
                if x in nodes:
                    edge(eid, x, t, "effects")
    for link in effects.get("links") or []:
        for end in (link.get("from"), link.get("to")):
            if str(end).startswith("construct:"):
                node(end, "construct", end[len("construct:"):])
            elif str(end).startswith("paper:"):
                node(end, "external", end)
        edge(link.get("from"), link.get("to"), link.get("type"), "effects.links",
             origin=link.get("origin"), evidenceIds=link.get("evidenceIds"))
    for name, x in extensions(effects).items():
        gaps.append({"what": "extension", "value": name, "category": x.get("category"), "parent": x.get("parent"),
                     "definition": x.get("definition")})
    # reached: from the hypotheses, along every non-procedure edge in either direction
    adj = {}
    for e in edges:
        if (e.get("core") or e["type"]) in PROCEDURE_EDGES:
            continue
        adj.setdefault(e["from"], set()).add(e["to"])
        adj.setdefault(e["to"], set()).add(e["from"])
    seen, todo = set(), [n for n in hyp_node.values() if n]
    todo += ["effect:" + str(e.get("id")) for e in effects.get("effects") or [] if not e.get("hypothesisIds")]
    while todo:
        x = todo.pop()
        if x not in seen:
            seen.add(x)
            todo.extend(adj.get(x, ()))
    unlinked = sorted(n["id"] for n in nodes.values() if n["kind"] in LINKED_KINDS and n["id"] not in seen)
    return {"schemaVersion": 2, "vocabulary": VOCAB_VERSION, "programId": pid, "nodes": list(nodes.values()),
            "edges": edges, "gaps": gaps, "unlinked": unlinked}


def types_registry():
    """The core vocabulary as data (pack/schema/graph-types.json is this, generated)."""
    return {
        "vocabulary": VOCAB_VERSION, "effectsSchemaVersions": list(SCHEMA_VERSIONS),
        "nodeKinds": list(NODE_KINDS),
        "edgeTypes": {t: {"from": d[0], "to": d[1], "definition": d[2], "procedure": t in PROCEDURE_EDGES}
                      for t, d in EDGE_TYPES.items()},
        "relationAliases": RELATION_ALIASES,
        "contrasts": {c: CONTRAST_DEFS[c] for c in CONTRASTS},
        "metrics": list(METRICS), "noOtherInSchema2": list(NO_OTHER),
        "promoted": PROMOTED,
        "extensions": {
            "name": "x: followed by lower-case letters, digits, _ . or -",
            "categories": list(EXT_CATEGORIES),
            "declare": "effects.json extensions[]: {name, category, parent (a core term), definition, "
                       "from/to node kinds for an edge}",
            "promote": "when several papers declare the same need, add it to the core, bump the vocabulary minor "
                       "version, and map each old x: name to it in PROMOTED; files are never rewritten",
            "merge": "two extensions that mean the same thing are both mapped in PROMOTED to one core term",
        },
    }


# ---- statistics (standard library) -------------------------------------------------------------------

def _betacf(a, b, x):
    qab, qap, qam, c, d = a + b, a + 1.0, a - 1.0, 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def p_t(t, df):
    """Two-sided p of Student's t."""
    if t is None or df is None or df <= 0 or t != t:
        return None
    return _ibeta(df / 2.0, 0.5, df / (df + t * t))


def p_chi2(x, df):
    """Upper tail of chi-square (regularized upper incomplete gamma)."""
    if x is None or df <= 0 or x != x:
        return None
    a, z = df / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    if z < a + 1:
        term = total = 1.0 / a
        for n in range(1, 1000):
            term *= z / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    b, c, d = z + 1 - a, 1e300, 1.0 / (z + 1 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c if abs(b + an / c) > 1e-300 else 1e-300
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * h


def _mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n if n else None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": m, "sd": sd}


# ---- computing effects from sessions --------------------------------------------------------------

def field(session, path):
    """A dotted path into a session: 'responses.favorable', 'parsed.choice', 'trials.0.answer' (a number
    indexes a list)."""
    cur = session
    for part in str(path).split("."):
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _rows(sessions, effect):
    comp = effect.get("compute") or {}
    study = comp.get("study", effect.get("studyId"))
    where = comp.get("where") or {}
    out = []
    for s in sessions:
        if study is not None and s.get("study") is not None and str(s.get("study")) != str(study):
            continue
        if any(field(s, k) != v for k, v in where.items()):
            continue
        out.append(s)
    return out


def _num(x):
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)) and x == x:
        return float(x)
    return None


def _groups(rows, effect):
    comp, con = effect["compute"], effect.get("contrast") or {}
    by = comp.get("groupBy", "condition")
    arms = [con.get("treatment"), con.get("reference")]
    return {a: [r for r in rows if str(field(r, by)) == str(a)] for a in arms}


def op_mean_difference(rows, effect):
    g = _groups(rows, effect)
    t_arm, r_arm = effect["contrast"]["treatment"], effect["contrast"]["reference"]
    vals = {a: [v for v in (_num(field(r, effect["compute"]["field"])) for r in rs) if v is not None]
            for a, rs in g.items()}
    cells = {a: _mean_sd(v) for a, v in vals.items()}
    a, b = cells[t_arm], cells[r_arm]
    if a["n"] < 2 or b["n"] < 2:
        return {"status": "not_ready", "reason": "needs at least 2 values in each arm", "cells": cells}
    va, vb = a["sd"] ** 2 / a["n"], b["sd"] ** 2 / b["n"]
    se = math.sqrt(va + vb)
    diff = a["mean"] - b["mean"]
    df = (va + vb) ** 2 / ((va ** 2 / (a["n"] - 1)) + (vb ** 2 / (b["n"] - 1))) if va + vb > 0 else None
    t = diff / se if se > 0 else None
    return {"status": "computed", "metric": "mean_difference", "value": diff, "se": se, "n": a["n"] + b["n"],
            "cells": cells, "test": {"statistic": "t", "value": t, "df": df, "p": p_t(t, df)},
            "method": "Welch t-test, treatment minus reference"}


def op_paired_difference(rows, effect):
    comp = effect["compute"]
    t_f, r_f = comp["field"], comp.get("referenceField")
    if not r_f:
        return {"status": "not_ready", "reason": "paired_difference needs compute.referenceField"}
    pairs = [(_num(field(r, t_f)), _num(field(r, r_f))) for r in rows]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    diffs = [x - y for x, y in pairs]
    d = _mean_sd(diffs)
    if d["n"] < 2:
        return {"status": "not_ready", "reason": "needs at least 2 complete pairs", "cells": {}}
    se = d["sd"] / math.sqrt(d["n"]) if d["sd"] is not None else None
    t = d["mean"] / se if se else None
    con = effect["contrast"]
    cells = {con["treatment"]: _mean_sd([x for x, _ in pairs]), con["reference"]: _mean_sd([y for _, y in pairs]),
             "difference": d}
    return {"status": "computed", "metric": "mean_difference", "value": d["mean"], "se": se, "n": d["n"],
            "cells": cells, "test": {"statistic": "t", "value": t, "df": d["n"] - 1, "p": p_t(t, d["n"] - 1)},
            "method": "paired t-test, treatment minus reference within participant"}


def _success(value, effect):
    s = effect["compute"].get("success", [True, 1, "yes"])
    s = s if isinstance(s, list) else [s]
    return value in s or str(value) in [str(x) for x in s]


def op_proportion_difference(rows, effect):
    g = _groups(rows, effect)
    t_arm, r_arm = effect["contrast"]["treatment"], effect["contrast"]["reference"]
    cells = {}
    for a, rs in g.items():
        vs = [field(r, effect["compute"]["field"]) for r in rs]
        vs = [v for v in vs if v is not None]
        k = sum(1 for v in vs if _success(v, effect))
        cells[a] = {"n": len(vs), "k": k, "proportion": k / len(vs) if vs else None}
    a, b = cells[t_arm], cells[r_arm]
    if not a["n"] or not b["n"]:
        return {"status": "not_ready", "reason": "an arm has no values", "cells": cells}
    diff = a["proportion"] - b["proportion"]
    se = math.sqrt(a["proportion"] * (1 - a["proportion"]) / a["n"] + b["proportion"] * (1 - b["proportion"]) / b["n"])
    table = [[a["k"], a["n"] - a["k"]], [b["k"], b["n"] - b["k"]]]
    chi = _chi_square(table)
    return {"status": "computed", "metric": "risk_difference", "value": diff, "se": se, "n": a["n"] + b["n"],
            "cells": cells, "table": {"rows": [t_arm, r_arm], "columns": ["success", "other"], "counts": table},
            "test": chi, "method": "difference in proportions, treatment minus reference; Pearson chi-square"}


def _chi_square(table):
    rows = [sum(r) for r in table]
    cols = [sum(c) for c in zip(*table)]
    n = sum(rows)
    if not n or any(x == 0 for x in rows + cols):
        return {"statistic": "chi_square", "value": None, "df": None, "p": None}
    x2 = sum((table[i][j] - rows[i] * cols[j] / n) ** 2 / (rows[i] * cols[j] / n)
             for i in range(len(rows)) for j in range(len(cols)))
    df = (len(rows) - 1) * (len(cols) - 1)
    return {"statistic": "chi_square", "value": x2, "df": df, "p": p_chi2(x2, df), "n": n}


def op_contingency(rows, effect):
    comp = effect["compute"]
    by = comp.get("groupBy", "condition")
    arms = comp.get("groups") or (effect.get("contrast") or {}).get("groups") or \
        sorted({str(field(r, by)) for r in rows if field(r, by) is not None})
    levels = comp.get("levels") or sorted({str(field(r, comp["field"])) for r in rows
                                          if field(r, comp["field"]) is not None})
    counts = [[sum(1 for r in rows if str(field(r, by)) == str(a) and str(field(r, comp["field"])) == str(lv))
               for lv in levels] for a in arms]
    chi = _chi_square(counts)
    if chi["value"] is None:
        return {"status": "not_ready", "reason": "a row or column of the table is empty",
                "table": {"rows": arms, "columns": levels, "counts": counts}}
    n, k = chi["n"], min(len(arms), len(levels))
    v = math.sqrt(chi["value"] / (n * (k - 1))) if k > 1 else None
    return {"status": "computed", "metric": "phi" if len(arms) == 2 and len(levels) == 2 else "cramers_v",
            "value": v, "n": n, "table": {"rows": arms, "columns": levels, "counts": counts}, "test": chi,
            "method": "Pearson chi-square without continuity correction; phi / Cramer's V = sqrt(chi2 / (N (k-1)))"}


def op_correlation(rows, effect):
    comp = effect["compute"]
    pairs = [(_num(field(r, comp["field"])), _num(field(r, comp.get("withField", "")))) for r in rows]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return {"status": "not_ready", "reason": "needs at least 3 complete pairs"}
    mx, my = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sx, sy = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs)), math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if not sx or not sy:
        return {"status": "not_ready", "reason": "a variable has no variance"}
    r = sxy / (sx * sy)
    t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else None
    return {"status": "computed", "metric": "correlation", "value": r, "n": n,
            "test": {"statistic": "t", "value": t, "df": n - 2, "p": p_t(t, n - 2)}, "method": "Pearson r"}


OP_FUNCS = {"mean_difference": op_mean_difference, "paired_difference": op_paired_difference,
            "proportion_difference": op_proportion_difference, "contingency": op_contingency,
            "correlation": op_correlation}


def package_block(effect, evaluation, module=None):
    """An effect computed by the package's own analysis. The evaluation module may define
    effect_estimates(evaluation) -> {key: result block}; otherwise the key_effects() entry with the
    same id gives a bare estimate."""
    key = effect["compute"]["key"]
    blocks = None
    if module is not None and hasattr(module, "effect_estimates"):
        blocks = module.effect_estimates(evaluation) or {}
        if key in blocks:
            return dict(blocks[key])
    return None  # the package has no block for this key


def key_of(effect):
    return (effect.get("compute") or {}).get("key")


def legacy_block(effect, key_effects):
    key = effect["compute"]["key"]
    for k in key_effects or []:
        if k.get("id") == key:
            est = _num(k.get("estimate"))
            if est is None:
                return {"status": "not_ready", "reason": "the package analysis could not compute it"}
            metric = (effect.get("compute") or {}).get("metric") or "other"
            return {"status": "computed", "metric": metric, "value": est,
                    "test": {"statistic": "other", "value": None, "df": None, "p": _num(k.get("p"))},
                    "method": "package key_effects (no cells)"}
    return {"status": "not_ready", "reason": "the package analysis has no key %r" % key}


def compute(effects, sessions, evaluation=None, module=None, key_effects=None, source=None):
    """One result block per effect: {effectId, source, status, metric, value, se, n, cells, table, test, method}."""
    out = []
    for e in effects.get("effects") or []:
        comp = e.get("compute") or {}
        op = comp.get("op")
        try:
            if op == "package":
                block = package_block(e, evaluation, module)
                if block is None:  # no effect_estimates entry: an older package's key_effects()
                    block = legacy_block(e, key_effects) if key_effects is not None else \
                        {"status": "not_ready", "reason": "the package analysis returned nothing for key %r" % key_of(e)}
            elif op in OP_FUNCS:
                block = OP_FUNCS[op](_rows(sessions, e), e)
            else:
                block = {"status": "not_ready", "reason": "unknown compute op %r" % op}
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            block = {"status": "not_ready", "reason": "%s: %s" % (type(exc).__name__, exc)}
        block = dict(block, effectId=e["id"])
        if source:
            block["source"] = source
        out.append(block)
    return out


def compare(effect, block):
    """Computed against reported, in the same quantity. Basis: the same metric; or the effect's
    compute.compareAs (a stated equivalence, e.g. an OLS coefficient on a condition dummy is the mean
    difference); or the difference of the reported cell means. Otherwise not comparable."""
    rep = effect.get("reported") or {}
    if block.get("status") != "computed" or block.get("value") is None:
        return {"comparable": False, "reason": block.get("reason") or "not computed"}
    metric = block.get("metric")
    for est in rep.get("estimates") or []:
        if est.get("metric") == metric and _num(est.get("value")) is not None:
            return _cmp(block["value"], est["value"], "same metric (%s)" % metric)
    alias = (effect.get("compute") or {}).get("compareAs")
    if alias:
        for est in rep.get("estimates") or []:
            if est.get("metric") == alias.get("metric") and _num(est.get("value")) is not None:
                return _cmp(block["value"], est["value"], "computed %s compared as reported %s: %s"
                            % (metric, alias["metric"], alias.get("reason")))
    con, cells = effect.get("contrast") or {}, rep.get("cells") or {}
    t, r = cells.get(con.get("treatment")) or {}, cells.get(con.get("reference")) or {}
    if metric == "mean_difference" and _num(t.get("mean")) is not None and _num(r.get("mean")) is not None:
        return _cmp(block["value"], t["mean"] - r["mean"], "difference of the reported cell means")
    pt, pr = _share(t), _share(r)
    if metric == "risk_difference" and pt is not None and pr is not None:
        return _cmp(block["value"], pt - pr, "difference of the reported proportions")
    return {"comparable": False, "reason": "the paper reports no %s for this effect" % metric}


def _share(cell):
    """A reported proportion, or a reported percentage / 100 (papers print either)."""
    if _num(cell.get("proportion")) is not None:
        return cell["proportion"]
    if _num(cell.get("percent")) is not None:
        return cell["percent"] / 100.0
    return None


def _cmp(computed, reported, basis):
    ok = abs(computed - reported) <= max(0.05, 0.15 * abs(reported))
    return {"comparable": True, "computed": computed, "reported": reported, "basis": basis, "withinTolerance": ok}


def phenomena_of(effect, hypotheses):
    """The effect's own phenomena when it lists them, else those of its hypotheses."""
    if isinstance(effect.get("phenomena"), list):
        return sorted(set(effect["phenomena"]))
    by_id = {h.get("id"): h for h in hypotheses or []}
    return sorted({p for h in effect.get("hypothesisIds") or [] for p in (by_id.get(h) or {}).get("phenomena") or []})


def _rounded(x):
    if isinstance(x, float):
        return float("%.6g" % x) if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v) for v in x]
    return x


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["types"]:
        print(json.dumps(types_registry(), ensure_ascii=False, indent=2))
        return 0
    if len(argv) < 2 or argv[0] not in ("check", "graph", "kg", "compute"):
        print(__doc__)
        return 2
    root = Path(argv[1])
    effects = load(root)
    if effects is None:
        print("effects.json is missing in %s" % root)
        return 1
    program = json.loads((root / "program.json").read_text(encoding="utf-8"))
    if argv[0] == "check":
        problems, warnings = check(root, effects, program)
        for w in warnings:
            print("warning: " + w)
        for p in problems:
            print("problem: " + p)
        n = len(effects.get("effects") or [])
        print("FAIL" if problems else "PASS (%d hypotheses, %d effects)" % (len(effects.get("hypotheses") or []), n))
        return 1 if problems else 0
    if argv[0] == "graph":
        print(json.dumps(graph(effects, program), ensure_ascii=False, indent=2))
        return 0
    if argv[0] == "kg":
        print(json.dumps(kg(effects, program), ensure_ascii=False, indent=2))
        return 0
    if len(argv) < 3:
        print("compute needs a sessions JSON file (a list, or {\"sessions\": [...]})")
        return 2
    data = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    sessions = data.get("sessions", []) if isinstance(data, dict) else data
    blocks = compute(effects, sessions)
    print(json.dumps(_rounded(blocks), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
