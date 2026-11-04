"""Package checks for what a replay shows and what it compares against.

Called by ``validate_package.py`` after the smoke test. Contract:
``agent_pipeline/CLAUDE.md``, sections "Participant memory within a session",
"Human results to compare against" and "Researcher reply".

1. Memory and leaks. The adapter is run with a stand-in model that records every
   prompt (one participant per condition, no network, no paid calls). A prompt
   after a participant's first must carry something this participant already saw
   (unless ``task/task.json`` ``memory`` declares a narrower design with a
   reason); a prompt that says earlier screens are "above" must contain them; and
   no prompt may contain a condition id or an internal screen id.
2. Human results. Every printed result table must be recorded cell by cell in the
   program's result nodes, and most reported test statistics, effect sizes and
   confidence limits must be recorded. Numbers are matched against tables and
   statistics found in the paper's PDF text; a paper without such tables or
   statistics passes.
3. Analyses. Each recorded human result names the ``evaluate()`` output that
   computes the paper's own test, or says why it is not ready. A missing
   scientific library is never a reason: the build and replay runtimes install
   numpy, scipy, pandas and statsmodels. The reply must not claim an analysis
   family the evaluation does not implement.
4. Researcher reply. ``studio-reply.md`` opens with three plain lines (ready to
   run / needs you / licence), asks at most three questions, gives a run-size
   estimate, reads the licence from the PDF, and avoids internal vocabulary and
   overclaims.

Run as ``python package_checks.py --probe <paper-folder>`` to print the
stand-in run as JSON (used by the validator and ``estimate_run.py``).
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import json
import math
import re
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterable

MAX_ARMS = 40
MAX_CALLS_PER_ARM = 400
PROBE_TIMEOUT = 120
# Libraries the build and replay runtimes install (run-humanstudy-pipeline.yml,
# run-playground.yml). "Not installed" is never a valid not-ready reason for them.
RUNTIME_LIBRARIES = ("numpy", "scipy", "statsmodels", "pandas")


# --------------------------------------------------------------------------
# Small helpers


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _job_purpose(job: Path) -> str | None:
    request = _load_json(job / "studio_request.json")
    return request.get("purpose") if isinstance(request, dict) else None


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _norm(text: str) -> str:
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-").replace("−", "-").replace(" ", " "))
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Stand-in run (executed in a subprocess so a broken adapter cannot hang or crash
# the validator)

STUB_NAME_RE = re.compile(r"(?i)(smoke|stub|stand_?in|fake|dummy|offline)")
STUB_ROLE_RE = re.compile(r"(?i)(llm|model|agent|participant|reply|answer)")


def _find_stub(module: Any):
    """The package's own offline stand-in model, if it has one at module level."""
    candidates = []
    for name, value in vars(module).items():
        if not callable(value) or inspect.isclass(value) or not STUB_NAME_RE.search(name) \
                or not STUB_ROLE_RE.search(name):
            continue
        try:
            params = [p for p in inspect.signature(value).parameters.values()
                      if p.default is inspect.Parameter.empty
                      and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            continue
        if len(params) == 1:
            candidates.append((0 if "llm" in name.lower() or "model" in name.lower() else 1, name, value))
    candidates.sort(key=lambda item: item[:2])
    return (candidates[0][1], candidates[0][2]) if candidates else (None, None)


def _import(path: Path, name: str) -> Any:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def condition_ids(task: Any) -> list[str]:
    if not isinstance(task, dict):
        return []
    ids: list[str] = []
    for key in ("conditions", "arms"):
        for item in task.get(key) or []:
            value = item.get("id") if isinstance(item, dict) else item
            if isinstance(value, str) and value and value not in ids:
                ids.append(value)
    return ids


class _Limit(BaseException):
    """Raised by the stand-in model at the call cap; a BaseException so an adapter's
    ``except Exception`` retry loop cannot swallow it."""


def _jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 30:
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)  # numpy scalars
    except (TypeError, ValueError):
        return str(value)


def probe(root: Path) -> dict:
    """Run every condition once with a recording stand-in model."""
    result: dict[str, Any] = {"stub": None, "runs": [], "errors": [], "evaluate": None}
    task = _load_json(root / "task/task.json")
    try:
        with redirect_stdout(io.StringIO()):
            adapter = _import(root / "task/adapter.py", "humanstudy_probe_adapter")
    except BaseException as exc:  # noqa: BLE001 - report, never crash
        result["errors"].append(f"task/adapter.py could not be imported ({type(exc).__name__}: {exc})")
        return result
    run_sessions = getattr(adapter, "run_sessions", None)
    if not callable(run_sessions):
        return result
    stub_name, stub = _find_stub(adapter)
    result["stub"] = stub_name
    try:
        accepts_arms = "arms" in inspect.signature(run_sessions).parameters
    except (TypeError, ValueError):
        accepts_arms = False
    arms = condition_ids(task)[:MAX_ARMS] if accepts_arms else []
    all_sessions: list = []

    # The pipeline's answer-aware stand-in (task/replies.py StubParticipant, read through
    # replies.ask's answer_for hook) gets past comprehension checks to the later screens.
    participant_class = getattr(sys.modules.get("replies"), "StubParticipant", None)
    if participant_class is not None:
        result["stub"] = "replies.StubParticipant"

    def one_run(arm: str | None, n: int) -> dict:
        calls: list[dict] = []
        participant = participant_class(seed=0) if participant_class is not None else None

        def record(prompt: Any, answer) -> str:
            if len(calls) >= MAX_CALLS_PER_ARM:
                raise _Limit()
            try:
                reply = answer()
            except _Limit:
                raise
            except Exception:  # noqa: BLE001 - a weak stub must not hide the prompts
                reply = "1"
            reply = reply if isinstance(reply, str) else str(reply)
            calls.append({"prompt": prompt if isinstance(prompt, str) else str(prompt), "reply": reply})
            return reply

        class Recorder:
            def __call__(self, prompt: str, key: Any = None) -> str:
                if participant is not None:
                    return record(prompt, lambda: participant(prompt))
                return record(prompt, (lambda: stub(prompt)) if stub is not None else (lambda: "1"))

        if participant is not None:
            def answer_for(prompt: str, questions: Any, **kwargs: Any) -> str:
                return record(prompt, lambda: participant.answer_for(prompt, questions, **kwargs))
            Recorder.answer_for = staticmethod(answer_for)  # type: ignore[attr-defined]
        llm = Recorder()

        run = {"arm": arm, "n": n, "calls": calls, "sessions": 0, "error": None, "truncated": False}
        try:
            with redirect_stdout(io.StringIO()):
                sessions = run_sessions(llm, 0, n, arms=[arm]) if arm is not None else run_sessions(llm, 0, n)
            if isinstance(sessions, list):
                run["sessions"] = len(sessions)
                all_sessions.extend(sessions)
        except _Limit:
            run["truncated"] = True
        except BaseException as exc:  # noqa: BLE001
            run["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return run

    if arms:
        for arm in arms:
            run = one_run(arm, 1)
            for n in (2, 4):
                # Pairs need an even n; random assignment may skip a condition at n = 1.
                if run["calls"] and not run["error"]:
                    break
                retry = one_run(arm, n)
                run = retry if retry["calls"] or not retry["error"] or run["error"] else run
            result["runs"].append(run)
    else:
        result["runs"].append(one_run(None, 1))

    evaluation_path = root / "evaluation/evaluation.py"
    if evaluation_path.is_file():
        try:
            with redirect_stdout(io.StringIO()):
                evaluation = _import(evaluation_path, "humanstudy_probe_evaluation")
                output = evaluation.evaluate(all_sessions)
            result["evaluate"] = {"ok": True, "output": _jsonable(output)}
        except BaseException as exc:  # noqa: BLE001
            result["evaluate"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return result


def run_probe(root: Path, timeout: int = PROBE_TIMEOUT) -> dict:
    """The stand-in run. The result goes to a file, so nothing the adapter prints
    can replace it; a run that cannot finish is reported as an error."""
    import tempfile

    def failed(message: str) -> dict:
        return {"stub": None, "runs": [], "errors": [message], "evaluate": None}

    with tempfile.TemporaryDirectory(prefix="humanstudy-probe-") as folder:
        out = Path(folder) / "probe.json"
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--probe", str(root), str(out)],
                cwd=root, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return failed(f"the stand-in run did not finish within {timeout} s")
        if completed.returncode != 0 or not out.is_file():
            return failed("the stand-in run failed: " + (completed.stderr or "")[-500:])
        try:
            return json.loads(out.read_text(encoding="utf-8"))
        except ValueError:
            return failed("the stand-in run wrote no readable result")


# --------------------------------------------------------------------------
# 1. Memory and leaks

CLAIM_RE = re.compile(
    r"(?i)\b(screens?|pages?|questions?|answers?|parts?)\s+(shown\s+|you\s+(have\s+)?(saw|seen|answered)\s+)?"
    r"(above|earlier|before|so far)\b|\b(previous|earlier|preceding)\s+(screens?|pages?|answers?|questions?|parts?)\b"
    r"|\bthe\s+(last|previous|first)\s+part\b")
ID_KEYS_RE = re.compile(r"(?i)(^id$|_id$|Id$|^qid$|^qsfid$|^key$|^ref$|^tag$|^qsftag$|pointer|^arm$|^arms$|"
                        r"^condition$|^conditions$|^cell$|^study$|^studies$|^source|path$|^evidence)")


def _chunks(prompt: str) -> set[str]:
    """Sentence-sized pieces of a prompt, long enough to identify a screen."""
    pieces = set()
    for line in prompt.splitlines():
        line = _norm(re.sub(r"^\s*([-*>•]|\d+[.)])\s+", "", line))
        for piece in re.split(r"(?<=[.!?:])\s+", line):
            piece = piece.strip(" -*\"'")
            # Answer options repeat from item to item and say nothing about memory.
            if len(piece) >= 30 and len(re.findall(r"[A-Za-z]{2,}", piece)) >= 6 and '", "' not in piece:
                pieces.add(piece)
    return pieces


def _visible_text(materials: Any) -> str:
    """Participant-facing strings of materials.json (identifier fields left out)."""
    out: list[str] = []

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if not ID_KEYS_RE.search(str(k)):
                    walk(v, str(k))
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str):
            out.append(value)

    walk(materials)
    return _norm("\n".join(out)).casefold()


def _identifier_like(token: str) -> bool:
    if len(token) < 4 or not re.fullmatch(r"[A-Za-z0-9_.\-]+", token):
        return False
    if not re.search(r"[A-Za-z]", token):
        return False
    return bool(re.search(r"[_.]", token) or re.search(r"[a-z][A-Z]", token)
                or (re.search(r"\d", token) and re.search(r"[A-Za-z]{2,}", token))
                or re.fullmatch(r"[a-z]+(-[a-z0-9]+)+", token))


def _material_ids(materials: Any) -> set[str]:
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, str) and re.search(r"(?i)(^id$|qid|qsfid|screenid|^qsftag$|^tag$)", str(k)):
                    found.add(v)
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(materials)
    return found


def _memory_design(task: Any) -> tuple[str, int]:
    memory = task.get("memory") if isinstance(task, dict) else None
    scope = "session"
    agents = 1
    if isinstance(memory, dict):
        if isinstance(memory.get("scope"), str):
            scope = memory["scope"].strip().lower()
        if isinstance(memory.get("agents_per_session"), int) and memory["agents_per_session"] >= 1:
            agents = memory["agents_per_session"]
    return scope, agents


MEMORY_CALLS = 80  # per condition; memory faults show in the first screens


def _visible_chunks(chunks: set[str], visible: str) -> set[str]:
    """Pieces of a prompt that are participant-facing screen text (materials.json),
    not the adapter's own instructions or reply-format lines."""
    return {chunk for chunk in chunks if chunk.casefold() in visible}


def check_memory(root: Path, result: dict | None) -> list[str]:
    if not result or not result.get("runs"):
        return []
    task = _load_json(root / "task/task.json")
    materials = _load_json(root / "materials/materials.json")
    scope, agents = _memory_design(task)
    problems: list[str] = []
    memory = task.get("memory") if isinstance(task, dict) else None
    if scope != "session" and not (isinstance(memory, dict) and isinstance(memory.get("reason"), str)
                                   and memory["reason"].strip()):
        problems.append("task/task.json memory.scope is not 'session' but gives no reason from the paper's design")

    runs = [run for run in result["runs"] if run.get("calls")]
    if not runs:
        return problems
    visible = _visible_text(materials)

    # Screen text every condition opens with (a shared introduction) does not show memory:
    # a memoryless adapter may repeat it in every prompt.
    firsts = [_chunks(run["calls"][0]["prompt"]) for run in runs]
    shared = set.intersection(*firsts) if len(firsts) >= 2 else set()

    examples: list[str] = []
    claim_examples: list[str] = []
    dropped_examples: list[str] = []
    for run in runs:
        calls = run["calls"][:MEMORY_CALLS]
        all_chunks = [_chunks(call["prompt"]) for call in calls]
        screen_chunks = [_visible_chunks(chunks, visible) - shared for chunks in all_chunks]
        if not any(screen_chunks):  # materials not in a readable form: use everything
            screen_chunks = [chunks - shared for chunks in all_chunks]
        seen: set[str] = set()
        fresh_after_first = 0
        participants = max(1, int(run.get("sessions") or 1)) * agents
        previous_new: set[str] = set()
        for index, call in enumerate(calls):
            screen = screen_chunks[index]
            # Substring match: a history line may wrap a screen ("Screen 1: ... (you answered 4)").
            text = _norm(call["prompt"])
            carried = any(chunk in text for chunk in seen)
            # One participant alone: the screen just answered must still be there.
            if participants == 1 and carried and previous_new \
                    and 2 * sum(chunk in text for chunk in previous_new) < len(previous_new) \
                    and len(dropped_examples) < 3:
                dropped_examples.append(f"{_label(run)} prompt {index + 1}")
            new = screen - seen
            if new:
                previous_new = new
            if index > 0 and seen and not carried:
                fresh_after_first += 1
                if fresh_after_first >= participants and len(examples) < 3:
                    examples.append(f"{_label(run)} prompt {index + 1}")
                if _claims_earlier(call["prompt"], visible) and len(claim_examples) < 3:
                    claim_examples.append(f"{_label(run)} prompt {index + 1}")
            seen |= screen
            seen |= _chunks(call["reply"])
        run["_fresh"] = fresh_after_first
    if scope == "session":
        over = [run for run in runs if run["_fresh"] >= max(1, int(run.get("sessions") or 1)) * agents]
        if over:
            problems.append(
                "participant memory: prompts after a participant's first screen carry none of what that participant "
                "already saw or answered (" + ", ".join(examples) + "). Each prompt must include this participant's "
                "own earlier screens and answers in this session, in order (task/task.json memory; contract "
                "'Participant memory within a session'). If one session has several agents, set "
                "task/task.json memory.agents_per_session.")
        if dropped_examples:
            problems.append(
                "participant memory: prompts drop the screen the participant has just answered ("
                + ", ".join(dropped_examples) + "). By default a participant keeps everything seen and answered "
                "so far in the session, in order; a narrower memory needs task/task.json memory with the reason.")
    if claim_examples:
        problems.append(
            "participant memory: prompts say earlier screens or answers are shown but contain none of them ("
            + ", ".join(claim_examples) + "). Send them or remove the claim.")

    # Leaks: condition ids and internal screen ids in what the participant reads.
    candidates: set[str] = set()
    for arm in condition_ids(task):
        candidates.add(arm)
        candidates |= {part for part in re.split(r"[-_.:/\s]+", arm) if part}
    candidates |= _material_ids(materials)
    candidates = {c for c in candidates if _identifier_like(c) and c.casefold() not in visible}
    leaked: dict[str, str] = {}
    if candidates:
        pattern = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(c) for c in
                             sorted(candidates, key=len, reverse=True)) + r")(?![A-Za-z0-9_])", re.IGNORECASE)
        lookup = {c.casefold(): c for c in candidates}
        for run in runs:
            # The stand-in's own replies are not the adapter's text.
            replies = sorted({call["reply"] for call in run["calls"] if len(call["reply"]) >= 3},
                             key=len, reverse=True)[:200]
            strip = re.compile("|".join(re.escape(r) for r in replies)) if replies else None
            for index, call in enumerate(run["calls"]):
                prompt = strip.sub(" ", call["prompt"]) if strip else call["prompt"]
                for match in pattern.finditer(prompt):
                    identifier = lookup.get(match.group(1).casefold(), match.group(1))
                    leaked.setdefault(identifier, f"{_label(run)} prompt {index + 1}")
    if leaked:
        # The parts of condition ids that tell conditions apart (not "study1" or "s2").
        parts = {c.casefold() for arm in condition_ids(task) for c in re.split(r"[-_.:/\s]+", arm)
                 if len(c) >= 3 and not re.fullmatch(r"(?i)(study|s|exp|experiment|part|cond|condition)?\d+", c)}

        def names_condition(identifier: str) -> bool:
            return any(re.search(r"(?<![a-z0-9])" + re.escape(part) + r"(?![a-z0-9])", identifier.casefold())
                       for part in parts)

        shown = sorted(leaked.items(), key=lambda item: (not names_condition(item[0]), item[0]))[:8]
        revealing = [k for k in leaked if names_condition(k)]
        problems.append(
            "participant prompts contain internal identifiers the participant never saw"
            + (f" ({len(revealing)} of them name the participant's condition)" if revealing else "") + ": "
            + ", ".join(f"'{k}' ({v})" for k, v in shown)
            + (" and more" if len(leaked) > 8 else "")
            + ". Never show condition ids or screen/question ids; key answers by neutral labels "
            "(q1, q2, or the number printed on the screen).")
    return problems


def _claims_earlier(prompt: str, visible: str) -> bool:
    """The adapter (not the survey's own wording) says earlier screens are shown."""
    for line in prompt.splitlines():
        if CLAIM_RE.search(line) and _norm(line).strip(" -*").casefold() not in visible:
            return True
    return False


def _label(run: dict) -> str:
    return f"condition {run['arm']}" if run.get("arm") else "the run"


# --------------------------------------------------------------------------
# 2. Human results: paper tables and statistics vs recorded result nodes

DECIMAL_RE = re.compile(r"(?<![\w.])-?(?:\d+\.\d+|\.\d+)(?![\w])")
CAPTION_RE = re.compile(r"(?:^|\s{2,})(?:Table|TABLE)\s+(\d+)[.:]?\s+([A-Z][^\n]*)")
RESULT_CAPTION_RE = re.compile(
    r"(?i)(result|mean|descriptive|statistic|effect|correlat|regression|model|analys|frequenc|proportion|"
    r"percent|anova|test|estimate|coefficient|rating|sample|outcome|choice|score|response)")
NON_RESULT_CAPTION_RE = re.compile(
    r"(?i)(power|sensitivity|taxonomy|methodolog|pre-?registration|procedure|material|stimul|wording|"
    r"instruction|hypothes[ie]s(?! and a comparison)|design\b|previous|prior\b|literature|meta-analy)")
# A statistic just after these words is a power analysis or someone else's result.
NOT_OWN_RESULT_RE = re.compile(r"(?i)power|sensitivity|g\*power|meta-analy|previous (stud|research|work)|"
                               r"prior (stud|research|work)|original (study|authors) (found|reported)")
NUMERIC_CELL_RE = re.compile(r"^[\[(]?-?(?:\d+(?:\.\d+)?|\.\d+)%?[\])]?[,;*†‡]*$")
STAT_RES = [
    ("t", re.compile(r"\bt\s*\(\s*\d+(?:\.\d+)?\s*\)\s*=\s*(-?\d*\.?\d+)")),
    ("F", re.compile(r"\bF\s*\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)\s*=\s*(\d*\.?\d+)")),
    ("chi2", re.compile(r"(?:χ|χ2|χ²|chi-square|chi2|Chi2|X2)\s*(?:2|²)?\s*\([^)=]{0,30}\)\s*=\s*(\d*\.?\d+)")),
    ("z", re.compile(r"\bz\s*=\s*(-?\d*\.\d+)")),
    ("effect", re.compile(r"(?:\b(?:d|g|r|OR|RR|b|B|β|φ|phi|f|w|V)|η\s*p?\s*2|η²|ηp²|R2|R²)\s*=\s*(-?\d*\.\d+)")),
    ("ci", re.compile(r"CI\s*[=:]?\s*[\[(]\s*(-?\d*\.?\d+)\s*[,;]\s*(-?\d*\.?\d+)\s*[\])]")),
]


_PAPER_TEXT: dict[tuple[str, bool], str] = {}


def paper_text(job: Path, layout: bool) -> str:
    key = (str(job.resolve()), layout)
    if key not in _PAPER_TEXT:
        _PAPER_TEXT[key] = _read_paper(job, layout)
    return _PAPER_TEXT[key]


def _read_paper(job: Path, layout: bool) -> str:
    pdf = job / "input/paper.pdf"
    if not pdf.is_file() or not shutil.which("pdftotext"):
        return ""
    command = ["pdftotext", "-q"] + (["-layout"] if layout else []) + [str(pdf), "-"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def _strip_references(text: str) -> str:
    match = None
    for match in re.finditer(r"(?im)^\s*(references|bibliography|literature cited)\s*$", text):
        pass
    return text[:match.start()] if match and match.start() > len(text) * 0.4 else text


def _table_row(line: str) -> bool:
    cells = [cell.strip() for cell in re.split(r"\s{2,}", line.strip()) if cell.strip()]
    numeric = [cell for cell in cells if NUMERIC_CELL_RE.match(cell.replace("–", "-").replace("−", "-"))]
    return len(numeric) >= 3 and any("." in cell for cell in numeric)


def paper_tables(layout_text: str) -> list[dict]:
    """Result tables in the PDF text: caption, number and the decimal values of their rows."""
    text = _strip_references(layout_text).replace("–", "-").replace("−", "-")
    lines = text.splitlines()
    captions = []
    for index, line in enumerate(lines):
        match = CAPTION_RE.search(line)
        if match:
            captions.append((index, match.group(1), _norm(match.group(2))[:120]))
    if not captions:
        return []
    numbers: dict[str, set[str]] = {}
    for index, line in enumerate(lines):
        if not _table_row(line):
            continue
        nearest = min(captions, key=lambda caption: abs(caption[0] - index))
        if abs(nearest[0] - index) > 45:
            continue
        values = {value for value in DECIMAL_RE.findall(line)}
        numbers.setdefault(nearest[1], set()).update(values)
    tables = []
    seen: set[str] = set()
    for _, number, title in captions:
        if number in seen:
            continue
        seen.add(number)
        if NON_RESULT_CAPTION_RE.search(title) and not re.search(r"(?i)result|effect|mean", title):
            continue
        if not RESULT_CAPTION_RE.search(title):
            continue
        values = numbers.get(number, set())
        if len(values) >= 8:
            tables.append({"number": number, "title": title, "values": sorted(values)})
    return tables


def paper_statistics(plain_text: str) -> list[str]:
    text = _norm(_strip_references(plain_text))
    found: list[str] = []
    for _, pattern in STAT_RES:
        for match in pattern.finditer(text):
            if NOT_OWN_RESULT_RE.search(text[max(0, match.start() - 160):match.start()]):
                continue
            for value in match.groups():
                if value and "." in value and value not in found:
                    found.append(value)
    return found


def _decimals(value: str) -> int:
    return len(value.split(".", 1)[1]) if "." in value else 0


def recorded_numbers(program: Any, tables_only: bool = False) -> set[str]:
    """Every number in the program's result nodes, at 0-4 decimals, as absolute values.
    With tables_only, only numbers inside structured (object or list) values count,
    so a table must be recorded as a table, not as a pasted paragraph."""
    keys: set[str] = set()
    nodes = program.get("nodes") if isinstance(program, dict) else None
    for node in nodes or []:
        if not isinstance(node, dict) or node.get("kind") != "result":
            continue
        for field in node.get("fields") or []:
            if not isinstance(field, dict):
                continue
            # Not extensions.evaluation: that link is outside what the repair guard compares.
            extensions = {k: v for k, v in (field.get("extensions") or {}).items() if k != "evaluation"}
            values: list[Any] = [field.get("value"), extensions]
            if tables_only:
                values = [field.get("value")] if isinstance(field.get("value"), (dict, list)) else []
            for value in values:
                stack = [value]
                while stack:
                    item = stack.pop()
                    if isinstance(item, dict):
                        stack.extend(item.values())
                    elif isinstance(item, list):
                        stack.extend(item)
                    elif isinstance(item, bool) or item is None:
                        continue
                    elif isinstance(item, (int, float)):
                        _add_number(keys, float(item))
                    elif isinstance(item, str):
                        for token in re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+)", _norm(item)):
                            _add_number(keys, float(token))
    return keys


def _add_number(keys: set[str], number: float) -> None:
    if not math.isfinite(number):
        return
    for places in range(0, 5):
        keys.add(f"{abs(number):.{places}f}")


def _is_recorded(value: str, keys: set[str]) -> bool:
    return f"{abs(float(value)):.{_decimals(value)}f}" in keys


def check_results(root: Path, job: Path) -> list[str]:
    if _job_purpose(job) == "accepted-model-sync":
        return []
    study = _load_json(root / "study.json")
    program = study.get("program") if isinstance(study, dict) else None
    if not isinstance(program, dict):
        return []
    keys = recorded_numbers(program)
    table_keys = recorded_numbers(program, tables_only=True)
    problems: list[str] = []
    missing_tables = []
    for table in paper_tables(paper_text(job, layout=True)):
        absent = [value for value in table["values"] if not _is_recorded(value, table_keys)]
        if len(absent) > len(table["values"]) / 2:
            missing_tables.append(f"Table {table['number']} ({len(table['values']) - len(absent)} of "
                                  f"{len(table['values'])} values recorded; missing e.g. "
                                  + ", ".join(absent[:5]) + ")")
    if missing_tables:
        problems.append(
            "human results: printed result tables are not recorded cell by cell in study.json.program result "
            "nodes: " + "; ".join(missing_tables) + ". Record each table as one field whose value is the table "
            "(rows and columns of numbers; contract 'Human results to compare against').")
    statistics = paper_statistics(paper_text(job, layout=False))
    if len(statistics) >= 10:
        absent = [value for value in statistics if not _is_recorded(value, keys)]
        if len(absent) > len(statistics) / 2:
            problems.append(
                f"human results: only {len(statistics) - len(absent)} of {len(statistics)} test statistics, "
                "effect sizes and confidence limits printed in the paper are recorded in study.json.program "
                "result nodes (missing e.g. " + ", ".join(absent[:8]) + "). Record every reported test, "
                "effect size, CI and N of this paper's studies.")
    return problems


# --------------------------------------------------------------------------
# 3. Analyses: each reported result is computed or explicitly not ready

LIBRARY_EXCUSE_RE = re.compile(
    r"(?i)\b(numpy|scipy|statsmodels|pandas)\b[^.;]{0,40}\b(not\s+(installed|available)|unavailable|no module|"
    r"import ?error|cannot be imported|missing from)\b|\b(needs?|requires?|without|no|lacks?)\s+(the\s+)?"
    r"(numpy|scipy|statsmodels|pandas)\b(?!\s*(\.|'s)?\s*(\w*_\w*|\w+\(|(?-i:[A-Z]\w*[A-Z]\w*)|\w+\.\w+))")
META_FIELD_RE = re.compile(r"(?i)^(result_kind|kind|source|evidence|note|notes|description|label|study|scope)$")


def _reported_fields(program: dict) -> list[tuple[str, dict]]:
    out = []
    for node in program.get("nodes") or []:
        if not isinstance(node, dict) or node.get("kind") != "result":
            continue
        fields = [f for f in node.get("fields") or [] if isinstance(f, dict)]
        kind = next((f.get("value") for f in fields if f.get("id") == "result_kind"), None)
        if isinstance(kind, str) and kind.strip().lower() not in ("reported", "original", "paper"):
            continue
        for field in fields:
            if META_FIELD_RE.match(str(field.get("id", ""))):
                continue
            # An unresolved result (not reported, needs the researcher) has nothing to compute.
            if field.get("origin") in ("missing", "unknown") or field.get("state") in ("missing", "decision"):
                continue
            if not re.search(r"\d", json.dumps(field.get("value"), ensure_ascii=False)):
                continue
            out.append((f"{node.get('id')}.{field.get('id')}", field))
    return out


def _lookup(output: Any, path: str) -> tuple[bool, Any]:
    current = output
    for part in [p for p in re.split(r"\.|/", path) if p]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def _not_ready(value: Any) -> bool:
    return isinstance(value, dict) and (bool(value.get("not_ready")) or value.get("status") == "not_ready")


def check_analyses(root: Path, job: Path, result: dict | None) -> list[str]:
    problems: list[str] = []
    study = _load_json(root / "study.json")
    program = study.get("program") if isinstance(study, dict) else None
    evaluate = (result or {}).get("evaluate")
    if isinstance(evaluate, dict) and evaluate.get("ok") is False:
        problems.append(
            "evaluate(sessions) raised " + str(evaluate.get("error")) + " on a one-participant-per-condition "
            "stand-in run; it must return a structured not_ready entry for statistics the data cannot support.")
    output = evaluate.get("output") if isinstance(evaluate, dict) and evaluate.get("ok") else None

    # A missing library is never the reason: the runtimes install them.
    excuses = set()
    for text in _strings(output):
        if LIBRARY_EXCUSE_RE.search(text):
            excuses.add(_norm(text)[:120])
    if isinstance(program, dict):
        for _, field in _reported_fields(program):
            reason = ((field.get("extensions") or {}).get("evaluation") or {})
            if isinstance(reason, dict) and isinstance(reason.get("not_ready"), str) \
                    and LIBRARY_EXCUSE_RE.search(reason["not_ready"]):
                excuses.add(_norm(reason["not_ready"])[:120])
    if excuses:
        problems.append(
            "analyses: a not-ready reason blames a missing library (" + "; ".join(sorted(excuses)[:3]) + "). "
            "The build and replay runtimes have " + ", ".join(RUNTIME_LIBRARIES) + ": implement the paper's test.")

    if isinstance(program, dict) and _job_purpose(job) != "accepted-model-sync":
        unlinked, broken = [], []
        for name, field in _reported_fields(program):
            link = (field.get("extensions") or {}).get("evaluation")
            if not isinstance(link, dict) or not (isinstance(link.get("output"), str) and link["output"].strip()
                                                  or isinstance(link.get("not_ready"), str) and link["not_ready"].strip()):
                unlinked.append(name)
            elif isinstance(link.get("output"), str) and output is not None:
                if not _lookup(output, link["output"])[0]:
                    broken.append(f"{name} -> {link['output']}")
        if unlinked:
            problems.append(
                "analyses: recorded human results do not say which evaluate() output computes them: "
                + ", ".join(unlinked[:10]) + (" and more" if len(unlinked) > 10 else "")
                + ". Give each result field extensions.evaluation = {\"output\": \"<key path in evaluate()>\"} "
                "computing the paper's own test, or {\"not_ready\": \"<reason>\"}.")
        if broken:
            problems.append(
                "analyses: these evaluate() outputs named by recorded results do not exist in evaluate()'s "
                "return value: " + ", ".join(broken[:10]) + ". evaluate() must always return every key, with a "
                "not_ready entry when the sessions cannot support the test.")

    problems += _t_test_variant(root, job)
    problems += _reply_analysis_claims(root, output)
    return problems


T_DF_RE = re.compile(r"\bt\s*\(\s*(\d+(?:\.\d+)?)\s*\)\s*=")


def _t_variants(path: Path) -> set[str]:
    """Which two-sample t-test variants a module computes: 'welch' and/or 'pooled'."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            if "welch" in name:
                found.add("welch")
            if re.search(r"student|pooled_?t|t_?pooled|paired|ttest_rel", name):
                found.add("pooled")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if name in ("ttest_ind", "ttest_ind_from_stats"):
                equal = next((k.value for k in node.keywords if k.arg == "equal_var"), None)
                welch = isinstance(equal, ast.Constant) and equal.value is False
                found.add("welch" if welch else "pooled")
            elif name in ("ttest_rel", "ttest_1samp", "ttost_paired", "ttest_ind_statsmodels"):
                found.add("pooled")
    return found


def _t_test_variant(root: Path, job: Path) -> list[str]:
    """The paper's t-tests and the evaluation's must be the same variant: whole-number df
    are a pooled-variance (or paired) t, fractional df are Welch's."""
    text = _norm(_strip_references(paper_text(job, layout=False)))
    dfs = T_DF_RE.findall(text)
    whole = [df for df in dfs if "." not in df]
    fractional = [df for df in dfs if "." in df]
    variants = _t_variants(root / "evaluation/evaluation.py")
    if len(whole) >= 2 and not fractional and variants == {"welch"}:
        return [f"analyses: the paper's t-tests have whole-number df (t({whole[0]}), t({whole[1]})), i.e. "
                "pooled-variance Student (or paired) t, but evaluation/evaluation.py computes only Welch's t. "
                "Compute the paper's variant so the df and p values are comparable."]
    if len(fractional) >= 2 and not whole and variants == {"pooled"}:
        return [f"analyses: the paper reports Welch t-tests (fractional df, e.g. t({fractional[0]})) but "
                "evaluation/evaluation.py computes only pooled-variance t. Compute Welch's t."]
    return []


ANALYSIS_FAMILIES = {
    "ANOVA": (r"\banovas?\b|analysis of variance",
              r"anova|f_oneway|anova_lm|AnovaRM|\bols\b|\bOLS\b|kruskal"),
    "regression": (r"\bregressions?\b|\bglm\b|linear model",
                   r"\bols\b|\bOLS\b|\bglm\b|\bGLM\b|\blogit\b|\bLogit\b|linregress|lstsq|regression|polyfit"),
    "logistic regression": (r"\blogistic\b|\blogit\b",
                            r"\blogit\b|\bLogit\b|Binomial|logistic"),
    "mediation": (r"\bmediation\b|indirect effect", r"mediat|indirect"),
    "equivalence test": (r"\btost\b|equivalence test", r"tost|equivalen"),
    "mixed model": (r"mixed[- ]effects?|mixed model|multilevel|random[- ]effects?", r"mixedlm|MixedLM|lmer|random_eff"),
    "chi-square test": (r"chi[- ]?square|χ2|χ²|\bchi2\b", r"chi2|chisquare|chi_square|chi-square"),
    "Welch t-test": (r"\bwelch\b", r"equal_var\s*=\s*False|welch"),
}
NEGATION_RE = re.compile(r"(?i)not[_ -]ready|returned as|\b(not|no|isn't|aren't|cannot|can't|won't|without|missing|yet|lacks?|skipped|"
                         r"left out|omitted|unavailable|needs?|pending|later)\b")


def _code_identifiers(path: Path) -> str:
    """Identifiers, attributes and dict keys in a module, without comments or docstrings."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return ""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.append(node.arg + ("=" + repr(node.value.value) if isinstance(node.value, ast.Constant) else ""))
        elif isinstance(node, ast.alias):
            names.append(node.name)
        elif isinstance(node, ast.Dict):
            names.extend(k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
    return " ".join(names)


def _computed_keys(output: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(output, dict) and not _not_ready(output):
        for key, value in output.items():
            if not _not_ready(value):
                keys.append(prefix + str(key))
                keys += _computed_keys(value, prefix + str(key) + ".")
    elif isinstance(output, list):
        for value in output:
            keys += _computed_keys(value, prefix)
    return keys


def _reply_analysis_claims(root: Path, output: Any) -> list[str]:
    reply = root / "studio-reply.md"
    if not reply.is_file():
        return []
    text = reply.read_text(encoding="utf-8", errors="replace")
    implemented = _code_identifiers(root / "evaluation/evaluation.py")
    implemented += " " + " ".join(_computed_keys(output)).replace("_", " ")
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not sentence.strip() or NEGATION_RE.search(sentence):
            continue
        for family, (mention, code) in ANALYSIS_FAMILIES.items():
            if re.search(mention, sentence, re.IGNORECASE) and not re.search(code, implemented):
                claims.append(f"{family} (\"{_norm(sentence)[:90]}\")")
    if claims:
        return ["studio-reply.md says the package runs analyses that evaluation/evaluation.py does not implement: "
                + "; ".join(claims[:4]) + ". Implement them or say plainly that they are not computed yet."]
    return []


# --------------------------------------------------------------------------
# 4. Researcher reply

JARGON = [
    (r"\bQIDs?\d*\b", "survey question ids (QID…)"),
    (r"\barms?\b", "'arm' (say 'condition' or 'group')"),
    (r"\bplumbing\b", "'plumbing'"),
    (r"\bnot_ready\b|\bnot-ready\b", "'not_ready'"),
    (r"\bSHA(-?256)?\b|\bsha256\b|\bhash(es)?\b", "file hashes / SHA"),
    (r"\bsmoke[- ]test\b|\bstub\b", "'smoke test' / 'stub'"),
    (r"\bTrash\b", "the survey's 'Trash' block (say 'unused questions')"),
    (r"\bblocked_arms\b|\bprogramBindings\b|\brun_sessions\b|\bevaluate\(\)|\badapter(\.py)?\b|\bHuman Program\b",
     "pipeline internals"),
    (r"\bJSON\b|\bQSF\b|\bschema\b", "file-format names (JSON, QSF, schema)"),
]
OVERCLAIM_RE = re.compile(
    r"(?i)\bnothing\s+(is\s+|was\s+)?missing\b|\bno\s+materials?\s+(are\s+|were\s+)?missing\b|"
    r"\b(fully|completely|exactly)\s+(faithful|reproduc\w*|replicat\w*|identical)\b|"
    r"\ball\s+(of\s+)?(the\s+)?(paper'?s?\s+)?(tests|analyses|statistics)\s+(are\s+)?(computed|implemented|reproduced)\b")
CC_RE = re.compile(
    r"(?i)\b(?:creative\s+commons\s+(attribution(?:[-\s]+(?:non-?commercial|no-?derivatives?|no-?derivs|share-?alike))*)"
    r"(?:\s+licen[cs]e)?|cc[-\s]?(by(?:[-\s](?:nc|nd|sa))*))\s*(?:\(cc[-\s]?by[^)]*\)\s*)?(?:licen[cs]e\s*)?"
    r"(?:version\s*)?(\d\.\d)?")
CC0_RE = re.compile(r"(?i)\bcc0\b|cc\s+zero|public domain dedication")


def _cc_variant(match: re.Match) -> str:
    words = (match.group(1) or match.group(2) or "").lower()
    parts = ["by"]
    if re.search(r"non-?commercial|\bnc\b", words):
        parts.append("nc")
    if re.search(r"no-?deriv|\bnd\b", words):
        parts.append("nd")
    if re.search(r"share-?alike|\bsa\b", words):
        parts.append("sa")
    return "CC " + "-".join(parts).upper() + (f" {match.group(3)}" if match.group(3) else "")


ARTICLE_LICENCE_RE = re.compile(r"(?i)this (is an )?(open[- ]access )?(article|work|paper)|distributed under|"
                                r"published under|licensed under|is licen[cs]ed|under the terms of")
OTHER_LICENCE_RE = re.compile(r"(?i)\b(data|dataset|materials|osf|figure|fig\.|image|photo|icon|wikimedia|code)\b")


def paper_licence(job: Path) -> str | None:
    """The licence the paper PDF prints for the article itself, e.g. 'CC BY 4.0', or None.
    Only a licence in an article-licence sentence counts, not a data, materials or
    figure-credit licence; when none is found the reply's licence is not checked."""
    text = _norm(paper_text(job, layout=False))
    matches = sorted(list(CC_RE.finditer(text)) + list(CC0_RE.finditer(text)), key=lambda m: m.start())
    for match in matches:
        before = text[max(0, match.start() - 300):match.start()]
        near = text[max(0, match.start() - 120):match.start()]
        if not ARTICLE_LICENCE_RE.search(before) or OTHER_LICENCE_RE.search(near):
            continue
        if match.re is CC0_RE:
            return "CC0"
        if match.group(1) or match.group(2):
            return _cc_variant(match)
    return None


def _line_mentions_licence(line: str, licence: str) -> bool:
    """The first licence the line names is the paper's (other sources may follow)."""
    if licence == "CC0":
        return bool(CC0_RE.search(line))
    match = CC_RE.search(line)
    if match is None:
        return False
    wanted = licence.split(" ")
    got = _cc_variant(match).split(" ")
    return got[1] == wanted[1] and (len(wanted) < 3 or len(got) < 3 or got[2] == wanted[2])


def _plain(line: str) -> str:
    return re.sub(r"^[#>\-*\s]+|[*_`]+", "", line).strip()


def check_reply(root: Path, job: Path, result: dict | None = None) -> list[str]:
    reply = root / "studio-reply.md"
    if not reply.is_file():
        return []
    text = reply.read_text(encoding="utf-8", errors="replace")
    problems: list[str] = []
    lines = [_plain(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    heads = ("ready to run:", "needs you:", ("licence:", "license:"))
    for position, head in enumerate(heads):
        options = head if isinstance(head, tuple) else (head,)
        if len(lines) <= position or not lines[position].lower().startswith(options):
            problems.append(
                "studio-reply.md must open with three plain lines: 'Ready to run: …', 'Needs you: …', "
                "'Licence: …' (contract 'Researcher reply').")
            break
    licence = paper_licence(job)
    licence_line = lines[2] if len(lines) > 2 and lines[2].lower().startswith(("licence:", "license:")) else next(
        (line for line in lines if CC_RE.search(line) or re.search(r"(?i)\blicen[cs]e", line)), None)
    if licence and licence_line is not None:
        if not _line_mentions_licence(licence_line, licence):
            problems.append(f"studio-reply.md licence line must state the licence printed in the paper PDF "
                            f"({licence}) first; other sources' licences come after it.")
    body = text
    questions = [s for s in re.split(r"(?<=[?])\s+|(?<=？)|\n", body)
                 if s.strip().rstrip("\"'”’)").endswith(("?", "？")) and s.strip().endswith(("?", "？"))]
    if len(questions) > 3:
        problems.append(f"studio-reply.md asks {len(questions)} questions; ask at most three, each answerable "
                        "without opening the survey file, and settle the rest from the sources.")
    found = []
    # A word the paper or its materials use ("arms race", "ticket stub", "self-schema") is not jargon.
    own_words = _norm(paper_text(job, layout=False)).casefold() + " " + _visible_text(
        _load_json(root / "materials/materials.json"))
    for pattern, name in JARGON:
        for match in re.finditer(pattern, body):
            word = match.group(0).casefold()
            if not re.search(r"(?<![a-z0-9_])" + re.escape(word) + r"(?![a-z0-9_])", own_words):
                found.append(f"{name} ('{match.group(0)}')")
                break
    for span in re.findall(r"`([^`\n]+)`", text):
        if re.search(r"[_/]|\.\w{1,4}\b|^\w+(-\w+)+$", span) and not CC_RE.fullmatch(span.strip()):
            found.append(f"code-formatted internal name ('{span[:40]}')")
            break
    if found:
        problems.append("studio-reply.md uses internal vocabulary a professor should not need: "
                        + "; ".join(found[:6]) + ". Say it in plain words.")
    overclaim = OVERCLAIM_RE.search(text)
    if overclaim:
        problems.append(f"studio-reply.md overclaims ('{overclaim.group(0)}'); say what was found and what "
                        "was not checked.")
    size = next((line for line in lines if line.lower().startswith("run size:")), None)
    if size is None or len(re.findall(r"\d[\d,.]*", size)) < 2 \
            or not re.search(r"(\$|US\$|USD|€|£|¥)\s?\d|\d\s?(USD|dollars|美元)", size):
        problems.append("studio-reply.md needs a 'Run size:' line with the paper's sample size, the number of "
                        "model calls and an approximate cost in dollars (from agent_pipeline/estimate_run.py).")
    return problems


# --------------------------------------------------------------------------


def check_package(root: Path, job: Path) -> list[str]:
    """All checks in this module; an empty list means the package passes."""
    problems: list[str] = []
    result = run_probe(root) if (root / "task/adapter.py").is_file() else None
    if result and result.get("errors"):
        problems += [f"stand-in run: {error}" for error in result["errors"]]
    failed_runs = [run for run in (result or {}).get("runs") or [] if run.get("error")]
    if failed_runs:
        problems.append("stand-in run: run_sessions raised for " + "; ".join(
            f"{_label(run)}: {run['error']}" for run in failed_runs[:4])
            + ". Every condition id in task/task.json must run with run_sessions(llm, seed, n, arms=[id]) for n = 1, 2 or 4; a "
            "condition that cannot run returns a blocked session record instead of raising.")
    for check in (lambda: check_memory(root, result), lambda: check_results(root, job),
                  lambda: check_analyses(root, job, result), lambda: check_reply(root, job, result)):
        try:
            problems += check()
        except Exception as exc:  # noqa: BLE001 - a malformed record is reported, not a crash
            problems.append(f"package check could not run ({type(exc).__name__}: {exc}); fix the malformed record")
    return problems


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--probe":
        Path(sys.argv[3]).write_text(json.dumps(probe(Path(sys.argv[2]).resolve()), ensure_ascii=False),
                                     encoding="utf-8")
    elif len(sys.argv) == 3:
        found = check_package(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
        print("\n".join(f"- {p}" for p in found) or "no problems")
        raise SystemExit(1 if found else 0)
    else:
        raise SystemExit("usage: package_checks.py --probe <paper-folder> <out.json> | <paper-folder> <job-dir>")
