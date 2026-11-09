"""The pinned hsenv engine (V080-PLATFORM B1, §2.1).

- scripts/research/effects_core.py is hsenv.effects_core, not a copy.
- Every community revision's effects.json checks exactly as it did under the Bench copy it replaced
  (tests/fixtures/effects_core_golden.json, recorded at b34d214). A difference is a changed verdict on a
  published study and must be listed in the hsenv release notes before the pin moves.
- Protocols are data. The engine modules never call eval/exec/compile/__import__, except hsenv.expr, which
  compiles an expression only after checking every syntax node against its whitelist (no attributes,
  lambdas, imports or dunder names) and evaluates it without builtins. The design (§2.1) says "no
  eval/exec"; the real guarantee is that whitelist, so that is what these tests pin.
"""
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

hsenv = pytest.importorskip("hsenv")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.research import effects_core  # noqa: E402
from scripts.research.catalog import digest  # noqa: E402


def test_effects_core_is_the_engine():
    from hsenv import effects_core as engine
    assert effects_core.check is engine.check
    assert effects_core.kg is engine.kg
    assert effects_core.VOCAB_VERSION == engine.VOCAB_VERSION


def test_loading_the_file_by_path_still_exposes_private_helpers():
    spec = importlib.util.spec_from_file_location("ec_by_path", ROOT / "scripts/research/effects_core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.quoted) and callable(module.rendered_quote)


PIN = re.compile(r"^hsenv @ git\+https://github\.com/HumanStudy-Hub/hsenv@(?P<ref>\S+)\s+#\s*v(?P<tag>\d+\.\d+\.\d+)$")


def hsenv_pin() -> re.Match:
    """The one hsenv pin. requirements.txt is the only place it is written; workflows grep this line."""
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip().startswith("hsenv")]
    assert len(lines) == 1, lines
    match = PIN.match(lines[0])
    assert match, f"hsenv must be pinned as 'hsenv @ git+https://github.com/HumanStudy-Hub/hsenv@<sha>  # vX.Y.Z': {lines[0]}"
    return match


def test_workflows_read_the_pin_from_requirements_only():
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        for line in path.read_text().splitlines():
            code = line.split("#", 1)[0]
            if re.search(r"hsenv\s*(@|==|>=|~=)|install[^|]*hsenv", code):
                assert "grep '^hsenv' requirements.txt" in code, f"{path.name}: {line.strip()}"


def test_requirements_pin_hsenv_by_commit_sha():
    pin = hsenv_pin()
    if pin["ref"] == "<SHA>":
        pytest.xfail(f"placeholder: fill the commit SHA of the hsenv v{pin['tag']} tag in requirements.txt")
    assert re.fullmatch(r"[0-9a-f]{40}", pin["ref"]), f"pin a commit SHA, not a mutable ref: {pin['ref']}"


def test_installed_engine_is_the_pinned_release():
    from importlib.metadata import version
    assert version("hsenv") == hsenv_pin()["tag"]


def test_community_effects_golden():
    golden = json.loads((ROOT / "tests/fixtures/effects_core_golden.json").read_text())
    assert golden["vocab"] == effects_core.VOCAB_VERSION
    # every community revision that has effects.json is covered, not a sample
    present = {f"{e.parents[2].name}/{e.parent.name}" for e in ROOT.glob("community/*/revisions/*/effects.json")}
    assert set(golden["revisions"]) == present
    for key, want in golden["revisions"].items():
        paper, rid = key.split("/")
        rev = ROOT / "community" / paper / "revisions" / rid
        effects = json.loads((rev / "effects.json").read_text())
        program = json.loads((rev / "program.json").read_text())
        problems, warnings = effects_core.check(Path("."), effects, program)
        assert problems == want["problems"], key
        assert warnings == want["warnings"], key
        assert digest(effects_core.kg(effects, program)) == want["kg"], key


FORBIDDEN = {"eval", "exec", "compile", "__import__"}


@pytest.mark.parametrize("module", ["engine", "checks", "schema_check", "effects_core", "evaldata", "rewards"])
def test_protocol_readers_never_execute_protocol_text(module):
    path = Path(hsenv.__file__).parent / f"{module}.py"
    if not path.is_file():
        pytest.skip(f"hsenv has no {module}.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ((isinstance(n.func, ast.Name) and n.func.id in FORBIDDEN)
                  or (isinstance(n.func, ast.Attribute) and n.func.attr in {"exec", "eval"}
                      and isinstance(n.func.value, ast.Name) and n.func.value.id == "builtins"))]
    assert not calls, f"{path.name}: {[ast.unparse(c)[:60] for c in calls]}"


def test_expr_whitelist_has_no_escape_hatches():
    from hsenv import expr
    for node in ("Attribute", "Lambda", "NamedExpr", "Starred", "JoinedStr", "Await", "Yield"):
        if hasattr(ast, node):
            assert getattr(ast, node) not in expr.ALLOWED, node
    tree = ast.parse((Path(expr.__file__)).read_text(encoding="utf-8"))
    builtins_cleared = any(isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == "__builtins__"
        for t in n.targets) for n in ast.walk(tree))
    assert builtins_cleared


@pytest.mark.parametrize("source", ["().__class__", "__import__('os')", "(lambda: 1)()", "open('x')",
                                    "[c for c in ().__class__.__mro__]", "getattr(1, 'real')", "__builtins__"])
def test_expr_refuses_code(source):
    from hsenv.expr import ExprError, evaluate
    with pytest.raises((ExprError, NameError, KeyError, TypeError)):
        evaluate(source, {})
