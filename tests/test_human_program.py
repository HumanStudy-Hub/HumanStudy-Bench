"""Shared v2 fixtures cover real package structures and extension-only studies."""
import copy
import json
from pathlib import Path
import pytest
from agent_pipeline.human_program import validate_human_program
from agent_pipeline.studio_output import validate_study_model, validate_build_sidecars

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = json.loads((ROOT / 'tests/fixtures/human-program-v2.json').read_text())

def sample(ident):
    return copy.deepcopy(next(c['program'] for c in FIXTURES['cases'] if c['id'] == ident))

@pytest.mark.parametrize('case', FIXTURES['cases'], ids=lambda c: c['id'])
def test_shared_cases_preserve_every_typed_value(case):
    program = case['program']
    assert validate_human_program(program) == program
    assert validate_study_model(program) == program
    assert validate_study_model({'program': program})['program'] == program

def test_references_cycles_scopes_and_derivations_are_checked():
    p = sample('study_009'); p['steps'][0]['actorIds'] = ['ghost']
    with pytest.raises(ValueError, match='unknown reference'): validate_human_program(p)
    p = sample('study_009'); p['steps'][1]['children'] = ['rounds']
    with pytest.raises(ValueError, match='containment cycle'): validate_human_program(p)
    p = sample('study_012'); p['steps'][0]['next'] = [{'to': 'send-1'}]
    with pytest.raises(ValueError, match='cross-study'): validate_human_program(p)
    p = sample('exploratory-extension'); p['nodes'][0]['fields'][0]['origin'] = 'derived'
    with pytest.raises(ValueError, match='derivation'): validate_human_program(p)
    p = sample('exploratory-extension'); p['schemaVersion'] = 3
    with pytest.raises(ValueError, match='unsupported schema version'): validate_human_program(p)

def test_core_sidecar_and_accepted_program_must_agree(tmp_path):
    root = tmp_path / 'package/paper'; root.mkdir(parents=True)
    p = sample('study_005')
    (root / 'study.json').write_text(json.dumps({'program': p}))
    (root / 'studio-model.json').write_text(json.dumps(p))
    (root / 'studio-reply.md').write_text('Built from accepted program')
    assert validate_build_sidecars(root.parent, {'program': p})[0]
    q = copy.deepcopy(p); q['nodes'][0]['fields'][0]['state'] = 'confirmed'
    (root / 'studio-model.json').write_text(json.dumps(q))
    valid, reason = validate_build_sidecars(root.parent)
    assert not valid and 'disagree' in reason
    (root / 'studio-model.json').write_text(json.dumps(p))
    valid, reason = validate_build_sidecars(root.parent, {'program': q})
    assert not valid and 'exactly match' in reason

def test_structurally_valid_is_not_automatically_execution_ready():
    p = sample('exploratory-extension')
    assert p['issues'] == []
    assert p['steps'] == []
    assert not any(n['kind'] == 'hypothesis' for n in p['nodes'])
    assert validate_human_program(p)['nodes'][0]['extensions']['codingRules']['multipleCodes'] is True

def test_unresolved_rule_requires_an_explicit_review_item():
    p = sample('study_009'); p['steps'][0]['rule']['state'] = 'missing'
    with pytest.raises(ValueError, match='requires a review issue'): validate_human_program(p)
    p['issues'] = [{'id': 'round-rule', 'title': 'Define repetition', 'severity': 'blocking', 'reason': 'Stopping rule is absent', 'impact': 'Cannot run faithfully', 'suggestedAction': 'Researcher supplies rule', 'stepId': 'rounds', 'evidenceIds': []}]
    assert validate_human_program(p)['issues'][0]['severity'] == 'blocking'

def test_package_validator_rejects_disconnected_execution_bindings(tmp_path):
    import subprocess
    import sys
    from agent_pipeline.validate_package import REQUIRED
    root = tmp_path / 'package/paper'; root.mkdir(parents=True)
    for name in REQUIRED:
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}' if target.suffix == '.json' else '# placeholder')
    p = sample('study_009')
    (root / 'study.json').write_text(json.dumps({'program': p}))
    command = [sys.executable, str(ROOT / 'agent_pipeline/validate_package.py'), str(root.parent)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and 'programBindings' in result.stderr
    (root / 'task/task.json').write_text(json.dumps({'programBindings': {'programId': p['id'], 'stepIds': [s['id'] for s in p['steps']], 'recordNodeIds': ['guesses'], 'analysisNodeIds': []}}))
    # This is a local test harness, not a runnable scientific study.
    (root / 'task/adapter.py').write_text('def run_sessions(llm, seed, n, arms=None):\n    return []\n')
    (root / 'evaluation/evaluation.py').write_text('def evaluate(sessions):\n    return {}\n')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
