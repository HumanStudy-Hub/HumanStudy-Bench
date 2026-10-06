"""Portable Human Program v2 validation; the JSON contract is shared with Studio."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path

SCHEMA = json.loads((Path(__file__).resolve().parent.parent / 'contracts/human-program.schema.json').read_text())

def validate_human_program(value: object) -> dict:
    def fail(path, message):
        raise ValueError(f'{path}: {message}')
    def finite(v, depth=0):
        if depth > 64:
            fail('program', 'structure is too deeply nested')
        if isinstance(v, float) and not math.isfinite(v):
            fail('program', 'invalid JSON value')
        if isinstance(v, dict):
            for item in v.values(): finite(item, depth + 1)
        elif isinstance(v, list):
            for item in v: finite(item, depth + 1)
        elif v is not None and not isinstance(v, (str, int, float, bool)):
            fail('program', 'invalid JSON value')
    finite(value)
    if len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) > 500_000:
        fail('program', 'exceeds 500 KB; reference large materials as resources')
    def check(v, rule, path='program', depth=0):
        if depth > 64: fail(path, 'structure is too deeply nested')
        if '$ref' in rule:
            return check(v, SCHEMA['$defs'][rule['$ref'].split('/')[-1]], path, depth + 1)
        if 'const' in rule and (type(v) is not int or v != rule['const']): fail(path, 'unsupported schema version')
        if 'enum' in rule and v not in rule['enum']: fail(path, 'invalid choice')
        kind = rule.get('type')
        if kind == 'string':
            if not isinstance(v, str) or not rule.get('minLength', 0) <= len(v) <= rule.get('maxLength', math.inf) or rule.get('pattern') and not re.search(rule['pattern'], v):
                fail(path, 'invalid string')
        if kind == 'integer' and (type(v) is not int or not rule.get('minimum', -math.inf) <= v <= rule.get('maximum', math.inf)): fail(path, 'invalid integer')
        if kind == 'array':
            if not isinstance(v, list) or len(v) > rule.get('maxItems', math.inf): fail(path, 'invalid array')
            for i, item in enumerate(v): check(item, rule.get('items', {}), f'{path}[{i}]', depth + 1)
        if kind == 'object':
            if not isinstance(v, dict): fail(path, 'must be an object')
            for key in rule.get('required', []):
                if key not in v: fail(path, f'missing {key}')
            for key, item in v.items():
                props = rule.get('properties', {})
                if rule.get('additionalProperties') is False and key not in props: fail(path, f'unknown field {key}; use extensions')
                if key in props: check(item, props[key], f'{path}.{key}', depth + 1)
    check(value, SCHEMA)
    p = value
    def unique(ids, path):
        if len(set(ids)) != len(ids): fail(path, 'duplicate ID')
    for key in ['studies', 'nodes', 'steps', 'relations', 'evidence', 'issues']:
        unique([x['id'] for x in p[key]], key)
    if len(p['relations']) + sum(len(s['next']) + len(s['children']) for s in p['steps']) > 1200:
        fail('program', 'exceeds 1200 display connections')
    nodes = {n['id']: n for n in p['nodes']}
    studies = {s['id']: s for s in p['studies']}
    steps = {s['id']: s for s in p['steps']}
    evidence = {e['id']: e for e in p['evidence']}
    def refs(ids, known, path):
        unique(ids, path)
        for ident in ids:
            if ident not in known: fail(path, f'unknown reference {ident}')
    def field(f, path):
        refs(f['evidenceIds'], evidence, path)
        refs(f.get('studyIds', []), studies, path)
        if f['origin'] == 'derived' and not f.get('derivation', '').strip(): fail(path, 'derived value requires derivation')
    for n in p['nodes']:
        if n['id'].startswith('flow:'): fail(n['id'], 'prefix flow: is reserved')
        refs(n['studyIds'], studies, n['id']); refs(n['evidenceIds'], evidence, n['id'])
        unique([f['id'] for f in n['fields']], n['id'])
        for f in n['fields']: field(f, n['id'])
    for s in p['steps']:
        refs([s['studyId']], studies, s['id'])
        for key in ['actorIds', 'inputIds', 'outputIds', 'visibleIds']: refs(s[key], nodes, s['id'])
        refs(s['children'], steps, s['id']); refs([e['to'] for e in s['next']], steps, s['id']); refs(s['evidenceIds'], evidence, s['id'])
        if 'rule' in s: field(s['rule'], s['id'])
        for ident in s['children'] + [e['to'] for e in s['next']]:
            if steps[ident]['studyId'] != s['studyId']: fail(s['id'], 'cross-study flow edge')
    done = set()
    def visit(ident, active):
        if ident in active: fail(ident, 'Procedure containment cycle')
        if ident in done: return
        for child in steps[ident]['children']: visit(child, active | {ident})
        done.add(ident)
    for s in p['steps']: visit(s['id'], set())
    finished = set()
    def study_visit(ident, active):
        if ident in active: fail(ident, 'Study dependency cycle')
        if ident in finished: return
        deps = studies[ident].get('dependsOn', [])
        refs(deps, studies, ident)
        for dep in deps: study_visit(dep, active | {ident})
        finished.add(ident)
    for s in p['studies']: study_visit(s['id'], set())
    for r in p['relations']: refs(list(dict.fromkeys([r['from'], r['to']])), nodes, r['id'])
    for e in p['evidence']:
        if not any(e.get(k) for k in ['sourceId', 'path', 'url']): fail(e['id'], 'source reference is required')
        loc = e['locator']
        if 'start' in loc and 'end' in loc and loc['end'] < loc['start']: fail(e['id'], 'reversed source range')
    for i in p['issues']:
        refs(i['evidenceIds'], evidence, i['id'])
        for key, known in [('studyId', studies), ('nodeId', nodes), ('stepId', steps)]:
            if key in i: refs([i[key]], known, i['id'])
        if 'fieldId' in i and not any(f['id'] == i['fieldId'] for f in nodes.get(i.get('nodeId'), {}).get('fields', [])):
            fail(i['id'], 'unknown field reference')
    for n in p['nodes']:
        for f in n['fields']:
            if f['state'] in ('missing', 'decision') and not any(i.get('nodeId') == n['id'] and (not i.get('fieldId') or i['fieldId'] == f['id']) for i in p['issues']):
                fail(n['id'], 'unresolved field requires a review issue')
    for s in p['steps']:
        if s.get('rule', {}).get('state') in ('missing', 'decision') and not any(i.get('stepId') == s['id'] for i in p['issues']):
            fail(s['id'], 'unresolved rule requires a review issue')
    return p


def canonical_model(model: dict) -> dict:
    return model.get('program', model)
