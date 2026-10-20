"""Sequential self-efficacy study. Explicit callbacks; no network or gold access."""
import copy
import json
import math
import time
from pathlib import Path

PARTS = ('introduction','argument_1','argument_2','argument_3','conclusion')


class PreparationRequired(RuntimeError):
    pass


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def check_ready(study):
    state=read_json(Path(study)/'source/readiness.json')
    if state['ready_for_simulation'] is not True:
        raise PreparationRequired('; '.join(x['description'] for x in state['blockers']))


def validate_battery(response,battery):
    if not isinstance(response,dict) or not isinstance(response.get('responses'),dict):
        raise ValueError('Questionnaire requires responses keyed by original question code')
    data=response['responses']
    expected={q['code'] for q in battery}
    if set(data)!=expected:
        raise ValueError('Questionnaire must include each original question exactly once')
    clean={}
    for q in battery:
        x=data[q['code']]
        if not isinstance(x,dict) or set(x)!={v['code'] for v in q['items']}:
            raise ValueError('Questionnaire item coverage mismatch: '+q['code'])
        values={}
        for item in q['items']:
            v=x[item['code']]
            if v is not None:
                if type(v) not in (int,float) or not math.isfinite(v) or v!=int(v):
                    raise ValueError('Likert responses must be integer codes or explicit null')
                if not q['minimum']<=v<=q['maximum']:
                    raise ValueError('Likert response out of range')
                v=int(v)
            values[item['code']]=v
        clean[q['code']]=values
    return clean


def _text(response):
    if not isinstance(response,dict) or not isinstance(response.get('text'),str):
        raise ValueError('Expected a text response')
    return response['text']


def run_participant(study,trial,participant,feedback_provider=None,
                    baseline_scorer=None,*,test_only=False,mode='full_surrogate'):
    """Persistent participant callback. Scoring and feedback are distinct roles.

    The feedback callback receives only current block text and its task/section.
    It never receives goals, self-efficacy, other blocks or human outcomes.
    The scorer receives only baseline task/text. Neither callback is implemented
    with a new replacement LLM. All test-only output is ineligible for scoring.
    """
    if not test_only:
        check_ready(study)
    if trial['condition'] not in ('control','feedback'):
        raise ValueError('Unknown condition')
    if mode!='full_surrogate':
        raise ValueError('Unknown reconstruction mode')
    if baseline_scorer is None:
        raise PreparationRequired('Frozen baseline text scorer is required')
    if trial['condition']=='feedback' and feedback_provider is None:
        raise PreparationRequired('Declared feedback provider is required')
    materials=read_json(Path(study)/'source/materials/protocol.json')
    history=[];questionnaires={};session_started=time.monotonic()

    def ask(context):
        context=copy.deepcopy(context)
        history.append(context['stage'])
        response=participant(context)
        if time.monotonic()-session_started>5400:
            raise TimeoutError('Session exceeded the 90-minute upper bound')
        return response

    def battery(stage,qs,extra=None):
        response=ask({'stage':stage,'questions':qs,**(extra or {})})
        result=validate_battery(response,qs)
        questionnaires[stage]=result
        return result

    for block in materials['pre_questionnaires']:
        battery(block['stage'],block['questions'])
    response=ask({'stage':'baseline_writing','task':materials['baseline_task'],
              'fields':materials['baseline_fields'],'time_limit_seconds':900,
              'assets':materials['baseline_assets']})
    fields={x['code'] for x in materials['baseline_fields']}
    if not isinstance(response,dict) or set(response.get('texts',{}))!=fields:
        raise ValueError('Baseline requires all five original text fields')
    if not all(isinstance(v,str) for v in response['texts'].values()):
        raise ValueError('Baseline text fields must be strings')
    baseline={x['code']:response['texts'][x['code']] for x in materials['baseline_fields']}
    score=baseline_scorer({'task':materials['baseline_task'],'texts':copy.deepcopy(baseline)})
    if not isinstance(score,dict) or not score.get('origin'):
        raise ValueError('Baseline scorer must supply provenance')
    if score.get('wle') is None:
        if not score.get('missing_reason'):
            raise ValueError('Missing baseline WLE requires an explicit reason')
    elif type(score['wle']) not in (int,float) or not math.isfinite(score['wle']):
        raise ValueError('Baseline WLE must be finite or explicitly missing')
    if mode=='full_surrogate' and (score.get('scoring_kind')!='surrogate' or not score.get('model_sha256')):
        raise PreparationRequired('Frozen surrogate provenance required')
    battery('after_baseline',materials['after_baseline_questionnaire'],
            {'instructions':materials['after_baseline_instructions']})
    essay={};cycles=[]
    for item in materials['cycles']:
        part=item['part']
        draft=_text(ask({'stage':'draft_'+part,'task':materials['writing_task'],
                         'instructions':item['draft_instruction'],'essay_so_far':essay,
                         'assets':materials['writing_assets']}))
        essay[part]=draft
        if trial['condition']=='control':
            feedback={'text':materials['control_message'],'origin':'verified_original_control'}
        else:
            feedback=feedback_provider({'part':part,'draft':draft,'task':materials['writing_task']})
            if not isinstance(feedback,dict) or not isinstance(feedback.get('text'),str) or not feedback.get('origin'):
                raise ValueError('Feedback must include text and provenance')
            if not test_only and not feedback.get('reconstruction_sha256'):
                raise PreparationRequired('Declared feedback reconstruction required')
        ratings=validate_battery(ask({'stage':'rating_'+part,'feedback':feedback['text'],
                                      'questions':[item['rating_question']]}),[item['rating_question']])
        revision=_text(ask({'stage':'revision_'+part,'instructions':item['revision_instruction'],
                            'draft':draft,'feedback':feedback['text'],'essay_so_far':essay}))
        cycles.append({'part':part,'draft':draft,'feedback':feedback,'ratings':ratings,'revision':revision})
        essay[part]=revision
    final=_text(ask({'stage':'holistic_revision','instructions':materials['holistic_instruction'],
                    'essay_so_far':essay}))
    battery('post_reflection',materials['reflection_questionnaire'],{'final_essay':final})
    post=battery('post_self_efficacy',materials['post_self_efficacy_questionnaire'])
    # Original outcome precedes later comparison/summative-feedback pages.
    return {'condition':trial['condition'],'cluster_id':trial['cluster_id'],'slot':trial.get('slot'),
            'reconstruction_mode':mode,
            'baseline_texts':baseline,'baseline_score':score,'questionnaires':questionnaires,
            'cycles':cycles,'final_essay':final,'post_self_efficacy':post,'history':history,
            'execution_kind':'synthetic_developer_fixture' if test_only else 'agent_run',
            'measurement_scope':'simulated writing self-efficacy responses; no learning or keystroke claim'}
