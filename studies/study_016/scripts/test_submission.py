import base64
import copy
import hashlib
import json
from pathlib import Path
import pytest
from config import StudyGeniusSelfEfficacyConfig
from evaluator import evaluate_study,records_to_rows,compare_to_targets
from protocol import run_participant
from runtime import ChatBackend,PersistentParticipant,ReconstructedFeedback,content_for_context,parse_object
from run_study import fixture,dry_run
from scorer_features import BaselineSurrogate

STUDY=Path(__file__).resolve().parents[1]
M=json.loads((STUDY/'source/materials/protocol.json').read_text())


def test_generated_baseline_role_boundary_and_persistent_order():
    assert not (STUDY/'source/materials/participant_profiles.json').exists()
    contexts=[]
    def participant(c):contexts.append(copy.deepcopy(c));return fixture(c)
    seen=[]
    def feedback(c):
        seen.append(c)
        return {'text':'SYNTHETIC','origin':'fixture'}
    r=run_participant(STUDY,{'condition':'feedback','cluster_id':'class_001','slot':0},participant,
                      feedback,BaselineSurrogate(STUDY),test_only=True)
    assert len(r['history'])==23 and r['history'][3]=='baseline_writing'
    assert r['history'][-1]=='post_self_efficacy'
    assert r['baseline_score']['scoring_kind']=='surrogate'
    assert all(set(c)=={'part','draft','task'} for c in seen)
    text=json.dumps(contexts)
    assert all(k not in text for k in ['baseline_wle','ground_truth','source_sha256','reported_coefficient','p3_se_w'])
    assert evaluate_study({'individual_data':[r]})['status']=='not_scored'


def test_multimodal_payload_preserves_images_and_history():
    seen=[]
    def transport(payload):
        seen.append(copy.deepcopy(payload))
        return {'choices':[{'message':{'content':'{"text":"SYNTHETIC"}'},'finish_reason':'stop'}],'model':'fixture'}
    backend=ChatBackend('fixture',{},transport=transport)
    p=PersistentParticipant(STUDY,backend,'TEST')
    c={'stage':'draft_introduction','task':M['writing_task'],'assets':M['writing_assets']}
    p(c);p({'stage':'revision_introduction'})
    assert len(seen[1]['messages'])==4
    images=[x for x in seen[0]['messages'][1]['content'] if x['type']=='image_url']
    assert len(images)==len(M['writing_assets'])>0
    for image,asset in zip(images,M['writing_assets']):
        data=base64.b64decode(image['image_url']['url'].split(',',1)[1])
        assert hashlib.sha256(data).hexdigest()==asset['sha256']


def test_feedback_uses_only_current_block_and_declares_adaptation():
    seen=[]
    class Backend:
        last_metadata={'requested_model':'fixture'}
        def __call__(self,messages,json_output=True):seen.append(messages);return 'SYNTHETIC'
    f=ReconstructedFeedback(STUDY,{},backend=Backend())
    r=f({'part':'argument_2','draft':'CURRENT BLOCK ONLY','task':'TASK'})
    assert r['validated_historical_configuration'] is False and r['reconstruction_sha256']
    text=json.dumps(seen)
    assert 'CURRENT BLOCK ONLY' in text and '{insert argument here}' not in text
    with pytest.raises(ValueError):f({'part':'argument_2','draft':'x','task':'x','goals':4})


def test_no_unconfigured_network_or_silent_json_repair(monkeypatch):
    for name in ('GENIUS_API_KEY','OPENAI_API_KEY'):monkeypatch.delenv(name,raising=False)
    with pytest.raises(ValueError):ChatBackend('explicit-model',{})
    with pytest.raises(ValueError):parse_object('not json')
    with pytest.raises(ValueError):parse_object('[]')
    with pytest.raises(ValueError):parse_object('{"responses":{"T3SE":{"1":2,"1":4}}}')


def test_frozen_scorer_is_portable_and_never_claims_original_equivalence():
    r=BaselineSurrogate(STUDY)({'texts':{'a':'Akkus benötigen Rohstoffe. Recycling kann diese wieder nutzbar machen.'}})
    assert isinstance(r['wle'],float) and r['scoring_kind']=='surrogate'
    assert r['validated_original_scoring'] is False


def test_single_protocol_has_offline_end_to_end_exercise():
    r=dry_run(STUDY)
    assert r['records']==2 and r['stages_each']==[23,23]
    assert set(r['conditions'])=={'control','feedback'}
    assert r['execution_kind']=='synthetic_developer_fixture'


def test_diagnostic_subsets_do_not_become_full_reference_comparisons():
    c=StudyGeniusSelfEfficacyConfig(STUDY)
    t=c.create_trials(2)
    assert {x['condition'] for x in t}=={'control','feedback'}
    assert all(x['run_scope']=='diagnostic_subset' for x in t)


def test_evaluator_checks_frozen_scorer_and_assignment_before_diagnostic_comparison():
    # Provider-record contract fixtures exercise provenance checks, not model behavior.
    config=StudyGeniusSelfEfficacyConfig(STUDY)
    feedback_sha=hashlib.sha256((STUDY/'source/materials/feedback_reconstruction.json').read_bytes()).hexdigest()
    feedback=lambda c:{'text':'FIXTURE','origin':'fixture','reconstruction_sha256':feedback_sha}
    records=[]
    for trial in config.create_trials(2):
        record=run_participant(STUDY,trial,fixture,feedback,BaselineSurrogate(STUDY),test_only=True)
        record.update(execution_kind='agent_run',run_scope='diagnostic_subset')
        records.append(record)
    assert evaluate_study({'individual_data':records})['status']=='diagnostic_only'
    altered=copy.deepcopy(records)
    altered[0]['baseline_score']['model_sha256']='wrong_model'
    assert evaluate_study({'individual_data':altered})['status']=='not_scored'
    altered=copy.deepcopy(records)
    altered[0]['cluster_id']='altered_cluster'
    assert evaluate_study({'individual_data':altered})['status']=='not_scored'


def test_comparison_uses_recalculated_801_case_reference():
    reference=json.loads((STUDY/'source/matched_reference.json').read_text())
    fits=json.loads((STUDY/'source/evaluator_only/human_source_method_reexecution.json').read_text())
    assert reference['n']==801
    assert reference['condition_sizes']=={'control':395,'feedback':406}
    assert all(x['n']==801 for x in fits.values())
    compared=compare_to_targets(fits,reference)
    assert len(compared)==24
    assert all(x['coefficient_difference']==0 and x['human_reference_n']==801 for x in compared)
    assert any(x['human_reference_coefficient']!=x['published_coefficient'] for x in compared)


def test_estimability_guard_uses_available_item_means_and_keeps_missingness():
    import csv
    from evaluator import estimability_issues
    with (STUDY/'source/evaluator_only/reference_items.csv').open() as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key=='baseline_wle' or key.startswith(('p1_','p3_')):
                row[key]=float(row[key]) if row[key] not in ('','NA') else None
    assert estimability_issues(rows)==[]
    altered=copy.deepcopy(rows)
    for row in altered:
        for i in range(1,5):row[f'p3_se_w_{i}']=4
    for i in range(1,5):altered[0][f'p3_se_w_{i}']=None
    issues=estimability_issues(altered)
    assert issues==[{'variable':'p3_se_w','observed_n':800,'distinct_observed_values':1,
        'reason':'At least two distinct observed scores are required by the source model'}]
    altered[1]['p3_se_w_1']=2
    assert estimability_issues(altered)==[]
