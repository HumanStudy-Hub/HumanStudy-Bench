import csv
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from protocol import PreparationRequired,run_participant,validate_battery
from config import StudyGeniusSelfEfficacyConfig
from evaluator import evaluate_study,records_to_rows

STUDY=HERE.parent
MATERIAL=json.loads((STUDY/'source/materials/protocol.json').read_text())


def fake_participant(context):
    if 'questions' in context:
        return {'responses':{q['code']:{x['code']:q['minimum'] for x in q['items']} for q in context['questions']}}
    if context['stage']=='baseline_writing':
        return {'texts':{x['code']:'ARTIFICIAL TEST TEXT' for x in context['fields']}}
    return {'text':'ARTIFICIAL '+context['stage']}


def fake_scorer(context):
    assert set(context)=={'task','texts'}
    assert len(context['texts'])==5
    return {'wle':0.0,'origin':'synthetic test fixture, not original scoring',
            'validated_original_scoring':False,'scoring_kind':'surrogate','model_sha256':'synthetic_fixture_hash'}


def fake_feedback(context):
    assert set(context)=={'part','draft','task'}
    return {'text':'ARTIFICIAL FEEDBACK','origin':'synthetic test fixture only'}


def test_readiness_prevents_silent_scientific_runs():
    state=json.loads((STUDY/'source/readiness.json').read_text())
    assert state['ready_for_technical_review'] is True
    assert state['ready_for_submission']==state.get('live_agent_workflow_pilot_passed',False)
    assert state['ready_for_public_release'] is True
    assert len(StudyGeniusSelfEfficacyConfig(STUDY).create_trials())==801
    with pytest.raises(PreparationRequired):
        run_participant(STUDY,{'condition':'control','cluster_id':'class_001'},fake_participant)


def test_persistent_order_and_correct_outcome_timing():
    contexts=[]
    def participant(context):
        contexts.append(context)
        return fake_participant(context)
    result=run_participant(STUDY,{'condition':'feedback','cluster_id':'class_001'},
                           participant,fake_feedback,fake_scorer,test_only=True)
    stages=result['history']
    assert len(stages)==23
    assert stages[:3]==['pre_group_2','pre_group_3','pre_group_4']
    assert stages[-3:]==['holistic_revision','post_reflection','post_self_efficacy']
    assert stages.index('baseline_writing')<stages.index('after_baseline')<stages.index('draft_introduction')
    second=next(x for x in contexts if x['stage']=='draft_argument_1')
    assert second['essay_so_far']['introduction']=='ARTIFICIAL revision_introduction'
    assert 'T3SE' in result['post_self_efficacy']
    serialized=json.dumps(contexts)
    for forbidden in ['mod2_var_cgm','reference_items.csv','ground_truth','reported_coefficient','human_source_method_reexecution']:
        assert forbidden not in serialized
    assert evaluate_study({'individual_data':[result]})['status']=='not_scored'


def test_missing_source_callbacks_cannot_be_faked_by_default():
    trial={'condition':'feedback','cluster_id':'class_001'}
    with pytest.raises(PreparationRequired):
        run_participant(STUDY,trial,fake_participant,fake_feedback,test_only=True)
    with pytest.raises(PreparationRequired):
        run_participant(STUDY,trial,fake_participant,baseline_scorer=fake_scorer,test_only=True)


def test_unscorable_baseline_stays_missing_with_a_reason():
    def missing_score(context):
        return {'wle':None,'origin':'artificial scorer fixture',
                'missing_reason':'artificial unscorable baseline','scoring_kind':'surrogate','model_sha256':'synthetic_fixture_hash'}
    r=run_participant(STUDY,{'condition':'control','cluster_id':'class_001'},
                       fake_participant,baseline_scorer=missing_score,test_only=True)
    assert r['baseline_score']['wle'] is None
    def unexplained(context):return {'wle':None,'origin':'artificial fixture'}
    with pytest.raises(ValueError):
        run_participant(STUDY,{'condition':'control','cluster_id':'class_001'},
                          fake_participant,baseline_scorer=unexplained,test_only=True)


def test_control_uses_the_exact_verified_instruction():
    r=run_participant(STUDY,{'condition':'control','cluster_id':'class_001'},
                       fake_participant,baseline_scorer=fake_scorer,test_only=True)
    assert all(c['feedback']['text']=='Nimm dir jetzt die Zeit, deinen Textabschnitt nochmal zu überarbeiten.' for c in r['cycles'])


def test_response_scales_missingness_and_item_coverage():
    battery=MATERIAL['post_self_efficacy_questionnaire']
    r=fake_participant({'questions':battery})
    r['responses']['T3SE']['1']=None
    assert validate_battery(r,battery)['T3SE']['1'] is None
    r['responses']['T3SE']['1']=6
    with pytest.raises(ValueError):validate_battery(r,battery)
    r['responses']['T3SE']['1']=2.5
    with pytest.raises(ValueError):validate_battery(r,battery)
    del r['responses']['T3SE']['1']
    with pytest.raises(ValueError):validate_battery(r,battery)


def test_reported_targets_include_nulls_and_robustness():
    reference=json.loads((STUDY/'source/ground_truth.json').read_text())
    assert len(reference['targets'])==24
    t={(x['goal'],x['parameter'],x['inference']):x for x in reference['targets']}
    assert t['lap','p2_wle_cgm:mod2_var_cgm:fb','primary']['p_value']==.046
    assert t['lap','p2_wle_cgm:mod2_var_cgm:fb','cr2']['p_value']==.151
    assert t['pap','mod2_var_cgm:fb','primary']['coefficient']==-.059
    assert t['pav','mod2_var_cgm:fb','primary']['coefficient']==-.087
    assert all(t[g,'fb','primary']['p_value']>.05 for g in ('lap','pap','pav','wav'))
    assert t['wav','mod2_var_cgm:fb','primary']['p_value']==.297


def test_instrument_mapping_and_source_audit():
    q={q['code']:q for block in MATERIAL['pre_questionnaires'] for q in block['questions']}
    assert all(len(q[code]['items'])==4 for code in ['T0AGL','T0AGPA','T0AGPV','T0AGEV','T0SE'])
    assert q['T0SE']['maximum']==5
    assert q['T0AGPA']['maximum']==8
    assert [x['code'] for x in q['T0AGPA']['items']]==['A1','A2','A3','A4']
    audit=json.loads((STUDY/'source/source_audit.json').read_text())
    assert audit['raw_n']==889 and audit['filtered_n']==801
    assert audit['raw_class_missing']==22 and audit['raw_observed_classes']==50
    assert audit['analysis_cluster_levels']==51
    assert audit['baseline_wle_nonmissing_count']==779
    assert audit['baseline_wle_max_abs_difference']==0


def test_reference_rows_have_no_original_identifiers_or_demographics():
    with (STUDY/'source/evaluator_only/reference_items.csv').open() as f:
        reader=csv.DictReader(f);names=reader.fieldnames;data=list(reader)
    assert len(data)==801
    assert {'id','token','schoolid','sex','age','seed','p3_revision','p3_revision_log'}.isdisjoint(names)
    assert sum(x['condition']=='feedback' for x in data)==406
    assert sum(x['condition']=='control' for x in data)==395
    assert all(x['cluster_id'].startswith('class_') for x in data)


def test_original_graphics_and_source_prompt_candidates_are_present():
    assets=json.loads((STUDY/'source/materials/asset_inventory.json').read_text())
    assert len(assets)==7
    for x in assets:
        assert hashlib.sha256((STUDY/x['path']).read_bytes()).hexdigest()==x['sha256']
    prompts=json.loads((STUDY/'source/materials/feedback_prompt_candidates.json').read_text())
    assert set(prompts['prompts'])=={'Introduction','Pro Argument','Contra Argument','Conclusion'}
    assert prompts['must_not_silently_translate_or_fill_placeholders'] is True


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is required for the R score-preparation check")
def test_r_scoring_preserves_original_available_item_mean_and_all_missing():
    r=subprocess.run(['Rscript',str(HERE/'test_statistics.R')],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
