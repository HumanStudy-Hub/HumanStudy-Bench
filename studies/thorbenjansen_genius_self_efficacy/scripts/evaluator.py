"""Original-scale outcome analysis; no fabricated PAS/ECS or learning claims."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

GOALS=('lap','pap','pav','wav')
QUESTION_MAPPING={'lap':('T0AGL',('1','2','3','4')),
                  'pap':('T0AGPA',('A1','A2','A3','A4')),
                  'pav':('T0AGPV',('1','2','3','4')),
                  'wav':('T0AGEV',('1','2','3','4'))}


def records_to_rows(records):
    rows=[]
    for r in records:
        if r.get('execution_kind')!='agent_run':
            raise ValueError('Developer fixtures cannot become scientific benchmark results')
        score=r.get('baseline_score',{})
        mode=r.get('reconstruction_mode')
        if mode!='full_surrogate' or score.get('scoring_kind')!='surrogate' or not score.get('model_sha256'):
            raise ValueError('Frozen text-scoring provenance missing')
        if score.get('wle') is None:
            if not score.get('missing_reason'):raise ValueError('Unexplained missing WLE')
        elif type(score['wle']) not in (int,float) or not math.isfinite(score['wle']):
            raise ValueError('Invalid calibrated WLE')
        if r.get('condition') not in ('control','feedback') or not r.get('cluster_id'):
            raise ValueError('Missing condition or cluster')
        if len(r.get('cycles',[]))!=5:
            raise ValueError('Incomplete five-cycle experiment')
        if [c.get('part') for c in r['cycles']]!=['introduction','argument_1','argument_2','argument_3','conclusion']:
            raise ValueError('Cycle order mismatch')
        history=r.get('history',[])
        if history[-3:]!=['holistic_revision','post_reflection','post_self_efficacy']:
            raise ValueError('Outcome must follow holistic revision and reflection')
        if r['condition']=='feedback':
            key='reconstruction_sha256'
            if any(not c.get('feedback',{}).get(key) for c in r['cycles']):
                raise ValueError('Feedback provenance is unconfirmed')
        q={k:v for b in r['questionnaires'].values() for k,v in b.items()}
        if q.get('T3SE')!=r['questionnaires']['post_self_efficacy'].get('T3SE'):
            raise ValueError('Ambiguous outcome questionnaire source')
        row={'condition':r['condition'],'cluster_id':r['cluster_id'],'baseline_wle':score['wle']}
        for g,(code,keys) in QUESTION_MAPPING.items():
            for i,key in enumerate(keys,1):row[f'p1_ach_{g}_{i}']=q[code][key]
        for tp,code in [('p1','T0SE'),('p3','T3SE')]:
            for i in range(1,5):row[f'{tp}_se_w_{i}']=q[code][str(i)]
        for k,v in row.items():
            if k.startswith(('p1_','p3_')) and v is not None:
                maximum=8 if '_ach_' in k else 5
                if type(v) not in (int,float) or not math.isfinite(v) or v!=int(v) or not 1<=v<=maximum:
                    raise ValueError('Questionnaire item outside original response scale')
        rows.append(row)
    if {r['condition'] for r in rows}!={'control','feedback'}:
        raise ValueError('Both condition arms are required')
    return rows


def compare_to_targets(fits,reference):
    comparisons=[]
    for goal in GOALS:
        for inference in ('primary','cr2'):
            fit={x['Parameter']:x for x in fits[goal][inference]}
            for target in reference['targets']:
                if target['goal']!=goal or target['inference']!=inference:continue
                actual=fit[target['parameter']]
                p=actual['p']
                comparisons.append({'finding_id':target['id'],'goal':goal,'inference':inference,
                     'parameter':target['parameter'],'coefficient':actual['Coefficient'],
                     'SE':actual['SE'],'statistic':actual.get('Statistic'),
                     'p_value':p,'df_error':actual.get('df_error'),
                     'human_reference_n':reference['n'],
                     'human_reference_coefficient':target['coefficient'],
                     'human_reference_p_value':target['p_value'],
                     'published_coefficient':target['published_coefficient'],
                     'published_p_value':target['published_p_value'],
                     'coefficient_difference':actual['Coefficient']-target['coefficient'],
                     'human_significant_at_05':target['p_value']<.05,
                     'agent_significant_at_05':p<.05,
                     'interpretation':'Inference and coefficient comparison only; nonsignificance is not equivalence'})
    return comparisons


def estimability_issues(rows):
    """Mirror the source model's observed-score guards before costly imputation."""
    def available_mean(row, keys):
        values=[row[k] for k in keys if row[k] is not None]
        return sum(values)/len(values) if values else None
    columns={'p2_wle':[r['baseline_wle'] for r in rows]}
    for tp in ('p1','p3'):
        columns[tp+'_se_w']=[available_mean(r,[f'{tp}_se_w_{i}' for i in range(1,5)]) for r in rows]
    for goal in GOALS:
        columns['p1_ach_'+goal]=[available_mean(r,[f'p1_ach_{goal}_{i}' for i in range(1,5)]) for r in rows]
    return [{'variable':name,'observed_n':sum(x is not None for x in values),
             'distinct_observed_values':len({x for x in values if x is not None}),
             'reason':'At least two distinct observed scores are required by the source model'}
            for name,values in columns.items() if len({x for x in values if x is not None})<2]


def evaluate_study(results):
    records=results.get('individual_data',[])
    try:rows=records_to_rows(records)
    except (ValueError,KeyError,TypeError) as e:
        return {'status':'not_scored','reason':str(e),'test_results':[]}
    study=Path(__file__).resolve().parents[1]
    modes={r.get('reconstruction_mode') for r in records}
    if len(modes)!=1:return {'status':'not_scored','reason':'Do not pool reconstruction modes','test_results':[]}
    plan=json.loads((study/'source/materials/assignment_blueprint.json').read_text())
    scorer=json.loads((study/'source/scorer/validation.json').read_text())
    feedback_sha=hashlib.sha256((study/'source/materials/feedback_reconstruction.json').read_bytes()).hexdigest()
    for record in records:
        slot=record.get('slot')
        if type(slot) is not int or not 0<=slot<len(plan):
            return {'status':'not_scored','reason':'Invalid planned slot','test_results':[]}
        if any(record[k]!=plan[slot][k] for k in ('condition','cluster_id')):
            return {'status':'not_scored','reason':'Planned allocation changed','test_results':[]}
        if record['baseline_score']['model_sha256']!=scorer['packaged_model_sha256']:
            return {'status':'not_scored','reason':'Text-scoring model differs from frozen model','test_results':[]}
        if record['condition']=='feedback' and any(c['feedback']['reconstruction_sha256']!=feedback_sha for c in record['cycles']):
            return {'status':'not_scored','reason':'Feedback configuration differs from declared reconstruction','test_results':[]}
    if len(records)!=801 or {r.get('slot') for r in records}!=set(range(801)) or any(r.get('run_scope')=='diagnostic_subset' for r in records):
        return {'status':'diagnostic_only','reason':'Reference comparison requires all 801 unique planned slots','n':len(records),'test_results':[]}
    readiness=json.loads((study/'source/readiness.json').read_text())
    if readiness['ready_for_scientific_scoring'] is not True:
        return {'status':'not_scored','reason':'Scientific source/method gates remain open','test_results':[]}
    issues=estimability_issues(rows)
    if issues:
        return {'status':'not_estimable','reason':'Insufficient observed-score variation for the source model',
                'n':len(records),'diagnostics':issues,'test_results':[],
                'interpretation':'A completed model run can yield non-estimable responses; no inferential comparison is fabricated.'}
    fits={}
    with tempfile.TemporaryDirectory(prefix='genius-selfeff-eval-') as tmp:
        tmp=Path(tmp);inputs=tmp/'items.csv'
        with inputs.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        for i,goal in enumerate(GOALS):
            output=tmp/(goal+'.json')
            run=subprocess.run(['Rscript',str(study/'scripts/run_statistics.R'),str(inputs),str(output),goal,
                                '5000','10000','20',str(20261020+i)],capture_output=True,text=True,timeout=14400)
            if run.returncode:
                return {'status':'not_scored','reason':'Original-method fit failed for '+goal,
                        'diagnostic':run.stderr[-2500:],'test_results':[]}
            fits[goal]=json.loads(output.read_text())
    reference=json.loads((study/'source/matched_reference.json').read_text())
    if reference['n']!=len(records):
        return {'status':'not_scored','reason':'Human and agent comparison sample sizes differ','test_results':[]}
    return {'status':'inference_comparison_only','reconstruction_mode':next(iter(modes)),
            'comparison_samples':{'human':reference['n'],'agent':len(records)},
            'test_results':compare_to_targets(fits,reference),
            'limitations':['No validated PAS/ECS aggregation implemented',
                           'Simulated self-efficacy is not actual motivation or independent learning',
                           'Reference is the frozen 801-case re-analysis; published estimates are contextual',
                           'Non-significance is not evidence of equivalence']}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Evaluate one complete declared GENIUS agent run')
    parser.add_argument('results',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new result file')
    result=evaluate_study(json.loads(args.results.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'status':result['status'],'output':str(args.output)}))
