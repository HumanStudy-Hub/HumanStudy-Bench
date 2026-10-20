"""Run from an extracted study directory; no calls without an explicit --live."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from config import StudyGeniusSelfEfficacyConfig
from protocol import run_participant


def fixture(context):
    if 'questions' in context:
        return {'responses':{q['code']:{i['code']:q['minimum'] for i in q['items']} for q in context['questions']}}
    if context['stage']=='baseline_writing':return {'texts':{x['code']:'SYNTHETIC FIXTURE' for x in context['fields']}}
    return {'text':'SYNTHETIC FIXTURE'}


def dry_run(study,mode='full_surrogate'):
    config=StudyGeniusSelfEfficacyConfig(study)
    trials=config.create_trials(2)
    from scorer_features import BaselineSurrogate
    scorer=BaselineSurrogate(study) if mode=='full_surrogate' else None
    feedback=lambda c:{'text':'SYNTHETIC FEEDBACK','origin':'developer_fixture'}
    records=[run_participant(study,t,fixture,feedback,scorer,test_only=True,mode=mode) for t in trials]
    return {'execution_kind':'synthetic_developer_fixture','mode':mode,
            'records':len(records),'stages_each':[len(r['history']) for r in records],
            'conditions':[r['condition'] for r in records],
            'claim':'Interface exercise only; no model performance or inferential evidence'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--feedback-model')
    parser.add_argument('--api-base',default=os.environ.get('GENIUS_API_BASE'))
    parser.add_argument('--limit',type=int,help='Diagnostic subset only; full reference comparison requires 801')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    study=Path(__file__).resolve().parents[1]
    if not args.live:
        print(json.dumps(dry_run(study,'full_surrogate'),indent=2));return
    if not args.model or not args.feedback_model or not args.output:
        parser.error('--live requires --model, --feedback-model, and --output')
    if args.output.exists():parser.error('Output exists; choose a new run directory')
    settings={'use_real_llm':True,'model':args.model,'feedback_model':args.feedback_model,
              'api_base':args.api_base,'reconstruction_mode':'full_surrogate','temperature':1.0}
    config=StudyGeniusSelfEfficacyConfig(study)
    # Validate provider configuration before creating output or starting calls.
    from runtime import make_participant_factory,ReconstructedFeedback
    settings['participant_factory']=make_participant_factory(study,settings)
    settings['feedback_provider']=ReconstructedFeedback(study,settings)
    args.output.mkdir(parents=True)
    trials=config.create_trials(args.limit)
    metadata={'mode':'full_surrogate','participant_model':args.model,'feedback_model':args.feedback_model,
              'n_planned':len(trials),'provider_base':args.api_base or 'https://api.openai.com/v1',
              'started_unix':time.time(),'status':'running','timing':'API timings are not human exposure times'}
    (args.output/'run.json').write_text(json.dumps(metadata,indent=2))
    records=[]
    try:
        for trial in trials:
            result=config.run_group_experiment([trial],config.get_instructions(),settings)
            record=result['individual_data'][0]
            records.append(record)
            (args.output/f"slot_{trial['slot']:04d}.json").write_text(json.dumps(record,ensure_ascii=False,indent=2))
        (args.output/'results.json').write_text(json.dumps({'individual_data':records},ensure_ascii=False,indent=2))
        metadata.update(status='completed',n_completed=len(records))
    except Exception as exc:
        metadata.update(status='failed',n_completed=len(records),error_type=type(exc).__name__)
        raise
    finally:
        metadata['ended_unix']=time.time()
        (args.output/'run.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'status':metadata['status'],'records':len(records),'output':str(args.output)}))


if __name__=='__main__':main()
