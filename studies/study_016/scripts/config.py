"""HumanStudy entry points for a persistent self-efficacy experiment."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import PreparationRequired,check_ready,run_participant
try:
    from src.core.study_config import BaseStudyConfig
except ImportError:
    from compat import BaseStudyConfig


class StudyGeniusSelfEfficacyConfig(BaseStudyConfig):
    REQUIRES_GROUP_TRIALS=True
    PLAYGROUND_PROTOCOL_VERSION=1
    MAX_MODEL_CALLS_PER_PARTICIPANT=28

    def __init__(self,study_path,specification=None):
        self.study_path=Path(study_path)
        self.specification=specification or json.loads((self.study_path/'source/specification.json').read_text())
        self.study_id=self.specification['study_id']

    def get_prompt_builder(self):
        return None

    def get_study_type(self):
        return 'persistent_writing_self_efficacy'

    def aggregate_results(self,raw_results):
        return raw_results

    def get_n_participants(self):
        return self.specification['reference_snapshot']['n']

    def get_instructions(self):
        return 'Bearbeite die Aufgaben aus der Perspektive einer Schülerin oder eines Schülers der Sekundarstufe. Antworte auf die jeweils gezeigten Fragen.'

    def create_trials(self,n_trials=None):
        check_ready(self.study_path)
        plan=json.loads((self.study_path/'source/materials/assignment_blueprint.json').read_text())
        if n_trials is not None and n_trials!=len(plan):
            if not 1<=int(n_trials)<=len(plan):raise ValueError('Invalid diagnostic sample size')
            arms=[[x for x in plan if x['condition']==c] for c in ('control','feedback')]
            plan=[x for pair in __import__('itertools').zip_longest(*arms) for x in pair if x is not None][:int(n_trials)]
            plan=[{**x,'run_scope':'diagnostic_subset'} for x in plan]
        return plan

    def run_group_experiment(self,trials,instructions,participant_pool_kwargs,prompt_builder=None):
        check_ready(self.study_path)
        factory=participant_pool_kwargs.get('participant_factory')
        if not callable(factory):
            if not participant_pool_kwargs.get('use_real_llm'):
                raise PreparationRequired('Real provider required; no simulated responses are used as agent results')
            from runtime import make_participant_factory,ReconstructedFeedback
            factory=make_participant_factory(self.study_path,participant_pool_kwargs)
        mode=participant_pool_kwargs.get('reconstruction_mode','full_surrogate')
        if mode!='full_surrogate':raise ValueError('Only the generated-baseline protocol is supported')
        feedback=participant_pool_kwargs.get('feedback_provider')
        if feedback is None:
            from runtime import ReconstructedFeedback
            feedback=ReconstructedFeedback(self.study_path,participant_pool_kwargs)
        scorer=participant_pool_kwargs.get('baseline_scorer')
        if mode=='full_surrogate' and scorer is None:
            from scorer_features import BaselineSurrogate
            scorer=BaselineSurrogate(self.study_path)
        records=[]
        for trial in trials:
            participant=factory({'population':self.specification['population'],
                                 'cluster_id':trial['cluster_id'],'instructions':instructions})
            record=run_participant(self.study_path,trial,participant,feedback,scorer,
                                   mode=mode)
            record['run_scope']=trial.get('run_scope','reference_sample')
            record['participant_calls']=getattr(participant,'calls',[])
            records.append(record)
        return {'individual_data':records,'execution_kind':'agent_run','reconstruction_mode':mode}

    def run_playground_protocol(self,trials,settings,profiles,on_participant):
        """Bridge the hosted runner to the original persistent protocol.

        Shared fixtures are explicitly synthetic; live calls use the runner's
        configured OpenRouter endpoint, with one participant context per slot.
        """
        import hashlib
        from runtime import make_participant_factory,ReconstructedFeedback
        from scorer_features import BaselineSurrogate
        live=settings['use_real_llm']
        scorer=BaselineSurrogate(self.study_path)
        if live:
            base_factory=make_participant_factory(self.study_path,settings)
            feedback=ReconstructedFeedback(self.study_path,settings)
        else:
            from run_study import fixture
            base_factory=lambda profile:fixture
            digest=hashlib.sha256((self.study_path/'source/materials/feedback_reconstruction.json').read_bytes()).hexdigest()
            feedback=lambda context:{'text':'SYNTHETIC FEEDBACK','origin':'developer_fixture','reconstruction_sha256':digest}
        records=[]
        for index,trial in enumerate(trials):
            def factory(profile):
                participant=base_factory({**profile,'instructions':profile['instructions']+'\n'+settings['participant_prompts'][index]})
                def ask(context):
                    settings['on_stage'](index,context['stage'])
                    return participant(context)
                ask.calls=getattr(participant,'calls',[])
                return ask
            result=self.run_group_experiment([trial],self.get_instructions(),{
                **settings,'participant_factory':factory,'feedback_provider':feedback,'baseline_scorer':scorer,
            })
            record=result['individual_data'][0]
            record['participant_id']=profiles[index]['participant_id']
            record['profile']=profiles[index]
            if not live:record['execution_kind']='synthetic_developer_fixture'
            records.append(record)
            on_participant(record)
        return {'individual_data':records,'execution_kind':'agent_run' if live else 'synthetic_developer_fixture',
                'participant_model':settings['model'],'feedback_model':settings['feedback_model'],
                'limitations':['Feedback generation and baseline scoring are declared methodological reconstructions.',
                               'The frozen scorer has not been validated on generated essays.']}
