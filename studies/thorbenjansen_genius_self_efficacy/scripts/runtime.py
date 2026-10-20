"""Persistent multimodal runtime with strict JSON parsing and no score/gold prompts.

Uses an explicitly configured OpenAI-compatible chat endpoint. No network calls
occur at import, trial creation, preview, or dry-run. No fallback model/formatter.
"""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error


def parse_object(text):
    text=text.strip()
    if text.startswith('```') and text.endswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    def unique_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON response key: '+key)
            result[key]=value
        return result
    result=json.loads(text,object_pairs_hook=unique_pairs)
    if not isinstance(result,dict):raise ValueError('Response must be a JSON object')
    return result


class ChatBackend:
    def __init__(self,model,settings,transport=None):
        if not model:raise ValueError('Choose an explicit model; no default participant model')
        self.model=model
        self.base=settings.get('api_base') or os.environ.get('GENIUS_API_BASE','https://api.openai.com/v1')
        self.key=settings.get('api_key') or os.environ.get('GENIUS_API_KEY') or os.environ.get('OPENAI_API_KEY')
        self.temperature=float(settings.get('temperature',1.0))
        self.transport=transport
        self.last_metadata={}
        if not self.key and transport is None:raise ValueError('Set GENIUS_API_KEY for the configured provider')

    def __call__(self,messages,json_output=True):
        payload={'model':self.model,'messages':copy.deepcopy(messages),'temperature':self.temperature,
                 'max_tokens':8192}
        if json_output:payload['response_format']={'type':'json_object'}
        if self.transport:
            result=self.transport(payload)
        else:
            req=urllib.request.Request(self.base.rstrip('/')+'/chat/completions',
                data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
            except urllib.error.HTTPError as error:
                raise RuntimeError('Provider request failed, HTTP '+str(error.code)) from None
        choice=result['choices'][0]
        if choice.get('finish_reason') not in ('stop',None):
            raise ValueError('Provider did not return a complete response: '+str(choice.get('finish_reason')))
        text=choice['message']['content']
        if not isinstance(text,str):raise ValueError('Provider returned no text')
        self.last_metadata={'requested_model':self.model,'returned_model':result.get('model'),
                            'response_id':result.get('id'),'usage':result.get('usage',{}),
                            'temperature':self.temperature}
        return text


def content_for_context(study,context):
    context=copy.deepcopy(context)
    assets=context.pop('assets',[])
    if context['stage']=='initial_state':
        schema={'acknowledged':True}
    elif 'questions' in context:
        schema={'responses':{q['code']:{i['code']:'integer scale code or null' for i in q['items']} for q in context['questions']}}
    elif context['stage']=='baseline_writing':
        schema={'texts':{x['code']:'your text' for x in context['fields']}}
    else:schema={'text':'your text'}
    content=[{'type':'text','text':json.dumps(context,ensure_ascii=False)+'\nAntworte ausschließlich als JSON in dieser Struktur: '+json.dumps(schema,ensure_ascii=False)}]
    root=Path(study).resolve()
    for asset in assets:
        p=(root/asset['path']).resolve()
        if not p.is_relative_to(root):raise ValueError('Asset outside study')
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=asset['sha256']:raise ValueError('Image checksum mismatch')
        content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode(),'detail':'high'}})
    return content


class PersistentParticipant:
    def __init__(self,study,backend,instructions):
        self.study=study;self.backend=backend
        self.messages=[{'role':'system','content':instructions+'\nAntworte auf Deutsch. Verwende ausschließlich das angeforderte JSON. Fehlende Fragebogenantworten dürfen null sein.'}]
        self.calls=[]

    def __call__(self,context):
        start=time.monotonic()
        message={'role':'user','content':content_for_context(self.study,context)}
        self.messages.append(message)
        text=self.backend(self.messages)
        result=parse_object(text)
        elapsed=time.monotonic()-start
        if context.get('time_limit_seconds') and elapsed>context['time_limit_seconds']:
            raise TimeoutError('Baseline generation exceeded protocol time limit')
        self.messages.append({'role':'assistant','content':text})
        self.calls.append({'stage':context['stage'],'elapsed_seconds':elapsed,**self.backend.last_metadata})
        return result


def make_participant_factory(study,settings):
    def factory(profile):
        return PersistentParticipant(study,ChatBackend(settings.get('model'),settings),profile['instructions'])
    return factory


class ReconstructedFeedback:
    def __init__(self,study,settings,backend=None):
        material=Path(study)/'source/materials'
        self.prompts=json.loads((material/'feedback_prompt_candidates.json').read_text())['prompts']
        path=material/'feedback_reconstruction.json'
        self.config=json.loads(path.read_text());self.sha=hashlib.sha256(path.read_bytes()).hexdigest()
        self.backend=backend or ChatBackend(settings.get('feedback_model') or self.config.get('model'),
                                           {**settings,'temperature':self.config['temperature']})

    def __call__(self,context):
        if set(context)!={'part','draft','task'}:raise ValueError('Feedback role received extra context')
        label=self.config['section_mapping'][context['part']]
        prompt=self.prompts[label]
        # Explicitly declared adaptation of the unbound published templates.
        prompt=prompt.replace('{}','folgenden Schreibaufgabe')
        placeholder={'Introduction':'{insert introduction here}','Conclusion':'{insert conclusion here}'}.get(label,'{insert argument here}')
        prompt=prompt.replace(placeholder,context['draft'])
        text=self.backend([{'role':'system','content':self.config['system_instruction']},
                           {'role':'user','content':'Aufgabe:\n'+context['task']+'\n\n'+prompt}],json_output=False)
        return {'text':text,'origin':'published_prompt_reconstruction_v1',
                'reconstruction_sha256':self.sha,'validated_historical_configuration':False,
                'provider':self.backend.last_metadata}
