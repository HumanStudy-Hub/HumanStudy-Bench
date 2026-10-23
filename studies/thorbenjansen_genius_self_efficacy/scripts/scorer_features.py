"""Features of the frozen local surrogate; no reference/outcome access."""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LengthFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result=[]
        for text in X:
            words=re.findall(r'\b\w+\b',text.lower())
            result.append([np.log1p(len(text)),np.log1p(len(words)),
                           np.log1p(len(set(words))),np.log1p(len(re.findall(r'[.!?]',text)))])
        return np.array(result)


class BaselineSurrogate:
    def __init__(self,study):
        import hashlib,json,joblib
        from pathlib import Path
        folder=Path(study)/'source/scorer'
        self.metadata=json.loads((folder/'validation.json').read_text())
        model=folder/'wle_surrogate.joblib'
        self.sha=hashlib.sha256(model.read_bytes()).hexdigest()
        if self.sha!=self.metadata['packaged_model_sha256']:
            raise ValueError('Scorer checksum mismatch')
        self.model=joblib.load(model)

    def __call__(self,context):
        text='\n'.join(context['texts'].values())
        if not text.strip():
            return {'wle':None,'missing_reason':'empty baseline','origin':'frozen_local_wle_surrogate',
                    'scoring_kind':'surrogate','model_sha256':self.sha}
        return {'wle':float(self.model.predict([text])[0]),'origin':'frozen_local_wle_surrogate',
                'scoring_kind':'surrogate','model_sha256':self.sha,
                'validated_original_scoring':False,
                'scope':'Cross-validated against prior automated WLE; not human gold or validated on generated text'}
