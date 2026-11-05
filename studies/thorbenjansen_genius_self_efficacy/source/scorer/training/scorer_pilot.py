"""Local text-to-WLE surrogate. Not the original analytical-rubric/TAM scorer.

Run from workspace: .venv/bin/python <this file>
No network, APIs, demographic predictors, feedback, or outcome inputs.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'articles/genius-multiple-feedback/data/2026.03_geniuswle'
EXPECTED = '9f269b76c4e33db75bc4daaa292620b73f9f2a7ca7962d0b10b5b859a75c34a9'
OUT = Path(__file__).resolve().parents[1] / 'output/reports/scorer-pilot'


class LengthFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for text in X:
            words = re.findall(r'\b\w+\b', text.lower())
            result.append([np.log1p(len(text)), np.log1p(len(words)),
                           np.log1p(len(set(words))),
                           np.log1p(len(re.findall(r'[.!?]', text)))])
        return np.array(result)


def length_model():
    return Pipeline([('length', LengthFeatures()), ('scale', StandardScaler()),
                     ('ridge', Ridge())])


def text_model():
    return Pipeline([
        ('features', FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                    max_features=15000, sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                    min_df=3, max_features=15000, sublinear_tf=True)),
            ('length', Pipeline([('length', LengthFeatures()),
                                 ('scale', StandardScaler())])),
        ])),
        ('ridge', Ridge(solver='lsqr')),
    ])


def metrics(y, pred):
    return {'n': len(y), 'pearson_r': float(pearsonr(y, pred).statistic),
            'spearman_r': float(spearmanr(y, pred).statistic),
            'r2': float(r2_score(y, pred)),
            'mae_wle': float(mean_absolute_error(y, pred)),
            'rmse_wle': float(np.sqrt(mean_squared_error(y, pred))),
            'bias_pred_minus_reference': float(np.mean(pred-y))}


def main(raw_survey):
    sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert sha == EXPECTED, 'Source snapshot changed'
    with SOURCE.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 801 and len({r['id'] for r in rows}) == 801
    # The analysis CSV combines unknown classes under one artificial token.
    # Use original observed classes, excluding unknowns rather than calling them a class.
    raw_classes = json.loads(subprocess.check_output([
        'Rscript', '-e',
        's<-readRDS(commandArgs(TRUE)[1]); '
        'cat(jsonlite::toJSON(setNames(as.list(as.character(s$token)),as.character(s$id)),na="null",auto_unbox=TRUE))',
        str(raw_survey)], text=True))
    kept = []
    excluded = {'missing_wle': 0, 'missing_class': 0, 'empty_text': 0}
    for row in rows:
        if row['p2_wle'] in ('', 'NA', 'NaN'):
            excluded['missing_wle'] += 1
            continue
        original_class = raw_classes[row['id']]
        if original_class is None or original_class in ('', 'NA', 'NaN'):
            excluded['missing_class'] += 1
            continue
        if not row['p2_draft'].strip() or row['p2_draft'] == 'NA':
            excluded['empty_text'] += 1
            continue
        kept.append({**row, 'observed_class': original_class})
    X = np.array([r['p2_draft'] for r in kept])
    y = np.array([float(r['p2_wle']) for r in kept])
    groups = np.array([r['observed_class'] for r in kept])
    # Merge classes connected by identical normalized texts to avoid copy leakage.
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    seen = {}
    duplicates = 0
    for text, g in zip(X, groups):
        key = ' '.join(text.lower().split())
        if key in seen:
            parent[find(g)] = find(seen[key])
            duplicates += 1
        else:
            seen[key] = g
    split_groups = np.array([find(g) for g in groups])
    predictions = {k: np.zeros(len(y)) for k in ('mean', 'length', 'text')}
    fold_records = []
    folds = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(GroupKFold(5).split(X, y, split_groups), 1):
        assert not set(split_groups[train]) & set(split_groups[test])
        folds[test] = fold
        predictions['mean'][test] = np.mean(y[train])
        record = {'fold': fold, 'n_train': len(train), 'n_test': len(test)}
        for name, model in [('length', length_model()), ('text', text_model())]:
            search = GridSearchCV(model, {'ridge__alpha': [1., 10., 100.]},
                                  scoring='neg_mean_squared_error',
                                  cv=GroupKFold(3), n_jobs=1)
            search.fit(X[train], y[train], groups=split_groups[train])
            pred = search.predict(X[test])
            predictions[name][test] = pred
            record[name] = {'alpha': search.best_params_['ridge__alpha'],
                            **metrics(y[test], pred)}
        fold_records.append(record)
        print(json.dumps(record), flush=True)
    summary = {
        'status': 'local_surrogate_pilot_not_validated_benchmark_scorer',
        'source_sha256': sha, 'source_rows': len(rows), 'included': len(y),
        'original_class_source_sha256': hashlib.sha256(raw_survey.read_bytes()).hexdigest(),
        'class_source': 'Original survey token; artificial pooled missing-class code in analysis CSV is not used.',
        'source_wle_crosscheck': 'All 801 IDs and WLE/missing values agree with published analysis data.rds (independently checked).',
        'source_reference': 'https://doi.org/10.2378/peu2026.art12d',
        'scoring_reference': 'https://aclanthology.org/2024.bea-1.18/',
        'excluded_sequential': excluded,
        'observed_classes': len(set(groups)),
        'split_components_after_duplicate_union': len(set(split_groups)),
        'duplicate_normalized_texts': duplicates,
        'target': 'Existing algorithm-derived baseline p2_wle, NOT human ratings',
        'predictors': 'Only baseline p2_draft; no outcomes, feedback, profiles or IDs',
        'validation': '5 outer class-grouped folds; alpha selected with 3 inner grouped folds; all vectorizers fit inside folds',
        'metrics': {k: metrics(y, p) for k, p in predictions.items()},
        'folds': fold_records, 'python_sklearn': sklearn.__version__,
        'limitations': [
            'Agreement with existing automated WLE is not validity against human gold.',
            'Direct WLE regression does not reproduce the original 10 indicators and PCM.',
            'No validation on model-generated essays or a separate student cohort.',
            'Classes, not schools, are held out.',
            'No inference-equivalence test for the original moderator effects.',
            'No claim of parity with the published scoring algorithms.',
            'No external training or participant-agent training performed.',
        ],
    }
    # Cluster bootstrap of fixed out-of-fold predictions; does not refit models.
    rng = np.random.default_rng(20261020)
    units = np.unique(split_groups)
    boot = []
    for _ in range(1000):
        sampled = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([np.flatnonzero(split_groups == g) for g in sampled])
        boot.append(metrics(y[idx], predictions['text'][idx]))
    summary['text_fixed_oof_cluster_bootstrap_95pct'] = {
        k: [float(v) for v in np.quantile([b[k] for b in boot], [.025, .975])]
        for k in ('pearson_r', 'r2', 'mae_wle', 'rmse_wle')}
    # Local deployable candidate; final fit has no independent test score of its own.
    final = GridSearchCV(text_model(), {'ridge__alpha': [1., 10., 100.]},
                         scoring='neg_mean_squared_error', cv=GroupKFold(5), n_jobs=1)
    final.fit(X, y, groups=split_groups)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(final.best_estimator_, OUT / 'local_wle_surrogate.joblib')
    summary['final_fit_alpha'] = final.best_params_['ridge__alpha']
    summary['model_sha256'] = hashlib.sha256((OUT/'local_wle_surrogate.joblib').read_bytes()).hexdigest()
    with (OUT/'out_of_fold_predictions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['local_row', 'fold', 'reference_wle', 'mean_prediction',
                         'length_prediction', 'text_prediction'])
        writer.writerows(zip(range(len(y)), folds, y, predictions['mean'],
                             predictions['length'], predictions['text']))
    (OUT/'validation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['metrics'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', type=Path,
                        help='Score one UTF-8 baseline essay with the local candidate')
    parser.add_argument('--raw-survey', type=Path, default=Path('/tmp/genius_original_survey.rds'),
                        help='Original public survey RDS for observed classroom labels')
    parser.add_argument('--source',type=Path,default=SOURCE,
                        help='Original 801-row authorized CSV, frozen hash checked')
    parser.add_argument('--output',type=Path,default=OUT,help='New local training output directory')
    args = parser.parse_args()
    SOURCE=args.source
    OUT=args.output
    if args.score:
        essay = args.score.read_text(encoding='utf-8').strip()
        if not essay:
            raise ValueError('Empty essay; no score assigned')
        model = joblib.load(OUT / 'local_wle_surrogate.joblib')
        print(json.dumps({'status': 'unvalidated_local_surrogate_prediction',
                          'predicted_wle': float(model.predict([essay])[0]),
                          'not_original_scorer': True,
                          'warning': 'Not validated on AI-generated essays.'}))
    else:
        main(args.raw_survey)
