"""Keep original regression evidence separate from the later Pearson benchmark."""
import json
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def test_every_legacy_aggregate_has_explicit_provenance():
    gt = json.loads((ROOT / 'source/ground_truth.json').read_text())
    values = gt['studies'][0]['findings'][0]['original_data_points']['data']
    for group, fields in values.items():
        for field, value in fields.items():
            pointer = f'/studies/0/findings/0/original_data_points/data/{group}/{field}'
            record = gt['value_provenance'][pointer]
            assert record['value'] == value
            assert record['status']
            assert isinstance(record['printed_in'], list)
    assert gt['value_provenance']['/studies/0/findings/0/original_data_points/data/overall/sd_reserve_price']['printed_value'] == 23.22
    assert gt['value_provenance']['/studies/0/findings/0/original_data_points/data/overall/sell_through_rate']['printed_value'] == .76


def test_missing_pearson_p_value_is_not_inferred():
    spec = importlib.util.spec_from_file_location('study014_provenance_evaluator', ROOT / 'scripts/evaluator.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    responses = [{'response_text': str(5 + i * 3), 'trial_info': {'num_bidders': [1, 4, 7, 10][i % 4]}} for i in range(24)]
    row = evaluator.evaluate_study({'individual_data': [{'responses': responses}]})['test_results'][0]
    # Effect size: the later Pearson r, which has no printed p-value.
    assert row['human_pearson_r'] == .42
    assert row['human_effect_source_id'] == 'feng2025'
    assert row['human_effect_p_value_status'] == 'missing'
    # Significance and direction: the original regression's printed bound, not an exact p.
    assert row['human_significance_source_id'] == 'davis2011'
    assert row['human_p_value'] is None
    assert row['human_p_value_bound'] == .01
    assert row['human_significant'] is True
