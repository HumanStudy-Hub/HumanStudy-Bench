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


def test_davis_bound_is_not_turned_into_an_exact_p_value():
    spec = importlib.util.spec_from_file_location('study014_provenance_evaluator', ROOT / 'scripts/evaluator.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    data = []
    for a in range(6):
        responses = [{'response_text': str(5 + a + i % 3 + [1, 4, 7, 10][(i + a) % 4] * 3),
                      'trial_info': {'num_bidders': [1, 4, 7, 10][(i + a) % 4]}} for i in range(12)]
        data.append({'responses': responses})
    row = evaluator.evaluate_study({'individual_data': data})['test_results'][0]
    # The only human statistic is the original regression's printed bound.
    assert row['human_source_id'] == 'davis2011'
    assert row['human_coefficient'] == 2.98
    assert row['human_p_value'] is None
    assert row['human_p_value_bound'] == .01
    assert row['human_significant'] is True
    assert 'human_pearson_r' not in row


def test_later_benchmark_is_reference_only():
    gt = json.loads((ROOT / 'source/ground_truth.json').read_text())
    finding = gt['studies'][0]['findings'][0]
    later = finding['later_benchmark_see_also']
    assert later['source_id'] == 'feng2025' and later['used_in_scoring'] is False
    assert later['pearson_r'] == .42 and later['pearson_p_value'] is None
    assert all(t['source_id'] == 'davis2011' for t in finding['statistical_tests'])
