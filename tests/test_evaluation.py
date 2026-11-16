"""Check policy identity and fair-pool selection with deliberately different scores."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reproduction/evaluation'
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('evaluation_primary', SOURCE / 'rescore_primary.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
sys.path.pop(0)


def test_native_mt_uses_raw_confidence_including_boundary():
    data = {'kf_preds': np.array([0, 1, 0]), 'kf_ts_msp': np.ones(3),
            'mt_preds': np.array([0, 1, 0]), 'mt_raw_msp': np.array([.49, .5, .8]),
            'mt_ts_msp': np.array([.9, .2, .1]), 'kraken_preds': np.array([0, 1, 0])}
    policy = {'kf_tau': .8, 'mt_native_tau': .5, 'mt_ts_control_tau': .6}
    native = scorer.policy_predictions(data, policy, 'native')
    extra = scorer.policy_predictions(data, policy, 'ts-control')
    np.testing.assert_array_equal(native['MT'], [-1, 1, 0])
    np.testing.assert_array_equal(extra['MT'], [0, -1, -1])


def test_common_pool_removes_union_of_declines(monkeypatch):
    monkeypatch.setattr(scorer, 'NBOOT', 10)
    y = np.array([0, 1, 0, 1, 0])
    predictions = {'KF': np.array([-1, 1, 0, 1, 0]),
                   'MT': np.array([0, -1, 0, 0, 0]),
                   'K2': np.array([0, 1, -1, 1, 0])}
    keep, result = scorer.fair_reads(predictions, y)
    np.testing.assert_array_equal(keep, [False, False, False, True, True])
    assert result['n_fair'] == 2
    assert result['n_wrong'] == {'KF': 0, 'MT': 1, 'K2': 0}
    assert result['declined'] == {'KF': 1, 'MT': 1, 'K2': 1}


def test_interpolation_and_realizable_threshold_are_distinct():
    fpr = np.array([0., .02, .10, 1.])
    tpr = np.array([0., .6, .8, 1.])
    thresholds = np.array([np.inf, .8, .5, 0.])
    rule = scorer.s1.operating_point(fpr, tpr, thresholds, .95)
    assert rule['sensitivity'] == .6
    assert np.isclose(scorer.s1.interp_sens(fpr, tpr, .95), .675)


def test_new_evidence_hashes_and_scope():
    data, protocol = scorer.load_inputs()
    assert len(data['labels']) == 100000
    assert protocol['mt_native_tau'] == .5
    assert 'after inspection' in protocol['policy_selection']
