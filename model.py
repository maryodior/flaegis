"""Loads the trained models and turns a raw transaction into a fraud score."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).parent / 'assets'

MODEL_NAMES = {
    'central': 'Central model (all data pooled)',
    'fedavg_iid': 'FedAvg, IID banks',
    'fedavgm_iid': 'FedAvgM, IID banks',
    'fedavg_noniid': 'FedAvg, non-IID banks',
    'fedavgm_noniid': 'FedAvgM, non-IID banks',
    'fedavgm05_noniid': 'FedAvgM (momentum 0.5), non-IID banks',
}

PRE = json.loads((ASSETS / 'preprocessing.json').read_text())
FEATURES = PRE['features']
MEAN = np.array(PRE['mean'])
SCALE = np.array(PRE['scale'])
THRESHOLD = PRE['threshold']

FEATURE_LABELS = {
    'type': 'Transaction type',
    'amount': 'Amount',
    'oldbalanceOrg': 'Sender balance before',
    'newbalanceOrig': 'Sender balance after',
    'oldbalanceDest': 'Receiver balance before',
    'newbalanceDest': 'Receiver balance after',
    'errorBalanceOrig': 'Sender balance mismatch',
    'errorBalanceDest': 'Receiver balance mismatch',
    'drained': 'Account emptied',
    'amount_ratio': 'Share of balance moved',
}


def weights_path(key):
    return ASSETS / ('central_weights.npz' if key == 'central' else f'fl_{key}_weights.npz')


def available_models():
    return [k for k in MODEL_NAMES if weights_path(k).exists()]


def load_weights(key):
    w = np.load(weights_path(key))
    return w['coef'].ravel(), float(np.ravel(w['intercept'])[0])


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as Stage 1 (add_features_v2)."""
    d = raw[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
             'oldbalanceDest', 'newbalanceDest']].copy()
    if not pd.api.types.is_numeric_dtype(d['type']):
        d['type'] = (d['type'] == 'TRANSFER').astype(int)
    d['errorBalanceOrig'] = d['newbalanceOrig'] + d['amount'] - d['oldbalanceOrg']
    d['errorBalanceDest'] = d['oldbalanceDest'] + d['amount'] - d['newbalanceDest']
    d['drained'] = ((d['oldbalanceOrg'] > 0) & (d['newbalanceOrig'] == 0)).astype(int)
    d['amount_ratio'] = d['amount'] / (d['oldbalanceOrg'] + 1)
    for c in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        d[c] = np.log1p(d[c])
    for c in ['errorBalanceOrig', 'errorBalanceDest']:
        d[c] = np.sign(d[c]) * np.log1p(np.abs(d[c]))
    d['amount_ratio'] = np.log1p(d['amount_ratio'])
    return d[FEATURES]


def scaled(raw):
    return (build_features(raw).to_numpy(dtype=float) - MEAN) / SCALE


def score(raw: pd.DataFrame, key: str) -> np.ndarray:
    coef, intercept = load_weights(key)
    z = scaled(raw) @ coef + intercept
    return 1 / (1 + np.exp(-z))


def explain(raw_row: pd.DataFrame, key: str) -> pd.DataFrame:
    """How much each feature pushed this transaction towards or away from fraud."""
    coef, _ = load_weights(key)
    contrib = scaled(raw_row)[0] * coef
    out = pd.DataFrame({'Feature': [FEATURE_LABELS.get(f, f) for f in FEATURES],
                        'Effect on score': contrib})
    return out.reindex(out['Effect on score'].abs().sort_values(ascending=False).index)
