"""Turns raw transactions into the Stage 1 features and scores them with a trained model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).parent / 'assets'

PRE = json.loads((ASSETS / 'preprocessing.json').read_text())
FEATURES = PRE['features']
MEAN = np.array(PRE['mean'])
SCALE = np.array(PRE['scale'])
THRESHOLD = PRE['threshold']

RAW_COLUMNS = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
TYPES = ['TRANSFER', 'CASH_OUT']

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


def load_weights(path):
    w = np.load(path)
    return w['coef'].ravel(), float(np.ravel(w['intercept'])[0])


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as Stage 1 (add_features_v2)."""
    d = raw[RAW_COLUMNS].copy()
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


def score(raw: pd.DataFrame, weights_path) -> np.ndarray:
    coef, intercept = load_weights(weights_path)
    z = scaled(raw) @ coef + intercept
    return 1 / (1 + np.exp(-z))


def explain(raw_row: pd.DataFrame, weights_path) -> pd.DataFrame:
    """How much each feature pushed this transaction towards or away from fraud."""
    coef, _ = load_weights(weights_path)
    contrib = scaled(raw_row)[0] * coef
    out = pd.DataFrame({'Feature': [FEATURE_LABELS.get(f, f) for f in FEATURES],
                        'Effect on score': contrib})
    return out.reindex(out['Effect on score'].abs().sort_values().index)


def validate_batch(df: pd.DataFrame):
    """Check an uploaded file before scoring. Returns (clean_df, list_of_errors)."""
    errors = []
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        return None, [f'Missing columns: {", ".join(missing)}']
    df = df[RAW_COLUMNS].copy()
    if len(df) == 0:
        return None, ['The file has no rows.']
    if len(df) > 5000:
        return None, ['Please upload at most 5,000 transactions at a time.']
    df['type'] = df['type'].astype(str).str.strip().str.upper()
    bad_type = ~df['type'].isin(TYPES)
    if bad_type.any():
        errors.append(f'{bad_type.sum()} row(s) have a type other than TRANSFER or CASH_OUT '
                      f'(first at row {int(bad_type.idxmax()) + 2}).')
    for c in RAW_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        bad = df[c].isna() | (df[c] < 0)
        if bad.any():
            errors.append(f'{bad.sum()} row(s) have a missing, non-numeric or negative {c} '
                          f'(first at row {int(bad.idxmax()) + 2}).')
    return (None, errors) if errors else (df, [])


def template_csv() -> bytes:
    example = pd.DataFrame([
        {'type': 'TRANSFER', 'amount': 181000, 'oldbalanceOrg': 181000, 'newbalanceOrig': 0,
         'oldbalanceDest': 0, 'newbalanceDest': 0},
        {'type': 'CASH_OUT', 'amount': 5000, 'oldbalanceOrg': 50000, 'newbalanceOrig': 45000,
         'oldbalanceDest': 100000, 'newbalanceDest': 105000},
    ])
    return example.to_csv(index=False).encode()
