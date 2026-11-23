"""Automated unit and integration tests for FLAegis.

Run from the flaegis folder:
    pytest -v                 (all tests, including one real training session)
    pytest -v -m "not slow"   (skip the real training session)
"""
import importlib.util
import io
import json
import time

import numpy as np
import pandas as pd
import pytest

FRAUD = {'type': 'TRANSFER', 'amount': 181000, 'oldbalanceOrg': 181000, 'newbalanceOrig': 0,
         'oldbalanceDest': 0, 'newbalanceDest': 0}
LEGIT = {'type': 'CASH_OUT', 'amount': 5000, 'oldbalanceOrg': 50000, 'newbalanceOrig': 45000,
         'oldbalanceDest': 100000, 'newbalanceDest': 105000}


def score_one(db, model, row):
    return float(model.score(pd.DataFrame([row]), db.weights_for(db.active_model()))[0])


# ======================================================================= unit tests
# ---- feature engineering and scoring (model.py)
def test_tc01_features_match_training_order(model):
    f = model.build_features(pd.DataFrame([FRAUD]))
    assert list(f.columns) == model.FEATURES and f.shape == (1, 10)


def test_tc02_drained_account_detected(model):
    f = model.build_features(pd.DataFrame([FRAUD, LEGIT]))
    assert f['drained'].tolist() == [1, 0]


def test_tc03_balance_mismatch_features(model):
    f = model.build_features(pd.DataFrame([FRAUD, LEGIT]))
    assert f['errorBalanceDest'].iloc[0] > 0          # money sent but receiver balance unchanged
    assert f['errorBalanceOrig'].iloc[1] == 0          # legitimate: sender's balance adds up


def test_tc04_scores_are_probabilities(db, model):
    sample = pd.read_csv(model.ASSETS / 'sample_transactions.csv')
    p = model.score(sample, db.weights_for(db.active_model()))
    assert ((p >= 0) & (p <= 1)).all()


def test_tc05_example_fraud_is_flagged(db, model):
    assert score_one(db, model, FRAUD) >= model.THRESHOLD


def test_tc06_example_legitimate_not_flagged(db, model):
    assert score_one(db, model, LEGIT) < model.THRESHOLD


def test_tc07_explanation_covers_all_features(db, model):
    ex = model.explain(pd.DataFrame([FRAUD]), db.weights_for(db.active_model()))
    assert len(ex) == 10 and ex['Effect on score'].notna().all()


# ---- batch upload validation (model.py)
def test_tc08_template_file_is_valid(model):
    df, errors = model.validate_batch(pd.read_csv(io.BytesIO(model.template_csv())))
    assert errors == [] and len(df) == 2


def test_tc09_missing_column_rejected(model):
    bad = pd.DataFrame([FRAUD]).drop(columns='amount')
    assert 'Missing columns' in model.validate_batch(bad)[1][0]


def test_tc10_unknown_transaction_type_rejected(model):
    assert model.validate_batch(pd.DataFrame([{**FRAUD, 'type': 'PAYMENT'}]))[1]


def test_tc11_negative_or_text_amount_rejected(model):
    assert model.validate_batch(pd.DataFrame([{**FRAUD, 'amount': -5}]))[1]
    assert model.validate_batch(pd.DataFrame([{**FRAUD, 'amount': 'abc'}]))[1]


def test_tc12_more_than_5000_rows_rejected(model):
    assert model.validate_batch(pd.DataFrame([FRAUD] * 5001))[1]


def test_tc13_lowercase_type_accepted(model):
    df, errors = model.validate_batch(pd.DataFrame([{**FRAUD, 'type': 'transfer'}]))
    assert errors == [] and df['type'].iloc[0] == 'TRANSFER'


# ---- passwords (database.py)
def test_tc14_passwords_stored_as_bcrypt_hashes(db):
    stored = db.query("SELECT password_hash FROM users WHERE username = 'admin'")['password_hash'][0]
    assert stored != 'Admin@123' and stored.startswith('$2')


def test_tc15_password_check(db):
    h = db.hash_password('Secret123')
    assert db.check_password('Secret123', h) and not db.check_password('wrong', h)


# ======================================================================= integration tests
# ---- setup and accounts
def test_tc16_first_run_creates_platform(db):
    assert len(db.list_institutions()) == 5
    users = db.list_users()
    assert (users['role'] == 'admin').sum() == 1 and (users['role'] == 'analyst').sum() == 5
    sessions = db.list_sessions()
    assert len(sessions) >= 5 and (sessions['status'] == 'Completed').all()


def test_tc17_login_with_correct_details(db):
    u = db.authenticate('analyst_a', 'BankA@123')
    assert u['role'] == 'analyst' and u['institution'] == 'Bank A' and 'password_hash' not in u


def test_tc18_login_with_wrong_password_or_unknown_user(db):
    assert db.authenticate('admin', 'wrong') is None
    assert db.authenticate('nobody', 'Admin@123') is None


def test_tc19_disabled_account_cannot_log_in(db):
    uid = db.authenticate('analyst_b', 'BankB@123')['id']
    db.set_user_active(uid, False)
    assert db.authenticate('analyst_b', 'BankB@123') is None


def test_tc20_change_password(db):
    uid = db.authenticate('admin', 'Admin@123')['id']
    db.change_password(uid, 'NewPass123')
    assert db.authenticate('admin', 'NewPass123') and db.authenticate('admin', 'Admin@123') is None


def test_tc21_register_institution_and_analyst(db):
    iid = db.create_institution('Bank F')
    db.create_user('analyst_f', 'Bank F Analyst', 'BankF@1234', 'analyst', iid)
    assert db.authenticate('analyst_f', 'BankF@1234')['institution'] == 'Bank F'


# ---- scoring and review
def test_tc22_scored_batch_saved_and_flagged_in_queue(db, model):
    a = db.authenticate('analyst_a', 'BankA@123')
    df, _ = model.validate_batch(pd.DataFrame([FRAUD, LEGIT]))
    df['fraud_score'] = model.score(df, db.weights_for(db.active_model()))
    df['flagged'] = (df['fraud_score'] >= model.THRESHOLD).astype(int)
    ids = db.add_transactions(df, a['institution_id'], db.active_model(), a['id'])
    queue = db.flagged_for(a['institution_id'])['txn_id'].tolist()
    assert ids[0] in queue and ids[1] not in queue


def test_tc23_analyst_sees_only_own_institution(db):
    a = db.authenticate('analyst_a', 'BankA@123')
    b = db.authenticate('analyst_b', 'BankB@123')
    qa = set(db.flagged_for(a['institution_id'])['txn_id'])
    qb = set(db.flagged_for(b['institution_id'])['txn_id'])
    assert qa and qb and not (qa & qb)


def test_tc24_cannot_review_another_banks_transaction(db):
    a = db.authenticate('analyst_a', 'BankA@123')
    b = db.authenticate('analyst_b', 'BankB@123')
    other = db.flagged_for(b['institution_id'])['txn_id'].iloc[0]
    with pytest.raises(PermissionError):
        db.review(other, 'Confirmed fraud', '', a['id'], a['institution_id'])


def test_tc25_review_decision_saved(db):
    a = db.authenticate('analyst_a', 'BankA@123')
    txn = db.flagged_for(a['institution_id'])['txn_id'].iloc[0]
    db.review(txn, 'False alarm', 'checked with customer', a['id'], a['institution_id'])
    row = db.flagged_for(a['institution_id']).set_index('txn_id').loc[txn]
    assert row['status'] == 'False alarm' and row['reviewed_by'] == 'analyst_a'


def test_tc26_my_institution_figures(db):
    a = db.authenticate('analyst_a', 'BankA@123')
    s = db.own_summary(a['institution_id'])
    assert s['flagged'] == len(db.flagged_for(a['institution_id'])) and s['scored'] >= s['flagged']


def test_tc27_other_banks_figures_hidden_from_analysts(db):
    cols = set(db.consortium_members().columns)
    assert not cols & {'transactions', 'flagged', 'scored', 'confirmed', 'false_alarms'}


# ---- training sessions
def test_tc28_new_session_is_in_progress(db):
    admin = db.authenticate('admin', 'Admin@123')
    sid = db.create_session('Test', 'FedAvgM', 'Non-IID', 5, 0.5, admin['id'])
    status = db.list_sessions().set_index('id').loc[sid, 'status']
    assert status in db.RUNNING_STATUSES


def test_tc29_cancelled_session_can_be_deleted(db):
    admin = db.authenticate('admin', 'Admin@123')
    sid = db.create_session('Mistake', 'FedAvg', 'IID', 5, None, admin['id'])
    db.cancel_session(sid)
    db.delete_session(sid)
    assert sid not in db.list_sessions()['id'].tolist()


def test_tc30_completed_session_cannot_be_deleted(db):
    with pytest.raises(ValueError):
        db.delete_session(1)


def test_tc31_archive_hides_and_restore_returns(db):
    sid = int(db.list_sessions().query("status == 'Completed'")['id'].iloc[0])
    if db.active_model() == f'session:{sid}':
        db.set_setting('active_model', 'central')
    db.archive_session(sid)
    assert f'session:{sid}' not in db.model_options()
    assert sid not in db.round_results()['session_id'].tolist()
    db.archive_session(sid, archived=False)
    assert f'session:{sid}' in db.model_options()


def test_tc32_model_in_use_cannot_be_archived(db):
    sid = int(db.active_model().split(':')[1])
    with pytest.raises(ValueError):
        db.archive_session(sid)


def test_tc33_uploaded_results_complete_session(db, model):
    admin = db.authenticate('admin', 'Admin@123')
    sid = db.create_session('Uploaded', 'FedAvg', 'IID', 2, None, admin['id'])
    history = [{'round': r, 'auc_pr': 0.9, 'precision': 0.9, 'recall': 0.9, 'f1': 0.9} for r in range(3)]
    w = np.load(model.ASSETS / 'central_weights.npz')
    fname = f'test_upload_{int(time.time() * 1000)}_weights.npz'
    np.savez(model.ASSETS / fname, coef=w['coef'], intercept=w['intercept'])
    try:
        db.import_results(sid, history, fname)
        assert db.list_sessions().set_index('id').loc[sid, 'status'] == 'Completed'
        assert f'session:{sid}' in db.model_options()
    finally:
        (model.ASSETS / fname).unlink(missing_ok=True)


def test_tc34_switching_model_in_use(db):
    db.set_setting('active_model', 'central')
    assert db.active_model() == 'central'
    assert db.weights_for('central').name == 'central_weights.npz'


# ---- dashboard pages (every page loads for each role without errors)
@pytest.mark.parametrize('user,password', [('admin', 'Admin@123'), ('analyst_a', 'BankA@123')])
def test_tc35_all_pages_load(db, user, password):
    from streamlit.testing.v1 import AppTest
    import database
    at = AppTest.from_file(str(database.Path(database.__file__).parent / 'app.py'),
                           default_timeout=120).run()
    at.text_input[0].input(user)
    at.text_input[1].input(password)
    at.button[0].click().run()
    pages = at.sidebar.radio[0].options
    for page in pages:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f'{page}: {at.exception}'


# ---- live federated training (needs training_data/stage1_data.npz and Flower installed)
from pathlib import Path
live = pytest.mark.skipif(
    importlib.util.find_spec('flwr') is None or
    not (Path(__file__).resolve().parent.parent / 'training_data' / 'stage1_data.npz').exists(),
    reason='live training needs training_data/stage1_data.npz and requirements-training.txt')


@pytest.mark.slow
@live
def test_tc36_live_training_session_completes(db):
    admin = db.authenticate('admin', 'Admin@123')
    sid = db.create_session('Automated test', 'FedAvg', 'Non-IID', 2, None, admin['id'])
    db.start_live_session(sid)
    deadline = time.time() + 900
    while time.time() < deadline:
        status = db.list_sessions().set_index('id').loc[sid, 'status']
        if status != 'Training':
            break
        time.sleep(3)
    try:
        assert status == 'Completed'
        assert db.latest_round(sid)['round'].tolist() == [0, 1, 2]
        assert f'session:{sid}' in db.model_options()
    finally:
        wf = db.list_sessions().set_index('id').loc[sid, 'weights_file']
        if isinstance(wf, str):
            (db.Path(db.__file__).parent / 'assets' / wf).unlink(missing_ok=True)
