"""FLAegis analyst dashboard. Run with: streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import database as db
import model

BRAND = Path(__file__).parent / 'brand'
APP_NAME = 'FLAegis'
TAGLINE = 'Many banks. One shield. Customer data never leaves the bank.'

st.set_page_config(page_title=f'{APP_NAME} | Analyst Dashboard', page_icon='🛡️', layout='wide')
st.logo(str(BRAND / 'logo.svg'), icon_image=str(BRAND / 'icon.svg'), size='large')
st.markdown("""
<style>
  [data-testid="stMetric"] {background: #F1F5F9; border-left: 4px solid #0F766E;
                            padding: 12px 16px; border-radius: 8px;}
  h1 {color: #0B2545;}
  .fs-footer {color: #64748B; font-size: 0.8rem; margin-top: 2rem;}
</style>
""", unsafe_allow_html=True)

RUNS = ['fedavg_iid', 'fedavgm_iid', 'fedavg_noniid', 'fedavgm_noniid', 'fedavgm05_noniid']


# ---------- load results ----------
@st.cache_data
def load_results():
    final, history = {}, {}
    central_file = model.ASSETS / 'central_metrics.json'
    if central_file.exists():
        final['central'] = json.loads(central_file.read_text())
    for run in RUNS:
        f = model.ASSETS / f'fl_{run}_history.json'
        if f.exists():
            h = json.loads(f.read_text())
            history[run] = pd.DataFrame(h)
            final[run] = h[-1]
    return final, history


final, history = load_results()
models = model.available_models()


def default_model():
    noniid = [r for r in ['fedavg_noniid', 'fedavgm_noniid', 'fedavgm05_noniid'] if r in final and r in models]
    if noniid:
        return max(noniid, key=lambda r: final[r]['auc_pr'])
    return models[0]


if 'model_key' not in st.session_state:
    st.session_state.model_key = default_model()
if db.is_empty():
    db.setup(st.session_state.model_key)

# ---------- sidebar ----------
st.sidebar.caption(TAGLINE)
page = st.sidebar.radio('Go to', ['Overview', 'Participating banks',
                                  'Score a transaction', 'Flagged transactions'])
st.sidebar.divider()
choice = st.sidebar.selectbox('Model in use', models,
                              index=models.index(st.session_state.model_key),
                              format_func=lambda k: model.MODEL_NAMES[k])
if choice != st.session_state.model_key:
    st.session_state.model_key = choice
    db.setup(choice)
    st.sidebar.success('Transactions re-scored with the selected model')
st.sidebar.caption(f'Fraud alert threshold: {model.THRESHOLD:.4f}')


def metric_table():
    rows = []
    for k in ['central'] + RUNS:
        if k in final:
            m = final[k]
            rows.append({'Model': model.MODEL_NAMES[k], 'AUC-PR': m['auc_pr'],
                         'Precision': m['precision'], 'Recall': m['recall'], 'F1': m['f1']})
    return pd.DataFrame(rows)


# ---------- pages ----------
if page == 'Overview':
    st.title('Model performance overview')
    st.write('Each bank trains on its own customers. Only model weights are shared with the '
             'coordination server, never transaction data. All models are tested on the same '
             'held-out test set.')

    if 'central' in final:
        c = final['central']
        best = default_model()
        cols = st.columns(4)
        cols[0].metric('Central AUC-PR', f"{c['auc_pr']:.4f}")
        if best in final and best != 'central':
            b = final[best]
            cols[1].metric(f'Best federated AUC-PR', f"{b['auc_pr']:.4f}",
                           f"{b['auc_pr'] - c['auc_pr']:+.4f} vs central")
            cols[2].metric('Federated precision', f"{b['precision']:.4f}")
            cols[3].metric('Federated recall', f"{b['recall']:.4f}")
            st.caption(f'Best federated model: {model.MODEL_NAMES[best]}')

    st.subheader('All models')
    table = metric_table()
    if len(table):
        st.dataframe(table.style.format({c: '{:.4f}' for c in table.columns if c != 'Model'}),
                     hide_index=True, width='stretch')

    if history:
        st.subheader('AUC-PR across federated rounds')
        curves = pd.DataFrame({model.MODEL_NAMES[r]: h.set_index('round')['auc_pr']
                               for r, h in history.items()})
        curves = curves.drop(index=0, errors='ignore')  # round 0 is the untrained start
        if 'central' in final:
            curves['Central model'] = final['central']['auc_pr']
        st.line_chart(curves)

        st.subheader('Final F1 score by model')
        st.bar_chart(table.set_index('Model')['F1'])
    else:
        st.info('Federated results not loaded yet. Copy the fl_* files into the assets folder.')

elif page == 'Participating banks':
    st.title('Participating banks')
    f = model.ASSETS / 'bank_partitions.csv'
    if f.exists():
        parts = pd.read_csv(f)
        parts['bank'] = parts['bank'].map(lambda b: f'Bank {"ABCDE"[b]}')
        parts['split'] = parts['split'].map({'iid': 'IID (similar banks)',
                                             'noniid': 'Non-IID (different banks)'})
        st.write('How the training data is spread across the 5 institutions. In the non-IID '
                 'setting each bank has a very different size and fraud rate, as real banks do.')
        for split in parts['split'].unique():
            st.subheader(split)
            p = parts[parts['split'] == split]
            c1, c2 = st.columns(2)
            c1.write('Transactions held')
            c1.bar_chart(p.set_index('bank')['rows'])
            c2.write('Fraud rate (%)')
            c2.bar_chart(p.set_index('bank')['fraud_rate_%'])
            st.dataframe(p.rename(columns={'bank': 'Bank', 'rows': 'Transactions',
                                           'frauds': 'Fraud cases',
                                           'fraud_rate_%': 'Fraud rate (%)'}).drop(columns='split'),
                         hide_index=True, width='stretch')
    st.subheader('Live activity in the dashboard')
    st.dataframe(db.counts().rename(columns={'bank': 'Bank', 'transactions': 'Transactions scored',
                                             'flagged': 'Flagged'}),
                 hide_index=True, width='stretch')

elif page == 'Score a transaction':
    st.title('Score a transaction')
    st.write(f'Using: **{model.MODEL_NAMES[st.session_state.model_key]}**')
    with st.form('score'):
        c1, c2 = st.columns(2)
        bank = c1.selectbox('Bank', ['Bank A', 'Bank B', 'Bank C', 'Bank D', 'Bank E'])
        ttype = c2.selectbox('Transaction type', ['TRANSFER', 'CASH_OUT'])
        amount = c1.number_input('Amount', min_value=0.0, value=181000.0, step=1000.0)
        old_org = c1.number_input('Sender balance before', min_value=0.0, value=181000.0, step=1000.0)
        new_org = c1.number_input('Sender balance after', min_value=0.0, value=0.0, step=1000.0)
        old_dest = c2.number_input('Receiver balance before', min_value=0.0, value=0.0, step=1000.0)
        new_dest = c2.number_input('Receiver balance after', min_value=0.0, value=0.0, step=1000.0)
        submitted = st.form_submit_button('Score transaction', type='primary')

    if submitted:
        raw = pd.DataFrame([{'type': ttype, 'amount': amount, 'oldbalanceOrg': old_org,
                             'newbalanceOrig': new_org, 'oldbalanceDest': old_dest,
                             'newbalanceDest': new_dest}])
        key = st.session_state.model_key
        p = float(model.score(raw, key)[0])
        is_flagged = p >= model.THRESHOLD
        txn_id = db.next_txn_id()
        db.add_transaction({'txn_id': txn_id, 'bank': bank, 'type': ttype, 'amount': amount,
                            'oldbalanceOrg': old_org, 'newbalanceOrig': new_org,
                            'oldbalanceDest': old_dest, 'newbalanceDest': new_dest,
                            'fraud_score': p, 'flagged': int(is_flagged), 'model_used': key})
        c1, c2 = st.columns(2)
        c1.metric('Fraud score', f'{p:.4f}')
        if is_flagged:
            c2.error(f'🚩 FLAGGED as likely fraud. Saved as {txn_id} in the review queue.')
        else:
            c2.success(f'✅ Looks legitimate. Saved as {txn_id}.')
        st.subheader('Why this score')
        st.caption('Positive values push towards fraud, negative values push towards legitimate.')
        st.bar_chart(model.explain(raw, key).set_index('Feature'))

elif page == 'Flagged transactions':
    st.title('Flagged transactions')
    df = db.flagged()
    c1, c2, c3 = st.columns(3)
    c1.metric('Flagged', len(df))
    c2.metric('Pending review', int((df['status'] == 'Pending review').sum()))
    c3.metric('Confirmed fraud', int((df['status'] == 'Confirmed fraud').sum()))

    f1, f2 = st.columns(2)
    banks = f1.multiselect('Bank', sorted(df['bank'].unique()))
    statuses = f2.multiselect('Status', ['Pending review', 'Confirmed fraud', 'False alarm'])
    view = df
    if banks:
        view = view[view['bank'].isin(banks)]
    if statuses:
        view = view[view['status'].isin(statuses)]
    st.dataframe(
        view.rename(columns={'txn_id': 'Transaction', 'bank': 'Bank', 'type': 'Type',
                             'amount': 'Amount', 'oldbalanceOrg': 'Sender before',
                             'newbalanceOrig': 'Sender after', 'fraud_score': 'Fraud score',
                             'scored_at': 'Scored at', 'status': 'Status', 'note': 'Note',
                             'reviewed_at': 'Reviewed at'})
        .style.format({'Amount': '{:,.2f}', 'Sender before': '{:,.2f}',
                       'Sender after': '{:,.2f}', 'Fraud score': '{:.4f}'}),
        hide_index=True, width='stretch')

    st.subheader('Review a transaction')
    if len(df):
        with st.form('review'):
            txn = st.selectbox('Transaction', df['txn_id'])
            status = st.radio('Decision', ['Confirmed fraud', 'False alarm'], horizontal=True)
            note = st.text_input('Note (optional)')
            if st.form_submit_button('Save decision', type='primary'):
                db.review(txn, status, note)
                st.success(f'{txn} marked as {status}')
                st.rerun()


st.markdown(f'<div class="fs-footer">{APP_NAME} v1.0 · Privacy-preserving federated fraud detection '
            'across 5 participating institutions</div>', unsafe_allow_html=True)
