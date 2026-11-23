"""FLAegis analyst dashboard. Run with: streamlit run app.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import database as db
import model

BRAND = Path(__file__).parent / 'brand'
APP_NAME = 'FLAegis'
TAGLINE = 'Many banks. One shield. Customer data never leaves the bank.'
NAVY, TEAL, MINT = '#0B2545', '#0F766E', '#2DD4BF'
PALETTE = [TEAL, NAVY, '#F59E0B', '#8B5CF6', '#EF4444', '#64748B']

st.set_page_config(page_title=f'{APP_NAME} | Fraud Detection Platform', page_icon='🛡️', layout='wide')
st.logo(str(BRAND / 'logo.svg'), icon_image=str(BRAND / 'icon.svg'), size='large')
st.markdown("""
<style>
  [data-testid="stMetric"] {background: #F1F5F9; border-left: 4px solid #0F766E;
                            padding: 12px 16px; border-radius: 8px;}
  h1, h2, h3 {color: #0B2545;}
  .fs-footer {color: #64748B; font-size: 0.8rem; margin-top: 2rem;}
  .fs-badge {display: inline-block; background: #CCFBF1; color: #0F766E; padding: 2px 10px;
             border-radius: 12px; font-size: 0.8rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

db.initialise()


def flash(msg):
    st.session_state['flash'] = msg


def show_flash():
    msg = st.session_state.pop('flash', None)
    if msg:
        st.toast(msg, icon='✅')
        st.success(msg)


def footer():
    st.markdown(f'<div class="fs-footer">{APP_NAME} v1.0 · Privacy-preserving federated fraud '
                'detection across participating institutions</div>', unsafe_allow_html=True)


# ====================================================================== login
def login_page():
    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.image(str(BRAND / 'logo.svg'), width=260)
        st.caption(TAGLINE)
        with st.form('login'):
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            if st.form_submit_button('Sign in', type='primary', width='stretch'):
                user = db.authenticate(username, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error('Incorrect username or password, or the account is disabled.')
        st.caption('Access is restricted to registered administrators and institution analysts.')
    footer()


if 'user' not in st.session_state:
    login_page()
    st.stop()

user = st.session_state.user
is_admin = user['role'] == 'admin'


# ====================================================================== helpers
def fmt4(df, cols):
    return df.style.format({c: '{:.4f}' for c in cols if c in df.columns})


def model_name(key):
    return db.model_options().get(key, key)


def load_central_metrics():
    f = model.ASSETS / 'central_metrics.json'
    return json.loads(f.read_text()) if f.exists() else None


def load_local_only():
    f = model.ASSETS / 'local_only_results.csv'
    return pd.read_csv(f) if f.exists() else None


def results_table():
    rows = []
    central = load_central_metrics()
    if central:
        rows.append({'Model': 'Central reference (data pooled)', 'Type': 'Central',
                     'AUC-PR': central['auc_pr'], 'Precision': central['precision'],
                     'Recall': central['recall'], 'F1': central['f1']})
    rr = db.round_results()
    for sid, g in rr.groupby('session_id'):
        last = g.sort_values('round').iloc[-1]
        rows.append({'Model': f"Session {sid}: {last['name']}", 'Type': 'Federated',
                     'AUC-PR': last['auc_pr'], 'Precision': last['precision'],
                     'Recall': last['recall'], 'F1': last['f1']})
    local = load_local_only()
    if local is not None:
        for split, g in local.groupby('split'):
            label = 'IID' if split == 'iid' else 'non-IID'
            rows.append({'Model': f'Local-only average, {label} banks', 'Type': 'Local-only',
                         'AUC-PR': g['auc_pr'].mean(), 'Precision': g['precision'].mean(),
                         'Recall': g['recall'].mean(), 'F1': g['f1'].mean()})
    return pd.DataFrame(rows)


def style_fig(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor='white', legend=dict(orientation='h', y=-0.25),
                      font=dict(family='sans-serif', color=NAVY))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#E2E8F0')
    return fig


# ====================================================================== pages
def page_overview():
    st.title('Model performance overview')
    st.write('Each institution trains on its own customers. Only model weights are shared with the '
             'coordination server, never transaction data. All models are tested on the same '
             'held-out test set.')

    table = results_table()
    central = load_central_metrics()
    active = db.active_model()
    fed = table[table['Type'] == 'Federated']
    cols = st.columns(4)
    if central:
        cols[0].metric('Central AUC-PR', f"{central['auc_pr']:.4f}")
    if len(fed):
        best = fed.sort_values('AUC-PR', ascending=False).iloc[0]
        cols[1].metric('Best federated AUC-PR', f"{best['AUC-PR']:.4f}",
                       f"{best['AUC-PR'] - central['auc_pr']:+.4f} vs central" if central else None)
    local = table[table['Model'] == 'Local-only average, non-IID banks']
    if len(local):
        cols[2].metric('Banks alone (avg AUC-PR)', f"{local['AUC-PR'].iloc[0]:.4f}",
                       help='Average AUC-PR when each bank trains only on its own data')
    cols[3].metric('Training sessions', len(db.list_sessions()))
    st.markdown(f'Model in use for scoring: <span class="fs-badge">{model_name(active)}</span>',
                unsafe_allow_html=True)

    st.subheader('Final results by model')
    st.dataframe(fmt4(table, ['AUC-PR', 'Precision', 'Recall', 'F1']),
                 hide_index=True, width='stretch')

    rr = db.round_results()
    if len(rr):
        st.subheader('Training progress across rounds')
        c1, c2 = st.columns([1, 3])
        metric = c1.selectbox('Metric', ['auc_pr', 'precision', 'recall', 'f1'],
                              format_func=lambda m: {'auc_pr': 'AUC-PR', 'precision': 'Precision',
                                                     'recall': 'Recall', 'f1': 'F1'}[m])
        names = rr.drop_duplicates('session_id')
        options = {int(r.session_id): f'Session {r.session_id}: {r.name}' for r in names.itertuples()}
        chosen = c2.multiselect('Sessions', list(options), default=list(options),
                                format_func=lambda s: options[s])
        data = rr[(rr['session_id'].isin(chosen)) & (rr['round'] > 0)].copy()
        data['Session'] = data['session_id'].map(options)
        fig = px.line(data, x='round', y=metric, color='Session', markers=True,
                      color_discrete_sequence=PALETTE,
                      labels={'round': 'Round', metric: metric.upper().replace('_', '-')})
        if central:
            fig.add_hline(y=central[metric], line_dash='dash', line_color='#94A3B8',
                          annotation_text='Central reference', annotation_position='bottom right')
        st.plotly_chart(style_fig(fig, 420), width='stretch')
        st.caption('Round 0 (the untrained starting model) is hidden.')

    local_df = load_local_only()
    if local_df is not None:
        st.subheader('Each bank alone vs the federation')
        split = st.radio('Data setting', ['noniid', 'iid'], horizontal=True,
                         format_func=lambda s: 'Non-IID banks' if s == 'noniid' else 'IID banks')
        l = local_df[local_df['split'] == split].copy()
        l['Bank'] = l['bank'].map(lambda b: f'Bank {"ABCDE"[b]}')
        fig = px.bar(l, x='Bank', y='auc_pr', text=l['auc_pr'].map('{:.4f}'.format),
                     color_discrete_sequence=['#94A3B8'], labels={'auc_pr': 'AUC-PR'})
        sessions = db.list_sessions()
        fedavg = sessions[(sessions['strategy'] == 'FedAvg') &
                          (sessions['data_split'] == ('Non-IID' if split == 'noniid' else 'IID'))]
        if len(fedavg) and pd.notna(fedavg.iloc[0]['final_auc_pr']):
            fig.add_hline(y=fedavg.iloc[0]['final_auc_pr'], line_color=TEAL, line_width=3,
                          annotation_text='Federated model (FedAvg)', annotation_position='top left')
        fig.update_yaxes(range=[max(0, l['auc_pr'].min() - 0.05), 1.0])
        st.plotly_chart(style_fig(fig), width='stretch')
        st.caption('Grey bars: a model trained by each bank on its own data only. '
                   'Teal line: the federated model trained across all banks.')

    ft_local = model.ASSETS / 'fraudtype_local_only.csv'
    if ft_local.exists():
        st.subheader('When banks see different kinds of fraud')
        st.write('Banks A and B only ever saw TRANSFER fraud; Banks C, D and E only saw CASH_OUT '
                 'fraud. Every model is tested on both kinds.')
        rows = []
        for r in pd.read_csv(ft_local).itertuples():
            rows.append({'Model': f'{r.bank} alone ({r.fraud_type_seen} fraud only)',
                         'AUC-PR': r.auc_pr, 'F1': r.f1,
                         'Caught TRANSFER fraud': r.recall_transfer_fraud,
                         'Caught CASH_OUT fraud': r.recall_cashout_fraud})
        for key, label in [('fedavg_fraudtype', 'Federated: FedAvg'),
                           ('fedavgm_fraudtype', 'Federated: FedAvgM (momentum 0.9)'),
                           ('fedavgm05_fraudtype', 'Federated: FedAvgM (momentum 0.5)')]:
            f = model.ASSETS / f'fl_{key}_history.json'
            if f.exists():
                h = json.loads(f.read_text())[-1]
                rows.append({'Model': label, 'AUC-PR': h['auc_pr'], 'F1': h['f1'],
                             'Caught TRANSFER fraud': h.get('recall_transfer_fraud'),
                             'Caught CASH_OUT fraud': h.get('recall_cashout_fraud')})
        ft = pd.DataFrame(rows)
        long = ft.melt(id_vars='Model', value_vars=['Caught TRANSFER fraud', 'Caught CASH_OUT fraud'],
                       var_name='Fraud type', value_name='Share caught')
        fig = px.bar(long, x='Share caught', y='Model', color='Fraud type', barmode='group',
                     orientation='h', color_discrete_sequence=[NAVY, TEAL])
        fig.update_xaxes(range=[0, 1], tickformat='.0%')
        st.plotly_chart(style_fig(fig, 460), width='stretch')
        st.dataframe(fmt4(ft, ['AUC-PR', 'F1', 'Caught TRANSFER fraud', 'Caught CASH_OUT fraud']),
                     hide_index=True, width='stretch')
        st.caption(f'Share of each fraud type caught at the {model.THRESHOLD:.4f} threshold.')


def page_banks():
    st.title('Participating institutions')
    f = model.ASSETS / 'bank_partitions.csv'
    if f.exists():
        parts = pd.read_csv(f)
        parts['Bank'] = parts['bank'].map(lambda b: f'Bank {"ABCDE"[b]}')
        st.write('How the training data is spread across the institutions. In the non-IID setting each '
                 'bank has a very different size and fraud rate, as real banks do.')
        split = st.radio('Data setting', ['noniid', 'iid'], horizontal=True,
                         format_func=lambda s: 'Non-IID (different banks)' if s == 'noniid'
                         else 'IID (similar banks)')
        p = parts[parts['split'] == split]
        c1, c2 = st.columns(2)
        c1.plotly_chart(style_fig(px.bar(p, x='Bank', y='rows', title='Transactions held',
                                         color_discrete_sequence=[NAVY]), 320), width='stretch')
        c2.plotly_chart(style_fig(px.bar(p, x='Bank', y='fraud_rate_%', title='Fraud rate (%)',
                                         color_discrete_sequence=[TEAL]), 320), width='stretch')
        st.dataframe(p[['Bank', 'rows', 'frauds', 'fraud_rate_%']].rename(
            columns={'rows': 'Transactions', 'frauds': 'Fraud cases', 'fraud_rate_%': 'Fraud rate (%)'}),
            hide_index=True, width='stretch')
    st.subheader('Platform activity (counts only)')
    st.dataframe(db.institution_summary().rename(columns={
        'institution': 'Institution', 'transactions_scored': 'Transactions scored',
        'flagged': 'Flagged', 'confirmed_fraud': 'Confirmed fraud', 'false_alarms': 'False alarms'}),
        hide_index=True, width='stretch')


def page_my_institution():
    st.title(f"My institution: {user['institution']}")
    s = db.own_summary(int(user['institution_id']))
    c = st.columns(5)
    c[0].metric('Transactions scored', f"{s['scored']:,}")
    c[1].metric('Flagged', f"{s['flagged']:,}")
    c[2].metric('Pending review', f"{s['pending']:,}")
    c[3].metric('Confirmed fraud', f"{s['confirmed']:,}")
    c[4].metric('False alarms', f"{s['false_alarms']:,}")
    reviewed = s['confirmed'] + s['false_alarms']
    if reviewed:
        st.caption(f"Of the flagged transactions you have reviewed, "
                   f"{s['confirmed'] / reviewed:.0%} were confirmed as fraud.")

    st.subheader('Consortium members')
    st.write('The institutions that train the shared fraud model with you. Other members\' '
             'transaction volumes and fraud figures are private to each institution.')
    m = db.consortium_members()
    m['You'] = m['name'].map(lambda n: '✓' if n == user['institution'] else '')
    st.dataframe(m[['name', 'You', 'created_at', 'participation']].rename(columns={
        'name': 'Institution', 'created_at': 'Joined', 'participation': 'Training participation'}),
        hide_index=True, width='stretch')


@st.fragment(run_every=2)
def live_progress(sid, name, rounds):
    """Refreshes every 2 seconds while a session trains on this machine."""
    status = db.query('SELECT status FROM training_sessions WHERE id = ?', (sid,))['status'].iloc[0]
    if status != 'Training':
        st.rerun()           # finished, failed or cancelled: redraw the whole page
    rr = db.latest_round(sid)
    done = int(rr['round'].max()) if len(rr) else 0
    st.info(f"🔄 **Session {sid} ({name}) is training on this machine.** "
            f"{db.get_setting(f'progress:{sid}') or 'Starting'}")
    st.progress(min(done / rounds, 1.0), text=f'Round {done} of {rounds}')
    if len(rr) > 1:
        last = rr.iloc[-1]
        c = st.columns(4)
        c[0].metric('AUC-PR', f"{last['auc_pr']:.4f}")
        c[1].metric('Precision', f"{last['precision']:.4f}")
        c[2].metric('Recall', f"{last['recall']:.4f}")
        c[3].metric('F1', f"{last['f1']:.4f}")
        fig = px.line(rr[rr['round'] > 0], x='round', y='auc_pr', markers=True,
                      color_discrete_sequence=[TEAL], labels={'round': 'Round', 'auc_pr': 'AUC-PR'})
        fig.update_xaxes(range=[0.5, rounds + 0.5])
        st.plotly_chart(style_fig(fig, 280), width='stretch')


def page_sessions():
    st.title('Training sessions')
    st.write('A training session runs the federated rounds on the training server. The coordination '
             'server sends the global model to each institution, each institution trains locally, '
             'and only the updated weights come back.')

    db.check_stalled()
    sessions = db.list_sessions()
    show_archived = st.toggle('Show archived sessions', value=False)
    visible = sessions if show_archived else sessions[sessions['status'] != 'Archived']
    show = visible.drop(columns=['weights_file']).rename(columns={
        'id': 'ID', 'name': 'Name', 'strategy': 'Strategy', 'data_split': 'Data split',
        'rounds': 'Rounds', 'server_momentum': 'Momentum', 'status': 'Status',
        'created_by': 'Created by', 'created_at': 'Created', 'completed_at': 'Completed',
        'final_auc_pr': 'Final AUC-PR', 'final_f1': 'Final F1'})
    st.dataframe(fmt4(show, ['Final AUC-PR', 'Final F1']), hide_index=True, width='stretch')

    st.subheader('Model in use for scoring')
    options = db.model_options()
    current = db.active_model()
    choice = st.selectbox('Model that analysts use to score transactions', list(options),
                          index=list(options).index(current), format_func=lambda k: options[k])
    if st.button('Set as model in use', type='primary', disabled=choice == current):
        db.set_setting('active_model', choice)
        flash(f'Analysts now score with: {options[choice]}')
        st.rerun()

    db.check_stalled()
    live = db.live_training_available()
    sessions = db.list_sessions()
    running = sessions[sessions['status'].isin(db.RUNNING_STATUSES)]
    st.subheader('Training status')
    st.caption('Training server: ' + ('**this machine** (sessions start and run automatically)' if live
               else '**external** (run the session on the training server, then upload its results)'))
    if len(running):
        r = running.iloc[0]
        if r['status'] == 'Training':
            live_progress(int(r['id']), r['name'], int(r['rounds']))
        else:
            st.warning(f"⏳ **Session {r['id']} ({r['name']}) is in progress** on the training server "
                       f"since {r['created_at']}. A new session can be created once its results are "
                       'uploaded or it is cancelled.')
        if st.button(f"Cancel session {r['id']}"):
            db.cancel_session(int(r['id']))
            flash(f"Session {r['id']} was cancelled.")
            st.rerun()
    else:
        st.success('✅ No training session is running. You can create a new one.')

    t1, t2, t3 = st.tabs(['Create a new session', 'Upload results from the training server',
                          'Archive or delete sessions'])
    with t1:
        busy = len(running) > 0
        with st.form('new_session', clear_on_submit=True):
            c1, c2 = st.columns(2)
            name = c1.text_input('Session name', placeholder='e.g. FedAvgM non-IID, 30 rounds',
                                 disabled=busy)
            strategy = c2.selectbox('Aggregation strategy', ['FedAvg', 'FedAvgM'], disabled=busy)
            split = c1.selectbox('Data split', ['Non-IID', 'IID', 'Fraud-type skew'], disabled=busy)
            rounds = c2.number_input('Number of rounds', min_value=1, max_value=100, value=20,
                                     disabled=busy)
            momentum = c1.slider('Server momentum (FedAvgM only)', 0.0, 0.99, 0.5, 0.05, disabled=busy)
            label = 'Training in progress, please wait' if busy else 'Create session'
            if st.form_submit_button(label, type='primary', disabled=busy):
                if not name.strip():
                    st.error('Please give the session a name.')
                elif db.list_sessions()['status'].isin(db.RUNNING_STATUSES).any():
                    st.error('A training session is already in progress.')
                else:
                    sid = db.create_session(name.strip(), strategy, split, int(rounds),
                                            momentum if strategy == 'FedAvgM' else None, user['id'])
                    if live:
                        db.start_live_session(sid)
                        flash(f'Session {sid} started. The coordination server is sending the model '
                              'to the 5 institutions. Progress appears below as each round finishes.')
                    else:
                        flash(f'Session {sid} created and marked as in progress. Run it on the training '
                              'server with these settings, then upload its results in the next tab.')
                    st.rerun()
    with t2:
        manual = running[running['status'] != 'Training']
        if len(manual) == 0:
            st.info('No session is waiting for uploaded results. '
                    + ('Sessions on this machine finish automatically.' if live else ''))
        else:
            with st.form('upload_results'):
                sid = st.selectbox('Session', manual['id'].tolist(),
                                   format_func=lambda s: f"Session {s}: "
                                   f"{manual.set_index('id').loc[s, 'name']}")
                hist_file = st.file_uploader('Round history (.json)', type=['json'])
                w_file = st.file_uploader('Final model weights (.npz)', type=['npz'])
                if st.form_submit_button('Upload results', type='primary'):
                    try:
                        history = json.load(hist_file)
                        assert isinstance(history, list) and all(
                            {'round', 'auc_pr', 'precision', 'recall', 'f1'} <= set(h) for h in history)
                        w = np.load(w_file)
                        assert w['coef'].size == len(model.FEATURES)
                        fname = f'session_{sid}_weights.npz'
                        np.savez(model.ASSETS / fname, coef=w['coef'], intercept=w['intercept'])
                        db.import_results(int(sid), history, fname)
                        flash(f'Results loaded for session {sid}. Training is complete.')
                        st.rerun()
                    except Exception:
                        st.error('These files are not valid session results. Upload the history '
                                 '.json and weights .npz saved by the training server.')
    with t3:
        st.caption('Completed sessions are kept as an audit trail of how each model was trained, so '
                   'they can be archived (hidden) but not deleted. Cancelled sessions never produced '
                   'a model and can be deleted permanently.')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Archive or restore a completed session**')
            arch = sessions[sessions['status'].isin(['Completed', 'Archived'])]
            if len(arch):
                sid = st.selectbox('Session', arch['id'].tolist(), key='arch_sel',
                                   format_func=lambda i: f"Session {i}: {arch.set_index('id').loc[i, 'name']}"
                                   f" ({arch.set_index('id').loc[i, 'status']})")
                is_arch = arch.set_index('id').loc[sid, 'status'] == 'Archived'
                if st.button('Restore session' if is_arch else 'Archive session'):
                    try:
                        db.archive_session(int(sid), archived=not is_arch)
                        flash(f"Session {sid} {'restored' if is_arch else 'archived'}.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
        with c2:
            st.markdown('**Delete a cancelled or failed session**')
            canc = sessions[sessions['status'].isin(['Cancelled', 'Failed'])]
            if len(canc) == 0:
                st.info('There are no cancelled or failed sessions.')
            else:
                sid = st.selectbox('Session', canc['id'].tolist(), key='del_sel',
                                   format_func=lambda i: f"Session {i}: {canc.set_index('id').loc[i, 'name']}")
                sure = st.checkbox('I understand this cannot be undone')
                if st.button('Delete session', disabled=not sure):
                    db.delete_session(int(sid))
                    flash(f'Session {sid} deleted.')
                    st.rerun()

def page_institutions():
    st.title('Institutions')
    st.dataframe(db.list_institutions().rename(columns={
        'id': 'ID', 'name': 'Institution', 'created_at': 'Registered', 'analysts': 'Analysts',
        'transactions': 'Transactions scored', 'flagged': 'Flagged'}),
        hide_index=True, width='stretch')
    st.subheader('Register an institution')
    with st.form('new_inst', clear_on_submit=True):
        name = st.text_input('Institution name')
        if st.form_submit_button('Register', type='primary'):
            if not name.strip():
                st.error('Please enter a name.')
            elif db.institution_id(name.strip()) is not None:
                st.error('An institution with this name already exists.')
            else:
                db.create_institution(name)
                st.success(f'{name} registered. You can now create analyst accounts for it.')
                st.rerun()


def page_users():
    st.title('User accounts')
    users = db.list_users()
    users['active'] = users['active'].map({1: 'Active', 0: 'Disabled'})
    st.dataframe(users.rename(columns={
        'id': 'ID', 'username': 'Username', 'full_name': 'Name', 'role': 'Role',
        'institution': 'Institution', 'active': 'Status', 'created_at': 'Created'}),
        hide_index=True, width='stretch')

    t1, t2 = st.tabs(['Create analyst account', 'Manage an account'])
    with t1:
        insts = db.list_institutions()
        with st.form('new_user', clear_on_submit=True):
            c1, c2 = st.columns(2)
            username = c1.text_input('Username')
            full_name = c2.text_input('Full name')
            inst = c1.selectbox('Institution', insts['id'].tolist(),
                                format_func=lambda i: insts.set_index('id').loc[i, 'name'])
            pw = c2.text_input('Temporary password (at least 8 characters)', type='password')
            if st.form_submit_button('Create account', type='primary'):
                if not username.strip() or not full_name.strip():
                    st.error('Username and name are required.')
                elif len(pw) < 8:
                    st.error('The password must be at least 8 characters.')
                elif username.strip().lower() in users['username'].tolist():
                    st.error('That username is already taken.')
                else:
                    db.create_user(username, full_name, pw, 'analyst', int(inst))
                    st.success(f'Analyst account {username.strip().lower()} created.')
                    st.rerun()
    with t2:
        others = users[users['id'] != user['id']]
        uid = st.selectbox('Account', others['id'].tolist(),
                           format_func=lambda i: others.set_index('id').loc[i, 'username'])
        c1, c2 = st.columns(2)
        status = others.set_index('id').loc[uid, 'active']
        if c1.button('Disable account' if status == 'Active' else 'Enable account'):
            db.set_user_active(int(uid), status != 'Active')
            st.rerun()
        with c2.form('reset_pw', clear_on_submit=True):
            new_pw = st.text_input('New password', type='password')
            if st.form_submit_button('Reset password'):
                if len(new_pw) < 8:
                    st.error('The password must be at least 8 characters.')
                else:
                    db.change_password(int(uid), new_pw)
                    st.success('Password reset.')


def page_score():
    st.title('Score transactions')
    key = db.active_model()
    weights = db.weights_for(key)
    st.markdown(f'Institution: **{user["institution"]}** · Model in use: '
                f'<span class="fs-badge">{model_name(key)}</span>', unsafe_allow_html=True)

    t1, t2 = st.tabs(['Single transaction', 'Batch upload'])
    with t1:
        with st.form('score'):
            c1, c2 = st.columns(2)
            ttype = c1.selectbox('Transaction type', model.TYPES)
            amount = c2.number_input('Amount', min_value=0.0, value=181000.0, step=1000.0)
            old_org = c1.number_input('Sender balance before', min_value=0.0, value=181000.0, step=1000.0)
            new_org = c1.number_input('Sender balance after', min_value=0.0, value=0.0, step=1000.0)
            old_dest = c2.number_input('Receiver balance before', min_value=0.0, value=0.0, step=1000.0)
            new_dest = c2.number_input('Receiver balance after', min_value=0.0, value=0.0, step=1000.0)
            submitted = st.form_submit_button('Score transaction', type='primary')
        if submitted:
            raw = pd.DataFrame([{'type': ttype, 'amount': amount, 'oldbalanceOrg': old_org,
                                 'newbalanceOrig': new_org, 'oldbalanceDest': old_dest,
                                 'newbalanceDest': new_dest}])
            p = float(model.score(raw, weights)[0])
            raw['fraud_score'], raw['flagged'] = p, int(p >= model.THRESHOLD)
            txn = db.add_transactions(raw, int(user['institution_id']), key, int(user['id']))[0]
            c1, c2 = st.columns(2)
            c1.metric('Fraud score', f'{p:.4f}', help=f'Flagged at {model.THRESHOLD:.4f} or above')
            if p >= model.THRESHOLD:
                c2.error(f'🚩 Flagged as likely fraud. Saved as {txn} in your review queue.')
            else:
                c2.success(f'✅ Looks legitimate. Saved as {txn}.')
            st.subheader('Why this score')
            ex = model.explain(raw, weights)
            fig = go.Figure(go.Bar(x=ex['Effect on score'], y=ex['Feature'], orientation='h',
                                   marker_color=[TEAL if v < 0 else '#EF4444'
                                                 for v in ex['Effect on score']]))
            st.plotly_chart(style_fig(fig, 360), width='stretch')
            st.caption('Red bars push the score towards fraud, teal bars push it towards legitimate.')
    with t2:
        st.write('Upload a CSV file with the columns: ' + ', '.join(f'`{c}`' for c in model.RAW_COLUMNS)
                 + '. Type must be TRANSFER or CASH_OUT. Up to 5,000 rows.')
        st.download_button('Download template', model.template_csv(), 'flaegis_template.csv', 'text/csv')
        up = st.file_uploader('Transactions file (.csv)', type=['csv'])
        if up is not None and st.button('Score file', type='primary'):
            try:
                df, errors = model.validate_batch(pd.read_csv(up))
            except Exception:
                df, errors = None, ['The file could not be read as a CSV.']
            if errors:
                for e in errors:
                    st.error(e)
            else:
                df['fraud_score'] = model.score(df, weights)
                df['flagged'] = (df['fraud_score'] >= model.THRESHOLD).astype(int)
                df.insert(0, 'txn_id', db.add_transactions(df, int(user['institution_id']), key,
                                                           int(user['id'])))
                c1, c2 = st.columns(2)
                c1.metric('Transactions scored', len(df))
                c2.metric('Flagged', int(df['flagged'].sum()))
                st.dataframe(df.style.format({'fraud_score': '{:.4f}'}), hide_index=True,
                             width='stretch')
                st.download_button('Download results', df.to_csv(index=False).encode(),
                                   'flaegis_results.csv', 'text/csv')
                st.caption('Flagged transactions have been added to your review queue.')


def page_flagged():
    st.title('Flagged transactions')
    st.caption(f'Showing only transactions belonging to {user["institution"]}.')
    df = db.flagged_for(int(user['institution_id']))
    c1, c2, c3 = st.columns(3)
    c1.metric('Flagged', len(df))
    c2.metric('Pending review', int((df['status'] == 'Pending review').sum()))
    c3.metric('Confirmed fraud', int((df['status'] == 'Confirmed fraud').sum()))

    statuses = st.multiselect('Filter by status', ['Pending review', 'Confirmed fraud', 'False alarm'])
    view = df[df['status'].isin(statuses)] if statuses else df
    st.dataframe(
        view.rename(columns={'txn_id': 'Transaction', 'type': 'Type', 'amount': 'Amount',
                             'oldbalanceOrg': 'Sender before', 'newbalanceOrig': 'Sender after',
                             'oldbalanceDest': 'Receiver before', 'newbalanceDest': 'Receiver after',
                             'fraud_score': 'Fraud score', 'scored_at': 'Scored at',
                             'status': 'Status', 'note': 'Note', 'reviewed_by': 'Reviewed by',
                             'reviewed_at': 'Reviewed at'})
        .style.format({'Amount': '{:,.2f}', 'Sender before': '{:,.2f}', 'Sender after': '{:,.2f}',
                       'Receiver before': '{:,.2f}', 'Receiver after': '{:,.2f}',
                       'Fraud score': '{:.4f}'}),
        hide_index=True, width='stretch')

    st.subheader('Review a transaction')
    if len(df):
        with st.form('review'):
            txn = st.selectbox('Transaction', df['txn_id'])
            status = st.radio('Decision', ['Confirmed fraud', 'False alarm'], horizontal=True)
            note = st.text_input('Note (optional)')
            if st.form_submit_button('Save decision', type='primary'):
                db.review(txn, status, note, int(user['id']), int(user['institution_id']))
                st.success(f'{txn} marked as {status}.')
                st.rerun()
    else:
        st.info('No flagged transactions for your institution yet.')


def page_account():
    st.title('My account')
    st.write(f"**Name:** {user['full_name']}  \n**Username:** {user['username']}  \n"
             f"**Role:** {user['role'].title()}"
             + (f"  \n**Institution:** {user['institution']}" if user.get('institution') else ''))
    st.subheader('Change password')
    with st.form('pw', clear_on_submit=True):
        cur = st.text_input('Current password', type='password')
        new = st.text_input('New password (at least 8 characters)', type='password')
        conf = st.text_input('Confirm new password', type='password')
        if st.form_submit_button('Change password', type='primary'):
            if not db.authenticate(user['username'], cur):
                st.error('Current password is incorrect.')
            elif len(new) < 8:
                st.error('The new password must be at least 8 characters.')
            elif new != conf:
                st.error('The new passwords do not match.')
            else:
                db.change_password(int(user['id']), new)
                st.success('Password changed.')


# ====================================================================== navigation
if is_admin:
    PAGES = {'Overview': page_overview, 'Training sessions': page_sessions,
             'Institutions': page_institutions, 'User accounts': page_users,
             'Participating institutions': page_banks, 'My account': page_account}
else:
    PAGES = {'Overview': page_overview, 'Score transactions': page_score,
             'Flagged transactions': page_flagged, 'My institution': page_my_institution,
             'My account': page_account}

st.sidebar.markdown(f"**{user['full_name']}**  \n"
                    f"{'Administrator' if is_admin else 'Analyst · ' + str(user['institution'])}")
page = st.sidebar.radio('Navigate', list(PAGES), label_visibility='collapsed')
st.sidebar.divider()
if st.sidebar.button('Sign out', width='stretch'):
    del st.session_state.user
    st.rerun()
st.sidebar.caption(TAGLINE)

show_flash()
PAGES[page]()
footer()
