"""SQLite database for FLAegis: institutions, users, training sessions, round results,
scored transactions and analyst reviews."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import bcrypt
import pandas as pd

import model

DB_PATH = Path(__file__).parent / 'flaegis.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS institutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'analyst')),
    institution_id INTEGER REFERENCES institutions(id),
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    strategy TEXT NOT NULL,
    data_split TEXT NOT NULL,
    rounds INTEGER NOT NULL,
    server_momentum REAL,
    status TEXT NOT NULL,
    weights_file TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS round_results (
    session_id INTEGER NOT NULL REFERENCES training_sessions(id),
    round INTEGER NOT NULL,
    auc_pr REAL, precision REAL, recall REAL, f1 REAL,
    PRIMARY KEY (session_id, round)
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id TEXT PRIMARY KEY,
    institution_id INTEGER NOT NULL REFERENCES institutions(id),
    type TEXT NOT NULL,
    amount REAL NOT NULL,
    oldbalanceOrg REAL, newbalanceOrig REAL,
    oldbalanceDest REAL, newbalanceDest REAL,
    fraud_score REAL NOT NULL,
    flagged INTEGER NOT NULL,
    model_used TEXT NOT NULL,
    scored_by INTEGER REFERENCES users(id),
    scored_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    txn_id TEXT PRIMARY KEY REFERENCES transactions(txn_id),
    status TEXT NOT NULL,
    note TEXT,
    reviewed_by INTEGER REFERENCES users(id),
    reviewed_at TEXT NOT NULL
);
"""

BANKS = ['Bank A', 'Bank B', 'Bank C', 'Bank D', 'Bank E']

# Demo accounts created on first run (for testing and UAT)
DEMO_ADMIN = ('admin', 'Platform Administrator', 'Admin@123')
DEMO_ANALYSTS = [(f'analyst_{b[-1].lower()}', f'{b} Analyst', f'Bank{b[-1]}@123', b) for b in BANKS]

# Training runs completed on the training server (Colab), loaded at setup
SEED_RUNS = [
    ('fedavg_iid', 'FedAvg, IID banks', 'FedAvg', 'IID', None),
    ('fedavgm_iid', 'FedAvgM, IID banks', 'FedAvgM', 'IID', 0.9),
    ('fedavg_noniid', 'FedAvg, non-IID banks', 'FedAvg', 'Non-IID', None),
    ('fedavgm_noniid', 'FedAvgM, non-IID banks', 'FedAvgM', 'Non-IID', 0.9),
    ('fedavgm05_noniid', 'FedAvgM (momentum 0.5), non-IID banks', 'FedAvgM', 'Non-IID', 0.5),
    ('fedavg_fraudtype', 'FedAvg, fraud-type skew', 'FedAvg', 'Fraud-type skew', None),
    ('fedavgm_fraudtype', 'FedAvgM, fraud-type skew', 'FedAvgM', 'Fraud-type skew', 0.9),
    ('fedavgm05_fraudtype', 'FedAvgM (momentum 0.5), fraud-type skew', 'FedAvgM', 'Fraud-type skew', 0.5),
]
DEFAULT_ACTIVE = 'fedavg_noniid'
RUNNING_STATUSES = ['Training', 'In progress', 'Awaiting results']


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys = ON')
    return con


def query(sql, params=()):
    con = connect()
    df = pd.read_sql(sql, con, params=params)
    con.close()
    return df


def execute(sql, params=()):
    con = connect()
    cur = con.execute(sql, params)
    con.commit()
    last = cur.lastrowid
    con.close()
    return last


# ---------- passwords ----------
def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def check_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except ValueError:
        return False


# ---------- setup ----------
def initialise():
    """Create tables and, on first run, the demo institutions, users, sessions and transactions."""
    con = connect()
    con.executescript(SCHEMA)
    con.commit()
    seeded = con.execute('SELECT COUNT(*) FROM institutions').fetchone()[0] > 0
    con.close()
    if seeded:
        return

    for b in BANKS:
        execute('INSERT INTO institutions (name, created_at) VALUES (?, ?)', (b, now()))
    admin_id = create_user(DEMO_ADMIN[0], DEMO_ADMIN[1], DEMO_ADMIN[2], 'admin', None)
    for username, name, pw, bank in DEMO_ANALYSTS:
        create_user(username, name, pw, 'analyst', institution_id(bank))

    for key, name, strategy, split, momentum in SEED_RUNS:
        hist = model.ASSETS / f'fl_{key}_history.json'
        if hist.exists():
            sid = create_session(name, strategy, split, 20, momentum, admin_id)
            import_results(sid, json.loads(hist.read_text()), f'fl_{key}_weights.npz')
            if key == DEFAULT_ACTIVE:
                set_setting('active_model', f'session:{sid}')
    if get_setting('active_model') is None:
        set_setting('active_model', 'central')
    load_sample_transactions()


def institution_id(name):
    row = query('SELECT id FROM institutions WHERE name = ?', (name,))
    return int(row['id'].iloc[0]) if len(row) else None


# ---------- users and institutions ----------
def create_user(username, full_name, password, role, inst_id):
    return execute(
        'INSERT INTO users (username, full_name, password_hash, role, institution_id, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (username.strip().lower(), full_name.strip(), hash_password(password), role, inst_id, now()))


def authenticate(username, password):
    df = query('SELECT u.*, i.name AS institution FROM users u '
               'LEFT JOIN institutions i ON u.institution_id = i.id WHERE u.username = ?',
               (username.strip().lower(),))
    if len(df) == 0:
        return None
    user = df.iloc[0].to_dict()
    if not user['active'] or not check_password(password, user['password_hash']):
        return None
    user.pop('password_hash')
    return user


def change_password(user_id, new_password):
    execute('UPDATE users SET password_hash = ? WHERE id = ?', (hash_password(new_password), user_id))


def set_user_active(user_id, active):
    execute('UPDATE users SET active = ? WHERE id = ?', (int(active), user_id))


def list_users():
    return query('SELECT u.id, u.username, u.full_name, u.role, i.name AS institution, '
                 'u.active, u.created_at FROM users u '
                 'LEFT JOIN institutions i ON u.institution_id = i.id ORDER BY u.role, u.username')


def list_institutions():
    return query("""SELECT i.id, i.name, i.created_at,
                           (SELECT COUNT(*) FROM users u WHERE u.institution_id = i.id
                              AND u.role = 'analyst') AS analysts,
                           (SELECT COUNT(*) FROM transactions t WHERE t.institution_id = i.id) AS transactions,
                           (SELECT COALESCE(SUM(flagged), 0) FROM transactions t
                              WHERE t.institution_id = i.id) AS flagged
                    FROM institutions i ORDER BY i.name""")


def create_institution(name):
    return execute('INSERT INTO institutions (name, created_at) VALUES (?, ?)', (name.strip(), now()))


# ---------- settings ----------
def get_setting(key):
    df = query('SELECT value FROM settings WHERE key = ?', (key,))
    return df['value'].iloc[0] if len(df) else None


def set_setting(key, value):
    execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, value))


# ---------- training sessions ----------
def create_session(name, strategy, split, rounds, momentum, created_by):
    return execute(
        'INSERT INTO training_sessions (name, strategy, data_split, rounds, server_momentum, status, '
        'created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (name, strategy, split, rounds, momentum, 'In progress', created_by, now()))


def import_results(session_id, history, weights_file):
    con = connect()
    con.execute('DELETE FROM round_results WHERE session_id = ?', (session_id,))
    con.executemany(
        'INSERT INTO round_results VALUES (?, ?, ?, ?, ?, ?)',
        [(session_id, int(h['round']), h['auc_pr'], h['precision'], h['recall'], h['f1'])
         for h in history])
    con.execute("UPDATE training_sessions SET status = 'Completed', weights_file = ?, "
                'completed_at = ? WHERE id = ?', (weights_file, now(), session_id))
    con.commit()
    con.close()


# ---------- live training on this machine ----------
TRAIN_DATA = Path(__file__).parent / 'training_data' / 'stage1_data.npz'


def live_training_available():
    """True when this machine can run training itself (data file present and Flower installed)."""
    if not TRAIN_DATA.exists():
        return False
    try:
        import importlib.util
        return importlib.util.find_spec('flwr') is not None and \
            importlib.util.find_spec('sklearn') is not None
    except Exception:
        return False


def start_live_session(session_id):
    import subprocess
    import sys
    log = open(Path(__file__).parent / 'training_data' / f'session_{session_id}.log', 'w')
    proc = subprocess.Popen([sys.executable, str(Path(__file__).parent / 'trainer.py'),
                             '--session-id', str(session_id)],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            cwd=str(Path(__file__).parent))
    execute("UPDATE training_sessions SET status = 'Training' WHERE id = ?", (session_id,))
    set_setting(f'pid:{session_id}', str(proc.pid))
    set_setting(f'progress:{session_id}', 'Starting')


def process_alive(session_id):
    import os
    pid = get_setting(f'pid:{session_id}')
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    try:   # a finished child that has not been collected yet counts as not alive
        with open(f'/proc/{int(pid)}/stat') as f:
            return f.read().split()[2] != 'Z'
    except OSError:
        return True


def stop_process(session_id):
    import os
    import signal
    import shutil
    import subprocess
    pid = get_setting(f'pid:{session_id}')
    if pid:
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass
    ray = shutil.which('ray')            # also stop any simulation workers left behind
    if ray:
        subprocess.run([ray, 'stop', '--force'], capture_output=True, timeout=60)


def add_round_result(session_id, rnd, m):
    execute('INSERT OR REPLACE INTO round_results VALUES (?, ?, ?, ?, ?, ?)',
            (session_id, int(rnd), m['auc_pr'], m['precision'], m['recall'], m['f1']))


def finish_session(session_id, weights_file):
    execute("UPDATE training_sessions SET status = 'Completed', weights_file = ?, completed_at = ? "
            "WHERE id = ? AND status = 'Training'", (weights_file, now(), session_id))
    set_setting(f'progress:{session_id}', 'Completed')


def fail_session(session_id, message):
    execute("UPDATE training_sessions SET status = 'Failed', completed_at = ? "
            "WHERE id = ? AND status = 'Training'", (now(), session_id))
    set_setting(f'progress:{session_id}', 'Failed: ' + message)


def check_stalled():
    """Mark a live session as failed if its training process has stopped without finishing."""
    df = query("SELECT id FROM training_sessions WHERE status = 'Training'")
    for sid in df['id']:
        if not process_alive(int(sid)):
            fail_session(int(sid), 'The training process stopped unexpectedly.')


def latest_round(session_id):
    df = query('SELECT * FROM round_results WHERE session_id = ? ORDER BY round', (session_id,))
    return df


def cancel_session(session_id):
    stop_process(session_id)
    execute("UPDATE training_sessions SET status = 'Cancelled', completed_at = ? WHERE id = ?",
            (now(), session_id))


def delete_session(session_id):
    """Permanently delete a cancelled session (it never produced a model)."""
    con = connect()
    status = con.execute('SELECT status FROM training_sessions WHERE id = ?', (session_id,)).fetchone()
    if status is None or status[0] not in ('Cancelled', 'Failed'):
        con.close()
        raise ValueError('Only cancelled or failed sessions can be deleted.')
    con.execute('DELETE FROM round_results WHERE session_id = ?', (session_id,))
    con.execute('DELETE FROM training_sessions WHERE id = ?', (session_id,))
    con.commit()
    con.close()


def archive_session(session_id, archived=True):
    """Hide a completed session from lists and charts, keeping it for the audit trail."""
    if archived and active_model() == f'session:{session_id}':
        raise ValueError('This session is the model in use. Choose another model first.')
    new, old = ('Archived', 'Completed') if archived else ('Completed', 'Archived')
    execute('UPDATE training_sessions SET status = ? WHERE id = ? AND status = ?',
            (new, session_id, old))


def list_sessions():
    return query("""SELECT s.id, s.name, s.strategy, s.data_split, s.rounds, s.server_momentum,
                           s.status, s.weights_file, u.username AS created_by, s.created_at,
                           s.completed_at,
                           (SELECT auc_pr FROM round_results r WHERE r.session_id = s.id
                              ORDER BY round DESC LIMIT 1) AS final_auc_pr,
                           (SELECT f1 FROM round_results r WHERE r.session_id = s.id
                              ORDER BY round DESC LIMIT 1) AS final_f1
                    FROM training_sessions s LEFT JOIN users u ON s.created_by = u.id
                    ORDER BY s.id""")


def round_results(session_ids=None):
    df = query("SELECT r.*, s.name FROM round_results r "
               "JOIN training_sessions s ON r.session_id = s.id "
               "WHERE s.status = 'Completed' ORDER BY r.session_id, r.round")
    if session_ids is not None:
        df = df[df['session_id'].isin(session_ids)]
    return df


# ---------- active model ----------
def model_options():
    """Map of model key to display name: the central reference plus completed sessions."""
    opts = {'central': 'Central reference model (data pooled)'}
    for _, s in list_sessions().iterrows():
        if s['status'] == 'Completed' and s['weights_file'] and \
                (model.ASSETS / s['weights_file']).exists():
            opts[f"session:{s['id']}"] = f"Session {s['id']}: {s['name']}"
    return opts


def weights_for(key):
    if key == 'central':
        return model.ASSETS / 'central_weights.npz'
    sid = int(key.split(':')[1])
    f = query('SELECT weights_file FROM training_sessions WHERE id = ?', (sid,))['weights_file'].iloc[0]
    return model.ASSETS / f


def active_model():
    key = get_setting('active_model') or 'central'
    if key not in model_options():
        key = 'central'
    return key


# ---------- transactions ----------
def load_sample_transactions():
    sample = pd.read_csv(model.ASSETS / 'sample_transactions.csv')
    key = active_model()
    sample['fraud_score'] = model.score(sample, weights_for(key))
    sample['flagged'] = (sample['fraud_score'] >= model.THRESHOLD).astype(int)
    ids = {b: institution_id(b) for b in BANKS}
    rows = [(r.txn_id, ids[r.bank], r.type, r.amount, r.oldbalanceOrg, r.newbalanceOrig,
             r.oldbalanceDest, r.newbalanceDest, float(r.fraud_score), int(r.flagged), key,
             None, now()) for r in sample.itertuples()]
    con = connect()
    con.execute('DELETE FROM reviews')
    con.execute('DELETE FROM transactions')
    con.executemany('INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    con.commit()
    con.close()


def next_txn_id():
    df = query("SELECT COUNT(*) AS n FROM transactions WHERE txn_id LIKE 'TXN2%'")
    return int(df['n'].iloc[0])


def add_transactions(df, inst_id, key, user_id):
    """Save scored transactions. df has the raw columns plus fraud_score and flagged."""
    start = next_txn_id()
    ids = [f'TXN{200000 + start + i}' for i in range(len(df))]
    rows = [(ids[i], inst_id, r.type, r.amount, r.oldbalanceOrg, r.newbalanceOrig,
             r.oldbalanceDest, r.newbalanceDest, float(r.fraud_score), int(r.flagged), key,
             user_id, now()) for i, r in enumerate(df.itertuples())]
    con = connect()
    con.executemany('INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    con.commit()
    con.close()
    return ids


def flagged_for(inst_id):
    return query("""SELECT t.txn_id, t.type, t.amount, t.oldbalanceOrg, t.newbalanceOrig,
                           t.oldbalanceDest, t.newbalanceDest, t.fraud_score, t.scored_at,
                           COALESCE(r.status, 'Pending review') AS status, r.note,
                           u.username AS reviewed_by, r.reviewed_at
                    FROM transactions t
                    LEFT JOIN reviews r ON t.txn_id = r.txn_id
                    LEFT JOIN users u ON r.reviewed_by = u.id
                    WHERE t.flagged = 1 AND t.institution_id = ?
                    ORDER BY t.fraud_score DESC""", (inst_id,))


def review(txn_id, status, note, user_id, inst_id):
    # Only allow reviewing a transaction that belongs to the analyst's own institution
    owner = query('SELECT institution_id FROM transactions WHERE txn_id = ?', (txn_id,))
    if len(owner) == 0 or int(owner['institution_id'].iloc[0]) != inst_id:
        raise PermissionError('This transaction belongs to another institution.')
    execute('INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?)',
            (txn_id, status, note, user_id, now()))


def institution_summary():
    """Aggregate counts only, so the administrator never sees individual transactions."""
    return query("""SELECT i.name AS institution, COUNT(t.txn_id) AS transactions_scored,
                           COALESCE(SUM(t.flagged), 0) AS flagged,
                           SUM(CASE WHEN r.status = 'Confirmed fraud' THEN 1 ELSE 0 END) AS confirmed_fraud,
                           SUM(CASE WHEN r.status = 'False alarm' THEN 1 ELSE 0 END) AS false_alarms
                    FROM institutions i
                    LEFT JOIN transactions t ON t.institution_id = i.id
                    LEFT JOIN reviews r ON r.txn_id = t.txn_id
                    GROUP BY i.id ORDER BY i.name""")


def consortium_members():
    """What any analyst may see about the consortium: names, join dates and training participation.
    No volumes or fraud figures."""
    inst = query('SELECT id, name, created_at FROM institutions ORDER BY name')
    completed = int(query("SELECT COUNT(*) AS n FROM training_sessions WHERE status = 'Completed'")['n'].iloc[0])
    # The original five institutions took part in every completed session; newly registered
    # institutions have not joined a training session yet.
    inst['sessions'] = inst['name'].map(lambda n: completed if n in BANKS else 0)
    inst['participation'] = inst['sessions'].map(lambda k: f'{k} of {completed} sessions')
    return inst


def own_summary(inst_id):
    """Activity figures for one institution only."""
    return query("""SELECT COUNT(t.txn_id) AS scored, COALESCE(SUM(t.flagged), 0) AS flagged,
                           SUM(CASE WHEN t.flagged = 1 AND r.status IS NULL THEN 1 ELSE 0 END) AS pending,
                           SUM(CASE WHEN r.status = 'Confirmed fraud' THEN 1 ELSE 0 END) AS confirmed,
                           SUM(CASE WHEN r.status = 'False alarm' THEN 1 ELSE 0 END) AS false_alarms
                    FROM transactions t LEFT JOIN reviews r ON r.txn_id = t.txn_id
                    WHERE t.institution_id = ?""", (inst_id,)).iloc[0].fillna(0).astype(int).to_dict()
