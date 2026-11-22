"""SQLite database for scored and flagged transactions."""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

import model

DB_PATH = Path(__file__).parent / 'fraud.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    txn_id TEXT PRIMARY KEY,
    bank TEXT NOT NULL,
    type TEXT NOT NULL,
    amount REAL NOT NULL,
    oldbalanceOrg REAL, newbalanceOrig REAL,
    oldbalanceDest REAL, newbalanceDest REAL,
    fraud_score REAL NOT NULL,
    flagged INTEGER NOT NULL,
    model_used TEXT NOT NULL,
    is_fraud_label INTEGER,
    scored_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    txn_id TEXT PRIMARY KEY REFERENCES transactions(txn_id),
    status TEXT NOT NULL,
    note TEXT,
    reviewed_at TEXT NOT NULL
);
"""


def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.executescript(SCHEMA)
    return con


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def setup(model_key):
    """Score the sample transactions with the chosen model and load them."""
    sample = pd.read_csv(model.ASSETS / 'sample_transactions.csv')
    sample['fraud_score'] = model.score(sample, model_key)
    sample['flagged'] = (sample['fraud_score'] >= model.THRESHOLD).astype(int)
    rows = pd.DataFrame({
        'txn_id': sample['txn_id'], 'bank': sample['bank'], 'type': sample['type'],
        'amount': sample['amount'], 'oldbalanceOrg': sample['oldbalanceOrg'],
        'newbalanceOrig': sample['newbalanceOrig'], 'oldbalanceDest': sample['oldbalanceDest'],
        'newbalanceDest': sample['newbalanceDest'], 'fraud_score': sample['fraud_score'],
        'flagged': sample['flagged'], 'model_used': model_key,
        'is_fraud_label': sample['isFraud'], 'scored_at': now(),
    })
    con = connect()
    con.execute('DELETE FROM reviews')
    con.execute('DELETE FROM transactions')
    rows.to_sql('transactions', con, if_exists='append', index=False)
    con.commit()
    con.close()


def is_empty():
    con = connect()
    n = con.execute('SELECT COUNT(*) FROM transactions').fetchone()[0]
    con.close()
    return n == 0


def add_transaction(row: dict):
    con = connect()
    con.execute(
        'INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (row['txn_id'], row['bank'], row['type'], row['amount'], row['oldbalanceOrg'],
         row['newbalanceOrig'], row['oldbalanceDest'], row['newbalanceDest'],
         row['fraud_score'], row['flagged'], row['model_used'], None, now()))
    con.commit()
    con.close()


def next_txn_id():
    con = connect()
    n = con.execute('SELECT COUNT(*) FROM transactions').fetchone()[0]
    con.close()
    return f'TXN{200000 + n}'


def flagged():
    con = connect()
    df = pd.read_sql(
        """SELECT t.txn_id, t.bank, t.type, t.amount, t.oldbalanceOrg, t.newbalanceOrig,
                  t.fraud_score, t.scored_at,
                  COALESCE(r.status, 'Pending review') AS status, r.note, r.reviewed_at
           FROM transactions t LEFT JOIN reviews r ON t.txn_id = r.txn_id
           WHERE t.flagged = 1 ORDER BY t.fraud_score DESC""", con)
    con.close()
    return df


def counts():
    con = connect()
    df = pd.read_sql(
        """SELECT bank, COUNT(*) AS transactions, SUM(flagged) AS flagged
           FROM transactions GROUP BY bank ORDER BY bank""", con)
    con.close()
    return df


def review(txn_id, status, note):
    con = connect()
    con.execute('INSERT OR REPLACE INTO reviews VALUES (?,?,?,?)', (txn_id, status, note, now()))
    con.commit()
    con.close()
