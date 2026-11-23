"""FLAegis training server: runs one federated training session with Flower.

Started in the background by the dashboard when the administrator creates a session:
    python trainer.py --session-id 12

It reads the session settings from the database, builds the five institutions' data from
training_data/stage1_data.npz (same splits and seed as the Colab experiments), runs the
federated rounds, writes each round's results to the database as it goes, and saves the
final global model.
"""
import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / 'training_data' / 'stage1_data.npz'
BANK_DIR = ROOT / 'training_data' / 'banks'
N_BANKS, LOCAL_ITERS, SEED = 5, 5, 42
SPLIT_KEYS = {'IID': 'iid', 'Non-IID': 'noniid', 'Fraud-type skew': 'fraudtype'}


# ---------- institution data (identical to Colab Cells 3 and 8) ----------
def build_banks(X, y):
    """Create all three splits once and cache them. Same code and seed as Colab, so the
    banks are identical to the ones used in the reported experiments."""
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    if (BANK_DIR / 'fraudtype_bank4.npz').exists():
        return

    # Cell 3: IID then non-IID (Dirichlet, alpha 0.5), sharing one random generator
    rng = np.random.default_rng(SEED)

    def iid_split(y):
        return np.array_split(rng.permutation(len(y)), N_BANKS)

    def non_iid_split(y, alpha=0.5):
        parts = [[] for _ in range(N_BANKS)]
        for c in [0, 1]:
            idx = rng.permutation(np.where(y == c)[0])
            props = 0.95 * rng.dirichlet([alpha] * N_BANKS) + 0.05 / N_BANKS
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for b, chunk in enumerate(np.split(idx, cuts)):
                parts[b].extend(chunk)
        return [rng.permutation(np.array(p)) for p in parts]

    for name, split in [('iid', iid_split(y)), ('noniid', non_iid_split(y))]:
        for b, idx in enumerate(split):
            np.savez(BANK_DIR / f'{name}_bank{b}.npz', X=X[idx], y=y[idx])

    # Cell 8: fraud-type skew (banks A, B only TRANSFER fraud; C, D, E only CASH_OUT fraud)
    rng = np.random.default_rng(SEED)
    is_transfer = X[:, 0] > 0
    legit_parts = np.array_split(rng.permutation(np.where(y == 0)[0]), N_BANKS)
    transfer_fraud = rng.permutation(np.where((y == 1) & is_transfer)[0])
    cashout_fraud = rng.permutation(np.where((y == 1) & ~is_transfer)[0])
    fraud_parts = list(np.array_split(transfer_fraud, 2)) + list(np.array_split(cashout_fraud, 3))
    for b in range(N_BANKS):
        idx = rng.permutation(np.concatenate([legit_parts[b], fraud_parts[b]]))
        np.savez(BANK_DIR / f'fraudtype_bank{b}.npz', X=X[idx], y=y[idx])


def main(session_id):
    sys.path.insert(0, str(ROOT))
    import database as db
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
    from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
    from flwr.serverapp import Grid, ServerApp
    from flwr.serverapp.strategy import FedAvg, FedAvgM
    from flwr.simulation import run_simulation

    s = db.query('SELECT * FROM training_sessions WHERE id = ?', (session_id,)).iloc[0]
    split = SPLIT_KEYS[s['data_split']]
    rounds = int(s['rounds'])
    momentum = None if s['strategy'] == 'FedAvg' else float(s['server_momentum'] or 0.9)
    db.set_setting(f'progress:{session_id}', 'Preparing institution data')

    d = np.load(DATA)
    X_test, y_test = d['X_test'], d['y_test']
    threshold = float(d['threshold'])
    n_features = X_test.shape[1]
    build_banks(d['X_train'], d['y_train'])
    del d

    def score(w):
        p = 1 / (1 + np.exp(-(X_test @ w[0].ravel() + w[1][0])))
        pred = (p >= threshold).astype(int)
        return {'auc_pr': float(average_precision_score(y_test, p)),
                'precision': float(precision_score(y_test, pred, zero_division=0)),
                'recall': float(recall_score(y_test, pred, zero_division=0)),
                'f1': float(f1_score(y_test, pred, zero_division=0))}

    client = ClientApp()

    @client.train()
    def train(msg: Message, context: Context):
        bank = context.node_config['partition-id']
        data = np.load(BANK_DIR / f'{split}_bank{bank}.npz')   # this institution's data only
        model = LogisticRegression(class_weight='balanced', max_iter=LOCAL_ITERS,
                                   warm_start=True, random_state=SEED)
        w = msg.content['arrays'].to_numpy_ndarrays()
        model.classes_ = np.array([0, 1])
        model.coef_, model.intercept_ = w[0].copy(), w[1].copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(data['X'], data['y'])
        return Message(content=RecordDict({
            'arrays': ArrayRecord([model.coef_, model.intercept_]),
            'metrics': MetricRecord({'num-examples': len(data['y'])})}), reply_to=msg)

    server = ServerApp()

    @server.main()
    def run(grid: Grid, context: Context):
        kw = dict(fraction_train=1.0, fraction_evaluate=0.0,
                  min_train_nodes=N_BANKS, min_available_nodes=N_BANKS)
        strategy = FedAvg(**kw) if momentum is None else \
            FedAvgM(**kw, server_learning_rate=1.0, server_momentum=momentum)

        def evaluate(rnd, arrays):
            m = score(arrays.to_numpy_ndarrays())
            db.add_round_result(session_id, rnd, m)            # saved live, round by round
            db.set_setting(f'progress:{session_id}', f'Round {rnd} of {rounds} complete')
            return MetricRecord(m)

        result = strategy.start(grid=grid, initial_arrays=ArrayRecord(
            [np.zeros((1, n_features)), np.zeros(1)]), num_rounds=rounds, evaluate_fn=evaluate)
        w = result.arrays.to_numpy_ndarrays()
        fname = f'session_{session_id}_weights.npz'
        np.savez(ROOT / 'assets' / fname, coef=w[0], intercept=w[1])
        db.finish_session(session_id, fname)

    db.set_setting(f'progress:{session_id}', 'Starting coordination server and 5 institution clients')
    run_simulation(server_app=server, client_app=client, num_supernodes=N_BANKS,
                   backend_config={'client_resources': {'num_cpus': 1}})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--session-id', type=int, required=True)
    sid = ap.parse_args().session_id
    try:
        main(sid)
    except Exception:
        sys.path.insert(0, str(ROOT))
        import database as db
        db.fail_session(sid, traceback.format_exc()[-500:])
        raise
