# FLAegis

Privacy-preserving federated fraud detection platform. Five institutions train a shared fraud
model with federated learning (Flower, FedAvg and FedAvgM). Only model weights leave each
institution, never transaction data.

## Run locally
1. Put the training outputs from Google Drive (FYP/dashboard_assets) into the assets folder.
2. pip install -r requirements.txt
3. streamlit run app.py

The SQLite database (flaegis.db) is created automatically on first run, with the demo
institutions, accounts and completed training sessions.

## Live training on this machine (optional)
1. Put stage1_data.npz (from Google Drive, FYP folder) in a folder called training_data.
2. pip install -r requirements-training.txt
3. Run the app. The Training sessions page now says "Training server: this machine", and creating a
   session starts federated training in the background with live round-by-round progress.

Without training_data (for example on Streamlit Community Cloud), sessions run on an external
training server and their results are uploaded.

## Demo accounts (change these after first login)
| Role | Username | Password |
|---|---|---|
| Administrator | admin | Admin@123 |
| Analyst, Bank A | analyst_a | BankA@123 |
| Analyst, Bank B | analyst_b | BankB@123 |
| Analyst, Bank C | analyst_c | BankC@123 |
| Analyst, Bank D | analyst_d | BankD@123 |
| Analyst, Bank E | analyst_e | BankE@123 |

## Modules
i. Authentication and user management: bcrypt password hashing, administrator and analyst roles
ii. Training sessions: create sessions, load results from the training server, choose the model in use
iii. Dashboard: results by model, round-by-round charts, local-only vs federated comparison
iv. Transaction scoring: single transactions and validated batch CSV upload, with explanations
v. Flagged transactions: each analyst sees and reviews only their own institution's cases
