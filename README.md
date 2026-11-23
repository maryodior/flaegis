# FLAegis

Privacy-preserving federated fraud detection platform. Five institutions train a shared fraud
model with federated learning (Flower, FedAvg and FedAvgM). Only model weights leave each
institution, never transaction data.

# INSTRUCTIONS TO RUN

## Run locally
1. Put the training outputs from Google Drive (FYP/dashboard_assets) into the assets folder.
2. pip install -r requirements.txt
3. streamlit run app.py

The SQLite database (flaegis.db) is created automatically on first run, with the demo
institutions, accounts and completed training sessions.

## Running FLAegis with live training on your own computer
Requirements: Python 3.10 to 3.12, Git, about 2 GB of free disk space and 8 GB of RAM.
Tested on Linux (WSL on Windows). Native Windows and macOS should also work; on Windows, WSL is the safest option.

1. Get the code:
   git clone https://github.com/maryodior/flaegis.git
   cd flaegis
2. Create an environment and install everything:
   python -m venv .venv
   source .venv/bin/activate        (Windows PowerShell: .venv\Scripts\Activate.ps1)
   pip install -r requirements.txt -r requirements-training.txt
3. Download stage1_data.npz from the shared Google Drive link and put it in a folder named
   training_data inside the flaegis folder (training_data/stage1_data.npz).
4. Start the app:
   streamlit run app.py
   It opens at http://localhost:8501. Sign in as admin / Admin@123.
5. Go to Training sessions. It should say "Training server: this machine". Create a session
   (for example FedAvg, Non-IID, 20 rounds) and watch it train live. The first session takes a few
   extra minutes while the five institutions' data is prepared.

## Live training on this machine (optional)
1. Put stage1_data.npz (from Google Drive, FYP folder) in a folder called training_data.
2. pip install -r requirements-training.txt
3. Run the app. The Training sessions page now says "Training server: this machine", and creating a
   session starts federated training in the background with live round-by-round progress.

Without training_data (for example on Streamlit Community Cloud), sessions run on an external
training server and their results are uploaded.

## Automated tests
pip install -r requirements-dev.txt
pytest -v                  (all 37 tests, including one real training session if training_data is present)
pytest -v -m "not slow"    (quick run, skips the real training session)
The tests use a temporary database, so they never change flaegis.db or the assets folder.

## Demo accounts 
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
