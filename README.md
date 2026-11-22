# FLAegis Analyst Dashboard

Analyst dashboard for the privacy-preserving federated fraud detection system.

## Run locally
1. Unzip dashboard_assets.zip (from Google Drive, FYP folder) into the assets folder.
2. pip install -r requirements.txt
3. streamlit run app.py

The SQLite database (fraud.db) is created automatically on first run.

## Pages
i. Overview: central vs FedAvg vs FedAvgM results
ii. Participating banks: how data is spread across the 5 institutions
iii. Score a transaction: score a new transaction and see why
iv. Flagged transactions: review queue stored in SQLite
