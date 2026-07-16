"""
STEP 1 — EDA with DuckDB
========================
Goal: understand class imbalance + find the signals that justify our features.
Every query here maps to a feature decision in 02_features_model.py.

Run: python 01_eda.py
(Each SQL block pastes directly into a Jupyter cell if you prefer notebooks.)
"""
import duckdb
import pandas as pd

pd.set_option("display.width", 120)
con = duckdb.connect()

con.execute("""
    CREATE OR REPLACE VIEW txn AS
    SELECT * FROM read_csv_auto('data/fraudTrain.csv')
""")

print("=" * 60)
print("1. Class imbalance — the single most important number")
print("=" * 60)
print(con.execute("""
    SELECT
        COUNT(*)                                   AS n_txn,
        SUM(is_fraud)                              AS n_fraud,
        ROUND(100.0 * AVG(is_fraud), 3)            AS fraud_pct
    FROM txn
""").df())
# WHY IT MATTERS: ~0.5% positives means accuracy is useless as a metric
# and we must handle imbalance (class_weight) + evaluate with PR-AUC.

print("=" * 60)
print("2. Amount distribution: fraud vs legit")
print("=" * 60)
print(con.execute("""
    SELECT
        is_fraud,
        ROUND(AVG(amt), 2)                          AS mean_amt,
        ROUND(MEDIAN(amt), 2)                       AS median_amt,
        ROUND(QUANTILE_CONT(amt, 0.95), 2)          AS p95_amt
    FROM txn GROUP BY is_fraud ORDER BY is_fraud
""").df())
# FEATURE DECISION: amt is right-skewed -> use log(amt).

print("=" * 60)
print("3. Fraud rate by hour of day")
print("=" * 60)
print(con.execute("""
    SELECT
        EXTRACT(hour FROM CAST(trans_date_trans_time AS TIMESTAMP)) AS hr,
        COUNT(*)                          AS n,
        ROUND(100.0 * AVG(is_fraud), 3)   AS fraud_pct
    FROM txn GROUP BY hr ORDER BY fraud_pct DESC
    LIMIT 8
""").df())
# FEATURE DECISION: fraud concentrates late night -> hour + is_night flag.

print("=" * 60)
print("4. Fraud rate by category (top 8)")
print("=" * 60)
print(con.execute("""
    SELECT
        category,
        COUNT(*)                          AS n,
        ROUND(100.0 * AVG(is_fraud), 3)   AS fraud_pct
    FROM txn GROUP BY category ORDER BY fraud_pct DESC
    LIMIT 8
""").df())
# FEATURE DECISION: card-not-present (\_net) categories are riskier
# -> one-hot encode category.

print("=" * 60)
print("5. Customer-to-merchant distance (degrees, rough proxy)")
print("=" * 60)
print(con.execute("""
    SELECT
        is_fraud,
        ROUND(AVG(SQRT(POW(lat - merch_lat, 2) + POW(long - merch_long, 2))), 3)
            AS avg_dist_deg
    FROM txn GROUP BY is_fraud ORDER BY is_fraud
""").df())
# FEATURE DECISION: fraud happens farther from home -> haversine distance km.

print("\nEDA complete. Each finding above maps to one feature in step 2.")
