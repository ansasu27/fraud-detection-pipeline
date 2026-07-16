"""
make_test_data.py
-----------------
Generates a small synthetic dataset matching the Kaggle Sparkov fraud
dataset schema (kartik2112/fraud-detection: fraudTrain.csv / fraudTest.csv).

PURPOSE: smoke-testing the pipeline only. Replace with the real Kaggle
files before producing portfolio results.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

def make_split(n, fraud_rate, start_ts):
    n_fraud = int(n * fraud_rate)
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[:n_fraud] = 1
    rng.shuffle(is_fraud)

    categories = np.array([
        "grocery_pos", "gas_transport", "shopping_net", "shopping_pos",
        "misc_net", "misc_pos", "entertainment", "food_dining",
        "health_fitness", "home", "kids_pets", "personal_care", "travel"
    ])
    # Fraud skews toward online categories
    cat_p_legit = np.ones(len(categories)) / len(categories)
    cat_p_fraud = np.where(np.isin(categories, ["shopping_net", "misc_net", "grocery_pos"]), 0.2, 0.04)
    cat_p_fraud = cat_p_fraud / cat_p_fraud.sum()

    cats, amts, hours = [], [], []
    for f in is_fraud:
        if f:
            cats.append(rng.choice(categories, p=cat_p_fraud))
            amts.append(float(np.round(rng.lognormal(5.5, 0.9), 2)))   # higher amounts
            hours.append(int(rng.choice([0,1,2,3,22,23], p=[.2,.2,.15,.15,.15,.15])))  # night-heavy
        else:
            cats.append(rng.choice(categories, p=cat_p_legit))
            amts.append(float(np.round(rng.lognormal(3.5, 1.0), 2)))
            hours.append(int(rng.integers(6, 23)))

    days = rng.integers(0, 180, n)
    ts = pd.to_datetime(start_ts) + pd.to_timedelta(days, "D") + pd.to_timedelta(hours, "h") \
         + pd.to_timedelta(rng.integers(0, 60, n), "m")

    lat = rng.uniform(25, 48, n); lon = rng.uniform(-124, -70, n)
    # fraud merchants tend to be farther away
    jitter = np.where(is_fraud == 1, rng.uniform(0.5, 3.0, n), rng.uniform(0.0, 0.7, n))
    ang = rng.uniform(0, 2*np.pi, n)

    dob_years = rng.integers(1950, 2004, n)
    df = pd.DataFrame({
        "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": rng.integers(4e15, 5e15, n).astype(np.int64),
        "merchant": ["fraud_" + m for m in rng.choice(["MerchA","MerchB","MerchC","MerchD"], n)],
        "category": cats,
        "amt": amts,
        "first": "Test", "last": "User",
        "gender": rng.choice(["M","F"], n),
        "street": "1 Test St", "city": "Testville", "state": "TX",
        "zip": 75080, "lat": lat, "long": lon,
        "city_pop": rng.integers(500, 3_000_000, n),
        "job": "Analyst",
        "dob": [f"{y}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}" for y in dob_years],
        "trans_num": [f"t{start_ts[:4]}{i:08d}" for i in range(n)],
        "unix_time": ts.astype("int64") // 10**9,
        "merch_lat": lat + jitter*np.sin(ang),
        "merch_long": lon + jitter*np.cos(ang),
        "is_fraud": is_fraud,
    })
    return df

train = make_split(40_000, 0.006, "2019-01-01")
test  = make_split(15_000, 0.006, "2020-06-21")
train.to_csv("data/fraudTrain.csv", index=True)
test.to_csv("data/fraudTest.csv", index=True)
print("train:", train.shape, "fraud rate:", train.is_fraud.mean().round(4))
print("test :", test.shape,  "fraud rate:", test.is_fraud.mean().round(4))
