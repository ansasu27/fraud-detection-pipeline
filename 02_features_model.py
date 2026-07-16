"""
STEP 2 — Feature engineering + Logistic Regression baseline
===========================================================
Design principles (say these in interviews):
1. Every feature traces back to an EDA finding — no kitchen-sink features.
2. Logistic regression FIRST: interpretable baseline that risk/compliance
   teams can sign off on. XGBoost comes later only if lift justifies it.
3. class_weight='balanced' instead of SMOTE: simpler, no synthetic data
   leakage risk, and we control the business trade-off at the THRESHOLD
   level anyway (step 3), not the sampling level.
4. Train on fraudTrain, evaluate on fraudTest = out-of-time validation,
   which mirrors how a model actually gets deployed.

Run: python 02_features_model.py
Outputs: outputs/model_scores_test.csv (used by step 3)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, classification_report,
    confusion_matrix,
)

RANDOM_STATE = 27


# ----------------------------------------------------------------------
# Feature engineering — one function so train/test are transformed
# identically (prevents train/serve skew)
# ----------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between customer home and merchant."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["trans_date_trans_time"])

    # EDA #2: amt right-skewed -> log transform
    out["log_amt"] = np.log1p(df["amt"])

    # EDA #3: fraud concentrates late night
    out["hour"] = ts.dt.hour
    out["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour <= 4)).astype(int)
    out["day_of_week"] = ts.dt.dayofweek

    # EDA #5: fraud farther from home
    out["dist_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # Demographic / context
    out["age"] = (ts - pd.to_datetime(df["dob"])).dt.days / 365.25
    out["log_city_pop"] = np.log1p(df["city_pop"])

    # EDA #4: category risk -> one-hot
    cat = pd.get_dummies(df["category"], prefix="cat", dtype=int)
    out = pd.concat([out, cat], axis=1)
    return out


def align_columns(train_X, test_X):
    """Ensure test has exactly the training columns (missing dummies -> 0)."""
    return test_X.reindex(columns=train_X.columns, fill_value=0)


# ----------------------------------------------------------------------
# Load + transform
# ----------------------------------------------------------------------
print("Loading data ...")
train = pd.read_csv("data/fraudTrain.csv", index_col=0)
test = pd.read_csv("data/fraudTest.csv", index_col=0)

X_train, y_train = build_features(train), train["is_fraud"]
X_test = align_columns(X_train, build_features(test))
y_test = test["is_fraud"]
print(f"train: {X_train.shape}, fraud rate {y_train.mean():.4f}")
print(f"test : {X_test.shape}, fraud rate {y_test.mean():.4f}")

# ----------------------------------------------------------------------
# Model — scaled logistic regression, balanced class weights
# ----------------------------------------------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",   # handles ~0.5% positive rate
        max_iter=2000,
        random_state=RANDOM_STATE,
    )),
])
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------------------
# Evaluation — metrics that survive class imbalance
# ----------------------------------------------------------------------
print("\n=== Threshold-free metrics (imbalance-safe) ===")
print(f"PR-AUC  (primary): {average_precision_score(y_test, scores):.4f}"
      f"   <- baseline to beat = fraud rate {y_test.mean():.4f}")
print(f"ROC-AUC (secondary): {roc_auc_score(y_test, scores):.4f}")

print("\n=== At default threshold 0.5 (for reference only) ===")
pred = (scores >= 0.5).astype(int)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=3))
print("NOTE: 0.5 is arbitrary. The RIGHT threshold is a business decision")
print("      -> that's exactly what step 3 (cost analysis) determines.")

# Top coefficients = interview talking points on interpretability
coefs = pd.Series(
    model.named_steps["clf"].coef_[0], index=X_train.columns
).sort_values(key=abs, ascending=False)
print("\n=== Top 10 features by |coefficient| (standardized) ===")
print(coefs.head(10).round(3))

# Persist scores for the cost analysis step
out = test[["amt"]].copy()
out["is_fraud"] = y_test
out["score"] = scores
out.to_csv("outputs/model_scores_test.csv", index=False)
print("\nSaved outputs/model_scores_test.csv -> feed into 03_cost_threshold.py")
