# Fraud Detection Thin Slice — Design Decisions & Interview Prep

**必讀。面試官問的不是 code,是這裡的每一條 reasoning。**

## How to run

```bash
pip install pandas scikit-learn duckdb matplotlib
# 1. Put the real Kaggle files (kartik2112/fraud-detection) in data/
#    as fraudTrain.csv and fraudTest.csv
python 01_eda.py
python 02_features_model.py
python 03_cost_threshold.py
```

## Results (real Kaggle data, 555,719 out-of-time test transactions)

| Metric | Value | Context |
|---|---|---|
| PR-AUC | **0.233** | no-skill baseline = fraud rate 0.0039 → **~60x lift** |
| ROC-AUC | 0.916 | secondary metric |
| Optimal threshold | **0.85** (at FP cost = $15) | vs. arbitrary default 0.5 |
| Total expected cost | $340,904 vs. $1,543,839 at 0.5 | **77.9% cost reduction** |
| Sensitivity | optimum stays at 0.80–0.85 across FP = $5/$15/$30 | conclusion robust to the assumption |
| Precision / recall at 0.85 | 14% / 58% | known limitation → motivates XGBoost + velocity features |

---

## The 2-minute story (口頭版骨架)

> "I built an end-to-end fraud detection pipeline on ~1.85M simulated credit
> card transactions. The interesting part isn't the model — it's that fraud
> detection is a **cost trade-off problem**: a missed fraud costs the
> transaction amount, but a false alarm costs analyst review time and
> customer friction. So instead of using the default 0.5 threshold, I built
> a cost matrix, swept thresholds, and chose the one minimizing total
> expected cost — then stress-tested it against different false-positive
> cost assumptions, since that number is a business estimate, not a fact.
> The baseline is deliberately logistic regression: in a risk context you
> need a model compliance can interpret before you earn the right to add
> complexity."

---

## Decision log (Q&A format — 每題都可能被問)

**Q: Why DuckDB for EDA instead of pandas?**
SQL-first EDA mirrors how analysts actually work at fintech companies
(data lives in warehouses). DuckDB gives warehouse-style SQL locally with
zero setup, and it handles the 1.3M-row file faster than pandas for
aggregations. It also demonstrates SQL skills directly.

**Q: Why logistic regression and not XGBoost from the start?**
Three reasons: (1) interpretability — in regulated risk settings, model
risk management teams need to explain decisions; coefficients do that,
(2) it establishes a baseline so any XGBoost lift is measurable, not
assumed, (3) it trains in seconds so iteration on features is fast.
XGBoost is the planned next step, justified only if it beats the baseline
on PR-AUC / cost.

**Q: Why class_weight='balanced' instead of SMOTE/undersampling?**
SMOTE creates synthetic minority samples, which risks leaking artificial
patterns and complicates the pipeline. Class weights achieve the same
rebalancing effect inside the loss function with zero data manipulation.
More importantly, the real business lever is the decision threshold —
which I handle explicitly in the cost analysis — so aggressive resampling
solves a problem I don't have.

**Q: Why PR-AUC as the primary metric, not accuracy or ROC-AUC?**
At 0.5% fraud rate, predicting "never fraud" gives 99.5% accuracy —
useless. ROC-AUC is inflated by the huge true-negative count. PR-AUC
focuses on the minority class: precision and recall of fraud calls, which
is what the business cares about. The no-skill baseline for PR-AUC equals
the fraud rate, so it's easy to communicate lift.

**Q: How did you pick your features?**
Every feature traces to an EDA finding: log(amt) because fraud amounts
are higher and right-skewed; hour + is_night because fraud concentrates
overnight; haversine distance because fraudulent merchants are farther
from the cardholder's home; category one-hots because card-not-present
categories carry more risk. No kitchen-sink features — each one has a
documented reason, which also makes the model easier to defend.

**Q: How did you set FP cost = $15?**
It's an assumption: roughly 10 minutes of analyst review time plus a
customer-friction penalty. Because it's soft, I ran sensitivity at $5 /
$15 / $30 and showed how the optimal threshold shifts. The point isn't
the exact number — it's that the framework lets the business plug in
their real number.

**Q: Why train on fraudTrain and test on fraudTest?**
The Kaggle split is chronological, so this is out-of-time validation —
the same way a deployed model faces the future. A random split would
leak temporal patterns and overstate performance.

**Q: Your precision at the optimal threshold is only 14% — isn't that bad?**
It means ~1 in 7 alerts is real fraud, and recall drops to 58%. For a
logistic baseline that's expected, and it's exactly why the roadmap adds
XGBoost and per-card velocity features next. The key point: because the
cost framework already exists, I can quantify in dollars whether the
added model complexity pays for itself — improvement is measured, not
assumed.

**Q: What are the limitations / what's next?**
(1) No per-card velocity features yet (txn count in trailing 24h) — the
highest-value next feature. (2) Logistic regression can't capture
interactions; XGBoost is next, evaluated on the same cost framework.
(3) Simulated data lacks adversarial adaptation — real fraudsters shift
patterns, which is why the roadmap ends with a PSI-based drift monitoring
dashboard. (4) FP cost is a point estimate; ideally calibrated with ops
data.

**Q: (如果被問) Did you use AI assistance?**
誠實版答案:"I used AI as a pair programmer for scaffolding, the same way
I'd use it on the job — but every design decision here is one I can
defend, and I rebuilt/verified each step myself." ← 這句要成立,前提是
你真的把每個檔案跑過、讀懂、能改。給自己排 2-3 小時做這件事。

---

## What "done" looks like for the thin slice

- [x] Replace data/ with real Kaggle fraudTrain/fraudTest
- [x] Re-run all three steps; record real PR-AUC + cost reduction %
- [x] Fill the real numbers into resume bullet / story
- [ ] Push to GitHub with this DECISIONS.md as part of the README
- [ ] Practice the 2-minute story out loud twice

## Resume bullet(final,可直接複製)

> Built an end-to-end fraud detection pipeline on 1.85M simulated
> credit-card transactions (DuckDB, scikit-learn); engineered EDA-driven
> features and applied cost-based threshold optimization with sensitivity
> analysis, reducing expected fraud-plus-review cost by 78% versus the
> default threshold (PR-AUC 0.23 vs. 0.004 no-skill baseline, ~60x lift).
