"""
STEP 3 — Cost-based threshold selection (the BA core of this project)
=====================================================================
A fraud model doesn't output decisions; a THRESHOLD does. Picking it is
a business problem, not an ML problem:

    Cost(FN) = the fraud amount we fail to catch (use actual amt!)
    Cost(FP) = friction of blocking/reviewing a good customer
               (analyst review time + customer-experience damage)

We sweep thresholds, compute total expected cost on the test set, and
pick the threshold that minimizes it. We also show sensitivity to the
FP-cost assumption — interviewers love this because it shows you know
the assumption is soft.

Run: python 03_cost_threshold.py   (after 02_features_model.py)
Outputs: outputs/cost_curve.png, outputs/threshold_table.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Assumptions — SAY THESE OUT LOUD in interviews, then show sensitivity
# ----------------------------------------------------------------------
FP_COST = 15.0        # $ per false alarm: ~10 min analyst review + friction
FP_COST_SCENARIOS = [5.0, 15.0, 30.0]   # sensitivity band

df = pd.read_csv("outputs/model_scores_test.csv")
y, amt, score = df["is_fraud"].values, df["amt"].values, df["score"].values

def total_cost(threshold, fp_cost):
    flag = score >= threshold
    fn_cost = amt[(y == 1) & (~flag)].sum()      # missed fraud = $ lost
    fp_n = int(((y == 0) & flag).sum())          # good txns flagged
    return fn_cost + fp_n * fp_cost, fn_cost, fp_n

thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
rows = []
for t in thresholds:
    cost, fn_cost, fp_n = total_cost(t, FP_COST)
    flag = score >= t
    tp = int(((y == 1) & flag).sum())
    rows.append({
        "threshold": t,
        "frauds_caught": tp,
        "frauds_missed": int((y == 1).sum()) - tp,
        "false_positives": fp_n,
        "missed_fraud_$": round(fn_cost, 0),
        "review_cost_$": round(fp_n * FP_COST, 0),
        "total_cost_$": round(cost, 0),
    })
table = pd.DataFrame(rows)
table.to_csv("outputs/threshold_table.csv", index=False)

best = table.loc[table["total_cost_$"].idxmin()]
naive = total_cost(0.5, FP_COST)[0]

print(table.to_string(index=False))
print("\n" + "=" * 60)
print(f"Optimal threshold @ FP=${FP_COST:.0f}: {best['threshold']:.2f}")
print(f"Total cost at optimum : ${best['total_cost_$']:,.0f}")
print(f"Total cost at 0.5     : ${naive:,.0f}")
if naive > 0:
    print(f"Cost reduction vs 0.5 : {100 * (naive - best['total_cost_$']) / naive:.1f}%")
print("=" * 60)

# ----------------------------------------------------------------------
# Sensitivity: does the optimum move when FP cost assumption changes?
# ----------------------------------------------------------------------
print("\nSensitivity to FP-cost assumption:")
fig, ax = plt.subplots(figsize=(9, 5.5))
for fpc in FP_COST_SCENARIOS:
    costs = [total_cost(t, fpc)[0] for t in thresholds]
    t_opt = thresholds[int(np.argmin(costs))]
    print(f"  FP cost ${fpc:>5.0f}  ->  optimal threshold {t_opt:.2f}, "
          f"min total cost ${min(costs):,.0f}")
    ax.plot(thresholds, costs, marker="o", ms=3, label=f"FP cost = ${fpc:.0f}")
    ax.axvline(t_opt, ls="--", lw=0.8, alpha=0.4)

ax.set_xlabel("Decision threshold")
ax.set_ylabel("Total expected cost on test set ($)")
ax.set_title("Cost-based threshold selection with FP-cost sensitivity")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/cost_curve.png", dpi=150)
print("\nSaved outputs/cost_curve.png and outputs/threshold_table.csv")
print("RESULT SENTENCE for interviews: 'Using a cost-based threshold instead")
print("of the default 0.5 reduced expected fraud + review cost by X%.'")
