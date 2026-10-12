"""Generate report-ready figures comparing Ridge and Lasso.

Reads backend/artifacts/ (models.joblib, test_data.joblib, metrics.json)
and writes PNG charts to frontend/figures/ (served by the Flask app,
so they also appear in the Graphs tab).

Run after train.py:
    python visualize.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIG = HERE.parent / "frontend" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Dark theme matching the web UI
BG, GRID, TEXT, MUTED = "#131a1e", "#26323a", "#e6ece9", "#aab8b2"
RIDGE, LASSO, BASE, REF = "#4cc38a", "#e3a73b", "#6b7a82", "#ff7b72"
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "legend.facecolor": "#1a2328",
    "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT,
    "font.family": ["Inter", "Segoe UI", "DejaVu Sans"],
    "font.size": 10,
})

bundle = joblib.load(ART / "models.joblib")
MODELS = bundle["models"]
test = joblib.load(ART / "test_data.joblib")
X_test, y_test = test["X_test"], test["y_test"].to_numpy()

ridge = MODELS["ridge"]
lasso = MODELS["lasso"]
feat_names = ridge.named_steps["pre"].get_feature_names_out()
short = [n.split("__", 1)[1] for n in feat_names]

pred_r = ridge.predict(X_test)
pred_l = lasso.predict(X_test)


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()
    print("  wrote", name)


# 1. Predicted vs actual -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
for ax, pred, title, color in zip(
    axes, [pred_r, pred_l], ["Ridge (L2)", "Lasso (L1)"], [RIDGE, LASSO]
):
    ax.scatter(y_test, pred, s=9, alpha=0.35, color=color)
    lim = max(y_test.max(), pred.max()) * 1.03
    ax.plot([0, lim], [0, lim], color=REF, ls="--", lw=1.2, label="perfect prediction")
    ax.set_xlabel("Actual runs")
    ax.set_ylabel("Predicted runs")
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle("Predicted vs actual runs — test set (2022–2023)")
save("pred_vs_actual.png")

# 2. Residuals ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, pred, title, color in zip(
    axes, [pred_r, pred_l], ["Ridge (L2)", "Lasso (L1)"], [RIDGE, LASSO]
):
    ax.scatter(pred, y_test - pred, s=9, alpha=0.35, color=color)
    ax.axhline(0, color=REF, ls="--", lw=1.2)
    ax.set_xlabel("Predicted runs")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title(title)
fig.suptitle("Residual plots — test set (random cloud around 0 is ideal)")
save("residuals.png")

# 3. Coefficient comparison (top 15 by |Ridge|) --------------------------------
rc = ridge.named_steps["model"].coef_
lc = lasso.named_steps["model"].coef_
order = np.argsort(np.abs(rc))[::-1][:15]
labels = [short[i] for i in order]
x = np.arange(len(order))
w = 0.4
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.bar(x - w / 2, rc[order], w, label="Ridge", color=RIDGE)
ax.bar(x + w / 2, lc[order], w, label="Lasso", color=LASSO)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_ylabel("Coefficient value")
ax.set_title("Top 15 features by |Ridge coefficient| — Lasso zeroes out the weak ones")
ax.legend()
save("coefficients.png")

# 4. Metrics comparison ---------------------------------------------------------
metrics = json.loads((ART / "metrics.json").read_text())
names = ["ridge", "lasso", "baseline"]
pretty = ["Ridge", "Lasso", "Baseline"]
mae = [metrics[m]["MAE"] for m in names]
rmse = [metrics[m]["RMSE"] for m in names]
r2 = [metrics[m]["R2"] for m in names]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
x = np.arange(len(names))
w = 0.35
axes[0].bar(x - w / 2, mae, w, label="MAE", color=RIDGE)
axes[0].bar(x + w / 2, rmse, w, label="RMSE", color=LASSO)
axes[0].set_xticks(x)
axes[0].set_xticklabels(pretty)
axes[0].set_ylabel("Runs")
axes[0].set_title("Test error — lower is better")
axes[0].legend()
axes[1].bar(pretty, r2, color=[RIDGE, LASSO, BASE])
axes[1].set_ylabel("R²")
axes[1].set_title("R² score — higher is better")
fig.suptitle("Model comparison — test set (2022–2023)")
save("metrics.png")

# 5. Lasso CV curve ---------------------------------------------------------------
lm = lasso.named_steps["model"]
alphas = lm.alphas_
mse_mean = lm.mse_path_.mean(axis=1)
mse_sd = lm.mse_path_.std(axis=1)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.errorbar(np.log10(alphas), mse_mean, yerr=mse_sd, fmt="-o", ms=3,
            color=LASSO, ecolor="#7a5a16", capsize=2)
ax.axvline(np.log10(lm.alpha_), color=REF, ls="--",
           label=f"selected α = {lm.alpha_:.4f}")
ax.set_xlabel("log₁₀(alpha)")
ax.set_ylabel("Cross-validated MSE")
ax.set_title("Lasso: CV error vs regularisation strength")
ax.legend()
save("lasso_cv.png")

print(f"Done — {len(list(FIG.glob('*.png')))} figures in {FIG}")
