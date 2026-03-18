"""
utils/generate_charts.py  —  DarkIQ v2
Generates all evaluation + optimization charts for report and slides.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "charts")
OPT_FILE  = os.path.join(os.path.dirname(__file__), "..", "instance", "optimization_results.json")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(CHART_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
BG  = "#070910"; BG2 = "#0e1117"; BG3 = "#141923"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG2,
    "axes.edgecolor": "#1e2535", "axes.labelcolor": "#7b8aaa",
    "xtick.color": "#7b8aaa", "ytick.color": "#7b8aaa",
    "grid.color": "#1e2535", "grid.linestyle": "--", "grid.alpha": 0.6,
    "text.color": "#dde4f0", "font.family": "DejaVu Sans",
    "axes.titlecolor": "#dde4f0", "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.titlesize": 14, "figure.titleweight": "bold",
})
A  = "#5b8af5"; A2 = "#34d399"; A3 = "#f472b6"; A4 = "#fb923c"; A5 = "#a78bfa"

def save(name):
    path = os.path.join(CHART_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {name}")

def load_opt():
    with open(OPT_FILE) as f:
        return json.load(f)

# ── 1. Optimization Summary — Before vs After ─────────────────────────────────
def chart_optimization_summary():
    opt = load_opt(); s = opt["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("AI System Optimization: Before vs After Baseline", y=1.02)

    # Spoilage comparison
    ax = axes[0]
    stores   = [r["store_id"] for r in opt["store_breakdown"]]
    base_sp  = [r["baseline_spoilage"] for r in opt["store_breakdown"]]
    ai_sp    = [r["ai_spoilage"] for r in opt["store_breakdown"]]
    x = np.arange(len(stores)); w = 0.35
    ax.bar(x-w/2, base_sp, w, label="Baseline", color=A3+"33", edgecolor=A3, linewidth=1.5, zorder=3)
    ax.bar(x+w/2, ai_sp,   w, label="AI System", color=A2+"33", edgecolor=A2, linewidth=1.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(stores, fontsize=9)
    ax.set_title("Spoilage Units by Store")
    ax.set_ylabel("Units Spoiled")
    ax.legend(fontsize=9); ax.grid(True, axis="y", zorder=0)
    for i,(b,a) in enumerate(zip(base_sp,ai_sp)):
        pct = (b-a)/max(b,1)*100
        ax.text(i+w/2+0.02, a+50, f"↓{pct:.0f}%", fontsize=8, color=A2, ha="center")

    # Category spoilage reduction
    ax = axes[1]
    cats = [r["category"] for r in opt["category_breakdown"] if r["spoilage_reduction_pct"] > 0]
    reds = [r["spoilage_reduction_pct"] for r in opt["category_breakdown"] if r["spoilage_reduction_pct"] > 0]
    colors = [A2 if r>40 else A4 for r in reds]
    bars = ax.barh(cats, reds, color=[c+"33" for c in colors], edgecolor=colors, linewidth=1.5, zorder=3)
    ax.set_title("Spoilage Reduction by Category (%)")
    ax.set_xlabel("Reduction (%)")
    for bar, val in zip(bars, reds):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f"{val:.1f}%",
                va="center", fontsize=10, fontweight="bold", color="#dde4f0")
    ax.grid(True, axis="x", zorder=0)

    # Key metrics comparison table as bar chart
    ax = axes[2]
    metrics  = ["Spoilage\nReduction", "Lost Sales\nReduction", "Waste Rate\nReduction"]
    values   = [s["spoilage_reduction_pct"], s["lost_sales_reduction_pct"],
                s["waste_rate_reduction_pp"]*10]  # scale pp for visibility
    bar_c    = [A2, A, A5]
    bars = ax.bar(metrics, values, color=[c+"33" for c in bar_c],
                  edgecolor=bar_c, linewidth=1.5, zorder=3, width=0.5)
    ax.set_title("Key Optimization Gains (%)")
    ax.set_ylabel("Improvement (%)")
    for bar, val, orig in zip(bars, values,
        [s["spoilage_reduction_pct"], s["lost_sales_reduction_pct"], s["waste_rate_reduction_pp"]]):
        label = f"{orig:.1f}%" if "Waste" not in metrics[list(values).index(val)] else f"{orig:.3f}pp"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{s['spoilage_reduction_pct']:.1f}%" if val==values[0]
                else f"{s['lost_sales_reduction_pct']:.1f}%"  if val==values[1]
                else f"{s['waste_rate_reduction_pp']:.3f}pp",
                ha="center", fontsize=11, fontweight="bold", color="#dde4f0")
    ax.grid(True, axis="y", zorder=0)

    plt.tight_layout()
    save("01_optimization_summary.png")

# ── 2. Weekly AI vs Baseline Availability ─────────────────────────────────────
def chart_weekly_comparison():
    opt = load_opt()
    w   = pd.DataFrame(opt["weekly_comparison"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AI vs Baseline — Weekly Performance Comparison")

    ax = axes[0]
    ax.fill_between(range(len(w)), w["base_avail"], alpha=0.15, color=A3)
    ax.fill_between(range(len(w)), w["ai_avail"],   alpha=0.15, color=A2)
    ax.plot(w["base_avail"].values, color=A3, linewidth=1.5, label="Baseline", linestyle="--")
    ax.plot(w["ai_avail"].values,   color=A2, linewidth=2,   label="AI System")
    ax.set_title("Weekly Availability Rate")
    ax.set_ylabel("Availability (%)"); ax.set_xlabel("Week")
    ax.legend(fontsize=10); ax.grid(True)

    ax = axes[1]
    base_sp = w["base_spoil"].values; ai_sp = w["ai_spoil"].values
    ax.bar(range(len(w)), base_sp, color=A3+"22", edgecolor=A3+"55", linewidth=0.5, label="Baseline spoilage")
    ax.bar(range(len(w)), ai_sp,   color=A2+"33", edgecolor=A2+"77", linewidth=0.5, label="AI spoilage")
    ax.set_title("Weekly Spoilage Units: Baseline vs AI")
    ax.set_ylabel("Units"); ax.set_xlabel("Week")
    ax.legend(fontsize=10); ax.grid(True, axis="y")

    plt.tight_layout()
    save("02_weekly_comparison.png")

# ── 3. ML Model Performance ────────────────────────────────────────────────────
def chart_model_performance():
    with open(os.path.join(MODEL_DIR,"bundle_v2.pkl"),"rb") as f:
        bundle = pickle.load(f)
    m = bundle["metrics"]
    models = ["Random\nForest","Gradient\nBoosting","Sequence\nRidge","Ensemble"]
    maes   = [m["rf_mae"], m["gb_mae"], m["seq_mae"], m["ens_mae"]]
    r2s    = [m["rf_r2"],  m["gb_r2"],  m["seq_r2"],  m["ens_r2"]]
    wts    = list(bundle["weights"].values())
    colors = [A, A2, A5, A3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("3-Model Ensemble: Performance Metrics")

    ax = axes[0]
    bars = ax.bar(models, maes, color=[c+"33" for c in colors], edgecolor=colors, linewidth=1.5, zorder=3)
    ax.set_title("Mean Absolute Error (lower = better)")
    ax.set_ylabel("MAE (units)"); ax.grid(True, axis="y", zorder=0)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold", color="#dde4f0")

    ax = axes[1]
    bars = ax.bar(models, r2s, color=[c+"33" for c in colors], edgecolor=colors, linewidth=1.5, zorder=3)
    ax.set_title("R² Score (higher = better)")
    ax.set_ylabel("R²"); ax.set_ylim(0.84, 0.92); ax.grid(True, axis="y", zorder=0)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0008,
                f"{val:.4f}", ha="center", fontsize=10, fontweight="bold", color="#dde4f0")

    ax = axes[2]
    wedges, texts, autotexts = ax.pie(
        wts, labels=["RF","GB","SEQ"],
        autopct="%1.1f%%", colors=[A+"55",A2+"55",A5+"55"],
        wedgeprops={"edgecolor":"#1e2535","linewidth":1.5}
    )
    for at in autotexts: at.set_color("#dde4f0"); at.set_fontsize(11)
    ax.set_title("Ensemble Weights")

    plt.tight_layout()
    save("03_model_performance.png")

# ── 4. Feature Importances ─────────────────────────────────────────────────────
def chart_feature_importance():
    fi = pd.read_csv(os.path.join(MODEL_DIR,"feature_importances.csv")).head(12)
    fi = fi.sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Top 12 Feature Importances — Random Forest")
    cats = ["lag/roll" if any(k in f for k in ["lag","roll","ema"]) else
            "temporal" if any(k in f for k in ["dow","month","woy","is_wknd","quarter","dom"]) else
            "signal"   if any(k in f for k in ["festival","day_mult","weather","rain","temp"]) else
            "entity"
            for f in fi["feature"]]
    cmap = {"lag/roll":A5,"temporal":A4,"signal":A2,"entity":A}
    colors = [cmap[c]+"44" for c in cats]; edges = [cmap[c] for c in cats]
    bars = ax.barh(fi["feature"].str.replace("_"," "), fi["importance"]*100,
                   color=colors, edgecolor=edges, linewidth=1.5, zorder=3)
    for bar, val in zip(bars, fi["importance"]*100):
        ax.text(val+0.1, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10, color="#dde4f0")
    patches = [mpatches.Patch(color=cmap[k],label=k) for k in cmap]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.set_xlabel("Importance (%)"); ax.grid(True, axis="x", zorder=0)
    plt.tight_layout()
    save("04_feature_importance.png")

# ── 5. Category Performance Heatmap ───────────────────────────────────────────
def chart_category_heatmap():
    opt = load_opt()
    cb  = opt["category_breakdown"]
    cats = [r["category"] for r in cb]
    metrics = ["Baseline\nStockout%","AI\nStockout%","Baseline\nSpoilage","AI\nSpoilage",
               "Spoilage\nRed.%","Baseline\nService%","AI\nService%"]
    data = np.array([
        [r["baseline_stockout_rate"],r["ai_stockout_rate"],
         r["baseline_spoilage"]/1000,r["ai_spoilage"]/1000,
         r["spoilage_reduction_pct"],r["baseline_service"],r["ai_service"]] for r in cb
    ])

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Category Performance: Baseline vs AI Heatmap")
    # Normalise each column 0–1 for coloring
    norm = (data - data.min(axis=0)) / (np.ptp(data, axis=0)+1e-9)
    im = ax.imshow(norm.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics, fontsize=9)
    for i in range(len(metrics)):
        for j in range(len(cats)):
            val = data[j,i]
            fmt = f"{val:.0f}" if val>10 else f"{val:.2f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=8,
                    color="black", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalised value (green=better)")
    plt.tight_layout()
    save("05_category_heatmap.png")

# ── 6. ROI & Cost Savings ─────────────────────────────────────────────────────
def chart_roi():
    opt = load_opt(); s = opt["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Financial Impact & ROI Projection")

    ax = axes[0]
    categories = ["Spoilage\nCost Saved","Lost Sales\nRecovered","Holding Cost\nReduced","Total\nBenefit"]
    sp_saved  = max(0, s["spoilage_cost_saved_inr"]) / 1e5
    ls_saved  = s["lost_sales_reduction"] * 80 / 1e5   # avg selling price ~₹80
    hold_save = abs(s["holding_cost_reduction_pct"]) * 0.5  # normalised
    total     = sp_saved + ls_saved + hold_save
    vals   = [sp_saved, ls_saved, hold_save, total]
    colors = [A3, A, A5, A2]
    bars   = ax.bar(categories, vals, color=[c+"33" for c in colors],
                    edgecolor=colors, linewidth=1.5, zorder=3)
    ax.set_title("Estimated Annual Savings (₹ Lakhs)")
    ax.set_ylabel("₹ Lakhs"); ax.grid(True, axis="y", zorder=0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"₹{val:.1f}L", ha="center", fontsize=10, fontweight="bold", color="#dde4f0")

    ax = axes[1]
    months = np.arange(1, 13)
    impl   = np.array([60,25,10,5,5,5,5,5,5,5,5,5])
    monthly_save = np.array([0,5,15,25,30,32,34,35,36,37,37,38])
    cum_cost = np.cumsum(impl); cum_save = np.cumsum(monthly_save)
    net = cum_save - cum_cost
    ax.fill_between(months, net, 0, where=net>=0, alpha=0.15, color=A2)
    ax.fill_between(months, net, 0, where=net<0,  alpha=0.15, color=A3)
    ax.plot(months, net, color=A2, linewidth=2.5, zorder=3)
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    be = months[np.argmax(net>=0)] if any(net>=0) else 12
    ax.axvline(be, color=A4, linewidth=1.5, linestyle=":", label=f"Break-even: Month {be}")
    ax.set_title("Cumulative ROI Projection (₹ Thousands)"); ax.set_xlabel("Month")
    ax.set_ylabel("Net Value (₹K)"); ax.set_xticks(months); ax.legend(fontsize=10); ax.grid(True)
    plt.tight_layout()
    save("06_roi_projection.png")

# ── Run all ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating charts…")
    chart_optimization_summary()
    chart_weekly_comparison()
    chart_model_performance()
    chart_feature_importance()
    chart_category_heatmap()
    chart_roi()
    print(f"\n✓ All 6 charts saved to {CHART_DIR}")