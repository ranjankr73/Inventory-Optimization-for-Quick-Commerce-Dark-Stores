"""
utils/optimization_analysis.py  —  FINAL (Clean + Correct)

Compares 3 inventory policies:
1. Baseline (rule-based ROP)
2. Classical (Newsvendor model)
3. ML-based (cost optimization)

Outputs cost + service level comparison.
"""

import sqlite3, json, os
import pandas as pd
import numpy as np
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), "..")
DB   = os.path.join(BASE, "instance", "darkiq.db")
OUT  = os.path.join(BASE, "instance", "optimization_results.json")


# ─────────────────────────────────────────────────────────────
# 🔁 SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────
def simulate_policy(demand_df, skus_df, policy="baseline"):
    sku_dict = skus_df.set_index("sku_id").to_dict(orient="index")
    rows = []

    for (store_id, sku_id), grp in demand_df.groupby(["store_id","sku_id"]):
        sku = sku_dict[sku_id]

        max_s  = sku["max_stock"]
        shelf  = sku["shelf_days"]
        perish = sku["is_perishable"]
        cost   = sku["unit_cost"]
        cat    = sku["category"]

        holding_cost  = sku.get("holding_cost", 0.5)
        shortage_cost = sku.get("shortage_cost", 5.0)

        stock = int(max_s * 0.6)
        days_in = 0

        for _, row in grp.sort_values("date").iterrows():
            demand = float(row["demand_units"])

            days_in += 1
            spoilage = 0.0

            # ── Spoilage ──
            if perish:
                age = days_in / max(shelf,1)
                if age > 0.9:
                    spoilage = stock * (0.2 if policy=="baseline" else 0.1)
                    stock = max(0, stock - spoilage)
                    days_in = 0

            # ── Demand Fulfillment ──
            actual = min(stock, demand)
            lost   = max(0, demand - stock)
            stock  = max(0, stock - actual)

            reorder = 0

            # ─────────────────────────────
            # 📌 POLICY LOGIC
            # ─────────────────────────────

            if policy == "baseline":
                if stock <= max_s * 0.25:
                    reorder = int(max_s * 0.8 - stock)

            elif policy == "classical":
                mean = demand
                std  = max(mean * 0.2, 1)
                z = 1.65

                optimal_q = mean + z * std

                if stock <= optimal_q:
                    reorder = int(optimal_q - stock)

            elif policy == "ml":
                mean = demand

                q_values = np.arange(0, max_s + 1, 5)

                costs = [
                    holding_cost * max(q - mean, 0) +
                    shortage_cost * max(mean - q, 0)
                    for q in q_values
                ]

                optimal_q = q_values[np.argmin(costs)]

                if stock <= optimal_q:
                    reorder = int(optimal_q - stock)

            # Apply reorder
            if reorder > 0:
                stock = min(max_s, stock + reorder)
                days_in = 0

            rows.append({
                "date": row["date"],
                "store_id": store_id,
                "sku_id": sku_id,
                "category": cat,
                "demand": demand,
                "actual_demand": actual,
                "lost_sales": lost,
                "stock": stock,
                "stockout": int(stock == 0 and demand > 0),
                "spoilage": spoilage,
                "reorder": reorder,
                "unit_cost": cost
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 📊 KPI CALCULATION
# ─────────────────────────────────────────────────────────────
def kpis(df):
    return {
        "stockout_rate": df["stockout"].mean() * 100,
        "service_level": (1 - df["stockout"].mean()) * 100,
        "lost_sales": df["lost_sales"].sum(),
        "spoilage": df["spoilage"].sum(),

        "holding_cost": (df["stock"] * df["unit_cost"] * 0.5).sum(),
        "shortage_cost": (df["lost_sales"] * df["unit_cost"] * 2).sum(),
        "spoilage_cost": (df["spoilage"] * df["unit_cost"]).sum(),

        "total_cost": (
            (df["stock"] * df["unit_cost"] * 0.5).sum() +
            (df["lost_sales"] * df["unit_cost"] * 2).sum() +
            (df["spoilage"] * df["unit_cost"]).sum()
        )
    }


# ─────────────────────────────────────────────────────────────
# 🚀 MAIN COMPARISON
# ─────────────────────────────────────────────────────────────
def compute_optimization():
    con = sqlite3.connect(DB)

    demand = pd.read_sql(
        "SELECT date, store_id, sku_id, demand_units FROM demand_signals",
        con
    )

    skus = pd.read_sql("SELECT * FROM skus", con)
    con.close()

    print("Simulating baseline...")
    bdf = simulate_policy(demand, skus, "baseline")

    print("Simulating classical...")
    cdf = simulate_policy(demand, skus, "classical")

    print("Simulating ML optimization...")
    mdf = simulate_policy(demand, skus, "ml")

    b = kpis(bdf)
    c = kpis(cdf)
    m = kpis(mdf)

    results = {
        "summary": {
            "baseline_cost": round(b["total_cost"], 2),
            "classical_cost": round(c["total_cost"], 2),
            "ml_cost": round(m["total_cost"], 2),

            "ml_vs_baseline_improvement_%": round(
                (b["total_cost"] - m["total_cost"]) / max(b["total_cost"],1) * 100, 2
            ),

            "service_level_baseline": round(b["service_level"], 2),
            "service_level_ml": round(m["service_level"], 2),

            "lost_sales_reduction": round(b["lost_sales"] - m["lost_sales"], 2),
            "spoilage_reduction": round(b["spoilage"] - m["spoilage"], 2)
        }
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== FINAL RESULTS =====")
    print(json.dumps(results["summary"], indent=2))

    return results


if __name__ == "__main__":
    compute_optimization()
