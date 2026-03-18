"""
utils/optimization_analysis.py  —  DarkIQ v2
Proper A/B simulation: both policies run on identical raw demand data.
  Baseline: fixed 25% ROP, no festival awareness, late spoilage detection
  AI system: dynamic ROP (demand-forecast-aware + festival signal), early spoilage scoring
"""
import sqlite3, json, os
import pandas as pd
import numpy as np
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), "..")
DB   = os.path.join(BASE, "instance", "darkiq.db")
OUT  = os.path.join(BASE, "instance", "optimization_results.json")

def simulate_policy(demand_df, skus_df, policy="baseline"):
    sku_dict = skus_df.set_index("sku_id").to_dict(orient="index")
    rows = []
    for (store_id, sku_id), grp in demand_df.groupby(["store_id","sku_id"]):
        sku    = sku_dict[sku_id]
        max_s  = sku["max_stock"]; shelf = sku["shelf_days"]
        perish = sku["is_perishable"]; cost = sku["unit_cost"]; cat = sku["category"]
        stock  = int(max_s * 0.62); days_in = 0
        for _, row in grp.sort_values("date").iterrows():
            demand = float(row["demand_units"])
            fm     = float(row.get("festival_mult",1.0))
            rn     = int(row.get("is_rain",0))
            days_in += 1
            spoilage = 0.0
            if perish:
                if policy == "baseline":
                    if days_in > shelf * 0.90:
                        spoilage = stock * 0.22; stock = max(0, stock-spoilage); days_in = 0
                else:
                    age   = days_in / max(shelf,1)
                    sr    = age*0.62 + age*(stock/max(max_s,1))*0.38
                    if sr >= 0.68:
                        spoilage = stock * 0.08; stock = max(0, stock-spoilage); days_in = 0
            actual = min(demand, stock); lost = max(0, demand-stock)
            stock = max(0, stock - actual)
            reorder = 0
            if policy == "baseline":
                if stock <= max_s * 0.25:
                    reorder = int(max_s*0.80) - stock; stock = min(max_s, stock+reorder); days_in=0
            else:
                sf = 0.12 if shelf<=5 else 0.16 if shelf<=14 else 0.20
                dd = demand * (fm if fm>1.1 else 1.0) * (1.30 if rn and cat in("dairy","bakery","staples") else 1.0)
                rop = dd*1 + max_s*sf
                if stock <= rop:
                    qty = max_s*0.90 - stock
                    if shelf<=7: qty = min(qty, dd*7*1.15)
                    reorder = max(0,round(qty)); stock = min(max_s, stock+reorder); days_in=0
            rows.append({"date":row["date"],"store_id":store_id,"sku_id":sku_id,
                "category":cat,"is_perishable":perish,"shelf_days":shelf,"max_stock":max_s,
                "unit_cost":cost,"demand":demand,"actual_demand":actual,"lost_sales":lost,
                "stock":stock,"stockout":int(stock==0 and demand>0),"spoilage":round(spoilage,2),
                "reorder":reorder,"stock_pct":round(stock/max(max_s,1)*100,1),"festival_mult":fm})
    return pd.DataFrame(rows)

def kpis(df):
    return {
        "stockout_rate":   df["stockout"].mean()*100,
        "service_level":   (1-df["stockout"].mean())*100,
        "total_spoilage":  df["spoilage"].sum(),
        "waste_rate":      df["spoilage"].sum()/max(df["demand"].sum(),1)*100,
        "avg_stock_pct":   df["stock_pct"].mean(),
        "holding_cost":    (df["stock"]*df["unit_cost"]).mean(),
        "reorder_events":  (df["reorder"]>0).sum(),
        "lost_sales":      df["lost_sales"].sum(),
        "avg_fill_rate":   (df["actual_demand"]/df["demand"].replace(0,np.nan)).mean()*100,
        "spoilage_cost":   (df["spoilage"]*df["unit_cost"]).sum(),
        "turnover":        df["demand"].sum()/max(df["stock"].mean(),1),
    }

def compute_optimization():
    con = sqlite3.connect(DB)
    demand = pd.read_sql(
        "SELECT d.date,d.store_id,d.sku_id,d.demand_units,"
        "d.festival_mult,d.day_mult,d.is_rain FROM demand_signals d "
        "ORDER BY store_id,sku_id,date", con)
    skus = pd.read_sql("SELECT * FROM skus", con)
    con.close()

    print("Simulating baseline (rule-based)…")
    bdf = simulate_policy(demand, skus, "baseline")
    print("Simulating AI-powered system…")
    adf = simulate_policy(demand, skus, "ai")

    b = kpis(bdf); a = kpis(adf)

    # Category breakdown
    cat_rows = []
    for cat in skus["category"].unique():
        bc = bdf[bdf.category==cat]; ac = adf[adf.category==cat]
        if not len(bc): continue
        cat_rows.append({
            "category":cat,
            "baseline_stockout_rate":round(bc["stockout"].mean()*100,2),
            "ai_stockout_rate":round(ac["stockout"].mean()*100,2),
            "stockout_reduction_pp":round((bc["stockout"].mean()-ac["stockout"].mean())*100,2),
            "baseline_spoilage":round(bc["spoilage"].sum(),0),
            "ai_spoilage":round(ac["spoilage"].sum(),0),
            "spoilage_reduction_pct":round((bc["spoilage"].sum()-ac["spoilage"].sum())/max(bc["spoilage"].sum(),1)*100,1),
            "baseline_service":round((1-bc["stockout"].mean())*100,1),
            "ai_service":round((1-ac["stockout"].mean())*100,1),
            "baseline_waste_rate":round(bc["spoilage"].sum()/max(bc["demand"].sum(),1)*100,2),
            "ai_waste_rate":round(ac["spoilage"].sum()/max(ac["demand"].sum(),1)*100,2),
            "baseline_fill_rate":round((bc["actual_demand"]/bc["demand"].replace(0,np.nan)).mean()*100,1),
            "ai_fill_rate":round((ac["actual_demand"]/ac["demand"].replace(0,np.nan)).mean()*100,1),
        })

    # Store breakdown
    store_rows = []
    for sid in demand["store_id"].unique():
        bs = bdf[bdf.store_id==sid]; as_ = adf[adf.store_id==sid]
        store_rows.append({
            "store_id":sid,
            "baseline_stockout_rate":round(bs["stockout"].mean()*100,2),
            "ai_stockout_rate":round(as_["stockout"].mean()*100,2),
            "stockout_reduction_pp":round((bs["stockout"].mean()-as_["stockout"].mean())*100,2),
            "baseline_service":round((1-bs["stockout"].mean())*100,1),
            "ai_service":round((1-as_["stockout"].mean())*100,1),
            "baseline_spoilage":round(bs["spoilage"].sum(),0),
            "ai_spoilage":round(as_["spoilage"].sum(),0),
            "baseline_lost_sales":round(bs["lost_sales"].sum(),0),
            "ai_lost_sales":round(as_["lost_sales"].sum(),0),
            "lost_sales_reduction":round(bs["lost_sales"].sum()-as_["lost_sales"].sum(),0),
        })

    # Weekly comparison
    for df,key in [(bdf,"base"),(adf,"ai")]:
        df["date"] = pd.to_datetime(df["date"])
        df["week"] = df["date"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    wb = bdf.groupby("week").agg(base_avail=("stockout",lambda x:round((1-x.mean())*100,1)),
                                  base_spoil=("spoilage","sum")).reset_index()
    wa = adf.groupby("week").agg(ai_avail=("stockout",lambda x:round((1-x.mean())*100,1)),
                                  ai_spoil=("spoilage","sum")).reset_index()
    weekly = wb.merge(wa,on="week").tail(52)

    results = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "baseline_stockout_rate":    round(b["stockout_rate"],3),
            "ai_stockout_rate":          round(a["stockout_rate"],3),
            "stockout_reduction_pp":     round(b["stockout_rate"]-a["stockout_rate"],3),
            "stockout_reduction_pct":    round((b["stockout_rate"]-a["stockout_rate"])/max(b["stockout_rate"],0.001)*100,1),
            "baseline_service_level":    round(b["service_level"],1),
            "ai_service_level":          round(a["service_level"],1),
            "service_level_improvement": round(a["service_level"]-b["service_level"],2),
            "baseline_spoilage_units":   round(b["total_spoilage"],0),
            "ai_spoilage_units":         round(a["total_spoilage"],0),
            "spoilage_units_saved":      round(b["total_spoilage"]-a["total_spoilage"],0),
            "spoilage_reduction_pct":    round((b["total_spoilage"]-a["total_spoilage"])/max(b["total_spoilage"],1)*100,1),
            "baseline_waste_rate":       round(b["waste_rate"],3),
            "ai_waste_rate":             round(a["waste_rate"],3),
            "waste_rate_reduction_pp":   round(b["waste_rate"]-a["waste_rate"],3),
            "spoilage_cost_saved_inr":   round(b["spoilage_cost"]-a["spoilage_cost"],0),
            "baseline_avg_stock_pct":    round(b["avg_stock_pct"],1),
            "ai_avg_stock_pct":          round(a["avg_stock_pct"],1),
            "holding_cost_reduction_pct":round((b["holding_cost"]-a["holding_cost"])/max(b["holding_cost"],1)*100,1),
            "baseline_inventory_turnover":round(b["turnover"],2),
            "ai_inventory_turnover":     round(a["turnover"],2),
            "turnover_improvement_pct":  round((a["turnover"]-b["turnover"])/max(b["turnover"],1)*100,1),
            "baseline_fill_rate":        round(b["avg_fill_rate"],2),
            "ai_fill_rate":              round(a["avg_fill_rate"],2),
            "fill_rate_improvement":     round(a["avg_fill_rate"]-b["avg_fill_rate"],2),
            "baseline_lost_sales":       round(b["lost_sales"],0),
            "ai_lost_sales":             round(a["lost_sales"],0),
            "lost_sales_reduction":      round(b["lost_sales"]-a["lost_sales"],0),
            "lost_sales_reduction_pct":  round((b["lost_sales"]-a["lost_sales"])/max(b["lost_sales"],1)*100,1),
            "baseline_reorder_events":   int(b["reorder_events"]),
            "ai_reorder_events":         int(a["reorder_events"]),
            "total_records_analysed":    len(adf),
            "date_range":                f"{adf['date'].min().date()} to {adf['date'].max().date()}",
            "stores":4,"skus":20,
        },
        "category_breakdown": cat_rows,
        "store_breakdown":    store_rows,
        "weekly_comparison":  weekly.to_dict(orient="records"),
    }
    with open(OUT,"w") as f:
        json.dump(results,f,indent=2,default=str)

    s = results["summary"]
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║           AI OPTIMIZATION RESULTS vs BASELINE               ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Stockout rate:       {s['baseline_stockout_rate']:6.2f}% → {s['ai_stockout_rate']:6.2f}%  ↓{s['stockout_reduction_pct']:5.1f}%  ║")
    print(f"║  Service level:       {s['baseline_service_level']:6.1f}% → {s['ai_service_level']:6.1f}%  +{s['service_level_improvement']:5.2f}pp ║")
    print(f"║  Fill rate:           {s['baseline_fill_rate']:6.2f}% → {s['ai_fill_rate']:6.2f}%  +{s['fill_rate_improvement']:5.2f}pp ║")
    print(f"║  Spoilage (units): {s['baseline_spoilage_units']:8.0f} → {s['ai_spoilage_units']:8.0f}  ↓{s['spoilage_reduction_pct']:5.1f}%  ║")
    print(f"║  Waste rate:          {s['baseline_waste_rate']:6.2f}% → {s['ai_waste_rate']:6.2f}%  ↓{s['waste_rate_reduction_pp']:.3f}pp ║")
    print(f"║  Lost sales (units):{s['baseline_lost_sales']:8.0f} → {s['ai_lost_sales']:8.0f}  ↓{s['lost_sales_reduction_pct']:5.1f}%  ║")
    print(f"║  Holding cost:                    ↓{s['holding_cost_reduction_pct']:5.1f}% reduction       ║")
    print(f"║  Inventory turnover:  {s['baseline_inventory_turnover']:6.1f}x → {s['ai_inventory_turnover']:6.1f}x  +{s['turnover_improvement_pct']:4.1f}%  ║")
    print(f"║  Spoilage cost saved:  ₹{s['spoilage_cost_saved_inr']:>12,.0f}                      ║")
    print(f"║  Records: {s['total_records_analysed']:,} | Stores: {s['stores']} | SKUs: {s['skus']}               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n✓ Saved → {OUT}")
    return results

if __name__ == "__main__":
    compute_optimization()