"""
models/reorder_engine.py  —  DarkIQ v2
Adaptive reorder engine pulling live state from SQLite.
Writes AI decisions back to the database.
"""
import sqlite3, os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

BASE = os.path.join(os.path.dirname(__file__), "..")
DB   = os.path.join(BASE, "instance", "darkiq.db")

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class SKUState:
    sku_id: str; 
    name: str; 
    category: str;
    stock: float; 
    max_stock: float; 
    reorder_pt: float
    shelf_days: int; 
    days_in_stock: int;
    unit_cost: float; 
    sell_price: float
    store_id: str; 
    is_perishable: int = 0;
    predicted_demand_7d: float = 0.0;
    holding_cost: float = 0.5;
    shortage_cost: float = 5.0;
    ordering_cost: float = 20.0;

@dataclass
class Decision:
    sku_id: str
    sku_name: str
    store_id: str

    # ML vs Classical comparison
    ml_qty: float = 0
    ml_cost: float = 0
    classical_qty: float = 0
    classical_cost: float = 0

    # Decision output
    action: str = ""
    urgency: str = ""
    qty: float = 0

    reason: str = ""
    cost_impact: float = 0
    confidence: float = 0

    days_cover: float = 0.0

@dataclass
class Transfer:
    sku_id: str; sku_name: str
    from_store: str; to_store: str
    qty: int; reason: str

# ── Engine ────────────────────────────────────────────────────────────────────
class ReorderEngine:
    def __init__(self, safety_factor=0.20, lead_time=1):
        self.safety_factor = safety_factor
        self.lead_time     = lead_time

    def _daily_demand(self, s: SKUState) -> float:
        return s.predicted_demand_7d / 7 if s.predicted_demand_7d > 0 else s.max_stock * 0.05

    def dynamic_rop(self, s: SKUState) -> float:
        """Reorder Point = demand during lead time + safety stock."""
        dd = self._daily_demand(s)
        sf = (self.safety_factor * 0.55 if s.shelf_days <= 5
              else self.safety_factor * 0.75 if s.shelf_days <= 14
              else self.safety_factor)
        return dd * self.lead_time + s.max_stock * sf

    def order_qty(self, s: SKUState) -> float:
        target = s.max_stock * 0.92
        qty    = target - s.stock
        if s.shelf_days <= 7:
            cap = s.predicted_demand_7d * 1.15 if s.predicted_demand_7d else qty
            qty = min(qty, cap)
        return max(0, round(qty))

    def spoilage_score(self, s: SKUState) -> float:
        if s.shelf_days > 30: return 0.0
        age   = s.days_in_stock / max(s.shelf_days, 1)
        stock = s.stock / max(s.max_stock, 1)
        return min(1.0, age*0.62 + age*stock*0.38)

    def compute_cost(self, Q, D, s: SKUState):
        holding = s.holding_cost * max(Q - D, 0)
        shortage = s.shortage_cost * max(D - Q, 0)
        return holding + shortage

    def classical_q(self, mean, std, service_level=0.95):
        z = 1.65  # approx for 95%
        return mean + z * std

    def optimize_q(self, s: SKUState):
        D = s.predicted_demand_7d
        std = max(D * 0.2, 1)

        q_values = np.arange(0, s.max_stock + 1, 5)
        costs = [self.compute_cost(q, D, s) for q in q_values]

        best_q = q_values[np.argmin(costs)]
        best_cost = min(costs)

        return best_q, best_cost
    
    def evaluate(self, s: SKUState) -> Decision:
        rop      = self.dynamic_rop(s)
        sp_risk  = self.spoilage_score(s)
        dd       = self._daily_demand(s)
        days_cov = s.stock / max(dd, 0.01)
        pct      = s.stock / max(s.max_stock, 1)

        if s.stock == 0:
            qty = self.order_qty(s)
            return Decision(s.sku_id, s.name, s.store_id, "critical", "critical",
                qty, "STOCKOUT — immediate emergency reorder required",
                round(qty * s.unit_cost, 2), 1.0, 0.0)

        if sp_risk >= 0.70:
            mv = round(s.stock * 0.60)
            return Decision(s.sku_id, s.name, s.store_id, "mark_down", "high",
                mv, f"Spoilage risk {sp_risk:.0%} — day {s.days_in_stock}/{s.shelf_days}. "
                    f"Move {mv} units via markdown or inter-store transfer.",
                round(mv * s.unit_cost * -0.25, 2), round(sp_risk, 2), days_cov)

        if s.stock <= rop and s.stock > 0:
            qty     = self.order_qty(s)
            urgency = ("critical" if days_cov < 1 else
                       "high"     if days_cov < 2 else "medium")
            return Decision(s.sku_id, s.name, s.store_id, "reorder", urgency,
                qty, f"Stock {s.stock:.0f} ≤ ROP {rop:.0f}. Days cover: {days_cov:.1f}. "
                    f"Forecast 7d demand: {s.predicted_demand_7d:.0f} units.",
                round(qty * s.unit_cost, 2), 0.88, days_cov)

        # ML optimization
        q_ml, cost_ml = self.optimize_q(s)

        # Classical model
        std = max(s.predicted_demand_7d * 0.2, 1)
        q_classic = self.classical_q(s.predicted_demand_7d, std)
        cost_classic = self.compute_cost(q_classic, s.predicted_demand_7d, s)

        return Decision(
    sku_id=s.sku_id,
    sku_name=s.name,
    store_id=s.store_id,

    ml_qty=q_ml,
    ml_cost=cost_ml,
    classical_qty=q_classic,
    classical_cost=cost_classic,

    action="ok",
    urgency="low",
    qty=0,

    reason=f"Healthy — {pct:.0%} stock, {days_cov:.1f}d cover.",
    cost_impact=0.0,
    confidence=0.95,
    days_cover=days_cov
)

    def recommend_transfers(self, states: List[SKUState]) -> List[Transfer]:
        transfers = []
        by_sku = {}
        for s in states:
            by_sku.setdefault(s.sku_id, []).append(s)

        for sku_id, ss in by_sku.items():
            surplus = [s for s in ss if s.stock/max(s.max_stock,1) > 0.62
                       and self.spoilage_score(s) > 0.35]
            deficit = [s for s in ss if s.stock <= self.dynamic_rop(s)]
            for sup in surplus:
                for def_ in deficit:
                    if sup.store_id == def_.store_id: continue
                    qty = min(round(sup.stock*0.28), round(def_.max_stock - def_.stock))
                    if qty > 0:
                        transfers.append(Transfer(
                            sku_id, sup.name, sup.store_id, def_.store_id, qty,
                            f"{sup.store_id} surplus {sup.stock:.0f} units "
                            f"(spoilage {self.spoilage_score(sup):.0%}) → "
                            f"{def_.store_id} below ROP {self.dynamic_rop(def_):.0f}"
                        ))
        return transfers

# ── DB helpers ────────────────────────────────────────────────────────────────
def latest_snapshot(store_id: Optional[str] = None) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    q = """
        SELECT i.store_id, i.sku_id, s.name, s.category, s.shelf_days,
               s.max_stock, s.reorder_pt, s.unit_cost, s.sell_price, s.is_perishable,
               i.closing_stock AS stock_level,
               ROUND(i.closing_stock*100.0/s.max_stock,1) AS stock_pct,
               i.spoilage_units, i.reorder_qty, i.stockout,
               i.demand_units, i.days_in_stock, i.date
        FROM inventory i JOIN skus s ON i.sku_id = s.sku_id
        WHERE i.date = (SELECT MAX(date) FROM inventory)
    """
    if store_id:
        q += " AND i.store_id = ?"
        df = pd.read_sql(q, con, params=(store_id,))
    else:
        df = pd.read_sql(q, con)
    con.close()
    return df

def run_decisions(store_id: Optional[str] = None, save: bool = True,
                  safety_factor: float = 0.20) -> dict:
    snap    = latest_snapshot(store_id)
    engine  = ReorderEngine(safety_factor=safety_factor)
    states  = []
    for _, r in snap.iterrows():
        states.append(SKUState(
            sku_id=r.sku_id, name=r.name, category=r.category,
            stock=r.stock_level, max_stock=r.max_stock, reorder_pt=r.reorder_pt,
            shelf_days=r.shelf_days, days_in_stock=r.days_in_stock,
            unit_cost=r.unit_cost, sell_price=r.sell_price,
            store_id=r.store_id, is_perishable=r.is_perishable,
            predicted_demand_7d=r.demand_units * 7
        ))

    decisions  = [engine.evaluate(s) for s in states]
    transfers  = engine.recommend_transfers(states)

    if save:
        con = sqlite3.connect(DB)
        for d in decisions:
            if d.action != "ok":
                con.execute(
                    "INSERT INTO ai_decisions(store_id,sku_id,action,urgency,qty,"
                    "reason,cost_impact,confidence) VALUES(?,?,?,?,?,?,?,?)",
                    (d.store_id,d.sku_id,d.action,d.urgency,d.qty,
                     d.reason,d.cost_impact,d.confidence)
                )
        for t in transfers:
            con.execute(
                "INSERT INTO transfer_log(sku_id,from_store,to_store,qty,reason) VALUES(?,?,?,?,?)",
                (t.sku_id,t.from_store,t.to_store,t.qty,t.reason)
            )
        con.commit(); con.close()

    total     = len(decisions)
    stockouts = sum(1 for d in decisions if d.action == "critical")
    reorders  = sum(1 for d in decisions if d.action == "reorder")
    markdowns = sum(1 for d in decisions if d.action == "mark_down")
    avail     = round((total - stockouts) / max(total,1) * 100, 1)

    return {
        "decisions": [vars(d) for d in decisions if d.action != "ok"],
        "all_decisions": [vars(d) for d in decisions],
        "transfers": [vars(t) for t in transfers],
        "kpis": {
            "availability_rate": avail,
            "stockouts": stockouts, "reorders": reorders,
            "markdowns": markdowns,
            "transfers": len(transfers),
            "reorder_spend": round(sum(d.cost_impact for d in decisions if d.action=="reorder"),2),
        }
    }


if __name__ == "__main__":
    r = run_decisions(save=False)
    print("KPIs:", r["kpis"])
    print(f"Active decisions: {len(r['decisions'])}")
    print(f"Transfers: {len(r['transfers'])}")
    print("✓ Reorder engine OK")