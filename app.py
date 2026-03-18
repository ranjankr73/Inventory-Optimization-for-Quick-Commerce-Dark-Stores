"""
app.py  —  DarkIQ v2
Flask application with session-based authentication, role-based views,
and full REST API backed by SQLite.
"""
import os, sys, sqlite3, hashlib, json
from functools import wraps
from datetime import datetime, timedelta
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, g)
import pandas as pd
import numpy as np

BASE = os.path.dirname(__file__)
sys.path.insert(0, BASE)

from models.reorder_engine import run_decisions, latest_snapshot, ReorderEngine, SKUState
from models.forecaster import predict as fc_predict, get_bundle

app = Flask(__name__)
app.secret_key = "darkiq-v2-secret-2024"
DB  = os.path.join(BASE, "instance", "darkiq.db")

# ── DB helper ──────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db: db.close()

def qry(sql, params=(), one=False):
    cur = get_db().execute(sql, params)
    r   = cur.fetchone() if one else cur.fetchall()
    return r

def sha(pw): return hashlib.sha256(pw.encode()).hexdigest()

# ── Auth ───────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if session.get("role") not in roles:
                return jsonify({"error": "Forbidden"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

# ── Routes: Auth ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        user = qry("SELECT * FROM users WHERE username=? AND password=?",
                   (username, sha(password)), one=True)
        if user:
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            session["store_id"] = user["store_id"]
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html",
        username=session["username"], role=session["role"],
        store_id=session.get("store_id"))

# ── API helpers ────────────────────────────────────────────────────────────────
def _store_scope():
    """Return the store filter based on role."""
    role     = session.get("role")
    store_id = request.args.get("store") or session.get("store_id")
    if role == "viewer" and session.get("store_id"):
        store_id = session["store_id"]
    return store_id

def _npclean(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# ── API: Auth check ────────────────────────────────────────────────────────────
@app.route("/api/me")
@login_required
def api_me():
    stores = [dict(r) for r in qry("SELECT store_id, name FROM stores WHERE active=1")]
    return jsonify({"username": session["username"], "role": session["role"],
                    "store_id": session.get("store_id"), "stores": stores})

# ── API: Dashboard summary ────────────────────────────────────────────────────
@app.route("/api/dashboard")
@login_required
def api_dashboard():
    store = _store_scope()
    result = run_decisions(store_id=store, save=False)
    kpis   = result["kpis"]

    snap = latest_snapshot(store)
    low     = int((snap.stock_pct < 25).sum())
    ok_     = int(((snap.stock_pct >= 25) & (snap.stock_pct < 70)).sum())
    healthy = int((snap.stock_pct >= 70).sum())

    # 7-day trend from DB
    con = sqlite3.connect(DB)
    q = "SELECT date, AVG(1-stockout)*100 AS avail FROM inventory"
    if store: q += " WHERE store_id=?"
    q += " GROUP BY date ORDER BY date DESC LIMIT 7"
    trend = pd.read_sql(q, con, params=(store,) if store else ())
    con.close()

    bundle  = get_bundle()
    metrics = bundle["metrics"]

    return jsonify({
        "kpis": kpis,
        "health": {"low": low, "ok": ok_, "healthy": healthy},
        "trend_7d": trend[["date","avail"]].to_dict(orient="records"),
        "model_metrics": metrics,
        "role": session["role"],
    })

# ── API: Inventory ─────────────────────────────────────────────────────────────
@app.route("/api/inventory")
@login_required
def api_inventory():
    store = _store_scope()
    snap  = latest_snapshot(store)
    snap["status"] = snap["stock_pct"].apply(
        lambda x: "critical" if x == 0 else "low" if x < 25
        else "warning" if x < 50 else "ok")
    cols = ["store_id","sku_id","name","category","stock_level","max_stock","stock_pct",
            "spoilage_units","reorder_qty","stockout","status","shelf_days",
            "unit_cost","sell_price","days_in_stock"]
    return jsonify(snap[cols].fillna(0).to_dict(orient="records"))

# ── API: AI Decisions ──────────────────────────────────────────────────────────
@app.route("/api/decisions")
@login_required
def api_decisions():
    store = _store_scope()
    result = run_decisions(store_id=store, save=False)
    decs   = sorted(result["decisions"],
                    key=lambda d: {"critical":0,"high":1,"medium":2,"low":3}.get(d["urgency"],4))
    return jsonify({"decisions": decs, "kpis": result["kpis"],
                    "transfers": result["transfers"]})

# ── API: Forecast ──────────────────────────────────────────────────────────────
@app.route("/api/forecast/<sku_id>")
@login_required
def api_forecast(sku_id):
    store = _store_scope() or "DS_NORTH"
    con   = sqlite3.connect(DB)
    hist  = pd.read_sql(
        "SELECT date, demand_units FROM demand_signals "
        "WHERE sku_id=? AND store_id=? ORDER BY date",
        con, params=(sku_id, store))
    sku_info = pd.read_sql("SELECT * FROM skus WHERE sku_id=?", con, params=(sku_id,))
    con.close()

    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.tail(60)
    vals = hist["demand_units"].values
    roll7  = float(np.mean(vals[-7:]))  if len(vals) >= 7  else float(np.mean(vals))
    roll14 = float(np.mean(vals[-14:])) if len(vals) >= 14 else roll7
    std7   = float(np.std(vals[-7:]))   if len(vals) >= 7  else roll7*0.16

    cat = sku_info["category"].iloc[0] if len(sku_info) else "staples"
    name = sku_info["name"].iloc[0]    if len(sku_info) else sku_id

    forecasts = []
    for i in range(1, 8):
        fd  = datetime.now() + timedelta(days=i)
        ctx = {
            "dow": fd.weekday(), "month": fd.month,
            "woy": fd.isocalendar()[1], "dom": fd.day,
            "quarter": (fd.month-1)//3+1, "is_wknd": int(fd.weekday()>=5),
            "lag_1": float(vals[-1]) if len(vals) else roll7,
            "lag_3": float(vals[-3]) if len(vals)>=3 else roll7,
            "lag_7": float(vals[-7]) if len(vals)>=7 else roll7,
            "lag_14": float(vals[-14]) if len(vals)>=14 else roll7,
            "roll_7_mean": roll7, "roll_14_mean": roll14, "roll_7_std": std7,
            "ema_7": roll7,
            "weather_mult": 1.0, "festival_mult": 1.0,
            "day_mult": [0.82,0.86,0.90,0.93,1.12,1.38,1.28][fd.weekday()],
            "is_rain": 0, "temp_c": 26.0, "recent_14": list(vals[-14:]),
        }
        p = fc_predict(store, sku_id, cat, ctx)
        forecasts.append({"date": fd.strftime("%Y-%m-%d"), **p})

    history = hist.tail(30).copy()
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")
    return jsonify({
        "sku_id": sku_id, "sku_name": name, "store": store,
        "history": history.to_dict(orient="records"),
        "forecast": forecasts, "avg_demand_7d": round(roll7, 1),
    })

# ── API: Analytics ─────────────────────────────────────────────────────────────
@app.route("/api/analytics")
@login_required
def api_analytics():
    con = sqlite3.connect(DB)
    # Weekly trend (last 52 weeks)
    weekly = pd.read_sql(
        "SELECT strftime('%Y-%W', date) AS week, "
        "  ROUND(AVG(1-stockout)*100,1) AS availability, "
        "  SUM(spoilage_units) AS spoilage, "
        "  SUM(CASE WHEN reorder_qty>0 THEN 1 ELSE 0 END) AS reorders "
        "FROM inventory GROUP BY week ORDER BY week DESC LIMIT 52", con)
    # Category performance
    cat_perf = pd.read_sql(
        "SELECT s.category, "
        "  ROUND(AVG(i.stockout)*100,2) AS stockout_rate, "
        "  ROUND(AVG(i.closing_stock*100.0/s.max_stock),1) AS avg_stock_pct, "
        "  ROUND(SUM(i.spoilage_units),0) AS total_spoilage "
        "FROM inventory i JOIN skus s ON i.sku_id=s.sku_id "
        "GROUP BY s.category", con)
    # Store comparison
    store_comp = pd.read_sql(
        "SELECT store_id, "
        "  ROUND(AVG(1-stockout)*100,1) AS availability, "
        "  ROUND(SUM(spoilage_units),0) AS total_spoilage, "
        "  ROUND(AVG(closing_stock*100.0/max_stock),1) AS avg_stock_pct "
        "FROM inventory i JOIN skus s ON i.sku_id=s.sku_id "
        "GROUP BY store_id", con)
    con.close()

    bundle = get_bundle()
    return jsonify({
        "weekly_trend":          weekly.iloc[::-1].to_dict(orient="records"),
        "category_performance":  cat_perf.to_dict(orient="records"),
        "store_comparison":      store_comp.to_dict(orient="records"),
        "model_metrics":         bundle["metrics"],
        "feature_importances":   bundle["feature_importances"][:10],
    })

# ── API: Network (multi-store) ─────────────────────────────────────────────────
@app.route("/api/network")
@login_required
def api_network():
    result = run_decisions(store_id=None, save=False)
    snap   = latest_snapshot()

    store_summary = []
    for sid in snap["store_id"].unique():
        s = snap[snap.store_id == sid]
        store_summary.append({
            "store_id":     sid,
            "availability": round((s.stock_pct > 0).mean()*100, 1),
            "avg_stock_pct":round(s.stock_pct.mean(), 1),
            "low_skus":     int((s.stock_pct < 25).sum()),
            "spoilage":     round(s.spoilage_units.sum(), 1),
            "reorders":     int((s.reorder_qty > 0).sum()),
        })

    return jsonify({"stores": store_summary, "transfers": result["transfers"]})

# ── API: Simulate event ────────────────────────────────────────────────────────
@app.route("/api/simulate", methods=["POST"])
@login_required
def api_simulate():
    data  = request.json or {}
    event = data.get("event", "none")
    store = data.get("store") or _store_scope()
    snap  = latest_snapshot(store).copy()

    multipliers = {"rain":1.45, "festival":1.75, "supply_delay":0.55, "heatwave":1.25, "none":1.0}
    mult = multipliers.get(event, 1.0)

    # Apply event shock to stock
    if event in ("rain","festival","heatwave"):
        snap["stock_level"]  = (snap["stock_level"] / mult).clip(lower=0).round()
    elif event == "supply_delay":
        snap["stock_level"]  = (snap["stock_level"] * mult).clip(lower=0).round()

    snap["stock_pct"] = (snap["stock_level"] / snap["max_stock"] * 100).round(1)
    snap["demand_units"] *= mult

    engine = ReorderEngine()
    states = [SKUState(
        sku_id=r.sku_id, name=r.name, category=r.category,
        stock=r.stock_level, max_stock=r.max_stock, reorder_pt=r.reorder_pt,
        shelf_days=r.shelf_days, days_in_stock=r.days_in_stock,
        unit_cost=r.unit_cost, sell_price=r.sell_price, store_id=r.store_id,
        is_perishable=r.is_perishable, predicted_demand_7d=r.demand_units*7
    ) for _, r in snap.iterrows()]

    decs = [engine.evaluate(s) for s in states]
    kpis = {
        "availability_rate": round(sum(1 for d in decs if d.action!="critical")/max(len(decs),1)*100,1),
        "stockouts":  sum(1 for d in decs if d.action=="critical"),
        "reorders":   sum(1 for d in decs if d.action=="reorder"),
        "markdowns":  sum(1 for d in decs if d.action=="mark_down"),
        "reorder_spend": round(sum(d.cost_impact for d in decs if d.action=="reorder"),2),
    }
    msgs = {
        "rain":          "Heavy rain: essentials demand +45%. AI triggered preemptive reorders.",
        "festival":      "Festival surge: demand +75% across all categories. Emergency restocking activated.",
        "supply_delay":  "Supplier delay: stock down 45%. AI flagged 7-14 day buffer strategy.",
        "heatwave":      "Heatwave: beverages & dairy demand +25%. Cold chain priority escalated.",
        "none":          "Normal day. AI monitoring nominal across all SKUs.",
    }

    return jsonify({
        "event": event, "multiplier": mult, "kpis": kpis,
        "decisions": [vars(d) for d in decs if d.action != "ok"][:15],
        "message": msgs.get(event,""),
    })

# ── API: SKU list (for dropdowns) ─────────────────────────────────────────────
@app.route("/api/skus")
@login_required
def api_skus():
    rows = qry("SELECT sku_id, name, category FROM skus ORDER BY category, name")
    return jsonify([dict(r) for r in rows])

# ── API: Decision history from DB ─────────────────────────────────────────────
@app.route("/api/history")
@login_required
def api_history():
    store = _store_scope()
    q = ("SELECT d.*, s.name AS sku_name FROM ai_decisions d "
         "JOIN skus s ON d.sku_id=s.sku_id WHERE d.resolved=0")
    params = ()
    if store:
        q += " AND d.store_id=?"; params = (store,)
    q += " ORDER BY d.created_at DESC LIMIT 50"
    rows = qry(q, params)
    return jsonify([dict(r) for r in rows])

if __name__ == "__main__":
    print("DarkIQ v2 starting on http://localhost:5050")
    print("Login: admin/admin123 | manager/manager123 | viewer/viewer123")
    app.run(debug=True, port=5050)

# ── API: Optimization A/B results ─────────────────────────────────────────────
@app.route("/api/optimization")
@login_required
def api_optimization():
    opt_path = os.path.join(BASE, "instance", "optimization_results.json")
    if not os.path.exists(opt_path):
        return jsonify({"error": "Run utils/optimization_analysis.py first"}), 404
    with open(opt_path) as f:
        return jsonify(json.load(f))