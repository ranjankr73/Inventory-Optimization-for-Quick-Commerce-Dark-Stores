"""
models/forecaster.py  —  DarkIQ v2
Three-model demand forecasting pipeline:
  1. RandomForest       — captures non-linear feature interactions
  2. GradientBoosting   — sequential error correction, best on festivals
  3. SequenceModel      — sliding-window linear regressor mimicking LSTM structure
                          (uses recent 14-day windows, matches LSTM philosophy
                          without requiring TensorFlow)
Final prediction = performance-weighted ensemble of all three.
"""
import os, sys, pickle, sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE  = os.path.join(os.path.dirname(__file__), "..")
DB    = os.path.join(BASE, "instance", "darkiq.db")
MDIR  = os.path.dirname(__file__)

WINDOW = 14   # days of history used by the sequence model

# ── Load data from SQLite ─────────────────────────────────────────────────────
def load_data():
    con = sqlite3.connect(DB)
    demand = pd.read_sql("SELECT * FROM demand_signals ORDER BY date,store_id,sku_id", con)
    skus   = pd.read_sql("SELECT sku_id, category, shelf_days, max_stock FROM skus", con)
    con.close()
    demand["date"] = pd.to_datetime(demand["date"])
    return demand.merge(skus, on="sku_id", how="left")

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer(df: pd.DataFrame):
    df = df.sort_values(["store_id","sku_id","date"]).copy()

    # Calendar
    df["dow"]    = df["date"].dt.dayofweek
    df["month"]  = df["date"].dt.month
    df["woy"]    = df["date"].dt.isocalendar().week.astype(int)
    df["dom"]    = df["date"].dt.day
    df["is_wknd"]= (df["dow"] >= 5).astype(int)
    df["quarter"]= df["date"].dt.quarter

    # Lag + rolling (per store+sku)
    grp = df.groupby(["store_id","sku_id"])["demand_units"]
    for lag in [1,3,7,14]:
        df[f"lag_{lag}"] = grp.shift(lag)
    df["roll_7_mean"]  = grp.shift(1).transform(lambda x: x.rolling(7,  min_periods=1).mean())
    df["roll_14_mean"] = grp.shift(1).transform(lambda x: x.rolling(14, min_periods=2).mean())
    df["roll_7_std"]   = grp.shift(1).transform(lambda x: x.rolling(7,  min_periods=2).std()).fillna(0)
    df["ema_7"]        = grp.shift(1).transform(lambda x: x.ewm(span=7, adjust=False).mean())

    # Encode categoricals
    df["store_enc"] = pd.factorize(df["store_id"])[0]
    df["sku_enc"]   = pd.factorize(df["sku_id"])[0]
    df["cat_enc"]   = pd.factorize(df["category"])[0]

    df = df.dropna()
    return df

FEATURES = [
    "store_enc","sku_enc","cat_enc",
    "dow","month","woy","dom","is_wknd","quarter",
    "lag_1","lag_3","lag_7","lag_14",
    "roll_7_mean","roll_14_mean","roll_7_std","ema_7",
    "weather_mult","festival_mult","day_mult","is_rain","temp_c",
]

# ── Sequence feature builder (LSTM-style sliding window) ─────────────────────
def build_sequence_features(df: pd.DataFrame, window=WINDOW):
    """
    For each sample, flatten the last `window` days of demand into a feature vector.
    This gives the Ridge regressor a sequence-aware input, approximating what an
    LSTM would receive — without requiring deep learning infrastructure.
    """
    records = []
    groups = df.groupby(["store_id","sku_id"])
    for (sid, skid), g in groups:
        g = g.sort_values("date").reset_index(drop=True)
        orig_indices = g.index.tolist()  # positional indices in df after reset
        vals = g["demand_units"].values
        for i in range(window, len(g)):
            window_feats = vals[i-window:i]
            trend = np.polyfit(range(window), window_feats, 1)[0]
            seasonal = window_feats[i % 7 if (i % 7) < len(window_feats) else -1]
            row = {
                "df_loc": g.iloc[i].name,  # actual df index
                **{f"seq_{j}": window_feats[j] for j in range(window)},
                "seq_trend": trend,
                "seq_seasonal_ref": seasonal,
                "store_enc": g["store_enc"].iloc[i],
                "sku_enc":   g["sku_enc"].iloc[i],
                "festival_mult": g["festival_mult"].iloc[i],
                "day_mult":  g["day_mult"].iloc[i],
                "is_wknd":   g["is_wknd"].iloc[i],
                "demand_units": vals[i],
            }
            records.append(row)
    return pd.DataFrame(records)

# ── Train ──────────────────────────────────────────────────────────────────────
def train():
    print("Loading data from DB…")
    df = load_data()
    df = engineer(df)

    X = df[FEATURES].values
    y = df["demand_units"].values

    # Temporal split (last 20% = test)
    split = int(len(df) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    df_te = df.iloc[split:]

    # ── Model 1: Random Forest ─────────────────────────────────────────────
    print("Training RandomForest…")
    rf = RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=2,
                               max_features=0.7, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_pred = np.maximum(0, rf.predict(X_te))
    rf_mae  = mean_absolute_error(y_te, rf_pred)
    rf_r2   = r2_score(y_te, rf_pred)
    print(f"  RF  → MAE: {rf_mae:.3f}  R²: {rf_r2:.4f}")

    # ── Model 2: Gradient Boosting ─────────────────────────────────────────
    print("Training GradientBoosting…")
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.07,
                                   subsample=0.75, min_samples_leaf=3, random_state=42)
    gb.fit(X_tr, y_tr)
    gb_pred = np.maximum(0, gb.predict(X_te))
    gb_mae  = mean_absolute_error(y_te, gb_pred)
    gb_r2   = r2_score(y_te, gb_pred)
    print(f"  GB  → MAE: {gb_mae:.3f}  R²: {gb_r2:.4f}")

    # ── Model 3: Sequence Ridge ("LSTM-style") ─────────────────────────────
    print("Building sequence features…")
    seq_df   = build_sequence_features(df)
    seq_cols = [c for c in seq_df.columns if c not in ("df_loc","demand_units")]
    Xs = seq_df[seq_cols].values
    ys = seq_df["demand_units"].values

    scaler = StandardScaler()
    Xs_sc  = scaler.fit_transform(Xs)

    split_s = int(len(Xs) * 0.80)
    seq_model = Ridge(alpha=1.0)
    seq_model.fit(Xs_sc[:split_s], ys[:split_s])
    seq_pred_full = np.maximum(0, seq_model.predict(Xs_sc))
    seq_pred_te   = seq_pred_full[split_s:]
    ys_te_seq     = ys[split_s:]

    # Align test lengths for ensemble weighting
    n_align = min(len(y_te), len(seq_pred_te))
    y_te_a  = y_te[-n_align:]
    rf_a    = rf_pred[-n_align:]
    gb_a    = gb_pred[-n_align:]
    sq_a    = seq_pred_te[-n_align:]

    seq_mae = mean_absolute_error(ys_te_seq, seq_pred_te)
    seq_r2  = r2_score(ys_te_seq, seq_pred_te)
    print(f"  SEQ → MAE: {seq_mae:.3f}  R²: {seq_r2:.4f}")

    # ── Ensemble weights (inverse-MAE) ────────────────────────────────────
    maes = np.array([rf_mae, gb_mae, seq_mae])
    inv  = 1.0 / maes
    wts  = inv / inv.sum()
    ens_pred = wts[0]*rf_a + wts[1]*gb_a + wts[2]*sq_a
    ens_mae  = mean_absolute_error(y_te_a, ens_pred)
    ens_rmse = np.sqrt(mean_squared_error(y_te_a, ens_pred))
    ens_r2   = r2_score(y_te_a, ens_pred)
    print(f"  ENS → MAE: {ens_mae:.3f}  RMSE: {ens_rmse:.3f}  R²: {ens_r2:.4f}")
    print(f"  Weights: RF={wts[0]:.3f}  GB={wts[1]:.3f}  SEQ={wts[2]:.3f}")

    # ── Feature importances ───────────────────────────────────────────────
    fi_df = pd.DataFrame({"feature": FEATURES, "importance": rf.feature_importances_})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

    bundle = {
        "rf": rf, "gb": gb, "seq_model": seq_model, "scaler": scaler,
        "weights": {"rf": float(wts[0]), "gb": float(wts[1]), "seq": float(wts[2])},
        "features": FEATURES, "seq_cols": seq_cols, "window": WINDOW,
        "store_map": {s: i for i,s in enumerate(df["store_id"].unique())},
        "sku_map":   {s: i for i,s in enumerate(df["sku_id"].unique())},
        "cat_map":   {s: i for i,s in enumerate(df["category"].unique())},
        "metrics": {
            "rf_mae": float(rf_mae), "rf_r2": float(rf_r2),
            "gb_mae": float(gb_mae), "gb_r2": float(gb_r2),
            "seq_mae": float(seq_mae), "seq_r2": float(seq_r2),
            "ens_mae": float(ens_mae), "ens_rmse": float(ens_rmse), "ens_r2": float(ens_r2),
        },
        "feature_importances": fi_df.to_dict(orient="records"),
    }

    out = os.path.join(MDIR, "bundle_v2.pkl")
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    fi_df.to_csv(os.path.join(MDIR, "feature_importances.csv"), index=False)
    print(f"\n✓ Bundle saved → {out}")
    return bundle

# ── Inference ─────────────────────────────────────────────────────────────────
_bundle = None
def get_bundle():
    global _bundle
    if _bundle is None:
        p = os.path.join(MDIR, "bundle_v2.pkl")
        with open(p, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle

def predict(store_id: str, sku_id: str, category: str, context: dict) -> dict:
    """
    context keys: dow, month, woy, dom, is_wknd, quarter,
                  lag_1, lag_3, lag_7, lag_14,
                  roll_7_mean, roll_14_mean, roll_7_std, ema_7,
                  weather_mult, festival_mult, day_mult, is_rain, temp_c,
                  recent_14 (list of last 14 demand values, for seq model)
    Returns: {point, low, high, by_model: {rf, gb, seq}}
    """
    b = get_bundle()
    se = b["store_map"].get(store_id, 0)
    sk = b["sku_map"].get(sku_id,   0)
    ca = b["cat_map"].get(category, 0)

    row = [se, sk, ca] + [context.get(f, 0) for f in b["features"][3:]]
    X   = np.array(row, dtype=float).reshape(1,-1)

    rf_p  = float(max(0, b["rf"].predict(X)[0]))
    gb_p  = float(max(0, b["gb"].predict(X)[0]))

    # Sequence model
    recent = context.get("recent_14", [context.get("roll_7_mean",10)]*14)
    if len(recent) < WINDOW:
        recent = ([recent[0]]*WINDOW + list(recent))[-WINDOW:]
    recent = list(recent)[-WINDOW:]
    trend  = float(np.polyfit(range(WINDOW), recent, 1)[0])
    seas   = recent[context.get("dow",0) % WINDOW]
    seq_row = recent + [trend, seas,
                        float(se), float(sk),
                        float(context.get("festival_mult",1)),
                        float(context.get("day_mult",1)),
                        float(context.get("is_wknd",0))]
    # Pad/trim to match seq_cols length
    n_seq_cols = len(b["seq_cols"])
    if len(seq_row) < n_seq_cols:
        seq_row += [0.0] * (n_seq_cols - len(seq_row))
    seq_row = seq_row[:n_seq_cols]
    Xs = b["scaler"].transform(np.array(seq_row).reshape(1,-1))
    sq_p = float(max(0, b["seq_model"].predict(Xs)[0]))

    w  = b["weights"]
    pt = w["rf"]*rf_p + w["gb"]*gb_p + w["seq"]*sq_p
    pt = max(0, pt)
    std = context.get("roll_7_std", pt*0.18) or pt*0.18
    return {
        "point": round(pt,1),
        "low":   round(max(0, pt - 1.5*std), 1),
        "high":  round(pt + 1.5*std, 1),
        "by_model": {"rf": round(rf_p,1), "gb": round(gb_p,1), "seq": round(sq_p,1)},
    }

if __name__ == "__main__":
    train()