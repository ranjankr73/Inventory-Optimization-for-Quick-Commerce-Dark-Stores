"""
data/seed_db.py  —  DarkIQ v2
Generates realistic synthetic data and seeds the SQLite database.
Run once: python data/seed_db.py
"""
import sqlite3, os, sys, hashlib, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
BASE   = os.path.join(os.path.dirname(__file__), "..")
DB     = os.path.join(BASE, "instance", "darkiq.db")
SCHEMA = os.path.join(os.path.dirname(__file__), "schema.sql")

# ─── Master data ──────────────────────────────────────────────────────────────
STORES = [
    ("DS_NORTH",   "DarkIQ North",   "Lucknow", "North", 26.8500, 80.9500),
    ("DS_CENTRAL", "DarkIQ Central", "Lucknow", "Central", 26.8467, 80.9462),
    ("DS_SOUTH",   "DarkIQ South",   "Lucknow", "South", 26.8100, 80.9200),
    ("DS_EAST",    "DarkIQ East",    "Lucknow", "East",  26.8700, 81.0100),
]

SKUS = [
    ("SKU001","Fresh Milk 1L",      "dairy",     5,  55,  80,  200, 55,  1),
    ("SKU002","Bread Loaf",         "bakery",    4,  35,  55,  150, 45,  1),
    ("SKU003","Eggs 12-pack",       "dairy",    21,  80, 115,  120, 35,  1),
    ("SKU004","Rice 5kg",           "staples", 365, 250, 340,   80, 20,  0),
    ("SKU005","Chips & Snacks",     "snacks",   90,  30,  52,  180, 50,  0),
    ("SKU006","Soft Drinks 6-pack", "beverages",180,120, 185,  160, 45,  0),
    ("SKU007","Bananas 1kg",        "produce",   5,  40,  65,  100, 30,  1),
    ("SKU008","Pasta 500g",         "staples", 365,  45,  72,  100, 25,  0),
    ("SKU009","Yoghurt 400g",       "dairy",    14,  60,  92,   90, 28,  1),
    ("SKU010","Cooking Oil 1L",     "staples", 365, 130, 188,   70, 18,  0),
    ("SKU011","Tomatoes 500g",      "produce",   7,  30,  52,  120, 38,  1),
    ("SKU012","Biscuits Pack",      "snacks",  120,  25,  45,  200, 55,  0),
    ("SKU013","Mineral Water 1L",   "beverages",365, 15,  32,  300, 80,  0),
    ("SKU014","Butter 500g",        "dairy",    30,  90, 135,   80, 22,  1),
    ("SKU015","Instant Noodles",    "staples", 365,  20,  38,  250, 70,  0),
    ("SKU016","Paneer 200g",        "dairy",     4, 110, 165,   80, 28,  1),
    ("SKU017","Green Vegetables",   "produce",   3,  25,  45,  100, 35,  1),
    ("SKU018","Fruit Juice 1L",     "beverages",30,  65, 100,  120, 35,  0),
    ("SKU019","Atta 5kg",           "staples", 180, 220, 295,   90, 25,  0),
    ("SKU020","Tea 250g",           "staples", 365,  95, 145,   70, 18,  0),
]

USERS = [
    ("admin",   "admin123",   "admin",   None),
    ("manager", "manager123", "manager", "DS_NORTH"),
    ("viewer",  "viewer123",  "viewer",  "DS_CENTRAL"),
]

START = datetime(2023, 1, 1)
END   = datetime(2024, 6, 30)
DATES = pd.date_range(START, END, freq="D")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def simple_hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def weather_cat_mult(date: datetime, category: str) -> float:
    m = date.month
    weather = {
        7:{"dairy":1.10,"bakery":1.15,"staples":1.20,"beverages":1.40,"snacks":0.88,"produce":0.78},
        8:{"dairy":1.10,"bakery":1.15,"staples":1.20,"beverages":1.40,"snacks":0.88,"produce":0.78},
        9:{"dairy":1.05,"bakery":1.10,"staples":1.15,"beverages":1.30,"snacks":0.92,"produce":0.82},
        12:{"dairy":1.20,"bakery":1.10,"staples":1.10,"beverages":0.82,"snacks":1.12,"produce":1.02},
        1:{"dairy":1.20,"bakery":1.10,"staples":1.10,"beverages":0.82,"snacks":1.12,"produce":1.02},
        2:{"dairy":1.15,"bakery":1.05,"staples":1.05,"beverages":0.85,"snacks":1.08,"produce":1.00},
    }
    return weather.get(m, {}).get(category, 1.0)

def festival_mult(date: datetime) -> float:
    festivals = [
        (datetime(2023,3,7),datetime(2023,3,9),1.50),    # Holi
        (datetime(2023,8,28),datetime(2023,8,30),1.30),  # Janmashtami
        (datetime(2023,10,20),datetime(2023,10,24),1.65),# Dussehra
        (datetime(2023,11,10),datetime(2023,11,14),1.85),# Diwali
        (datetime(2023,12,24),datetime(2023,12,26),1.40),# Christmas/New Year
        (datetime(2024,1,14),datetime(2024,1,15),1.30),  # Makar Sankranti
        (datetime(2024,1,26),datetime(2024,1,26),1.25),  # Republic Day
        (datetime(2024,3,25),datetime(2024,3,27),1.55),  # Holi 2024
        (datetime(2024,4,14),datetime(2024,4,15),1.30),  # Baisakhi
    ]
    for s,e,m in festivals:
        if s <= date <= e: return m
    return 1.0

def day_mult(date: datetime) -> float:
    return [0.82,0.86,0.90,0.93,1.12,1.38,1.28][date.weekday()]

def temp_c(date: datetime) -> float:
    base = 18 + 14 * np.sin(2*np.pi*(date.month-3)/12)
    return round(base + np.random.normal(0,2), 1)

def is_rain(date: datetime) -> int:
    return 1 if date.month in [7,8,9] and np.random.rand() < 0.60 else 0

# ─── Seed ─────────────────────────────────────────────────────────────────────
def seed():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    if os.path.exists(DB): os.remove(DB)
    con = sqlite3.connect(DB)
    cur = con.cursor()

    with open(SCHEMA) as f:
        cur.executescript(f.read())

    # Users
    for uname, pw, role, store in USERS:
        cur.execute("INSERT INTO users(username,password,role,store_id) VALUES(?,?,?,?)",
                    (uname, simple_hash(pw), role, store))

    # Stores
    cur.executemany("INSERT INTO stores VALUES(?,?,?,?,?,?,1)", STORES)

    # SKUs
    cur.executemany("INSERT INTO skus VALUES(?,?,?,?,?,?,?,?,?)", SKUS)

    con.commit()

    # ── Demand + Inventory ──────────────────────────────────────────────────
    print("Generating demand & inventory (this may take ~30s)...")
    demand_rows, inv_rows = [], []
    sku_dict = {s[0]: s for s in SKUS}

    for store_id, *_ in STORES:
        store_bias = round(0.82 + np.random.rand()*0.42, 3)

        for sku_row in SKUS:
            sku_id, name, cat, shelf, ucost, sprice, maxs, rop, perish = sku_row
            base  = maxs * (0.055 + np.random.rand()*0.075)
            stock = round(maxs * (0.45 + np.random.rand()*0.45))
            days_in = 0

            for date in DATES:
                wm  = weather_cat_mult(date, cat)
                fm  = festival_mult(date)
                dm  = day_mult(date)
                tmp = temp_c(date)
                rn  = is_rain(date)
                noise = float(np.random.lognormal(0, 0.17))
                demand = max(0, round(base * store_bias * wm * fm * dm * noise))

                demand_rows.append((
                    str(date.date()), store_id, sku_id,
                    demand, round(wm,3), round(fm,3), round(dm,3), rn, tmp
                ))

                opening = stock
                days_in += 1
                spoilage = 0.0
                if perish and days_in >= shelf * 0.80:
                    spoilage = round(stock * 0.12, 1)
                    stock = max(0, stock - spoilage)
                    days_in = 0

                stock = max(0, stock - demand)
                reorder_qty = 0
                if stock <= rop:
                    reorder_qty = maxs - stock
                    stock = maxs
                    days_in = 0

                inv_rows.append((
                    str(date.date()), store_id, sku_id,
                    opening, demand, reorder_qty,
                    spoilage, stock, int(stock == 0), days_in
                ))

    cur.executemany(
        "INSERT INTO demand_signals(date,store_id,sku_id,demand_units,weather_mult,"
        "festival_mult,day_mult,is_rain,temp_c) VALUES(?,?,?,?,?,?,?,?,?)",
        demand_rows
    )
    cur.executemany(
        "INSERT INTO inventory(date,store_id,sku_id,opening_stock,demand_units,"
        "reorder_qty,spoilage_units,closing_stock,stockout,days_in_stock) VALUES(?,?,?,?,?,?,?,?,?,?)",
        inv_rows
    )

    con.commit()
    con.close()

    total = len(demand_rows)
    print(f"  Inserted {total:,} demand rows and {len(inv_rows):,} inventory rows")
    print(f"  Stores: {len(STORES)} | SKUs: {len(SKUS)} | Users: {len(USERS)}")
    print(f"  DB: {DB}")
    print("✓ Database seeded successfully.\n")
    print("Login credentials:")
    for u,p,r,_ in USERS:
        print(f"  {u} / {p}  ({r})")

if __name__ == "__main__":
    seed()