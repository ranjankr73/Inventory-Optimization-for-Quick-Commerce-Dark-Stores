-- DarkIQ v2 — SQLite Schema
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL UNIQUE,
    password    TEXT    NOT NULL,        -- bcrypt hash
    role        TEXT    NOT NULL DEFAULT 'viewer', -- admin | manager | viewer
    store_id    TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stores (
    store_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT NOT NULL,
    zone        TEXT NOT NULL,
    lat         REAL,
    lng         REAL,
    active      INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS skus (
    sku_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    shelf_days  INTEGER NOT NULL,
    unit_cost   REAL NOT NULL,
    sell_price  REAL NOT NULL,
    max_stock   INTEGER NOT NULL,
    reorder_pt  INTEGER NOT NULL,
    is_perishable INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS inventory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    store_id    TEXT NOT NULL,
    sku_id      TEXT NOT NULL,
    opening_stock   INTEGER DEFAULT 0,
    demand_units    INTEGER DEFAULT 0,
    reorder_qty     INTEGER DEFAULT 0,
    spoilage_units  REAL    DEFAULT 0,
    closing_stock   INTEGER DEFAULT 0,
    stockout        INTEGER DEFAULT 0,
    days_in_stock   INTEGER DEFAULT 0,
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (sku_id)   REFERENCES skus(sku_id)
);

CREATE TABLE IF NOT EXISTS demand_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,
    store_id        TEXT NOT NULL,
    sku_id          TEXT NOT NULL,
    demand_units    REAL NOT NULL,
    weather_mult    REAL DEFAULT 1.0,
    festival_mult   REAL DEFAULT 1.0,
    day_mult        REAL DEFAULT 1.0,
    is_rain         INTEGER DEFAULT 0,
    temp_c          REAL DEFAULT 25.0,
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (sku_id)   REFERENCES skus(sku_id)
);

CREATE TABLE IF NOT EXISTS ai_decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT DEFAULT (datetime('now')),
    store_id    TEXT NOT NULL,
    sku_id      TEXT NOT NULL,
    action      TEXT NOT NULL,   -- reorder|transfer|mark_down|ok|critical
    urgency     TEXT NOT NULL,   -- critical|high|medium|low
    qty         REAL DEFAULT 0,
    reason      TEXT,
    cost_impact REAL DEFAULT 0,
    confidence  REAL DEFAULT 0,
    resolved    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transfer_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT DEFAULT (datetime('now')),
    sku_id      TEXT NOT NULL,
    from_store  TEXT NOT NULL,
    to_store    TEXT NOT NULL,
    qty         INTEGER NOT NULL,
    reason      TEXT,
    status      TEXT DEFAULT 'pending'  -- pending|completed|cancelled
);

CREATE INDEX IF NOT EXISTS idx_inv_date_store  ON inventory(date, store_id);
CREATE INDEX IF NOT EXISTS idx_demand_sku_date ON demand_signals(sku_id, date);
CREATE INDEX IF NOT EXISTS idx_decisions_store ON ai_decisions(store_id, resolved);