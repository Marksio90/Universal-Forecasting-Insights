#!/usr/bin/env python
from pathlib import Path
import sqlite3

DB_PATH = Path("data/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS healthcheck (id INTEGER PRIMARY KEY, ts TEXT)")
cur.execute("INSERT INTO healthcheck (ts) VALUES (datetime('now'))")
conn.commit()
conn.close()
print("Seed DB: OK ->", DB_PATH)
