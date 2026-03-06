"""
database.py — All database access for the Smart Attendance System
==================================================================
Single source of truth for every SQLite read/write.
gui.py, recognize.py, and register.py all import from here.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH      = "database/attendance.db"
FACE_DATA_DIR = "face_data"
MODEL_PATH   = "models/embeddings.npy"

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def init_db():
    """Create tables and required directories if they don't exist."""
    for d in ["database", FACE_DATA_DIR, "models", "exports"]:
        os.makedirs(d, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        user_code TEXT UNIQUE NOT NULL,
        registered_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")

    c.execute("""CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date DATE NOT NULL,
        in_time TIME NOT NULL,
        out_time TIME,
        status TEXT DEFAULT 'PRESENT',
        confidence REAL DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id))""")

    # Migration: rename old 'time' column if it exists
    try:
        c.execute("ALTER TABLE attendance RENAME COLUMN time TO in_time")
        c.execute("ALTER TABLE attendance ADD COLUMN out_time TIME")
    except Exception:
        pass

    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# USER QUERIES
# ─────────────────────────────────────────────

def get_users():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id,name,user_code,registered_at FROM users ORDER BY name")
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_users_dict():
    """Returns {user_code: name} dict — used by recognition engine."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT user_code, name FROM users")
    rows = {code: name for code, name in c.fetchall()}
    conn.close()
    return rows

def user_exists(user_code):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id FROM users WHERE user_code=?", (user_code,))
    row  = c.fetchone()
    conn.close()
    return row is not None

def save_user_db(name, user_code):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("INSERT INTO users(name,user_code) VALUES(?,?)", (name, user_code))
    conn.commit()
    conn.close()

def delete_user_db(user_code):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id,name FROM users WHERE user_code=?", (user_code,))
    row  = c.fetchone()
    if row:
        c.execute("DELETE FROM attendance WHERE user_id=?", (row[0],))
        c.execute("DELETE FROM users WHERE user_code=?",    (user_code,))
        conn.commit()
    conn.close()
    return row

def delete_all_users_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("DELETE FROM attendance")
    c.execute("DELETE FROM users")
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# ATTENDANCE QUERIES
# ─────────────────────────────────────────────

def get_today_attendance():
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("""
        SELECT u.name, u.user_code, a.in_time, a.out_time,
               COALESCE(a.status, 'ABSENT'), COALESCE(a.confidence, 0.0)
        FROM users u
        LEFT JOIN attendance a ON u.id = a.user_id AND a.date = ?
        ORDER BY a.in_time DESC, u.name ASC
    """, (today,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_attendance_filtered(date_from=None, date_to=None, user_code=None):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    q    = ["SELECT u.name, u.user_code, a.date, a.in_time, a.out_time, "
            "a.status, ROUND(a.confidence, 1) "
            "FROM attendance a JOIN users u ON u.id = a.user_id WHERE 1=1 "]
    p    = []
    if date_from: q.append("AND a.date >= ? ");      p.append(date_from)
    if date_to:   q.append("AND a.date <= ? ");      p.append(date_to)
    if user_code: q.append("AND u.user_code=? ");    p.append(user_code.upper())
    q.append("ORDER BY a.date DESC, a.in_time DESC, u.name ASC")
    c.execute("".join(q), p)
    rows = c.fetchall()
    conn.close()
    return rows

def mark_attendance_db(user_code, confidence=0):
    """
    Mark IN time on first detection, OUT time on second (after 60 s).
    A new IN cycle starts 2 minutes after the last OUT.
    Returns True if a record was written, False if within cooldown.
    """
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id FROM users WHERE user_code=?", (user_code,))
    row  = c.fetchone()
    if not row:
        conn.close()
        return False
    uid   = row[0]
    today = datetime.now().strftime("%Y-%m-%d")
    now_t = datetime.now().strftime("%H:%M:%S")
    fmt   = "%H:%M:%S"

    c.execute(
        "SELECT id, in_time, out_time FROM attendance "
        "WHERE user_id=? AND date=? ORDER BY in_time DESC LIMIT 1",
        (uid, today)
    )
    att = c.fetchone()

    if att:
        aid, in_t, out_t = att
        if out_t:
            try:    diff = (datetime.strptime(now_t, fmt) - datetime.strptime(out_t, fmt)).total_seconds()
            except: diff = 121
            if diff > 120:
                c.execute(
                    "INSERT INTO attendance(user_id,date,in_time,status,confidence) VALUES(?,?,?,'PRESENT',?)",
                    (uid, today, now_t, confidence)
                )
                conn.commit(); conn.close(); return True
            else:
                conn.close(); return False
        else:
            try:    diff = (datetime.strptime(now_t, fmt) - datetime.strptime(in_t, fmt)).total_seconds()
            except: diff = 61
            if diff > 60:
                c.execute(
                    "UPDATE attendance SET out_time=?, confidence=? WHERE id=?",
                    (now_t, confidence, aid)
                )
                conn.commit(); conn.close(); return True
            else:
                conn.close(); return False
    else:
        c.execute(
            "INSERT INTO attendance(user_id,date,in_time,status,confidence) VALUES(?,?,?,'PRESENT',?)",
            (uid, today, now_t, confidence)
        )
        conn.commit(); conn.close(); return True

def clear_attendance_db(date=None):
    """Clear attendance for a specific date, or ALL records if date=None."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    if date:
        c.execute("DELETE FROM attendance WHERE date=?", (date,))
    else:
        c.execute("DELETE FROM attendance")
    count = conn.total_changes
    conn.commit()
    conn.close()
    return count
