"""
db.py — our "database"
-----------------------
This uses SQLite, which is just a single file on your computer (hia.db)
that stores tables, the same idea as Supabase/Postgres but with zero setup:
no account, no server, no internet needed. Python can talk to it out of
the box with the built-in `sqlite3` module.

Two tables:
  reports  -> one row per report you analyze (the text + the AI's analysis)
  messages -> one row per chat message (linked to a report by report_id)
"""

import sqlite3

DB_FILE = "hia.db"


def get_connection():
    """Open a connection to the database file (creates it if it doesn't exist)."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # lets us access columns by name, e.g. row["filename"]
    return conn


def init_db():
    """Create the tables if they don't already exist. Safe to call every time the app starts."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            report_text TEXT,
            analysis TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (report_id) REFERENCES reports (id)
        )
    """)
    conn.commit()
    conn.close()


def save_report(filename: str, report_text: str, analysis: str) -> int:
    """Save a new report + its analysis. Returns the new report's id."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO reports (filename, report_text, analysis) VALUES (?, ?, ?)",
        (filename, report_text, analysis),
    )
    conn.commit()
    report_id = cursor.lastrowid
    conn.close()
    return report_id


def list_reports():
    """Return all past reports, most recent first, for showing in the sidebar."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, filename, created_at FROM reports ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return rows


def load_report(report_id: int):
    """Return (report_text, analysis) for a given report id."""
    conn = get_connection()
    row = conn.execute(
        "SELECT report_text, analysis FROM reports WHERE id = ?", (report_id,)
    ).fetchone()
    conn.close()
    return (row["report_text"], row["analysis"]) if row else (None, None)


def save_message(report_id: int, role: str, content: str):
    """Save one chat message (role is 'user' or 'assistant')."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (report_id, role, content) VALUES (?, ?, ?)",
        (report_id, role, content),
    )
    conn.commit()
    conn.close()


def load_messages(report_id: int):
    """Return all chat messages for a report, in order, as a list of dicts."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE report_id = ? ORDER BY created_at ASC",
        (report_id,),
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]
