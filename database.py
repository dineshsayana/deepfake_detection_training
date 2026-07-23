import sqlite3
from datetime import datetime

DB_PATH = 'app_data.db'


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            media_type TEXT NOT NULL,
            original_filename TEXT,
            output_path TEXT,
            result TEXT NOT NULL,
            confidence REAL,
            explanation TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()


def create_user(username, password_hash):
    conn = get_db()
    conn.execute(
        'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
        (username, password_hash, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_user_by_username(username):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return user


def add_history(user_id, media_type, original_filename, output_path, result, confidence, explanation):
    conn = get_db()
    conn.execute('''
        INSERT INTO history
            (user_id, media_type, original_filename, output_path, result, confidence, explanation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, media_type, original_filename, output_path, result, confidence, explanation,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_history(user_id):
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM history WHERE user_id = ? ORDER BY created_at DESC', (user_id,)
    ).fetchall()
    conn.close()
    return rows


def clear_history(user_id):
    conn = get_db()
    conn.execute('DELETE FROM history WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
