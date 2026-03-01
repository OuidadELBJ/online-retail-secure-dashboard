import sqlite3
import pandas as pd
from contextlib import contextmanager

DB_PATH = "users.db"

@contextmanager
def get_db_connection():
    """Gestionnaire de contexte pour sécuriser les connexions SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Permet d'accéder aux colonnes par leur nom
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialise la table des utilisateurs."""
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def create_user(username, password_hash, role):
    """Insère un nouvel utilisateur dans la base."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role)
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # L'utilisateur existe déjà

def get_user_by_username(username):
    """Récupère les informations d'un utilisateur spécifique."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()

def get_all_users_df():
    """Retourne tous les utilisateurs sous forme de DataFrame Pandas (pour le panel Admin)."""
    with get_db_connection() as conn:
        return pd.read_sql_query("SELECT id, username, role, created_at FROM users", conn)