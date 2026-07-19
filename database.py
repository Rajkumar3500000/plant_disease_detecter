import os
import sqlite3
import warnings
from werkzeug.security import generate_password_hash
from flask import session

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    mysql = None
    Error = Exception
    warnings.warn(
        "mysql-connector-python is not installed. Falling back to SQLite.",
        UserWarning,
    )

SQLITE_DB_PATH = "plant.db"

MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "plant_disease_db"),
    "autocommit": True,
}

DB_TYPE = os.environ.get("DB_TYPE", "mysql").lower()
if DB_TYPE == "mysql" and mysql is None:
    DB_TYPE = "sqlite"

DB_AVAILABLE = False


def row_to_dict(row):
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    return {key: row[key] for key in row.keys()}


def get_db_connection(use_database=True):
    if DB_TYPE == "sqlite":
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    config = dict(MYSQL_CONFIG)
    if not use_database:
        config.pop("database", None)
    return mysql.connector.connect(**config)


def get_cursor(conn, dictionary=False):
    if DB_TYPE == "mysql":
        return conn.cursor(dictionary=dictionary)
    return conn.cursor()


def init_db():
    global DB_AVAILABLE, DB_TYPE

    try:
        if DB_TYPE == "mysql":
            if mysql is None:
                raise RuntimeError("mysql-connector-python is unavailable")
            conn = get_db_connection(use_database=False)
            cursor = get_cursor(conn)
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}`")
            conn.close()
            conn = get_db_connection()
        else:
            conn = get_db_connection()

        cursor = get_cursor(conn)

        if DB_TYPE == "mysql":
            create_user_table = """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(80) UNIQUE NOT NULL,
                    email VARCHAR(120) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            create_report_table = """
                CREATE TABLE IF NOT EXISTS reports (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    plant VARCHAR(100) NOT NULL,
                    disease VARCHAR(200) NOT NULL,
                    confidence DECIMAL(5, 2) NOT NULL,
                    severity VARCHAR(50) NOT NULL,
                    treatment TEXT NOT NULL,
                    prevention TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """
        else:
            create_user_table = """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            create_report_table = """
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    plant TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    severity TEXT NOT NULL,
                    treatment TEXT NOT NULL,
                    prevention TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """

        cursor.execute(create_user_table)
        cursor.execute(create_report_table)

        if DB_TYPE == "mysql":
            cursor.execute("SHOW COLUMNS FROM users LIKE 'role'")
            if cursor.fetchone() is None:
                cursor.execute(
                    "ALTER TABLE users ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user'"
                )
            cursor.execute("SELECT id FROM users WHERE username = %s", ("admin",))
        else:
            cursor.execute("SELECT id FROM users WHERE username = ?", ("admin",))

        if not cursor.fetchone():
            if DB_TYPE == "mysql":
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                    (
                        "admin",
                        "admin@example.com",
                        generate_password_hash("admin123"),
                        "admin",
                    ),
                )
            else:
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                    (
                        "admin",
                        "admin@example.com",
                        generate_password_hash("admin123"),
                        "admin",
                    ),
                )

        conn.commit()
        conn.close()
        DB_AVAILABLE = True
        print(f"{DB_TYPE.upper()} database ready for authentication and reports")
    except Exception as exc:
        print(f"Database initialization unavailable ({DB_TYPE}): {exc}")
        if DB_TYPE == "mysql":
            DB_TYPE = "sqlite"
            print("Falling back to SQLite database.")
            try:
                init_db()
            except Exception as fallback_exc:
                DB_AVAILABLE = False
                print(f"SQLite fallback failed: {fallback_exc}")
        else:
            DB_AVAILABLE = False


from flask import session

def get_current_user():
    if not DB_AVAILABLE:
        return None

    user_id = session.get("user_id")

    if not user_id:
        return None

    conn = get_db_connection()
    cursor = get_cursor(conn, dictionary=(DB_TYPE == "mysql"))

    if DB_TYPE == "mysql":
        cursor.execute(
            "SELECT id, username, email, role FROM users WHERE id = %s",
            (user_id,)
        )
    else:
        cursor.execute(
            "SELECT id, username, email, role FROM users WHERE id = ?",
            (user_id,)
        )

    user = row_to_dict(cursor.fetchone())

    cursor.close()
    conn.close()

    return user

def is_admin_user(user):
    return bool(user and str(user.get("role", "")).lower() == "admin")
