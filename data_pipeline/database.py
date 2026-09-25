"""
Module 1: Database Management
Implements normalized SQLite storage with Foreign Key constraints
between `categories` and `books` tables.
"""

import os
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Tuple

DEFAULT_DB_PATH = "data_pipeline/database/zepto_catalog.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_books_category ON books(category_id);
CREATE INDEX IF NOT EXISTS idx_books_price_inr ON books(price_inr);
"""

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Connect to SQLite database and enforce foreign keys."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create schema tables if they do not exist."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)

def populate_database(
    df: pd.DataFrame, 
    db_path: str = DEFAULT_DB_PATH, 
    replace_existing: bool = True
) -> Tuple[int, int]:
    """
    Inserts categories and books into normalized tables.
    Returns (categories_inserted, books_inserted).
    """
    init_db(db_path)
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        if replace_existing:
            cursor.execute("DELETE FROM books;")
            cursor.execute("DELETE FROM categories;")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('books', 'categories');")
            
        # 1. Insert unique categories
        categories = sorted(df["category"].dropna().unique().tolist())
        cat_map = {}
        for cat in categories:
            cursor.execute(
                "INSERT OR IGNORE INTO categories (category_name) VALUES (?);",
                (cat,)
            )
            cursor.execute(
                "SELECT category_id FROM categories WHERE category_name = ?;",
                (cat,)
            )
            cat_id = cursor.fetchone()[0]
            cat_map[cat] = cat_id
            
        # 2. Insert books with foreign key
        books_data = []
        for _, row in df.iterrows():
            books_data.append((
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                1 if row["in_stock"] else 0,
                cat_map[row["category"]]
            ))
            
        cursor.executemany("""
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?);
        """, books_data)
        
        conn.commit()
        return len(cat_map), len(books_data)

def execute_query(query: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Execute a read query and return as a pandas DataFrame."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)
