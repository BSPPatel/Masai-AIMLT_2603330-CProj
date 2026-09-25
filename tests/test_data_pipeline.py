"""
Automated Test Suite for Module 1: Data Engineering Pipeline
Tests scraping, cleaning, currency conversion, SQLite storage, SQL queries, and Pandas equivalence.
"""

import os
import sqlite3
import pytest
import pandas as pd

from data_pipeline.cleaner import (
    clean_price_gbp,
    clean_rating,
    clean_availability,
    clean_and_transform_books,
    EXCHANGE_RATE_GBP_TO_INR
)
from data_pipeline.database import (
    init_db,
    populate_database,
    get_connection,
    execute_query
)
from data_pipeline.run_pipeline import (
    load_individual_queries,
    demonstrate_pandas_equivalence
)

def test_fixed_currency_exchange_rate():
    """Ensure the exchange rate strictly follows the 105.50 INR/GBP assignment constant."""
    assert EXCHANGE_RATE_GBP_TO_INR == 105.50

def test_clean_price_gbp():
    """Validate parsing of various price formats and symbols."""
    assert clean_price_gbp("£51.77") == 51.77
    assert clean_price_gbp("Â£22.60") == 22.60
    assert clean_price_gbp("15.99") == 15.99
    assert clean_price_gbp("Invalid") is None
    assert clean_price_gbp(None) is None

def test_clean_rating():
    """Validate textual rating mapping to 1-5."""
    assert clean_rating("One") == 1
    assert clean_rating("Two") == 2
    assert clean_rating("Three") == 3
    assert clean_rating("Four") == 4
    assert clean_rating("Five") == 5
    assert clean_rating("five") == 5
    assert clean_rating("6") is None
    assert clean_rating(None) is None

def test_clean_availability():
    """Validate in-stock boolean extraction."""
    assert clean_availability("In stock (22 available)") is True
    assert clean_availability("In stock") is True
    assert clean_availability("Out of stock") is False
    assert clean_availability("Not in stock") is False
    assert clean_availability(None) is False

def test_clean_and_transform_pipeline():
    """Validate dataframe transformation, missing value imputation, and price_inr computation."""
    raw_sample = [
        {"title": "Book A", "price": "£10.00", "star_rating": "Four", "availability": "In stock", "category": "Fiction"},
        {"title": "Book B", "price": "£20.00", "star_rating": "Five", "availability": "In stock", "category": "Fiction"},
        {"title": "Book C (missing price)", "price": None, "star_rating": "Three", "availability": "In stock", "category": "Fiction"},
        {"title": "Book D (missing rating)", "price": "£30.00", "star_rating": None, "availability": "In stock", "category": "Fiction"},
        {"title": "", "price": "£50.00", "star_rating": "One", "availability": "In stock", "category": "Fiction"}, # Should drop
    ]
    df_cleaned, audit = clean_and_transform_books(raw_sample)
    
    # 5 rows raw -> 1 dropped (empty title) -> 4 rows remaining
    assert len(df_cleaned) == 4
    assert audit["dropped_rows"] == 1
    assert audit["imputed_price_count"] == 1
    assert audit["imputed_rating_count"] == 1
    
    # Imputed price should match median of available Fiction books: (10 + 20 + 30)/3 = 20.0
    book_c = df_cleaned[df_cleaned["title"] == "Book C (missing price)"].iloc[0]
    assert book_c["price_gbp"] == 20.00
    assert book_c["price_inr"] == round(20.00 * 105.50, 2)
    
    # Check types
    assert pd.api.types.is_float_dtype(df_cleaned["price_gbp"])
    assert pd.api.types.is_float_dtype(df_cleaned["price_inr"])
    assert pd.api.types.is_integer_dtype(df_cleaned["rating"])
    assert pd.api.types.is_bool_dtype(df_cleaned["in_stock"])

def test_sqlite_schema_and_integrity(tmp_path):
    """Test schema creation, foreign key constraints, and population."""
    db_file = str(tmp_path / "test_catalog.db")
    init_db(db_file)
    
    df_test = pd.DataFrame([
        {"title": "Test Book 1", "category": "Travel", "price_gbp": 12.00, "price_inr": 1266.00, "rating": 4, "in_stock": True},
        {"title": "Test Book 2", "category": "Mystery", "price_gbp": 25.00, "price_inr": 2637.50, "rating": 5, "in_stock": True},
        {"title": "Test Book 3", "category": "Mystery", "price_gbp": 15.00, "price_inr": 1582.50, "rating": 3, "in_stock": False},
    ])
    
    num_cats, num_books = populate_database(df_test, db_file)
    assert num_cats == 2
    assert num_books == 3
    
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM categories;")
        assert cursor.fetchone()[0] == 2
        cursor.execute("SELECT COUNT(*) FROM books;")
        assert cursor.fetchone()[0] == 3
        
        # Test foreign key enforcement
        cursor.execute("PRAGMA foreign_keys;")
        assert cursor.fetchone()[0] == 1
        
        # Violating FK should raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                "INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) VALUES (?, ?, ?, ?, ?, ?);",
                ("Orphan Book", 10.0, 1055.0, 3, 1, 9999) # Non-existent category_id
            )

def test_queries_contain_all_required_clauses():
    """Verify that the SQL suite covers SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN/IN, and JOIN."""
    queries = load_individual_queries("data_pipeline/queries.sql")
    assert len(queries) >= 5
    
    full_sql = " ".join([q["sql"].upper() for q in queries])
    assert "SELECT" in full_sql
    assert "WHERE" in full_sql
    assert "ORDER BY" in full_sql
    assert "LIMIT" in full_sql
    assert "DISTINCT" in full_sql
    assert "BETWEEN" in full_sql or "IN" in full_sql
    assert "JOIN" in full_sql

def test_pandas_equivalence_and_join(tmp_path):
    """Verify that pd.read_sql and pd.merge produce identical results to SQL queries."""
    db_file = str(tmp_path / "equiv_test.db")
    df_test = pd.DataFrame([
        {"title": f"Book {i}", "category": f"Genre {i%3}", "price_gbp": 10.0+i, "price_inr": (10.0+i)*105.50, "rating": 5 if i%2==0 else 3, "in_stock": True}
        for i in range(10)
    ])
    populate_database(df_test, db_file)
    
    conn = get_connection(db_file)
    try:
        assert demonstrate_pandas_equivalence(conn) is True
    finally:
        conn.close()
