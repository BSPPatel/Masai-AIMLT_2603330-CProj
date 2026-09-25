"""
Module 1: Pipeline Orchestrator
Executes: SCRAPE -> CLEAN -> CONVERT (105.50) -> STORE -> QUERY -> PANDAS REPRODUCTION
"""

import os
import sys
import json
import sqlite3
import pandas as pd
from typing import Dict, Any

# Ensure data_pipeline directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from scraper import run_scraper
from cleaner import clean_and_transform_books, EXCHANGE_RATE_GBP_TO_INR
from database import init_db, populate_database, get_connection, DEFAULT_DB_PATH

RAW_JSON_PATH = "data_pipeline/data/raw_books.json"
CLEANED_CSV_PATH = "data_pipeline/data/cleaned_books.csv"
QUERIES_SQL_PATH = "data_pipeline/queries.sql"
QUERY_OUTPUT_LOG = "data_pipeline/data/sql_query_results.txt"

def load_individual_queries(sql_file_path: str = QUERIES_SQL_PATH):
    """Splits a multi-query SQL file into individual labeled queries."""
    with open(sql_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    raw_queries = content.split(";")
    queries = []
    current_comment = ""
    
    for segment in raw_queries:
        trimmed = segment.strip()
        if not trimmed:
            continue
        lines = trimmed.split("\n")
        comments = [l for l in lines if l.strip().startswith("--")]
        code_lines = [l for l in lines if not l.strip().startswith("--")]
        query_code = "\n".join(code_lines).strip()
        label = comments[0].replace("--", "").strip() if comments else f"Query {len(queries)+1}"
        if query_code:
            queries.append({"label": label, "sql": query_code})
            
    return queries

def execute_and_log_queries(conn: sqlite3.Connection, output_log_path: str = QUERY_OUTPUT_LOG):
    """Executes all project SQL queries and writes results to a text log."""
    queries = load_individual_queries()
    results = {}
    
    with open(output_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("ZEPTO DATA PLATFORM — MODULE 1 SQL QUERY EXECUTION LOG\n")
        log_file.write("=" * 80 + "\n\n")
        
        for idx, q_info in enumerate(queries, 1):
            label = q_info["label"]
            sql = q_info["sql"]
            df_res = pd.read_sql_query(sql, conn)
            results[f"query_{idx}"] = df_res
            
            header = f"[{idx}] {label}"
            print(f"Executing: {header}")
            
            log_file.write(f"{header}\n")
            log_file.write("-" * len(header) + "\n")
            log_file.write("SQL QUERY:\n")
            log_file.write(sql + ";\n\n")
            log_file.write(f"OUTPUT ROWS: {len(df_res)}\n")
            log_file.write(df_res.to_string(index=False))
            log_file.write("\n\n" + "=" * 80 + "\n\n")
            
    print(f"Logged {len(queries)} query outputs to {output_log_path}")
    return results

def demonstrate_pandas_equivalence(conn: sqlite3.Connection):
    """
    Demonstrates:
    1. Reading at least 2 SQL queries via pd.read_sql
    2. Reproducing the SQL JOIN query via pd.merge
    3. Asserting and proving output equivalence
    """
    print("\n" + "=" * 50)
    print("DEMONSTRATING PANDAS EQUIVALENCE")
    print("=" * 50)
    
    # Demonstration 1: Reading via pd.read_sql
    sql_top5 = """
    SELECT book_id, title, price_gbp, price_inr, rating
    FROM books
    WHERE in_stock = 1
    ORDER BY price_inr DESC
    LIMIT 5;
    """
    df_sql_top5 = pd.read_sql_query(sql_top5, conn)
    print("1. Read SQL Query 1 via pd.read_sql_query: SUCCESS")
    print(df_sql_top5[["title", "price_inr", "rating"]].head(2))

    # Demonstration 2: Reproducing JOIN query using pd.merge
    # The SQL JOIN query (Query 5):
    sql_join = """
    SELECT 
        b.book_id,
        b.title,
        c.category_name,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock
    FROM books b
    INNER JOIN categories c ON b.category_id = c.category_id
    WHERE b.rating = 5
    ORDER BY b.price_inr DESC;
    """
    df_sql_join = pd.read_sql_query(sql_join, conn)
    
    # Pandas reproduction:
    df_books = pd.read_sql_query("SELECT * FROM books", conn)
    df_categories = pd.read_sql_query("SELECT * FROM categories", conn)
    
    # Perform pd.merge
    df_pandas_merged = pd.merge(
        df_books, 
        df_categories, 
        on="category_id", 
        how="inner"
    )
    # Apply filtering and ordering
    df_pandas_join = df_pandas_merged[df_pandas_merged["rating"] == 5].sort_values(
        by="price_inr", 
        ascending=False
    )[["book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"]].reset_index(drop=True)
    
    # Reset index for df_sql_join for clean comparison
    df_sql_clean = df_sql_join.reset_index(drop=True)
    
    # Assert equivalence
    columns_to_compare = ["book_id", "title", "category_name", "price_inr", "rating", "in_stock"]
    diff = (df_sql_clean[columns_to_compare] != df_pandas_join[columns_to_compare]).sum().sum()
    
    print("\n2. Comparing SQL JOIN vs. Pandas pd.merge reproduction:")
    print(f"   SQL result shape:    {df_sql_clean.shape}")
    print(f"   Pandas result shape: {df_pandas_join.shape}")
    print(f"   Discrepancies found: {diff}")
    
    assert diff == 0, "SQL and Pandas JOIN outputs must be strictly equivalent!"
    print("   --> STRICT EQUIVALENCE VERIFIED: SQL JOIN == pd.merge")
    print("=" * 50 + "\n")
    return True

def run_module_1() -> Dict[str, Any]:
    """Runs the complete Module 1 data pipeline end-to-end."""
    print("=== STARTING MODULE 1: DATA ENGINEERING PIPELINE ===")
    
    # 1. Scrape raw books
    raw_books = run_scraper(output_path=RAW_JSON_PATH, target_count=60)
    
    # 2. Clean & convert
    df_cleaned, audit = clean_and_transform_books(raw_books)
    print(f"Cleaned {len(df_cleaned)} books across {df_cleaned['category'].nunique()} categories.")
    print(f"Imputed prices: {audit['imputed_price_count']}, Imputed ratings: {audit['imputed_rating_count']}")
    print(f"Fixed Conversion Rate: 1 GBP = {EXCHANGE_RATE_GBP_TO_INR} INR")
    
    # Save cleaned CSV
    os.makedirs(os.path.dirname(CLEANED_CSV_PATH), exist_ok=True)
    df_cleaned.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"Saved cleaned dataset to {CLEANED_CSV_PATH}")
    
    # 3. SQLite Storage
    num_cats, num_books = populate_database(df_cleaned, DEFAULT_DB_PATH)
    print(f"Populated SQLite database: {num_cats} categories, {num_books} books.")
    
    # 4. Execute Queries and Pandas Equivalency
    conn = get_connection(DEFAULT_DB_PATH)
    try:
        query_results = execute_and_log_queries(conn)
        demonstrate_pandas_equivalence(conn)
    finally:
        conn.close()
        
    print("=== MODULE 1 COMPLETED SUCCESSFULLY ===\n")
    return {
        "book_count": len(df_cleaned),
        "category_count": df_cleaned['category'].nunique(),
        "database_path": DEFAULT_DB_PATH,
        "exchange_rate": EXCHANGE_RATE_GBP_TO_INR
    }

if __name__ == "__main__":
    run_module_1()
