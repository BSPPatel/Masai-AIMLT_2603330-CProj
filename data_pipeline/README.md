# Module 1: Data Engineering Pipeline

## Overview
This module implements an automated end-to-end data engineering pipeline that extracts raw catalog data, cleans and normalizes it, enforces relational constraints in SQLite, executes complex analytical queries, and mathematically validates Pandas reproduction equivalence.

$$\textbf{SCRAPE} \longrightarrow \textbf{CLEAN} \longrightarrow \textbf{CONVERT} \longrightarrow \textbf{STORE} \longrightarrow \textbf{QUERY} \longrightarrow \textbf{ANALYZE}$$

---

## 1. Data Collection (`scraper.py`)
- **Target Source:** `http://books.toscrape.com` (public scraping sandbox).
- **Libraries:** Python `requests`, `BeautifulSoup4`.
- **Extraction Scope:** Scrapes across targeted categories (`Mystery`, `Historical Fiction`, `Travel`, `Classics`, `Sequential Art`) plus catalog pages.
- **Yield:** Meets and exceeds the threshold of $\ge 60$ books across $\ge 3$ categories.
- **Collected Fields:** `title`, `price` (raw GBP string), `star_rating` (text), `availability` (raw text), `category`.

---

## 2. Data Cleaning & Type Conversion (`cleaner.py`)
- **`price_gbp` (`float`):** Strips currency symbols (`£`, `Â£`, `$`) and converts to 2-decimal floating point.
- **`rating` (`int` 1–5):** Maps textual numbers (`One` $\to$ 1, `Two` $\to$ 2, `Three` $\to$ 3, `Four` $\to$ 4, `Five` $\to$ 5).
- **`in_stock` (`bool`):** Evaluates availability string (True for in-stock, False otherwise).
- **Missing Value Handling:**
  - Missing title: Row is dropped (unusable catalog entry).
  - Missing numeric price: Category-level median imputation (fallback to global median).
  - Missing rating: Global median rating imputation.
- **Fixed Currency Baseline:**
  $$\mathbf{1\text{ GBP} = 105.50\text{ INR}}$$
  *Strict project-defined constant. No live network API dependency.*
  `price_inr` is computed as:
  $$\text{price\_inr} = \text{round}(\text{price\_gbp} \times 105.50, 2)$$

---

## 3. Relational Schema (`database.py`)
Data is stored in SQLite (`zepto_catalog.db`) with Foreign Key enforcement (`PRAGMA foreign_keys = ON;`).

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id) ON DELETE CASCADE
);
```

---

## 4. SQL Analysis & Pandas Equivalence (`queries.sql` & `run_pipeline.py`)
The pipeline runs 6 SQL queries demonstrating:
1. `SELECT`, `WHERE`, `ORDER BY`, `LIMIT` (Top 5 most expensive in-stock books).
2. `DISTINCT`, `WHERE` (Distinct ratings for premium items).
3. `BETWEEN`, `ORDER BY` (Affordable highly-rated items).
4. `IN`, `JOIN`, `WHERE` (Books in selected customer-favorite genres).
5. `JOIN`, `ORDER BY` (Inner join of books and categories for 5-star titles).
6. `GROUP BY`, Aggregate metrics (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`).

### Equivalence Proof
- Query results are loaded via `pd.read_sql_query`.
- The relational `INNER JOIN` is reproduced natively in Pandas using:
  ```python
  df_pandas_merged = pd.merge(df_books, df_categories, on="category_id", how="inner")
  ```
- Strict equality (`assert diff == 0`) confirms SQL and Pandas outputs are identical.

---

## 5. Execution Instructions
Run the entire module pipeline using the dedicated project virtual environment:
```bash
.venv\Scripts\python.exe data_pipeline/run_pipeline.py
```
Run the automated test suite:
```bash
.venv\Scripts\pytest.exe tests/test_data_pipeline.py -v
```
