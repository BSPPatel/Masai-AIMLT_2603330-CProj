"""
Module 1: Cleaner
Transforms raw scraped catalog data into typed, validated, and normalized records.
Implements the fixed baseline currency conversion: 1 GBP = 105.50 INR.
"""

import re
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

EXCHANGE_RATE_GBP_TO_INR = 105.50

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}

def clean_price_gbp(raw_price: Any) -> Optional_Float:
    """
    Parses raw price string (e.g., '£51.77', 'Â£22.60', '51.77') to float.
    Returns None if parsing fails.
    """
    if raw_price is None or pd.isna(raw_price):
        return None
    raw_str = str(raw_price)
    # Strip any currency symbols (£, Â, $, €, etc.) and whitespace
    match = re.search(r"[-+]?\d*\.\d+|\d+", raw_str)
    if match:
        try:
            return round(float(match.group(0)), 2)
        except ValueError:
            return None
    return None

def clean_rating(raw_rating: Any) -> Optional_Int:
    """
    Parses star rating text ('One', 'Two', etc.) or integer to 1-5.
    Returns None if unparseable.
    """
    if raw_rating is None or pd.isna(raw_rating):
        return None
    raw_str = str(raw_rating).strip().lower()
    if raw_str in RATING_MAP:
        return RATING_MAP[raw_str]
    # Check if already a number
    try:
        val = int(float(raw_str))
        if 1 <= val <= 5:
            return val
    except ValueError:
        pass
    return None

def clean_availability(raw_avail: Any) -> bool:
    """
    Parses availability text to boolean (True if in stock, else False).
    """
    if raw_avail is None or pd.isna(raw_avail):
        return False
    raw_str = str(raw_avail).lower()
    return "in stock" in raw_str and "not" not in raw_str

def clean_and_transform_books(
    raw_books: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw book entries, calculates price_inr using the 105.50 conversion constant,
    applies median imputation for missing numeric prices/ratings,
    and returns a cleaned pandas DataFrame along with an audit dictionary.
    """
    audit = {
        "raw_count": len(raw_books),
        "dropped_rows": 0,
        "imputed_price_count": 0,
        "imputed_rating_count": 0,
        "exchange_rate": EXCHANGE_RATE_GBP_TO_INR
    }

    records = []
    for item in raw_books:
        title = str(item.get("title", "")).strip()
        category = str(item.get("category", "")).strip() or "General"
        
        # If title is missing entirely, drop row
        if not title:
            audit["dropped_rows"] += 1
            continue
            
        p_gbp = clean_price_gbp(item.get("price"))
        rating = clean_rating(item.get("star_rating"))
        in_stock = clean_availability(item.get("availability"))
        
        records.append({
            "title": title,
            "category": category,
            "price_gbp": p_gbp,
            "rating": rating,
            "in_stock": in_stock
        })
        
    df = pd.DataFrame(records)
    
    if df.empty:
        df = pd.DataFrame(columns=["title", "category", "price_gbp", "price_inr", "rating", "in_stock"])
        return df, audit

    # Impute missing price_gbp using category median (or overall median fallback)
    overall_median_price = df["price_gbp"].median()
    if pd.isna(overall_median_price):
        overall_median_price = 25.00
        
    missing_price_mask = df["price_gbp"].isna()
    audit["imputed_price_count"] = int(missing_price_mask.sum())
    if audit["imputed_price_count"] > 0:
        df["price_gbp"] = df.groupby("category")["price_gbp"].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else overall_median_price)
        )
        df["price_gbp"] = df["price_gbp"].fillna(overall_median_price)
        
    # Impute missing rating using median rating
    overall_median_rating = int(df["rating"].median()) if not pd.isna(df["rating"].median()) else 3
    missing_rating_mask = df["rating"].isna()
    audit["imputed_rating_count"] = int(missing_rating_mask.sum())
    df["rating"] = df["rating"].fillna(overall_median_rating).astype(int)

    # Compute price_inr using required fixed constant: 1 GBP = 105.50 INR
    df["price_inr"] = (df["price_gbp"] * EXCHANGE_RATE_GBP_TO_INR).round(2)
    df["in_stock"] = df["in_stock"].astype(bool)

    return df, audit

# Type alias helper
Optional_Float = Any
Optional_Int = Any
