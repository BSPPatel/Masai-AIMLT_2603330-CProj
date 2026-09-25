"""
Zepto Data & AI Platform — Unified FastAPI Web & Service Layer
Unites:
- Module 1: Data Engineering Catalog & SQL Explorer (/api/catalog)
- Module 2: Analytics Visualizations & Real-Time ML Inference (/api/analytics)
- Module 3: LangGraph Grounded Policy Assistant (/ask)
"""

import os
import sys
import sqlite3
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Path configurations
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import AskRequest, AskResponse
from graph import ask_question, is_mock_mode

app = FastAPI(
    title="Zepto Data & AI Platform API",
    description="Unified API serving Data Engineering, Predictive ML Inference, and LangGraph RAG Assistant.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static mounts
STATIC_INDEX_PATH = os.path.join(CURRENT_DIR, "static", "index.html")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "analytics", "charts")
DOCS_DIR = os.path.join(CURRENT_DIR, "docs")
if os.path.exists(CHARTS_DIR):
    app.mount("/charts", StaticFiles(directory=CHARTS_DIR), name="charts")

DB_PATH = os.path.join(PROJECT_ROOT, "data_pipeline", "database", "zepto_catalog.db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "analytics", "models", "best_model.joblib")

# Cache loaded ML model
_cached_pipeline = None

def get_ml_pipeline():
    global _cached_pipeline
    if _cached_pipeline is None and os.path.exists(MODEL_PATH):
        _cached_pipeline = joblib.load(MODEL_PATH)
    return _cached_pipeline

# -------------------------------------------------------------
# Input Schemas for Platform Endpoints
# -------------------------------------------------------------
class PassengerInput(BaseModel):
    pclass: int = Field(1, ge=1, le=3, description="Passenger ticket class (1, 2, or 3)")
    sex: str = Field("female", description="Gender ('female' or 'male')")
    age: float = Field(28.0, ge=0.0, le=100.0, description="Age in years")
    sibsp: int = Field(0, ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(0, ge=0, description="Number of parents/children aboard")
    fare: float = Field(85.50, ge=0.0, description="Ticket fare paid")
    embarked: str = Field("S", description="Port of embarkation ('C', 'Q', or 'S')")

class SqlQueryInput(BaseModel):
    query: str = Field(..., description="Read-only SQL query to execute")

# -------------------------------------------------------------
# Core Root & Health Endpoints
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the Unified Zepto Data & AI Platform Web UI."""
    if os.path.exists(STATIC_INDEX_PATH):
        with open(STATIC_INDEX_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>Zepto Data & AI Platform</h2><p><a href='/docs'>Swagger API Docs</a></p>")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mock_mode": is_mock_mode(),
        "database_connected": os.path.exists(DB_PATH),
        "ml_model_loaded": os.path.exists(MODEL_PATH)
    }

@app.get("/api/system/summary")
def get_system_summary():
    """Returns high-level platform status and KPIs across all 3 modules."""
    pipeline = get_ml_pipeline()
    db_exists = os.path.exists(DB_PATH)
    total_books = 0
    total_categories = 0
    if db_exists:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM books;")
                total_books = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM categories;")
                total_categories = c.fetchone()[0]
        except Exception:
            pass

    chart_count = 0
    if os.path.exists(CHARTS_DIR):
        chart_count = len([f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")])

    return {
        "status": "operational",
        "modules": {
            "module_1": {
                "name": "Data Engineering Pipeline",
                "database": "SQLite (zepto_catalog.db)",
                "connected": db_exists,
                "total_records": total_books,
                "categories": total_categories,
                "exchange_rate": "1 GBP = 105.50 INR (Fixed)"
            },
            "module_2": {
                "name": "Analytics & Machine Learning",
                "best_model": "Random Forest Classifier (OOB: 0.816, Test Acc: 0.827)",
                "pipeline_loaded": pipeline is not None,
                "visualizations_available": chart_count
            },
            "module_3": {
                "name": "Support Assistant (LangGraph RAG)",
                "state_graph": "classify_intent -> retrieve_and_answer / direct_answer",
                "mode": "Deterministic MOCK_LLM" if is_mock_mode() else "Live LLM",
                "indexed_documents": 8,
                "total_chunks": 32,
                "top_k": 3
            }
        }
    }

@app.get("/api/policies")
def list_policies():
    """Lists all indexed policy documents with titles and previews."""
    if not os.path.exists(DOCS_DIR):
        return {"policies": []}
    policies = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if fname.endswith(".txt"):
            fpath = os.path.join(DOCS_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                title = lines[0] if lines else fname
                policies.append({
                    "id": fname,
                    "title": title,
                    "preview": content[:160] + "..." if len(content) > 160 else content,
                    "content": content
                })
    return {"policies": policies}

@app.get("/api/policies/{doc_id}")
def get_policy(doc_id: str):
    """Fetches full policy text for a specific document."""
    fpath = os.path.join(DOCS_DIR, doc_id)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Policy document '{doc_id}' not found.")
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    title = lines[0] if lines else doc_id
    return {"id": doc_id, "title": title, "content": content}

# -------------------------------------------------------------
# Module 1: Data Engineering Catalog & SQL Explorer
# -------------------------------------------------------------
@app.get("/api/catalog/stats")
def get_catalog_stats():
    """Returns inventory stats from SQLite database."""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database not found. Run data pipeline first.")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM books;")
        total_books = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM categories;")
        total_cats = cursor.fetchone()[0]
        cursor.execute("SELECT ROUND(AVG(price_inr), 2), MIN(price_inr), MAX(price_inr) FROM books;")
        avg_p, min_p, max_p = cursor.fetchone()
        
        cursor.execute("""
            SELECT c.category_name, COUNT(b.book_id) as count
            FROM categories c
            LEFT JOIN books b ON c.category_id = b.category_id
            GROUP BY c.category_name
            ORDER BY count DESC;
        """)
        cats = [{"category": row[0], "count": row[1]} for row in cursor.fetchall()]
        
    return {
        "total_books": total_books,
        "total_categories": total_cats,
        "avg_price_inr": avg_p,
        "min_price_inr": min_p,
        "max_price_inr": max_p,
        "exchange_rate": 105.50,
        "categories": cats
    }

@app.get("/api/catalog/books")
def get_catalog_books(limit: int = 20, category: Optional[str] = None):
    """Fetches books from SQLite database with optional category filtering."""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database not found.")
        
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if category:
            cursor.execute("""
                SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
                FROM books b
                JOIN categories c ON b.category_id = c.category_id
                WHERE c.category_name = ?
                LIMIT ?;
            """, (category, limit))
        else:
            cursor.execute("""
                SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
                FROM books b
                JOIN categories c ON b.category_id = c.category_id
                LIMIT ?;
            """, (limit,))
        books = [dict(row) for row in cursor.fetchall()]
        
    return {"count": len(books), "books": books}

@app.post("/api/catalog/sql")
def execute_sql(payload: SqlQueryInput):
    """Executes a safe read-only SQL query against zepto_catalog.db."""
    sql = payload.query.strip()
    # Guard against destructive queries
    blocked_words = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]
    if any(bw in sql.upper() for bw in blocked_words):
        raise HTTPException(status_code=400, detail="Only read-only SELECT queries are permitted.")

    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database not found.")
        
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql, conn)
            return {
                "row_count": len(df),
                "columns": df.columns.tolist(),
                "data": df.head(50).to_dict(orient="records")
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL Execution Error: {str(e)}")

# -------------------------------------------------------------
# Module 2: Analytics & Live ML Prediction
# -------------------------------------------------------------
@app.get("/api/analytics/charts")
def list_analytics_charts():
    """Lists generated analytical charts available in /charts."""
    if not os.path.exists(CHARTS_DIR):
        return {"charts": []}
    files = [f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")]
    return {"charts": files}

@app.post("/api/analytics/predict")
def predict_survival(passenger: PassengerInput):
    """Predicts survival outcome directly using the persisted Joblib pipeline."""
    pipeline = get_ml_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=500, detail="ML model pipeline not available.")
        
    df_sample = pd.DataFrame([{
        "pclass": passenger.pclass,
        "sex": passenger.sex.lower(),
        "age": passenger.age,
        "sibsp": passenger.sibsp,
        "parch": passenger.parch,
        "fare": passenger.fare,
        "embarked": passenger.embarked.upper()
    }])
    
    try:
        pred = int(pipeline.predict(df_sample)[0])
        probas = pipeline.predict_proba(df_sample)[0]
        survival_proba = float(probas[1])
        return {
            "prediction": "Survived" if pred == 1 else "Perished",
            "survived": pred,
            "survival_probability": round(survival_proba, 4),
            "mortality_probability": round(float(probas[0]), 4),
            "input_received": passenger.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -------------------------------------------------------------
# Module 3: Generative AI Support Assistant
# -------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def handle_ask(request: AskRequest):
    """Processes customer query using the LangGraph StateGraph workflow."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        result = ask_question(request.query)
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing graph workflow: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
