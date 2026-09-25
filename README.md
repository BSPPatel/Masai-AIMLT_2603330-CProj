# Zepto Data & AI Platform — Capstone Project

## 1. Project Vision & Architecture
The **Zepto Data & AI Platform** is a connected, modular platform uniting Data Engineering, Predictive Machine Learning, and Generative AI into a cohesive engineering system:

$$\textbf{GET DATA} \longrightarrow \textbf{UNDERSTAND DATA} \longrightarrow \textbf{BUILD INTELLIGENCE} \longrightarrow \textbf{SERVE IT}$$

```text
zepto-data-ai-platform/
├── data_pipeline/         # Module 1: Web scraping, cleaning, SQLite storage & SQL analytics
│   ├── scraper.py
│   ├── cleaner.py
│   ├── database.py
│   ├── queries.sql
│   ├── run_pipeline.py
│   ├── data/
│   ├── database/
│   └── README.md
│
├── analytics/             # Module 2: Profiling, statistical EDA, ML models & regression
│   ├── dataset_loader.py
│   ├── eda_and_modeling.py
│   ├── generate_notebooks.py
│   ├── titanic.csv
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── charts/
│   ├── models/
│   └── README.md
│
├── support_assistant/     # Module 3: Policy RAG service, LangGraph workflow, FastAPI & Docker
│   ├── docs/ (8 policy texts)
│   ├── models.py
│   ├── prompts.py
│   ├── vector_store.py
│   ├── graph.py
│   ├── main.py
│   ├── Dockerfile
│   └── README.md
│
├── tests/                 # Automated Test Suite (23 test cases)
│   ├── conftest.py
│   ├── test_data_pipeline.py
│   ├── test_analytics.py
│   └── test_support_assistant.py
│
├── requirements.txt       # Environment dependencies
├── .gitignore             # Ignore venv, cache, checkpoints
└── README.md              # Root architectural documentation
```

---

## 2. Zero-Cost & Offline Guarantee
In strict compliance with the project specifications:
- The entire platform runs on **100% free, offline baselines**.
- No paid cloud services, external LLM API tokens, or live currency APIs are required.
- Module 1 uses a project-defined constant of **$1\text{ GBP} = 105.50\text{ INR}$**.
- Module 2 persists `titanic.csv` directly in the repository.
- Module 3 operates deterministically in mock mode (`MOCK_LLM=1` or unset) using local embeddings and cosine similarity.

---

## 3. Environment Setup & Installation

All files, virtual environments, and dependencies are isolated within the project directory to prevent conflicts:

```bash
# 1. Create a dedicated project virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt
```

---

## 4. Module-by-Module Execution

### Module 1: Data Engineering Pipeline
Scrapes `books.toscrape.com`, parses prices and ratings, applies the $105.50$ INR/GBP rate, stores in normalized SQLite, executes 6 SQL queries, and proves Pandas `pd.merge` equivalence:
```bash
.venv\Scripts\python.exe data_pipeline/run_pipeline.py
```

### Module 2: Analytics & Machine Learning
Loads the Titanic dataset, executes threshold-based cleaning, statistical EDA, leakage-free stratified modeling (Logistic Regression, Decision Tree, Random Forest), class-imbalance experiments, GridSearchCV tuning with OOB score, fare regression, and Joblib serialization:
```bash
.venv\Scripts\python.exe analytics/eda_and_modeling.py
```

### Module 3: Generative AI Support Assistant
Starts the FastAPI policy assistant backed by a 3-node LangGraph StateGraph, cosine similarity retrieval across 8 Zepto policies, and Pydantic validation:
```bash
.venv\Scripts\uvicorn.exe support_assistant.main:app --host 0.0.0.0 --port 7860
```

---

## 5. Automated Testing Suite

A suite of **23 automated tests** verifies all functional contracts, schema constraints, data leakage prevention, and API behavior:

```bash
.venv\Scripts\pytest.exe tests/ -v
```

### Test Coverage Summary:
- **Module 1 (`tests/test_data_pipeline.py` - 8 tests):**
  - Currency conversion rate ($105.50$).
  - Cleaning logic for `price_gbp`, `rating` (1–5), and `in_stock`.
  - Median imputation and malformed row handling.
  - SQLite schema integrity and Foreign Key enforcement.
  - SQL clauses coverage (`SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`/`IN`, `JOIN`).
  - Mathematical equivalence between SQL `INNER JOIN` and Pandas `pd.merge`.

- **Module 2 (`tests/test_analytics.py` - 7 tests):**
  - Titanic CSV presence and 891-row structure.
  - Threshold missing-value logic ($<5\%$ drop rows, $>30\%$ drop column).
  - Skewness condition ($\text{Mean} > \text{Median} > \text{Mode}$).
  - Exact 6-column correlation matrix restriction.
  - Bivariate boolean masking calculations.
  - Production Joblib pipeline deserialization and raw feature inference.
  - Verification of all 10 generated PNG charts.

- **Module 3 (`tests/test_support_assistant.py` - 8 tests):**
  - Verification of all 8 policy documents.
  - Vector indexing and top-3 cosine similarity retrieval.
  - Intent classification keyword heuristics.
  - LangGraph conditional edge routing.
  - Grounded mock response format and direct answer fallback.
  - Pydantic schema validation.
  - FastAPI `/health` and `/ask` endpoints with HTTP 200/400 validation.

---

## 6. Docker Container Execution

Build and run the container locally:
```bash
# Build the Docker image
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .

# Run the container exposing port 7860
docker run -p 7860:7860 zepto-support-assistant
```

Test the live endpoint:
```bash
curl -X POST "http://localhost:7860/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the return policy for dairy products?"}'
```

---

## 7. Key Findings & Design Decisions

1. **Currency Standardization:** A fixed constant of $105.50$ INR per GBP guarantees deterministic output without reliance on fluctuating or rate-limited live exchange APIs.
2. **Relational vs. In-Memory:** Relational SQLite storage with foreign keys guarantees data normalization, and strict `pd.merge` verification proves analytical parity across SQL and Pandas.
3. **Leakage Prevention:** All data transformations, scalers, imputers, and SMOTE resampling in Module 2 are fit strictly on training partitions after a stratified split.
4. **Model Selection:** Random Forest with $OOB=0.816$ and $F1=0.768$ was selected for production deployment because it captures non-linear socio-economic interactions while remaining robust to multicollinearity.
5. **Deterministic RAG Baseline:** By using an intent-routed LangGraph `StateGraph` and local cosine vector retrieval, Module 3 guarantees policy grounding and reproducibility regardless of external cloud API availability.
