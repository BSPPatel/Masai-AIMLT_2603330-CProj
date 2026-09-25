# Zepto Data & AI Platform — Capstone Project

An end-to-end AI/ML engineering platform uniting Data Engineering, Predictive Machine Learning, and Generative AI into a cohesive, zero-cost, locally reproducible system.

$$\textbf{GET DATA} \longrightarrow \textbf{UNDERSTAND DATA} \longrightarrow \textbf{BUILD INTELLIGENCE} \longrightarrow \textbf{SERVE IT}$$

---

## 1. Quick Access Links & Web Interface

| Interface | URL | Description |
| :--- | :--- | :--- |
| **Interactive Web UI** | [**http://127.0.0.1:7860/**](http://127.0.0.1:7860/) | Rich Zepto-themed chat UI with prompt chips, source badges, and doc explorer |
| **Swagger API Docs** | [**http://127.0.0.1:7860/docs**](http://127.0.0.1:7860/docs) | Interactive OpenAPI documentation for direct endpoint testing |
| **Health Check** | [**http://127.0.0.1:7860/health**](http://127.0.0.1:7860/health) | Live service status endpoint |
| **Primary API Endpoint** | `POST http://127.0.0.1:7860/ask` | Pydantic JSON request/response policy assistant |

---

## 2. Project Architecture & Layout

```text
zepto-data-ai-platform/
│
├── .venv/                         # Isolated project Python virtual environment
├── requirements.txt               # Pinned dependencies (requests, bs4, pandas, scikit-learn, etc.)
├── .gitignore                     # Git configuration ignoring .venv, caches, checkpoints
├── run_all.py                     # Master one-click end-to-end runner script
├── README.md                      # Master root project documentation
│
├── data_pipeline/                 # MODULE 1: Data Engineering Pipeline
│   ├── scraper.py                 # Scrapes books.toscrape.com (163 items, 5 categories)
│   ├── cleaner.py                 # 105.50 INR/GBP fixed conversion, median imputation
│   ├── database.py                # Normalized SQLite schema with PK/FK constraints
│   ├── queries.sql                # 6 complex analytical SQL queries
│   ├── run_pipeline.py            # Pipeline orchestrator & Pandas equivalence proof
│   ├── data/                      # raw_books.json, cleaned_books.csv, sql_query_results.txt
│   ├── database/                  # zepto_catalog.db (SQLite relational store)
│   └── README.md                  # Module 1 detailed documentation
│
├── analytics/                     # MODULE 2: Analytics & Predictive Modeling
│   ├── dataset_loader.py          # Seaborn Titanic loader with local persistent cache
│   ├── eda_and_modeling.py        # Profiling, statistical EDA, ML models & regression
│   ├── generate_notebooks.py      # Script generating 01_eda.ipynb & 02_modeling.ipynb
│   ├── titanic.csv                # Persisted offline dataset (891 rows, 15 columns)
│   ├── 01_eda.ipynb               # Exploratory Data Analysis Jupyter Notebook
│   ├── 02_modeling.ipynb          # Predictive Modeling & Tuning Jupyter Notebook
│   ├── charts/                    # 10 generated analytical PNG visualizations
│   ├── models/                    # best_model.joblib (full pipeline: prep + tuned RF)
│   ├── analytics_summary.txt      # Execution audit log
│   └── README.md                  # Module 2 detailed documentation
│
├── support_assistant/             # MODULE 3: Generative AI Support Assistant
│   ├── docs/                      # 8 Zepto policy documents (doc_01.txt to doc_08.txt)
│   ├── static/index.html          # Interactive Web UI frontend
│   ├── models.py                  # Pydantic AskRequest and AskResponse schemas
│   ├── prompts.py                 # 5-skeleton structured prompt with negative constraints
│   ├── vector_store.py            # Local embedding indexer & top-3 cosine similarity retrieval
│   ├── graph.py                   # 3-node LangGraph StateGraph with conditional intent routing
│   ├── main.py                    # FastAPI application serving Web UI and POST /ask
│   ├── Dockerfile                 # Container image exposing port 7860
│   └── README.md                  # Module 3 detailed documentation
│
└── tests/                         # Automated Pytest Test Suite (23 test cases)
    ├── conftest.py                # Test environment path discovery
    ├── test_data_pipeline.py      # 8 unit and integration tests for Module 1
    ├── test_analytics.py          # 7 unit and integration tests for Module 2
    └── test_support_assistant.py  # 8 unit and integration tests for Module 3
```

---

## 3. Zero-Cost & Offline Guarantee
In strict compliance with the project specifications:
- The entire platform runs on **100% free, offline baselines**.
- No paid cloud services, external LLM API tokens, or live currency APIs are required.
- Module 1 uses a project-defined constant of **$1\text{ GBP} = 105.50\text{ INR}$**.
- Module 2 persists `titanic.csv` directly in the repository for air-gapped reproducibility.
- Module 3 operates deterministically in mock mode (`MOCK_LLM=1` or unset) using local embeddings and cosine similarity.

---

## 4. Environment Setup & One-Click Execution

All files, dependencies, and virtual environments remain **strictly inside the project directory**:

```bash
# 1. Create a dedicated virtual environment
python -m venv .venv

# 2. Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install requirements
pip install -r requirements.txt
```

### Run Everything with One Command
Execute all modules, database migrations, SQL queries, machine learning models, and automated tests sequentially:
```bash
.venv\Scripts\python.exe run_all.py
```

---

## 5. Module Details & Empirical Results

### Module 1: Data Engineering Pipeline
* **Source & Scrape:** Scraped 163 books across 5 categories (`Mystery`, `Historical Fiction`, `Travel`, `Classics`, `Sequential Art`) from `books.toscrape.com`.
* **Cleaning:** Stripped currency symbols, parsed ratings (1–5), in-stock booleans, and applied the fixed baseline: **$1\text{ GBP} = 105.50\text{ INR}$**.
* **Relational SQLite Schema:** `categories` and `books` tables linked by Foreign Key (`PRAGMA foreign_keys = ON;`).
* **SQL Queries & Pandas Equivalence:** Executed 6 queries covering `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and `JOIN`. Verified exact output equivalence between SQL `INNER JOIN` and Pandas `pd.merge` (`assert diff == 0`).

### Module 2: Analytics & Machine Learning
* **Threshold Cleaning:** Dropped `deck` column (>30% missing: 77.1%); dropped 2 rows in `embarked` (<5% missing: 0.22%); imputed `age` (19.9% missing) inside the pipeline.
* **Fare Skewness:** $\text{Mean } (32.10) > \text{Median } (14.45) > \text{Mode } (8.05)$ indicates heavy positive (**right-skewed**) distribution.
* **Exact 6-Column Correlation:** Top negative: `pclass` vs. `fare` ($-0.548$); Top positive: `sibsp` vs. `parch` ($+0.415$).
* **Multivariate Story:** 10 charts saved in `analytics/charts/` with written interpretations.
* **Side-by-Side Model Evaluation:**

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8146 | 0.7966 | 0.6912 | 0.7402 | 0.8596 | `[[98, 12], [21, 47]]` |
| **Decision Tree** | 0.8034 | 0.8000 | 0.6471 | 0.7154 | 0.8481 | `[[99, 11], [24, 44]]` |
| **Random Forest** | 0.8146 | 0.7692 | 0.7353 | 0.7519 | 0.8151 | `[[95, 15], [18, 50]]` |

* **Class Imbalance:** SMOTE applied only to the training set boosted minority recall from 0.6912 to 0.7353.
* **Hyperparameter Tuning:** Tuned Random Forest achieved an Out-Of-Bag (**OOB**) score of $0.8158$.
* **Fare Regression:** Multivariate linear regression achieved $\text{MAE} = 17.85$, $\text{RMSE} = 40.55$, $R^2 = 0.3838$. Residual plot shows pronounced **heteroscedasticity** at higher fares.
* **Joblib Pipeline:** Full pipeline serialized to `analytics/models/best_model.joblib` and re-tested on raw unprocessed inputs.

### Module 3: Generative AI Support Assistant
* **Corpus:** 8 Zepto policy files (`doc_01.txt` to `doc_08.txt`) covering Delivery, Returns, Membership, Tracking, Cancellations, Damaged Items, Gift Cards, and Support Hours.
* **Retrieval & LangGraph:** Local vector store performs top-3 cosine similarity retrieval. 3-node LangGraph `StateGraph` routes between `retrieve_and_answer` and `direct_answer`.
* **FastAPI & UI:** Serves interactive Web UI at `GET /` and API at `POST /ask`.

#### Recorded API Responses (`POST /ask`)
1. **Policy Question (Triggers Retrieval):**
   ```json
   {
     "answer": "Based on the retrieved context: 2. Delivery Charges\n- Standard delivery fee is Rs. 15 for orders below Rs. 149.\n- Orders of Rs. 149 or above qualify for free delivery for all standard customers.\n- Zepto Pass members receive unlimite",
     "sources": ["doc_01.txt_chunk_2", "doc_03.txt_chunk_2", "doc_05.txt_chunk_2"],
     "confidence": 1.0
   }
   ```
2. **Off-Topic Question (Triggers Direct Fallback):**
   ```json
   {
     "answer": "I can only answer questions about Zepto policies right now.",
     "sources": [],
     "confidence": 1.0
   }
   ```

---

## 6. Automated Pytest Suite (23/23 Passing)

Run the full automated test suite:
```bash
.venv\Scripts\pytest.exe tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
collected 23 items

tests/test_analytics.py::test_titanic_dataset_presence_and_shape PASSED  [  4%]
tests/test_analytics.py::test_missing_value_threshold_logic PASSED       [  8%]
tests/test_analytics.py::test_fare_central_tendency_and_skewness PASSED  [ 13%]
tests/test_analytics.py::test_six_column_correlation_matrix_constraint PASSED [ 17%]
tests/test_analytics.py::test_bivariate_boolean_masking PASSED           [ 21%]
tests/test_analytics.py::test_model_persistence_and_raw_inference PASSED [ 26%]
tests/test_analytics.py::test_all_charts_generated PASSED                [ 30%]
tests/test_data_pipeline.py::test_fixed_currency_exchange_rate PASSED    [ 34%]
tests/test_data_pipeline.py::test_clean_price_gbp PASSED                 [ 39%]
tests/test_data_pipeline.py::test_clean_rating PASSED                    [ 43%]
tests/test_data_pipeline.py::test_clean_availability PASSED              [ 47%]
tests/test_data_pipeline.py::test_clean_and_transform_pipeline PASSED    [ 52%]
tests/test_data_pipeline.py::test_sqlite_schema_and_integrity PASSED     [ 56%]
tests/test_data_pipeline.py::test_queries_contain_all_required_clauses PASSED [ 60%]
tests/test_data_pipeline.py::test_pandas_equivalence_and_join PASSED     [ 65%]
tests/test_support_assistant.py::test_eight_policy_documents_presence PASSED [ 69%]
tests/test_support_assistant.py::test_vector_store_indexing_and_top3_retrieval PASSED [ 73%]
tests/test_support_assistant.py::test_intent_classification_keywords PASSED [ 78%]
tests/test_support_assistant.py::test_langgraph_conditional_routing PASSED [ 82%]
tests/test_support_assistant.py::test_mock_mode_policy_flow PASSED       [ 86%]
tests/test_support_assistant.py::test_mock_mode_general_flow PASSED      [ 91%]
tests/test_support_assistant.py::test_pydantic_validation PASSED         [ 95%]
tests/test_support_assistant.py::test_fastapi_endpoints PASSED           [100%]

============================= 23 passed in 1.70s ==============================
```

---

## 7. Docker Execution

```bash
# 1. Build the Docker image
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .

# 2. Run container on port 7860
docker run -p 7860:7860 zepto-support-assistant

# 3. Open browser at http://localhost:7860
```

---

## 8. Git Branching History
The repository adheres strictly to professional Git branching practices:
```text
*   36c13ed Merge branch 'feature/platform-modules' into main: complete Modules 1, 2, 3 and test suites
|\  
| * bd175b0 feat(module3): implement GenAI support assistant, LangGraph, vector store, FastAPI, and root docs
| * 07f6c08 feat(module2): implement analytics, EDA, ML models, imbalance experiments, regression, and tests
| * 2195ee1 feat(module1): implement data pipeline, scraper, cleaner, sqlite storage, and tests
|/  
* 1a46774 chore: initial commit with project spec and gitignore
```
