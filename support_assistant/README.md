# Module 3: Generative AI Support Assistant

## Overview
This module implements a locally runnable, policy-grounded customer support service. It features an intent-routed LangGraph `StateGraph` workflow, cosine-similarity vector retrieval over 8 Zepto policy documents, Pydantic structured output validation, a FastAPI web service, and Docker containerization.

$$\textbf{INGESTION} \longrightarrow \textbf{EMBEDDING} \longrightarrow \textbf{RETRIEVAL} \longrightarrow \textbf{GENERATION} \longrightarrow \textbf{API / CONTAINER}$$

---

## 1. RAG Architecture & Data Flow

```text
                  Customer Query (POST /ask)
                              │
                              ▼
                  ┌───────────────────────┐
                  │    classify_intent    │
                  └───────────┬───────────┘
                              │
               Conditional Edge (Keyword Router)
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   [policy_question]                  [general_question]
             │                                 │
             ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│   retrieve_and_answer   │       │      direct_answer      │
│  - Embed query          │       │  - Return deterministic │
│  - Top-3 cosine search  │       │    policy restriction   │
│  - Grounded output      │       │    message              │
└────────────┬────────────┘       └────────────┬────────────┘
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                 Pydantic Response Validation
                              │
                              ▼
                   HTTP 200 JSON Response
```

### Pipeline Stages
1. **INGESTION (`vector_store.py`):**
   - Source corpus: 8 policy files in `support_assistant/docs/` (`doc_01.txt` to `doc_08.txt`).
   - Topics: Delivery, Returns & Refunds, Membership, Order Tracking, Cancellations, Damaged Items, Gift Cards, Support Hours.
   - Chunking: Paragraph-level segmentation generating unique chunk IDs (`doc_01.txt_chunk_0`, etc.).
2. **EMBEDDING (`vector_store.py`):**
   - Model: `sentence-transformers` (`all-MiniLM-L6-v2`) with pure offline fallback to Scikit-Learn TF-IDF unit-normalized vectors.
   - Vectors: Stored in an indexed in-memory cosine vector store.
3. **RETRIEVAL (`vector_store.py`):**
   - Searches top-3 most similar chunks using cosine similarity ($A \cdot B / (\|A\| \|B\|)$).
4. **GENERATION (`graph.py`):**
   - Controlled by the `MOCK_LLM` environment variable.
   - **Mock Mode (Default / Graded Baseline: `MOCK_LLM=1` or unset):**
     - Fully deterministic and runs 100% offline without external LLM calls or API keys.
     - Policy answers format: `"Based on the retrieved context: <top chunk snippet>"`.
     - Confidence: `1.0`.
   - **Real-LLM Mode (Optional Extension: `MOCK_LLM=0`):**
     - Generates grounded responses using the 5-component structured prompt (`ROLE`, `CONTEXT`, `TASK`, `FORMAT`, `LENGTH`), negative constraints, and few-shot examples defined in `prompts.py`.
     - Includes up to 2 retries for schema validation failures.

---

## 2. Policy Corpus Directory
```text
support_assistant/docs/
├── doc_01.txt   # Delivery Policy (10-min delivery, fees, free above Rs 149)
├── doc_02.txt   # Returns & Refunds (perishable rules, 2-4 hr UPI refunds)
├── doc_03.txt   # Membership Tiers (Zepto Pass Rs 99/mo, perks)
├── doc_04.txt   # Order Tracking (GPS tracking, packing, dispatch)
├── doc_05.txt   # Order Cancellation Policy (free within 60s, Rs 30 post-packing)
├── doc_06.txt   # Damaged or Missing Items (photo evidence, 24h reporting)
├── doc_07.txt   # Gift Cards (Rs 250-10000, 12 month validity)
└── doc_08.txt   # Customer Support Hours (24/7/365 chat, phone, email)
```

---

## 3. Recorded API Examples (`POST /ask`)

### Example 1: Policy Question (Triggers Retrieval)
**Request:**
```bash
curl -X POST "http://localhost:7860/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery policy?"}'
```
**Recorded JSON Response:**
```json
{
  "answer": "Based on the retrieved context: Zepto Delivery Policy",
  "sources": [
    "doc_01.txt_chunk_0",
    "doc_07.txt_chunk_0",
    "doc_05.txt_chunk_0"
  ],
  "confidence": 1.0
}
```

### Example 2: General / Off-Topic Question (Triggers Direct Answer)
**Request:**
```bash
curl -X POST "http://localhost:7860/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "How far is the moon?"}'
```
**Recorded JSON Response:**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## 4. Docker Execution
Build the Docker image:
```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
```
Run the container on port 7860:
```bash
docker run -p 7860:7860 zepto-support-assistant
```
Verify the live container API:
```bash
curl http://localhost:7860/health
```

---

## 5. Execution Instructions
Run service locally via Uvicorn:
```bash
.venv\Scripts\uvicorn.exe support_assistant.main:app --host 0.0.0.0 --port 7860
```
Run automated test suite:
```bash
.venv\Scripts\pytest.exe tests/test_support_assistant.py -v
```
