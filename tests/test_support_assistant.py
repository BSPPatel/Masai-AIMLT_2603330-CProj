"""
Automated Test Suite for Module 3: Generative AI Support Assistant
Tests policy documents, vector indexing, top-3 cosine retrieval,
LangGraph workflow, intent classification, Pydantic schemas, and FastAPI endpoints.
"""

import os
import glob
import pytest
from fastapi.testclient import TestClient

from support_assistant.models import AskRequest, AskResponse
from support_assistant.vector_store import get_vector_store, PolicyVectorStore
from support_assistant.graph import (
    classify_intent_node,
    route_intent,
    retrieve_and_answer_node,
    direct_answer_node,
    ask_question,
    POLICY_KEYWORDS
)
from support_assistant.main import app

client = TestClient(app)

def test_eight_policy_documents_presence():
    """Verify exactly 8 policy documents exist in support_assistant/docs/."""
    doc_paths = sorted(glob.glob("support_assistant/docs/doc_*.txt"))
    assert len(doc_paths) == 8, f"Expected 8 policy docs, found {len(doc_paths)}"
    
    expected_filenames = [f"doc_0{i}.txt" for i in range(1, 9)]
    actual_filenames = [os.path.basename(p) for p in doc_paths]
    assert actual_filenames == expected_filenames

def test_vector_store_indexing_and_top3_retrieval():
    """Verify vector store chunks documents and performs top-3 similarity search."""
    vs = get_vector_store()
    assert len(vs.chunks) >= 8
    
    # Query for delivery policy
    results = vs.search("What are the delivery hours and standard delivery fee?", top_k=3)
    assert len(results) == 3
    assert all("id" in r and "text" in r and "score" in r for r in results)
    
    # The top retrieved chunk should be from doc_01.txt (Delivery Policy)
    assert "doc_01.txt" in results[0]["id"]
    assert "delivery" in results[0]["text"].lower()

def test_intent_classification_keywords():
    """Test required keyword classification logic."""
    for kw in POLICY_KEYWORDS:
        state = {"query": f"Can you explain the {kw} rules?", "intent": "", "retrieved_chunks": [], "answer": "", "sources": [], "confidence": 0.0}
        out = classify_intent_node(state)
        assert out["intent"] == "policy_question", f"Keyword '{kw}' should trigger policy_question"

    general_state = {"query": "Write a poem about the sunrise.", "intent": "", "retrieved_chunks": [], "answer": "", "sources": [], "confidence": 0.0}
    assert classify_intent_node(general_state)["intent"] == "general_question"

def test_langgraph_conditional_routing():
    """Verify conditional edge router selects appropriate node destination."""
    policy_state = {"intent": "policy_question"}
    assert route_intent(policy_state) == "retrieve_and_answer"

    general_state = {"intent": "general_question"}
    assert route_intent(general_state) == "direct_answer"

def test_mock_mode_policy_flow():
    """Verify deterministic mock response format for policy inquiries."""
    res = ask_question("What is the refund policy for damaged milk?")
    assert res["answer"].startswith("Based on the retrieved context:")
    assert len(res["sources"]) > 0
    assert res["confidence"] == 1.0

def test_mock_mode_general_flow():
    """Verify deterministic mock response for general off-topic inquiries."""
    res = ask_question("What is the capital of Spain?")
    assert res["answer"] == "I can only answer questions about Zepto policies right now."
    assert res["sources"] == []
    assert res["confidence"] == 1.0

def test_pydantic_validation():
    """Test schema validation on AskRequest and AskResponse models."""
    req = AskRequest(query="What is Zepto Pass membership?")
    assert req.query == "What is Zepto Pass membership?"
    
    with pytest.raises(Exception):
        AskRequest(query="") # Empty query validation
        
    resp = AskResponse(answer="Sample answer", sources=["doc_03.txt"], confidence=1.0)
    assert resp.confidence == 1.0
    
    with pytest.raises(Exception):
        AskResponse(answer="Sample", sources=[], confidence=1.5) # Out of range confidence

def test_fastapi_endpoints():
    """Test FastAPI GET /health, and POST /ask for policy and general queries."""
    # Health check
    h_res = client.get("/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"
    
    # Policy query
    pol_res = client.post("/ask", json={"query": "How do gift cards work?"})
    assert pol_res.status_code == 200
    data = pol_res.json()
    assert "answer" in data and "sources" in data and "confidence" in data
    assert data["answer"].startswith("Based on the retrieved context:")
    assert len(data["sources"]) > 0
    assert any("doc_07.txt" in s for s in data["sources"])
    
    # General query
    gen_res = client.post("/ask", json={"query": "How many planets are in the solar system?"})
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["answer"] == "I can only answer questions about Zepto policies right now."
    assert gen_data["sources"] == []
    assert gen_data["confidence"] == 1.0
    
    # Empty query should return 400 or 422
    err_res = client.post("/ask", json={"query": "   "})
    assert err_res.status_code in [400, 422]
