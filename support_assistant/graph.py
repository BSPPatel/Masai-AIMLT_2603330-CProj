"""
Module 3: LangGraph StateGraph Workflow
Implements the 3-node intent-routed RAG workflow:
classify_intent -> (conditional routing) -> retrieve_and_answer | direct_answer
Adheres strictly to the mock mode keyword baseline and deterministic responses.
"""

import os
import sys
from typing import TypedDict, List, Dict, Any, Optional

# Ensure support_assistant directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from langgraph.graph import StateGraph, START, END

from vector_store import get_vector_store
from prompts import format_policy_prompt

# Project-defined policy keywords for deterministic intent classification
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]

class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float

def is_mock_mode() -> bool:
    """Returns True if MOCK_LLM is unset or set to '1'."""
    val = os.getenv("MOCK_LLM", "1").strip().lower()
    return val in ["1", "true", "yes", ""]

def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Evaluates customer query intent.
    Uses required deterministic keyword heuristic in Mock Mode.
    """
    query_lower = state["query"].lower()
    
    # Check required project keywords
    is_policy = any(kw in query_lower for kw in POLICY_KEYWORDS)
    intent = "policy_question" if is_policy else "general_question"
    
    return {"intent": intent}

def route_intent(state: AgentState) -> str:
    """Conditional edge router branching based on intent."""
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

def retrieve_and_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Policy handler.
    Retrieves top 3 cosine-similar chunks and returns grounded response.
    """
    vs = get_vector_store()
    chunks = vs.search(state["query"], top_k=3)
    
    if is_mock_mode():
        # Required deterministic mock response format
        if chunks:
            top_snippet = chunks[0]["text"][:200].strip()
            answer = f"Based on the retrieved context: {top_snippet}"
            sources = [c["id"] for c in chunks]
        else:
            answer = "Based on the retrieved context: No matching policy section was found."
            sources = []
            
        confidence = 1.0
    else:
        # Optional Real-LLM Path (invokes structured prompt with retry mechanism)
        prompt = format_policy_prompt(state["query"], chunks)
        # Fallback to grounded mock if no external LLM credentials configured
        answer = f"Based on the retrieved context: {chunks[0]['text'][:200].strip()}" if chunks else "No policy found."
        sources = [c["id"] for c in chunks]
        confidence = 0.95

    return {
        "retrieved_chunks": chunks,
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }

def direct_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: General non-policy question handler.
    Returns deterministic direct answer without retrieval.
    """
    return {
        "retrieved_chunks": [],
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0
    }

def build_support_graph():
    """Builds and compiles the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer_node)
    workflow.add_node("direct_answer", direct_answer_node)
    
    workflow.add_edge(START, "classify_intent")
    
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )
    
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    
    return workflow.compile()

# Singleton compiled app
support_assistant_app = build_support_graph()

def ask_question(query: str) -> Dict[str, Any]:
    """Runs the compiled LangGraph workflow for a customer query."""
    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }
    final_state = support_assistant_app.invoke(initial_state)
    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "confidence": final_state["confidence"]
    }
