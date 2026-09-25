"""
Module 3: FastAPI Web Service
Exposes POST /ask endpoint wrapping the LangGraph RAG workflow.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add support_assistant directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fastapi.responses import HTMLResponse
from models import AskRequest, AskResponse
from graph import ask_question, is_mock_mode

app = FastAPI(
    title="Zepto Customer Support Policy Assistant",
    description="Grounded GenAI RAG assistant answering Zepto policy questions via LangGraph and ChromaDB.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

STATIC_INDEX_PATH = os.path.join(CURRENT_DIR, "static", "index.html")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the rich interactive Zepto AI Support Web Interface."""
    if os.path.exists(STATIC_INDEX_PATH):
        with open(STATIC_INDEX_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>Zepto Support Assistant API</h2><p><a href='/docs'>Swagger Docs</a></p>")

@app.get("/api/info")
def get_api_info():
    """Returns service metadata and endpoint map."""
    return {
        "service": "Zepto Support Assistant API",
        "status": "online",
        "mock_mode": is_mock_mode(),
        "endpoints": {
            "web_ui": "GET /",
            "ask": "POST /ask",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "mock_mode": is_mock_mode()}

@app.post("/ask", response_model=AskResponse)
def handle_ask(request: AskRequest):
    """
    Primary endpoint for processing customer questions.
    Routes between policy retrieval and direct general answer.
    """
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
