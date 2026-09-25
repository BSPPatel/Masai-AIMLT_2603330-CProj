"""
Module 3: Data Contracts
Defines Pydantic models for request and response validation.
"""

from typing import List
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's question")

class AskResponse(BaseModel):
    answer: str = Field(..., description="The grounded answer or direct response")
    sources: List[str] = Field(default_factory=list, description="IDs of source documents/chunks used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Zepto delivers in 10 minutes between 6:00 AM and 2:00 AM daily.",
                "sources": ["doc_01.txt"],
                "confidence": 1.0
            }
        }
    }
