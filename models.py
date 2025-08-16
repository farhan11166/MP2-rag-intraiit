# models.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel): 
    question: str
    messages: Optional[List[str]] = []

class SummaryResponse(BaseModel): 
    summary: str
    textforbot: List[str]
    citations: List[Dict[str, str]] = []

class ChatResponse(BaseModel): 
    answer: str
    sources: List[Dict[str, str]] = []

class ErrorResponse(BaseModel): 
    error: str