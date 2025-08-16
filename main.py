# Standard library imports
import os
import logging
import asyncio
import tempfile
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Third-party libraries
import numpy as np
import fitz  # PyMuPDF
import faiss
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

# Local imports
from config import logger, chat_model, embedding_model
from agents import run_summarization_crew, get_citations_from_topics
from tools import search_web_content

# --- FastAPI App Setup ---
app = FastAPI(
    title="AI Document Summarizer with Citations",
    description="An API for summarizing documents with internet citations using CrewAI agents.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
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

# --- Vector Store ---
@dataclass
class VectorStore:
    text_chunks: List[str] = field(default_factory=list)
    vector_index: Optional[faiss.Index] = None

    def build_index(self, chunks: List[str]):
        self.text_chunks = chunks
        if not chunks:
            self.vector_index = None
            return
        
        logger.info(f"Building index for {len(chunks)} chunks...")
        embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
        embedding_dim = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(embedding_dim)
        self.vector_index.add(np.array(embeddings, dtype='float32'))

    def search(self, query: str, k: int = 3) -> List[str]:
        if self.vector_index is None or not self.text_chunks:
            return []
        
        query_embedding = embedding_model.encode([query])
        distances, indices = self.vector_index.search(np.array(query_embedding, dtype='float32'), k)
        return [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]

vector_store = VectorStore()

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Extracted {len(text)} characters from TXT")
        return text
    except Exception as e:
        logger.error(f"TXT extraction error: {e}")
        return ""

def split_text_into_chunks(text: str) -> List[str]:
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks

def extract_key_topics(text: str) -> List[str]:
    """Extract key topics from text."""
    try:
        words = text.lower().split()
        skip_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        technical_terms = []
        for word in words[:1000]:
            if len(word) > 4 and word not in skip_words and word.isalpha():
                technical_terms.append(word)
        
        from collections import Counter
        topic_counts = Counter(technical_terms)
        return [topic for topic, count in topic_counts.most_common(5)]
    except Exception as e:
        logger.error(f"Topic extraction error: {e}")
        return []

async def enhanced_summarization(full_text: str, filename: str) -> Dict[str, Any]:
    """Enhanced summarization with CrewAI."""
    try:
        key_topics = extract_key_topics(full_text)
        logger.info(f"Key topics: {key_topics}")
        
        # Run CrewAI summarization
        summary = await run_summarization_crew(full_text, filename, key_topics)
        
        # Get citations
        citations = get_citations_from_topics(key_topics)
        
        return {"summary": summary, "citations": citations}
        
    except Exception as e:
        logger.error(f"Enhanced summarization error: {e}")
        return {"summary": await basic_summarization(full_text[:5000]), "citations": []}

async def basic_summarization(full_text: str) -> str:
    """Basic fallback summarization."""
    try:
        if chat_model is None:
            return "# Error\nChat model not initialized."
        
        prompt = f"""
        Create a comprehensive summary of this document:
        
        {full_text}
        
        Structure as:
        # Document Summary
        ## Executive Summary
        ## Key Findings
        ## Methodology
        ## Technical Details
        ## Conclusions
        """
        
        response = await chat_model.ainvoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"# Error\nSummarization failed: {str(e)}"

# --- API Endpoints ---
@app.get("/health")
def health_check(): 
    return {"status": "ok", "message": "AI Document Summarizer API is running!"}

@app.post("/summarize_arxiv/", response_model=SummaryResponse)
async def summarize_document(file: UploadFile = File(...)):
    """Summarize uploaded document with citations."""
    temp_file_path = None
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Create temporary file
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Extract text
        if suffix == '.pdf':
            full_text = extract_text_from_pdf(temp_file_path)
        elif suffix in ['.txt', '.csv']:
            full_text = extract_text_from_txt(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from file.")

        # Split for chat
        text_chunks = split_text_into_chunks(full_text)
        vector_store.build_index(text_chunks)

        # Summarize with citations
        result = await enhanced_summarization(full_text, file.filename)

        return SummaryResponse(
            summary=result["summary"], 
            textforbot=text_chunks,
            citations=result["citations"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path): 
            os.remove(temp_file_path)

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """Chat with processed document."""
    try:
        if not vector_store.vector_index:
            raise HTTPException(status_code=400, detail="No document processed yet.")
        
        if chat_model is None:
            raise HTTPException(status_code=500, detail="Chat model not initialized.")
        
        # Get relevant chunks
        relevant_chunks = vector_store.search(request.question, k=3)
        document_context = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else "No relevant context found."
        
        # Get web context
        web_context = ""
        web_sources = []
        try:
            web_results = search_web_content(request.question, num_results=3)
            if web_results:
                web_context = "\n".join([
                    f"Source: {r['title']}\nURL: {r['url']}\nContent: {r['content'][:300]}" 
                    for r in web_results if r.get('title')
                ])
                
                for result in web_results[:3]:
                    if result.get('title') and result.get('url'):
                        web_sources.append({
                            "title": result["title"],
                            "url": result["url"],
                            "type": "web"
                        })
        except Exception as e:
            logger.warning(f"Web search error: {e}")
        
        # Create prompt
        system_prompt = """You are a helpful AI assistant. Answer based on the document context and web information. 
        Prioritize document content and cite sources when possible."""
        
        user_prompt = f"""Question: {request.question}

Document Context:
{document_context}

Web Context:
{web_context if web_context else "No web context available."}

Please provide a comprehensive answer."""

        # Get response
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = await chat_model.ainvoke(messages)
        
        answer = response.content if response and hasattr(response, 'content') else "Sorry, I couldn't generate a response."
        
        # Add document source
        sources = web_sources.copy()
        sources.append({"title": "Processed Document", "url": "#", "type": "document"})
        
        return ChatResponse(answer=answer, sources=sources)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Document Summarizer API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)