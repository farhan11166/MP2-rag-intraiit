# Standard library imports
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Third-party libraries
import numpy as np
import fitz  # PyMuPDF
import faiss
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException

# Machine learning & NLP
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration & Initialization ---

load_dotenv("a.env")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable in your 'a.env' file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Document Summarizer Backend",
    description="An API for summarizing documents and answering questions using Gemini and RAG.",
    version="1.3.0" # Version updated for longer summaries
)

# --- Constants ---
# Using character count as a proxy for tokens (1 token ~ 4 chars)
# Gemini 1.5 Flash has a 1M token context window. We'll use a safe limit.
# 800,000 tokens * 4 chars/token = 3,200,000 characters per section
MAX_CHARS_PER_SECTION = 3200000

# --- In-Memory Vector Store ---
@dataclass
class VectorStore:
    text_chunks: List[str] = field(default_factory=list)
    vector_index: Optional[faiss.Index] = None
    embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    def build_index(self, chunks: List[str]):
        self.text_chunks = chunks
        if not self.text_chunks:
            logger.warning("No text chunks to index.")
            self.vector_index = None
            return
        logger.info(f"Generating embeddings for {len(self.text_chunks)} chunks...")
        embeddings = self.embedding_model.encode(self.text_chunks, convert_to_tensor=False)
        embedding_dim = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(embedding_dim)
        self.vector_index.add(np.array(embeddings, dtype='float32'))
        logger.info("FAISS index built successfully.")

    def search(self, query: str, k: int = 3) -> List[str]:
        if self.vector_index is None or not self.text_chunks:
            return []
        logger.info(f"Searching for top {k} chunks for query: '{query[:50]}...'")
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.vector_index.search(np.array(query_embedding, dtype='float32'), k)
        return [self.text_chunks[i] for i in indices[0]]

vector_store = VectorStore()

# --- Language Model Initialization ---
try:
    chat_model = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-1.5-flash",
        temperature=0.2,
        max_output_tokens=16384 # Increased token limit for longer summaries
    )
    logger.info("Gemini 1.5 Flash model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")
    chat_model = None

# --- Pydantic Models ---
class ChatRequest(BaseModel): question: str
class SummaryResponse(BaseModel): summary: str; textforbot: List[str]
class ChatResponse(BaseModel): answer: str
class ErrorResponse(BaseModel): error: str

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        logger.info(f"Extracted {len(text)} characters from PDF: {os.path.basename(pdf_path)}")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return ""

def split_text_into_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""], length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks for chat context.")
    return chunks

# --- New Map-Reduce Summarization Logic ---

async def summarize_section(section_text: str, section_num: int, total_sections: int) -> str:
    """(MAP STEP) Summarizes one large section of the document."""
    logger.info(f"Summarizing section {section_num}/{total_sections}...")
    # UPDATED PROMPT: Asks for a detailed summary, not a concise one.
    prompt = f"""
    You are a technical analyst. Your task is to extract and provide a detailed and comprehensive summary of the key information from the following section of a larger document. 
    Focus on technical details, methodologies, results, and conclusions. Elaborate on the main points.

    Document Section:
    ---
    {section_text}
    ---
    """
    try:
        message = HumanMessage(content=prompt)
        response = await chat_model.ainvoke([message])
        return response.content
    except Exception as e:
        logger.error(f"Error summarizing section {section_num}: {e}")
        return f"Error processing section {section_num}."

async def generate_final_summary(intermediate_summaries: str) -> str:
    """(REDUCE STEP) Creates a final, structured, and DETAILED summary from intermediate summaries."""
    logger.info("Generating final cohesive and detailed summary...")
    # UPDATED PROMPT: Explicitly asks for a long, detailed, multi-paragraph summary.
    prompt = f"""
    As an expert technical writer, create a single, in-depth, and comprehensive summary based on the following collection of section summaries. 
    Synthesize the information into a cohesive document. Your final output should be substantially longer and more detailed than a simple overview.

    Focus on these key areas, writing detailed, multi-paragraph explanations for each:
    1.  **System Architecture & Design:** Describe the overall structure, components, data flow, and design principles in detail.
    2.  **Technical Implementation:** Thoroughly explain the core algorithms, methodologies, frameworks, and technologies used.
    3.  **Infrastructure & Setup:** Provide a comprehensive overview of the environment, datasets, tools, and configurations involved.
    4.  **Performance Analysis & Results:** Summarize the key findings, metrics, and evaluations with as much detail as provided.
    5.  **Optimization & Future Work:** Elaborate on any improvements, optimizations, or proposed next steps.

    Instructions:
    - Use Markdown for clear formatting (e.g., # Title, ## Section, - Bullet).
    - Ensure a logical flow. Do not just list the section summaries; integrate them smoothly.
    - Be precise and factually correct based ONLY on the provided content.
    - **Crucially, aim for a substantial and thorough output, at least three times longer than a brief overview. Elaborate on every point.**

    Provided Section Summaries:
    ---
    {intermediate_summaries}
    ---
    """
    try:
        message = HumanMessage(content=prompt)
        response = await chat_model.ainvoke([message])
        logger.info("Final detailed summary generation complete.")
        return response.content
    except Exception as e:
        logger.error(f"Error during final summary generation: {e}")
        return "Error: Could not generate the final summary."

async def map_reduce_summarizer(full_text: str) -> str:
    """Orchestrates the map-reduce summarization process."""
    if len(full_text) <= MAX_CHARS_PER_SECTION:
        logger.info("Document is small enough for a single summary call.")
        return await generate_final_summary(full_text)

    sections = [full_text[i:i+MAX_CHARS_PER_SECTION] for i in range(0, len(full_text), MAX_CHARS_PER_SECTION)]
    logger.info(f"Document split into {len(sections)} large sections for map-reduce.")

    tasks = [summarize_section(sec, i+1, len(sections)) for i, sec in enumerate(sections)]
    intermediate_summaries = await asyncio.gather(*tasks)

    combined_summaries = "\n\n---\n\n".join(intermediate_summaries)

    final_summary = await generate_final_summary(combined_summaries)
    return final_summary

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check(): return {"status": "ok", "message": "FastAPI backend is running!"}

@app.post("/summarize_arxiv/", response_model=SummaryResponse, responses={500: {"model": ErrorResponse}}, summary="Summarize a Document")
async def summarize_and_index_document(file: UploadFile = File(...)):
    temp_pdf_path = f"/tmp/{file.filename}"
    try:
        content = await file.read()
        with open(temp_pdf_path, "wb") as f: f.write(content)

        full_text = extract_text_from_pdf(temp_pdf_path)
        if not full_text or not full_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")

        text_chunks_for_bot = split_text_into_chunks(full_text)
        vector_store.build_index(text_chunks_for_bot)

        summary = await map_reduce_summarizer(full_text)

        return {"summary": summary, "textforbot": text_chunks_for_bot}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in /summarize_arxiv/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)


@app.post("/chat/", response_model=ChatResponse, responses={500: {"model": ErrorResponse}}, summary="Chat with the Document")
async def chat_with_document(request: ChatRequest):
    if not vector_store.vector_index:
        raise HTTPException(status_code=400, detail="No document has been processed yet. Please upload a document first.")
    try:
        relevant_chunks = vector_store.search(request.question, k=3)
        context = "\n\n---\n\n".join(relevant_chunks)
        
        system_prompt = "You are a helpful AI assistant. Answer the user's question based *only* on the provided context. If the answer is not in the context, say 'I cannot answer that based on the provided document.'"
        user_prompt = f"Context from the document:\n---\n{context}\n---\nQuestion: {request.question}"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = await chat_model.ainvoke(messages)
        return {"answer": response.content}

    except Exception as e:
        logger.error(f"An unexpected error occurred in /chat/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
