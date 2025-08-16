
import os
import json
import logging
import asyncio
import tempfile
import traceback

import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Machine learning & NLP
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from models import ChatRequest, SummaryResponse, ChatResponse, ErrorResponse
from agents import enhanced_summarization_with_citations,search_web_content
from crewai import LLM

from vector_store import VectorStore
from agents import enhanced_summarization_with_citations
from llm_setup import chat_model
from helpers import (
    extract_text_from_pdf,
    extract_text_from_txt,
    split_text_into_chunks
)


load_dotenv("a.env")
api_key = os.getenv("GOOGLE_API_KEY")
exa_api_key = os.getenv("EXA_API_KEY")

if not api_key:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable in your 'a.env' file.")
if not exa_api_key:
    raise ValueError("Please set the EXA_API_KEY environment variable in your 'a.env' file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Document Summarizer with Citations",
    description="An API for summarizing documents with internet citations using CrewAI agents and Exa search.",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Exa client
vector_store = VectorStore()

# --- Language Model Initialization (Fixed) ---
try:
    # For CrewAI agents - use CrewAI LLM
    crew_llm = LLM(
        api_key=api_key,
        model="gemini/gemini-1.5-flash"  # Correct model name for CrewAI
    )
    
    # For chat functionality - use LangChain ChatGoogleGenerativeAI
    chat_model = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-1.5-flash",
        temperature=0.2,
        max_output_tokens=8192
    )
    
    logger.info("Both Gemini models initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini models: {e}")
    crew_llm = None
    chat_model = None

# --- Enhanced Exa Search Tool ---
# --- Helper Functions ---



# --- Enhanced Summarization with Citations ---


# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check(): 
    return {"status": "ok", "message": "AI Document Summarizer API is running!"}

@app.post("/summarize/", response_model=SummaryResponse, responses={500: {"model": ErrorResponse}}, summary="Summarize Document with Citations")
async def summarize_and_index_document(file: UploadFile = File(...)):
    """Summarize uploaded document with web citations"""
    
    temp_file_path = None
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Create temporary file
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Extract text based on file type
        if suffix == '.pdf':
            full_text = extract_text_from_pdf(temp_file_path)
        elif suffix in ['.txt', '.csv']:
            full_text = extract_text_from_txt(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
        
        if not full_text or not full_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file.")

        # Split text for chat functionality
        text_chunks_for_bot = split_text_into_chunks(full_text)
        vector_store.build_index(text_chunks_for_bot)

        # Enhanced summarization with citations
        logger.info("Starting enhanced summarization process...")
        result = await enhanced_summarization_with_citations(full_text, file.filename)

        logger.info(f"Summarization completed. Citations found: {len(result['citations'])}")

        return SummaryResponse(
            summary=result["summary"], 
            textforbot=text_chunks_for_bot,
            citations=result["citations"]
        )

    except HTTPException as http_exc:
        logger.error(f"HTTP exception in summarize_arxiv: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in summarize_arxiv: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path): 
            try:
                os.remove(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {e}")

@app.post("/chat/", response_model=ChatResponse, responses={500: {"model": ErrorResponse}}, summary="Chat with Document")
async def chat_with_document(request: ChatRequest):
    """Chat with the processed document using enhanced context - FIXED"""
    
    try:
        # Check if vector store is initialized
        if not vector_store.vector_index:
            logger.warning("No document processed for chat")
            raise HTTPException(status_code=400, detail="No document has been processed yet. Please upload a document first.")
        
        # Check if chat model is initialized
        if chat_model is None:
            logger.error("Chat model not initialized")
            raise HTTPException(status_code=500, detail="Chat model not properly initialized.")
        
        logger.info(f"Processing chat query: {request.question}")
        
        # Get relevant document chunks
        relevant_chunks = vector_store.search(request.question, k=3)
        if not relevant_chunks:
            logger.warning("No relevant chunks found for query")
            document_context = "No relevant context found in the document."
        else:
            document_context = "\n\n---\n\n".join(relevant_chunks)
        
        # Search for additional web context (with error handling)
        web_context = ""
        web_sources = []
        try:
            web_results = search_web_content.run(query=request.question, num_results=3)
            if web_results:
                web_context = "\n".join([
                    f"Source: {r['title']}\nURL: {r['url']}\nContent: {r['content'][:300]}" 
                    for r in web_results if r.get('title') and r.get('content')
                ])
                
                # Prepare sources for response
                for result in web_results[:3]:
                    if result.get('title') and result.get('url'):
                        web_sources.append({
                            "title": result["title"],
                            "url": result["url"],
                            "type": "web"
                        })
        except Exception as web_error:
            logger.warning(f"Error in web search for chat: {web_error}")
            web_context = "Web search unavailable."
        
        # Create comprehensive prompt
        system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided 
        document context and supplementary web information. 
        
        Guidelines:
        - Prioritize information from the document context
        - Use web sources to provide additional context and validation
        - Always cite your sources when possible
        - If the information is not available in either context, clearly state this
        - Provide detailed and informative answers
        - Be concise but comprehensive"""
        
        user_prompt = f"""Question: {request.question}

Document Context:
{document_context}

Additional Web Context:
{web_context if web_context else "No additional web context available."}

Please provide a comprehensive answer based on the above contexts."""

        # Make the API call with proper error handling
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = await chat_model.ainvoke(messages)
            
            if not response or not hasattr(response, 'content'):
                raise Exception("Invalid response from chat model")
            
            answer = response.content
            if not answer or not answer.strip():
                answer = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as model_error:
            logger.error(f"Error calling chat model: {model_error}")
            answer = f"I encountered an error while processing your question: {str(model_error)}"
        
        # Add document source
        sources = web_sources.copy()
        sources.append({
            "title": "Processed Document",
            "url": "#",
            "type": "document"
        })
        
        logger.info(f"Chat response generated with {len(sources)} sources")
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )

    except HTTPException as http_exc:
        logger.error(f"HTTP exception in chat: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Document Summarizer API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)