
import os
import json
import logging
from typing import List, Dict, Any, Optional
import asyncio
import tempfile
import traceback

import fitz  # PyMuPDF
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Machine learning & NLP
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from vector_store import VectorStore
from exa_py import Exa



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
exa_client = Exa(api_key=exa_api_key)
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
@tool("exa_search")
def search_web_content(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web for relevant content using Exa API.
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return
    
    Returns:
        List[Dict]: List of search results with title, URL, and content
    """
    try:
        logger.info(f"Searching web for: {query}")
        search_response = exa_client.search_and_contents(
            query=query,
            num_results=num_results,
            text=True,
            highlights=True,
            summary=True
        )
        
        results = []
        for result in search_response.results:
            results.append({
                "title": result.title or "Untitled",
                "url": result.url or "",
                "content": result.text[:800] if result.text else "",
                "summary": getattr(result, 'summary', '') or "",
                "highlights": getattr(result, 'highlights', []) or []
            })
        
        logger.info(f"Found {len(results)} web results")
        return results
    
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return []

# --- CrewAI Agents (Fixed) ---
def create_agents():
    """Create CrewAI agents with error handling"""
    try:
        if crew_llm is None:
            logger.error("CrewAI LLM not initialized")
            return None, None, None
            
        # Document Analysis Agent
        document_analyzer = Agent(
            role="Document Analyst",
            goal="Analyze and extract key information from documents for comprehensive summarization",
            backstory="""You are an expert document analyst with years of experience in technical 
            document review. You excel at identifying key concepts, methodologies, and findings 
            from complex documents.""",
            verbose=False,
            allow_delegation=False,
            llm=crew_llm
        )

        # Web Research Agent
        web_researcher = Agent(
            role="Web Research Specialist",
            goal="Find relevant online sources and citations that complement document analysis",
            backstory="""You are a skilled researcher who knows how to find credible online sources 
            that provide additional context and validation for technical documents. You excel at 
            identifying authoritative sources and relevant citations.""",
            verbose=False,
            allow_delegation=False,
            tools=[search_web_content],
            llm=crew_llm
        )

        # Summary Synthesizer Agent
        summary_synthesizer = Agent(
            role="Technical Writer and Synthesizer",
            goal="Create comprehensive summaries that integrate document content with web citations",
            backstory="""You are an expert technical writer who specializes in creating detailed, 
            well-structured summaries that combine multiple sources of information. You excel at 
            presenting complex information in a clear, organized manner with proper citations.""",
            verbose=False,
            allow_delegation=False,
            llm=crew_llm
        )
        
        return document_analyzer, web_researcher, summary_synthesizer
    except Exception as e:
        logger.error(f"Error creating agents: {e}")
        return None, None, None

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

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF with improved error handling"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text("text")
                text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num}: {e}")
                continue
        doc.close()
        logger.info(f"Extracted {len(text)} characters from PDF: {os.path.basename(pdf_path)}")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Extracted {len(text)} characters from TXT file")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from TXT file: {e}")
        return ""

def split_text_into_chunks(text: str) -> List[str]:
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""], 
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks for chat context.")
    return chunks

def extract_key_topics(text: str) -> List[str]:
    """Extract key topics from document text for web search"""
    try:
        words = text.lower().split()
        # Filter out common words and find technical terms
        technical_terms = []
        skip_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "been", "be", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "must"}
        
        for word in words[:1000]:  # Look at first 1000 words
            if len(word) > 4 and word not in skip_words and word.isalpha():
                technical_terms.append(word)
        
        # Return most frequent terms as topics
        from collections import Counter
        topic_counts = Counter(technical_terms)
        return [topic for topic, count in topic_counts.most_common(5)]
    except Exception as e:
        logger.error(f"Error extracting key topics: {e}")
        return []

# --- Enhanced Summarization with Citations ---
async def enhanced_summarization_with_citations(full_text: str, filename: str) -> Dict[str, Any]:
    """Enhanced summarization using CrewAI agents with web citations"""
    try:
        logger.info("Starting enhanced summarization with citations...")
        
        # Create agents
        document_analyzer, web_researcher, summary_synthesizer = create_agents()
        
        if not all([document_analyzer, web_researcher, summary_synthesizer]):
            logger.error("Failed to create agents, falling back to basic summarization")
            return {
                "summary": await basic_summarization(full_text[:5000]),
                "citations": []
            }
        
        # Extract key topics for web research
        key_topics = extract_key_topics(full_text)
        logger.info(f"Extracted key topics: {key_topics}")
        
        # Task 1: Document Analysis
        document_analysis_task = Task(
            description=f"""
            Analyze the following document and extract key information:
            
            Document: {filename}
            Content (first 3000 chars): {full_text[:3000]}
            
            Focus on:
            1. Main objectives and purpose
            2. Key findings and results
            3. Methodologies and approaches used
            4. Important concepts and terminology
            5. Conclusions and implications
            
            Provide a comprehensive analysis of the document's content and structure.
            """,
            agent=document_analyzer,
            expected_output="Detailed analysis of document content, methodology, and key findings"
        )
        
        # Task 2: Web Research
        web_research_task = Task(
            description=f"""
            Research relevant information online for the following topics derived from the document:
            
            Topics: {', '.join(key_topics)}
            
            For each topic, find:
            1. Authoritative sources and recent research
            2. Related methodologies and best practices
            3. Current developments in the field
            4. Supporting evidence and validation
            
            Focus on finding credible, academic, and industry sources that complement the document.
            """,
            agent=web_researcher,
            expected_output="List of relevant web sources with titles, URLs, and descriptions"
        )
        
        # Task 3: Summary Synthesis
        synthesis_task = Task(
            description=f"""
            Create a comprehensive summary that integrates the document analysis with web research findings.
            
            Requirements:
            1. Start with the document analysis as the primary content
            2. Integrate relevant web citations to support and expand on key points
            3. Structure the summary using clear Markdown formatting
            4. Include proper citations and references
            5. Make it detailed, informative, and well-organized
            
            Structure the summary as follows:
            # Document Summary: {filename}
            
            ## Executive Summary
            [Brief overview of the document's main points]
            
            ## Key Findings and Results
            [Main discoveries and outcomes]
            
            ## Methodology and Approach
            [Methods and techniques used]
            
            ## Technical Details and Implementation
            [Technical aspects and details]
            
            ## Related Research and Context
            [How this relates to current research, supported by web citations]
            
            ## Conclusions and Implications
            [Final thoughts and implications]
            
            ## References and Citations
            [List of web sources and citations]
            """,
            agent=summary_synthesizer,
            expected_output="Comprehensive markdown summary with integrated web citations",
            context=[document_analysis_task, web_research_task]
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[document_analyzer, web_researcher, summary_synthesizer],
            tasks=[document_analysis_task, web_research_task, synthesis_task],
            verbose=True
        )
        
        logger.info("Running CrewAI agents for enhanced summarization...")
        result = crew.kickoff()
        
        # Extract citations from web research results
        citations = []
        try:
            # Get web results from the search tool calls
            web_results = search_web_content(" ".join(key_topics[:2]), num_results=5)
            for idx, result in enumerate(web_results, 1):
                if result.get('title') and result.get('url'):
                    citations.append({
                        "title": result['title'],
                        "url": result['url'],
                        "description": result.get('summary', result.get('content', '')[:200] + "...")
                    })
        except Exception as e:
            logger.warning(f"Could not extract citations: {e}")
        
        logger.info(f"Enhanced summarization completed with {len(citations)} citations")
        
        return {
            "summary": str(result),
            "citations": citations
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced summarization: {e}")
        logger.error(traceback.format_exc())
        # Fallback to basic summarization
        return {
            "summary": await basic_summarization(full_text[:5000]),
            "citations": []
        }

async def basic_summarization(full_text: str) -> str:
    """Fallback basic summarization using LangChain model"""
    try:
        logger.info("Using basic summarization fallback")
        
        if chat_model is None:
            return "# Error in Summary Generation\n\nChat model not initialized properly."
        
        prompt = f"""
        Create a comprehensive summary of the following document. Structure your response with clear sections and provide detailed analysis:

        Document Content:
        {full_text}

        Please structure your summary as follows:
        # Document Summary

        ## Executive Summary
        [Provide a brief overview of the main points]

        ## Key Findings
        [Detail the main discoveries and results]

        ## Methodology
        [Describe the approaches and methods used]

        ## Technical Details
        [Include important technical information]

        ## Conclusions
        [Summarize the main conclusions and implications]

        Make the summary detailed, informative, and well-structured using Markdown formatting.
        """
        
        message = HumanMessage(content=prompt)
        response = await chat_model.ainvoke([message])
        return response.content
    except Exception as e:
        logger.error(f"Error in basic summarization: {e}")
        return f"# Error in Summary Generation\n\nAn error occurred while generating the summary: {str(e)}"

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
            web_results = search_web_content(request.question, num_results=3)
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