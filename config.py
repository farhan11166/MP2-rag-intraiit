
import os
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import LLM
from exa_py import Exa

# Load environment variables
load_dotenv("a.env")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable in your 'a.env' file.")
if not EXA_API_KEY:
    raise ValueError("Please set the EXA_API_KEY environment variable in your 'a.env' file.")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize models
try:
    # For CrewAI agents
    crew_llm = LLM(api_key=GOOGLE_API_KEY, model="gemini/gemini-1.5-flash")
    
    # For chat functionality
    chat_model = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model="gemini-1.5-flash",
        temperature=0.2,
        max_output_tokens=8192
    )
    
    # For embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # For web search
    exa_client = Exa(api_key=EXA_API_KEY)
    
    logger.info("All models initialized successfully.")
    
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    crew_llm = None
    chat_model = None
    embedding_model = None
    exa_client = None