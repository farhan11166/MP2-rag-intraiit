from crewai import LLM
import os
import logging
from dotenv import load_dotenv
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv("a.env")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable in your 'a.env' file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize LLMs for CrewAI and general chat
try:
    crew_llm = LLM(
        api_key=api_key,
        model="gemini/gemini-1.5-flash"
    )
    
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
