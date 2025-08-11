import os
import logging
import requests
import fitz
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv("a.env")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Initialize the Gemini chat model
chat = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.5-flash",  # Fast & high-token model
    max_output_tokens=10000
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

class URLRequest(BaseModel):
    url: str

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI backend is running!"}

@app.post("/summarize_arxiv/")
async def summarize_arxiv(file: UploadFile = File(...)):
    """Download PDF, extract text, summarize with Gemini."""
    try:
        temp_pdf_path = f"/tmp/{file.filename}"
        with open(temp_pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract text
        text = extract_text_from_pdf(temp_pdf_path)
        if not text:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        summary = await summarize_text_parallel(text)
        return {"summary": summary}    



        logger.info(f"Extracted {len(text)} characters of text")

       

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": "Failed to process PDF"}




def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text("text") for page in doc])
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return ""

async def summarize_chunk_with_retry(chunk, chunk_id, total_chunks, max_retries=2):
    """Retry wrapper for chunk summaries."""
    retries = 0
    while retries <= max_retries:
        try:
            if retries > 0:
                logger.info(f"Retry {retries}/{max_retries} for chunk {chunk_id}")

            result = await summarize_chunk_wrapper(chunk, chunk_id, total_chunks)

            if isinstance(result, str) and result.startswith("Error"):
                retries += 1
                await asyncio.sleep(5 * (2 ** (retries - 1)))
            else:
                return result

        except Exception as e:
            retries += 1
            logger.error(f"Exception in chunk {chunk_id}: {e}")
            await asyncio.sleep(5 * (2 ** (retries - 1)))

    return f"Error: Failed after {max_retries+1} attempts"

async def summarize_text_parallel(text):
    """Split into chunks & summarize in parallel."""
    chunk_size = 1000 * 4  # ~32K tokens
    chunk_overlap = 100
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split into {len(chunks)} chunks")

    tasks = [summarize_chunk_with_retry(chunk, i+1, len(chunks)) for i, chunk in enumerate(chunks)]
    summaries = await asyncio.gather(*tasks, return_exceptions=True)

    summaries = [s if isinstance(s, str) else f"Error: {s}" for s in summaries]

    combined_text = "\n\n".join(f"Section {i+1}:\n{s}" for i, s in enumerate(summaries))

    return await generate_final_summary(combined_text)

async def summarize_chunk_wrapper(chunk, chunk_id, total_chunks):
    """Summarize one chunk using Gemini."""
    logger.info(f"Processing chunk {chunk_id}/{total_chunks}")
    try:
        message = HumanMessage(content=f"Extract only technical details from this text:\n{chunk}")
        response = await chat.ainvoke([message])
        return response.content
    except Exception as e:
        logger.error(f"Error chunk {chunk_id}: {e}")
        return f"Error processing chunk {chunk_id}: {e}"

async def generate_final_summary(combined_chunks):
    """Generate a final structured summary using Gemini."""
    prompt = f"""
    You are an expert technical and academic writer with deep knowledge across diverse domains, including software engineering, scientific research, data analysis, engineering systems, and experimental studies.

    Your task:
    Create a **comprehensive, well-structured technical document** focusing **only** on:
    1. System Architecture  
    2. Technical Implementation  
    3. Infrastructure & Setup  
    4. Performance Analysis  
    5. Optimization Techniques  

    Instructions:
    - Use ONLY the provided content below as the source of information:  
      {combined_chunks}
    - Adapt each section appropriately for the given domain, even if it is not directly related to computing or system design.
        - For example, in non-technical fields:
            - "System Architecture" can describe experimental design, workflows, or conceptual models.
            - "Technical Implementation" can explain how the methodology or process was executed.
            - "Infrastructure & Setup" can describe tools, environments, datasets, or experimental setups.
            - "Performance Analysis" can include results, metrics, evaluation criteria, or observations.
            - "Optimization Techniques" can cover improvements, refinements, troubleshooting, or alternative methods.
    - Ignore irrelevant theory, background, and unrelated narrative unless needed for clarity.
    - Maintain precision, factual correctness, and a professional tone.
    - Present information in a way that is **logical, readable, and actionable**.
    - Do not fabricate information — use only what is in the provided content.
    """
    try:
        message = HumanMessage(content=prompt)
        response = await chat.ainvoke([message])
        return response.content
    except Exception as e:
        logger.error(f"Final summary error: {e}")
        return f"Error generating final summary:"


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
