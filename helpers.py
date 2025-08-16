import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import Counter
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
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
