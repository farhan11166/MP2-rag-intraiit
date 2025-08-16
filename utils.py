"""
Utility functions for document processing and text handling
"""
import os
import logging
import tempfile
import fitz  # PyMuPDF
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT/CSV file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Extracted {len(text)} characters from text file")
        return text
    except Exception as e:
        logger.error(f"Error extracting text file: {e}")
        return ""

def split_text_into_chunks(text: str) -> List[str]:
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def validate_file(file):
    """Validate uploaded file"""
    if file is None:
        return False, "Please upload a file"
    
    # Check file size (50MB max)
    if file.size > 50 * 1024 * 1024:
        return False, "File too large (max 50MB)"
    
    # Check file type
    allowed_types = ['application/pdf', 'text/plain', 'text/csv']
    if file.content_type not in allowed_types:  # ✅ Correct for FastAPI
        return False, f"Unsupported file type: {file.content_type}"
    
    return True, "File is valid"

def save_uploaded_file(file) -> str:
    """Save uploaded file to temporary location"""
    suffix = os.path.splitext(file.filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file.read())
        return temp_file.name

class SimpleVectorStore:
    """Simple vector store for document chunks"""
    
    def __init__(self):
        self.chunks = []
        self.index = None
    
    def add_chunks(self, chunks: List[str]):
        """Add text chunks to vector store"""
        try:
            self.chunks = chunks
            if not chunks:
                logger.warning("No chunks to add to vector store")
                return
                
            # Generate embeddings
            embeddings = embedding_model.encode(chunks)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings, dtype='float32'))
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant chunks"""
        try:
            if not self.index or not self.chunks:
                return []
            
            # Generate query embedding
            query_embedding = embedding_model.encode([query])
            
            # Search
            distances, indices = self.index.search(
                np.array(query_embedding, dtype='float32'), k
            )
            
            # Return relevant chunks
            relevant_chunks = []
            for idx in indices[0]:
                if idx < len(self.chunks):
                    relevant_chunks.append(self.chunks[idx])
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

# Global vector store instance
vector_store = SimpleVectorStore()