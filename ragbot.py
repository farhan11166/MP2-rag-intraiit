from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# CrewAI
class Config:
    """Configuration class for API keys and settings"""
    gemini_api_key: str = ""
    serper_api_key: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    max_search_results: int = 5
class VectorStore:
    """FAISS-based vector store for document retrieval"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.index = None
        self.texts = []
        self.embeddings = None

    def add_texts(self, texts: List[str]):
        """Add texts to the vector store"""
        self.texts = texts
        print(f"Generating embeddings for {len(texts)} text chunks...")

        # Generate embeddings
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        print(f"Vector store created with {len(texts)} documents")

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            return []

        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append({
                    'text': self.texts[idx],
                    'score': float(score),
                    'index': int(idx)
                })

        return results    