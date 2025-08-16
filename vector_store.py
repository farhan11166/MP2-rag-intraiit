from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

@dataclass
class VectorStore:
    text_chunks: List[str] = field(default_factory=list)
    vector_index: Optional[faiss.Index] = None
    embedding_model: SentenceTransformer = None

    def __post_init__(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
        return [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]
