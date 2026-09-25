"""
Module 3: Vector Store and Retrieval
Loads the 8 Zepto policy documents, chunks them, computes embeddings (supporting all-MiniLM-L6-v2
with offline Scikit-Learn Tfidf/cosine fallback), and performs top-3 cosine similarity retrieval.
"""

import os
import glob
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

class PolicyVectorStore:
    def __init__(self, docs_dir: str = DOCS_DIR):
        self.docs_dir = docs_dir
        self.chunks = []
        self.chunk_ids = []
        self.vectorizer = None
        self.embeddings = None
        self._st_model = None
        self.load_and_index()

    def _init_model(self):
        """Attempts to load sentence-transformers all-MiniLM-L6-v2 if installed."""
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Loaded SentenceTransformer ('all-MiniLM-L6-v2') successfully.")
        except Exception as e:
            # Fully supported offline fallback
            self._st_model = None

    def load_and_index(self):
        """Loads all doc_01.txt to doc_08.txt and indexes chunks."""
        doc_files = sorted(glob.glob(os.path.join(self.docs_dir, "doc_*.txt")))
        if not doc_files:
            # Fallback if path relative
            doc_files = sorted(glob.glob("support_assistant/docs/doc_*.txt"))
            
        self.chunks = []
        self.chunk_ids = []
        
        for file_path in doc_files:
            doc_name = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
            # Document chunking: split by paragraphs or sections
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for p_idx, para in enumerate(paragraphs):
                chunk_id = f"{doc_name}_chunk_{p_idx}"
                self.chunks.append({
                    "id": chunk_id,
                    "doc_name": doc_name,
                    "text": para
                })
                self.chunk_ids.append(chunk_id)

        corpus_texts = [c["text"] for c in self.chunks]
        
        # Initialize embedding model / vectorizer
        self._init_model()
        if self._st_model:
            self.embeddings = self._st_model.encode(corpus_texts, convert_to_numpy=True)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-12)
        else:
            # Pure offline TF-IDF vectorizer producing unit-norm vectors
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.embeddings = self.vectorizer.fit_transform(corpus_texts).toarray()

        print(f"Indexed {len(self.chunks)} policy chunks from {len(doc_files)} policy documents.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most similar chunks using cosine similarity.
        """
        if not self.chunks:
            return []

        if self._st_model:
            query_vec = self._st_model.encode([query], convert_to_numpy=True)
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
            scores = np.dot(self.embeddings, query_vec.T).flatten()
        else:
            query_vec = self.vectorizer.transform([query]).toarray()
            scores = cosine_similarity(query_vec, self.embeddings).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "id": self.chunks[idx]["id"],
                "doc_name": self.chunks[idx]["doc_name"],
                "text": self.chunks[idx]["text"],
                "score": float(scores[idx])
            })
            
        return results

# Singleton instance
_vector_store_instance = None

def get_vector_store() -> PolicyVectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = PolicyVectorStore()
    return _vector_store_instance
