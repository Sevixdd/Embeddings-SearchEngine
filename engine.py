from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from vectordb import VectorDatabase, SearchResult
from utils.load_documents import load_documents_from_directory
from utils.compute_embeddings import compute_embeddings, get_embedding_model


class EmbeddingSearchEngine:
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db: VectorDatabase | None = None,
                 collection_name: str = "search_engine") -> None:
        self.model_name = model_name
        # Use object to avoid hard import-time dependency on sentence_transformers
        self.model: object | None = None
        # Use provided vector database or create a new one
        self.vector_db = vector_db or VectorDatabase(collection_name=collection_name)
        self.documents: List[str] = []
        self.paths: List[str] = []

    def _ensure_model(self):
        if self.model is None:
            self.model = get_embedding_model(self.model_name)
        return self.model

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        return compute_embeddings(texts, model=model, normalize=True, show_progress=False)

    def load_directory(self, directory: str, pattern: str = "*.txt") -> None:
        """Load documents from a directory using utility functions."""
        try:
            docs, paths = load_documents_from_directory(directory, pattern)
            self.build(docs, paths)
        except Exception as e:
            print(f"Warning: Failed to load documents from directory '{directory}': {e}")
            # Fallback to empty build
            self.build([], [])

    def build(self, documents: List[str], paths: List[str] | None = None) -> None:
        self.documents = documents
        self.paths = paths if paths is not None else [f"doc_{i}" for i in range(len(documents))]
        if not self.documents:
            return

        embeddings = self._encode(self.documents)
        dim = embeddings.shape[1]
        
        # Create collection if it doesn't exist or has wrong dimensions
        if not self.vector_db.collection_exists():
            self.vector_db.create_collection(dim)
        else:
            collection_info = self.vector_db.get_collection_info()
            if collection_info and collection_info["vector_size"] != dim:
                self.vector_db.delete_collection()
                self.vector_db.create_collection(dim)
        
        # Clear existing points and add new ones
        self.vector_db.clear_collection()
        self.vector_db.upsert_vectors(
            vectors=embeddings.tolist(),
            documents=self.documents,
            paths=self.paths
        )

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if not query:
            return []
        if len(self.documents) == 0:
            return []
        
        query_emb = self._encode([query])  # shape (1, dim)
        
        # Search using the vector database
        return self.vector_db.search_vectors(
            query_vector=query_emb[0].tolist(),
            limit=min(k, len(self.documents))
        )
    
    def get_stats(self) -> dict:
        """Get statistics about the current search engine."""
        return {
            "documents_count": len(self.documents),
            "model_name": self.model_name,
            "collection_stats": self.vector_db.get_collection_stats()
        }
    
    def clear_index(self) -> None:
        """Clear the current index."""
        self.vector_db.clear_collection()
        self.documents = []
        self.paths = []


def build_default_engine(data_dir: str = "data") -> EmbeddingSearchEngine:
    engine = EmbeddingSearchEngine()
    engine.load_directory(data_dir)
    return engine


