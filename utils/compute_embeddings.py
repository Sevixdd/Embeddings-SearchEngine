"""
Embedding computation utilities for the semantic search engine.

This module provides functions for computing text embeddings using
various models and handling embedding-related operations.
"""

import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any
from sentence_transformers import SentenceTransformer


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder: Optional[str] = None
) -> SentenceTransformer:
    """
    Get or create an embedding model instance.
    
    Args:
        model_name: Name of the SentenceTransformer model to use
        cache_folder: Optional cache folder for the model
        
    Returns:
        SentenceTransformer model instance
        
    Raises:
        Exception: If model loading fails
    """
    try:
        model = SentenceTransformer(model_name, cache_folder=cache_folder)
        return model
    except Exception as e:
        raise Exception(f"Failed to load embedding model '{model_name}': {e}")


def compute_embeddings(
    texts: List[str],
    model: Optional[SentenceTransformer] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    show_progress: bool = False,
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model: Pre-loaded SentenceTransformer model (if None, will load model_name)
        model_name: Name of the model to use if model is None
        normalize: Whether to normalize embeddings to unit length
        show_progress: Whether to show progress bar during computation
        batch_size: Batch size for processing texts
        
    Returns:
        Numpy array of shape (n_texts, embedding_dim) containing the embeddings
        
    Raises:
        ValueError: If texts list is empty
        Exception: If embedding computation fails
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    # Load model if not provided
    if model is None:
        model = get_embedding_model(model_name)
    
    try:
        # Compute embeddings
        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=batch_size
        )
        
        # Ensure embeddings are float32 for better performance
        embeddings = embeddings.astype(np.float32)
        
        return embeddings
        
    except Exception as e:
        raise Exception(f"Failed to compute embeddings: {e}")


def compute_single_embedding(
    text: str,
    model: Optional[SentenceTransformer] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True
) -> np.ndarray:
    """
    Compute embedding for a single text.
    
    Args:
        text: Text string to embed
        model: Pre-loaded SentenceTransformer model (if None, will load model_name)
        model_name: Name of the model to use if model is None
        normalize: Whether to normalize embedding to unit length
        
    Returns:
        Numpy array of shape (embedding_dim,) containing the embedding
        
    Raises:
        ValueError: If text is empty
        Exception: If embedding computation fails
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    return compute_embeddings(
        [text], 
        model=model, 
        model_name=model_name, 
        normalize=normalize
    )[0]


def get_embedding_dimension(
    model: Optional[SentenceTransformer] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> int:
    """
    Get the dimension of embeddings produced by a model.
    
    Args:
        model: Pre-loaded SentenceTransformer model (if None, will load model_name)
        model_name: Name of the model to use if model is None
        
    Returns:
        Embedding dimension as integer
        
    Raises:
        Exception: If model loading or dimension detection fails
    """
    if model is None:
        model = get_embedding_model(model_name)
    
    try:
        # Compute embedding for a dummy text to get dimension
        dummy_embedding = compute_single_embedding("dummy text", model=model)
        return len(dummy_embedding)
    except Exception as e:
        raise Exception(f"Failed to get embedding dimension: {e}")


def compute_similarity_matrix(
    embeddings: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute pairwise similarity matrix between embeddings.
    
    Args:
        embeddings: Array of shape (n_embeddings, embedding_dim)
        normalize: Whether to normalize embeddings before computing similarity
        
    Returns:
        Similarity matrix of shape (n_embeddings, n_embeddings)
    """
    if normalize:
        # Normalize embeddings to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    return similarity_matrix


def find_most_similar_pairs(
    embeddings: np.ndarray,
    top_k: int = 5,
    exclude_self: bool = True
) -> List[Tuple[int, int, float]]:
    """
    Find the most similar pairs of embeddings.
    
    Args:
        embeddings: Array of shape (n_embeddings, embedding_dim)
        top_k: Number of top similar pairs to return
        exclude_self: Whether to exclude self-similarity (diagonal elements)
        
    Returns:
        List of tuples (index1, index2, similarity_score) sorted by similarity
    """
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    if exclude_self:
        # Set diagonal to -1 to exclude self-similarity
        np.fill_diagonal(similarity_matrix, -1)
    
    # Get indices of top similarities
    flat_indices = np.argpartition(similarity_matrix.flatten(), -top_k)[-top_k:]
    row_indices, col_indices = np.unravel_index(flat_indices, similarity_matrix.shape)
    
    # Create list of (index1, index2, similarity) tuples
    similar_pairs = []
    for i, j in zip(row_indices, col_indices):
        if i != j or not exclude_self:  # Avoid duplicates when excluding self
            similar_pairs.append((i, j, similarity_matrix[i, j]))
    
    # Sort by similarity score (descending)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return similar_pairs[:top_k]


def get_embedding_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about an embedding model.
    
    Args:
        model_name: Name of the SentenceTransformer model
        
    Returns:
        Dictionary containing model information
    """
    try:
        model = get_embedding_model(model_name)
        
        # Get model dimension
        dimension = get_embedding_dimension(model=model)
        
        # Get model info
        model_info = {
            "name": model_name,
            "dimension": dimension,
            "max_seq_length": getattr(model, 'max_seq_length', 'Unknown'),
            "device": str(model.device) if hasattr(model, 'device') else 'Unknown'
        }
        
        return model_info
        
    except Exception as e:
        return {
            "name": model_name,
            "error": str(e)
        }

