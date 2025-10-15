"""
Document loading utilities for the semantic search engine.

This module provides functions for loading and processing documents
from various sources including directories, file lists, and direct content.
"""

import os
import glob
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


def load_documents_from_directory(
    directory: str, 
    pattern: str = "*.txt",
    encoding: str = "utf-8"
) -> Tuple[List[str], List[str]]:
    """
    Load documents from a directory matching a specific pattern.
    
    Args:
        directory: Path to the directory containing documents
        pattern: Glob pattern to match files (default: "*.txt")
        encoding: Text encoding to use when reading files
        
    Returns:
        Tuple of (documents, paths) where:
        - documents: List of document contents as strings
        - paths: List of file paths corresponding to documents
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no files match the pattern
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    # Find all files matching the pattern
    search_pattern = os.path.join(directory, pattern)
    file_paths = sorted(glob.glob(search_pattern))
    
    if not file_paths:
        raise ValueError(f"No files found matching pattern '{pattern}' in directory '{directory}'")
    
    documents = []
    valid_paths = []
    
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read().strip()
                if content:  # Only include non-empty documents
                    documents.append(content)
                    valid_paths.append(file_path)
        except Exception as e:
            print(f"Warning: Could not read file '{file_path}': {e}")
            continue
    
    if not documents:
        raise ValueError(f"No readable documents found in directory '{directory}'")
    
    return documents, valid_paths


def load_documents_from_list(
    documents: List[str], 
    paths: Optional[List[str]] = None,
    base_name: str = "doc"
) -> Tuple[List[str], List[str]]:
    """
    Load documents from a list of strings.
    
    Args:
        documents: List of document contents
        paths: Optional list of paths (if None, generates default names)
        base_name: Base name for generated paths if paths not provided
        
    Returns:
        Tuple of (documents, paths)
        
    Raises:
        ValueError: If documents list is empty
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")
    
    if paths is None:
        # Generate default paths
        paths = [f"{base_name}_{i+1}.txt" for i in range(len(documents))]
    elif len(paths) != len(documents):
        raise ValueError("Paths list must have the same length as documents list")
    
    # Filter out empty documents
    valid_documents = []
    valid_paths = []
    
    for doc, path in zip(documents, paths):
        if doc.strip():  # Only include non-empty documents
            valid_documents.append(doc.strip())
            valid_paths.append(path)
    
    if not valid_documents:
        raise ValueError("No non-empty documents found")
    
    return valid_documents, valid_paths


def save_documents_to_directory(
    documents: List[str], 
    paths: List[str], 
    directory: str,
    encoding: str = "utf-8"
) -> List[str]:
    """
    Save documents to a directory.
    
    Args:
        documents: List of document contents
        paths: List of file names (not full paths)
        directory: Target directory to save files
        encoding: Text encoding to use when writing files
        
    Returns:
        List of full paths where files were saved
        
    Raises:
        ValueError: If documents and paths lists have different lengths
    """
    if len(documents) != len(paths):
        raise ValueError("Documents and paths lists must have the same length")
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    saved_paths = []
    
    for doc, filename in zip(documents, paths):
        # Ensure filename has .txt extension if not present
        if not filename.lower().endswith('.txt'):
            filename += '.txt'
        
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, "w", encoding=encoding) as f:
                f.write(doc)
            saved_paths.append(file_path)
        except Exception as e:
            print(f"Warning: Could not save file '{file_path}': {e}")
    
    return saved_paths


def get_document_info(documents: List[str], paths: List[str]) -> Dict[str, Any]:
    """
    Get information about a collection of documents.
    
    Args:
        documents: List of document contents
        paths: List of document paths
        
    Returns:
        Dictionary containing document statistics
    """
    if not documents:
        return {
            "count": 0,
            "total_chars": 0,
            "total_words": 0,
            "avg_chars_per_doc": 0,
            "avg_words_per_doc": 0,
            "min_chars": 0,
            "max_chars": 0
        }
    
    char_counts = [len(doc) for doc in documents]
    word_counts = [len(doc.split()) for doc in documents]
    
    return {
        "count": len(documents),
        "total_chars": sum(char_counts),
        "total_words": sum(word_counts),
        "avg_chars_per_doc": sum(char_counts) / len(documents),
        "avg_words_per_doc": sum(word_counts) / len(documents),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "paths": paths
    }


def validate_documents(documents: List[str], paths: List[str]) -> bool:
    """
    Validate that documents and paths are properly formatted.
    
    Args:
        documents: List of document contents
        paths: List of document paths
        
    Returns:
        True if valid, False otherwise
    """
    if not documents or not paths:
        return False
    
    if len(documents) != len(paths):
        return False
    
    # Check for empty documents
    if any(not doc.strip() for doc in documents):
        return False
    
    return True

