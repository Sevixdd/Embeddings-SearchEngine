"""
Utils package for the semantic search engine.

This package contains utility functions for:
- Document loading and processing
- Embedding computation
- Search result display and visualization
- Evaluation metrics and code search evaluation
"""

from .load_documents import load_documents_from_directory, load_documents_from_list
from .compute_embeddings import compute_embeddings, get_embedding_model
from .cosqa_dataset import CoSQADataset, load_cosqa_dataset, create_sample_evaluation_data
__all__ = [
    # Document loading
    'load_documents_from_directory',
    'load_documents_from_list',
    
    # Embedding computation
    'compute_embeddings',
    'get_embedding_model',
    
    # Search result display
    'display_search_results',
    'analyze_search_results',
    'create_search_visualizations',
    
    # Evaluation metrics
    'recall_at_k',
    'mrr_at_k', 
    'ndcg_at_k',
    'evaluate_single_query',
    'evaluate_batch_queries',
    'print_evaluation_results',
    'create_evaluation_visualization',
    
    # Dataset handling
    'CoSQADataset',
    'load_cosqa_dataset',
    'create_sample_evaluation_data',
    
    # Code search evaluation
    'CodeSearchEvaluator',
    'run_code_search_evaluation'
]
