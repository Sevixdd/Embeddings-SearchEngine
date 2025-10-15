"""
CoSQA dataset loader and processor for code search evaluation.

This module provides functions for loading and processing the CoSQA dataset
from Hugging Face for evaluating code search engines.
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
from datasets import load_dataset


class CoSQADataset:
    """CoSQA dataset loader and processor using Hugging Face datasets."""
    
    def __init__(self, data_dir: str = "data/cosqa"):
        """
        Initialize CoSQA dataset loader.
        
        Args:
            data_dir: Directory to store the dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.queries = []
        self.code_snippets = []
        self.ground_truth = []
        self.metadata = []
        self.dataset = None
        
    def load_from_huggingface(self, split: str = "test") -> Tuple[List[str], List[str], List[List[int]]]:
        """
        Load CoSQA dataset from Hugging Face.
        
        Args:
            split: Dataset split ('train', 'test', 'valid')
            
        Returns:
            Tuple of (queries, code_snippets, ground_truth)
        """
        print(f"📥 Loading CoSQA dataset from Hugging Face ({split} split)...")
        
        try:
            # Load the dataset from Hugging Face
            dataset = load_dataset("CoIR-Retrieval/cosqa", split=split)
            self.dataset = dataset
            
            # Extract queries and corpus
            queries_data = load_dataset("CoIR-Retrieval/cosqa", split=split, name="queries")
            corpus_data = load_dataset("CoIR-Retrieval/cosqa", split=split, name="corpus")
            
            # Create mappings
            query_map = {item['query-id']: item['query'] for item in queries_data}
            corpus_map = {item['corpus-id']: item['corpus'] for item in corpus_data}
            
            # Process the main dataset
            queries = []
            code_snippets = []
            ground_truth = []
            
            # Group by query-id to create ground truth
            query_groups = {}
            for item in dataset:
                query_id = item['query-id']
                corpus_id = item['corpus-id']
                score = item['score']
                
                if query_id not in query_groups:
                    query_groups[query_id] = {
                        'query': query_map.get(query_id, ''),
                        'relevant_docs': [],
                        'all_docs': []
                    }
                
                query_groups[query_id]['all_docs'].append(corpus_id)
                if score == 1:  # Relevant document
                    query_groups[query_id]['relevant_docs'].append(corpus_id)
            
            # Convert to lists
            for query_id, group in query_groups.items():
                if group['query']:  # Only include queries with text
                    queries.append(group['query'])
                    ground_truth.append(group['relevant_docs'])
            
            # Get all unique code snippets
            all_corpus_ids = set()
            for group in query_groups.values():
                all_corpus_ids.update(group['all_docs'])
            
            code_snippets = []
            corpus_id_to_index = {}
            for i, corpus_id in enumerate(sorted(all_corpus_ids)):
                if corpus_id in corpus_map:
                    code_snippets.append(corpus_map[corpus_id])
                    corpus_id_to_index[corpus_id] = i
            
            # Convert ground truth to use indices
            for i, gt in enumerate(ground_truth):
                ground_truth[i] = [corpus_id_to_index[doc_id] for doc_id in gt if doc_id in corpus_id_to_index]
            
            print(f"✅ Loaded {len(queries)} queries and {len(code_snippets)} code snippets")
            
            return queries, code_snippets, ground_truth
            
        except Exception as e:
            print(f"❌ Failed to load from Hugging Face: {e}")
            print("🔄 Falling back to sample data...")
            return self._create_sample_data(split)
    
    def _create_sample_data(self, split: str) -> Tuple[List[str], List[str], List[List[int]]]:
        """
        Create sample data for demonstration purposes.
        
        Args:
            split: Dataset split name
            
        Returns:
            Tuple of (queries, code_snippets, ground_truth)
        """
        print(f"📝 Creating sample data for {split} split...")
        
        # Sample code search data
        sample_data = {
            "queries": [
                "binary search algorithm implementation",
                "sorting array using quicksort",
                "calculate fibonacci numbers recursively",
                "merge sort implementation",
                "linear search in array",
                "bubble sort algorithm",
                "factorial calculation using recursion",
                "prime number checking function",
                "greatest common divisor calculation",
                "reverse string function"
            ],
            "code_snippets": [
                "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
                "def linear_search(arr, target):\n    for i, item in enumerate(arr):\n        if item == target:\n            return i\n    return -1",
                "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
                "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
                "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
                "def reverse_string(s):\n    return s[::-1]"
            ],
            "ground_truth": [
                [0],  # binary search query -> binary search code
                [1],  # quicksort query -> quicksort code
                [2],  # fibonacci query -> fibonacci code
                [3],  # merge sort query -> merge sort code
                [4],  # linear search query -> linear search code
                [5],  # bubble sort query -> bubble sort code
                [6],  # factorial query -> factorial code
                [7],  # prime query -> prime code
                [8],  # gcd query -> gcd code
                [9]   # reverse query -> reverse code
            ]
        }
        
        # Adjust data based on split
        if split == "train":
            queries = sample_data["queries"][:7]
            code_snippets = sample_data["code_snippets"]
            ground_truth = sample_data["ground_truth"][:7]
        elif split == "test":
            queries = sample_data["queries"][7:]
            code_snippets = sample_data["code_snippets"]
            ground_truth = sample_data["ground_truth"][7:]
        else:  # valid
            queries = sample_data["queries"][5:8]
            code_snippets = sample_data["code_snippets"]
            ground_truth = sample_data["ground_truth"][5:8]
        
        print(f"✅ Created sample {split} data: {len(queries)} queries, {len(code_snippets)} code snippets")
        
        return queries, code_snippets, ground_truth
    
    def load_split(self, split: str) -> Tuple[List[str], List[str], List[List[int]]]:
        """
        Load a specific split of the dataset.
        
        Args:
            split: Dataset split ('train', 'test', 'valid')
            
        Returns:
            Tuple of (queries, code_snippets, ground_truth)
        """
        return self.load_from_huggingface(split)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        info = {}
        
        for split in ['train', 'test', 'valid']:
            try:
                queries, code_snippets, ground_truth = self.load_split(split)
                
                info[split] = {
                    'num_queries': len(queries),
                    'num_code_snippets': len(code_snippets),
                    'avg_query_length': sum(len(q.split()) for q in queries) / len(queries) if queries else 0,
                    'avg_code_length': sum(len(c.split()) for c in code_snippets) / len(code_snippets) if code_snippets else 0,
                    'avg_relevant_per_query': sum(len(gt) for gt in ground_truth) / len(ground_truth) if ground_truth else 0
                }
            except Exception as e:
                info[split] = {'error': str(e)}
        
        return info
    
    def create_evaluation_set(self, split: str = 'test') -> Tuple[List[str], List[Tuple[int, str]], List[List[int]]]:
        """
        Create an evaluation set for code search.
        
        Args:
            split: Dataset split to use for evaluation
            
        Returns:
            Tuple of (queries, corpus, ground_truth)
        """
        # Load queries and ground truth
        queries, code_snippets, ground_truth = self.load_split(split)
        
        # Create corpus (all code snippets with their IDs)
        corpus = [(i, code) for i, code in enumerate(code_snippets)]
        
        return queries, corpus, ground_truth


def load_cosqa_dataset(data_dir: str = "data/cosqa") -> CoSQADataset:
    """
    Load the CoSQA dataset.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        CoSQADataset instance
    """
    dataset = CoSQADataset(data_dir)
    return dataset


def create_sample_evaluation_data(num_queries: int = 50, num_docs: int = 200) -> Tuple[List[str], List[Tuple[int, str]], List[List[int]]]:
    """
    Create sample evaluation data for testing.
    
    Args:
        num_queries: Number of sample queries
        num_docs: Number of sample documents
        
    Returns:
        Tuple of (queries, corpus, ground_truth)
    """
    # Sample queries
    sample_queries = [
        "binary search algorithm implementation",
        "sorting array using quicksort",
        "calculate fibonacci numbers recursively",
        "merge sort implementation",
        "linear search in array",
        "bubble sort algorithm",
        "factorial calculation using recursion",
        "prime number checking function",
        "greatest common divisor calculation",
        "reverse string function",
        "find maximum element in array",
        "check if string is palindrome",
        "calculate sum of array elements",
        "find duplicate elements in array",
        "implement stack data structure",
        "queue implementation using list",
        "linked list node class",
        "tree traversal algorithms",
        "graph depth first search",
        "breadth first search implementation"
    ]
    
    # Generate queries
    queries = []
    for i in range(num_queries):
        queries.append(sample_queries[i % len(sample_queries)] + f" {i}")
    
    # Generate corpus
    corpus = []
    for i in range(num_docs):
        corpus.append((i, f"def function_{i}():\n    # Sample code implementation {i}\n    pass"))
    
    # Generate ground truth (each query has 1-3 relevant documents)
    ground_truth = []
    for i in range(num_queries):
        # Randomly select 1-3 relevant documents
        import random
        num_relevant = random.randint(1, 3)
        relevant_docs = random.sample(range(num_docs), num_relevant)
        ground_truth.append(relevant_docs)
    
    return queries, corpus, ground_truth

