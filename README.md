# SearchEngine - Code Search with Fine-tuned Embeddings

A comprehensive code search engine using Sentence Transformers, Qdrant vector database, and fine-tuned embeddings on the CoSQA dataset. This project demonstrates advanced semantic search capabilities for code retrieval using state-of-the-art embedding models.

## 🚀 Features

- **Semantic Code Search**: Find relevant code snippets using natural language queries
- **Fine-tuned Model**: Custom-trained embedding model optimized for code search tasks
- **Qdrant Integration**: High-performance vector database for scalable search operations
- **Comprehensive Evaluation**: Multiple metrics (Recall@K, MRR@K, NDCG@K) for performance assessment
- **Interactive Notebook**: Complete implementation in Jupyter notebook with step-by-step execution
- **Dataset Analysis**: Detailed visualization and analysis of the CoSQA dataset
- **Real-time Search**: Interactive search demonstration with sample queries

## 📁 Project Structure

```
SearchEngine/
├── SearchEngine.ipynb                   # Main Jupyter notebook (complete implementation)
├── engine.py                           # EmbeddingSearchEngine class
├── vectordb.py                         # Qdrant vector database interface
├── utils/                              # Utility modules
│   ├── __init__.py                     # Package initialization
│   ├── cosqa_dataset.py               # CoSQA dataset loading and processing
│   ├── evaluation_metrics.py          # Evaluation metrics (Recall@K, MRR@K, NDCG@K)
│   ├── compute_embeddings.py          # Embedding computation utilities
│   ├── code_search_evaluator.py       # Code search evaluation utilities
│   ├── display_search_results.py      # Search results display utilities
│   └── load_documents.py              # Document loading utilities
├── cosqa-ft-all-MiniLM-L6-v2-*/       # Fine-tuned model directory
│   ├── config.json                     # Model configuration
│   ├── model.safetensors              # Model weights
│   ├── tokenizer.json                 # Tokenizer configuration
│   └── ...                            # Other model files
├── data/                              # Data directory
│   └── cosqa/                         # CoSQA dataset cache
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── PROJECT_STRUCTURE.md               # Detailed project structure
├── SETUP.md                          # Setup instructions
└── venv/                             # Virtual environment (not tracked in git)
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SearchEngine
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# For Qdrant Cloud (recommended)
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key

# For local Qdrant (alternative)
# QDRANT_MODE=local
# QDRANT_URL=http://localhost:6333
```

### 5. Run the Notebook

```bash
jupyter notebook SearchEngine.ipynb
```

## 📊 Dataset

This project uses the **CoSQA (Code Search Question Answering)** dataset, which contains:
- **20,604 queries**: Natural language questions about code
- **20,604 documents**: Python code snippets  
- **500 relevant pairs**: Query-document relevance annotations
- **Source**: Stack Overflow questions and answers
- **Task**: Code search and retrieval evaluation

### Dataset Characteristics

- **Query Length**: Average ~6-8 words per query
- **Document Length**: Average ~30-50 words per code snippet
- **Relevance Distribution**: 1-2 relevant documents per query (sparse)
- **Content Types**: Functions, classes, algorithms, and utility code
- **Language**: Primarily Python code snippets
- **Domain**: Stack Overflow questions and answers

## 🔧 Usage

### Main Notebook (`SearchEngine.ipynb`)

The main notebook contains the complete implementation:

1. **Environment Setup**: Qdrant connection and search engine initialization
2. **Sample Data Testing**: Test the search engine with sample code documents
3. **Dataset Loading**: Load CoSQA dataset from Hugging Face (20,604 queries, 20,604 docs, 500 relevant pairs)
4. **Dataset Analysis**: Comprehensive visualization and statistics with 6 detailed plots
5. **Model Training**: Fine-tune Sentence Transformers on code search task (3 epochs, custom loss function)
6. **Search Implementation**: Build search index and implement search functionality with Qdrant
7. **Evaluation**: Performance evaluation with multiple metrics (Recall@K, MRR@K, NDCG@K)
8. **Interactive Testing**: Sample queries and search demonstrations

### Expected Output

When you run the notebook, you'll see:
- **Dataset Statistics**: Query/document length distributions, word frequency analysis
- **Training Progress**: Loss curves and gradient tracking during fine-tuning
- **Search Results**: Real-time search with similarity scores and code snippets
- **Performance Metrics**: Detailed evaluation results comparing baseline vs fine-tuned models
- **Visualizations**: Histograms, scatter plots, word clouds, and statistical charts

### Key Components

#### EmbeddingSearchEngine (`engine.py`)
```python
from engine import EmbeddingSearchEngine

# Initialize search engine
search_engine = EmbeddingSearchEngine(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    vector_db=vector_db
)

# Build search index
search_engine.build(documents, paths)

# Search for relevant documents
results = search_engine.search("python binary search", k=5)
```

#### VectorDatabase (`vectordb.py`)
```python
from vectordb import VectorDatabase

# Initialize Qdrant connection
vector_db = VectorDatabase(
    collection_name="code_search",
    use_local=False  # Use cloud Qdrant
)

# Create collection
vector_db.create_collection(vector_size=384)
```

## 📈 Model Performance

The fine-tuned model shows significant improvements over the baseline on the CoSQA test set:

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Recall@10 | 47% | 60.2% | +13.2% |
| MRR@10 | 36.8% | 46.7% | +9.9% |
| NDCG@10 | 39.3% | 50% | +10.7% |


*Note: Performance may vary based on training configuration and dataset splits.*

## 🔍 Search Examples

### Sample Queries and Results

1. **Query**: "python binary search algorithm"
   - **Result**: Binary search implementation with proper indexing

2. **Query**: "fibonacci sequence calculation"
   - **Result**: Recursive and iterative Fibonacci implementations

3. **Query**: "prime number checker"
   - **Result**: Prime number validation functions

4. **Query**: "sorting algorithms"
   - **Result**: Bubble sort, quick sort, and merge sort implementations

## 🧪 Evaluation Metrics

The project implements comprehensive evaluation metrics:

- **Recall@K**: Fraction of relevant documents retrieved in top-K results
- **MRR@K**: Mean Reciprocal Rank of the first relevant document
- **NDCG@K**: Normalized Discounted Cumulative Gain at rank K

## 🚀 Advanced Features

### Dataset Analysis
- **Length Distribution**: Query and document length analysis
- **Word Frequency**: Most common programming terms
- **Document Classification**: Function, class, and algorithm categorization
- **Visualization**: Comprehensive plots and word clouds

### Fine-tuning Process
- **Custom Loss Function**: Optimized for code search tasks
- **Gradient Tracking**: Proper gradient flow for model training
- **Device Management**: Automatic CUDA/CPU device handling
- **Model Persistence**: Save and load fine-tuned models

### Search Capabilities
- **Semantic Search**: Natural language to code translation
- **Similarity Scoring**: Cosine similarity with normalized embeddings
- **Batch Processing**: Efficient batch search operations
- **Result Ranking**: Relevance-based result ordering

## 🛠️ Development

### Adding New Features

1. **New Evaluation Metrics**: Add to `utils/evaluation_metrics.py`
2. **Dataset Support**: Extend `utils/cosqa_dataset.py`
3. **Search Algorithms**: Modify `engine.py` for new search methods
4. **Visualizations**: Add plots to the analysis cells

### Testing

```bash
# Run sample data tests
python -c "from engine import EmbeddingSearchEngine; print('✅ Engine imports successfully')"

# Test Qdrant connection
python -c "from vectordb import VectorDatabase; print('✅ Qdrant imports successfully')"
```

## 📚 Dependencies

### Core Dependencies
- `sentence-transformers==3.0.1` - Embedding models
- `qdrant-client==1.15.1` - Vector database client
- `torch==2.8.0+cu129` - Deep learning framework
- `datasets==4.2.0` - Hugging Face datasets
- `matplotlib==3.10.7` - Plotting and visualization

### Development Dependencies
- `jupyter` - Notebook environment
- `tqdm` - Progress bars
- `numpy` - Numerical computing
- `pandas` - Data manipulation

See `requirements.txt` for complete dependency list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Sentence Transformers library
- **Qdrant** for the vector database platform
- **CoSQA Dataset** creators for the evaluation dataset
- **Sentence Transformers** community for model improvements

## 🔧 Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   ```bash
   # Ensure environment variables are set
   echo $QDRANT_URL
   echo $QDRANT_API_KEY
   ```

2. **CUDA Out of Memory**
   ```python
   # Reduce batch size in training
   batch_size = 8  # Instead of 16
   ```

3. **Dataset Loading Issues**
   ```python
   # Check internet connection and Hugging Face access
   from datasets import load_dataset
   dataset = load_dataset("CoIR-Retrieval/cosqa", split="test")
   ```

1. **Data Loading**: Loads CoSQA dataset from Hugging Face
2. **Model Training**: Fine-tunes Sentence Transformers on code search task
3. **Qdrant Setup**: Configures vector database for search
4. **Search Engine**: Implements semantic search with fine-tuned model
5. **Evaluation**: Comprehensive performance evaluation with multiple metrics
6. **Interactive Testing**: Sample queries and interactive search demonstration

### Performance Tips

- Use GPU for training (CUDA-compatible)
- Increase batch size if you have more memory
- Use Qdrant Cloud for better performance
- Monitor training loss for convergence

## Requirements

- Python 3.8+
- PyTorch
- Sentence Transformers
- Qdrant Client
- Jupyter Notebook
- See `requirements.txt` for complete list


