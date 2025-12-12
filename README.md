# 🚀 Enhanced RAG System with DeepEval Evaluation

A production-ready **Retrieval-Augmented Generation (RAG)** system with comprehensive evaluation metrics, human-in-the-loop feedback, and interactive web interface.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cohere](https://img.shields.io/badge/Powered%20by-Cohere-orange)](https://cohere.com/)
[![DeepEval](https://img.shields.io/badge/Evaluated%20with-DeepEval-green)](https://docs.confident-ai.com/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [DeepEval Metrics](#deepeval-metrics)
- [Usage Guide](#usage-guide)
- [API Configuration](#api-configuration)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This system combines state-of-the-art technologies to create a powerful, evaluable RAG pipeline:

- **Document Processing**: PDF, DOCX, TXT, MD support
- **Vector Search**: FAISS for efficient similarity search
- **Reranking**: Cohere reranking for improved relevance
- **LLM Generation**: Cohere Command-R-Plus for high-quality answers
- **Evaluation**: DeepEval metrics for comprehensive quality assessment
- **Human Feedback**: Built-in feedback collection and analytics
- **Interactive UI**: Gradio-based web interface

### 🎬 Demo

![RAG System Demo](demo_screenshot.png)

## ✨ Features

### Core Capabilities

- ✅ **Multi-format Document Support**: PDF, DOCX, TXT, Markdown
- ✅ **Smart Text Chunking**: Recursive character splitting with overlap
- ✅ **Vector Database**: FAISS with Cohere embeddings (1024-dim)
- ✅ **Semantic Search**: Top-K retrieval with configurable parameters
- ✅ **Reranking**: Cohere rerank-english-v3.0 for precision
- ✅ **LLM Generation**: Command-R-Plus-08-2024 for answers

### Evaluation & Feedback

- 📊 **DeepEval Integration**: 4 comprehensive metrics
  - Faithfulness (factual consistency)
  - Answer Relevancy (question alignment)
  - Contextual Precision (retrieval quality)
  - Contextual Recall (retrieval completeness)
- 👍 **Human Feedback**: Thumbs up/down, 5-star ratings
- ✏️ **Ground Truth Collection**: Correction system for continuous improvement
- 📈 **Analytics Dashboard**: Track satisfaction rates and trends
- 🔬 **Batch Evaluation**: Evaluate multiple queries simultaneously

### User Interface

- 🖥️ **Web Interface**: Intuitive Gradio-based UI
- 📂 **Document Management**: Upload, browse, and manage documents
- 💬 **Interactive Q&A**: Real-time question answering
- 📊 **Live Metrics**: View evaluation scores instantly
- 📜 **Query History**: Track all interactions
- 💾 **Data Export**: Export feedback and evaluations (JSON)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Gradio)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       RAG Pipeline                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Document   │→ │    Vector    │→ │   Retrieval  │     │
│  │   Processor  │  │    Store     │  │   + Rerank   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Chunking   │  │    FAISS     │  │   Cohere     │     │
│  │  (RecChar)   │  │   + Cohere   │  │   Rerank     │     │
│  │              │  │  Embeddings  │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Generation (Cohere)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              DeepEval Evaluation (4 Metrics)                 │
│    Faithfulness | Answer Relevancy | Precision | Recall     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         Feedback Collection & Analytics Dashboard            │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager
- Cohere API key ([Get one here](https://dashboard.cohere.com/api-keys))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/rag-deepeval-system.git
cd rag-deepeval-system
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Create a configuration file or set environment variables:

```python
# In your notebook or script
config.COHERE_API_KEY = "your-cohere-api-key-here"
```

Or use environment variables:

```bash
export COHERE_API_KEY="your-cohere-api-key-here"
```

## 🚀 Quick Start

### 1. Basic Setup (Jupyter Notebook)

```python
# Run all setup cells in order:
# Cell 1: Installations & Imports
# Cell 2: Configuration
# Cell 3-8: Core Components (Document Loading, Chunking, Vector Store, etc.)
# Cell 9: RAG Pipeline
# Cell 10: DeepEval Setup
# Cell 11: Enhanced AppState
# Cell 12: DeepEval Evaluator
# Cell 13: Query Functions
# Cell 14: Feedback Functions
# Cell 15: Helper Functions
# Cell 16: Gradio Interface
```

### 2. Initialize the System

```python
# Upload or browse for documents
loader = DocumentLoader()
file_paths = loader.upload_files()  # Or use loader.load_from_drive()

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()
rag_pipeline.ingest_documents(file_paths)
```

### 3. Ask Questions

```python
# Simple query
result = rag_pipeline.query("What is MFCC?", use_reranking=True)
print(result['answer'])
```

### 4. Query with Evaluation

```python
# Query with DeepEval evaluation
result = query_with_evaluation(
    question="MFCC signifie quoi?",
    use_reranking=True,
    top_k=5,
    evaluate_response=True,
    ground_truth="MFCC est l'acronyme de Mel Frequency Cepstral Coefficients"
)

# View scores
answer, sources, report, query_id, scores = result
print(scores)
# {'faithfulness': 1.0, 'answer_relevancy': 0.98, 'contextual_precision': 0.85, 'contextual_recall': 0.92}
```

### 5. Run Batch Evaluation

```python
# After collecting multiple queries with ground truth
report, file = run_batch_evaluation()
print(report)
```

## 📊 DeepEval Metrics

### Faithfulness (No Ground Truth Required)

**Measures**: Factual consistency between answer and retrieved context

**Good Score**: ≥ 0.7

**Interpretation**:
- ✅ High (≥0.8): Answer is well-grounded, no hallucinations
- ⚠️ Medium (0.5-0.7): Some unsupported claims
- ❌ Low (<0.5): Significant hallucinations

**Example**:
```
Context: "MFCC stands for Mel Frequency Cepstral Coefficients"
Answer: "MFCC stands for Mel Frequency Cepstral Coefficients" → Score: 1.0 ✅
Answer: "MFCC was invented in 1980" (not in context) → Score: 0.3 ❌
```

### Answer Relevancy (No Ground Truth Required)

**Measures**: How well the answer addresses the question

**Good Score**: ≥ 0.7

**Interpretation**:
- ✅ High: Direct, complete answer
- ⚠️ Medium: Partially addresses question
- ❌ Low: Off-topic or incomplete

**Example**:
```
Question: "What is MFCC?"
Answer: "MFCC stands for Mel Frequency Cepstral Coefficients" → Score: 1.0 ✅
Answer: "Audio processing uses many techniques..." → Score: 0.4 ❌
```

### Contextual Precision (Requires Ground Truth)

**Measures**: Quality of retrieved documents (relevance)

**Good Score**: ≥ 0.7

**Interpretation**:
- ✅ High: Relevant docs ranked higher
- ⚠️ Medium: Some irrelevant docs retrieved
- ❌ Low: Poor retrieval quality

### Contextual Recall (Requires Ground Truth)

**Measures**: Completeness of retrieval (did we find everything?)

**Good Score**: ≥ 0.7

**Interpretation**:
- ✅ High: All relevant info retrieved
- ⚠️ Medium: Some info missing
- ❌ Low: Critical information not found

## 📖 Usage Guide

### Web Interface (Gradio)

#### Document Setup Tab

1. **Browse Documents**: Navigate to folder containing your documents
2. **Upload Files**: Or upload files directly
3. **Initialize System**: Click "Initialize System" and wait for completion

#### Ask & Evaluate Tab

1. **Enter Question**: Type your question in the text box
2. **Configure Settings**:
   - Toggle reranking (recommended: ON)
   - Adjust Top-K (recommended: 5)
   - Enable DeepEval evaluation (optional)
3. **Add Ground Truth** (optional): For complete evaluation
4. **Get Answer**: Click to retrieve answer
5. **Provide Feedback**: Use thumbs, ratings, or corrections

#### Batch Evaluation Tab

1. **Check Readiness**: See if you have enough ground truth
2. **Run Evaluation**: Click "Run Batch Evaluation"
3. **View Report**: See average scores and recommendations
4. **Download**: Save report for later analysis

### Programmatic Usage

#### Basic Query

```python
from rag_pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()
pipeline.ingest_documents(['doc1.pdf', 'doc2.pdf'])

# Query
result = pipeline.query("Your question here")
print(result['answer'])
```

#### With Evaluation

```python
from deepeval_evaluator import DeepEvalEvaluator

# Initialize evaluator
evaluator = DeepEvalEvaluator(model=cohere_model)

# Evaluate
scores = evaluator.evaluate_single_query(
    question="Your question",
    answer=result['answer'],
    contexts=[doc.page_content for doc in result['sources']],
    ground_truth="Expected answer"
)

print(scores)
```

#### Batch Evaluation

```python
# Prepare Q&A pairs
qa_pairs = [
    {
        'question': "Q1",
        'answer': "A1",
        'contexts': ["context1", "context2"],
        'ground_truth': "GT1"
    },
    # ... more pairs
]

# Evaluate
avg_scores = evaluator.evaluate_batch(qa_pairs)
report = evaluator.generate_report(avg_scores)
print(report)
```

## ⚙️ API Configuration

### Cohere Models

```python
class Config:
    # LLM for generation
    COHERE_MODEL = "command-r-plus-08-2024"
    
    # Embedding model
    COHERE_EMBEDDING_MODEL = "embed-english-v3.0"
    
    # Reranking model
    COHERE_RERANK_MODEL = "rerank-english-v3.0"
```

### Chunking Parameters

```python
class Config:
    CHUNK_SIZE = 1000        # Characters per chunk
    CHUNK_OVERLAP = 200      # Overlap between chunks
```

### Retrieval Parameters

```python
class Config:
    TOP_K = 10              # Initial retrieval count
    TOP_K_RERANK = 5        # Final count after reranking
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "API Key Error"

**Problem**: Invalid or missing Cohere API key

**Solution**:
```python
# Update API key
config.COHERE_API_KEY = "your-new-key"
os.environ["COHERE_API_KEY"] = config.COHERE_API_KEY
```

#### 2. "Evaluation Failed"

**Problem**: DeepEval evaluation returns no scores

**Solutions**:
- Check console for detailed errors
- Verify API key has remaining quota
- Try without ground truth first
- Wait 5 minutes (rate limits)
- Check internet connection

#### 3. "No Ground Truth Data Available"

**Problem**: Can't run batch evaluation

**Solutions**:
- Submit queries with evaluation enabled
- Or provide corrections to existing queries
- Need at least 1 query with ground truth
- Run: `prepare_for_batch_evaluation()`

#### 4. Low Evaluation Scores

**Faithfulness Low** → Check for hallucinations, improve context
**Answer Relevancy Low** → Answer may be off-topic
**Contextual Precision Low** → Retrieval finding wrong docs
**Contextual Recall Low** → Missing important information

### Performance Optimization

```python
# Faster retrieval (approximate search)
config.FAISS_INDEX_TYPE = "IndexIVFFlat"

# Reduce chunk size for faster processing
config.CHUNK_SIZE = 500

# Reduce Top-K for faster queries
config.TOP_K = 5
```

## 📁 Project Structure

```
rag-deepeval-system/
│
├── course_rag_22.ipynb          # Main Jupyter notebook
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
│
├── components/
│   ├── document_loader.py       # Document loading utilities
│   ├── text_chunker.py          # Text chunking logic
│   ├── vector_store.py          # FAISS vector store manager
│   ├── reranker.py              # Cohere reranking
│   └── llm_generator.py         # LLM generation
│
├── evaluation/
│   ├── deepeval_evaluator.py   # DeepEval metrics
│   └── feedback_collector.py   # Human feedback system
│
├── interface/
│   └── gradio_app.py            # Gradio web interface
│
├── utils/
│   ├── config.py                # Configuration settings
│   └── helpers.py               # Helper functions
│
└── data/
    ├── uploaded_documents/      # User uploaded files
    ├── faiss_index/             # Vector store index
    └── exports/                 # Exported data (JSON)
```

## 📊 Example Results

### Single Query Evaluation

```
Question: "MFCC signifie quoi?"
Ground Truth: "MFCC est l'acronyme de Mel Frequency Cepstral Coefficients"

Results:
✅ Faithfulness:           1.0000
✅ Answer Relevancy:       1.0000
✅ Contextual Precision:   1.0000
✅ Contextual Recall:      1.0000

Overall Score: 1.0000
🏆 EXCELLENT - System performing very well!
```

### Batch Evaluation

```
Evaluated 5 queries with ground truth

Average Scores:
✅ Faithfulness:           0.9400
✅ Answer Relevancy:       0.9600
✅ Contextual Precision:   0.8800
✅ Contextual Recall:      0.9100

Overall Score: 0.9225
🏆 EXCELLENT - System performing very well!
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/rag-deepeval-system.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .
```

### Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits atomic and well-described


## 👤 Contact

**Abdelaziz Ouayazza**

- 📧 Email: ouayazza.abdelaziz@gmail.com
- 💼 LinkedIn: [Abdelaziz Ouayazza](https://www.linkedin.com/in/abdelaziz-ouayazza-61b19227b/)
- 💻 GitHub: [@abdelaziz2003vvb](https://github.com/abdelaziz2003vvb)

## 🙏 Acknowledgments

- **Cohere** - For powerful LLM and embedding APIs
- **DeepEval** - For comprehensive evaluation metrics
- **LangChain** - For RAG pipeline utilities
- **FAISS** - For efficient vector search
- **Gradio** - For the amazing UI framework

## 📚 References

- [Cohere Documentation](https://docs.cohere.com/)
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

## 🗺️ Roadmap

### Version 2.1 (Planned)
- [ ] Support for more document formats (PPTX, HTML)
- [ ] Multi-language support
- [ ] Advanced filtering options
- [ ] Custom metric definitions
- [ ] Integration with MLflow for experiment tracking

### Version 3.0 (Future)
- [ ] Multi-modal RAG (images, tables)
- [ ] Conversational memory
- [ ] Agent-based workflows
- [ ] Fine-tuning support
- [ ] Production deployment templates

---

**⭐ Star this repository if you find it helpful!**

**🐛 Found a bug? [Open an issue](https://github.com/abdelaziz2003vvb/Enhanced-RAG-System-with-DeepEval-Evaluation/issues)**

**💡 Have a feature request? [Start a discussion](https://github.com/abdelaziz2003vvb/Enhanced-RAG-System-with-DeepEval-Evaluation/discussions)**

---

Made by Abdelaziz Ouayazza | © 2025 |
