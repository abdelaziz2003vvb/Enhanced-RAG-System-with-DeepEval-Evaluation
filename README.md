# 🚀 Enhanced RAG System with DeepEval & Human Feedback

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Cohere](https://img.shields.io/badge/Cohere-API-orange.svg)](https://cohere.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com/)
[![DeepEval](https://img.shields.io/badge/DeepEval-Metrics-purple.svg)](https://docs.confident-ai.com/)
[![License](https://img.shields.io/badge/License-Educational-red.svg)](LICENSE)

A production-ready Retrieval-Augmented Generation (RAG) system with advanced evaluation metrics and human-in-the-loop feedback. Built with Cohere, FAISS, LangChain, and DeepEval for comprehensive document Q&A with quality tracking.

## ✨ Key Features

### 🔍 **Advanced RAG Pipeline**
- **Multi-format Document Support**: PDF, DOCX, TXT, Markdown
- **Intelligent Chunking**: Recursive text splitting with overlap
- **FAISS Vector Search**: Fast approximate nearest neighbor search
- **Cohere Reranking**: Improved retrieval accuracy
- **LLM Generation**: High-quality answers with source citations

### 📊 **DeepEval Evaluation (4 Metrics)**
- **Faithfulness**: Measures factual consistency with retrieved context
- **Answer Relevancy**: Evaluates if answer addresses the question
- **Contextual Precision**: Assesses retrieval quality (requires ground truth)
- **Contextual Recall**: Measures retrieval completeness (requires ground truth)

### 👥 **Human-in-the-Loop Feedback**
- **Thumbs Up/Down**: Quick satisfaction tracking
- **Star Ratings**: 1-5 scale quality assessment
- **Corrections**: Users provide correct answers (auto-builds ground truth)
- **Feedback Dashboard**: Analytics and satisfaction metrics
- **Data Export**: JSON exports for analysis

### 📈 **Batch Evaluation & Analytics**
- Run comprehensive evaluations on multiple queries
- Automatic aggregation of metrics
- Performance tracking over time
- Detailed reports with recommendations

### 🎨 **User-Friendly Interface**
- Gradio web interface with 6 intuitive tabs
- Real-time processing and feedback
- Complete documentation and help section
- Example queries and troubleshooting guides

---

## 🏗️ Architecture

```
┌─────────────────┐
│  User Interface │
│    (Gradio)     │
└────────┬────────┘
         │
    ┌────▼────┐
    │   RAG   │
    │Pipeline │
    └────┬────┘
         │
    ┌────▼──────────────────────────┐
    │                               │
┌───▼────┐  ┌──────────┐  ┌────────▼─────┐
│Document│  │  Vector  │  │  Reranking   │
│Loading │─▶│  Store   │─▶│  (Cohere)    │
│        │  │  (FAISS) │  │              │
└────────┘  └──────────┘  └──────┬───────┘
                                 │
                          ┌──────▼──────┐
                          │     LLM     │
                          │  (Cohere)   │
                          └──────┬──────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │                            │
            ┌──────▼──────┐         ┌──────────▼────────┐
            │  DeepEval   │         │  Human Feedback   │
            │ Evaluation  │         │   Collection      │
            └─────────────┘         └───────────────────┘
```

---

## 📋 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Configuration](#-configuration)
4. [Usage Guide](#-usage-guide)
5. [DeepEval Metrics](#-deepeval-metrics-explained)
6. [API Reference](#-api-reference)
7. [Troubleshooting](#-troubleshooting)
8. [Examples](#-examples)
9. [Contributing](#-contributing)
10. [License](#-license)

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local Jupyter environment
- Cohere API Key ([Get one free](https://dashboard.cohere.com/api-keys))

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/enhanced-rag-system.git
cd enhanced-rag-system
```

### Step 2: Install Dependencies
```python
# Core dependencies
!pip install -q langchain langchain-community langchain-cohere langchain-text-splitters
!pip install -q cohere faiss-cpu pypdf python-docx
!pip install -q sentence-transformers tiktoken

# Evaluation and UI
!pip install -q datasets deepeval --upgrade
!pip install -q gradio --upgrade
```

### Step 3: Set API Key
```python
# In the Config class
COHERE_API_KEY = "your-api-key-here"
```

---

## 🚀 Quick Start

### 1️⃣ **Initialize the System (5 minutes)**

```python
# Load all code cells in order
# The system will auto-initialize when you run the final cell

# Launch the Gradio interface
demo.launch(share=True)
```

### 2️⃣ **Upload Documents**

Navigate to **📁 Document Setup** tab:
- Click **"📁 Browse"** to select a folder, OR
- Use **"📤 Upload Files"** to upload directly
- Click **"🚀 Initialize System"**
- Wait for confirmation

### 3️⃣ **Ask Your First Question**

Go to **💬 Ask & Evaluate** tab:
```
Question: "What is MFCC?"
☑️ Enable Reranking
☐ Evaluate with DeepEval (optional)
```
Click **"Get Answer"**

### 4️⃣ **Enable Evaluation (Optional)**

```
Question: "MFCC signifie quoi?"
☑️ Enable Reranking
☑️ Evaluate with DeepEval
Ground Truth: "MFCC est l'acronyme de Mel Frequency Cepstral Coefficients"
```
Click **"Get Answer"** → View evaluation scores

---

## ⚙️ Configuration

### Core Settings

```python
class Config:
    # API Keys
    COHERE_API_KEY = "your-key-here"
    
    # Models
    COHERE_MODEL = "command-r-plus-08-2024"
    COHERE_EMBEDDING_MODEL = "embed-english-v3.0"
    COHERE_RERANK_MODEL = "rerank-english-v3.0"
    
    # Chunking Parameters
    CHUNK_SIZE = 1000          # Characters per chunk
    CHUNK_OVERLAP = 200        # Overlap between chunks
    
    # Retrieval Parameters
    TOP_K = 10                 # Documents to retrieve
    TOP_K_RERANK = 5           # Documents after reranking
    
    # Storage
    VECTOR_STORE_PATH = "./faiss_index"
    UPLOAD_FOLDER = "./uploaded_documents"
```

### Customization Tips

**For Long Documents:**
```python
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 15
```

**For Short Queries:**
```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
```

**For High Precision:**
```python
TOP_K = 20
TOP_K_RERANK = 3
```

---

## 📖 Usage Guide

### Document Management

#### Supported Formats
- **PDF** (`.pdf`) - Multi-page documents
- **Word** (`.docx`) - Microsoft Word documents
- **Text** (`.txt`, `.md`) - Plain text and Markdown

#### Loading Documents

**Option A: From Folder**
```python
loader = DocumentLoader()
file_paths = loader.load_from_drive("/content/drive/MyDrive/documents")
documents = loader.load_documents(file_paths)
```

**Option B: Upload Files**
```python
uploaded = loader.upload_files()  # Opens file picker
documents = loader.load_documents(uploaded)
```

### Querying the System

#### Basic Query
```python
result = rag_pipeline.query(
    question="What is machine learning?",
    use_reranking=True
)
print(result['answer'])
```

#### Query with Evaluation
```python
result = query_with_evaluation(
    question="What is MFCC?",
    use_reranking=True,
    top_k=5,
    evaluate_response=True,
    ground_truth="MFCC stands for Mel Frequency Cepstral Coefficients"
)
```

### Providing Feedback

#### Quick Feedback
```python
# Thumbs up
record_thumbs_feedback(query_id=0, feedback_type='thumbs_up')

# Star rating
record_rating_feedback(query_id=0, rating=5, comment="Very helpful!")
```

#### Corrections (Builds Ground Truth)
```python
record_correction(
    query_id=0,
    corrected_answer="The correct answer is...",
    comment="Original answer was incomplete"
)
```

### Batch Evaluation

```python
# Automatic evaluation of all queries with ground truth
report, file = run_batch_evaluation()
print(report)
```

### Data Export

```python
# Export all feedback, evaluations, and ground truth
summary, feedback_file, eval_file, gt_file = export_all_data()
```

---

## 📊 DeepEval Metrics Explained

### 1. **Faithfulness** (No Ground Truth Needed)
**Question:** Is the answer factually consistent with the retrieved context?

**What it Measures:**
- No hallucinations or fabricated information
- All claims are grounded in source documents
- No contradictions with retrieved context

**Score Interpretation:**
- **0.8 - 1.0**: Excellent - Fully grounded
- **0.7 - 0.8**: Good - Mostly grounded
- **0.5 - 0.7**: Fair - Some hallucinations
- **< 0.5**: Poor - Significant hallucinations

**Example:**
```
✅ Good (0.95): "MFCC stands for Mel Frequency Cepstral Coefficients" 
   (if this exact info is in documents)

❌ Bad (0.3): "MFCC was invented by John Smith in 1980" 
   (if this info is NOT in documents)
```

### 2. **Answer Relevancy** (No Ground Truth Needed)
**Question:** Does the answer directly address what was asked?

**What it Measures:**
- Answer is on-topic
- No unnecessary information
- Directly responds to the question

**Score Interpretation:**
- **0.8 - 1.0**: Excellent - Highly relevant
- **0.7 - 0.8**: Good - Relevant with minor extras
- **0.5 - 0.7**: Fair - Partially off-topic
- **< 0.5**: Poor - Not addressing question

**Example for "What is MFCC?":**
```
✅ Good (0.9): "MFCC stands for Mel Frequency Cepstral Coefficients, 
   used in audio signal processing"

❌ Bad (0.4): "Audio processing involves many techniques like FFT, 
   spectrograms, wavelets, and various filtering methods..."
```

### 3. **Contextual Precision** (Requires Ground Truth)
**Question:** Are the retrieved documents relevant and useful?

**What it Measures:**
- Relevant docs ranked higher
- Irrelevant docs ranked lower
- Quality of retrieval system

**Score Interpretation:**
- **0.8 - 1.0**: Excellent retrieval
- **0.7 - 0.8**: Good retrieval
- **0.5 - 0.7**: Fair - some noise
- **< 0.5**: Poor - too much noise

**Why It Matters:** High precision means less noise in context, leading to better answers.

### 4. **Contextual Recall** (Requires Ground Truth)
**Question:** Did we retrieve all necessary information?

**What it Measures:**
- All important facts retrieved
- No critical information missing
- Completeness of retrieval

**Score Interpretation:**
- **0.8 - 1.0**: Excellent - Complete info
- **0.7 - 0.8**: Good - Mostly complete
- **0.5 - 0.7**: Fair - Some gaps
- **< 0.5**: Poor - Major gaps

**Why It Matters:** High recall ensures comprehensive answers with all relevant details.

---

## 🔧 API Reference

### Core Classes

#### `RAGPipeline`
Main orchestration class for the RAG system.

```python
pipeline = RAGPipeline()

# Ingest documents
pipeline.ingest_documents(file_paths)

# Query
result = pipeline.query(
    question="Your question",
    use_reranking=True
)

# Load existing index
pipeline.load_existing_index()
```

#### `DocumentLoader`
Handles document loading from various sources.

```python
loader = DocumentLoader()

# Upload files
paths = loader.upload_files()

# Load from Google Drive
paths = loader.load_from_drive("/path/to/folder")

# Load documents
docs = loader.load_documents(paths)
```

#### `FAISSVectorStore`
Manages FAISS vector database.

```python
store = FAISSVectorStore()

# Create index
store.create_vectorstore(chunks)

# Save/Load
store.save_vectorstore()
store.load_vectorstore()

# Search
docs = store.similarity_search("query", k=10)
```

#### `CohereReranker`
Reranks retrieved documents using Cohere.

```python
reranker = CohereReranker()

# Rerank documents
reranked = reranker.rerank(
    query="question",
    documents=retrieved_docs,
    top_k=5
)
```

#### `DeepEvalEvaluator`
Evaluates responses using DeepEval metrics.

```python
evaluator = DeepEvalEvaluator(model=cohere_model)

# Single evaluation
scores = evaluator.evaluate_single_query(
    question="What is X?",
    answer="X is...",
    contexts=["context1", "context2"],
    ground_truth="X is..."  # Optional
)

# Batch evaluation
avg_scores = evaluator.evaluate_batch(qa_pairs)

# Generate report
report = evaluator.generate_report(scores)
```

### Helper Functions

#### Feedback Management
```python
# Record feedback
record_thumbs_feedback(query_id, 'thumbs_up')
record_rating_feedback(query_id, rating=5)
record_correction(query_id, corrected_answer, comment)

# Get dashboard
dashboard = get_feedback_dashboard()

# Export data
export_all_data()
```

#### Ground Truth Management
```python
# Add ground truth to existing query
add_ground_truth_to_query(query_id, ground_truth_text)

# Show queries without ground truth
show_queries_without_ground_truth()

# Check batch evaluation readiness
prepare_for_batch_evaluation()
```

---

## 🐛 Troubleshooting

### Common Issues

#### ❌ "System Not Initialized"
**Problem:** Cannot ask questions

**Solution:**
1. Go to "Document Setup" tab
2. Upload or browse for documents
3. Click "Initialize System"
4. Wait for success message

#### ❌ "Evaluation Failed" or No Scores
**Problem:** DeepEval returns no scores

**Solutions:**
1. Check console for detailed errors
2. Verify Cohere API key is valid: [Dashboard](https://dashboard.cohere.com/)
3. Check internet connection
4. Wait 5 minutes (rate limits)
5. Try without ground truth first
6. Restart kernel and reload

#### ❌ "No Ground Truth Data Available"
**Problem:** Cannot run batch evaluation

**Solutions:**
1. Submit queries with evaluation enabled
2. Provide corrections to existing queries
3. Need at least 1 query with ground truth
4. Run: `prepare_for_batch_evaluation()`

#### ⚠️ Evaluation Takes Too Long

**Normal Timing:**
- Single query (no GT): 5-10 seconds
- Single query (with GT): 10-20 seconds
- Batch (3 queries): 30-60 seconds
- Batch (5 queries): 50-100 seconds

**If Slower:**
- Check internet connection
- Cohere API might be slow (retry later)
- Rate limits (wait 5-10 minutes)

#### 📉 Low Evaluation Scores

**Low Faithfulness:**
- System is hallucinating
- Check document quality
- Improve chunking strategy
- Verify documents are relevant

**Low Answer Relevancy:**
- Answers are off-topic
- Rephrase questions
- Check if documents contain relevant info
- Adjust prompt template

**Low Contextual Precision:**
- Wrong documents retrieved
- Adjust `TOP_K` parameter
- Enable reranking
- Improve embeddings

**Low Contextual Recall:**
- Missing information
- Increase `TOP_K`
- Check if info exists in documents
- Reduce `CHUNK_SIZE` for granular retrieval

### Error Messages

#### "API Key Error"
```
Solution: Get new key at https://dashboard.cohere.com/api-keys
```

#### "Rate Limit Exceeded"
```
Solution: Wait 5-10 minutes, check usage limits on dashboard
```

#### "Context is empty"
```
Solution: Documents not loaded properly, reinitialize system
```

#### "Model generation failed"
```
Solution: Check Cohere API status, verify internet connection
```

---

## 💡 Examples

### Example 1: Basic RAG Query
```python
# Initialize system
rag = RAGPipeline()
rag.ingest_documents(["document1.pdf", "document2.docx"])

# Query
result = rag.query("What is machine learning?")
print(result['answer'])

# View sources
for doc in result['sources']:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}...")
```

### Example 2: Query with Full Evaluation
```python
result = query_with_evaluation(
    question="What are MFCC features used for?",
    use_reranking=True,
    top_k=5,
    evaluate_response=True,
    ground_truth="MFCC features are used for audio signal processing and speech recognition"
)

print(f"Answer: {result['answer']}")
print(f"Scores: {result['evaluation_scores']}")
```

### Example 3: Batch Evaluation Workflow
```python
# Step 1: Submit multiple queries with ground truth
questions = [
    ("What is MFCC?", "MFCC stands for Mel Frequency Cepstral Coefficients"),
    ("How to calculate MFCC?", "MFCC is calculated using FFT and mel-scale filtering"),
    ("What is a spectrogram?", "A spectrogram is a visual representation of signal frequencies")
]

for q, gt in questions:
    query_with_evaluation(q, True, 5, True, gt)

# Step 2: Run batch evaluation
report, file = run_batch_evaluation()
print(report)
```

### Example 4: Human Feedback Loop
```python
# User asks question
result = rag.query("What is deep learning?")
query_id = len(app_state.chat_history) - 1

# User provides feedback
record_thumbs_feedback(query_id, 'thumbs_up')
record_rating_feedback(query_id, rating=4, comment="Good but could be more detailed")

# User provides correction (builds ground truth)
record_correction(
    query_id,
    corrected_answer="Deep learning is a subset of machine learning using neural networks with multiple layers",
    comment="Original answer was too vague"
)
```

### Example 5: Export and Analyze Feedback
```python
# Export all data
summary, feedback_file, eval_file, gt_file = export_all_data()

# Load and analyze in pandas
import json
import pandas as pd

with open(feedback_file, 'r') as f:
    feedback = json.load(f)

df = pd.DataFrame(feedback)
print(df[['timestamp', 'feedback_type', 'rating']].head())

# Calculate satisfaction rate
thumbs_up = df[df['feedback_type'] == 'thumbs_up'].shape[0]
thumbs_down = df[df['feedback_type'] == 'thumbs_down'].shape[0]
satisfaction = thumbs_up / (thumbs_up + thumbs_down) * 100
print(f"Satisfaction Rate: {satisfaction:.1f}%")
```

---

## 🎯 Best Practices

### Document Preparation
1. **Clean your documents** - Remove headers, footers, page numbers
2. **Structure matters** - Use clear headings and sections
3. **Avoid scans** - Use native PDFs, not scanned images
4. **Size appropriately** - 10-100 pages per document is ideal
5. **Related content** - Keep topically related documents together

### Ground Truth Creation
**Good Ground Truth:**
- ✅ Accurate and complete
- ✅ Based on your documents
- ✅ Clear and concise (1-3 sentences)
- ✅ Consistent in quality

**Bad Ground Truth:**
- ❌ Too vague: "It's a thing"
- ❌ Too detailed: Full paragraphs
- ❌ Inconsistent: Sometimes brief, sometimes detailed
- ❌ Inaccurate: Doesn't match documents

### Query Optimization
1. **Be specific** - "What is MFCC?" vs "Tell me about audio"
2. **Use keywords** from your documents
3. **Avoid ambiguity** - Clear, focused questions
4. **Test variations** - Try rephrasing for better results

### Evaluation Strategy
1. **Start simple** - Test without ground truth first
2. **Collect diverse questions** - Different topics and complexity
3. **Provide accurate ground truth** - Quality over quantity
4. **Evaluate regularly** - After every 10-20 queries
5. **Track trends** - Compare scores over time

### Performance Tuning
**For Better Precision (fewer wrong documents):**
```python
TOP_K = 5
TOP_K_RERANK = 3
# Enable reranking
```

**For Better Recall (more complete answers):**
```python
TOP_K = 15
TOP_K_RERANK = 7
CHUNK_SIZE = 800
```

**For Faster Responses:**
```python
TOP_K = 5
TOP_K_RERANK = 3
# Disable evaluation for regular queries
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the Repository**
```bash
git fork https://github.com/yourusername/enhanced-rag-system
```

2. **Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make Your Changes**
- Follow existing code style
- Add comments and documentation
- Test thoroughly

4. **Commit Changes**
```bash
git commit -m "Add: Your feature description"
```

5. **Push and Create Pull Request**
```bash
git push origin feature/your-feature-name
```

### Contribution Ideas
- 🐛 **Bug Fixes** - Found a bug? Fix it!
- ✨ **New Features** - Add new evaluation metrics
- 📚 **Documentation** - Improve guides and examples
- 🎨 **UI Improvements** - Enhance Gradio interface
- ⚡ **Performance** - Optimize speed and efficiency
- 🧪 **Tests** - Add unit tests
- 🌍 **Translations** - Add multi-language support

---

## 📄 License

This project is licensed for **Educational Use**.

**You may:**
- ✅ Use for learning and education
- ✅ Modify and experiment
- ✅ Share with attribution

**You may not:**
- ❌ Use for commercial purposes without permission
- ❌ Remove attribution
- ❌ Claim as your own work

---

## 🙏 Acknowledgments

### Technologies
- **[Cohere](https://cohere.com/)** - LLM and embeddings
- **[LangChain](https://python.langchain.com/)** - RAG framework
- **[FAISS](https://faiss.ai/)** - Vector similarity search
- **[DeepEval](https://docs.confident-ai.com/)** - Evaluation metrics
- **[Gradio](https://gradio.app/)** - Web interface

### Inspiration
This project was built to demonstrate production-ready RAG systems with proper evaluation and human feedback loops for continuous improvement.

---

## 📞 Support & Contact

**Created by:** Abdelaziz Ouayazza

- 📧 **Email:** ouayazza.abdelaziz@gmail.com
- 💼 **LinkedIn:** [Connect with me](https://www.linkedin.com/in/abdelaziz-ouayazza-61b19227b/)
- 🐙 **GitHub:** [@abdelaziz2003vvb](https://github.com/abdelaziz2003vvb)

### Getting Help

1. **Check Documentation** - Read this README and in-app help
2. **Console Logs** - Most errors show detailed info in console
3. **GitHub Issues** - [Open an issue](https://github.com/yourusername/enhanced-rag-system/issues)
4. **Email** - For direct support

---

## 📊 Project Stats

- **Version:** 2.0 with DeepEval
- **Last Updated:** December 2024
- **Python Version:** 3.8+
- **Dependencies:** 15+ packages
- **Code Lines:** ~2,000+
- **Features:** 20+ major features

---

## 🗺️ Roadmap

### Current Version (v2.0)
- ✅ Basic RAG pipeline
- ✅ DeepEval integration (4 metrics)
- ✅ Human feedback collection
- ✅ Batch evaluation
- ✅ Gradio interface

### Planned Features (v2.1)
- 🔄 Async query processing
- 🔄 Multi-model support (OpenAI, Anthropic)
- 🔄 Advanced analytics dashboard
- 🔄 A/B testing framework
- 🔄 API endpoints (FastAPI)

### Future (v3.0)
- 🔮 Fine-tuning on feedback data
- 🔮 Multi-lingual support
- 🔮 Image and table extraction
- 🔮 Streaming responses
- 🔮 Production deployment guide

---

## 📚 Additional Resources

### Documentation
- [Cohere Documentation](https://docs.cohere.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Gradio Documentation](https://gradio.app/docs/)

### Tutorials
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [LLM Evaluation Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [Vector Database Comparison](https://www.kdnuggets.com/2023/08/vector-database-comparison.html)

### Papers
- [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [FAISS: A Library for Efficient Similarity Search](https://arxiv.org/abs/1702.08734)
- [Cohere Command-R Technical Report](https://cohere.com/research/papers)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

*Last updated: December 2025*
