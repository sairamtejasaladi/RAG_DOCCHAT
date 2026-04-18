# DocChat - Multi-Agent RAG System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An intelligent document question-answering system powered by multi-agent architecture and hybrid retrieval**

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Overview

**DocChat** is a state-of-the-art Retrieval-Augmented Generation (RAG) system that leverages a multi-agent architecture to provide accurate, verified answers from your documents. Unlike traditional RAG systems, DocChat employs specialized AI agents that collaborate to ensure answer quality, relevance, and factual accuracy.

### Why DocChat?

- 🎯 **Multi-Agent Intelligence**: Three specialized agents work together for optimal results
- 🔍 **Hybrid Search**: Combines BM25 (keyword) and vector (semantic) search for comprehensive retrieval
- ✅ **Automatic Verification**: Built-in fact-checking ensures answer accuracy
- 🔄 **Self-Correction Loop**: Automatically refines answers that fail verification
- 📄 **Multiple Formats**: Supports PDF, DOCX, TXT, and Markdown files
- ⚡ **Smart Caching**: Optimized processing with intelligent document caching
- 🖥️ **User-Friendly UI**: Clean Gradio interface for easy interaction

---

## ✨ Features

### Core Capabilities

- **Document Processing**
  - Multi-format support (PDF, DOCX, TXT, MD)
  - Intelligent text chunking with overlap
  - Document caching for faster reprocessing
  - Deduplication across uploaded files

- **Hybrid Retrieval System**
  - BM25 keyword-based retrieval (40% weight)
  - Vector semantic search (60% weight)
  - Ensemble approach for better coverage
  - Built-in ONNX embeddings (all-MiniLM-L6-v2)

- **Multi-Agent Workflow**
  - **Relevance Checker**: Validates question relevance before processing
  - **Research Agent**: Generates comprehensive answers from context
  - **Verification Agent**: Fact-checks answers against source documents
  - Self-correction loop for continuous improvement

- **Production Ready**
  - Comprehensive error handling
  - Structured logging with loguru
  - Environment-based configuration
  - Containerization ready

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Gradio)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  DOCUMENT PROCESSOR                         │
│  • Extracts text from PDF/DOCX/TXT/MD                      │
│  • Chunks documents (1000 chars, 200 overlap)              │
│  • Caches processed documents (7 days)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  HYBRID RETRIEVER                           │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ BM25 (40%)   │              │ Vector (60%) │            │
│  │ Keyword      │    Ensemble  │ Semantic     │            │
│  │ Search       │◄────────────►│ Search       │            │
│  └──────────────┘              └──────────────┘            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 MULTI-AGENT WORKFLOW                        │
│                                                             │
│  ┌──────────────────┐                                      │
│  │ RELEVANCE        │  Classifies: CAN_ANSWER,             │
│  │ CHECKER          │  PARTIAL, NO_MATCH                   │
│  └────────┬─────────┘                                      │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                      │
│  │ RESEARCH         │  Generates draft answer              │
│  │ AGENT            │  from retrieved context              │
│  └────────┬─────────┘                                      │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                      │
│  │ VERIFICATION     │  Fact-checks answer:                 │
│  │ AGENT            │  - Supported?                        │
│  └────────┬─────────┘  - Contradictions?                   │
│           │            - Unsupported claims?               │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                      │
│  │ Decision Point   │  Failed? → Loop back to Research     │
│  │                  │  Passed? → Return answer             │
│  └──────────────────┘                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ FINAL ANSWER  │
                    │ + VERIFICATION│
                    │ REPORT        │
                    └───────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Azure OpenAI API access
- 4GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/doc_chat.git
cd doc_chat
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements_docchat.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_MODEL_NAME=gpt-4
```

---

## ⚙️ Configuration

All configuration is managed through environment variables and `docchat/config/settings.py`:

### Environment Variables (.env)

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key | ✅ Yes | - |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | ✅ Yes | - |
| `AZURE_OPENAI_API_VERSION` | API version | No | 2024-12-01-preview |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Model deployment name | No | gpt-4 |
| `AZURE_OPENAI_MODEL_NAME` | Model name | No | gpt-4 |

### Application Settings

Edit `docchat/config/settings.py` to customize:

```python
# Retrieval settings
VECTOR_SEARCH_K: int = 10                    # Top-k documents to retrieve
HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]  # [BM25, Vector] weights

# Cache settings
CACHE_DIR: str = "document_cache"
CACHE_EXPIRE_DAYS: int = 7

# File limits
MAX_FILE_SIZE: int = 50 * 1024 * 1024        # 50 MB per file
MAX_TOTAL_SIZE: int = 200 * 1024 * 1024      # 200 MB total
```

---

## 💻 Usage

### Running the Application

```bash
python run_docchat.py
```

The application will start on `http://127.0.0.1:7860`

### Using the Web Interface

1. **Upload Documents**
   - Click "Upload Documents" button
   - Select one or more files (PDF, DOCX, TXT, MD)
   - Wait for processing confirmation

2. **Ask Questions**
   - Type your question in the "Question" text box
   - Click "Submit" button
   - Wait for the multi-agent system to process

3. **Review Results**
   - **Answer**: The verified answer to your question
   - **Verification Report**: Detailed fact-check results including:
     - Supported: YES/NO
     - Unsupported Claims: List of unverified statements
     - Contradictions: Any contradictory information
     - Relevance: YES/NO
     - Additional Details: Extra context

### Example Questions

```
1. "What are the main features of the product described in the manual?"
2. "Summarize the key findings from the research paper."
3. "What installation steps are required according to the documentation?"
4. "Compare the pricing models mentioned in the proposal."
```

---

## 📁 Project Structure

```
doc_chat/
│
├── run_docchat.py              # Application entry point
├── requirements_docchat.txt     # Python dependencies
├── .env                         # Environment variables (create this)
├── README.md                    # This file
├── TECHNICAL_DOCUMENTATION.md   # Detailed technical guide
│
└── docchat/                     # Main package
    ├── __init__.py
    ├── app.py                   # Gradio UI and main orchestration
    │
    ├── agents/                  # Multi-agent system
    │   ├── __init__.py
    │   ├── relevance_checker.py # Validates question relevance
    │   ├── research_agent.py    # Generates answers
    │   ├── verification_agent.py # Fact-checks answers
    │   └── workflow.py          # LangGraph orchestration
    │
    ├── config/                  # Configuration
    │   ├── __init__.py
    │   ├── settings.py          # Settings and env loading
    │   └── constants.py         # Constants
    │
    ├── document_processor/      # Document handling
    │   ├── __init__.py
    │   └── file_handler.py      # PDF/DOCX/TXT parsing
    │
    ├── retriever/               # Retrieval system
    │   ├── __init__.py
    │   └── builder.py           # Hybrid retriever builder
    │
    ├── utils/                   # Utilities
    │   ├── __init__.py
    │   └── logging.py           # Logging configuration
    │
    └── examples/                # Sample documents
        └── sample_ai_agents.md
```

---

## 🔧 Technical Details

### Document Processing

- **Text Extraction**: Uses `pypdf` for PDFs, `python-docx` for DOCX
- **Chunking Strategy**: Recursive character splitting with 1000 char chunks, 200 char overlap
- **Caching**: SHA-256 based caching with 7-day expiration
- **Deduplication**: Content-based deduplication across multiple files

### Retrieval System

- **Vector Store**: ChromaDB with ONNX embeddings (all-MiniLM-L6-v2)
- **BM25 Retriever**: Traditional keyword-based retrieval
- **Ensemble Weighting**: 40% BM25, 60% Vector (configurable)
- **Top-K**: Retrieves 10 most relevant chunks

### Agent System

Built on **LangGraph** for workflow orchestration:

1. **Relevance Checker**
   - Uses GPT-4 to classify relevance
   - Returns: CAN_ANSWER, PARTIAL, or NO_MATCH
   - Prevents processing of irrelevant questions

2. **Research Agent**
   - Generates comprehensive answers
   - Uses retrieved context
   - Temperature: 0.3, Max tokens: 800

3. **Verification Agent**
   - Fact-checks generated answers
   - Validates claims against source documents
   - Temperature: 0.0, Max tokens: 500

### LLM Configuration

- **Provider**: Azure OpenAI
- **Model**: GPT-4
- **Temperature**: 0.0-0.3 (agent-specific)
- **Max Tokens**: 10-800 (agent-specific)

---

## 📚 Documentation

- **[README.md](README.md)** - This file (overview and quick start)
- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** - Comprehensive technical guide covering:
  - Detailed agent logic
  - Chunking strategies
  - Hybrid search implementation
  - Code walkthrough
  - Troubleshooting guides

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use

```
OSError: Cannot find empty port in range: 7860-7860
```

**Solution**: Change the port in `docchat/app.py`:
```python
demo.launch(server_name="127.0.0.1", server_port=8080, share=False)
```

#### 2. Azure OpenAI Authentication Error

```
Error: Invalid API key
```

**Solution**: Verify your `.env` file:
- Ensure `AZURE_OPENAI_API_KEY` is correct
- Check `AZURE_OPENAI_ENDPOINT` format
- Verify deployment name matches your Azure resource

#### 3. Out of Memory

```
MemoryError during document processing
```

**Solution**: Reduce file sizes or adjust settings:
```python
MAX_TOTAL_SIZE: int = 100 * 1024 * 1024  # Reduce to 100 MB
```

#### 4. Import Errors

```
ModuleNotFoundError: No module named 'langchain'
```

**Solution**: Reinstall dependencies:
```bash
pip install -r requirements_docchat.txt --force-reinstall
```

### Getting Help

- Check [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for detailed explanations
- Review logs in console output
- Verify environment variables are set correctly

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **LangChain** - Framework for LLM applications
- **LangGraph** - Workflow orchestration
- **Azure OpenAI** - LLM provider
- **ChromaDB** - Vector database
- **Gradio** - Web UI framework

---

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

## 📊 Evaluation Results

### Retrieval Metrics

| Metric | Score |
|--------|-------|
| recall@5 | 0.7429 |
| recall@10 | 0.8143 |
| precision@5 | 0.1943 |
| precision@10 | 0.1114 |
| mrr | 0.4876 |
| hit@5 | 0.8000 |
| hit@10 | 0.8286 |

### End-to-End Metrics

| Metric | Score |
|--------|-------|
| exact_match | 1.0000 |
| semantic_similarity | 1.0000 |
| satisfaction_proxy | 1.0000 |

---

<div align="center">

**Built with ❤️ using Azure OpenAI, LangChain, and LangGraph**

[⬆ Back to Top](#docchat---multi-agent-rag-system)

</div>
