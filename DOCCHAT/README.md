# DocChat - Multi-Agent RAG with AI Observability

## Overview

DocChat is a multi-agent Retrieval-Augmented Generation (RAG) system adapted from
[IBM's zzpwx-docchat](https://github.com/ibm-developer-skills-network/zzpwx-docchat),
powered by **Azure OpenAI GPT-4** and fully instrumented with the **ai_observability SDK**.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Document Upload    │  ← PDF, DOCX, TXT, MD
│   & Processing       │  ← Cache hit/miss metrics
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Hybrid Retriever   │  ← BM25 + ChromaDB vector search
│  (HuggingFace Emb.) │  ← DB query duration metrics
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│         LangGraph Multi-Agent Pipeline       │
│  ┌───────────────┐                           │
│  │  Relevance    │ → agent_starts_total      │
│  │  Checker      │ → agent_execution_latency │
│  └───────┬───────┘                           │
│          │ handoff ← agent_handoffs_total    │
│          ▼                                   │
│  ┌───────────────┐                           │
│  │  Research     │ → llm_requests_total      │
│  │  Agent        │ → llm_prompt_tokens_total │
│  └───────┬───────┘                           │
│          │ handoff                           │
│          ▼                                   │
│  ┌───────────────┐                           │
│  │ Verification  │ → llm_model_latency_secs  │
│  │  Agent        │ → Self-correction loop    │
│  └───────────────┘                           │
│              ← workflow_duration_seconds      │
└─────────────────────────────────────────────┘
           │
           ▼
     Gradio Web UI
```

## Observability Metrics Captured

| Category | Metrics |
|---|---|
| **Agent Lifecycle** | `agent_starts_total`, `agent_completions_total`, `agent_failures_total` |
| **Agent Performance** | `agent_execution_latency_seconds`, `agent_active_count` |
| **LLM** | `llm_requests_total`, `llm_prompt_tokens_total`, `llm_completion_tokens_total`, `llm_model_latency_seconds`, `llm_api_errors_total` |
| **Workflow** | `workflow_starts_total`, `workflow_completions_total`, `workflow_duration_seconds` |
| **Collaboration** | `agent_handoffs_total`, `agent_inter_agent_latency_seconds` |
| **Infrastructure** | `cache_hits_total`, `cache_misses_total`, `db_query_duration_seconds` |

## Traces Captured

- `workflow_execution` (root span for RAG pipeline)
- `agent_execution` spans for each agent (relevance_checker, research_agent, verification_agent)
- `agent_handoff` spans between agents
- LLM call spans (auto-instrumented via OpenLLMetry)
- Document processing spans
- Retriever build spans

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_docchat.txt
```

### 2. Configure Azure OpenAI

Ensure your `.env` file has:
```
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

### 3. Start Infrastructure (Optional)

```bash
docker-compose up -d
```

This starts:
- **OTel Collector** on port 4317 (gRPC) / 4318 (HTTP)
- **Prometheus** on port 9090
- **Jaeger** on port 16686

### 4. Run DocChat

```bash
python run_docchat.py
```

Open **http://127.0.0.1:5000** in your browser.

### 5. View Observability Data

- **Prometheus** → http://localhost:9090 → query `agent_starts_total`, `llm_requests_total`, etc.
- **Jaeger** → http://localhost:16686 → select service `docchat-rag-app`

## Usage

1. Upload one or more documents (PDF, DOCX, TXT, MD)
2. Enter a question about the document content
3. Click **Submit**
4. View the answer and verification report
5. Check Prometheus/Jaeger for live metrics and traces!
