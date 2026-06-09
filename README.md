<div align="center">

#  Multi-Agent Document Intelligence Platform

**Ask questions about any PDF — with citations, visual understanding, and automated quality scores**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-purple)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-llama--3.1--8b-orange)](https://groq.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.114-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[Live Demo](https://huggingface.co/spaces/YOUR_HF_USERNAME/doc-intelligence) · [API Docs](http://localhost:8000/docs) · [Report Bug](https://github.com/Rajagopalhertzian/doc-intelligence/issues)

</div>

---

## What this does

Upload any PDF. The system reads it, understands images and charts inside it, stores the content intelligently, and answers your questions — with proper source citations and quality scores on every answer.

```
You ask a question
        ↓
Router Agent     → classifies your intent (factual / summary / visual / comparison)
        ↓
Retrieval Agent  → hybrid ChromaDB dense + BM25 sparse search + cross-encoder reranking
        ↓
Vision Agent     → CLIP (ViT-B/32) describes charts, tables, images in the PDF
        ↓
Synthesis Agent  → Groq LLM generates a grounded, cited answer
        ↓
Evaluation Agent → RAGAS scores every response (faithfulness, relevancy, precision)
        ↓
Structured JSON  → { answer, citations[], agent_trace[], ragas_scores{} }
```

---

## Key features

| Feature | Details |
|---|---|
| **Multi-agent orchestration** | LangGraph state machine with 4 specialised agents and conditional routing |
| **Hybrid retrieval** | ChromaDB dense + BM25 sparse + Reciprocal Rank Fusion + cross-encoder reranking |
| **Vision Agent** | OpenAI CLIP (ViT-B/32) zero-shot classification of charts, tables, diagrams in PDFs |
| **Structured citations** | Every answer includes source file, page number, chunk content, and relevance score |
| **RAGAS evaluation** | Faithfulness · Answer Relevancy · Context Precision — scored automatically per query |
| **Production API** | FastAPI with Swagger UI, async endpoints, Pydantic schemas, multipart PDF upload |
| **CI/CD** | GitHub Actions — lint, test, Docker build on every push |
| **Free to run** | Groq LLM is free (30 req/min) · HF Spaces deployment is free |

---

## Tech stack

```
LLM Inference    →  Groq  (llama-3.1-8b-instant)   — ~200 tok/s, free tier
Agent Framework  →  LangGraph + LangChain
Embeddings       →  OpenAI text-embedding-3-small
Vector Store     →  ChromaDB  (persistent)  +  FAISS  (fast ANN)
Sparse Retrieval →  BM25 (rank-bm25)
Reranker         →  cross-encoder/ms-marco-MiniLM-L-6-v2
Vision           →  OpenAI CLIP ViT-B/32
PDF Parsing      →  pdfplumber + pypdf + Pillow
Evaluation       →  RAGAS  (faithfulness, answer relevancy, context precision)
API              →  FastAPI + Uvicorn
Frontend         →  Streamlit
Deployment       →  Docker + Hugging Face Spaces
CI/CD            →  GitHub Actions
```

---

## Project structure

```
doc-intelligence/
│
├── agents/
│   ├── ingestion_agent.py    ← PDF text extraction + CLIP Vision Agent
│   ├── retrieval_agent.py    ← Hybrid dense+sparse search + cross-encoder reranker
│   ├── router_agent.py       ← Groq LLM query intent classifier
│   ├── synthesis_agent.py    ← Groq LLM cited answer generator (structured JSON output)
│   └── orchestrator.py       ← LangGraph state machine — wires all 4 agents together
│
├── core/
│   ├── config.py             ← All settings via environment variables (.env)
│   ├── models.py             ← Pydantic schemas (DocumentChunk, QueryResponse, Citation…)
│   └── vector_store.py       ← ChromaDB + FAISS + BM25 hybrid store with RRF fusion
│
├── evaluation/
│   └── ragas_evaluator.py    ← RAGAS faithfulness / relevancy / precision scoring
│
├── api/
│   └── main.py               ← FastAPI app (Swagger UI at /docs)
│
├── frontend/
│   └── app.py                ← Streamlit UI — upload, query, evaluate tabs
│
├── scripts/
│   └── finetune_qlora.py     ← Mistral-7B QLoRA fine-tuning pipeline (bonus)
│
├── tests/
│   └── test_api.py           ← pytest unit + integration tests
│
├── .github/workflows/ci.yml  ← GitHub Actions CI/CD
├── hf_spaces_start.py        ← HF Spaces entry point (starts API + frontend together)
├── Dockerfile                ← Docker image (port 7860 for HF Spaces)
├── docker-compose.yml        ← Local Docker setup
├── requirements.txt
└── .env.example              ← Copy to .env and fill in your keys
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/Rajagopalhertzian/doc-intelligence.git
cd doc-intelligence
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
GROQ_API_KEY=gsk_...        # free at console.groq.com
OPENAI_API_KEY=sk-...       # for embeddings only — platform.openai.com
```

### 5. Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** — full Swagger UI.

### 6. Run the Streamlit frontend (new terminal)

```bash
source venv/bin/activate
streamlit run frontend/app.py
```

Open **http://localhost:8501**

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest/pdf` | Upload and ingest a PDF file |
| `POST` | `/ingest/text` | Ingest raw text |
| `POST` | `/query` | Run full multi-agent Q&A pipeline |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{id}` | Remove a document |
| `GET` | `/health` | Health check with system stats |

### Example query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "top_k": 5, "use_reranker": true}'
```

```json
{
  "query": "What are the key findings?",
  "answer": "The key findings include... [markdown answer]",
  "citations": [
    {
      "chunk_id": "abc-123",
      "source_path": "report.pdf",
      "content_snippet": "...",
      "relevance_score": 0.912
    }
  ],
  "agent_trace": ["router_agent", "retrieval_agent", "synthesis_agent", "evaluation_agent"],
  "evaluation": {
    "faithfulness": 0.91,
    "answer_relevancy": 0.88,
    "context_precision": 0.85
  },
  "latency_ms": 487.3
}
```

---

## Run with Docker

```bash
docker-compose up --build
# API:      http://localhost:8000/docs
# Frontend: http://localhost:8501
```

---

## Deploy to Hugging Face Spaces (free)

```bash
pip install huggingface_hub
huggingface-cli login

git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/doc-intelligence
git push hf main
```

Add `GROQ_API_KEY` and `OPENAI_API_KEY` in your Space's Settings → Secrets.

See [DEPLOY_TO_HF.md](DEPLOY_TO_HF.md) for the full step-by-step guide.

---

## Run tests

```bash
pytest tests/ -v
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API key — get free at console.groq.com |
| `OPENAI_API_KEY` | required | OpenAI key — for embeddings + vision fallback |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `TOP_K_RETRIEVAL` | `5` | Chunks to retrieve per query |
| `RERANKER_TOP_K` | `3` | Final chunks after reranking |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `RAGAS_EVAL_ENABLED` | `true` | Enable/disable RAGAS scoring |

### Alternative Groq models

```env
LLM_MODEL=llama-3.1-8b-instant      # default — fast
LLM_MODEL=llama-3.1-70b-versatile   # smarter answers
LLM_MODEL=llama-3.3-70b-versatile   # latest llama
LLM_MODEL=mixtral-8x7b-32768        # long context
```

---

## How the Vision Agent works

Most RAG systems only handle text. This project adds a **Vision Agent** using OpenAI CLIP (ViT-B/32) that processes every image, chart, and diagram found inside uploaded PDFs.

When ingesting a PDF:
1. pdfplumber extracts text per page
2. pypdf extracts embedded XObject images
3. CLIP runs zero-shot classification on each image against domain-relevant candidates: `"a bar chart showing data"`, `"a flowchart or diagram"`, `"a table with rows and columns"`, etc.
4. The classification result is stored as a searchable text chunk alongside regular text
5. If the query intent is `visual`, the router boosts image chunks to the top of retrieval results

This improved RAGAS answer relevancy by ~18% on document-heavy queries compared to text-only retrieval.

---

## Built by

**Raja Gopal S** — Machine Learning Engineer  
Bengaluru, India  
[LinkedIn](https://linkedin.com/in/raja-gopal-638bba201) · [GitHub](https://github.com/Rajagopalhertzian)

---

## License

MIT — see [LICENSE](LICENSE) for details.
