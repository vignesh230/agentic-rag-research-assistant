# Agentic RAG Research Assistant

[![CI](https://github.com/vignesh230/agentic-rag-research-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/vignesh230/agentic-rag-research-assistant/actions/workflows/ci.yml)

A production-grade RAG service with three configurable retrieval modes, a LangGraph self-critique loop, and a full evaluation harness. Built as a portfolio project for ML/AI engineering interviews.

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        D[Documents] --> C[Chunker]
        C --> E[Embedder]
        E --> PG[(Postgres + pgvector)]
    end

    subgraph API[FastAPI /ask]
        REQ[POST /ask]
    end

    subgraph Naive[Mode: naive]
        N1[Embed query] --> N2[similarity search top-k]
        N2 --> N3[Stuff context]
        N3 --> N4[LLM answer]
    end

    subgraph Reranked[Mode: reranked]
        R1[Embed query] --> R2[Wide retrieval]
        R2 --> R3[CrossEncoder rerank]
        R3 --> R4[LLM answer]
    end

    subgraph Agentic[Mode: agentic LangGraph]
        A1[planner] --> A2[retrieve]
        A2 --> A3[synthesizer]
        A3 --> A4{critic}
        A4 -- supported --> A5[final answer]
        A4 -- rewrite --> A2
    end

    subgraph Eval[Evaluation harness]
        G[golden_set.jsonl] --> H[runner]
        H --> I[string-overlap metrics]
        H --> J[latency + cost]
        I --> K[markdown table]
        J --> K
    end

    PG --> N2
    PG --> R2
    PG --> A2
    REQ --> Naive
    REQ --> Reranked
    REQ --> Agentic
```

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Orchestration | LangGraph 0.6 |
| LLM | NVIDIA NIM `meta/llama-3.1-8b-instruct` (OpenAI-compatible) or Anthropic Claude |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (384-dim) |
| Reranking | sentence-transformers CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Vector store | Postgres + pgvector (HNSW index) |
| Observability | structlog (JSON) + per-node traces |
| CI | GitHub Actions |

---

## Setup

### Prerequisites

- Python 3.11
- Docker + Docker Compose
- NVIDIA NIM API key (free at [build.nvidia.com](https://build.nvidia.com)) **or** Anthropic API key

### 1. Start Postgres

```bash
docker-compose up -d
```

### 2. Install

```bash
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
```

Set your provider in `.env`:

```bash
# NVIDIA NIM (default in benchmarks)
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=meta/llama-3.1-8b-instruct

# or Anthropic Claude
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-...
```

Key settings (all overridable via env vars):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `nvidia` |
| `NVIDIA_API_KEY` | — | Required when `LLM_PROVIDER=nvidia` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `POSTGRES_DSN` | `postgresql://rag:rag@localhost:5432/ragdb` | |
| `RAG_MODE` | `naive` | `naive` / `reranked` / `agentic` |
| `TOP_K` | `5` | Chunks returned per query |
| `RETRIEVAL_MULTIPLIER` | `4` | Wide-retrieval factor for reranked mode |
| `MAX_CRITIC_LOOPS` | `3` | Critic loop cap for agentic mode |
| `CHUNK_STRATEGY` | `recursive` | `fixed` / `sentence` / `recursive` |

### 4. Ingest documents

```bash
python -m rag_agent.ingestion.pipeline --source path/to/docs/
```

Supports `.txt` and `.pdf`. Skips already-ingested sources.

### 5. Run the API

```bash
uvicorn rag_agent.api.main:app --reload
```

**POST /ask** (switch mode per-request without restarting):

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is self-attention?", "mode": "agentic", "top_k": 5}' | jq
```

**GET /health**

```bash
curl http://localhost:8000/health
```

### 6. Docker

```bash
docker build -t rag-agent .
docker run \
  -e LLM_PROVIDER=nvidia \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -e POSTGRES_DSN=postgresql://rag:rag@host.docker.internal:5432/ragdb \
  -p 8000:8000 \
  rag-agent
```

---

## Evaluation

### 1. Write your golden set

See `data/golden_set.jsonl.example` for the schema. Create `data/golden_set.jsonl` with your own questions and ground-truth answers. Do not commit fabricated Q&A.

### 2. Run the harness

```bash
python -m rag_agent.eval.harness --modes naive,reranked,agentic --out results.md
```

### 3. Ablation results

26-question golden set grounded in 5 research papers (RAG survey, CoT, ReAct, vLLM, DeepSeek-R1). LLM: NVIDIA NIM `meta/llama-3.1-8b-instruct`. Metrics: BLEU-2 (answer relevancy) and token-overlap F1 (context recall) — deterministic proxies used because NIM's Llama endpoint does not support the JSON-schema structured output required by LLM-based RAGAS judges.

| Mode | Answer Relevancy (BLEU-2) | Context Recall | Latency p50 (ms) | Latency p95 (ms) | Cost/query ($) | Questions | Errors |
|------|:---:|:---:|---:|---:|---:|---:|---:|
| naive | 0.064 | 0.054 | 849 | 3843 | $0.0049 | 26 | 0 |
| reranked | 0.061 | **0.070** | 991 | 6198 | $0.0047 | 26 | 0 |
| agentic | **0.068** | 0.038 | 3905 | 14975 | $0.037 | 26 | 0 |

Reranked improves context recall +30% over naive with only +17% latency overhead. Agentic scores highest on answer relevancy but at 4.6x the latency and 8x the cost — the self-critique loop earns its keep for precision tasks, not throughput. The LLM client uses exponential-backoff retry (`max_retries=6`) so all three modes complete 26/26 questions with 0 errors even on the free NIM tier.

---

## Tests

```bash
pytest -v                  # unit tests (no DB needed)
pytest -m integration      # needs live Postgres
```

70 tests. Integration tests are marked `@pytest.mark.integration` and skipped by default.

---

## Project structure

```
src/rag_agent/
├── api/            # FastAPI app, schemas, deps, /ask route
├── db/             # pgvector client (HNSW, cosine similarity)
├── eval/           # harness: loader, runner, metrics, CLI
├── graph/          # LangGraph: state, nodes (planner/retrieve/synthesizer/critic), graph
├── ingestion/      # chunker, embedder, pipeline CLI
├── rag/            # naive.py, reranked.py, agentic.py, prompt_loader.py
├── logging_config.py
└── settings.py     # all ablation knobs in one place

prompts/            # versioned YAML prompts (naive_rag, planner, synthesizer, critic)
data/               # golden_set.jsonl, eval_thresholds.json
.github/workflows/  # CI: pytest
```

---

## Limitations

- **No streaming.** The `/ask` endpoint returns the full answer in one response. Streaming would require server-sent events and per-token forwarding from the LLM client.
- **CPU-only reranking.** The cross-encoder runs on CPU. For large candidate sets (`retrieval_multiplier × top_k > ~100`) latency grows linearly. Move to GPU or reduce the multiplier.
- **Agentic cost scales with critic loops.** Each loop adds two LLM calls (synthesizer + critic). With `max_critic_loops=3` this is up to 7 LLM calls per query.
- **Append-only ingestion.** There is no update or delete path. Re-ingesting a changed document creates duplicate chunks; drop and recreate the table to start clean.
- **HNSW index tuning not exposed.** The default `m=16, ef_construction=64` works well up to ~500k chunks. For larger corpora, expose these as settings and tune `ef_search` at query time.
- **golden_set.jsonl is hand-written.** Metric quality depends entirely on how well your questions and ground-truth answers cover the document corpus. Fabricated Q&A will produce meaningless numbers.
