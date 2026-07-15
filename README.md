# Agentic RAG Research Assistant

A production-grade RAG service with three configurable retrieval modes, a LangGraph self-critique loop, and a full RAGAS evaluation harness. Built as a portfolio project for ML/AI engineering interviews.

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        D[Documents] --> C[Chunker\nfixed / sentence / recursive]
        C --> E[Embedder\nall-MiniLM-L6-v2]
        E --> PG[(Postgres\n+ pgvector\nHNSW index)]
    end

    subgraph API["FastAPI  /ask"]
        REQ[POST /ask\nmode: naive | reranked | agentic]
    end

    subgraph Naive["Mode: naive"]
        N1[Embed query] --> N2[similarity_search top-k]
        N2 --> N3[Stuff context]
        N3 --> N4[Claude → answer + citations]
    end

    subgraph Reranked["Mode: reranked"]
        R1[Embed query] --> R2[similarity_search top-k × multiplier]
        R2 --> R3[CrossEncoder rerank → top-k]
        R3 --> R4[Claude → answer + citations]
    end

    subgraph Agentic["Mode: agentic  (LangGraph)"]
        direction TB
        A1[planner\ndecompose question] --> A2[retrieve\nembed + search + dedup]
        A2 --> A3[synthesizer\ndraft answer]
        A3 --> A4{critic\ngrounded?}
        A4 -- supported / max loops --> A5[final answer]
        A4 -- rewrite query --> A2
    end

    subgraph Eval["Evaluation harness"]
        G[golden_set.jsonl] --> H[runner\nall 3 modes]
        H --> I[RAGAS metrics\nfaithfulness · relevancy\nprecision · recall]
        H --> J[latency p50/p95\ncost per query]
        I & J --> K[markdown table]
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
| LLM | Anthropic Claude (claude-sonnet-4-6) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (384-dim) |
| Reranking | sentence-transformers CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Vector store | Postgres + pgvector (HNSW index) |
| Evaluation | RAGAS 0.4 |
| Observability | structlog (JSON) + per-node traces |
| CI | GitHub Actions |

---

## Setup

### Prerequisites

- Python 3.11
- Docker + Docker Compose
- Anthropic API key

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
cp .env.example .env          # then set ANTHROPIC_API_KEY
```

Or export directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Key settings (all overridable via env vars):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
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
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
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

### 3. Ablation table

*Run the harness to populate this table. Values below are placeholders.*

| Mode | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Latency p50 (ms) | Latency p95 (ms) | Cost/query ($) |
|------|:---:|:---:|:---:|:---:|---:|---:|---:|
| naive | — | — | — | — | — | — | — |
| reranked | — | — | — | — | — | — | — |
| agentic | — | — | — | — | — | — | — |

---

## Tests

```bash
pytest --ignore=tests/test_pipeline.py -v   # unit tests (no DB needed)
pytest -m integration                        # needs live Postgres
```

65 tests. Integration tests are skipped by default and marked `@pytest.mark.integration`.

---

## Project structure

```
src/rag_agent/
├── api/            # FastAPI app, schemas, deps, /ask route
├── db/             # pgvector client (HNSW, cosine similarity)
├── eval/           # RAGAS harness: loader, runner, metrics, CLI
├── graph/          # LangGraph: state, nodes (planner/retrieve/synthesizer/critic), graph
├── ingestion/      # chunker, embedder, pipeline CLI
├── rag/            # naive.py, reranked.py, agentic.py, prompt_loader.py
├── logging_config.py
└── settings.py     # all ablation knobs in one place

prompts/            # versioned YAML prompts (naive_rag, planner, synthesizer, critic)
data/               # golden_set.jsonl (you write), eval_thresholds.json (you fill)
.github/workflows/  # CI: pytest + eval schema gate
```

---

## Limitations

- **No streaming.** The `/ask` endpoint returns the full answer in one response. Streaming would require server-sent events and per-token forwarding from the Anthropic client.
- **CPU-only reranking.** The cross-encoder runs on CPU. For large candidate sets (retrieval_multiplier × top_k > ~100) latency grows linearly. Move to GPU or reduce the multiplier.
- **Agentic cost scales with critic loops.** Each loop adds two Claude calls (synthesizer + critic). With `max_critic_loops=3` and a hard question this is up to ~7 Claude calls per query.
- **Append-only ingestion.** There is no update or delete path. Re-ingesting a changed document creates duplicate chunks; drop and recreate the table to start clean.
- **HNSW index tuning not exposed.** The default `m=16, ef_construction=64` works well up to ~500k chunks. For larger corpora, expose these as settings and tune `ef_search` at query time.
- **RAGAS judge cost.** Running the evaluation harness makes LLM calls for every metric on every golden-set entry. Budget ~$0.01–0.05 per question depending on context length.
- **golden_set.jsonl is hand-written.** Metric quality depends entirely on how well your questions and ground-truth answers cover the document corpus. Fabricated Q&A will produce meaningless numbers.
