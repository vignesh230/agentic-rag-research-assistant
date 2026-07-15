# Design Decisions

Each entry lists the choice made, the alternatives that were seriously considered, and a blank reasoning section for you to fill in after reflecting on the tradeoffs. Filling this in before an interview is highly recommended — recruiters ask about these directly.

---

## 1. Vector store: pgvector over a dedicated vector DB

**Choice:** Postgres + pgvector extension.

**Alternatives considered:**
- FAISS (in-process, no persistence)
- Qdrant (purpose-built vector DB, REST API)
- Weaviate (multi-modal, GraphQL interface)
- Chroma (lightweight, Python-native)

**Reasoning:**

---

## 2. HNSW index over IVFFlat

**Choice:** HNSW (`vector_cosine_ops`, m=16, ef_construction=64).

**Alternatives considered:**
- IVFFlat — requires a training step (k-means on existing vectors); cannot be created on an empty table, which complicates cold-start setup.
- Exact nearest-neighbour (no index) — correct but O(n) per query; unacceptable above ~10k chunks.

**Reasoning:**

---

## 3. Embedding model: all-MiniLM-L6-v2 (384-dim)

**Choice:** `sentence-transformers/all-MiniLM-L6-v2`.

**Alternatives considered:**
- `all-mpnet-base-v2` (768-dim) — higher quality, 2× dimension cost in storage and dot-product time.
- NVIDIA nv-embedqa-e5-v5 (1024-dim) — top-tier quality, requires an external API call per embed.
- OpenAI `text-embedding-3-small` — API-based, adds cost and latency, creates a second vendor dependency.

**Reasoning:**

---

## 4. Default chunking strategy: recursive

**Choice:** `RecursiveCharacterTextSplitter` with `add_start_index=True`.

**Alternatives considered:**
- Fixed (sliding window) — ignores semantic boundaries; a sentence split mid-paragraph loses context.
- Sentence (NLTK groups) — better than fixed but struggles with code blocks and tables.
- Semantic (embedding-based boundaries) — highest quality, but requires an embedding call per split decision, making ingestion 5-10× slower.

**Reasoning:**

---

## 5. Flat TypedDict for AgentState

**Choice:** A single flat `TypedDict` (`AgentState`) for all LangGraph node state.

**Alternatives considered:**
- Nested Pydantic models — easier validation but harder to serialize for logging and LangGraph's internal checkpointing.
- Dataclass — no JSON-serialization out of the box.
- Dict without schema — no type checking; silent key typos become runtime bugs.

**Reasoning:**

---

## 6. No separate router node in the agentic graph

**Choice:** The critic node doubles as the router — it either finalises the answer or triggers re-retrieval.

**Alternatives considered:**
- Dedicated `router` node after planner: decides whether to retrieve or answer directly. Saves retrieval on simple questions, but adds an LLM call on every query to make a decision the critic would make anyway after synthesis.
- Conditional entry: skip retrieval if the question is in a cache. Adds complexity for a feature not in scope.

**Reasoning:**

---

## 7. Chunks accumulate across critic loops (not replaced)

**Choice:** `retrieved_chunks` grows with each critic loop; the rewrite query appends new evidence rather than discarding prior chunks.

**Alternatives considered:**
- Replace chunks on each loop — the synthesizer starts fresh each time, which could help if the original retrieval was completely off-track, but discards genuinely useful context that happened to be retrieved early.
- Fixed retrieval window (no looping) — simpler, but the whole point of the critic is to recover from gaps; without accumulation the recovery is incomplete.

**Reasoning:**

---

## 8. Module-level model cache for embedder and reranker

**Choice:** `_MODEL_CACHE: dict[str, Model]` at module level in `embedder.py` and `reranker.py`.

**Alternatives considered:**
- Singleton class with `@classmethod` — same effect, more boilerplate.
- `functools.lru_cache` on the loader function — works, but `lru_cache` adds a layer of indirection that's harder to inspect in tests.
- Reload per request — correct but adds 1-2s on every call while the weights load into memory.

**Reasoning:**

---

## 9. Claude Haiku as RAGAS judge LLM

**Choice:** `claude-haiku-4-5-20251001` for RAGAS metric evaluation (faithfulness, relevancy, etc.).

**Alternatives considered:**
- Claude Sonnet (same family, higher capability) — more accurate judgements, ~5× cost increase on eval runs.
- GPT-4o-mini — would require an OpenAI API key, adding a second vendor.
- Local model (Ollama + Llama) — zero cost, but RAGAS's judge prompts are calibrated for instruction-tuned frontier models; quality degrades significantly on smaller local models.

**Reasoning:**

---

## 10. Versioned prompts in YAML files

**Choice:** All prompts live in `prompts/*.yaml` with a `version` field logged on every generation call.

**Alternatives considered:**
- Hardcoded f-strings in Python — fast to write, impossible to diff across experiments.
- Jinja2 templates — more powerful templating but adds a dependency for no benefit given the current prompt complexity.
- LangChain `PromptTemplate` objects — vendor-coupled, harder to version-control as plain text.
- Database-backed prompt registry — overkill for a single-service research assistant; adds operational complexity without a concrete need.

**Reasoning:**

---

## 11. Cross-encoder model: ms-marco-MiniLM-L-6-v2

**Choice:** `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking.

**Alternatives considered:**
- `cross-encoder/ms-marco-MiniLM-L-12-v2` — higher accuracy, ~2× inference time on CPU.
- `BAAI/bge-reranker-base` — strong multilingual support, heavier model.
- Cohere Rerank API — top-tier quality, adds cost and a third vendor dependency.

**Reasoning:**

---

## 12. Retrieval multiplier for reranked mode (default 4×)

**Choice:** Retrieve `top_k × retrieval_multiplier` candidates before reranking, default multiplier = 4.

**Alternatives considered:**
- Fixed candidate count (e.g. always 20) — ignores top_k, so a user requesting top_k=2 still gets the same candidate pool as top_k=10.
- Multiplier of 2× — too narrow; the cross-encoder can't compensate for a bad first-stage retrieval pool.
- Multiplier of 8× — diminishing returns on quality; latency grows linearly with candidates on CPU.

**Reasoning:**
