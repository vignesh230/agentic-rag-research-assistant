# Contributing

Thanks for your interest in contributing to Agentic RAG Research Assistant.

## Getting started

```bash
git clone https://github.com/vignesh230/agentic-rag-research-assistant.git
cd agentic-rag-research-assistant
pip install -e ".[dev]"
docker-compose up -d   # Postgres for integration tests
```

## Running tests

```bash
pytest -v                  # unit tests (no DB needed)
pytest -m integration      # needs live Postgres
```

All 70 tests must pass before submitting a PR.

## Making changes

1. Fork the repo and create a branch from `main`
2. Make your changes with clear, focused commits
3. Add or update tests as needed
4. Open a pull request — describe what you changed and why

## What's welcome

- Bug fixes
- Evaluation improvements (new metrics, better golden-set entries)
- Documentation clarifications
- Performance improvements to the retrieval pipeline

## What to discuss first

Open an issue before starting work on new RAG modes, major refactors, or new dependencies — to avoid duplicate effort.

## Code style

```bash
ruff check src/ tests/
mypy src/
```

Both must pass cleanly. No new `type: ignore` comments without explanation.
