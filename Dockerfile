FROM python:3.11-slim

WORKDIR /app

# Install build deps in one layer, then drop them after pip install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.cfg ./
# Install deps before copying source so Docker cache survives src-only changes.
RUN pip install --no-cache-dir -e "." && pip install --no-cache-dir uvicorn[standard]

COPY src/ ./src/
COPY prompts/ ./prompts/

ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "rag_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
