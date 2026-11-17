# App image. The model is served by Ollama (see docker-compose.yml), which
# keeps the model server and the RAG app separable — swap Ollama for vLLM on
# the GPU target without touching this image.
FROM python:3.12-slim

WORKDIR /app

# System deps are minimal: numpy/pandas wheels cover the rest.
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY eval ./eval
COPY web ./web

RUN pip install --no-cache-dir .

ENV OLLAMA_HOST=http://ollama:11434
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "neet_rag.app:app", "--host", "0.0.0.0", "--port", "8000"]