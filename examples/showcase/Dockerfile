# Single self-contained image for the Oronts Knowledge Terminal.
# Build context is the repository root:
#   docker build -f examples/showcase/Dockerfile -t oronts-terminal .
#
# The image runs the whole app in one container: SQLite plus in-process vector search,
# in-memory rate limiter, no external service. The only external dependency is the
# OpenAI REST API, and only when REFAAT_OPENAI_API_KEY is set.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    REFAAT_SHOWCASE_DATA_DIR=/data \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install refaat from source with the framework integration extra, plus langgraph,
# fastapi and uvicorn (fastapi and uvicorn are already refaat core dependencies).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[integrations]" "langgraph>=1.0.10,<2"

# The showcase app is not part of the refaat package, so it is copied and run from PYTHONPATH.
COPY examples/showcase ./showcase

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown -R app:app /data /app
USER app

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/healthz').read()"

CMD ["sh", "-c", "uvicorn showcase.app.main:create_app --factory --host 0.0.0.0 --port ${PORT}"]
