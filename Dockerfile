# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.13-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /build

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

RUN python -m venv /opt/refaat \
    && /opt/refaat/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/refaat/bin/python -m pip wheel --wheel-dir /wheels ".[server]" \
    && /opt/refaat/bin/python -m pip install --no-index --find-links=/wheels "refaat[server]"

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="Refaat" \
      org.opencontainers.image.description="Retrieval & Evidence Foundation for Adaptive Agentic Tooling" \
      org.opencontainers.image.source="https://github.com/oronts/refaat" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PATH=/opt/refaat/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/refaat

RUN groupadd --system --gid 10001 refaat \
    && useradd --system --uid 10001 --gid 10001 --home-dir /home/refaat --create-home refaat \
    && mkdir -p /app/data /home/refaat/.cache \
    && chown -R 10001:10001 /app /home/refaat

WORKDIR /app
COPY --from=builder /opt/refaat /opt/refaat
COPY --chown=10001:10001 alembic.ini ./
COPY --chown=10001:10001 migrations ./migrations

USER 10001:10001
EXPOSE 8000


CMD ["refaat-api"]
