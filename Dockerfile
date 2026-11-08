# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv and app code.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 pipecast
WORKDIR /app
COPY --from=build --chown=pipecast:pipecast /app /app
# The engine cache is written here at first start; the non-root user must own it.
RUN mkdir -p /app/data && chown pipecast:pipecast /app/data
USER pipecast
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8854 PIPECAST_CACHE=/app/data/forecaster.pkl
EXPOSE 8854
# First start generates the CRM, replays 6 quarters, and trains 7 models: about 20 to 60 seconds.
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8854/healthz', timeout=2)"
CMD ["pipecast", "serve"]
