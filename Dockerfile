# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13-python3.12-trixie-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
# Dependencies first, so code changes do not invalidate the dependency layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-trixie
RUN useradd --create-home --uid 10001 agent
WORKDIR /app
COPY --from=build --chown=agent:agent /app/.venv /app/.venv
# Scenario fixtures and offline scripts, so the container can run evals without a cluster.
COPY --chown=agent:agent scenarios ./scenarios
COPY --chown=agent:agent evals/scripted ./evals/scripted
RUN mkdir -p traces evals/results && chown agent:agent /app traces evals/results
USER agent
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
ENTRYPOINT ["aiops"]
CMD ["--help"]
