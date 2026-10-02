FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/classifier/pyproject.toml,target=packages/classifier/pyproject.toml \
    uv sync --locked --no-install-project --package classifier

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY packages/classifier packages/classifier

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --package classifier

FROM python:3.14-slim

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1

WORKDIR /app/packages/classifier

CMD ["fastapi", "run", "--port", "8000", "--workers", "4"]