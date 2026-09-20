FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /code

COPY uv.lock /code/uv.lock
COPY pyproject.toml /code/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable

COPY src/serving/classifier /code/src/serving/classifier

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

CMD ["fastapi", "run", "classifier/main.py", "--port", "8080"]