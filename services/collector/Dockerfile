FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN pip install --no-cache-dir uv==0.9.5

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 agenttrace && chown -R agenttrace:agenttrace /app
USER agenttrace

EXPOSE 8000
CMD ["agenttrace", "collector", "--host", "0.0.0.0", "--port", "8000", "--log-format", "json"]
