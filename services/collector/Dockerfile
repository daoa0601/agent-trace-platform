FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN pip install --no-cache-dir uv==0.9.5

# uv.lock resolves benchmark-audit-core as ../benchmark-audit-core. Supply the sibling as a named
# build context (compose.yaml does this; with plain docker build pass
# --build-context benchmark-audit-core=../benchmark-audit-core).
COPY --from=benchmark-audit-core pyproject.toml README.md LICENSE /benchmark-audit-core/
COPY --from=benchmark-audit-core src /benchmark-audit-core/src
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 agenttrace && chown -R agenttrace:agenttrace /app
USER agenttrace

EXPOSE 8000
CMD ["agenttrace", "collector", "--host", "0.0.0.0", "--port", "8000", "--log-format", "json"]
