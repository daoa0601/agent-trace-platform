FROM python:3.11-slim

RUN useradd --create-home --uid 10001 task && mkdir /workspace && chown task:task /workspace
USER task
WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "--version"]
