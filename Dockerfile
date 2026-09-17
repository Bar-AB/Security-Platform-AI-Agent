FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agent ./agent
COPY api ./api
COPY mcp_client ./mcp_client
COPY mock_server ./mock_server
COPY rag ./rag
COPY docs ./docs
COPY main.py .
