FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

COPY ai/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN mkdir -p /app/.cache/huggingface /app/ai/model/tag_classification

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "ai.app:app", "--host", "0.0.0.0", "--port", "8000"]
