
FROM python:3.11-slim

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HuggingFace model cache
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Application directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir rank-bm25==0.2.2
RUN pip install --no-cache-dir python-multipart

# Copy application source code only
COPY app ./app

# Create runtime directories
RUN mkdir -p \
    chroma_storage \
    uploaded_documents \
    sample_documents \
    logs \
    .cache/huggingface

# FastAPI port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
