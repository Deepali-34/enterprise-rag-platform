FROM python:3.11-slim

# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Keep HuggingFace cache inside the container volume
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Application directory
# ---------------------------------------------------------

WORKDIR /app

# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# Copy application
# ---------------------------------------------------------

COPY app ./app

# Copy directories required by the application
COPY sample_documents ./sample_documents
COPY data ./data

# Create runtime directories
RUN mkdir -p \
    chroma_storage \
    uploaded_documents \
    logs \
    .cache/huggingface

# ---------------------------------------------------------
# Expose FastAPI port
# ---------------------------------------------------------

EXPOSE 8000

# ---------------------------------------------------------
# Start FastAPI
# ---------------------------------------------------------

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]