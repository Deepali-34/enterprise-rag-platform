# Enterprise RAG Platform

An enterprise-grade AI Knowledge Platform built using **FastAPI, LangChain, ChromaDB, Google Gemini, and Retrieval-Augmented Generation (RAG)**.

The platform enables users to upload PDF documents, create vector embeddings, retrieve relevant document chunks, and generate context-aware answers using a Large Language Model (LLM) with source citations.

---

# Features

## ✅ Phase 1 – Core Enterprise RAG (Completed)

- PDF Document Upload & Indexing
- Automatic Text Chunking
- Vector Embedding Generation
- ChromaDB Vector Database Integration
- Semantic Similarity Search
- Retrieval-Augmented Generation (RAG)
- Google Gemini LLM Integration
- FastAPI REST APIs
- Document Management APIs
- Source Citation Support
- Metadata Enrichment
- Production Logging
- Swagger API Documentation

---

# Tech Stack

### Backend
- Python
- FastAPI
- LangChain

### AI & LLM
- Google Gemini
- Hugging Face Embeddings
- sentence-transformers/all-MiniLM-L6-v2

### Vector Database
- ChromaDB

### Document Processing
- LangChain PDF Loader
- Recursive Character Text Splitter

### Development Tools
- Swagger UI
- Uvicorn
- Git
- GitHub

---

# Current API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | API Health Check |
| POST | `/upload` | Upload and index PDF documents |
| POST | `/ask` | Ask questions from uploaded documents |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{filename}` | Delete indexed documents |

---

# Project Architecture

```
Enterprise RAG Platform
│
├── FastAPI REST API
├── PDF Upload Service
├── Document Loader
├── Text Chunking
├── Embedding Generator
├── ChromaDB Vector Store
├── Semantic Retriever
├── Gemini LLM
└── Source Citation Engine
```

---

# Project Structure

```
backend/
│
├── app/
│   ├── api/
│   ├── embeddings/
│   ├── llm/
│   ├── loaders/
│   ├── models/
│   ├── pipelines/
│   ├── preprocessing/
│   ├── rag/
│   ├── retriever/
│   ├── uploader/
│   ├── utils/
│   └── vectordb/
│
├── sample_documents/
├── tests/
├── requirements.txt
└── README.md
```

---

# Workflow

```
Upload PDF
      │
      ▼
PDF Loader
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
      │
      ▼
ChromaDB Storage
      │
      ▼
User Question
      │
      ▼
Semantic Retrieval
      │
      ▼
Gemini LLM
      │
      ▼
Answer + Source Citations
```

---

# Current Progress

## ✅ Phase 1 — Core Enterprise RAG (Completed)

- Document Upload Pipeline
- Vector Database Integration
- Semantic Retrieval
- RAG Question Answering
- Document Management APIs
- Metadata Enrichment
- Source Citations
- Production Logging

---

## 🚧 Phase 2 — Advanced RAG (In Progress)

Planned Features:

- Hybrid Search (Dense + BM25)
- Multi-Query Retrieval
- Context Compression
- Re-ranking
- Conversational Memory
- Streaming Responses
- Evaluation Pipeline

---

# Future Enhancements

- LangGraph Agentic RAG
- Authentication & Authorization
- Docker Deployment
- CI/CD Pipeline
- Cloud Deployment (AWS/GCP/Azure)
- Monitoring & Observability
- Frontend Dashboard (Streamlit/React)

---

# Installation

```bash
git clone https://github.com/Deepali-34/enterprise-rag-platform.git

cd enterprise-rag-platform/backend

pip install -r requirements.txt

uvicorn app.api.main:app --reload
```

---

# API Documentation

After starting the server:

```
http://127.0.0.1:8000/docs
```

---

# Project Status

**Current Version:** Phase 1 Completed ✅

**Next Milestone:** Advanced Enterprise RAG Pipeline 🚀
