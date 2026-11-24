# Enterprise RAG Platform

An AI-powered document question-answering platform built using **FastAPI, LangChain, ChromaDB, Google Gemini, and Retrieval-Augmented Generation (RAG)**.

Upload PDF documents, index their content, retrieve relevant information using advanced retrieval techniques, and generate context-aware answers with source references.

## Features

### Phase 1 – Core RAG
- PDF upload and document ingestion
- Automatic text extraction and chunking
- Hugging Face embedding generation
- ChromaDB vector storage
- Semantic similarity search
- Gemini-powered question answering
- Source citations and metadata enrichment
- Document listing and deletion APIs
- Production logging

### Phase 2 – Advanced RAG and Reliability
- Hybrid retrieval: dense vector and BM25 search
- Multi-query retrieval
- Context compression
- Cross-encoder reranking
- Streaming question-answering endpoint
- Retrieval evaluation pipeline
- API and reliability test suite
- Docker-based backend deployment
- Persistent ChromaDB storage
- Gemini quota and service-error handling

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| RAG Framework | LangChain |
| LLM | Google Gemini |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Reranking | Cross-Encoder |
| Testing | Pytest |
| Containerization | Docker |
| API Documentation | Swagger UI |

## Architecture

```text
PDF Upload
    |
    v
Document Loader
    |
    v
Text Chunking
    |
    v
Embedding Generation
    |
    v
ChromaDB + BM25
    |
    v
Multi-Query Retrieval
    |
    v
Context Compression
    |
    v
Cross-Encoder Reranking
    |
    v
Gemini LLM
    |
    v
Answer + Source References
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API health check |
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Generate an answer with sources |
| POST | `/ask/stream` | Stream a generated answer |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{filename}` | Delete a document and its indexed chunks |

## Retrieval Evaluation

The initial retrieval benchmark contains six queries.

| Metric | Result |
|---|---:|
| Successful queries | 6/6 |
| Average Precision@3 | 0.7778 |
| Source Recall@3 | 1.0000 |
| Mean Reciprocal Rank | 1.0000 |
| Average Retrieval Latency | 0.838 seconds |

These are preliminary source-level evaluation results, not human-judged semantic chunk relevance.

Detailed report: `docs/RETRIEVAL_EVALUATION.md`

## Testing

Run from the backend directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Latest recorded result: **22 passed, 2 non-blocking warnings**.

## Project Structure

```text
enterprise-rag-platform/
|
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── embeddings/
│   │   ├── llm/
│   │   ├── loaders/
│   │   ├── models/
│   │   ├── pipelines/
│   │   ├── preprocessing/
│   │   ├── rag/
│   │   ├── retriever/
│   │   ├── uploader/
│   │   ├── utils/
│   │   └── vectordb/
│   ├── evaluation/
│   ├── tests/
│   ├── sample_documents/
│   ├── requirements.txt
│   └── Dockerfile
|
├── docs/
│   ├── API_DOCUMENTATION.md
│   ├── RETRIEVAL_EVALUATION.md
│   ├── ADVANCED_RAG_ARCHITECTURE.md
│   └── PROJECT_OVERVIEW.md
|
└── README.md
```

## Local Setup

```powershell
cd backend

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn app.api.main:app --reload
```

Configure the required Gemini API key in your environment before using generation endpoints.

## API Documentation

Once the server is running:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

See `docs/API_DOCUMENTATION.md` for endpoint details.

## Project Status

**Version:** 1.0.0

- Phase 1 – Core RAG: Completed
- Phase 2 – Advanced RAG, evaluation and reliability: Technical verification completed
- Live Gemini generation: Final confirmation pending API quota availability
- Phase 3 – Agentic AI: Planned

## Future Enhancements

- LangGraph-based Agentic RAG
- Authentication and authorization
- CI/CD automation
- Cloud deployment
- Monitoring and observability
- React or Streamlit frontend
