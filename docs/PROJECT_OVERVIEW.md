# Enterprise RAG Platform – Project Overview

## 1. Project Objective

Build an AI-powered knowledge platform that allows users to upload PDF documents and ask questions using Retrieval-Augmented Generation (RAG).

## 2. Technology Stack

- Python
- FastAPI
- LangChain
- ChromaDB
- Google Gemini
- Docker
- Sentence Transformers
- BM25 Retrieval

## 3. Core Workflow

1. Upload PDF documents.
2. Extract and preprocess document text.
3. Split text into smaller chunks.
4. Generate vector embeddings.
5. Store embeddings in ChromaDB.
6. Retrieve relevant document chunks.
7. Compress and rerank retrieved context.
8. Send context to Google Gemini.
9. Generate an answer with source information.

## 4. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | API health check |
| GET | /documents | List indexed documents |
| POST | /upload | Upload a PDF |
| POST | /ask | Ask a question |
| POST | /ask/stream | Stream an answer |
| DELETE | /documents/{filename} | Delete a document |

## 5. Advanced RAG Components

- Hybrid retrieval
- BM25 retrieval
- Multi-query retrieval
- Reciprocal Rank Fusion
- Context compression
- Reranking
- Conversation memory

## 6. Deployment

The backend is containerized using Docker and exposed on port 8000.

## 7. Project Status

Core document ingestion, retrieval, and streaming answer generation have been tested. Further improvements include comprehensive API testing, retrieval evaluation, and production deployment.
