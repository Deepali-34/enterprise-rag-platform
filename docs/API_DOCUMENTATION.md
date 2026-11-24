# Enterprise RAG Platform — API Documentation

**Version:** 1.0.0
**Framework:** FastAPI
**API Type:** REST API
**Content Types:** JSON, Multipart Form Data, Plain Text

## 1. Overview

The Enterprise RAG Platform provides APIs for PDF ingestion, document indexing, retrieval-augmented question answering, streaming responses, and document management.

The backend uses ChromaDB for vector storage, a hybrid retrieval pipeline, and Google Gemini for answer generation.

## 2. Running the API

From the backend directory:

```powershell
uvicorn app.api.main:app --reload
```

Default address:

`http://localhost:8000`

Interactive documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 3. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API health check |
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question and receive a structured answer |
| POST | `/ask/stream` | Receive a streamed answer |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{filename}` | Delete an indexed document |

## 4. Endpoint Details

### 4.1 Health Check

**Endpoint:** `GET /`

Checks whether the API is running.

Example response:

```json
{
  "message": "Enterprise RAG Platform API is running!"
}
```

**Status:** `200 OK`

### 4.2 Upload PDF

**Endpoint:** `POST /upload`

Uploads a PDF, extracts its text, splits it into chunks, generates embeddings, and indexes the chunks.

**Request type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|---|---|---|---|
| file | File | Yes | PDF document |

Example:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

Example response:

```json
{
  "message": "Document indexed successfully",
  "filename": "document.pdf",
  "chunks_added": 15
}
```

Possible status codes:

- `200` — Upload and indexing succeeded.
- `400` — Invalid upload or document validation error.
- `409` — Duplicate document.
- `422` — Request validation failed.
- `500` — Unexpected upload or ingestion failure.

### 4.3 Ask a Question

**Endpoint:** `POST /ask`

Accepts a question and returns an answer generated using retrieved document context.

Request body:

```json
{
  "question": "Explain the main concepts of RAG architecture.",
  "session_id": "default"
}
```

| Parameter | Type | Required | Default |
|---|---|---|---|
| question | String | Yes | — |
| session_id | String | No | `default` |

Example response:

```json
{
  "question": "Explain the main concepts of RAG architecture.",
  "answer": "Retrieval-Augmented Generation combines document retrieval with language model generation...",
  "sources": [
    {
      "filename": "RAG_LLM_Architecture_Guide.pdf",
      "page": 2
    }
  ]
}
```

The response contains the original question, generated answer, and source references.

Possible status codes:

- `200` — Answer generated successfully.
- `422` — Invalid request body.
- `429` — Gemini API quota or rate limit exceeded.
- `503` — AI service temporarily unavailable.
- `500` — Unexpected internal processing error.

**Quota handling:** When Gemini's quota is exhausted, the API returns HTTP `429` with a clear error message. This is an external provider limitation, not a retrieval failure.

### 4.4 Streaming Question Answering

**Endpoint:** `POST /ask/stream`

Returns generated answer text as a streaming response.

Request body:

```json
{
  "question": "What is semantic search?",
  "session_id": "default"
}
```

Example:

```bash
curl -N -X POST "http://localhost:8000/ask/stream" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is semantic search?\",\"session_id\":\"default\"}"
```

Response content type:

`text/plain`

Unlike `/ask`, this endpoint streams text instead of returning a structured JSON answer.

Errors that occur after streaming begins may be represented within the stream rather than as a new HTTP response status.

### 4.5 List Indexed Documents

**Endpoint:** `GET /documents`

Returns the currently indexed documents and their chunk counts.

Example response:

```json
{
  "documents": [
    {
      "filename": "document.pdf",
      "chunks": 15
    }
  ]
}
```

**Status:** `200 OK`

### 4.6 Delete an Indexed Document

**Endpoint:** `DELETE /documents/{filename}`

Removes the specified document's indexed chunks and its corresponding stored PDF.

Example:

```http
DELETE /documents/document.pdf
```

Example response:

```json
{
  "message": "Document deleted successfully.",
  "filename": "document.pdf",
  "chunks_deleted": 15
}
```

Possible status codes:

- `200` — Document deleted successfully.
- `404` — Document not found.

**Warning:** This operation deletes both the stored PDF and its corresponding indexed chunks.

## 5. Error Handling

| Status | Meaning |
|---|---|
| 200 | Request completed successfully |
| 400 | Invalid upload or document validation error |
| 404 | Document not found |
| 409 | Duplicate upload |
| 422 | Request validation failed |
| 429 | Gemini quota or rate limit exceeded |
| 500 | Unexpected internal error |
| 503 | AI service temporarily unavailable |

Example quota error:

```json
{
  "detail": "Gemini API quota exceeded. Please retry after your quota resets."
}
```

## 6. Technology Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| Programming Language | Python |
| RAG Framework | LangChain |
| Language Model | Google Gemini |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Reranking | Cross-Encoder |
| Testing | Pytest |
| Containerization | Docker |

## 7. API Workflow

1. Upload a PDF using `/upload`.
2. Extract document text and split it into chunks.
3. Generate embeddings and store document chunks.
4. Submit a question through `/ask` or `/ask/stream`.
5. Retrieve relevant context through the retrieval pipeline.
6. Compress and rerank retrieved context.
7. Generate a response using Gemini.
8. Return the answer with source references, or stream the answer.

## 8. Testing

Run the automated backend tests:

```powershell
python -m pytest -q
```

Check API syntax:

```powershell
python -m py_compile app/api/main.py
```

The interactive Swagger UI is available at `/docs`.

## 9. Project Status

**API Version:** 1.0.0

The documented routes reflect the current FastAPI implementation. Example answers, filenames, and chunk counts are illustrative and depend on the indexed documents and runtime behavior.

The `/ask` endpoint's live generation availability depends on the configured Gemini API quota.
