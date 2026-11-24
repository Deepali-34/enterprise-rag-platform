
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.upload import router as upload_router
from app.rag.rag_chain import ask_rag, stream_rag
from app.models.response_models import QuestionResponse
from app.utils.logger import logger
from app.uploader.upload_service import upload_pdf


app = FastAPI(
    title="Enterprise RAG Platform",
    description="AI-powered document question answering system",
    version="1.0.0",
)

app.include_router(upload_router)


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


def is_gemini_rate_limited(error: Exception) -> bool:
    """Identify Gemini API quota and rate-limit errors."""

    status_code = getattr(error, "status_code", None)
    error_code = getattr(error, "code", None)
    error_message = str(error).upper()

    return (
        status_code == 429
        or error_code == 429
        or "429" in error_message
        or "RESOURCE_EXHAUSTED" in error_message
    )


def is_gemini_unavailable(error: Exception) -> bool:
    """Identify temporary Gemini service availability errors."""

    status_code = getattr(error, "status_code", None)
    error_code = getattr(error, "code", None)
    error_message = str(error).upper()

    return (
        status_code == 503
        or error_code == 503
        or "503" in error_message
        or "UNAVAILABLE" in error_message
    )


@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {"message": "Enterprise RAG Platform API is running!"}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    logger.info(
        f"Question: {request.question} | "
        f"Session: {request.session_id}"
    )

    try:
        result = ask_rag(
            question=request.question,
            session_id=request.session_id
        )

        logger.info("Answer generated successfully.")

        return QuestionResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        logger.error(
            "Error while processing question",
            exc_info=True
        )

        if is_gemini_rate_limited(e):
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota exceeded. Please retry after your quota resets."
            )

        if is_gemini_unavailable(e):
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable. Please try again later."
            )

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question due to an internal error."
        )


@app.post("/ask/stream")
def stream_question(request: QuestionRequest):
    logger.info(
        f"Streaming question: {request.question} | "
        f"Session: {request.session_id}"
    )

    return StreamingResponse(
        stream_rag(
            question=request.question,
            session_id=request.session_id
        ),
        media_type="text/plain"
    )


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    try:
        result = upload_pdf(file)

        logger.info(f"Uploaded: {file.filename}")

        return {
            "message": "Document indexed successfully",
            "filename": result["filename"],
            "chunks_added": result["chunks"]
        }

    except HTTPException:
        raise

    except Exception:
        logger.error(
            "Error while uploading document",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail="Document upload failed."
        )
