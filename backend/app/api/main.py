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


# ---------------------------------------------------------
# Routers
# ---------------------------------------------------------

app.include_router(upload_router)


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    logger.info("Root endpoint accessed")

    return {
        "message": "Enterprise RAG Platform API is running!"
    }


# ---------------------------------------------------------
# Standard RAG Endpoint
# ---------------------------------------------------------

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

        logger.info(
            "Answer generated successfully."
        )

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

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# Streaming RAG Endpoint
# ---------------------------------------------------------

@app.post("/ask/stream")
def stream_question(request: QuestionRequest):

    logger.info(
        f"Streaming question: {request.question} | "
        f"Session: {request.session_id}"
    )

    try:

        return StreamingResponse(
            stream_rag(
                question=request.question,
                session_id=request.session_id
            ),
            media_type="text/plain"
        )

    except Exception as e:

        logger.error(
            "Error while streaming response",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# Upload Endpoint
# ---------------------------------------------------------

@app.post("/upload")
def upload_document(
    file: UploadFile = File(...)
):

    try:

        result = upload_pdf(file)

        logger.info(
            f"Uploaded: {file.filename}"
        )

        return {
            "message": "Document indexed successfully",
            "filename": result["filename"],
            "chunks_added": result["chunks"]
        }

    except Exception as e:

        logger.error(
            "Error while uploading document",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )