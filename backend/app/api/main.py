from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.api.upload import router as upload_router
from app.rag.rag_chain import ask_rag
from app.models.response_models import QuestionResponse
from app.utils.logger import logger
from app.uploader.upload_service import upload_pdf

app = FastAPI(
    title="Enterprise RAG Platform",
    description="AI-powered document question answering system",
    version="1.0.0",
)

# Register Document Management API
app.include_router(upload_router)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():

    logger.info("Root endpoint accessed")

    return {
        "message": "Enterprise RAG Platform API is running!"
    }


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    logger.info(f"Question: {request.question}")

    try:

        result = ask_rag(request.question)

        logger.info("Answer generated successfully.")

        return QuestionResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:

        logger.exception("Error while processing question")

        raise HTTPException(
            status_code=500,
            detail=str(e)
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

    except Exception as e:

        logger.exception("Upload failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )