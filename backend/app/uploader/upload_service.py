import os
import shutil

from fastapi import HTTPException

from app.pipelines.ingestion_pipeline import ingest_document

UPLOAD_FOLDER = "sample_documents"


def upload_pdf(file):
    """
    Save PDF and ingest it into ChromaDB.
    Prevent duplicate uploads.
    """

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Prevent duplicate uploads
    if os.path.exists(file_path):
        raise HTTPException(
            status_code=400,
            detail=f"'{file.filename}' already exists."
        )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = ingest_document(file_path)

    return {
        "filename": file.filename,
        "chunks": chunks
    }