from fastapi import APIRouter, HTTPException
from pathlib import Path
import os

from app.vectordb.chroma_db import (
    list_documents,
    delete_document,
)

router = APIRouter(
    tags=["Document Management"]
)

UPLOAD_FOLDER = "sample_documents"


@router.get("/documents")
def get_documents():

    return {
        "documents": list_documents()
    }


@router.delete("/documents/{filename}")
def remove_document(filename: str):

    file_path = Path(UPLOAD_FOLDER) / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    deleted_chunks = delete_document(filename)

    os.remove(file_path)

    return {
        "message": "Document deleted successfully.",
        "filename": filename,
        "chunks_deleted": deleted_chunks
    }