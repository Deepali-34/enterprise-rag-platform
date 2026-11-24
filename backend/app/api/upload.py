from fastapi import APIRouter, HTTPException
from pathlib import Path

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

    # Reject path traversal and invalid document filenames.
    safe_filename = filename.replace("\\", "/")

    if (
        Path(safe_filename).name != safe_filename
        or safe_filename in {"", ".", ".."}
        or Path(safe_filename).suffix.lower() != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid document filename."
        )

    file_path = Path(UPLOAD_FOLDER) / safe_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    deleted_chunks = delete_document(safe_filename)

    file_path.unlink()

    return {
        "message": "Document deleted successfully.",
        "filename": safe_filename,
        "chunks_deleted": deleted_chunks
    }