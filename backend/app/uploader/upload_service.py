
from pathlib import Path
import shutil

from fastapi import HTTPException

from app.pipelines.ingestion_pipeline import ingest_document
from app.vectordb.chroma_db import delete_document

UPLOAD_FOLDER = Path("sample_documents")


def upload_pdf(file):
    """Save an uploaded PDF and ingest it into ChromaDB."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    filename = Path(file.filename.replace("\\", "/")).name

    if not filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_FOLDER / filename

    if file_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"'{filename}' already exists."
        )

    try:
        file.file.seek(0)

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = ingest_document(str(file_path))

        return {
            "filename": filename,
            "chunks": chunks
        }

    except HTTPException:
        raise

    except Exception:
        try:
            delete_document(filename)
        except Exception:
            pass

        if file_path.exists():
            file_path.unlink()

        raise
