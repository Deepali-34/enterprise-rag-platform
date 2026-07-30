from app.loaders.pdf_loader import load_pdf
from app.preprocessing.text_splitter import split_documents
from app.vectordb.chroma_db import add_documents


def ingest_document(pdf_path: str):
    """
    Complete ingestion pipeline.

    PDF
        ↓
    Loader
        ↓
    Splitter
        ↓
    ChromaDB
    """

    documents = load_pdf(pdf_path)

    total_chunks = add_documents(documents)

    return total_chunks