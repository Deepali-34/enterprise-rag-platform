from pathlib import Path
from collections import Counter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.loaders.multi_pdf_loader import load_multiple_pdfs
from app.preprocessing.text_splitter import split_documents


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

PERSIST_DIRECTORY = "chroma_storage"


def get_vector_db():
    """
    Open existing Chroma database.
    """

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )


def create_vector_database():
    """
    Create vector database from all PDFs.
    """

    pdf_folder = Path("sample_documents")

    documents = load_multiple_pdfs(pdf_folder)

    chunks = split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_db


def add_documents(documents):
    """
    Add new documents to existing Chroma database.
    """

    chunks = split_documents(documents)

    db = get_vector_db()

    db.add_documents(chunks)

    return len(chunks)


def list_documents():
    """
    Return all indexed documents with chunk counts.
    """

    db = get_vector_db()

    data = db.get(include=["metadatas"])

    if not data or "metadatas" not in data:
        return []

    filenames = []

    for metadata in data["metadatas"]:
        if metadata and "source" in metadata:
            filenames.append(Path(metadata["source"]).name)

    counts = Counter(filenames)

    return [
        {
            "filename": filename,
            "chunks": count
        }
        for filename, count in counts.items()
    ]


def delete_document(filename):
    """
    Delete all chunks belonging to a document.
    """

    db = get_vector_db()

    data = db.get(include=["metadatas"])

    if not data:
        return 0

    ids_to_delete = []

    for doc_id, metadata in zip(data["ids"], data["metadatas"]):
        if metadata and "source" in metadata:
            if Path(metadata["source"]).name == filename:
                ids_to_delete.append(doc_id)

    if ids_to_delete:
        db.delete(ids=ids_to_delete)

    return len(ids_to_delete)


if __name__ == "__main__":

    create_vector_database()

    print("Database created successfully.")