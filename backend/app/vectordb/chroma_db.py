from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.loaders.multi_pdf_loader import load_multiple_pdfs
from app.preprocessing.text_splitter import split_documents


# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_database():

    pdf_folder = Path("sample_documents")

    documents = load_multiple_pdfs(pdf_folder)

    chunks = split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_storage"
    )

    return vector_db, chunks


def main():

    vector_db, chunks = create_vector_database()

    print("\n" + "=" * 60)
    print("Chroma Vector Database Created Successfully")
    print("=" * 60)

    print(f"\nTotal Chunks Stored : {len(chunks)}")

    print("\nDatabase Location:")
    print("backend/chroma_storage")


if __name__ == "__main__":
    main()