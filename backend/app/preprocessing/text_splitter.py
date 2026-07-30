from pathlib import Path
from datetime import datetime
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.loaders.multi_pdf_loader import load_multiple_pdfs


def split_documents(documents):
    """
    Split LangChain documents into smaller chunks and enrich metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for index, chunk in enumerate(chunks):

        metadata = chunk.metadata

        source = metadata.get("source", "")

        filename = Path(source).name if source else "Unknown"

        extension = Path(filename).suffix.lower()

        chunk.metadata.update({
            "filename": filename,
            "document_type": extension.replace(".", ""),
            "chunk_number": index + 1,
            "chunk_id": str(uuid.uuid4()),
            "upload_timestamp": upload_time,
            "page_number": metadata.get("page", 1)
        })

    return chunks


def main():
    pdf_folder = Path("sample_documents")

    documents = load_multiple_pdfs(pdf_folder)

    print("=" * 60)
    print("Documents Loaded Successfully")
    print("=" * 60)

    print(f"Total Pages Loaded : {len(documents)}")

    chunks = split_documents(documents)

    print(f"Total Chunks Created : {len(chunks)}")

    print("\n" + "=" * 60)
    print("FIRST CHUNK")
    print("=" * 60)

    print(chunks[0].page_content)

    print("\n" + "=" * 60)
    print("METADATA")
    print("=" * 60)

    print(chunks[0].metadata)


if __name__ == "__main__":
    main()