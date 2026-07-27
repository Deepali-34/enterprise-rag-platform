from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.loaders.multi_pdf_loader import load_multiple_pdfs


def split_documents(documents):
    """
    Split LangChain documents into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    return chunks


def main():
    # Load all PDFs
    pdf_folder = Path("sample_documents")
    documents = load_multiple_pdfs(pdf_folder)

    print("=" * 60)
    print("Documents Loaded Successfully")
    print("=" * 60)

    print(f"Total Pages Loaded : {len(documents)}")

    # Split into chunks
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