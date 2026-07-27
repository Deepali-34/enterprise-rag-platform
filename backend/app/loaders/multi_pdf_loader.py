from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
def load_multiple_pdfs(pdf_folder: Path):
    all_documents = []

    pdf_files = list(pdf_folder.glob("*.pdf"))

    print(f"\nFound {len(pdf_files)} PDF(s):\n")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            all_documents.extend(documents)

        except Exception as e:
            print(f"Failed to load {pdf_file.name}: {e}")

    return all_documents


def main():
    pdf_folder = Path("sample_documents")

    documents = load_multiple_pdfs(pdf_folder)

    print("\n" + "=" * 60)
    print("All PDFs Loaded Successfully")
    print("=" * 60)

    print(f"Total Pages Loaded: {len(documents)}")

    print("\nSources:\n")

    for doc in documents:
        print(doc.metadata["source"])


if __name__ == "__main__":
    main()