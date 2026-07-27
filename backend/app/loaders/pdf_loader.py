from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(pdf_path: str):
    """
    Load a PDF document and return a list of LangChain Document objects.
    """
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()


def main():
    # Path to your PDF
    pdf_path = "sample_documents/resume.pdf"

    # Load the PDF
    documents = load_pdf(pdf_path)

    # Display details
    print("=" * 60)
    print("PDF Loaded Successfully")
    print("=" * 60)

    print(f"Total Pages: {len(documents)}")

    print("\nFirst Page Content:\n")
    print(documents[0].page_content[:1000])

    print("\nMetadata:")
    print(documents[0].metadata)


if __name__ == "__main__":
    main()