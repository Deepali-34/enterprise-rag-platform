from sentence_transformers import SentenceTransformer
from pathlib import Path

from app.loaders.multi_pdf_loader import load_multiple_pdfs
from app.preprocessing.text_splitter import split_documents


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(texts)

    return embeddings


def main():

    pdf_folder = Path("sample_documents")

    documents = load_multiple_pdfs(pdf_folder)

    chunks = split_documents(documents)

    embeddings = generate_embeddings(chunks)

    print("\n" + "=" * 60)
    print("Embedding Generation Successful")
    print("=" * 60)

    print(f"\nTotal Chunks : {len(chunks)}")
    print(f"Total Embeddings : {len(embeddings)}")

    print("\nShape of One Embedding Vector:")
    print(len(embeddings[0]))

    print("\nFirst 10 Values of First Embedding:\n")

    print(embeddings[0][:10])


if __name__ == "__main__":
    main()