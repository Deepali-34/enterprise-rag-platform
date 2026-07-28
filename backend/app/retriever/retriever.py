from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="chroma_storage",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)


def search_documents(query: str):

    results = retriever.invoke(query)

    return results


def main():

    query = input("Enter your question: ")

    results = search_documents(query)

    print("\n" + "=" * 60)
    print("Top Matching Chunks")
    print("=" * 60)

    for i, doc in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 40)
        print(f"Source : {doc.metadata['source']}")
        print("\nContent:\n")
        print(doc.page_content[:500])
        print()


if __name__ == "__main__":
    main()