from app.retriever.retriever import search_documents
from app.llm.gemini_llm import llm


def build_context(documents):
    context = ""

    for doc in documents:
        context += doc.page_content
        context += "\n\n"

    return context


def create_prompt(context, question):

    prompt = f"""
You are an AI assistant.

Answer the question ONLY using the context below.

If the answer is not present in the context, say:

"I could not find that information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def ask_rag(question):

    documents = search_documents(question)

    context = build_context(documents)

    prompt = create_prompt(context, question)

    response = llm.invoke(prompt)

    return response.text()


def main():

    print("=" * 60)
    print("Enterprise RAG Platform")
    print("=" * 60)

    question = input("\nAsk a question: ")

    answer = ask_rag(question)

    print("\n" + "=" * 60)
    print("Answer")
    print("=" * 60)

    print(answer)


if __name__ == "__main__":
    main()