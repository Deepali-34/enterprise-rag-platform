import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)


def main():
    response = llm.invoke("Say Hello!")

    print("\n" + "=" * 60)
    print("Gemini Connected Successfully")
    print("=" * 60)
    print(response.text())


if __name__ == "__main__":
    main()