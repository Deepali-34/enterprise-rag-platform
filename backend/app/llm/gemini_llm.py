import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load backend/.env explicitly
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Get API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Raise an error if the key is missing
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please add it to backend/.env"
    )

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
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