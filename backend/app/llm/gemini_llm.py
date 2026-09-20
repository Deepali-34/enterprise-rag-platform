import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------
# Load backend/.env explicitly
# ---------------------------------------------------------

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(ENV_FILE)


# ---------------------------------------------------------
# Get API key
# ---------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Please add GOOGLE_API_KEY to backend/.env"
    )


# ---------------------------------------------------------
# Initialize Gemini
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


def main():

    print("=" * 60)
    print("Gemini LLM Test")
    print("=" * 60)

    response = llm.invoke(
        "Say Hello!"
    )

    print("\nGemini Response:")
    print(response.text)


if __name__ == "__main__":
    main()