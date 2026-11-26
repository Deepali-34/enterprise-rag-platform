import os
import ssl
from pathlib import Path
from typing import Iterator

import httpx
import truststore
from dotenv import load_dotenv
from google import genai
from google.genai import types


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_FILE)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing from backend/.env")

MODEL_NAME = "gemini-3.5-flash-lite"

# Windows certificate trust configuration
ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ssl_context.load_default_certs()

http_client = httpx.Client(
    verify=ssl_context,
    timeout=httpx.Timeout(120.0, connect=30.0),
    trust_env=True,
)

client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options=types.HttpOptions(httpxClient=http_client),
)


def extract_interaction_text(response) -> str:
    """Extract generated text from a Gemini Interaction response."""

    # Convenience property, if provided by the SDK.
    direct_text = getattr(response, "output_text", None)

    if direct_text:
        return str(direct_text).strip()

    collected = []

    for item in getattr(response, "output", []) or []:

        item_text = getattr(item, "text", None)

        if item_text:
            collected.append(str(item_text))
            continue

        content = getattr(item, "content", None)

        if content:
            for part in content:
                part_text = getattr(part, "text", None)

                if part_text:
                    collected.append(str(part_text))

    answer = "\n".join(collected).strip()

    if not answer:
        raise RuntimeError(
            "Gemini returned no extractable text. "
            f"Response structure: {response!r}"
        )

    return answer


class GeminiResponse:
    """Compatibility wrapper for the existing RAG pipeline."""

    def __init__(self, text: str):
        self.text = text


class TextChunk:
    def __init__(self, text: str):
        self.text = text


class GeminiLLM:

    def invoke(self, prompt: str):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
        )

        return GeminiResponse(
            extract_interaction_text(interaction)
        )

    def stream(self, prompt: str) -> Iterator:
        """
        Compatibility streaming interface.
        Generates the response first, then yields text chunks.
        """

        response = self.invoke(prompt)
        answer = response.text

        for start in range(0, len(answer), 40):
            yield TextChunk(answer[start:start + 40])


llm = GeminiLLM()


def main():

    print("=" * 60)
    print("Enterprise RAG - Gemini Integration Test")
    print("=" * 60)
    print("Model:", MODEL_NAME)

    print("\nTesting generation...")

    response = llm.invoke(
        "Reply with exactly: GEMINI_INTEGRATION_SUCCESS"
    )

    print("Gemini response:", response.text)

    print("\nTesting stream compatibility...")

    chunks = list(
        llm.stream(
            "Reply with exactly: GEMINI_STREAM_SUCCESS"
        )
    )

    print("".join(chunk.text for chunk in chunks))

    print("\nAll Gemini integration tests passed.")


if __name__ == "__main__":
    main()
