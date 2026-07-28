from fastapi import FastAPI
from pydantic import BaseModel

from app.rag.rag_chain import ask_rag

app = FastAPI(
    title="Enterprise RAG Platform",
    description="AI-powered document question answering system",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Enterprise RAG Platform API is running!"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    answer = ask_rag(request.question)

    return {
        "question": request.question,
        "answer": answer
    }