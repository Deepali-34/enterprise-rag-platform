from fastapi import FastAPI

app = FastAPI(
    title="Enterprise RAG Platform",
    description="AI-powered document question answering system",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Enterprise RAG Platform API is running!"
    }