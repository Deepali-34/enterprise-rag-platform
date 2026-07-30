from pydantic import BaseModel


class Source(BaseModel):
    filename: str
    page: int


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]