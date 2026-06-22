from pydantic import BaseModel, Field
from backend.config import MAX_QUESTION_LENGTH


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
