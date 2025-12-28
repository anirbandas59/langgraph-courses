from typing import List
from pydantic import BaseModel, Field


class AnswerQuestion(BaseModel):
    """Generate an answer to a research question."""

    answer: str = Field(
        description="Detailed answer to the research question with citations"
    )
    search_queries: List[str] = Field(
        description="1-3 search queries to find supporting information",
        max_length=3,
    )


class ReviseAnswer(BaseModel):
    """Revise a previous answer based on critique and new information."""

    critique: str = Field(
        description="Critique of the previous answer, identifying gaps or issues"
    )
    search_queries: List[str] = Field(
        description="1-3 search queries to find information addressing the critique",
        max_length=3,
    )
    revised_answer: str = Field(
        description="Improved answer incorporating critique and new information"
    )
