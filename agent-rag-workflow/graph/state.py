"""State definitions for the graph workflow."""

from typing import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of the graph

    Attributes:
    question: question to ask
    generation: LLM generation
    web_search: Whether to add search
    documents: List of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: list[str]
