"""State definitions for the graph workflow."""

from typing import List, TypedDict

from langchain_core.documents import Document


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
    documents: List[Document]  # Fixed: was list[str], should be List[Document]
