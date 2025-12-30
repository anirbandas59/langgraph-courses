from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever
from utils import print_step


def retrieve(state: GraphState) -> Dict[str, Any]:
    # print(f"{'-' * 7} RETRIEVE {'-' * 7}")  # Replaced with print_step
    print_step("RETRIEVE", "Fetching documents from vector store", "cyan")
    question = state["question"]
    documents = retriever.invoke(question)

    print_step("RETRIEVE", f"✓ Retrieved {len(documents)} documents", "green")
    return {"documents": documents, "question": question}
