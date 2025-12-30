from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print(f"{'-' * 7} RETRIEVE {'-' * 7}")
    question = state["question"]
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}
