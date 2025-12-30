from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState
from utils import print_step


def generate(state: GraphState) -> Dict[str, Any]:
    # print(f"{'-' * 7} Generate {'-' * 7}")  # Replaced with print_step
    print_step("GENERATE", "Creating answer from documents", "cyan")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})

    print_step("GENERATE", "✓ Answer generated", "green")
    return {"documents": documents, "question": question, "generation": generation}
