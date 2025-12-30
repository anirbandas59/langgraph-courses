from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.nodes import web_search
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run a web search."""

    print(f"{'-' * 7} CHECK DOCUMENT RELEVANCE TO QUESTION {'-' * 7}")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )

        grade = score.binary_score

        if grade.lower() == "yes":
            print(f"{'-' * 3} GRADE: DOCUMENT RELEVANT {'-' * 3}")
            filtered_docs.append(doc)
        else:
            print(f"{'-' * 3} GRADE: DOCUMENT NOT RELEVANT {'-' * 3}")
            web_search = True
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
