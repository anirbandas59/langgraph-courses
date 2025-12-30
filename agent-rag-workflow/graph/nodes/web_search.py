from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState
from utils import print_step

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    # print(f"{'-' * 7} WEB SEARCH {'-' * 7}")  # Replaced with print_step
    print_step("WEB SEARCH", "Searching the web for additional information", "cyan")
    question = state["question"]

    if "documents" in state:
        documents = state["documents"]
    else:
        documents = None

    tavily_search_results = web_search_tool.invoke({"query": question})["results"]

    joined_search_result = "\n\n".join(
        [search_result["content"] for search_result in tavily_search_results]
    )

    web_results = Document(page_content=joined_search_result)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    print_step("WEB SEARCH", f"✓ Found {len(tavily_search_results)} web results", "green")
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
