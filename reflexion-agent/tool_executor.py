from typing import List
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# Initialize Tavily search tool
# Set max_results to control how many search results to retrieve
tavily_search = TavilySearch(max_results=3)


def execute_searches(search_queries: List[str]) -> str:
    """Execute multiple search queries and aggregate results.

    Args:
        search_queries: List of search query strings

    Returns:
        Formatted string containing all search results
    """
    all_results = []

    for query in search_queries:
        try:
            results = tavily_search.invoke(query)
            all_results.append(f"\n--- Results for: {query} ---")

            if isinstance(results, list):
                for i, result in enumerate(results, 1):
                    content = result.get("content", "No content available")
                    url = result.get("url", "No URL")
                    all_results.append(f"\n{i}. {content}\nSource: {url}")
            else:
                all_results.append(str(results))

        except Exception as e:
            all_results.append(f"\nError searching for '{query}': {str(e)}")

    return "\n".join(all_results) if all_results else "No search results found."
