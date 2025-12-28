from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from chains import actor_chain, revisor_chain
from tool_executor import execute_searches


# Define the state structure for our graph
class GraphState(TypedDict):
    """State schema for the reflexion agent graph."""

    question: str  # The research question
    answer: str  # Current answer
    search_queries: list[str]  # Search queries to execute
    search_results: str  # Aggregated search results
    critique: str  # Critique of the current answer
    revision_count: int  # Number of revisions made
    max_revisions: int  # Maximum allowed revisions


# Node 1: Draft - Generate initial answer
def draft_answer(state: GraphState) -> dict:
    """Actor node: Generate initial answer and search queries."""
    print("\n--- DRAFTING INITIAL ANSWER ---")

    # Invoke the actor chain to generate answer and search queries
    response = actor_chain.invoke({"question": state["question"]})

    print(f"Generated answer: {response.answer[:100]}...")
    print(f"Search queries: {response.search_queries}")

    return {
        **state,
        "answer": response.answer,
        "search_queries": response.search_queries,
        "revision_count": 0,
    }


# Node 2: Execute - Run search queries
def execute_tools(state: GraphState) -> dict:
    """Tool execution node: Execute search queries and gather results."""
    print("\n--- EXECUTING SEARCH QUERIES ---")

    # Execute all search queries
    search_results = execute_searches(state["search_queries"])

    print(f"Retrieved {len(search_results)} characters of search results")

    return {
        **state,
        "search_results": search_results,
    }


# Node 3: Revise - Critique and improve answer
def revise_answer(state: GraphState) -> dict:
    """Revisor node: Critique answer and generate improved version."""
    print(f"\n--- REVISING ANSWER (Revision #{state['revision_count'] + 1}) ---")

    # Invoke the revisor chain to critique and revise
    response = revisor_chain.invoke(
        {
            "question": state["question"],
            "previous_answer": state["answer"],
            "search_results": state["search_results"],
        }
    )

    print(f"Critique: {response.critique[:100]}...")
    print(f"New search queries: {response.search_queries}")
    print(f"Revised answer: {response.revised_answer[:100]}...")

    return {
        **state,
        "critique": response.critique,
        "answer": response.revised_answer,
        "search_queries": response.search_queries,
        "revision_count": state["revision_count"] + 1,
    }


# Conditional edge: Decide whether to continue revising or finish
def should_continue(state: GraphState) -> Literal["execute_tools", "end"]:
    """Determine if we should continue revising or end the process."""

    # Check if we've reached the maximum number of revisions
    if state["revision_count"] >= state["max_revisions"]:
        print("\n--- MAX REVISIONS REACHED - ENDING ---")
        return "end"

    # If we have search queries, continue to execute them
    if state.get("search_queries") and len(state["search_queries"]) > 0:
        print("\n--- CONTINUING TO SEARCH ---")
        return "execute_tools"

    print("\n--- NO MORE SEARCH QUERIES - ENDING ---")
    return "end"


# Build the graph
def create_graph():
    """Create and compile the reflexion agent graph."""

    # Initialize the graph with our state schema
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("draft_answer", draft_answer)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("revise_answer", revise_answer)

    # Define edges
    # Start -> Draft initial answer
    workflow.set_entry_point("draft_answer")

    # Draft -> Execute searches
    workflow.add_edge("draft_answer", "execute_tools")

    # Execute -> Revise
    workflow.add_edge("execute_tools", "revise_answer")

    # Revise -> Conditional (either execute more searches or end)
    workflow.add_conditional_edges(
        "revise_answer",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "end": END,
        },
    )

    # Compile the graph
    return workflow.compile()


# Create the compiled graph
graph = create_graph()
