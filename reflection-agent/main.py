from typing import Annotated, TypedDict

from chains import generate_chain, reflect_chain
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# Node name constants
GENERATE = "generate"
REFLECT = "reflect"


class AgentState(TypedDict):
    """State of the reflection agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    revision_count: int
    max_revisions: int


def generate(state: AgentState) -> dict:
    """Generate a tweet using the generate_chain."""
    revision_count = state.get("revision_count", 0)

    # Invoke the generate chain with message history
    response = generate_chain.invoke({"messages": state["messages"]})

    # Return only the new message - add_messages reducer handles appending
    return {"messages": [response], "revision_count": revision_count + 1}


def reflect(state: AgentState) -> dict:
    """Critique the generated tweet using the reflect_chain."""
    # Invoke the reflect chain with message history
    response = reflect_chain.invoke({"messages": state["messages"]})

    # Return critique as HumanMessage - add_messages reducer handles appending
    return {"messages": [HumanMessage(content=response.content)]}


def should_continue(state: AgentState) -> str:
    """Check if the tweet is satisfactory or needs revision."""
    messages = state.get("messages", [])
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 5)

    # Check if max revisions reached
    if revision_count >= max_revisions:
        print(f"\n⚠️  Max revisions ({max_revisions}) reached. Ending process.\n")
        return "end"

    # Get the last critique message
    if len(messages) > 0:
        last_message = messages[-1].content
        # Check if critique indicates satisfaction (simple keyword check)
        if isinstance(last_message, str) and any(
            keyword in last_message.lower()
            for keyword in ["satisfactory", "excellent", "perfect", "great job"]
        ):
            print(f"\n✓ Tweet is satisfactory after {revision_count} revision(s)!\n")
            return "end"

    print(
        f"\n↻ Tweet needs improvement. Revision {revision_count + 1}/{max_revisions}\n"
    )
    return "continue"


def create_workflow():
    """Create the reflection agent workflow."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node(GENERATE, generate)
    builder.add_node(REFLECT, reflect)

    # Set entry point
    builder.set_entry_point(GENERATE)

    # Add edges
    builder.add_edge(GENERATE, REFLECT)
    builder.add_conditional_edges(
        REFLECT, should_continue, {"continue": GENERATE, "end": END}
    )

    graph = builder.compile()

    # Optional: Print graph visualization for debugging
    # print(graph.get_graph().draw_mermaid_png(output_file_path="graph.png"))
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    # graph.get_graph().print_ascii()

    return graph


def main():
    """Run the reflection agent."""
    print("=" * 60)
    print("REFLECTION AGENT - Tweet Generator")
    print("=" * 60)

    # Get topic from user
    topic = input("\nEnter a topic for the tweet: ").strip()

    if not topic:
        topic = "artificial intelligence"
        print(f"No topic provided. Using default: {topic}")

    # Initialize state with messages
    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content=f"Write a viral tweet about {topic}. Keep it under 280 characters."
            )
        ],
        "revision_count": 0,
        "max_revisions": 5,
    }

    # Create and run workflow
    app = create_workflow()

    print(f"\n🚀 Starting reflection process for topic: '{topic}'")
    print("-" * 60)

    # Run the workflow
    result = app.invoke(initial_state)

    # Display results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    # Extract the final tweet (should be in AI messages)
    messages = result["messages"]
    tweets = [msg.content for msg in messages if isinstance(msg, AIMessage)]
    critiques = [
        msg.content
        for msg in messages
        if isinstance(msg, HumanMessage)
        and isinstance(msg.content, str)
        and len(msg.content) > len(topic)
    ]

    if tweets:
        final_tweet = tweets[-1]
        print(f"\nFinal Tweet:\n{final_tweet}")
        print(f"\nTweet Length: {len(final_tweet)} characters")

    if critiques:
        print(f"\nFinal Critique:\n{critiques[-1]}")

    print(f"\nTotal Revisions: {result['revision_count']}")
    print(f"\nMessage History Length: {len(messages)} messages")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
