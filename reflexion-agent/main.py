from graph import graph


def main():
    """Run the Reflexion agent with a research question."""

    # Example research question
    # question = "What are the key architectural components of a Reflexion agent?"
    question = "What is the key difference in Reflection agent and Reflexion agent?"

    print("=" * 80)
    print("REFLEXION AGENT - AI Research Assistant")
    print("=" * 80)
    print(f"\nQuestion: {question}\n")

    # Initial state
    initial_state = {
        "question": question,
        "answer": "",
        "search_queries": [],
        "search_results": "",
        "critique": "",
        "revision_count": 0,
        "max_revisions": 2,  # Allow up to 2 revisions
    }

    # Run the graph
    try:
        final_state = graph.invoke(initial_state)

        # Display final results
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"\nTotal Revisions: {final_state['revision_count']}")
        print(f"\nFinal Answer:\n{final_state['answer']}")

        if final_state.get("critique"):
            print(f"\nLast Critique:\n{final_state['critique']}")

    except Exception as e:
        print(f"\n❌ Error running agent: {e}")
        raise


if __name__ == "__main__":
    main()
