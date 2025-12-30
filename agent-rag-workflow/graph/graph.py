"""Graph workflow definition."""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from utils import print_step

load_dotenv()


def decide_to_generate(state: GraphState):
    # print(f"{'=' * 7} ASSESS GRADED DOCUMENTS {'=' * 7}")  # Replaced with print_step
    print_step("ASSESS GRADED DOCUMENTS", "Evaluating document relevance", "yellow")

    if state["web_search"]:
        # print(f"{'-' * 7} DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH. {'-' * 7}")
        print_step("DECISION", "Some documents not relevant → Including web search", "magenta")
        return WEBSEARCH
    else:
        # print(f"{'-' * 7} DECISION: GENERATE {'-' * 7}")
        print_step("DECISION", "All documents relevant → Generating answer", "green")
        return GENERATE


def route_question(state: GraphState) -> str:
    # print(f"{'=' * 7} ROUTE QUESTION {'=' * 7}")  # Replaced with print_step
    print_step("ROUTE QUESTION", "Analyzing query topic", "cyan")
    question = state["question"]

    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == WEBSEARCH:
        # print(f"{'-' * 7} DECISION: ROUTE QUESTION TO WEB SEARCH {'-' * 7}")
        print_step("DECISION", "Routing to WEB SEARCH", "magenta")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        # print(f"{'-' * 7} DECISION: ROUTE QUESTION TO RAG {'-' * 7}")
        print_step("DECISION", "Routing to VECTOR STORE (RAG)", "green")
        return RETRIEVE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    # print(f"{'=' * 7} CHECK HALLUCINATION {'=' * 7}")  # Replaced with print_step
    print_step("CHECK HALLUCINATION", "Validating answer is grounded in facts", "yellow")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # score = hallucination_grader.invoke({"documents": documents, "question": question})  # Incorrect: expects 'generation', not 'question'
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        # print(f"{'-' * 7} DECISION: GENERATION IS GROUNDED IN DOCUMENTS {'-' * 7}")
        print_step("DECISION", "✓ Generation is grounded in documents", "green")
        # print(f"{'-' * 7} GRADE GENERATION vs QUESION {'-' * 7}")
        print_step("GRADE ANSWER", "Checking if answer addresses question", "yellow")
        score = answer_grader.invoke({"question": question, "generation": generation})

        if answer_grade := score.binary_score:
            # print(f"{'-' * 7} DECISION: GENERATION ADDRESSES QUESTION {'-' * 7}")
            print_step("DECISION", "✓ Answer addresses question → Complete", "green")
            return "useful"
        else:
            # print(f"{'-' * 7} DECISION: GENERATION DOES NOT ADDRESSES QUESTION {'-' * 7}")
            print_step("DECISION", "✗ Answer doesn't address question → Web search", "red")
            return "not useful"

    else:
        # print(f"{'-' * 7} DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS. RETRY. {'-' * 7}")
        print_step("DECISION", "✗ Not grounded in documents → Regenerating", "red")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question, {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE}
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"not supported": GENERATE, "useful": END, "not useful": WEBSEARCH},
)

workflow.add_edge(WEBSEARCH, GENERATE)
# workflow.add_edge(GENERATE, END)  # Commented: This creates a duplicate edge - conditional edges above already handle GENERATE -> END

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
