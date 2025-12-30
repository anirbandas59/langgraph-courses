"""Graph workflow definition."""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState):
    print(f"{'=' * 7} ASSESS GRADED DOCUMENTS {'=' * 7}")

    if state["web_search"]:
        print(
            f"{'-' * 7} DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH. {'-' * 7}"
        )
        return WEBSEARCH
    else:
        print(f"{'-' * 7} DECISION: GENERATE {'-' * 7}")
        return GENERATE


def route_question(state: GraphState) -> str:
    print(f"{'=' * 7} ROUTE QUESTION {'=' * 7}")
    question = state["question"]

    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == WEBSEARCH:
        print(f"{'-' * 7} DECISION: ROUTE QUESTION TO WEB SEARCH {'-' * 7}")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print(f"{'-' * 7} DECISION: ROUTE QUESTION TO RAG {'-' * 7}")
        return RETRIEVE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print(f"{'=' * 7} CHECK HALLUCINATION {'=' * 7}")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # score = hallucination_grader.invoke({"documents": documents, "question": question})  # Incorrect: expects 'generation', not 'question'
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print(f"{'-' * 7} DECISION: GENERATION IS GROUNDED IN DOCUMENTS {'-' * 7}")
        print(f"{'-' * 7} GRADE GENERATION vs QUESION {'-' * 7}")
        score = answer_grader.invoke({"question": question, "generation": generation})

        if answer_grade := score.binary_score:
            print(f"{'-' * 7} DECISION: GENERATION ADDRESSES QUESTION {'-' * 7}")
            return "useful"
        else:
            print(
                f"{'-' * 7} DECISION: GENERATION DOES NOT ADDRESSES QUESTION {'-' * 7}"
            )
            return "not useful"

    else:
        print(
            f"{'-' * 7} DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS. RETRY. {'-' * 7}"
        )
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
