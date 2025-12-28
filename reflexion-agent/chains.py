import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# Actor Chain - Initial answer generation
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher tasked with answering questions accurately and thoroughly.
Current time: {time}

Your goal is to provide a detailed, well-researched answer to the user's question.

Instructions:
1. Provide a comprehensive answer (~250 words) based on your knowledge
2. Be specific, factual, and use clear professional language
3. If you include any claims that should be verified, note them
4. Identify 1-3 search queries that would help find supporting evidence and citations to validate your answer

Quality criteria:
- Accuracy and factual correctness
- Completeness of key information
- Clear and organized structure
- Appropriate depth without unnecessary verbosity""",
        ),
        ("human", "{question}"),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

# Bind the structured output schema to the LLM
actor_chain = actor_prompt_template | llm.with_structured_output(AnswerQuestion)

# Revisor Chain - Answer revision based on critique and new information
revisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert research critic and editor.
Current time: {time}

Your task is to critically evaluate and improve the previous answer using the search results.

Step 1 - CRITIQUE (be thorough and constructive):
- Identify what information is missing or incomplete
- Point out any inaccuracies or claims that need verification
- Note what is superfluous or could be removed for clarity
- Assess the quality and organization of the response

Step 2 - SEARCH STRATEGY:
- Generate 1-3 targeted search queries to address the gaps identified in your critique
- Focus on finding specific facts, statistics, or sources that would strengthen the answer

Step 3 - REVISE:
- Incorporate relevant information from the search results
- Add important missing information identified in your critique
- Remove or condense superfluous content
- MUST include numerical citations [1], [2], etc. in the text for verifiable claims
- Add a "References" section at the bottom listing the ACTUAL URLs from the search results
- Use this format for references:
  - [1] <actual URL from search results>
  - [2] <actual URL from search results>
- DO NOT use placeholder URLs like "https://example.com"
- ONLY cite sources that were actually provided in the search results
- Keep the revised answer to ~250 words (excluding References section)

Quality standards:
- Accuracy and factual correctness (highest priority)
- Completeness with proper citations
- Clear organization and readability
- Appropriate depth without verbosity""",
        ),
        (
            "human",
            """Question: {question}

Previous Answer:
{previous_answer}

Search Results:
{search_results}

Please provide your critique, search queries, and revised answer following the steps above.""",
        ),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

# Bind the structured output schema to the LLM
revisor_chain = revisor_prompt_template | llm.with_structured_output(ReviseAnswer)
