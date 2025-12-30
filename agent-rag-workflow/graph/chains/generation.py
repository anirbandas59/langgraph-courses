import os

from langchain_core.output_parsers import StrOutputParser
from langsmith import Client

from model.model import llm

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)

generation_chain = prompt | llm | StrOutputParser()
