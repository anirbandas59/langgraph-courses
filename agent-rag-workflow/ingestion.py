import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# Create vector store from documents with a collection name
collection_name = "agent_rag_collection"
vectorstore = FAISS.from_documents(documents=doc_splits, embedding=OpenAIEmbeddings())

# Save the vector store locally in a separate folder
# vectorstore_path = os.path.join("vectorstore", collection_name)
# os.makedirs(vectorstore_path, exist_ok=True)
# vectorstore.save_local(vectorstore_path)

# Create retriever from the vector store
retriever = vectorstore.as_retriever()
