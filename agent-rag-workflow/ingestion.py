"""Document ingestion and vectorstore management."""

# import os
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configuration
VECTORSTORE_PATH = Path("vectorstore/agent_rag_collection")
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


@lru_cache(maxsize=1)
def get_retriever(force_refresh: bool = False) -> VectorStoreRetriever:
    """
    Get or create the retriever instance with caching.

    Uses cached vectorstore if available, otherwise creates new one.
    This prevents expensive re-downloading and re-embedding on every import.

    Args:
        force_refresh: If True, recreate vectorstore from source URLs

    Returns:
        Configured retriever instance with MMR search
    """
    # Try to load existing vectorstore
    if VECTORSTORE_PATH.exists() and not force_refresh:
        print(f"📂 Loading cached vectorstore from {VECTORSTORE_PATH}")
        try:
            vectorstore = FAISS.load_local(
                str(VECTORSTORE_PATH),
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True,
            )
            print(f"✓ Loaded vectorstore with {vectorstore.index.ntotal} vectors")
        except Exception as e:
            print(f"⚠️  Failed to load cached vectorstore: {e}")
            print("Creating new vectorstore...")
            vectorstore = _create_vectorstore()
    else:
        if force_refresh:
            print("🔄 Force refresh requested...")
        print("Creating new vectorstore from URLs...")
        vectorstore = _create_vectorstore()

    # Configure retriever with MMR for better diversity
    return vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 4,  # Return top 4 results
            "fetch_k": 20,  # Fetch 20 candidates before MMR filtering
            "lambda_mult": 0.7,  # Balance: 0.7 = 70% relevance, 30% diversity
        },
    )


def _create_vectorstore() -> FAISS:
    """
    Create vectorstore from source URLs.

    Returns:
        FAISS vectorstore instance

    Raises:
        ValueError: If no documents loaded successfully
    """
    try:
        # Load documents with error handling per URL
        print("📥 Downloading documents...")
        all_docs: List[Document] = []
        for url in URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  ✓ Loaded {url}")
            except Exception as e:
                print(f"  ✗ Failed to load {url}: {e}")

        if not all_docs:
            raise ValueError("No documents loaded successfully from any URL")

        print(f"✓ Loaded {len(all_docs)} documents total")

        # Split documents with improved settings
        print("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,  # Larger chunks for better context (was 250)
            chunk_overlap=50,  # 10% overlap to preserve context (was 0)
        )
        doc_splits = text_splitter.split_documents(all_docs)
        print(f"✓ Created {len(doc_splits)} chunks")

        # Create embeddings and FAISS index
        print("🔢 Creating embeddings and building FAISS index...")
        print("   (This may take a minute and will call OpenAI API...)")
        vectorstore = FAISS.from_documents(
            documents=doc_splits, embedding=OpenAIEmbeddings()
        )
        print(f"✓ Created vectorstore with {vectorstore.index.ntotal} vectors")

        # Save for future use
        print(f"💾 Saving vectorstore to {VECTORSTORE_PATH}...")
        VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(VECTORSTORE_PATH))
        print("✓ Saved successfully")

        return vectorstore

    except Exception as e:
        print(f"❌ Failed to create vectorstore: {e}")
        raise


# Lazy-loaded retriever (only created when accessed)
retriever = get_retriever()


# CLI for manual ingestion management
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage vectorstore ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refresh vectorstore from source URLs
  python ingestion.py --refresh

  # Check current vectorstore status
  python ingestion.py
        """,
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh vectorstore from source URLs (re-download and re-embed)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("📚 Vectorstore Management")
    print("=" * 60 + "\n")

    retriever = get_retriever(force_refresh=args.refresh)

    print(f"\n{'=' * 60}")
    print("✅ Retriever ready!")
    print(f"   Vectors: {retriever.vectorstore.index.ntotal}")
    print(f"   Location: {VECTORSTORE_PATH}")
    print("   Search type: MMR (Maximum Marginal Relevance)")
    print(f"{'=' * 60}\n")
