import os
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    TOP_K,
    MAX_DISTANCE
)
from .data_models import RetrievedChunk, RetrievalOutput

# Global variable to store the initialized embeddings model for singleton pattern
EMBEDDINGS_MODEL = None


def get_embeddings_model():
    """
        Initializes or returns the singleton instance of the OpenAI Embeddings model.

        This ensures the model is loaded only once per process lifecycle.
    """
    global EMBEDDINGS_MODEL

    if EMBEDDINGS_MODEL is None:
        try:
            EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            raise RuntimeError("Embeddings initialization failed.")
    else:
        print("EMBEDDINGS_MODEL already initialized, reusing it")

    return EMBEDDINGS_MODEL


def load_chroma_db() -> Chroma:
    """Loads the existing ChromaDB index from disk.
    Raises:
        FileNotFoundError: If the Chroma persistence directory does not exist.
        Exception: If there is a generic error during Chroma loading.

    Returns:
        Chroma: The initialized Chroma vector store object.
    """

    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"Chroma DB not found at {CHROMA_PERSIST_DIR}. Please run ingestion.py first."
        )


    try:
        embeddings = get_embeddings_model()

        db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
        print(f"Successfully loaded Chroma DB")
        return db
    except Exception as e:
        print(f"Error loading Chroma DB: {e}")
        raise


def retrieve_context(query_text: str) -> RetrievalOutput:
    """
        Performs the similarity search against the vector database.

        Args:
            query_text: The user's input query string.

        Returns:
            RetrievalOutput: An object containing the list of retrieved chunks
                             that passed the MAX_DISTANCE filter.
    """

    try:
        db = load_chroma_db()
        raw_results = db.similarity_search_with_score(query_text, k=TOP_K)

        final_chunks = []
        for doc, distance in raw_results:
            print(f"distance={distance:.4f}")

            if distance <= MAX_DISTANCE:
                final_chunks.append(
                    RetrievedChunk(
                        id=doc.metadata.get("id", "N/A"),
                        score=round(distance, 4),
                        text=doc.page_content,
                    )
                )

        return RetrievalOutput(
            results=final_chunks,
            num_results=len(final_chunks)
        )

    except Exception as e:
        return RetrievalOutput(
            results=[RetrievedChunk(
                id="ERROR",
                score=0.0,
                text=f"Retrieval error: {e}"
            )],
            num_results=0
        )