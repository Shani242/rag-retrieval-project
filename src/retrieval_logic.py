import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    TOP_K,
    MAX_DISTANCE
)
from .data_models import RetrievedChunk, RetrievalOutput

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalBaseError(Exception):
    """Base exception for the context retrieval module."""
    pass


class RetrievalChromaLoadError(RetrievalBaseError):
    """Error occurring when loading the ChromaDB index."""
    pass


class RetrievalEmbeddingsError(RetrievalBaseError):
    """Error occurring during the initialization of the Embeddings model."""
    pass


EMBEDDINGS_MODEL = None


def get_embeddings_model():
    """
    Initializes or returns the singleton instance of the OpenAI Embeddings model.

    Raises:
        RetrievalEmbeddingsError: If initialization fails (e.g., missing API key).
    """
    global EMBEDDINGS_MODEL

    if EMBEDDINGS_MODEL is None:
        try:
            EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
            logger.info("Embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Embeddings initialization failed: {e}")
            raise RetrievalEmbeddingsError(f"Embeddings initialization failed: {e}")

    return EMBEDDINGS_MODEL


def load_chroma_db() -> Chroma:
    """
    Loads the existing ChromaDB index from disk.

    Raises:
        RetrievalChromaLoadError: If the DB directory is missing or loading fails.

    Returns:
        Chroma: The initialized Chroma vector store object.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        error_msg = f"Chroma DB not found at {CHROMA_PERSIST_DIR}. Please run ingestion.py first."
        logger.error(error_msg)
        raise RetrievalChromaLoadError(error_msg)

    try:
        embeddings = get_embeddings_model()
        db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
        logger.info("ChromaDB loaded successfully")
        return db
    except RetrievalEmbeddingsError:
        raise  # Re-raise embeddings errors as-is
    except Exception as e:
        logger.error(f"Error loading Chroma DB: {e}")
        raise RetrievalChromaLoadError(f"Error loading Chroma DB: {e}") from e


def retrieve_context(query_text: str) -> RetrievalOutput:
    """
    Performs similarity search against the vector database.

    Args:
        query_text: The user's input query string.

    Returns:
        RetrievalOutput: An object containing the list of retrieved chunks.

    Raises:
        RetrievalBaseError: If retrieval fails (caller should handle this).
    """
    try:
        db = load_chroma_db()
        raw_results = db.similarity_search_with_score(query_text, k=TOP_K)

        final_chunks = []
        for doc, distance in raw_results:
            if distance <= MAX_DISTANCE:
                final_chunks.append(
                    RetrievedChunk(
                        id=doc.metadata.get("id", "N/A"),
                        distance=round(distance, 4),
                        text=doc.page_content,
                    )
                )

        logger.info(f"Retrieved {len(final_chunks)} chunks for query")
        return RetrievalOutput(
            results=final_chunks,
            num_results=len(final_chunks)
        )

    except RetrievalBaseError as e:
        logger.error(f"Retrieval setup error: {e}")
        raise  # Let caller decide how to handle
    except Exception as e:
        logger.error(f"Unexpected retrieval error: {type(e).__name__}: {e}")
        raise RetrievalBaseError(f"Unexpected error during retrieval: {e}") from e