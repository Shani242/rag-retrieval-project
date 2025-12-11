from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import logging

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()


from .config import (
    DATA_FILE_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Configure logging
logger = logging.getLogger(__name__)

class IngestionError(Exception):
    """Custom exception for errors during the ingestion pipeline."""
    pass


def run_ingestion():
    """
    Executes the entire RAG data ingestion pipeline.

    Raises:
        IngestionError: If any critical step fails.
    """
    try:
        file_path = DATA_FILE_PATH
        if not file_path.exists():
            raise IngestionError(f"Data file not found at {file_path}")

        loader = TextLoader(file_path.as_posix(), encoding="utf-8")
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} document(s).")
    except Exception as e:
        raise IngestionError(f"Error loading file: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "\r\n", " ", ""]
    )

    texts = text_splitter.split_documents(documents)
    logger.info(f"Text split into {len(texts)} chunks.")

    for i, doc in enumerate(texts):
        doc.metadata["id"] = f"chunk_{i}"

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info("OpenAI Embeddings model defined successfully.")
    except Exception as e:
        raise IngestionError(f"Error defining OpenAI Embeddings. Check OPENAI_API_KEY: {e}")

    logger.info(f"Embedding chunks and saving to ChromaDB at: {CHROMA_PERSIST_DIR}")

    try:
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR.as_posix(),
        )
        db.persist()
        logger.info(f"--- Ingestion Complete. {len(texts)} vectors created. ---")
    except Exception as e:
        raise IngestionError(f"Error storing in ChromaDB: {e}")


if __name__ == "__main__":
    run_ingestion()