
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()


from .config import (
    DATA_FILE_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def run_ingestion():
    """
    Executes the entire RAG data ingestion pipeline.

    This process includes:
    1. Loading the source text file.
    2. Splitting the document into smaller chunks.
    3. Generating vector embeddings using OpenAI.
    4. Storing the chunks and their embeddings in the Chroma vector database.
    """
    try:
        file_path = DATA_FILE_PATH
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist")
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} document.")
    except Exception as e:
        print(f"Error loading file: {e}. Check if {DATA_FILE_PATH} exists.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    texts = text_splitter.split_documents(documents)
    print(f"Text split into {len(texts)} chunks.")

    for i, doc in enumerate(texts):
        doc.metadata["id"] = f"chunk_{i}"

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("OpenAI Embeddings model defined successfully.")
    except Exception as e:
        print(f"Error defining OpenAI Embeddings. Check OPENAI_API_KEY: {e}")
        return

    print(f"Embedding chunks and saving to ChromaDB at: {CHROMA_PERSIST_DIR}")

    try:
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
        db.persist()
        print(f"--- Ingestion Complete. {len(texts)} vectors created. ---")
    except Exception as e:
        print(f"Error storing in ChromaDB: {e}")


if __name__ == "__main__":
    run_ingestion()