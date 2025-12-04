from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_FILE_PATH = BASE_DIR / "data" / "accountant_article.txt"
CHROMA_COLLECTION_NAME = "accountant_rag"
CHROMA_PERSIST_DIR = "chroma_db"


CHUNK_SIZE = 500

CHUNK_OVERLAP = 50

TOP_K = 3
MAX_DISTANCE=1.35
SIMILARITY_THRESHOLD = 0.35