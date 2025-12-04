from pydantic import BaseModel
from typing import List


class QueryInput(BaseModel):
    """
    Pydantic model for the query text sent from the Frontend to the FastAPI backend.
    """
    query_text: str

class RetrievedChunk(BaseModel):
    """
    Model representing a single text chunk retrieved from the Vector Database.
    """
    id: str             # The unique ID or index of the chunk
    score: float        # The Similarity Score (0 to 1)
    text: str           # The content of the text chunk

class RetrievalOutput(BaseModel):
    """
    Model for the final API response, containing filtered results and count.
    """
    results: List[RetrievedChunk] # List of chunks that passed the similarity threshold
    num_results: int              # The actual number of chunks returned