import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .data_models import QueryInput, RetrievalOutput
from .retrieval_logic import retrieve_context


app = FastAPI(title="RAG Retrieval API")

# Define CORS origins to allow cross-origin requests (e.g., from the separate frontend server)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """
        Serves the main frontend application (index.html) from the static folder.
    """
    html_file = Path(__file__).resolve().parent.parent / "static" / "index.html"
    if html_file.exists():
        return FileResponse(html_file, media_type="text/html")
    else:
        return {"error": f"index.html not found at {html_file}"}


@app.post("/api/retrieve", response_model=RetrievalOutput)
async def retrieval_endpoint(query_input: QueryInput):
    """
    Handles the core retrieval logic for RAG.

    Accepts a user query, retrieves relevant chunks from the vector store,
    and returns a list of filtered context chunks.

    Args:
        query_input: The Pydantic model containing the query text.
        request: The incoming request object (FastAPI dependency).

    Returns:
        RetrievalOutput: A Pydantic model containing the list of retrieved
                         chunks and the total count.

    Raises:
        HTTPException: 400 if the query is empty.
        HTTPException: 500 if an internal retrieval error occurs (e.g., DB not found).
    """
    print(f"[API] Received query: {query_input.query_text}")

    query_text = query_input.query_text.strip()

    if not query_text:
        print(f"[API] Query is empty!")
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    print(f"[API] Calling retrieve_context()...")
    results = retrieve_context(query_text)
    print(f"[API] Got {results.num_results} results")

    if results.num_results == 0 and len(results.results) > 0 and results.results[0].id == "ERROR":
        print(f"[API] Error in retrieval: {results.results[0].text}")
        raise HTTPException(status_code=500, detail=results.results[0].text)

    return results


@app.on_event("startup")
async def startup_event():
    """
        Startup event handler to check the status of the Chroma DB persistence directory.
    """
    from .config import CHROMA_PERSIST_DIR
    print("API Server Startup: Checking Chroma DB status")
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print("!!! WARNING: Chroma DB not found. Run ingestion.py before queries will work. !!!")
    else:
        print("Chroma DB found. Ready for queries.")