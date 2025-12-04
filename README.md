# 🌟 RAG Retrieval System: Junior AI Engineer Home Exam

## 🎯 1. Overview and Project Goal

This project implements the **Retrieval (R)** component of a Retrieval-Augmented Generation (RAG) pipeline. The system ingests a small text dataset, converts it into vector embeddings, stores them in a **Chroma** vector database, and retrieves the most relevant context chunks based on a user query.

**Objective:** Demonstrate a fully functional retrieval pipeline (Ingestion and Retrieval) with a minimal user interface. **(The LLM generation step is deliberately omitted as per the assignment instructions.)**

---

## 📊 2. Reasoning and Design Choices (75% of Grade)

### A. Dataset Selection

| Parameter | Choice | Justification |
| :--- | :--- | :--- |
| **Source** | "Why Hiring an Accountant Saves Time and Money" (Kaggle) | The data is entirely **text-based** and **very small** (~8,200 characters), easily meeting the <30,000 character requirement and ensuring costs remain far below $1 USD. |
| **Expected Queries** | Quantitative data (e.g., "How much money can I save?"), Lists (e.g., "Recommended software?"), and Contextual comparisons (e.g., "Local vs. remote accountant?"). | The diverse, information-rich content allows for robust testing of semantic retrieval capabilities. |

### B. Vector Database Selection: ChromaDB

* **Choice:** ChromaDB (Local Persisted) 
* **Justification:** Chroma was selected for its **ease of local setup** (In-memory/Disk-based persistence). This eliminated the need for cloud credentials (Pinecone/Weaviate) and complex networking, allowing for rapid development and meeting the requirement for a cost-free solution suitable for a small prototype.

### C. Embeddings and RAG Parameters

| Parameter | Choice | Justification |
| :--- | :--- | :--- |
| **Embeddings Model** | OpenAI `text-embedding-3-small` | Adheres to the requirement to use an OpenAI model while providing state-of-the-art embedding quality at a negligible cost. |
| **Chunking Strategy** | Size: 500 chars, Overlap: 50 chars | This provides a balanced approach: the chunk size is large enough to maintain semantic **context** (e.g., a full paragraph) but small enough to maintain **relevance** to specific queries. The overlap prevents important sentences from being cut across boundaries. |
| **Retrieval Output Logic** | **1 Consolidated Chunk** (Merging Top K results) | The dataset is short, resulting in high similarity scores and redundant output when returning K individual chunks. The logic was modified to **consolidate** the Top K results into one comprehensive context chunk to provide a more readable and relevant single output. |
| **Similarity Filter** | **MAX\_DISTANCE = 1.25** (Approximate Cosine Distance) | Instead of a fixed similarity threshold, we filter results based on a **Maximum Distance**. This ensures that even with slight variations in the distance metric returned by Chroma, only chunks highly related to the query (distance below 1.25, confirmed through debugging) are merged into the final output. |

---

## 💻 3. Technical Implementation and Usage

### A. System Architecture

The system runs on a two-server architecture for development simplicity:

* **Backend (API):** FastAPI (Python) on **Port 9000**. Handles retrieval logic and filtering.
* **Frontend (UI):** Simple Python `http.server` on **Port 8001**. Serves the static HTML/JS interface.

### B. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [your_repo_link]
    cd rag-retrieval-project
    ```
2.  **Create Environment & Install Dependencies:**
    ```bash
    python -m venv .venv
    # Activate environment (varies by OS)
    pip install -r requirements.txt
    ```
3.  **Set API Key:**
    Create a file named **`.env`** in the project root directory and add your OpenAI key:
    ```
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
    ```
4.  **Run Ingestion (One-Time Step):**
    This creates the vector index in the `chroma_db/` folder.
    ```bash
    python -m src.ingestion
    ```

### C. Running the System

1.  **Start API Server (Backend):** (Terminal 1)

    ```bash
    uvicorn src.main:app --reload --port 9000 --env-file .env
    ```

    *(The `--env-file .env` ensures the API key is loaded correctly.)*

2.  **Start Frontend Server:** (Terminal 2 - must be run from the `static` folder)

    ```bash
    cd static
    python -m http.server 8001
    ```

3.  **Access UI:**
    Open your browser to: **`http://127.0.0.1:8001/`**

### D. Testing Example

| Query | Expected Result (Consolidated Context) | Score Type |
| :--- | :--- | :--- |
| "how much money can I save?" | Context containing "20% per year" and "pays for itself many times over." | Highest raw similarity score from merged chunks. |
| "accreditations I should look for" | Context containing "ACCA, ICAEW, or CIMA." | |

---

