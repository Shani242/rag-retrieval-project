# 🌟 RAG Retrieval System: Junior AI Engineer Home Exam

## 🎯 1. Overview and Project Goal

This project implements the **Retrieval (R)** component of a Retrieval-Augmented Generation (RAG) pipeline. The system ingests a small text dataset, converts it into vector embeddings, stores them in a **Chroma** vector database, and retrieves the most relevant context chunks based on a user query.

**Note:** The LLM generation step is intentionally omitted as required by the assignment. 

---

## 📊 2. Reasoning and Design Choices (75% of Grade)

### A. Dataset Selection

| Parameter | Choice | Justification |
|-----------|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| **Source** | ["Why Hiring an Accountant Saves Time and Money" – Kaggle](https://www.kaggle.com/datasets/deancooperacca/why-hiring-an-accountant-saves-time-and-money) | Small, clean, text-based dataset (~8,200 characters), well under the 30,000-character requirement. |
| **Expected Queries** | Examples: “How much money can I save?”, “What accreditations should I look for?”, “Why hire an accountant?” | The article contains rich semantically meaningful text suitable for robust testing of retrieval capabilities. |

### B. Vector Database Selection: ChromaDB

**Choice:** Chroma (local persistent mode)

**Justification:**
* No cloud credentials required (unlike Pinecone / Weaviate).
* Ideal for small prototype RAG systems, providing simplicity and rapid development.
* Persisted locally under `chroma_db/` allowing reuse after ingestion.

### C. Embeddings and RAG Parameters (Based on `src/config.py`)

| Parameter | Choice | Justification |
|-----------|--------------------------|----------------------------------------------------------------------------------------------------------------------|
| **Embeddings Model** | `text-embedding-3-small` | Adheres to the assignment requirement; offers state-of-the-art quality at a negligible cost ($<\$1$). |
| **Chunk Size / Overlap** | **500 chars** / **50 chars** | Large enough to preserve context (e.g., a full paragraph) but small enough for targeted retrieval. Overlap prevents context loss between chunk boundaries. |
| **Top-K** | **3** | Returns the closest 3 chunks per query, ensuring sufficient context without adding too much noise. |
| **Distance Filter** | **MAX\_DISTANCE = 1.35** | Acts as a **quality gate**. Only chunks where the distance is $\le 1.35$ are considered relevant and returned to the user, regardless of the `Top-K` value. |

#### Important: Distance vs Similarity
Chroma's `similarity_search_with_score()` returns **distance** (L2 / Euclidean), where:
* $0$ = perfect semantic match.
* Lower distance = more similar.
* The system filters using: `distance <= MAX_DISTANCE`. This satisfies the “similarity threshold” requirement.

---

## 💻 3. Technical Implementation and Usage

The system runs on a **Single-Server Architecture**, leveraging FastAPI's capabilities for both serving the API and the static user interface:

* **Backend (API & UI Server):** FastAPI (Python) on **Port 9000** (or the default 8000 if not specified).
    * **Function:** Handles the core retrieval logic (`/api/retrieve`).
    * **Function:** Serves the static `index.html` file (`/`) and other assets (`/static`).
* **Frontend (UI):** The client-side `index.html` (HTML) makes a **relative fetch request** to the same host's `/api/retrieve` endpoint.
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

1.  **Start the Single FastAPI Server:** (Terminal 1)

    ```bash
    uvicorn src.main:app --reload --port 9000 --env-file .env
    ```

    *(The `--env-file .env` ensures the API key is loaded correctly.)*

2.  **Access UI:**
    Open your browser to: **`http://127.0.0.1:9000/`** ```

### D. Testing Example

| Query | Expected Result (Consolidated Context) | Score Type |
| :--- | :--- | :--- |
| "how much money can I save?" | Context containing "20% per year" and "pays for itself many times over." | Highest raw similarity score from merged chunks. |
| "accreditations I should look for" | Context containing "ACCA, ICAEW, or CIMA." | |

---

