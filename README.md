# Advanced Agentic RAG Chatbot

This project implements a sophisticated, agent-based Retrieval-Augmented Generation (RAG) chatbot capable of answering questions from a knowledge base of multiple document types. The architecture is built on a scalable Model Context Protocol (MCP) using Redis, and features an advanced two-stage retrieval pipeline for high-accuracy answers.

### Demo Video & Presentation

*   **[Link to Your 5-Minute Video Presentation]**
*   **[Link to the Architecture Presentation PDF in this Repo]**

---

### Architecture and System Flow

The system uses a decoupled, agentic architecture where specialized agents communicate via a Redis message bus.

**System Flow Diagram:**
*To be created with a tool like [Excalidraw](https://excalidraw.com/) or PowerPoint and saved as `architecture.png` in the repo.*

![Architecture Diagram](architecture.png) 

**Workflow:**
1.  **UI (Streamlit)**: User uploads documents and submits a query.
2.  **Coordinator & Redis Bus**: An `INGEST` message is sent to the Redis queue for each file.
3.  **IngestionAgent**: Parses the documents into smaller, overlapping text chunks.
4.  **RetrievalAgent (Indexing)**: Receives chunks, converts them to embeddings using `all-MiniLM-L6-v2`, and stores them in a FAISS vector database.
5.  **RetrievalAgent (Retrieval)**: On a user query, it performs a two-stage retrieval:
    -   **Stage 1 (Recall)**: A fast FAISS search retrieves the top 10 potential chunks.
    -   **Stage 2 (Reranking)**: A `CrossEncoder` model (`ms-marco-MiniLM-L-6-v2`) reranks these chunks for semantic relevance, selecting the top 3. This significantly improves context quality.
6.  **LLMResponseAgent**: Receives the curated context, formats it into a detailed prompt, and generates a final answer using the generative LLM.
7.  **UI Display**: The final answer and its sources are displayed in the Streamlit chat interface.

---

### Tech Stack

| Component               | Technology                                | Reason for Choice                                           |
| ----------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| **UI Framework**        | Streamlit                                 | Rapid development of interactive data science applications. |
| **Agent Communication**   | Redis (Pub/Sub)                           | Provides a scalable, robust, and decoupled message bus.     |
| **Document Processing**   | LangChain                                 | A comprehensive library for loaders and text splitters.     |
| **Vector Database**     | FAISS                                     | Highly efficient in-memory similarity search.               |
| **Embedding Model**     | `all-MiniLM-L6-v2`                        | Excellent balance of speed and performance for embeddings.  |
| **Reranking Model**     | `ms-marco-MiniLM-L-6-v2`                  | Boosts retrieval accuracy for higher quality context.       |
| **Generative LLM**      | `stabilityai/stablelm-zephyr-3b`          | A powerful 3B parameter model for high-quality generation.  |
| **Core Language**       | Python                                    | The standard for AI and data science development.           |

---

### Setup & How to Run

**Prerequisites:**
- Python 3.9+
- Redis Server (run easily with Docker)

**1. Clone the repository:**
```bash
git clone [Your-GitHub-Repo-Link]
cd agentic-rag-project
```

**2. Create and activate a Python virtual environment:**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Start Redis Server using Docker:**
```bash
docker run -d --name rag-redis -p 6379:6379 redis
```

**5. Run the Streamlit Application:**
*The AI models will be downloaded automatically on the first run.*
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.
