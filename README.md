# Advanced Agentic RAG Chatbot

This project implements a sophisticated, agent-based Retrieval-Augmented Generation (RAG) chatbot capable of answering complex questions from a knowledge base of multiple document types. The architecture is built on a scalable Model Context Protocol (MCP) using Redis and features an advanced two-stage retrieval pipeline to ensure high-accuracy, contextually relevant answers.

### Demo Video & Presentation

*   **[Link to Your 5-Minute Video Presentation]**
*   **[Link to the Architecture Presentation PDF in this Repo]**

---

## Architecture and System Flow

The system is designed with a decoupled, agentic architecture where specialized agents communicate asynchronously via a Redis message bus. This design ensures scalability, modularity, and maintainability.

**System Flow Diagram:**
*(This is where you would place a generated diagram named `architecture.png`)*

![Architecture Diagram](architecture.png) 

### Workflow Steps:

1.  **UI & Ingestion Trigger (Streamlit)**: The user interacts with a Streamlit-based web UI to upload documents. For each new file, a message is dispatched to the Redis message queue.

2.  **Agent 1: IngestionAgent (Document Processing)**:
    *   **Responsibility**: To parse and chunk documents.
    *   **Action**: It consumes `INGEST` messages, uses LangChain's document loaders (`PyPDFLoader`, `UnstructuredWordDocumentLoader`, etc.) to read the file content, and then employs a `RecursiveCharacterTextSplitter` to break the text into smaller, overlapping chunks of 1000 characters. The 200-character overlap is crucial for maintaining context across chunk boundaries.

3.  **Agent 2: RetrievalAgent (Indexing)**:
    *   **Responsibility**: To create and manage the knowledge base.
    *   **Action**: It consumes `ADD_DOCUMENT` messages containing the text chunks. Using the `all-MiniLM-L6-v2` embedding model, it converts each chunk into a numerical vector and stores it in a high-performance **FAISS** vector database.

4.  **User Query & Retrieval Trigger**: The user asks a question in the chat interface. A `RETRIEVE` message containing the query is sent to the Redis queue.

5.  **RetrievalAgent (Two-Stage Retrieval)**:
    *   **Responsibility**: To find the most relevant information for a given query. This is the core of the RAG pipeline's accuracy.
    *   **Stage 1: Vector Search (Recall)**: A fast, broad search is performed on the FAISS index to find the top 10 chunks that are mathematically closest to the query's embedding.
    *   **Stage 2: Cross-Encoder Reranking (Precision)**: These 10 candidates are then passed to a more powerful `CrossEncoder` model (`ms-marco-MiniLM-L-6-v2`). This model compares the query directly against each chunk and re-orders them based on true semantic relevance, selecting the top 3. This step is critical for eliminating noise and providing a clean, focused context to the final LLM.

6.  **Agent 3: LLMResponseAgent (Answer Generation)**:
    *   **Responsibility**: To generate a human-like answer.
    *   **Action**: It receives the curated top 3 chunks and the original query. It uses a sophisticated prompt template to instruct the final LLM on its task, persona, and constraints (e.g., "answer only from the context").

7.  **UI Display**: The final generated text and a list of the source documents are sent back to the Streamlit UI and displayed to the user.

---

## Model Selection and Rationale

The selection of AI models was an iterative process focused on balancing performance, resource consumption, and the quality of results.

| Component          | Final Model Selected                    | Other Models Considered                   | Rationale for Final Choice                                                                                                                              |
| ------------------ | --------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Embedding**      | `all-MiniLM-L6-v2`                        | N/A                                       | This model is an industry standard for its excellent balance of speed, small size, and high performance in semantic search tasks. It was a clear first choice. |
| **Reranking**      | `ms-marco-MiniLM-L-6-v2`                  | N/A                                       | Chosen specifically for its proven ability to significantly boost RAG accuracy. Implementing a reranking step was a core design decision.                    |
| **Generative LLM** | `stabilityai/stablelm-zephyr-3b`        | `google/flan-t5-base`, `Mistral-7B-Instruct` | **`Flan-T5`** was used for initial pipeline testing due to its speed but lacked the ability to perform complex summarization. **`Mistral-7B`** offered the highest quality but its resource requirements (~15 GB) were prohibitive for consumer hardware. **`Zephyr-3B`** provided the perfect compromise: it's a powerful instruction-following model capable of high-quality summarization and Q&A, with a manageable size (~6 GB). |

---

## Tech Stack Deep Dive

| Component               | Technology                                | Detailed Reason for Choice                                           |
| ----------------------- | ----------------------------------------- | ---------------------------------------------------------------------- |
| **UI Framework**        | Streamlit                                 | Enables rapid, Python-native development of interactive data science applications, perfect for prototyping and demos. |
| **Agent Communication**   | Redis (Pub/Sub)                           | Chosen over in-memory queues to provide a scalable, robust, and decoupled message bus that allows agents to operate as independent microservices. |
| **Document Processing**   | LangChain                                 | Provides a standardized, comprehensive library for document loaders and advanced text splitters, accelerating development. |
| **Vector Database**     | FAISS                                     | Selected for its extreme efficiency and high speed in in-memory similarity search, crucial for a responsive user experience. |
| **Core Language**       | Python                                    | The standard for AI and data science development due to its rich ecosystem of libraries like PyTorch, Transformers, and LangChain. |

---

## Setup & How to Run

**Prerequisites:**
- Python 3.9+
- Redis Server (most easily run with Docker)
- Git

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

**3. Install dependencies from the requirements file:**
```bash
pip install -r requirements.txt
```

**4. Start the Redis Server in the background:**
```bash
docker run -d --name rag-redis -p 6379:6379 redis
```

**5. Run the Streamlit Application:**
*Note: The required AI models will be downloaded automatically from Hugging Face on the first run and cached in the `models/` directory.*
```bash
streamlit run app.py
```
The application will become available in your web browser, typically at `http://localhost:8501`.
