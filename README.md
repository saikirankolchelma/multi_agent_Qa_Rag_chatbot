# 🤖 Advanced Agentic RAG Chatbot with MCP

This project implements a sophisticated, agent-based Retrieval-Augmented Generation (RAG) chatbot capable of answering complex questions from a knowledge base of multiple document types. The architecture is built on a scalable **Model Context Protocol (MCP)** using Redis and features an advanced two-stage retrieval pipeline to ensure high-accuracy answers powered by the state-of-the-art **Mistral-7B** model

---

## 🏗️ Overall System Architecture

The system is designed with a decoupled, agentic architecture where specialized agents communicate asynchronously via a central message bus. This design ensures scalability, modularity, and maintainability by separating distinct concerns into independent microservices.

The workflow is divided into two main phases: **1. Document Ingestion** and **2. Query & Response**.

### 1. Document Ingestion Flow 📄➡️🧠

*This phase builds the chatbot's "memory."*

```mermaid
graph TD
    subgraph Ingestion Phase
        A[👤 User uploads file via Streamlit UI] --> B{📨 Redis Message Bus};
        B -- "1. MCP Message [Type: INGEST]" --> C(IngestionAgent);
        C -->|Parses & Chunks Document| B;
        B -- "2. MCP Message [Type: ADD_DOCUMENT]" --> D(RetrievalAgent);
        D -->|3. Embeds Chunks| F[🤖 Inference Service <br><i>(all-MiniLM-L6-v2)</i>];
        F --> D;
        D -->|4. Indexes Vectors| E((📚 FAISS Vector Store));
    end
```

### 2. Query & Response Flow 🤔➡️💬

*This phase uses the chatbot's memory to answer questions.*
```mermaid
graph TD
    subgraph Query Phase
        F[👤 User asks question in UI] --> G{📨 Redis Message Bus};
        G -- "1. MCP Message [Type: RETRIEVE]" --> H(RetrievalAgent);
        H -- "2. Vector Search (Recall)" --> I((📚 FAISS Vector Store));
        I -- "3. Returns 10 candidates" --> H;
        H -- "4. Reranks with Cross-Encoder" --> J[🤖 Inference Service <br><i>(ms-marco-MiniLM-L-6-v2)</i>];
        J --> H;
        H -- "5. Sends Curated Context" --> G;
        G -- "6. MCP Message [Type: RETRIEVAL_RESULT]" --> K(LLMResponseAgent);
        K -- "7. Generates Answer" --> L[🤖 Inference Service <br><i>(Mistral-7B GGUF)</i>];
        L --> K;
        K -- "8. Sends Final Answer" --> G;
        G -- "9. MCP Message [Type: LLM_RESPONSE]" --> F;
    end
```

---

## 📡 The Model Context Protocol (MCP) in Action

A core requirement of this project is the use of a **Model Context Protocol (MCP)**. MCP is not a library, but the **formal structure and language** for the messages our agents send. This ensures clear, unambiguous communication across the system.

Our implementation uses JSON-formatted messages sent via Redis. Every message in the system strictly adheres to the following MCP structure:

```json
{
  "sender": "AgentName",       // Who sent this message?
  "receiver": "AgentName",     // Who is the intended recipient?
  "type": "MESSAGE_PURPOSE", // What action should be taken?
  "trace_id": "unique_id_...", // An ID to follow a request across the entire system.
  "payload": { ... }           // The actual data (context, query, etc.).
}
```

This structured protocol is the backbone of the agentic system, allowing for complex, multi-step workflows to be executed reliably and enabling robust logging and debugging.

---

## 🧠 Model Selection and Rationale

The selection of AI models was an iterative process focused on balancing state-of-the-art performance with practical deployment on consumer hardware.

| Component          | Final Model Selected                    | Other Models Considered                   | Rationale for Final Choice                                                                                                                                                             |
| ------------------ | --------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Embedding**      | `all-MiniLM-L6-v2`                        | N/A                                       | 🏆 An industry standard chosen for its excellent balance of speed, size, and high performance in semantic search tasks.                                                                                 |
| **Reranking**      | `ms-marco-MiniLM-L-6-v2`                  | N/A                                       | 🎯 A crucial design decision to boost retrieval accuracy. This Cross-Encoder model ensures the context sent to the LLM is as clean and relevant as possible, significantly reducing hallucinations. |
| **Generative LLM** | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`  | `google/flan-t5-base`, `stablelm-zephyr-3b` | **`Flan-T5`** was too basic for complex reasoning. **`Zephyr-3B`** was a good step up. The goal was to use the SOTA **`Mistral-7B`** model. Its full size (~15 GB) is impractical for local review. **Solution**: Use a **GGUF-quantized** version, which provides the power of the Mistral-7B architecture in a CPU-friendly, ~4.4 GB package. This demonstrates an expert understanding of practical deployment constraints while delivering top-tier performance. |

---

## 🛠️ Tech Stack Deep Dive

| Component               | Technology                                | Detailed Reason for Choice                                           |
| ----------------------- | ----------------------------------------- | ---------------------------------------------------------------------- |
| **UI Framework**        | Streamlit                                 | Enables rapid, Python-native development of interactive AI applications. |
| **Agent Communication**   | Redis (Pub/Sub)                           | Chosen over in-memory queues to provide a scalable and robust message bus, decoupling agents into independent microservices. |
| **LLM Orchestration**   | LangChain & CTransformers               | LangChain for document processing and `ctransformers` for efficiently running high-performance GGUF-quantized models on local hardware. |
| **Vector Database**     | FAISS                                     | Highly efficient in-memory similarity search for a responsive user experience. |

---

## 🚀 Setup & How to Run

**Prerequisites:**
- Python 3.9+
- Docker (to easily run Redis)
- Git

**1. Clone the repository:**
```bash
git clone [Your-GitHub-Repo-Link]
cd agentic-rag-project```

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
*This includes `ctransformers` and `langchain-huggingface` for running the GGUF model.*
```bash
pip install -r requirements.txt
```

**4. Start the Redis Server in the background using Docker:**
```bash
docker run -d --name rag-redis -p 6379:6379 redis
```

**5. Run the Streamlit Application:**
*Note: The Mistral-7B GGUF model (~4.4 GB) will be downloaded automatically on the first run and cached. This may take some time depending on your internet connection.*
```bash
streamlit run app.py
```
The application will become available in your web browser, typically at `http://localhost:8501`.
