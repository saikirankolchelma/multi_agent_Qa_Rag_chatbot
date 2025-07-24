# app.py

import os
import uuid
import tempfile
import streamlit as st
from message_bus import RedisBus
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.inference import LocalInference
from utils.logging_config import setup_logging
from config import config

# --- START OF THE FIX ---
def initialize_system():
    """Initialize and persist all stateful components across reruns."""
    setup_logging()
    
    # Initialize expensive, stateless singletons only once
    if "bus" not in st.session_state:
        st.session_state.bus = RedisBus()
    if "inference_service" not in st.session_state:
        st.session_state.inference_service = LocalInference()
    
    # Initialize stateful agents only once
    if "agents" not in st.session_state:
        st.session_state.agents = {
            "IngestionAgent": IngestionAgent(st.session_state.bus),
            "RetrievalAgent": RetrievalAgent(st.session_state.bus, st.session_state.inference_service),
            "LLMResponseAgent": LLMResponseAgent(st.session_state.bus, st.session_state.inference_service),
        }
    
    # Always return the persisted objects from the session state
    return st.session_state.bus, st.session_state.agents
# --- END OF THE FIX ---

def process_agent_queues(agents):
    """Process all messages for each agent in a loop."""
    processed_message = True
    while processed_message:
        processed_message = False
        for agent_name, agent in agents.items():
            # The agent's name matches its queue name
            if not agent.bus.is_empty(agent.name):
                message = agent.bus.receive(agent.name, block=False)
                if message:
                    agent.handle_message(message)
                    processed_message = True

def main():
    st.set_page_config(page_title="Mutli  Agentic Rag", layout="wide")
    st.title("🧠 Multi Agent  QA Chatbot")

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    bus, agents = initialize_system()
    
    with st.sidebar:
        st.header("📁 Document Upload")
        st.info("Upload documents to build the knowledge base.")
        uploaded_files = st.file_uploader(
            "Supported formats: PDF, DOCX, PPTX, TXT, MD",
            type=["pdf", "docx", "pptx", "txt", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            if new_files:
                with st.spinner("Processing documents..."):
                    for file in new_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp_file:
                            tmp_file.write(file.getvalue())
                            file_path = tmp_file.name
                        
                        bus.send({
                            "sender": "Coordinator",
                            "receiver": "IngestionAgent",
                            "type": "INGEST",
                            "trace_id": str(uuid.uuid4()),
                            "payload": {"file_path": file_path, "file_name": file.name}
                        })
                        st.session_state.processed_files.add(file.name)
                    
                    process_agent_queues(agents)
                st.sidebar.success(f"{len(new_files)} new document(s) processed!")
    
    st.header("💬 Chat with your Documents")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                sources_text = ", ".join([os.path.basename(s) for s in msg["sources"] if s])
                if sources_text:
                    st.info(f"Retrieved from: **{sources_text}**")
    
    if prompt := st.chat_input("Ask a question about the content of your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            trace_id = str(uuid.uuid4())
            bus.send({
                "sender": "Coordinator",
                "receiver": "RetrievalAgent",
                "type": "RETRIEVE",
                "trace_id": trace_id,
                "payload": {"query": prompt}
            })
            process_agent_queues(agents)
            
            response_message = bus.receive("Coordinator", block=True, timeout=30)
            
            if response_message and response_message.get("trace_id") == trace_id:
                response_payload = response_message["payload"]
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_payload["answer"],
                    "sources": response_payload["sources"]
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Sorry, I could not retrieve an answer in time. Please try again."
                })
        st.rerun()

if __name__ == "__main__":
    main()