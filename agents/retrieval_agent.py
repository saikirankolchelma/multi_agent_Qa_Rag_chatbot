from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.inference import LocalInference
from utils.logging_config import logger
from config import config

class RetrievalAgent:
    def __init__(self, bus, inference_service: LocalInference):
        self.name = "RetrievalAgent"
        self.bus = bus
        self.inference = inference_service
        self.vector_store = None

        self.embedding_interface = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            cache_folder=config.MODEL_CACHE_DIR,
            model_kwargs={'device': config.DEVICE}
        )

    def handle_message(self, message):
        try:
            if message["type"] == "ADD_DOCUMENT":
                self._handle_add_document(message)
            elif message["type"] == "RETRIEVE":
                self._handle_retrieve(message)
        except Exception as e:
            logger.error(f"[{self.name}] Failed to handle message: {e}", exc_info=True)

    def _handle_add_document(self, message):
        chunks = message["payload"]["chunks"]
        metadatas = message["payload"]["metadatas"]
        doc_id = message['payload']['document_id']

        logger.info(f"[{self.name}] Creating or updating FAISS index for document: {doc_id}")
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts=chunks, embedding=self.embedding_interface, metadatas=metadatas
            )
        else:
            self.vector_store.add_texts(texts=chunks, metadatas=metadatas)
        logger.info(f"[{self.name}] Successfully added {doc_id} to vector store.")

    def _handle_retrieve(self, message):
        query = message["payload"]["query"]
        top_chunks, sources = [], []

        if self.vector_store:
            logger.info(f"[{self.name}] Retrieving documents for query: '{query}'")
            # 1. Initial retrieval
            results = self.vector_store.similarity_search(query, k=config.INITIAL_RETRIEVAL_K)
            initial_docs = {doc.page_content: doc.metadata for doc in results}

            # 2. Reranking for relevance
            logger.info(f"[{self.name}] Reranking {len(initial_docs)} documents.")
            reranked_chunks = self.inference.rerank_documents(query, list(initial_docs.keys()))

            # 3. Select top K and get their sources
            top_chunks = reranked_chunks[:config.FINAL_RETRIEVAL_K]
            final_sources = {initial_docs[chunk].get('source', '') for chunk in top_chunks}
            sources = list(final_sources)
            logger.info(f"[{self.name}] Finalized {len(top_chunks)} chunks from {len(sources)} source(s).")
        else:
            logger.warning(f"[{self.name}] Retrieval attempt failed: Vector store not initialized.")

        self.bus.send({
            "sender": self.name,
            "receiver": "LLMResponseAgent",
            "type": "RETRIEVAL_RESULT",
            "trace_id": message["trace_id"],
            "payload": {"query": query, "top_chunks": top_chunks, "sources": sources}
        })