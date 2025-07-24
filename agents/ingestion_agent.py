import os
from utils.file_utils import load_and_split_document
from utils.logging_config import logger

class IngestionAgent:
    def __init__(self, bus):
        self.name = "IngestionAgent"
        self.bus = bus

    def handle_message(self, message):
        if message["type"] != "INGEST":
            return

        file_path = message["payload"]["file_path"]
        file_name = message["payload"]["file_name"]

        try:
            logger.info(f"[{self.name}] Processing file: {file_name}")
            chunks = load_and_split_document(file_path)
            if not chunks:
                logger.warning(f"[{self.name}] Could not extract any chunks from {file_name}. It might be empty or an unsupported format.")
                return

            logger.info(f"[{self.name}] Processed {file_name} into {len(chunks)} chunks.")

            self.bus.send({
                "sender": self.name,
                "receiver": "RetrievalAgent",
                "type": "ADD_DOCUMENT",
                "trace_id": message["trace_id"],
                "payload": {
                    "document_id": file_name,
                    "chunks": [chunk.page_content for chunk in chunks],
                    "metadatas": [chunk.metadata for chunk in chunks]
                }
            })
        except Exception as e:
            logger.error(f"[{self.name}] Error processing {file_name}: {e}", exc_info=True)
        finally:
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.unlink(file_path)