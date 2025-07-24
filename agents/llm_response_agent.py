# agents/llm_response_agent.py

from utils.inference import LocalInference, InferenceError
from utils.logging_config import logger

class LLMResponseAgent:
    def __init__(self, bus, inference_service: LocalInference):
        self.name = "LLMResponseAgent"
        self.bus = bus
        self.inference = inference_service
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self):
        # A robust prompt template designed for Mistral Instruct models
        system_prompt = """You are an expert AI assistant. Your task is to provide accurate and concise answers based *only* on the context provided.
- If the user asks for a summary, synthesize the key points from the context into a coherent summary.
- If the context does not contain the answer, you must state that the information is not available in the documents. Do not use outside knowledge."""
        
        # This is the Mistral instruction format
        template = f"<s>[INST] {system_prompt} [/INST]</s>\n[INST] Based on the context below:\n\n---\nContext:\n{{context}}\n---\n\nAnswer this question: {{query}} [/INST]"
        return template

    def handle_message(self, message):
        if message["type"] != "RETRIEVAL_RESULT":
            return
        
        context = "\n\n---\n\n".join(message["payload"]["top_chunks"])
        query = message["payload"]["query"]
        
        prompt = self.prompt_template.format(context=context, query=query)
        
        response_text = ""
        try:
            if not message["payload"]["top_chunks"]:
                response_text = "I couldn't find any relevant information in the uploaded documents to answer your question."
            else:
                logger.info(f"[{self.name}] Generating response for query: '{query}'")
                response_text = self.inference.generate_text(prompt)
        except InferenceError as e:
            logger.error(f"[{self.name}] Inference failed: {e}")
            response_text = str(e)
        except Exception as e:
            logger.error(f"[{self.name}] An unexpected error occurred: {e}", exc_info=True)
            response_text = "I encountered a critical error while generating a response."
        
        self.bus.send({
            "sender": self.name,
            "receiver": "Coordinator", 
            "type": "LLM_RESPONSE",
            "trace_id": message["trace_id"],
            "payload": {"answer": response_text.strip(), "sources": message["payload"]["sources"]}
        })