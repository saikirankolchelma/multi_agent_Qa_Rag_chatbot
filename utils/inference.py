# utils/inference.py

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download
# --- THIS IS THE FINAL, CORRECT IMPORT ---
from ctransformers import AutoModelForCausalLM as CTransformersModel
from utils.logging_config import logger
from config import config

class ModelLoaderError(Exception): pass
class InferenceError(Exception): pass

class LocalInference:
    def __init__(self):
        logger.info("Initializing local inference services...")
        self.embedding_model = self._load_embedding_or_reranker_model(SentenceTransformer, config.EMBEDDING_MODEL_NAME)
        self.reranker_model = self._load_embedding_or_reranker_model(CrossEncoder, config.RERANKER_MODEL_NAME)
        self.text_generator = self._load_gguf_llm()
        logger.info(f"All models loaded successfully to device: {config.DEVICE}")

    def _load_embedding_or_reranker_model(self, model_class, model_name):
        try:
            logger.info(f"Loading model: {model_name}")
            return model_class(model_name, cache_folder=config.MODEL_CACHE_DIR, device=config.DEVICE)
        except Exception as e:
            raise ModelLoaderError(f"Failed to load {model_name}: {e}")
    
    def _load_gguf_llm(self):
        try:
            logger.info(f"Downloading and loading GGUF LLM: {config.LLM_MODEL_NAME}, file: {config.LLM_MODEL_FILE}")
            model_path = hf_hub_download(
                repo_id=config.LLM_MODEL_NAME,
                filename=config.LLM_MODEL_FILE,
                cache_dir=config.MODEL_CACHE_DIR
            )
            # We load the model directly without the problematic LangChain wrapper
            llm = CTransformersModel.from_pretrained(
                model_path,
                model_type="mistral",
                gpu_layers=0, # Ensures CPU usage for consistent performance
                context_length=4096,
            )
            logger.info("GGUF LLM loaded successfully for CPU execution.")
            return llm
        except Exception as e:
            raise ModelLoaderError(f"Failed to load GGUF LLM: {e}")

    def get_embeddings(self, texts):
        try:
            return self.embedding_model.encode(texts, convert_to_tensor=False).tolist()
        except Exception as e:
            raise InferenceError(f"Embedding encoding failed: {e}")

    def rerank_documents(self, query: str, documents: list[str]) -> list[str]:
        if not documents: return []
        try:
            pairs = [[query, doc] for doc in documents]
            scores = self.reranker_model.predict(pairs)
            return [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original order.")
            return documents

    def generate_text(self, prompt, max_new_tokens=512):
        try:
            # We call the model directly. No more .invoke()
            return self.text_generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7
            )
        except Exception as e:
            raise InferenceError(f"Text generation failed: {e}")