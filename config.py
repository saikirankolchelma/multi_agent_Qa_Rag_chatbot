# config.py

import os
import torch

class Config:
    # --- Model Configuration ---
    # Embedding and Reranker models remain the same
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # We are using a high-quality "quantized" model optimized for CPU.
    LLM_MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" 
    
    # THE FINAL, CORRECT FILENAME with case-sensitivity and period fix.
    LLM_MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    MODEL_CACHE_DIR = "models"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- RAG Configuration ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    INITIAL_RETRIEVAL_K = 10
    FINAL_RETRIEVAL_K = 3
    
    # --- Message Bus Configuration (Redis) ---
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    
    # --- Logging Configuration ---
    LOG_LEVEL = "INFO"
    LOG_FORMAT = '%(asctime)s - [%(name)s:%(levelname)s] - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
config = Config()