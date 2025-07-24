import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import config

def download_all_models():
    """Downloads and caches all models specified in the config file."""
    print(f"Creating models directory at: {os.path.abspath(config.MODEL_CACHE_DIR)}")
    os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)

    models_to_download = [
        (SentenceTransformer, config.EMBEDDING_MODEL_NAME),
        (CrossEncoder, config.RERANKER_MODEL_NAME),
    ]

    for model_class, model_name in models_to_download:
        print(f"\n--- Downloading {model_name} ---")
        try:
            model_class(model_name, cache_folder=config.MODEL_CACHE_DIR)
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")

    # Download LLM separately due to different loading mechanism
    print(f"\n--- Downloading LLM: {config.LLM_MODEL_NAME} ---")
    try:
        AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME, cache_dir=config.MODEL_CACHE_DIR)
        AutoModelForCausalLM.from_pretrained(config.LLM_MODEL_NAME, cache_dir=config.MODEL_CACHE_DIR, trust_remote_code=True)
        print(f"Successfully downloaded {config.LLM_MODEL_NAME}")
    except Exception as e:
        print(f"Error downloading {config.LLM_MODEL_NAME}: {e}")

    print("\nAll model downloads attempted successfully!")

if __name__ == "__main__":
    download_all_models()