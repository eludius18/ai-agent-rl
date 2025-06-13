

import logging
import requests
from langchain_ollama import ChatOllama

def is_ollama_running(host="http://localhost:11434"):
    try:
        response = requests.get(f"{host}/api/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        logging.error("❌ Ollama server not running at http://localhost:11434")
        return False

def get_available_models(host="http://localhost:11434"):
    try:
        response = requests.get(f"{host}/api/tags")
        return [model["name"] for model in response.json().get("models", [])]
    except Exception as e:
        logging.error(f"❌ Could not fetch models: {e}")
        return []

def get_llm():
    if not is_ollama_running():
        raise RuntimeError("❌ Ollama is not running. Start it with `ollama serve`.")

    available_models = get_available_models()
    model_name = next((m for m in available_models if "mistral" in m), None)

    if not model_name:
        raise RuntimeError("❌ No Mistral model found. Run `ollama pull mistral`.")

    logging.info(f"✅ Using local model: {model_name}")

    return ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.3
    )