import ollama
from router.config import OLLAMA_MODEL, OLLAMA_URL


def ask(prompt: str) -> str:
    """Send a prompt to the local Ollama model and return the text response."""
    client = ollama.Client(host=OLLAMA_URL)
    response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response["response"]
