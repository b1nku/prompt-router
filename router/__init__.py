from router.classifier import classify
from router import claude_client, ollama_client


def route(prompt: str) -> tuple[str, str, str]:
    """
    Classify and respond to a prompt.
    Returns (route, reason, response).
    """
    destination, reason = classify(prompt)

    if destination == "claude":
        response = claude_client.ask(prompt)
    else:
        response = ollama_client.ask(prompt)

    return destination, reason, response
