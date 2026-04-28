from router.classifier import classify
from router import claude_client, ollama_client


def route(prompt: str) -> tuple[str, str, str, dict | float | None]:
    """
    Classify and respond to a prompt.
    Returns (route, reason, response, energy_meta).
    """
    destination, reason = classify(prompt)

    if destination == "claude":
        response, energy_meta = claude_client.ask(prompt)
    else:
        response, energy_meta = ollama_client.ask(prompt)

    return destination, reason, response, energy_meta
