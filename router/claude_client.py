import anthropic
from router.config import ANTHROPIC_API_KEY, CLAUDE_RESPONSE_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def ask(prompt: str) -> str:
    """Send a prompt to Claude and return the text response."""
    message = _client.messages.create(
        model=CLAUDE_RESPONSE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
