import json
import anthropic
from router.config import ANTHROPIC_API_KEY, CLAUDE_CLASSIFIER_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM = """You are a routing classifier for an AI prompt router.
 
Decide if a prompt requires SIGNIFICANT resources or can be handled by a small local model.
 
Route to "claude" for:
- Complex reasoning or multi-step problems
- Code generation, debugging, or explanation
- Research, analysis, or synthesis
- Long-form writing or structured output
- Anything ambiguous but potentially complex
 
Route to "local" for:
- Simple factual questions with short answers
- Casual conversation or small talk
- Basic definitions or quick lookups
- Simple rephrasing or formatting
 
Respond ONLY with valid JSON. No markdown, no explanation, nothing else.
Format: {"route": "claude", "reason": "one short sentence"}
     or {"route": "local", "reason": "one short sentence"}"""


def classify(prompt: str) -> tuple[str, str]:
    """
    Classify a prompt as 'claude' or 'local'.
    Returns (route, reason).
    """
    message = _client.messages.create(
        model=CLAUDE_CLASSIFIER_MODEL,
        max_tokens=100,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw)
    return data["route"], data["reason"]
