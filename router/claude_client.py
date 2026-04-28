import anthropic
from router.config import ANTHROPIC_API_KEY, CLAUDE_RESPONSE_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------------------------------------------------------------------------
# Energy estimation methodology
#
# INFERENCE:
#   0.24 - 2.9 Wh per query - typical AI exchange 500 - 1500 tokens (input + output)
#   midpoint energy: (0.24 + 2.9) / 2 = 1.54 Wh
#   midpoint tokens: (500 + 1500) / 2 = 1000 tokens
#   Coefficient -> 1.54 Wh per 1000 tokens
#   
#   # Source: Brookings (2025) https://brookings.edu/articles/...
#   # Source: IEA Energy and AI (2025) https://iea.org/reports/energy-and-ai
#   # Source: Patterson et al. (2021) https://arxiv.org/abs/2104.10350
#   # PUE: 1.2 (typical modern hyperscaler - Google/Microsoft/AWS public reports)
#
# TRAINING AMORTIZATION:
#   Training energy proxy: 1,000 - 5,000 MWh
#     Source: GPT-3 reported at ~1,287 MWh (Patterson et al. 2021).
#             Claude is undisclosed but likely in the same order of magnitude
#             for a model of comparable scale. Epoch AI model registry used
#             for cross-reference: https://epochai.org/data/notable-ai-models
#   Lifetime queries: 10B - 100B (estimated range for a widely deployed model)
#     Source: No public Anthropic figure. Range derived from:
#             - Google processes ~8.5B searches/day as scale reference
#             - Assumed Claude serves a fraction of that volume
#   Per-query training attribution:
#     low:  1,000 MWh / 100B queries = 0.00001 Wh
#     high: 5,000 MWh /  10B queries = 0.0005  Wh
#     midpoint used in calculation:    0.00025 Wh
#
# IMPORTANT: These are speculative estimates based on public proxies.
#            Anthropic does not publish per-query or training energy figures.
#            All values should be interpreted as order-of-magnitude indicators
#            only, not measurements.
# ---------------------------------------------------------------------------

_INFERENCE_WH_PER_1K_TOKENS = 1.54
_PUE = 1.2
_TRAINING_WH_PER_QUERY_LOW = 0.00001
_TRAINING_WH_PER_QUERY_HIGH = 0.0005
_TRAINING_WH_PER_QUERY_MID = (_TRAINING_WH_PER_QUERY_LOW + _TRAINING_WH_PER_QUERY_HIGH) / 2


def ask(prompt: str) -> tuple[str, float]:
    """
    Send a prompt to Claude and return the text response plus energy metadata.
    Returns (response, eneergy_meta) where energy_meta contains:
        inference_wh    - estimated inference energy
        training_wh     - midpoint training amorization estimate
        training_low    - low bound training estimate
        training_high   - high bound training estimate
        tokens          - total tokens used
    """
    message = _client.messages.create(
        model=CLAUDE_RESPONSE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = next(
        block.text for block in message.content if hasattr(block, "text")
    )

    tokens = message.usage.input_tokens + message.usage.output_tokens
    inference_wh = (tokens / 1000) * _INFERENCE_WH_PER_1K_TOKENS * _PUE

    energy_meta = {
        "inference_wh": inference_wh,
        "training_wh": _TRAINING_WH_PER_QUERY_MID,
        "training_low": _TRAINING_WH_PER_QUERY_LOW,
        "training_high": _TRAINING_WH_PER_QUERY_HIGH,
        "tokens": tokens,
    }

    return response_text, energy_meta
