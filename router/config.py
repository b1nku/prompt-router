from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

CLAUDE_CLASSIFIER_MODEL: str = "claude-haiku-4-5-20251001"
CLAUDE_RESPONSE_MODEL: str = "claude-sonnet-4-6"

if not ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Check your .env file.")
