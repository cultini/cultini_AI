"""List the Gemini models your API key can access.

Use this to pick a real model string for AZETTA_LLM_MODEL / AZETTA_EMBED_MODEL
instead of hardcoding a guess.

  uv run python -m app.list_models
"""

from __future__ import annotations

from google import genai

from app import config


def main() -> None:
    client = genai.Client(api_key=config.require_api_key())
    print("Models available to this key:\n")
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or getattr(
            m, "supported_generation_methods", None
        )
        print(f"  {m.name}    {actions or ''}")


if __name__ == "__main__":
    main()
