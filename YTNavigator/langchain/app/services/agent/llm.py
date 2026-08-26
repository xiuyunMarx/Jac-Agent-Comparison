"""Provider-agnostic chat model construction.

Model names come from settings (INSTANT_LLM / POWERFUL_LLM), hard-coded to
ollama_chat/glm-5.2; no env override.
"""

from langchain.chat_models import init_chat_model


def get_chat_model(model_name: str, temperature: float = 0.0):
    """Build a chat model for the given provider-agnostic model name."""
    return init_chat_model(model_name, temperature=temperature)
