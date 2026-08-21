"""Provider-agnostic chat model construction.

Model names come from settings (INSTANT_LLM / POWERFUL_LLM). Plain names
infer their provider from the name ("gpt-4o-mini" -> OpenAI, needs
OPENAI_API_KEY); prefixed names select one explicitly
("groq:llama-3.1-8b-instant", needs GROQ_API_KEY). The byLLM counterpart
accepts the same values, so one env setting configures both implementations.
"""

from langchain.chat_models import init_chat_model


def get_chat_model(model_name: str, temperature: float = 0.0):
    """Build a chat model for the given provider-agnostic model name."""
    return init_chat_model(model_name, temperature=temperature)
