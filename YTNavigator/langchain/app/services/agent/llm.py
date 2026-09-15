"""Provider-agnostic chat model construction.

Model names come from settings (INSTANT_LLM / POWERFUL_LLM), i.e. $BENCH_MODEL.
"""

from langchain.chat_models import init_chat_model


def get_chat_model(model_name: str, temperature: float = 0.0):
    """Build a chat model for the given provider-agnostic model name."""
    # Strip the litellm provider prefix and speak OpenAI wire format to
    # OPENAI_BASE_URL (langchain-openai reads it natively), like the SDK arm.
    # tiktoken_model_name: trim_messages(token_counter=model) calls
    # ChatOpenAI.get_num_tokens_from_messages, which langchain-openai only
    # implements for gpt-*/o* names and raises NotImplementedError for anything
    # else (e.g. glm-5.2). Counting with gpt-4o's encoding is an approximation,
    # but it is only used to bound the history window at 1000 tokens.
    # gpt-5 / o-series bill their thinking against the completion budget and
    # take only the default temperature; "minimal" switches the thinking off,
    # matching the other two arms. Every other model keeps the pinned value.
    bare = model_name.split("/", 1)[-1]
    extra = ({"reasoning_effort": "minimal"}
             if bare.startswith(("gpt-5", "o1", "o3", "o4"))
             else {"temperature": temperature})
    return init_chat_model(
        bare, model_provider="openai", tiktoken_model_name="gpt-4o", **extra,
    )
