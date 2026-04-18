"""
LLM factory — returns the appropriate LLM client based on settings.LLM_PROVIDER.
"""
from docchat.config.settings import settings


def get_llm(temperature: float = 0, max_tokens: int = None, num_ctx: int = 4096):
    """
    Return a LangChain chat model based on the configured provider.

    Parameters
    ----------
    temperature : float
        Sampling temperature.
    max_tokens : int | None
        Max tokens to generate (used by Azure; maps to num_predict for Ollama).
    num_ctx : int
        Context window size (Ollama only).
    """
    if settings.LLM_PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI

        kwargs = dict(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=temperature,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return AzureChatOpenAI(**kwargs)

    else:
        from langchain_ollama import ChatOllama

        kwargs = dict(
            model=settings.LLM_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        if max_tokens is not None:
            kwargs["num_predict"] = max_tokens
        return ChatOllama(**kwargs)
