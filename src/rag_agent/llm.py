"""LLM provider factory.

Returns a LangChain BaseChatModel so every module works with the same
interface regardless of whether the backend is Anthropic or NVIDIA NIM.

Supported providers (set LLM_PROVIDER env var):
  anthropic  — Claude via the Anthropic API (default)
  nvidia     — Any NIM-hosted model via the OpenAI-compatible NIM endpoint
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from rag_agent.settings import Settings


def get_llm(settings: Settings, max_tokens: int = 1024) -> BaseChatModel:
    """Return a configured LangChain chat model for the active provider."""
    if settings.llm_provider == "nvidia":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.nvidia_model,
            api_key=settings.nvidia_api_key,  # type: ignore[arg-type]
            base_url=settings.nvidia_base_url,
            max_tokens=max_tokens,
            temperature=0,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=0,
    )
