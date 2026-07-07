"""
Factory to get the appropriate LLM client by provider.
(Uses OpenAI-compatible clients for OpenAI, Google, OpenRouter, and vLLM.)
"""

import os
from typing import Optional

from generation_pipeline.settings import PROVIDER_API_KEY_ENV, PROVIDER_DEFAULT_API_BASE, SUPPORTED_PROVIDERS
from src.llm.base import BaseLLMClient
from src.llm.openai_client import OpenAIClient
from src.llm.anthropic_client import AnthropicClient


def get_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> BaseLLMClient:
    """
    Return a unified LLM client for the given provider.

    Args:
        provider: One of "openai", "google", "anthropic", "openrouter", "vllm"
        model: Model name (e.g. "claude-sonnet-4-6", "gpt-4o", "grok-2")
        api_key: Optional; otherwise read from provider-specific key env
        api_base: Optional; otherwise read from BASE_URL or provider default

    Returns:
        BaseLLMClient implementation
    """
    provider = (provider or "openai").lower().strip()
    key = api_key
    base = api_base or os.getenv("BASE_URL") or PROVIDER_DEFAULT_API_BASE.get(provider)

    if provider == "anthropic":
        key = key or os.getenv(PROVIDER_API_KEY_ENV[provider])
        return AnthropicClient(model=model, api_key=key, api_base=base)
    if provider == "openai":
        key = key or os.getenv(PROVIDER_API_KEY_ENV[provider])
        return OpenAIClient(model=model, api_key=key, api_base=base)
    if provider == "google":
        key = key or os.getenv(PROVIDER_API_KEY_ENV[provider])
        if not key:
            raise ValueError(f"{PROVIDER_API_KEY_ENV[provider]} not set and api_key not provided")
        return OpenAIClient(model=model, api_key=key, api_base=base)
    if provider == "openrouter":
        key = key or os.getenv(PROVIDER_API_KEY_ENV[provider])
        if not key:
            raise ValueError(f"{PROVIDER_API_KEY_ENV[provider]} not set and api_key not provided")
        return OpenAIClient(model=model, api_key=key, api_base=base)
    if provider == "vllm":
        key = key or os.getenv(PROVIDER_API_KEY_ENV[provider]) or "EMPTY"
        return OpenAIClient(model=model, api_key=key, api_base=base)

    raise ValueError(f"Unknown provider: {provider}. Use one of: {', '.join(SUPPORTED_PROVIDERS)}")
