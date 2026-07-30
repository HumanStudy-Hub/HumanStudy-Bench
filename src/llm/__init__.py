"""
Unified multimodal LLM client abstraction.

Supports: openai, google, anthropic, openrouter, vllm.
"""

from src.llm.base import BaseLLMClient, Message, ContentPart
from src.llm.factory import get_client
from src.llm.helpers import generate_text, generate_json

__all__ = [
    "BaseLLMClient",
    "Message",
    "ContentPart",
    "get_client",
    "generate_text",
    "generate_json",
]
