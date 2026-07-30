"""
Base types and protocol for unified LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

# Content: str (text only) or list of parts for multimodal
# Part: {"type": "text", "text": str} | {"type": "image", "image": path_or_base64_or_bytes, "mime": optional}
ContentPart = Dict[str, Any]
MessageContent = Union[str, List[ContentPart]]


class Message:
    """Single message in a conversation."""

    def __init__(self, role: str, content: MessageContent):
        self.role = role
        self.content = content


class BaseLLMClient(ABC):
    """Abstract base for unified LLM clients (text + optional images)."""

    def __init__(self, model: str, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

    @abstractmethod
    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Generate text from messages.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str or list of parts}
            system: Optional system message (some providers require it separately)
            temperature: Sampling temperature
            max_tokens: Max output tokens
            timeout: Optional request timeout in seconds

        Returns:
            Generated text string
        """
        pass

    def generate_content(
        self,
        prompt: Union[str, List[Any]],
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Legacy-style single prompt generation (for pipeline components).
        If prompt is a list (e.g. [uploaded_file, text]), only text parts are used; file refs are ignored.
        """
        if isinstance(prompt, list):
            text_parts = []
            for p in prompt:
                if isinstance(p, str):
                    text_parts.append(p)
                # Skip non-string (e.g. uploaded file refs) when using unified client
            prompt = "\n\n".join(text_parts) if text_parts else ""
        return self.generate_text(
            messages=[{"role": "user", "content": prompt}],
            system=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
