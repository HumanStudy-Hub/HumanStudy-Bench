"""
OpenAI API client (also used for OpenAI-compatible endpoints like xAI).
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from generation_pipeline.settings import PROVIDER_API_KEY_ENV
from src.llm.base import BaseLLMClient


def _image_to_data_url(image: Union[str, bytes, Path], mime: Optional[str] = None) -> str:
    """Convert image path/bytes/base64 to data URL for OpenAI content."""
    if isinstance(image, Path):
        image = str(image)
    if isinstance(image, str):
        if image.startswith("data:"):
            return image
        path = Path(image)
        if path.exists():
            mime = mime or "image/png"
            if path.suffix.lower() in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif path.suffix.lower() == ".webp":
                mime = "image/webp"
            data = path.read_bytes()
            b64 = base64.standard_b64encode(data).decode("ascii")
            return f"data:{mime};base64,{b64}"
        b64 = image
        mime = mime or "image/png"
        return f"data:{mime};base64,{b64}"
    if isinstance(image, bytes):
        b64 = base64.standard_b64encode(image).decode("ascii")
        mime = mime or "image/png"
        return f"data:{mime};base64,{b64}"
    raise TypeError(f"image must be path, bytes, or base64 str, got {type(image)}")


def _normalize_content(content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize content to OpenAI format: list of {type, text} or {type, image_url}."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for part in content:
        if part.get("type") == "text":
            out.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image":
            url = _image_to_data_url(
                part["image"],
                mime=part.get("mime"),
            )
            out.append({"type": "image_url", "image_url": {"url": url}})
        else:
            out.append(part)
    return out


def _messages_to_openai(messages: List[Dict[str, Any]], system: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert unified messages to OpenAI API format."""
    result = []
    if system:
        result.append({"role": "system", "content": system})
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system" and not result and not system:
            result.append({"role": "system", "content": content if isinstance(content, str) else content[0].get("text", "")})
            continue
        normalized = _normalize_content(content) if isinstance(content, list) else [{"type": "text", "text": content}]
        result.append({"role": role, "content": normalized})
    return result


class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions API (and OpenAI-compatible endpoints)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        api_key = api_key or os.getenv(PROVIDER_API_KEY_ENV["openai"])
        if not api_key:
            raise ValueError(f"{PROVIDER_API_KEY_ENV['openai']} not set and api_key not provided")
        super().__init__(model=model, api_key=api_key, api_base=api_base)

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": 0,
        }
        if timeout is not None and timeout > 0:
            client_kwargs["timeout"] = timeout
        client = OpenAI(**client_kwargs)
        openai_messages = _messages_to_openai(messages, system=system)
        kwargs: dict = {
            "model": self.model,
            "messages": openai_messages,
        }
        _default_only = self.model.startswith(("o1", "o3", "o4", "gpt-5"))
        if temperature is not None and not _default_only:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            if _default_only:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
