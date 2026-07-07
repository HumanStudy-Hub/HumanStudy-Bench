"""
Anthropic Claude Messages API client.
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from generation_pipeline.settings import PROVIDER_API_KEY_ENV
from src.llm.base import BaseLLMClient


def _image_to_base64(image: Union[str, bytes, Path], mime: Optional[str] = None) -> tuple[str, str]:
    """Return (base64_str, media_type) for Anthropic content block."""
    if isinstance(image, Path):
        image = str(image)
    if isinstance(image, str):
        if image.startswith("data:"):
            rest = image.split(",", 1)
            if len(rest) == 2:
                mt = rest[0].replace("data:", "").replace(";base64", "").strip()
                return rest[1].strip(), mt
            return rest[0], "image/png"
        path = Path(image)
        if path.exists():
            media_type = mime or "image/png"
            if path.suffix.lower() in (".jpg", ".jpeg"):
                media_type = "image/jpeg"
            elif path.suffix.lower() == ".webp":
                media_type = "image/webp"
            data = path.read_bytes()
            return base64.standard_b64encode(data).decode("ascii"), media_type
        return image, mime or "image/png"
    if isinstance(image, bytes):
        return base64.standard_b64encode(image).decode("ascii"), mime or "image/png"
    raise TypeError(f"image must be path, bytes, or base64 str, got {type(image)}")


def _content_to_anthropic(content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert unified content to Anthropic content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for part in content:
        if part.get("type") == "text":
            out.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image":
            data, media_type = _image_to_base64(part["image"], mime=part.get("mime"))
            out.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        else:
            out.append(part)
    return out


def _messages_to_anthropic(messages: List[Dict[str, Any]], system: Optional[str] = None) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Return (system, [content_blocks]) for Anthropic. System and first user message handled."""
    system_text = system
    blocks = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_text = content if isinstance(content, str) else (content[0].get("text", "") if content else "")
            continue
        if role == "user":
            blocks.extend(_content_to_anthropic(content))
        elif role == "assistant":
            # Anthropic expects alternating user/assistant; we flatten to single user content + optional system
            pass
    return system_text, blocks


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude Messages API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        api_key = api_key or os.getenv(PROVIDER_API_KEY_ENV["anthropic"])
        if not api_key:
            raise ValueError(f"{PROVIDER_API_KEY_ENV['anthropic']} not set and api_key not provided")
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
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        kwargs = {"api_key": self.api_key}
        if self.api_base:
            kwargs["base_url"] = self.api_base
        if timeout is not None and timeout > 0:
            kwargs["timeout"] = timeout
        client = Anthropic(**kwargs)
        system_text, content_blocks = _messages_to_anthropic(messages, system=system)
        if not content_blocks:
            raise ValueError("At least one user content block required")
        # Single user message: merge all text parts for simplicity
        text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        image_blocks = [b for b in content_blocks if b.get("type") == "image"]
        user_content = []
        if text_parts:
            user_content.append({"type": "text", "text": "\n".join(text_parts)})
        user_content.extend(image_blocks)

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or 8192,
            "messages": [{"role": "user", "content": user_content}],
        }
        if system_text:
            kwargs["system"] = system_text
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = client.messages.create(**kwargs)
        return (response.content[0].text if response.content else "") or ""
