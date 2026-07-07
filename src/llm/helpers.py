"""
Helper functions: generate_text and generate_json using any BaseLLMClient.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict, dataclass
import json
import re
import signal
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from src.llm.base import BaseLLMClient


class LLMTimeoutError(TimeoutError):
    """Raised when an LLM call exceeds the helper-level timeout."""


def call_with_timeout(fn: Callable[[], str], *, timeout: Optional[float]) -> str:
    return _call_with_timeout(fn, timeout=timeout)


@dataclass(frozen=True)
class LLMGenerationMetadata:
    attempts: int
    elapsed_seconds: float
    usage: Dict[str, int]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMTextResult:
    text: str
    metadata: LLMGenerationMetadata


@dataclass(frozen=True)
class LLMJSONResult:
    data: Dict[str, Any]
    text: str
    metadata: LLMGenerationMetadata


def generate_text(
    client: BaseLLMClient,
    messages: List[Dict[str, Any]],
    *,
    system: Optional[str] = None,
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    retries: int = 0,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None,
    count_tokens: bool = False,
    token_counter: Optional[Callable[[str], int]] = None,
    return_metadata: bool = False,
) -> str | LLMTextResult:
    """Convenience wrapper around client.generate_text with retry/timeout support."""
    start = time.monotonic()
    errors: list[str] = []
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            text = _call_with_timeout(
                lambda: client.generate_text(
                    messages=messages,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                ),
                timeout=timeout,
            )
            metadata = LLMGenerationMetadata(
                attempts=attempt + 1,
                elapsed_seconds=round(time.monotonic() - start, 3),
                usage=_token_usage(messages, system, text, count_tokens=count_tokens, token_counter=token_counter),
                errors=errors,
            )
            if return_metadata:
                return LLMTextResult(text=text, metadata=metadata)
            return text
        except Exception as exc:
            last_exc = exc
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt < retries and retry_delay > 0:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"LLM text generation failed after {retries + 1} attempt(s): {' | '.join(errors)}"
    ) from last_exc


def _strip_json_fence(text: str) -> str:
    """Remove markdown code fence around JSON."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def generate_json(
    client: BaseLLMClient,
    messages: List[Dict[str, Any]],
    *,
    system: Optional[str] = None,
    temperature: Optional[float] = 0.3,
    max_tokens: Optional[int] = None,
    json_suffix: str = "\n\nRespond with valid JSON only, no markdown code blocks.",
    retries: int = 1,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None,
    count_tokens: bool = False,
    token_counter: Optional[Callable[[str], int]] = None,
    return_metadata: bool = False,
) -> Dict[str, Any] | LLMJSONResult:
    """
    Generate text then parse as JSON. Appends json_suffix to the last user message.
    Strips markdown fences and retries on parse/API errors up to retries times.
    """
    # Append JSON instruction to last user message
    msgs = []
    last_user_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_idx = i
    for i, m in enumerate(messages):
        if i == last_user_idx and last_user_idx >= 0:
            content = m.get("content", "")
            if isinstance(content, str):
                content = content + json_suffix
            else:
                content = list(content) + [{"type": "text", "text": json_suffix}]
            msgs.append({**m, "content": content})
        else:
            msgs.append(dict(m))
    text = ""
    errors: list[str] = []
    combined_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    start = time.monotonic()
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            text_result = generate_text(
                client,
                msgs,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=0,
                timeout=timeout,
                count_tokens=count_tokens,
                token_counter=token_counter,
                return_metadata=True,
            )
            assert isinstance(text_result, LLMTextResult)
            text = text_result.text
            _merge_usage(combined_usage, text_result.metadata.usage)
            cleaned = _strip_json_fence(text)
            data = json.loads(cleaned)
            metadata = LLMGenerationMetadata(
                attempts=attempt + 1,
                elapsed_seconds=round(time.monotonic() - start, 3),
                usage=combined_usage if count_tokens else {},
                errors=errors,
            )
            if return_metadata:
                return LLMJSONResult(data=data, text=text, metadata=metadata)
            return data
        except json.JSONDecodeError as exc:
            last_exc = exc
            errors.append(f"JSONDecodeError: {exc}")
            if attempt < retries and last_user_idx >= 0:
                _append_to_message(msgs[last_user_idx], "\n\nFix the JSON and output again.")
        except Exception as exc:
            last_exc = exc
            errors.append(f"{type(exc).__name__}: {exc}")
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    raise ValueError(
        f"Failed to generate valid JSON after {retries + 1} attempt(s): {' | '.join(errors)}. "
        f"Response preview: {text[:500] if text else 'empty'}"
    ) from last_exc


def estimate_tokens(value: Any, token_counter: Optional[Callable[[str], int]] = None) -> int:
    """Estimate token count for text/messages; uses tiktoken when available."""
    text = _value_to_text(value)
    if token_counter is not None:
        return int(token_counter(text))
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        lexical = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
        return max(1, lexical, len(text) // 4)


def _call_with_timeout(fn: Callable[[], str], *, timeout: Optional[float]) -> str:
    if timeout is None or timeout <= 0:
        return fn()
    if threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer"):
        return _call_with_signal_timeout(fn, timeout=timeout)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise LLMTimeoutError(f"LLM call exceeded timeout={timeout}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _call_with_signal_timeout(fn: Callable[[], str], *, timeout: float) -> str:
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def _raise_timeout(signum: int, frame: Any) -> None:
        raise LLMTimeoutError(f"LLM call exceeded timeout={timeout}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _append_to_message(message: Dict[str, Any], extra: str) -> None:
    content = message.get("content", "")
    if isinstance(content, str):
        message["content"] = content + extra
    else:
        message["content"] = list(content) + [{"type": "text", "text": extra}]


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_value_to_text(item) for item in value)
    if isinstance(value, dict):
        if "content" in value:
            return _value_to_text(value.get("content"))
        if value.get("type") == "text":
            return str(value.get("text", ""))
        return "\n".join(_value_to_text(v) for v in value.values())
    return str(value)


def _token_usage(
    messages: List[Dict[str, Any]],
    system: Optional[str],
    completion: str,
    *,
    count_tokens: bool,
    token_counter: Optional[Callable[[str], int]],
) -> Dict[str, int]:
    if not count_tokens:
        return {}
    prompt_tokens = estimate_tokens([system or "", *messages], token_counter=token_counter)
    completion_tokens = estimate_tokens(completion, token_counter=token_counter)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _merge_usage(total: Dict[str, int], usage: Dict[str, int]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + usage.get(key, 0)
