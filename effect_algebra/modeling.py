"""Shared Hugging Face model helpers for Colab training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required. In Colab choose Runtime > Change runtime type > GPU."
        )


def compute_dtype() -> Any:
    import torch

    require_cuda()
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def precision_flags() -> Dict[str, bool]:
    import torch

    bf16 = bool(torch.cuda.is_bf16_supported())
    return {"bf16": bf16, "fp16": not bf16}


def quantization_config() -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(),
    )


def load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_4bit_base(
    model_name: str,
    *,
    for_training: bool,
) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    require_cuda()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config(),
        torch_dtype=compute_dtype(),
        device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa",
    )
    if for_training:
        model.config.use_cache = False
    else:
        model.eval()
    return model


def make_lora_config(
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
    )


def prompt_token_lengths(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> List[int]:
    lengths: List[int] = []
    for row in rows:
        token_ids = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )
        lengths.append(len(token_ids))
    return lengths


def enforce_prompt_limit(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    max_prompt_length: int,
    *,
    dataset_name: str,
) -> Dict[str, float]:
    lengths = prompt_token_lengths(tokenizer, rows)
    if not lengths:
        raise ValueError("{} is empty".format(dataset_name))
    longest = max(lengths)
    if longest > max_prompt_length:
        raise ValueError(
            "{} has a {}-token prompt, exceeding --max-prompt-length {}. "
            "Increase the limit; truncating a stateful B episode is forbidden.".format(
                dataset_name,
                longest,
                max_prompt_length,
            )
        )
    return {
        "minimum": float(min(lengths)),
        "mean": float(sum(lengths) / len(lengths)),
        "maximum": float(longest),
    }


def adapter_config(path: Path) -> Dict[str, Any]:
    resolved = resolve_adapter_dir(path)
    return json.loads((resolved / "adapter_config.json").read_text(encoding="utf-8"))


def resolve_adapter_dir(path: Path) -> Path:
    path = Path(path)
    if (path / "adapter_config.json").exists():
        return path
    candidates = sorted(path.glob("*/adapter_config.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    raise FileNotFoundError(
        "could not resolve one adapter_config.json under {}".format(path)
    )


def comparable_adapter_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    target_modules = config.get("target_modules") or []
    return {
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": sorted(target_modules),
        "rank_pattern": config.get("rank_pattern") or {},
        "alpha_pattern": config.get("alpha_pattern") or {},
    }


def validate_adapter_compatibility(paths: Iterable[Path]) -> Dict[str, Any]:
    resolved_paths = [resolve_adapter_dir(path) for path in paths]
    signatures = [
        comparable_adapter_signature(adapter_config(path))
        for path in resolved_paths
    ]
    first = signatures[0]
    for path, signature in zip(resolved_paths[1:], signatures[1:]):
        if signature != first:
            raise ValueError(
                "adapter {} is incompatible with the first adapter:\n{}\n!=\n{}".format(
                    path,
                    json.dumps(signature, indent=2, sort_keys=True),
                    json.dumps(first, indent=2, sort_keys=True),
                )
            )
    return first


def load_adapter_for_evaluation(
    model: Any,
    adapter_path: Optional[Path],
) -> Any:
    if adapter_path is None:
        return model
    from peft import PeftModel

    resolved = resolve_adapter_dir(adapter_path)
    loaded = PeftModel.from_pretrained(model, str(resolved), is_trainable=False)
    loaded.eval()
    return loaded
