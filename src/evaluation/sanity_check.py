"""
Sanity Check Module for Response Extraction Validation

Checks if agent responses can be fully extracted by evaluator's parse functions.
If extraction fails, formats responses using deepseek formatter.
"""

import json
import re
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def is_refusal_response(response_text: str) -> bool:
    """
    Check if a response is a refusal (agent declined to answer).

    Looks for common refusal patterns like:
    - "I cannot", "I can't", "I'm unable"
    - "I don't", "I won't", "I will not"
    - "I apologize", "I'm sorry"
    - "I'm not able", "I'm not comfortable"
    - "I decline", "I refuse"
    - "As an AI", "As a language model" (often used in refusals)

    Args:
        response_text: The response text to check

    Returns:
        True if the response appears to be a refusal
    """
    if not response_text or not isinstance(response_text, str):
        return False

    text_lower = response_text.lower().strip()

    # Empty or very short responses might be refusals
    if len(text_lower) < 10:
        return False  # Too short to determine, don't count as refusal

    # Common refusal patterns
    refusal_patterns = [
        r'\bi\s+(cannot|can\'t|can not)\s+',
        r"\bi\s+(am|'m)\s+(unable|not able|not comfortable)",
        r'\bi\s+(do|don\'t|do not|won\'t|will not)\s+(want|wish|feel comfortable)',
        r'\bi\s+(apologize|am sorry|regret)',
        r'\bi\s+(decline|refuse|cannot answer|cannot provide)',
        r'\bas\s+(an|a)\s+(ai|artificial intelligence|language model|assistant)',
        r'\b(unable|cannot|can\'t)\s+to\s+(answer|respond|provide|give)',
        r'\b(not\s+appropriate|not\s+suitable|not\s+ethical)',
        r'\b(decline|refuse|cannot)\s+to\s+(answer|respond|provide)',
    ]

    for pattern in refusal_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    return False


def calculate_raw_failure_rate(benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate raw failure rate - includes format issues, empty responses, and refusals.

    Raw failures include:
    1. Empty/None responses
    2. Refusal responses (agent declined to answer)
    3. Responses that are too short or malformed

    Args:
        benchmark_data: The benchmark data dictionary

    Returns:
        {
            "raw_failed": int,
            "raw_total": int,
            "raw_failure_rate": float,  # Percentage
            "raw_failure_breakdown": {
                "empty": int,
                "refusal": int,
                "other": int
            }
        }
    """
    individual_data = benchmark_data.get("individual_data", [])

    if not individual_data:
        return {
            "raw_failed": 0,
            "raw_total": 0,
            "raw_failure_rate": 0.0,
            "raw_failure_breakdown": {"empty": 0, "refusal": 0, "other": 0}
        }

    # Check if data is flat or nested
    is_flat = len(individual_data) > 0 and 'responses' not in individual_data[0]

    raw_failed = 0
    raw_total = 0
    empty_count = 0
    refusal_count = 0
    other_count = 0

    if is_flat:
        # Flat structure: each item is a response
        for resp in individual_data:
            raw_total += 1
            response_text = resp.get("response_text", "")

            # Check for empty/None
            if not response_text or response_text == "None" or (isinstance(response_text, str) and not response_text.strip()):
                raw_failed += 1
                empty_count += 1
                continue

            # Check for refusal
            if is_refusal_response(response_text):
                raw_failed += 1
                refusal_count += 1
                continue

            # Check for very short responses (might be malformed)
            if len(response_text.strip()) < 5:
                raw_failed += 1
                other_count += 1
    else:
        # Nested structure: each item is a participant with multiple responses
        for participant in individual_data:
            for resp in participant.get("responses", []):
                raw_total += 1
                response_text = resp.get("response_text", "")

                # Check for empty/None
                if not response_text or response_text == "None" or (isinstance(response_text, str) and not response_text.strip()):
                    raw_failed += 1
                    empty_count += 1
                    continue

                # Check for refusal
                if is_refusal_response(response_text):
                    raw_failed += 1
                    refusal_count += 1
                    continue

                # Check for very short responses (might be malformed)
                if len(response_text.strip()) < 5:
                    raw_failed += 1
                    other_count += 1

    raw_failure_rate = (raw_failed / raw_total * 100.0) if raw_total > 0 else 0.0

    return {
        "raw_failed": raw_failed,
        "raw_total": raw_total,
        "raw_failure_rate": raw_failure_rate,
        "raw_failure_breakdown": {
            "empty": empty_count,
            "refusal": refusal_count,
            "other": other_count
        }
    }


def run_sanity_check(
    study_id: str,
    benchmark_file: Path,
    evaluator_path: Path
) -> Dict[str, Any]:
    """
    检查所有响应是否能被evaluator的正则表达式完全提取。

    Args:
        study_id: Study ID (e.g., "study_001")
        benchmark_file: Path to full_benchmark.json
        evaluator_path: Path to evaluator.py file

    Returns:
        {
            "all_passed": bool,
            "failed_responses": List[Dict],  # 包含participant_id, response_index, missing_q_numbers
            "total_checked": int,
            "passed": int,
            "failed": int
        }
    """
    # 1. 加载benchmark数据
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)

    # 2. 动态加载evaluator模块
    try:
        spec = importlib.util.spec_from_file_location(f"{study_id}_evaluator", evaluator_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{study_id}_evaluator"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"Failed to load evaluator for {study_id}: {e}")
        return {
            "all_passed": False,
            "failed_responses": [],
            "total_checked": 0,
            "passed": 0,
            "failed": 0,
            "error": str(e)
        }

    # 3. 获取evaluator的解析函数
    parse_func = getattr(module, 'parse_agent_responses', None)
    get_required_func = getattr(module, 'get_required_q_numbers', None)

    if not parse_func:
        logger.warning(f"No parse_agent_responses function found in {study_id}_evaluator")
        return {
            "all_passed": True,  # 无法检查，假设通过
            "failed_responses": [],
            "total_checked": 0,
            "passed": 0,
            "failed": 0,
            "warning": "No parse_agent_responses function found"
        }

    # 4. 检查每个响应
    failed_responses = []
    total_responses = 0
    skipped_responses = 0
    total_checked = 0
    passed = 0

    individual_data = benchmark_data.get("individual_data", [])

    # Check if data is flat structure (legacy format) or nested structure (from Stage 5)
    # Note: Flat structure is kept for backward compatibility with legacy results
    is_flat_structure = len(individual_data) > 0 and 'responses' not in individual_data[0]

    if is_flat_structure:
        # Handle flat structure: each item is a trial response
        for resp_idx, response in enumerate(individual_data):
            total_responses += 1
            participant_id = response.get("participant_id", resp_idx)
            response_text = response.get("response_text", "")

            # Skip if response_text is None, "None", or empty
            if not response_text or response_text == "None" or (isinstance(response_text, str) and not response_text.strip()):
                skipped_responses += 1
                logger.debug(f"Skipping response with empty/None text: participant {participant_id}, response {resp_idx}")
                continue

            total_checked += 1

            trial_info = response.get("trial_info", {})

            # 确定需要的Q编号
            if get_required_func:
                try:
                    required_q_numbers = get_required_func(trial_info)
                except Exception as e:
                    logger.warning(f"Error calling get_required_q_numbers: {e}")
                    required_q_numbers = set()
            else:
                # 如果没有get_required_q_numbers函数，无法检查
                logger.warning(f"No get_required_q_numbers function in {study_id}_evaluator")
                continue

            if not required_q_numbers:
                # 无法确定需要的Q编号，跳过
                continue

            # 使用evaluator的解析函数提取Q值
            try:
                parsed = parse_func(response_text)
            except Exception as e:
                logger.warning(f"Error parsing response: {e}")
                parsed = {}

            # 检查是否所有需要的Q编号都被提取
            extracted_q_numbers = set(parsed.keys())
            missing = required_q_numbers - extracted_q_numbers

            if missing:
                failed_responses.append({
                    "participant_id": participant_id,
                    "response_index": resp_idx,
                    "missing_q_numbers": list(missing),
                    "required_q_numbers": list(required_q_numbers),
                    "extracted_q_numbers": list(extracted_q_numbers),
                    "response_text_preview": response_text[:200]
                })
            else:
                passed += 1
    else:
        # Handle nested structure: each item is a participant with multiple responses
        for participant in individual_data:
            participant_id = participant.get("participant_id")
            for resp_idx, response in enumerate(participant.get("responses", [])):
                total_responses += 1
                response_text = response.get("response_text", "")

                # Skip if response_text is None, "None", or empty
                if not response_text or response_text == "None" or (isinstance(response_text, str) and not response_text.strip()):
                    skipped_responses += 1
                    logger.debug(f"Skipping response with empty/None text: participant {participant_id}, response {resp_idx}")
                    continue

                total_checked += 1

                trial_info = response.get("trial_info", {})

                # 确定需要的Q编号
                if get_required_func:
                    try:
                        required_q_numbers = get_required_func(trial_info)
                    except Exception as e:
                        logger.warning(f"Error calling get_required_q_numbers: {e}")
                        required_q_numbers = set()
                else:
                    # 如果没有get_required_q_numbers函数，无法检查
                    logger.warning(f"No get_required_q_numbers function in {study_id}_evaluator")
                    continue

                if not required_q_numbers:
                    # 无法确定需要的Q编号，跳过
                    continue

                # 使用evaluator的解析函数提取Q值
                try:
                    parsed = parse_func(response_text)
                except Exception as e:
                    logger.warning(f"Error parsing response: {e}")
                    parsed = {}

                # 检查是否所有需要的Q编号都被提取
                extracted_q_numbers = set(parsed.keys())
                missing = required_q_numbers - extracted_q_numbers

                if missing:
                    failed_responses.append({
                        "participant_id": participant_id,
                        "response_index": resp_idx,
                        "missing_q_numbers": list(missing),
                        "required_q_numbers": list(required_q_numbers),
                        "extracted_q_numbers": list(extracted_q_numbers),
                        "response_text_preview": response_text[:200]
                    })
                else:
                    passed += 1

    return {
        "all_passed": len(failed_responses) == 0,
        "failed_responses": failed_responses,
        "total_responses": total_responses,
        "skipped_responses": skipped_responses,
        "total_checked": total_checked,
        "passed": passed,
        "failed": len(failed_responses)
    }


def format_with_deepseek(
    raw_response: str,
    trial_info: Dict[str, Any],
    study_id: str,
    trial_prompt: Optional[str] = None,
    missing_q_numbers: Optional[List[str]] = None,
    required_q_numbers: Optional[List[str]] = None,
    extracted_q_numbers: Optional[List[str]] = None,
    debug: bool = False
) -> str:
    """
    使用deepseek格式化响应。
    复用llm_participant_agent的格式化逻辑。

    Args:
        raw_response: 原始响应文本
        trial_info: Trial信息（可能包含prompt）
        study_id: Study ID
        trial_prompt: 可选的trial prompt（用于提取RESPONSE_SPEC）

    Returns:
        格式化后的响应文本
    """
    import os
    from openai import OpenAI
    from src.core.study_config import get_study_config
    from pathlib import Path

    # 如果响应为空、None或字符串"None"，直接返回
    if not raw_response or raw_response == "None" or (isinstance(raw_response, str) and not raw_response.strip()):
        return raw_response if raw_response and raw_response != "None" else ""

    # 注意：即使响应中已经有部分Q格式，如果sanity check标记为失败，
    # 说明缺少某些Q，仍然需要格式化（formatter会提取所有能找到的Q）

    # 提取RESPONSE_SPEC
    trial_prompt_source = "provided as parameter"
    if not trial_prompt:
        # 尝试从trial_info中获取prompt
        trial_prompt = trial_info.get("trial_prompt", "")
        trial_prompt_source = "from trial_info" if trial_prompt else "not found in trial_info"

        # 如果没有保存的prompt，尝试从study config重新生成
        if not trial_prompt:
            try:
                study_dir = Path(f"data/studies/{study_id}")
                study_config = get_study_config(study_id, study_dir, {})
                if hasattr(study_config, 'get_prompt_builder'):
                    builder = study_config.get_prompt_builder()
                    # 使用trial_info重新构建prompt
                    trial_prompt = builder.build_trial_prompt(trial_info)
                    trial_prompt_source = "regenerated from study_config"
            except Exception as e:
                trial_prompt_source = f"regeneration failed: {e}"
                logger.debug(f"Could not regenerate trial_prompt: {e}")

    response_spec = ""
    response_spec_extracted = False
    if trial_prompt:
        response_spec_match = re.search(
            r'RESPONSE_SPEC\s*\([^)]*\)\s*:?\s*\n(.*?)(?=\n\n|\n[A-Z]|\Z)',
            trial_prompt,
            re.IGNORECASE | re.DOTALL
        )
        if response_spec_match:
            response_spec = response_spec_match.group(1).strip()
            response_spec_extracted = True
        else:
            response_spec_extracted = False

    if not response_spec:
        # 使用默认格式
        response_spec = """- Output ONLY answer lines in the format: Qk=<value>
- Use this format for ALL questions: Q1=X, Q2=Y, Q3=Z, etc."""

    # 构建 sanity check 信息部分
    sanity_info = ""
    if required_q_numbers or missing_q_numbers or extracted_q_numbers:
        sanity_info = "\n\nSANITY CHECK RESULTS:\n"
        if required_q_numbers:
            sanity_info += f"- Required Q numbers: {', '.join(sorted(required_q_numbers))}\n"
        if extracted_q_numbers:
            sanity_info += f"- Currently extracted: {', '.join(sorted(extracted_q_numbers))}\n"
        if missing_q_numbers:
            sanity_info += f"- MISSING Q numbers: {', '.join(sorted(missing_q_numbers))}\n"
            sanity_info += "  ⚠️  These Q numbers MUST be present in the formatted output!\n"

    # 构建格式化prompt
    format_prompt = f"""Convert the following participant response to the standardized format specified in RESPONSE_SPEC.

RESPONSE_SPEC (from trial instructions):
{response_spec}
{sanity_info}

ORIGINAL RESPONSE:
{raw_response}

INSTRUCTIONS:
- Extract ONLY the answers that are explicitly present in the original response
- Format them according to the RESPONSE_SPEC above
- Use Qk=<value> format for single responses per question
- Use Qk.n=<value> format for multiple responses per question (e.g., Q1.1=X, Q1.2=Y)
- Output ONLY the formatted answer lines, one per line
- CRITICAL: If an answer is missing in the original response, DO NOT fill it in or make up values - simply skip that Q number
- DO NOT generate or invent answers that are not in the original response
- Only extract what is actually present in the original response
- IMPORTANT: If you detect duplicate Q numbers (e.g., Q1.6 appears twice), check if one should be a different Q number (e.g., Q1.8) based on the sequence and context. Only correct obvious typos where the intended Q number is clear from context.
- PAY SPECIAL ATTENTION: If the SANITY CHECK shows missing Q numbers, look carefully in the original response for those Q numbers. They might be:
  * Written with a colon instead of equals (Q1.8: 7 instead of Q1.8=7)
  * Written as a duplicate of another Q number (Q1.6=4 when it should be Q1.8=4)
  * Written in a different format or location in the response

FORMATTED OUTPUT:"""

    try:
        # 使用可配置的formatter provider/model（env: FORMATTER_PROVIDER, FORMATTER_MODEL）
        formatter_provider = os.getenv("FORMATTER_PROVIDER", "openrouter").lower()
        formatter_model = os.getenv("FORMATTER_MODEL", "deepseek/deepseek-chat")

        # 构建 system prompt，包含 sanity check 信息
        system_content = "You are a data formatter. Extract ONLY the Q values that are explicitly present in the response. Do NOT fill in missing answers or generate new values. Only format what is actually present in the original response. Use Qk=<value> or Qk.n=<value> format as specified. If an answer is missing, skip that Q number entirely. If you detect duplicate Q numbers (e.g., Q1.6 appears twice), check if one should be a different Q number based on the sequence and context. Only correct obvious typos where the intended Q number is clear."

        if missing_q_numbers:
            system_content += f" IMPORTANT: The sanity check indicates that these Q numbers are missing: {', '.join(sorted(missing_q_numbers))}. Look carefully in the original response for these Q numbers - they might be written with colons, as duplicates, or in a different format."

        # Route to anthropic if configured
        if formatter_provider == "anthropic" or "claude" in formatter_model.lower():
            try:
                from anthropic import Anthropic
            except ImportError:
                logger.warning("anthropic package not installed, returning original response")
                return raw_response

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found, returning original response")
                return raw_response

            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=formatter_model,
                max_tokens=8192,
                temperature=0.1,
                system=system_content,
                messages=[{"role": "user", "content": format_prompt}],
            )
            formatted = response.content[0].text if response.content else ""
        else:
            # Use OpenAI SDK for openai/xai/openrouter
            if formatter_provider == "xai":
                api_key = os.getenv("XAI_API_KEY")
                base_url = "https://api.x.ai/v1"
            elif formatter_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = None
            else:  # openrouter (default)
                api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
                base_url = "https://openrouter.ai/api/v1"

            if not api_key:
                logger.warning(f"No API key found for formatter provider {formatter_provider}, returning original response")
                return raw_response

            client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )

            response = client.chat.completions.create(
                model=formatter_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": format_prompt}
                ],
                max_tokens=8192,
                temperature=0.1
            )

            formatted = response.choices[0].message.content.strip()

        # 清理formatted输出（移除可能的markdown代码块）
        if formatted.startswith("```"):
            # 移除markdown代码块标记
            lines = formatted.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            formatted = "\n".join(lines).strip()

        # 调试日志：记录第一个失败响应的详细信息
        if debug:
            print(f"\n{'='*80}")
            print(f"FORMATTER DEBUG - First Failed Response")
            print(f"{'='*80}")

            # Show trial_prompt source and content
            print(f"\n{'─'*80}")
            print("TRIAL PROMPT STATUS:")
            print(f"{'─'*80}")
            print(f"Source: {trial_prompt_source}")
            print(f"Length: {len(trial_prompt) if trial_prompt else 0} characters")
            print(f"Empty: {not trial_prompt}")

            # Show full trial_prompt
            if trial_prompt:
                print(f"\n{'─'*80}")
                print("FULL TRIAL PROMPT:")
                print(f"{'─'*80}")
                print(trial_prompt)
                print(f"{'─'*80}")
            else:
                print(f"\n⚠️  TRIAL PROMPT IS EMPTY!")
                print(f"   This means RESPONSE_SPEC will use default format")

            # Show RESPONSE_SPEC extraction
            print(f"\n{'─'*80}")
            print("RESPONSE_SPEC EXTRACTION:")
            print(f"{'─'*80}")
            print(f"Extracted from trial_prompt: {response_spec_extracted}")
            print(f"Using default format: {not response_spec_extracted}")
            print(f"\nRESPONSE_SPEC (what formatter sees):")
            print(f"{'─'*80}")
            print(response_spec)
            print(f"{'─'*80}")

            print(f"\nSystem Prompt:")
            print(f"{system_content}")
            print(f"\nUser Prompt (full):")
            print(f"{format_prompt}")

            print(f"\n{'─'*80}")
            print("ORIGINAL RESPONSE (full, with repr for hidden chars):")
            print(f"{'─'*80}")
            print(f"Raw (repr): {repr(raw_response)}")
            print(f"\nFormatted display:")
            print(raw_response)
            print(f"\nLine by line:")
            for line_num, line in enumerate(raw_response.splitlines(), 1):
                print(f"  {line_num:3d}: {repr(line)}")

            # Check for Q1.11
            if "Q1.11" in raw_response:
                print(f"\n✓ Q1.11 IS present in the response!")
                for match in re.finditer(r'Q1\.11[:\s=]+([^\n]+)', raw_response):
                    print(f"  Found: {match.group(0)}")
            else:
                print(f"\n✗ Q1.11 IS NOT present in the response")

            # Count Q numbers
            q_matches = re.findall(r'(Q\d+(?:\.\d+)?)', raw_response)
            print(f"\nQ numbers found: {sorted(set(q_matches))}")
            print(f"Total unique Q numbers: {len(set(q_matches))}")
            if required_q_numbers:
                print(f"Required Q numbers: {sorted(required_q_numbers)}")
                missing = set(required_q_numbers) - set(q_matches)
                if missing:
                    print(f"Missing Q numbers: {sorted(missing)}")

            print(f"\n{'─'*80}")
            print("FORMATTER RESPONSE:")
            print(f"{'─'*80}")
            print(f"Raw (repr): {repr(formatted)}")
            print(f"\nFormatted display:")
            print(formatted)

            print(f"\nComparison:")
            print(f"  Original length: {len(raw_response)}")
            print(f"  Formatted length: {len(formatted)}")
            print(f"  Changed: {formatted != raw_response}")
            print(f"{'='*80}\n")

        return formatted

    except Exception as e:
        logger.warning(f"Formatting failed: {e}, using original response")
        return raw_response


def format_failed_responses(
    study_id: str,
    benchmark_file: Path,
    failed_responses: List[Dict],
    evaluator_path: Path,
    num_workers: int = 32
) -> int:
    """
    使用deepseek formatter多线程格式化失败的响应。

    Args:
        study_id: Study ID
        benchmark_file: Path to full_benchmark.json
        failed_responses: 失败的响应列表
        evaluator_path: Path to evaluator.py（用于获取trial_prompt等信息）
        num_workers: 线程数

    Returns:
        成功格式化的响应数量
    """
    # 1. 加载benchmark数据
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)

    # 2. 创建响应索引映射
    response_map = {}
    for participant in benchmark_data.get("individual_data", []):
        participant_id = participant.get("participant_id")
        for resp_idx, response in enumerate(participant.get("responses", [])):
            key = (participant_id, resp_idx)
            response_map[key] = response

    # 3. 多线程格式化
    def format_single_response(failed_resp_and_idx):
        # 解包：failed_resp 和它在列表中的索引
        failed_resp, idx = failed_resp_and_idx
        key = (failed_resp["participant_id"], failed_resp["response_index"])
        response = response_map.get(key)
        if not response:
            logger.warning(f"Response not found for participant {failed_resp['participant_id']}, response {failed_resp['response_index']}")
            return False

        # 获取原始响应和trial_info
        raw_response = response.get("raw_response_text") or response.get("response_text", "")
        trial_info = response.get("trial_info", {})

        # Skip if response is None, "None", or empty
        if not raw_response or raw_response == "None" or (isinstance(raw_response, str) and not raw_response.strip()):
            logger.debug(f"Skipping formatting for empty/None response: participant {failed_resp['participant_id']}, response {failed_resp['response_index']}")
            return False

        # 尝试获取trial_prompt（如果保存在trial_info中）
        trial_prompt = trial_info.get("trial_prompt")

        # 获取 sanity check 信息
        missing_q_numbers = failed_resp.get("missing_q_numbers", [])
        required_q_numbers = failed_resp.get("required_q_numbers", [])
        extracted_q_numbers = failed_resp.get("extracted_q_numbers", [])

        # 使用deepseek formatter
        try:
            # 记录第一个失败的响应用于调试
            is_first_failed = (idx == 0)

            formatted = format_with_deepseek(
                raw_response,
                trial_info,
                study_id,
                trial_prompt,
                missing_q_numbers=missing_q_numbers,
                required_q_numbers=required_q_numbers,
                extracted_q_numbers=extracted_q_numbers,
                debug=is_first_failed  # 只对第一个失败响应记录详细日志
            )

            if formatted and formatted != raw_response and formatted.strip():
                # 更新response_text
                response["response_text"] = formatted
                return True
            else:
                if is_first_failed:
                    print(f"\n⚠️  Formatter returned same or empty response")
                    print(f"  Original: {raw_response}")
                    print(f"  Formatted: {formatted if formatted else 'EMPTY'}")
                logger.debug(f"Formatter returned same or empty response for participant {failed_resp['participant_id']}, response {failed_resp['response_index']}")
                return False
        except Exception as e:
            logger.warning(f"Error formatting response for participant {failed_resp['participant_id']}, response {failed_resp['response_index']}: {e}")
            return False

    formatted_count = 0
    errors = []

    # 打印第一个失败的响应详情用于调试
    if failed_responses:
        first_failed = failed_responses[0]
        key = (first_failed["participant_id"], first_failed["response_index"])
        response = response_map.get(key)
        if response:
            raw_resp = response.get("raw_response_text") or response.get("response_text", "")
            logger.info(f"Formatting first failed response: Participant {first_failed['participant_id']}, Response {first_failed['response_index']}")
            logger.info(f"  Raw response (first 200 chars): {raw_resp[:200]}")
            logger.info(f"  Missing Q: {first_failed['missing_q_numbers']}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 传递索引以便识别第一个失败的响应
        futures = {executor.submit(format_single_response, (fr, idx)): (fr, idx) for idx, fr in enumerate(failed_responses)}

        # 添加进度条
        with tqdm(total=len(failed_responses), desc="Formatting responses", unit="resp") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        formatted_count += 1
                except Exception as e:
                    errors.append(str(e))
                    logger.warning(f"Error formatting response: {e}")
                finally:
                    pbar.update(1)

    if errors and len(errors) <= 5:
        logger.info(f"Formatting errors (showing first {len(errors)}): {errors}")

    # 4. 保存更新后的benchmark数据
    if formatted_count > 0:
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)

    return formatted_count
