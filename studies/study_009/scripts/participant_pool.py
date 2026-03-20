"""
LLM Participant Agent - AI acts as a single human participant.

This agent simulates a single participant in a psychological experiment.
The LLM takes on the role of a human participant with specific characteristics
and responds to experimental trials.
"""

import os
from typing import Dict, Any, List, Optional, Callable
import json


def _v2_human_prompt(profile: Dict[str, Any]) -> str:
    return "You are participating in a psychology experiment as a human participant."


def _v3_human_plus_demo_prompt(profile: Dict[str, Any]) -> str:
    base_sentence = "You are participating in a psychology experiment as a human participant."
    age = profile.get("age", "unknown age")
    gender = profile.get("gender")
    education = profile.get("education", "college student")
    background = profile.get("background")
    identity_parts = [f"- Age: {age} years old"]
    if gender is not None:
        identity_parts.append(f"- Gender: {gender}")
    if background:
        identity_parts.append(f"- Background: {background}")
    else:
        identity_parts.append(f"- Education: {education}")
    identity_section = "\n".join(identity_parts)
    return f"""{base_sentence}

YOUR IDENTITY:
{identity_section}

Follow the experimenter's instructions and answer each task in the requested format.
Be concise. Do not add extra explanations unless explicitly asked."""


class SystemPromptRegistry:
    _presets: Dict[str, Any] = {}

    @classmethod
    def get_prompt(cls, name: str, profile: Dict[str, Any]) -> str:
        normalized = name.replace("-", "_")
        lookup = normalized if normalized in cls._presets else name
        if lookup not in cls._presets:
            raise ValueError(f"Unknown system prompt preset: {name}. Available: {list(cls._presets.keys())}")
        handler = cls._presets[lookup]
        if callable(handler):
            return handler(profile)
        try:
            return handler.format(**profile)
        except KeyError:
            return handler


SystemPromptRegistry._presets["empty"] = ""
SystemPromptRegistry._presets["v2_human"] = _v2_human_prompt
SystemPromptRegistry._presets["v3_human_plus_demo"] = _v3_human_plus_demo_prompt


class LLMParticipantAgent:
    """
    LLM agent acting as a single participant in an experiment.
    
    The agent:
    1. Takes on a participant profile (age, gender, personality traits, etc.)
    2. Reads experimental instructions
    3. Responds to individual trials based on the experimental context
    4. Generates responses that reflect human-like behavior and decision-making
    """
    
    def __init__(
        self,
        participant_id: int,
        profile: Dict[str, Any],
        model: str = "mistralai/mistral-nemo",
        api_key: Optional[str] = None,
        use_real_llm: bool = False,
        system_prompt_override: Optional[str] = None,
        api_base: Optional[str] = None,
        prompt_builder: Optional[Any] = None,
        system_prompt_preset: str = "v3_human_plus_demo",
        reasoning: str = "default",
        enable_reasoning: bool = False,
        temperature: float = 1.0
    ):
        """
        Initialize a participant agent.
        
        Args:
            participant_id: Unique identifier for this participant
            profile: Participant characteristics (age, gender, traits, etc.)
            model: LLM model to use (default: "mistralai/mistral-nemo" for OpenRouter)
                   Examples: "mistralai/mistral-nemo", "gpt-4", "gpt-3.5-turbo", "anthropic/claude-3-sonnet"
            api_key: API key for LLM service (OPENROUTER_API_KEY or OPENAI_API_KEY)
            use_real_llm: If True, makes actual API calls. If False, simulates responses.
            system_prompt_override: Optional custom system prompt (deprecated, use prompt_builder instead)
            api_base: Optional API base URL (default: "https://openrouter.ai/api/v1" for OpenRouter models)
            prompt_builder: Optional PromptBuilder for custom system prompt support
            system_prompt_preset: One of "empty", "v2_human", "v3_human_plus_demo"
            reasoning: Reasoning effort for supported models ("default", "none", "low", "medium", "high", "xhigh", "minimal")
            enable_reasoning: Force enable reasoning for OpenRouter models
            temperature: Sampling temperature for the LLM (default: 1.0)
        """
        self.participant_id = participant_id
        self.profile = profile
        self.model = model
        self.prompt_builder = prompt_builder
        self.system_prompt_preset = system_prompt_preset
        self.reasoning = reasoning
        self.enable_reasoning = enable_reasoning
        self.temperature = temperature
        
        # Infer provider from model name (backward compatible)
        self.provider = self._infer_provider(model, api_base)
        
        # Determine API key and base URL based on provider
        if self.provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.api_base = api_base
            self.is_openrouter = False
        elif self.provider == "xai":
            self.api_key = api_key or os.environ.get("XAI_API_KEY")
            self.api_base = api_base or "https://api.x.ai/v1"
            self.is_openrouter = False
        elif self.provider == "openrouter":
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            self.api_base = api_base or "https://openrouter.ai/api/v1"
            self.is_openrouter = True
        else:  # openai or default
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.api_base = api_base
            self.is_openrouter = False
        
        self.use_real_llm = use_real_llm
        self.system_prompt_override = system_prompt_override
        
        # Conversation history for maintaining context
        self.conversation_history = []
        self._conversation_initialized = False  # Track if start_conversation has been called
        
        # Trial responses
        self.trial_responses = []
        
        # Store formatter usage info
        self._last_formatter_usage = {}
        
        if use_real_llm and not self.api_key:
            key_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "xai": "XAI_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "openai": "OPENAI_API_KEY",
            }
            key_name = key_map.get(self.provider, "OPENAI_API_KEY")
            raise ValueError(
                f"API key required for real LLM usage. "
                f"Set {key_name} environment variable or pass api_key parameter."
            )
    
    def _infer_provider(self, model: str, api_base: Optional[str]) -> str:
        """
        Infer provider from model name or api_base (for backward compatibility).
        
        Returns: "openai", "anthropic", "xai", or "openrouter"
        """
        m = (model or "").lower()
        # Check API base first
        if api_base:
            if "openrouter.ai" in api_base:
                return "openrouter"
            if "x.ai" in api_base or "xai" in api_base:
                return "xai"
            if "anthropic.com" in api_base:
                return "anthropic"
        # Check model name patterns
        if "claude" in m or m.startswith("anthropic/"):
            return "anthropic"
        if "grok" in m or m.startswith("xai/") or m.startswith("x-ai/"):
            return "xai"
        if "/" in m:  # OpenRouter uses "provider/model" format
            return "openrouter"
        # Default to openai for simple model names like "gpt-4"
        return "openai"
    
    def _construct_system_prompt(self) -> str:
        """
        Construct system prompt based on the selected preset using the SystemPromptRegistry.
        """
        # 1. Legacy support: if system_prompt_override is provided, use it
        if self.system_prompt_override:
            return self.system_prompt_override
        
        # 2. Get prompt from registry
        base_prompt = SystemPromptRegistry.get_prompt(self.system_prompt_preset, self.profile)
        
        # 3. If prompt_builder has custom system prompt template, append it
        if self.prompt_builder and hasattr(self.prompt_builder, 'build_system_prompt'):
            try:
                custom_content = self.prompt_builder.build_system_prompt()
                if custom_content:
                    # Don't append to empty prompt
                    if not base_prompt:
                        return custom_content
                    return f"{base_prompt}\n\n{custom_content}"
            except Exception:
                pass
        
        return base_prompt
    
    def _clean_llm_response_text(self, text: str) -> str:
        """
        Ensures text is valid UTF-8, replacing invalid characters.
        
        Args:
            text: Raw text from LLM response
            
        Returns:
            Cleaned UTF-8 text
        """
        if not isinstance(text, str):
            return str(text)
        try:
            # Try to encode/decode to ensure it's valid UTF-8
            text.encode('utf-8')
            return text
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Replace invalid characters
            return text.encode('utf-8', errors='replace').decode('utf-8')
    
    def start_conversation(self, system_prompt: Optional[str] = None) -> None:
        """
        Initialize a new conversation session with a system prompt.
        
        Args:
            system_prompt: System prompt to initialize the conversation.
                          If None, uses the agent's default system prompt.
        """
        self.conversation_history = []
        self._conversation_initialized = True
        
        if system_prompt is None:
            system_prompt = self._construct_system_prompt()
        
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def continue_conversation(self, user_message: str, max_tokens: int = 8192) -> Dict[str, Any]:
        """
        Continue an existing conversation by adding a user message and getting a response.
        
        This method:
        1. Appends the user message to conversation history
        2. Calls the LLM with the full conversation history
        3. Appends the assistant's response to history
        4. Returns the assistant's response and usage info
        
        Args:
            user_message: User message to add to the conversation
            max_tokens: Maximum tokens for the response
            
        Returns:
            Dict containing "response_text" and "usage"
            
        Raises:
            ValueError: If conversation hasn't been started (history is empty and not initialized)
        """
        if not self.conversation_history and not getattr(self, '_conversation_initialized', False):
            raise ValueError(
                "Conversation not started. Call start_conversation() first, "
                "or use complete_trial() for stateless interactions."
            )
        
        # Append user message
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        usage_info = {}
        # Call LLM with full history
        if self.use_real_llm:
            llm_result = self._call_llm_with_history(self.conversation_history, max_tokens=max_tokens)
            response_text = llm_result.get("response_text", "")
            usage_info = llm_result.get("usage", {})
        else:
            # Simulated response for testing - try to extract round number from user message
            import re
            round_match = re.search(r"Round (\d+)", user_message)
            round_num = round_match.group(1) if round_match else "1"
            
            # Simple heuristic for simulated choices
            if round_num == "1":
                val = 50.0
            elif round_num == "2":
                val = 25.0
            elif round_num == "3":
                val = 12.5
            else:
                val = 6.25
                
            response_text = f"Q{round_num}={val}"
            usage_info = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0}
        
        # Append assistant response
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        
        return {"response_text": response_text, "usage": usage_info}
    
    def clear_conversation(self) -> None:
        """
        Clear the conversation history. Useful for resetting between trials.
        """
        self.conversation_history = []
        self._conversation_initialized = False
    
    def receive_instructions(self, instructions: str) -> str:
        """
        Participant receives and acknowledges experimental instructions.
        
        Args:
            instructions: Experimental instructions text
            
        Returns:
            Participant's acknowledgment or questions
        """
        system_prompt = self._construct_system_prompt()
        
        user_message = f"""The experimenter gives you the following instructions:

{instructions}

Proceed."""
        
        if self.use_real_llm:
            llm_result = self._call_llm(system_prompt, user_message)
            response = llm_result.get("response_text", "")
        else:
            response = "Yes, I understand. I'll judge which line matches the standard."
        
        # Store in history
        self.conversation_history.append({
            "role": "instructions",
            "content": instructions,
            "response": response
        })
        
        return response
    
    def _format_response(self, raw_response: str, trial_prompt: str, trial_info: Dict[str, Any]) -> str:
        """
        Format free-form response to standardized format based on RESPONSE_SPEC from trial_prompt.
        
        Extracts RESPONSE_SPEC from the trial prompt and uses it to format the response.
        Supports Qk=<value> and Qk.n=<value> formats as specified in RESPONSE_SPEC.
        
        Args:
            raw_response: Agent's raw response text
            trial_prompt: The original trial prompt containing RESPONSE_SPEC
            trial_info: Trial metadata (optional)
            
        Returns:
            Formatted response in Qk=<value> or Qk.n=<value> format as specified in RESPONSE_SPEC
        """
        import re
        
        # If raw_response is empty or None, return it as-is (don't try to format empty responses)
        if not raw_response or not raw_response.strip():
            return raw_response or ""
        
        # Quick check: if already in Q format, return as-is
        q_pattern = re.compile(r'Q\d+(?:\.\d+)?\s*=', re.IGNORECASE)
        if q_pattern.search(raw_response):
            # Already has Q format, return as-is
            return raw_response
        
        # Extract RESPONSE_SPEC from trial_prompt
        response_spec_match = re.search(
            r'RESPONSE_SPEC\s*\([^)]*\)\s*:?\s*\n(.*?)(?=\n\n|\n[A-Z]|\Z)',
            trial_prompt,
            re.IGNORECASE | re.DOTALL
        )
        
        if not response_spec_match:
            # No RESPONSE_SPEC found, use default format
            response_spec = """- Output ONLY answer lines in the format: Qk=<value>
- Use this format for ALL questions: Q1=X, Q2=Y, Q3=Z, etc."""
        else:
            response_spec = response_spec_match.group(1).strip()
        
        # Build formatting prompt using RESPONSE_SPEC
        format_prompt = f"""Convert the following participant response to the standardized format specified in RESPONSE_SPEC.

RESPONSE_SPEC (from trial instructions):
{response_spec}

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

FORMATTED OUTPUT:"""

        try:
            # Use deepseek for formatting (lightweight, fast)
            formatted_result = self._call_llm_with_model(
                system_prompt="You are a data formatter. Extract ONLY the Q values that are explicitly present in the response. Do NOT fill in missing answers or generate new values. Only format what is actually present in the original response. Use Qk=<value> or Qk.n=<value> format as specified. If an answer is missing, skip that Q number entirely. If you detect duplicate Q numbers (e.g., Q1.6 appears twice), check if one should be a different Q number based on the sequence and context. Only correct obvious typos where the intended Q number is clear.",
                user_message=format_prompt,
                model="deepseek/deepseek-v3.2",
                max_tokens=8192
            )
            # Store formatter usage info for later (will be added to response_data)
            self._last_formatter_usage = formatted_result.get("usage", {})
            return formatted_result.get("response_text", "").strip()
        except Exception as e:
            # Log error but fallback to original if formatting fails
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Formatting failed: {e}, using original response")
            return raw_response
    
    def _call_llm_with_model(self, system_prompt: str, user_message: str, model: str, max_tokens: int = 8192) -> Dict[str, Any]:
        """
        Call LLM with a specific model (for formatting agent).
        Supports env vars: FORMATTER_PROVIDER (openai/anthropic/xai/openrouter, default openrouter)
                          FORMATTER_MODEL (default deepseek/deepseek-v3.2)
        """
        import logging
        import time
        import random as _rnd
        logger = logging.getLogger(__name__)
        
        # Check for formatter config from env
        formatter_provider = os.getenv("FORMATTER_PROVIDER", "openrouter").lower()
        formatter_model = os.getenv("FORMATTER_MODEL", model).lower()
        
        # Route to anthropic if needed
        if formatter_provider == "anthropic" or "claude" in formatter_model:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY required for formatter")
            
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=formatter_model,
                max_tokens=max_tokens,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            result = response.content[0].text if response.content else ""
            usage_info = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
            }
            return {"response_text": result, "usage": usage_info, "full_api_response": {}}
        
        # Use OpenAI SDK for openai/xai/openrouter
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        # Determine API key and base URL
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
            raise ValueError(f"API key required for formatter provider: {formatter_provider}")
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0,
            max_retries=0
        )
        
        try:
            response = client.chat.completions.create(
                model=formatter_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,  # Low temperature for consistent formatting
                max_tokens=max_tokens
            )
            
            # Validate response
            if response is None:
                raise ValueError("LLM API returned None response")
            if not hasattr(response, 'choices') or not response.choices:
                raise ValueError("LLM API response has no choices")
            if len(response.choices) == 0:
                raise ValueError("LLM API response has empty choices list")
            
            # Capture full API response dictionary
            try:
                if hasattr(response, 'model_dump'):
                    # Pydantic v2
                    full_api_response = response.model_dump()
                elif hasattr(response, 'dict'):
                    # Pydantic v1
                    full_api_response = response.dict()
                else:
                    # Fallback: convert to dict manually
                    full_api_response = {
                        "id": getattr(response, 'id', None),
                        "object": getattr(response, 'object', None),
                        "created": getattr(response, 'created', None),
                        "model": getattr(response, 'model', None),
                        "choices": [
                            {
                                "index": getattr(choice, 'index', None),
                                "message": {
                                    "role": getattr(choice.message, 'role', None),
                                    "content": getattr(choice.message, 'content', None),
                                },
                                "finish_reason": getattr(choice, 'finish_reason', None),
                            }
                            for choice in response.choices
                        ],
                        "usage": {
                            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                            "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                            "total_tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                        }
                    }
            except Exception as e:
                logger.warning(f"Failed to capture full API response: {e}")
                full_api_response = {}
            
            # Extract usage information
            usage_info = {}
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                usage_info = {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                    "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                    "total_tokens": getattr(usage, 'total_tokens', 0) or 0
                }
                # Try to get cost if available (OpenRouter provides this)
                if hasattr(usage, 'total_cost') or hasattr(usage, 'cost'):
                    usage_info["cost"] = getattr(usage, 'total_cost', None) or getattr(usage, 'cost', None)
                elif hasattr(response, 'total_cost') or hasattr(response, 'cost'):
                    usage_info["cost"] = getattr(response, 'total_cost', None) or getattr(response, 'cost', None)
            
            result = response.choices[0].message.content.strip()
            # Sanitize response text to ensure UTF-8 compatibility
            result = self._clean_llm_response_text(result)
            return {"response_text": result, "usage": usage_info, "full_api_response": full_api_response}
        except Exception as e:
            # Silently fail formatting - fallback to original response
            raise
    
    def complete_trial(
        self,
        trial_prompt: str,  # 改为直接接受 prompt
        trial_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Participant completes a single experimental trial.
        
        Args:
            trial_prompt: Pre-built trial prompt (from PromptBuilder)
            trial_info: Optional metadata about the trial
            
        Returns:
            Participant's response and metadata
        """
        raw_response_text = ""  # Store raw API response (default to empty string, not None)
        usage_info = {}  # Store token usage and cost info
        formatter_used = False  # Track if formatter was used
        
        if self.use_real_llm:
            system_prompt = self._construct_system_prompt()
            # Determine max_tokens based on trial type
            max_tokens = self._get_max_tokens_for_trial(trial_info or {})
            llm_result = self._call_llm(system_prompt, trial_prompt, max_tokens=max_tokens)  # Save raw response
            raw_response_text = llm_result.get("response_text", "") or ""
            # Clean response text to ensure UTF-8 compatibility
            raw_response_text = self._clean_llm_response_text(raw_response_text)
            usage_info = llm_result.get("usage", {})
            full_api_response = llm_result.get("full_api_response", {})
            
            # Stage 5: No formatter - use raw response directly
            # Formatting will be done in Stage 6 if needed (sanity check)
            response_text = raw_response_text
            formatter_used = False
            
            # Parse response to extract choice
            choice = self._parse_response(response_text, trial_info or {})
        else:
            # Simulated response
            choice, response_text = self._simulate_response(trial_info or {}, None)
            raw_response_text = response_text  # In simulation mode, raw response is the same as response_text
        
        # Record response - include raw_response_text for later extraction
        response_data = {
            "participant_id": self.participant_id,
            "trial_number": trial_info.get("trial_number") if trial_info else len(self.trial_responses) + 1,
            "response": choice,
            "response_text": response_text,
            "raw_response_text": raw_response_text or "",  # Store raw response (formatted if formatter was used)
            "usage": usage_info,  # Store token usage and cost
            "formatter_used": formatter_used,  # Track if formatter was used
            "full_api_response": full_api_response if self.use_real_llm else {},  # Store complete API response dictionary
            "correct_answer": trial_info.get("correct_answer") if trial_info else None,
            "is_correct": choice == trial_info.get("correct_answer") if trial_info and trial_info.get("correct_answer") else None,
            "trial_info": trial_info
        }
        
        self.trial_responses.append(response_data)
        
        return response_data
    
    def _get_max_tokens_for_trial(self, trial_info: Dict[str, Any]) -> int:
        """
        Determine appropriate max_tokens based on trial type.
        
        Args:
            trial_info: Trial metadata
            
        Returns:
            Max tokens for this trial
        """
        # All trials use 8192 (4096 * 2) max tokens
        return 8192
    
    def _call_llm_with_history(self, messages: List[Dict[str, str]], max_retries: int = 3, max_tokens: int = 8192) -> Dict[str, Any]:
        """
        Make actual API call to LLM with conversation history.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            max_retries: Maximum number of retry attempts (default: 3)
            max_tokens: Maximum tokens for response
            
        Returns:
            Dict with "response_text" and "usage" keys containing token usage and cost info
        """
        import logging
        import time
        import random as _rnd
        logger = logging.getLogger(__name__)
        
        # Route to Anthropic SDK if provider is anthropic
        if self.provider == "anthropic":
            return self._call_anthropic_with_history(messages, max_retries=max_retries, max_tokens=max_tokens)
        
        # Otherwise use OpenAI SDK (openai/xai/openrouter)
        try:
            from openai import OpenAI
            # Import exception classes for robust retry detection (best-effort)
            try:
                from openai import APITimeoutError, APIConnectionError, APIStatusError  # type: ignore
            except Exception:  # pragma: no cover - optional availability across versions
                APITimeoutError = APIConnectionError = APIStatusError = None  # type: ignore
            try:
                import httpx  # type: ignore
            except Exception:  # pragma: no cover
                httpx = None  # type: ignore
            try:
                import httpcore  # type: ignore
            except Exception:  # pragma: no cover
                httpcore = None  # type: ignore
        except ImportError:
            raise ImportError(
                "OpenAI package required for LLM agent. "
                "Install with: pip install openai"
            )
        
        # Create client with appropriate base URL and timeout
        http_client = None
        try:
            if httpx is not None:
                http_client = httpx.Client(
                    limits=httpx.Limits(max_keepalive_connections=20, max_connections=20)
                )
        except Exception:
            http_client = None
        
        if self.is_openrouter:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=45.0,
                max_retries=0,
                http_client=http_client
            )
        else:
            client = OpenAI(
                api_key=self.api_key,
                timeout=45.0,
                max_retries=0,
                http_client=http_client
            )
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    base_wait = 2 ** (attempt - 1)
                    wait_time = base_wait + _rnd.uniform(0.0, 0.75)
                    time.sleep(wait_time)
                
                # Prepare request parameters
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens
                }
                
                # Add reasoning configuration for OpenRouter if specified and not default
                if self.is_openrouter and (getattr(self, "reasoning", "default") != "default" or getattr(self, "enable_reasoning", False)):
                    # If reasoning is "none", explicitly disable reasoning
                    if self.reasoning.lower() == "none":
                        kwargs["extra_body"] = {
                            "reasoning": {
                                "enabled": False
                            }
                        }
                    else:
                        # Initialize extra_body if not exists
                        if "extra_body" not in kwargs:
                            kwargs["extra_body"] = {}
                        
                        reasoning_config = {"enabled": True}
                        # If reasoning is a numeric string, treat it as max_tokens
                        if self.reasoning.isdigit():
                            reasoning_config["max_tokens"] = int(self.reasoning)
                            # For Qwen models, OpenRouter may map max_tokens to thinking_budget
                            # According to OpenRouter docs, some Qwen models support thinking_budget
                            # We set it both in reasoning.max_tokens and top-level extra_body.thinking_budget
                            if "qwen" in self.model.lower():
                                kwargs["extra_body"]["thinking_budget"] = int(self.reasoning)
                        elif self.reasoning != "default":
                            # Otherwise treat as effort (OpenAI-style: "low", "medium", "high", etc.)
                            reasoning_config["effort"] = self.reasoning
                        
                        # Set reasoning config (standard OpenRouter format)
                        kwargs["extra_body"]["reasoning"] = reasoning_config
                        # Note: include_reasoning should be in extra_body, not as a top-level parameter
                        # Some models may also support include_reasoning in extra_body if needed
                        if "qwen" in self.model.lower():
                            kwargs["extra_body"]["include_reasoning"] = True
                
                response = client.chat.completions.create(**kwargs)
                
                # Validate response
                if response is None:
                    raise ValueError("LLM API returned None response")
                if not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("LLM API response has no choices")
                if len(response.choices) == 0:
                    raise ValueError("LLM API response has empty choices list")
                
                # Capture full API response dictionary
                try:
                    if hasattr(response, 'model_dump'):
                        # Pydantic v2
                        full_api_response = response.model_dump()
                    elif hasattr(response, 'dict'):
                        # Pydantic v1
                        full_api_response = response.dict()
                    else:
                        # Fallback: convert to dict manually
                        full_api_response = {
                            "id": getattr(response, 'id', None),
                            "object": getattr(response, 'object', None),
                            "created": getattr(response, 'created', None),
                            "model": getattr(response, 'model', None),
                            "choices": [
                                {
                                    "index": getattr(choice, 'index', None),
                                    "message": {
                                        "role": getattr(choice.message, 'role', None),
                                        "content": getattr(choice.message, 'content', None),
                                        "reasoning": getattr(choice.message, 'reasoning', None),
                                    },
                                    "finish_reason": getattr(choice, 'finish_reason', None),
                                }
                                for choice in response.choices
                            ],
                            "usage": {
                                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                                "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                                "total_tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                            }
                        }
                        # Add any additional fields that might exist
                        if hasattr(response, 'system_fingerprint'):
                            full_api_response["system_fingerprint"] = getattr(response, 'system_fingerprint', None)
                except Exception as e:
                    logger.warning(f"Failed to capture full API response: {e}")
                    full_api_response = {}
                
                # Extract usage information
                usage_info = {}
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    usage_info = {
                        "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                        "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                        "total_tokens": getattr(usage, 'total_tokens', 0) or 0
                    }
                    # Try to get cost if available (OpenRouter provides this)
                    if hasattr(usage, 'total_cost') or hasattr(usage, 'cost'):
                        usage_info["cost"] = getattr(usage, 'total_cost', None) or getattr(usage, 'cost', None)
                    elif hasattr(response, 'total_cost') or hasattr(response, 'cost'):
                        usage_info["cost"] = getattr(response, 'total_cost', None) or getattr(response, 'cost', None)
                
                # Get main content
                message = response.choices[0].message
                result = message.content.strip() if message.content else ""
                
                # Check finish_reason to detect truncation
                finish_reason = None
                if hasattr(response.choices[0], 'finish_reason'):
                    finish_reason = response.choices[0].finish_reason
                elif hasattr(response.choices[0], 'finishReason'):  # Alternative naming
                    finish_reason = response.choices[0].finishReason
                
                # For reasoning models (DeepSeek R1, o1, o3, etc.), check for reasoning content
                reasoning_text = None
                
                # Check for explicit reasoning field (DeepSeek R1, OpenAI o1/o3)
                if hasattr(message, 'reasoning') and message.reasoning:
                    reasoning_text = message.reasoning
                # Check for reasoning_details array (some models provide structured reasoning)
                elif hasattr(message, 'reasoning_details') and message.reasoning_details:
                    # Extract text from reasoning_details (for models like DeepSeek R1)
                    reasoning_parts = []
                    for detail in message.reasoning_details:
                        if hasattr(detail, 'text') and detail.text:
                            reasoning_parts.append(detail.text)
                        elif isinstance(detail, dict) and detail.get('text'):
                            reasoning_parts.append(detail['text'])
                    if reasoning_parts:
                        reasoning_text = '\n\n'.join(reasoning_parts)
                # Check for reasoning in response object (alternative location)
                elif hasattr(response.choices[0], 'reasoning') and response.choices[0].reasoning:
                    reasoning_text = response.choices[0].reasoning
                # Check if content itself contains reasoning tags (some models embed reasoning)
                elif result and any(tag in result.lower() for tag in ['<thinking>', '<reasoning>', '<think>']):
                    # Content already contains reasoning, return as-is with usage
                    return {"response_text": result, "usage": usage_info}
                
                # Combine reasoning and content if reasoning exists
                if reasoning_text:
                    if result:
                        full_response = f"<reasoning>\n{reasoning_text}\n</reasoning>\n\n{result}"
                    else:
                        # For models where content is empty but reasoning exists
                        # This can happen when:
                        # 1. Model hit max_tokens limit before outputting final answer (finish_reason == "length")
                        # 2. Model only output reasoning without final answer
                        # Try to extract answer from reasoning if possible
                        import re
                        q_pattern = re.compile(r'Q\d+(?:\.\d+)?\s*=\s*[^\n]+', re.IGNORECASE)
                        q_matches = q_pattern.findall(reasoning_text)
                        
                        if q_matches:
                            # Found Q format in reasoning, extract it
                            extracted = '\n'.join(q_matches)
                            logger.warning(f"Content empty but found Q format in reasoning. Extracted: {extracted[:100]}...")
                            full_response = f"<reasoning>\n{reasoning_text}\n</reasoning>\n\n{extracted}"
                        else:
                            # No Q format found, just return reasoning with warning
                            if finish_reason == "length":
                                logger.warning(f"Response truncated (finish_reason=length). Only reasoning available, no final answer. Consider increasing max_tokens.")
                            else:
                                logger.warning(f"Content empty but reasoning exists (finish_reason={finish_reason}). Only reasoning available, no final answer.")
                            full_response = f"<reasoning>\n{reasoning_text}\n</reasoning>"
                    return {"response_text": full_response, "usage": usage_info, "full_api_response": full_api_response}
                
                return {"response_text": result, "usage": usage_info, "full_api_response": full_api_response}
            
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                retryable_exc = False
                try:
                    if APITimeoutError is not None and isinstance(e, APITimeoutError):
                        retryable_exc = True
                except Exception:
                    pass
                
                if not retryable_exc:
                    retryable_exc = any(indicator in error_msg for indicator in [
                        "timeout", "connection", "rate limit", "429", "503", "502", "500"
                    ])
                
                if retryable_exc and attempt < max_retries - 1:
                    logger.warning(f"Retryable error on attempt {attempt + 1}/{max_retries}: {e}")
                    continue
                else:
                    logger.error(f"Failed to call LLM after {attempt + 1} attempts: {e}")
                    raise
        
        if last_exception:
            raise last_exception
        raise RuntimeError("Failed to call LLM: unknown error")
    
    def _call_anthropic_with_history(self, messages: List[Dict[str, str]], max_retries: int = 3, max_tokens: int = 8192) -> Dict[str, Any]:
        """
        Call Anthropic Claude API with conversation history.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            max_retries: Maximum retry attempts
            max_tokens: Maximum tokens for response
            
        Returns:
            Dict with "response_text" and "usage" keys
        """
        import logging
        import time
        import random as _rnd
        logger = logging.getLogger(__name__)
        
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        client = Anthropic(api_key=self.api_key)
        
        # Convert messages to Anthropic format (extract system, ensure user/assistant alternation)
        system_msg = None
        claude_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_msg = content
            elif role in ("user", "assistant"):
                claude_messages.append({"role": role, "content": content})
        
        if not claude_messages or claude_messages[0].get("role") != "user":
            # Anthropic requires first message to be user
            claude_messages.insert(0, {"role": "user", "content": "Hello"})
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** (attempt - 1) + _rnd.uniform(0, 0.75)
                    time.sleep(wait_time)
                
                kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": claude_messages,
                }
                if system_msg:
                    kwargs["system"] = system_msg
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                
                response = client.messages.create(**kwargs)
                result = response.content[0].text if response.content else ""
                result = self._clean_llm_response_text(result)
                
                # Extract usage
                usage_info = {}
                if hasattr(response, "usage") and response.usage:
                    usage_info = {
                        "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                        "completion_tokens": getattr(response.usage, "output_tokens", 0),
                        "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
                    }
                
                full_api_response = {}
                try:
                    if hasattr(response, 'model_dump'):
                        full_api_response = response.model_dump()
                    elif hasattr(response, 'dict'):
                        full_api_response = response.dict()
                except:
                    pass
                
                return {"response_text": result, "usage": usage_info, "full_api_response": full_api_response}
            
            except Exception as e:
                last_exception = e
                logger.warning(f"Anthropic API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    break
        
        error_msg = f"Failed to get response from Anthropic after {max_retries} attempts: {last_exception}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _call_llm(self, system_prompt: str, user_message: str, max_retries: int = 3, max_tokens: int = 8192) -> Dict[str, Any]:
        """
        Make actual API call to LLM with automatic retry on failure (stateless mode).
        
        Supports both OpenRouter (for models like mistralai/mistral-nemo) and OpenAI API.
        This is a convenience wrapper that creates a simple message list and calls _call_llm_with_history.
        
        Args:
            system_prompt: System prompt for the LLM
            user_message: User message/prompt
            max_retries: Maximum number of retry attempts (default: 3)
            max_tokens: Maximum tokens for response
            
        Returns:
            Dict with "response_text" and "usage" keys containing token usage and cost info
        """
        # Stateless mode: create simple message list and delegate to history-based method
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self._call_llm_with_history(messages, max_retries=max_retries, max_tokens=max_tokens)
    
    def _call_llm_old(self, system_prompt: str, user_message: str, max_retries: int = 3, max_tokens: int = 8192) -> str:
        """
        OLD IMPLEMENTATION - kept for reference, not used.
        """
        import logging
        import time
        import random as _rnd
        logger = logging.getLogger(__name__)
        
        try:
            from openai import OpenAI
            # Import exception classes for robust retry detection (best-effort)
            try:
                from openai import APITimeoutError, APIConnectionError, APIStatusError  # type: ignore
            except Exception:  # pragma: no cover - optional availability across versions
                APITimeoutError = APIConnectionError = APIStatusError = None  # type: ignore
            try:
                import httpx  # type: ignore
            except Exception:  # pragma: no cover
                httpx = None  # type: ignore
            try:
                import httpcore  # type: ignore
            except Exception:  # pragma: no cover
                httpcore = None  # type: ignore
        except ImportError:
            raise ImportError(
                "OpenAI package required for LLM agent. "
                "Install with: pip install openai"
            )
        
        # Create client with appropriate base URL and timeout
        # Optional HTTP client with connection pooling
        http_client = None
        try:
            # Use modest pool limits to improve reuse and avoid connection churn
            if httpx is not None:
                http_client = httpx.Client(
                    limits=httpx.Limits(max_keepalive_connections=20, max_connections=20)
                )
        except Exception:
            http_client = None

        if self.is_openrouter:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=45.0,  # Bump timeout to reduce spurious request timeouts
                max_retries=0,  # We handle retries manually for better control
                http_client=http_client
            )
        else:
            client = OpenAI(
                api_key=self.api_key,
                timeout=45.0,
                max_retries=0,
                http_client=http_client
            )
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff + jitter to avoid synchronized retries
                    base_wait = 2 ** (attempt - 1)
                    wait_time = base_wait + _rnd.uniform(0.0, 0.75)
                    time.sleep(wait_time)
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=1.0,
                    max_tokens=max_tokens
                )
                
                # Validate response
                if response is None:
                    raise ValueError("LLM API returned None response")
                if not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("LLM API response has no choices")
                if len(response.choices) == 0:
                    raise ValueError("LLM API response has empty choices list")
                
                result = response.choices[0].message.content.strip()
                
                return result
            
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Determine if we should retry (types + common transient indicators)
                retryable_exc = False
                # By exception type
                try:
                    if APITimeoutError is not None and isinstance(e, APITimeoutError):
                        retryable_exc = True
                except Exception:
                    pass
                try:
                    if httpx is not None and isinstance(e, getattr(httpx, 'ReadTimeout', tuple())):
                        retryable_exc = True
                except Exception:
                    pass
                try:
                    if httpcore is not None and isinstance(e, getattr(httpcore, 'ReadTimeout', tuple())):
                        retryable_exc = True
                except Exception:
                    pass
                # HTTP status based
                status_code = getattr(e, 'status_code', None) or getattr(e, 'status', None)
                if isinstance(status_code, int) and status_code in (429, 500, 502, 503):
                    retryable_exc = True

                # Message patterns
                keywords = ['connection', 'timeout', 'timed out', 'rate limit', '429', '503', '502', '500', 'deadline exceeded']
                message_hint = any(k in error_msg for k in keywords)

                should_retry = retryable_exc or message_hint

                if should_retry and attempt < max_retries - 1:
                    logger.warning(f"[P{self.participant_id}] API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    # Last attempt or non-retryable error
                    provider = "OpenRouter" if self.is_openrouter else "OpenAI"
                    logger.error(f"[P{self.participant_id}] API call failed after {attempt + 1} attempts: {e}")
                    raise RuntimeError(f"{provider} API call failed: {e}")
        
        # Should not reach here, but just in case
        provider = "OpenRouter" if self.is_openrouter else "OpenAI"
        raise RuntimeError(f"{provider} API call failed after {max_retries} attempts: {last_exception}")
    
    def _simulate_response(
        self,
        trial_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """
        Simulate participant response without LLM (for testing).
        
        Handles different study types:
        - Conformity studies (Asch): ~37% conformity on critical trials
        - Obedience studies (Milgram): Decreasing compliance with shock level
        - Framing studies (Tversky & Kahneman): Frame-dependent risk preferences
        - Representativeness studies (Kahneman & Tversky 1972): Representativeness bias
        - False Consensus Effect (Ross et al. 1977): Estimates bias based on own choice
        """
        import random
        
        study_type = trial_info.get("study_type", "")
        sub_study_id = trial_info.get("sub_study_id", "")
        items = trial_info.get("items", [])
        
        # Detect study type from sub_study_id if study_type not set
        if not study_type:
            if sub_study_id.startswith("study_1_") or sub_study_id.startswith("study_2_") or sub_study_id == "study_3_sandwich_board":
                study_type = "false_consensus_effect"
        
        # Handle representativeness heuristic studies
        if study_type == "representativeness_heuristic":
            # Get assigned problem from participant profile
            assigned_problem = self.profile.get("assigned_problem", "birth_sequence")
            
            if assigned_problem == "birth_sequence":
                # Birth sequence problem: 81.5% show representativeness bias
                # (judge BGBBBB as less likely than GBGBBG, though they're equally likely)
                if random.random() < 0.815:
                    # Show bias: think BGBBBB is less likely
                    choice = "less_likely"
                    # Simulate various ways of expressing this
                    responses = [
                        "BGBBBB seems less likely",
                        "I think GBGBBG is more representative and thus more likely",
                        "BGBBBB appears less probable",
                        "The first sequence (GBGBBG) seems more likely"
                    ]
                    response_text = random.choice(responses)
                else:
                    # Correct answer: they're equally likely
                    choice = "equal"
                    response_text = "They are equally likely"
            
            else:  # program_choice
                # Program choice problem: 75.3% choose representative answer (Program A)
                # Correct answer is Program B (higher variance at p=0.45)
                if random.random() < 0.753:
                    # Show bias: choose based on representativeness (55% closer to 65%)
                    choice = "A"
                    response_text = "Program A"
                else:
                    # Correct answer: Program B (considers variance)
                    choice = "B"
                    response_text = "Program B"
            
            return choice, response_text
        
        # Handle False Consensus Effect (Study 001)
        if study_type == "false_consensus_effect":
            # For bundled trials, generate responses for all items
            if items:
                # First pass: identify and make the personal choice
                my_choice = None
                choice_item = None
                for item in items:
                    if item.get("type") == "multiple_choice" and "personal_choice" in item.get("id", ""):
                        options = item.get("options", [])
                        if options:
                            my_choice = random.choice(options)
                            choice_item = item
                        break
                
                # If no choice made, pick randomly between option 1 and option 2
                if not my_choice:
                    my_choice = "Option 1" if random.random() < 0.5 else "Option 2"
                
                # Determine if this is an "agree" choice or "refuse" choice for FCE bias
                is_agree_choice = not any(word in my_choice.lower() for word in ["refuse", "contest", "oppose", "against"])
                
                # Second pass: generate all responses
                response_lines = []
                for item in items:
                    item_id = item.get("id", "")
                    item_type = item.get("type", "")
                    
                    if item_type == "multiple_choice":
                        response_lines.append(f"[{item_id}]: {my_choice}")
                    
                    elif item_type == "estimation":
                        # Generate estimate based on False Consensus Effect
                        # "estimation_1" typically asks about % who would choose option 1
                        if "estimation_1" in item_id or "1" in item_id:
                            if is_agree_choice:
                                estimate = int(random.gauss(65, 10))  # I agree, so I think others will too
                            else:
                                estimate = int(random.gauss(35, 10))  # I refuse, so I think others will too
                        else:  # estimation_2 (usually the complement)
                            if is_agree_choice:
                                estimate = int(random.gauss(35, 10))
                            else:
                                estimate = int(random.gauss(65, 10))
                        
                        estimate = max(0, min(100, estimate))
                        response_lines.append(f"[{item_id}]: {estimate}")
                    
                    elif item_type == "likert":
                        # Generate random Likert response (assuming -3 to +3 scale)
                        likert_val = random.randint(-3, 3)
                        response_lines.append(f"[{item_id}]: {likert_val}")
                
                response_text = "\n".join(response_lines)
                return my_choice, response_text
            
            # Fallback for old format (legacy)
            profile = self.profile
            scenario = profile.get("assigned_scenario", "unknown")
            
            # Handle Study 2 Full Questionnaire
            if scenario == "study_2_questionnaire_full":
                # Generate JSON list for 34 items
                items_list = []
                # Simulate 34 items
                for i in range(1, 35):
                    # Random choice A or B (assume 50/50 split for simulation)
                    my_choice = "Option A" if random.random() < 0.5 else "Option B"
                    
                    # FCE Simulation:
                    # If I choose A, I estimate A higher (e.g., 60-80%)
                    # If I choose B, I estimate A lower (e.g., 20-40%)
                    if my_choice == "Option A":
                        est_a = int(random.gauss(65, 10))
                    else:
                        est_a = int(random.gauss(35, 10))
                    
                    est_a = max(0, min(100, est_a))
                    est_b = 100 - est_a
                    
                    items_list.append({
                        "id": f"item_{i}",
                        "my_choice": my_choice,
                        "estimate_a": est_a,
                        "estimate_b": est_b
                    })
                
                response_text = json.dumps(items_list)
                choice = "JSON_RESPONSE" # Special marker
                
                return choice, response_text
                
            # Handle Study 1 & 3 (Single Scenarios) - legacy
            else:
                # Random choice
                choice_idx = 0 if random.random() < 0.5 else 1
                choices = ["Option A", "Option B"]
                choice = choices[choice_idx]
                
                # FCE Simulation
                if choice == "Option A":
                    est_a = int(random.gauss(70, 10))
                else:
                    est_a = int(random.gauss(40, 10))
                
                est_a = max(0, min(100, est_a))
                
                response_text = f"I would choose {choice}. I estimate {est_a}% of students would choose Option A."
                
                # Extract just A or B for the 'response' field
                choice_letter = "A" if choice == "Option A" else "B"
                
                return choice_letter, response_text

        
        # Handle framing effect studies
        if study_type == "framing_effect":
            # Get framing condition from participant profile
            frame = self.profile.get("framing_condition", "positive_frame")
            
            # Simulate framing effect based on original Tversky & Kahneman findings:
            # Positive frame: 72% choose certain option (Program A)
            # Negative frame: 22% choose certain option (Program A), 78% choose risky (Program B)
            if frame == "positive_frame":
                # Positive frame: risk-averse (favor certain option)
                if random.random() < 0.72:
                    choice = "A"
                    response_text = "Program A"
                else:
                    choice = "B"
                    response_text = "Program B"
            else:  # negative_frame
                # Negative frame: risk-seeking (favor risky option)
                if random.random() < 0.78:
                    choice = "B"
                    response_text = "Program B"
                else:
                    choice = "A"
                    response_text = "Program A"
            
            return choice, response_text
        
        # Handle obedience studies (Milgram)
        shock_level = trial_info.get("shock_level", trial_info.get("voltage", 0))
        if shock_level > 0:
            # Fixed obedience probability (no longer using personality_traits)
            # Base: 65% go to max, declining probability
            base_prob = 0.65
            shock_factor = shock_level / 30.0  # Normalize to 0-1
            continue_prob = base_prob * (1.0 - 0.3 * shock_factor)  # Decreases with shock level
            
            if random.random() < continue_prob:
                choice = "continue"
                response_text = "continue"
            else:
                choice = "stop"
                response_text = "stop"
            
            return choice, response_text
        
        correct = trial_info.get("correct_answer")
        confederates = trial_info.get("confederate_responses", [])
        
        # Fixed conformity tendency (no longer using personality_traits)
        conformity_tendency = 0.37
        
        if confederates:
            # Critical trial - group gives wrong answer
            confederate_answer = confederates[0]  # They're all the same
            
            # Decide whether to conform
            if random.random() < conformity_tendency:
                # Conform to group
                choice = confederate_answer
                response_text = f"{confederate_answer}"
            else:
                # Give correct answer (resist pressure)
                choice = correct
                response_text = f"{correct}"
        else:
            # Neutral trial - almost always correct
            if random.random() < 0.01:  # 1% error rate
                # Rare error
                options = trial_info.get("comparison_lines", {})
                if options:
                    wrong_options = [o for o in options.keys() if o != correct]
                    if wrong_options:
                        choice = random.choice(wrong_options)
                    else:
                        choice = correct
                else:
                    choice = correct
                response_text = f"{choice}"
            else:
                choice = correct
                response_text = f"{correct}"
        
        return choice, response_text
    
    def _parse_response(self, response_text: str, trial_info: Dict[str, Any]) -> str:
        """
        Parse LLM response to extract the actual choice or numerical answer.
        
        Tries multiple extraction strategies in order of preference:
        1. For numeric responses: extract the first number
        2. Look for "Program X" or "Option X" patterns (for framing studies)
        3. Look for quoted responses like '"A"' or "'B'"
        4. Look for single letter at start of line or after colon
        5. Fall back to first occurrence of A/B/C
        """
        response_upper = response_text.upper()
        response_stripped = response_text.strip()
        
        import re
        
        # Strategy 0: Check if this is a pure numeric response (e.g., "36", "72")
        # This is common for estimation tasks like Study 004 birth sequence
        if response_stripped.isdigit():
            return response_stripped
        
        # Strategy 0.5: Extract first number from response
        # Useful when response is like "I estimate 36 families" or "3.5 million"
        # Support integers, decimals, and comma-separated numbers (e.g., "29,032", "1,000")
        number_match = re.search(r'\b(\d+(?:,\d+)*(?:\.\d+)?)\b', response_text)
        if number_match:
            # Only use numeric parsing if response doesn't contain A/B/C choice indicators
            if not re.search(r'\b(PROGRAM|OPTION)\s+[ABC]\b', response_upper):
                # Return the number (we keep commas to preserve the original format, 
                # or we could strip them. Study configs usually handle parsing.)
                # But to be safe for simple int() conversions, maybe we should strip?
                # The previous behavior returned the substring. Let's return the substring 
                # but ensure it captures the full number.
                return number_match.group(1).replace(",", "")
        
        # Strategy 1: Look for "PROGRAM A/B" or "OPTION A/B/C" patterns
        program_match = re.search(r'\b(PROGRAM|OPTION)\s+([A-F])\b', response_upper)
        if program_match:
            return program_match.group(2)
        
        # Strategy 2: Look for quoted single letters (e.g., "A", 'B', or "Program A")
        quoted_match = re.search(r'["\'](?:PROGRAM\s+)?([A-F])["\']', response_upper)
        if quoted_match:
            return quoted_match.group(1)
        
        # Strategy 2.5: Look for "Yes" or "No"
        # Prioritize explicit Yes/No answers
        yes_match = re.search(r'\bYES\b', response_upper)
        no_match = re.search(r'\bNO\b', response_upper)
        if yes_match and not no_match:
            return "Yes"
        if no_match and not yes_match:
            return "No"
        # If both appear, fall through to other strategies or heuristics
        
        # Strategy 3: Look for A/B/C at start of response or after newline/colon
        clean_start = response_upper.strip()
        if clean_start and clean_start[0] in ['A', 'B', 'C', 'D', 'E', 'F']:
            return clean_start[0]
        
        line_start_match = re.search(r'(?:^|\n|:\s*)([A-F])\b', response_upper)
        if line_start_match:
            return line_start_match.group(1)
        
        # Strategy 4: Last resort - find first A-F in text
        # Note: checking specific letters to avoid false positives
        for letter in ['A', 'B', 'C', 'D', 'E', 'F']:
            if letter in response_upper:
                # Simple presence check might be too aggressive for 'A' (e.g. in "A cat")
                # Require word boundary for single letter
                if re.search(r'\b' + letter + r'\b', response_upper):
                    return letter
        
        # Default to ? if completely unparseable
        return "?"
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of participant's performance.
        """
        if not self.trial_responses:
            return {
                "participant_id": self.participant_id,
                "total_trials": 0,
                "profile": self.profile
            }
        
        total = len(self.trial_responses)
        correct = sum(1 for r in self.trial_responses if r["is_correct"])
        
        # Calculate conformity rate (only for critical trials)
        critical_trials = [
            r for r in self.trial_responses 
            if r.get("trial_info", {}).get("confederate_responses")
        ]
        
        if critical_trials:
            conformed = sum(
                1 for r in critical_trials 
                if not r["is_correct"]  # Wrong answer = conformed
            )
            conformity_rate = conformed / len(critical_trials)
        else:
            conformity_rate = None
        
        return {
            "participant_id": self.participant_id,
            "profile": self.profile,
            "total_trials": total,
            "correct_responses": correct,
            "accuracy": correct / total if total > 0 else 0,
            "critical_trials": len(critical_trials),
            "conformity_rate": conformity_rate,
            "responses": self.trial_responses
        }


class ParticipantPool:
    """
    Manages a pool of LLM participant agents for an experiment.
    
    This is what users interact with when running the benchmark.
    """
    
    def __init__(
        self,
        study_specification: Dict[str, Any],
        n_participants: Optional[int] = None,
        use_real_llm: bool = False,
        model: str = "mistralai/mistral-nemo",
        api_key: Optional[str] = None,
        random_seed: Optional[int] = None,
        api_base: Optional[str] = None,
        num_workers: Optional[int] = None,
        profiles: Optional[List[Dict[str, Any]]] = None,
        prompt_builder: Optional[Any] = None,
        system_prompt_override: Optional[str] = None,
        system_prompt_preset: str = "v3_human_plus_demo",
        reasoning: str = "default",
        enable_reasoning: bool = False,
        study_id: Optional[str] = None,
        existing_responses: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0
    ):
        """
        Initialize participant pool based on study specification.
        
        Args:
            study_specification: Study specification with participant requirements
            n_participants: Number of participants (default: use study's n)
            use_real_llm: Whether to use real LLM API calls
            model: LLM model to use (default: "mistralai/mistral-nemo" via OpenRouter)
            api_key: API key for LLM service (OPENROUTER_API_KEY or OPENAI_API_KEY)
            random_seed: Random seed for reproducible profile generation
            api_base: Optional API base URL for OpenRouter or custom endpoints
            num_workers: Number of parallel workers for participant execution
            profiles: Optional pre-generated participant profiles (if None, will auto-generate)
            prompt_builder: Optional PromptBuilder for custom system prompt generation
            system_prompt_override: Optional custom system prompt
            system_prompt_preset: One of "empty", "v2_human", "v3_human_plus_demo", etc.
            reasoning: Reasoning effort for supported models ("default", "none", "low", "medium", "high", "xhigh", "minimal")
            enable_reasoning: Force enable reasoning for OpenRouter models
            study_id: Optional study identifier (for presets that need it)
            existing_responses: Optional list of already collected responses to resume from
            temperature: Sampling temperature for the LLM (default: 1.0)
        """
        self.specification = study_specification
        self.n_participants = n_participants or study_specification["participants"]["n"]
        self.use_real_llm = use_real_llm
        self.model = model
        self.api_key = api_key
        self.random_seed = random_seed
        self.api_base = api_base
        self.prompt_builder = prompt_builder
        self.system_prompt_preset = system_prompt_preset
        self.reasoning = reasoning
        self.enable_reasoning = enable_reasoning
        self.temperature = temperature
        self.study_id = study_id
        # Number of worker threads to use for parallel participant execution
        # If None, and using real LLMs, default to min(8, n_participants)
        self.num_workers = num_workers if num_workers is not None else (
            min(8, self.n_participants) if self.use_real_llm else 1
        )
        
        # Create participant profiles from specification or use provided ones
        if profiles is not None:
            # Ensure study_id is added to provided profiles if not present
            if self.study_id:
                for profile in profiles:
                    if "study_id" not in profile:
                        profile["study_id"] = self.study_id
            self.profiles = profiles
        else:
            self.profiles = self._generate_profiles()
        
        # Store prompt_builder for later use in system prompt construction
        self.prompt_builder = prompt_builder
        self.system_prompt_override = system_prompt_override
        
        # Create participant agents
        self.participants: List[LLMParticipantAgent] = []
        for i, profile in enumerate(self.profiles):
            agent = LLMParticipantAgent(
                participant_id=i,
                profile=profile,
                model=model,
                api_key=api_key,
                use_real_llm=use_real_llm,
                system_prompt_override=system_prompt_override,
                api_base=api_base,
                prompt_builder=prompt_builder,  # Pass prompt_builder for custom system prompt support
                system_prompt_preset=system_prompt_preset,
                reasoning=reasoning,
                enable_reasoning=enable_reasoning,
                temperature=temperature
            )
            
            # Load existing responses for this participant if provided
            if existing_responses:
                agent.trial_responses = [r for r in existing_responses if r.get('participant_id') == i]
                
            self.participants.append(agent)
    
    def _generate_profiles(self) -> List[Dict[str, Any]]:
        """
        Generate participant profiles based on study specification.
        
        Uses the recruitment criteria from the literature to create
        realistic participant profiles.
        """
        import numpy as np
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        spec = self.specification["participants"]
        profiles = []
        
        # Extract profile requirements
        # If age_range is provided, default to uniform sampling over the range.
        # If age_mean/age_sd are explicitly provided, use clipped Normal sampling.
        #
        # If neither is provided, infer a reasonable default from population text
        # (e.g., "Undergraduates" -> typical college age) to avoid unrealistic ages.
        age_range = spec.get("age_range")
        if age_range is None:
            population = (spec.get("population") or "").lower()
            if "undergraduate" in population or "college" in population or "university" in population:
                age_range = [18, 25]
            else:
                age_range = [18, 65]
        age_mean = spec.get("age_mean")
        age_sd = spec.get("age_sd")
        
        gender_dist = spec.get("gender_distribution", {"male": 50, "female": 50})
        total_gender = sum(gender_dist.values())
        
        for i in range(self.n_participants):
            # Sample age
            if age_mean is not None and age_sd is not None:
                age = np.random.normal(age_mean, age_sd)
                age = int(np.clip(age, age_range[0], age_range[1]))
            else:
                # Uniform integer age within [min,max]
                lo, hi = int(age_range[0]), int(age_range[1])
                if hi < lo:
                    lo, hi = hi, lo
                age = int(np.random.randint(lo, hi + 1))
            
            # Sample gender based on distribution
            rand = np.random.random() * total_gender
            cumsum = 0
            gender = "male"
            for g, count in gender_dist.items():
                cumsum += count
                if rand < cumsum:
                    gender = g
                    break
            
            # Note: personality_traits removed - no longer used for behavior modeling
            profile = {
                "participant_id": i,
                "age": age,
                "gender": gender,
                # Keep a simple text field that can be used in the system prompt
                # (Historically called "education" in this codebase).
                "education": spec.get("recruitment_source", "college student"),
                "population": spec.get("population"),
                "recruitment_source": spec.get("recruitment_source"),
            }
            
            # Add study_id if available (for presets that need it)
            if self.study_id:
                profile["study_id"] = self.study_id
            
            profiles.append(profile)
        
        return profiles
    
    def run_experiment(
        self,
        trials: List[Dict[str, Any]],
        instructions: str,
        prompt_builder: Optional[Any] = None,
        one_to_one: bool = False,
        save_callback: Optional[Callable[[], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the experiment with all participants.
        
        Args:
            trials: List of trial specifications
            instructions: Experimental instructions
            prompt_builder: Optional PromptBuilder for generating trial prompts.
                           If None, trials must contain pre-built prompts or use legacy format.
            one_to_one: If True, each participant runs exactly one trial (trials[i] for participant[i]).
                       Requires len(trials) == len(self.participants).
            
        Returns:
            Aggregated results from all participants
        """
        if one_to_one and len(trials) != len(self.participants):
            raise ValueError(f"one_to_one=True requires len(trials) ({len(trials)}) == len(participants) ({len(self.participants)})")

        print(f"\n{'='*70}")
        print(f"Running experiment with {len(self.participants)} participants")
        print(f"Model: {self.model} (Real LLM: {self.use_real_llm})")
        print(f"{'='*70}\n")
        
        # Prepare progress bar and parallel execution across participants
        if one_to_one:
            total_api_calls = len(trials)
            print(f"Running 1-to-1 experiment: each participant runs 1 trial. (total API calls: {total_api_calls})")
        else:
            total_api_calls = len(self.participants) * len(trials)
            print(f"Running {len(trials)} trials per participant... (total API calls: {total_api_calls})")

        # If only one worker or not using real LLMs, run sequentially but keep progress prints
        if self.num_workers <= 1 or not self.use_real_llm:
            from tqdm import tqdm
            pbar = tqdm(total=total_api_calls, desc="API calls", unit="call")
            
            if one_to_one:
                for i, (participant, trial) in enumerate(zip(self.participants, trials)):
                    trial_with_profile = {**trial, "participant_profile": participant.profile}
                    if prompt_builder:
                        trial_prompt = prompt_builder.build_trial_prompt(trial_with_profile)
                    else:
                        trial_prompt = trial.get("prompt", f"Trial {trial.get('trial_number', '?')}: Please respond.")
                    resp_data = participant.complete_trial(trial_prompt, trial_with_profile)
                    pbar.update(1)
                    # Call save callback after each API call completes
                    if save_callback:
                        try:
                            save_callback(resp_data)
                        except Exception as e:
                            import logging
                            logging.getLogger(__name__).warning(f"Save callback failed: {e}")
            else:
                for trial_idx, trial in enumerate(trials):
                    for participant in self.participants:
                        trial_with_profile = {**trial, "participant_profile": participant.profile}
                        if prompt_builder:
                            trial_prompt = prompt_builder.build_trial_prompt(trial_with_profile)
                        else:
                            trial_prompt = trial.get("prompt", f"Trial {trial.get('trial_number', '?')}: Please respond.")
                        participant.complete_trial(trial_prompt, trial_with_profile)
                        pbar.update(1)
                        # Call save callback after each API call completes
                        if save_callback:
                            try:
                                save_callback()
                            except Exception as e:
                                import logging
                                logging.getLogger(__name__).warning(f"Save callback failed: {e}")

            pbar.close()
            print("Experiment complete!\n")
        else:
            # Parallelize
            from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
            from tqdm import tqdm
            import logging
            
            logging.basicConfig(level=logging.WARNING, format='%(message)s')
            logger = logging.getLogger(__name__)

            if one_to_one:
                def run_single_trial(idx):
                    participant = self.participants[idx]
                    trial = trials[idx]
                    
                    # RESUME: Skip if already has response
                    if len(participant.trial_responses) > 0:
                        return idx
                        
                    try:
                        import time as _t
                        import random as _r
                        _t.sleep(_r.uniform(0.0, 0.5))
                        trial_with_profile = {**trial, "participant_profile": participant.profile}
                        if prompt_builder:
                            trial_prompt = prompt_builder.build_trial_prompt(trial_with_profile)
                        else:
                            trial_prompt = trial.get("prompt", f"Trial {trial.get('trial_number', '?')}: Please respond.")
                        resp_data = participant.complete_trial(trial_prompt, trial_with_profile)
                        
                        # Call callback immediately if in same thread or handle via main loop
                        if save_callback:
                            try:
                                save_callback(resp_data)
                            except:
                                pass
                                
                        return idx
                    except Exception as e:
                        logger.error(f"[P{participant.participant_id}] Trial failed: {e}")
                        raise

                pbar = tqdm(total=total_api_calls, desc="Progress", unit="call", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
                
                # Pre-update progress bar for existing responses
                initial_count = sum(len(p.trial_responses) for p in self.participants)
                if initial_count > 0:
                    pbar.update(initial_count)
                    
                try:
                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        futures = {executor.submit(run_single_trial, i): self.participants[i] for i in range(len(self.participants))}
                        
                        import time
                        last_count = initial_count
                        timeout_counter = 0
                        max_stall_time = 60
                        
                        while any(not f.done() for f in futures):
                            current_count = sum(len(p.trial_responses) for p in self.participants)
                            delta = current_count - last_count
                            if delta > 0:
                                pbar.update(delta)
                                last_count = current_count
                                timeout_counter = 0
                                # Call save callback after each batch of API calls complete
                                if save_callback:
                                    try:
                                        save_callback()
                                    except Exception as e:
                                        logger.warning(f"Save callback failed: {e}")
                            else:
                                timeout_counter += 1
                                if timeout_counter > max_stall_time * 20:
                                    raise TimeoutError(f"Stalled at {current_count}/{total_api_calls} calls")
                            time.sleep(0.05)
                        
                        for future, participant in futures.items():
                            future.result()
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    raise
                finally:
                    pbar.close()
            else:
                def run_for_participant(participant: LLMParticipantAgent):
                    """Run all trials for a single participant."""
                    try:
                        import time as _t
                        import random as _r
                        _t.sleep(_r.uniform(0.0, 0.5))
                        for trial_idx, trial in enumerate(trials):
                            # RESUME: Skip if already has response for this trial index
                            if len(participant.trial_responses) > trial_idx:
                                continue
                                
                            try:
                                trial_with_profile = {**trial, "participant_profile": participant.profile}
                                if prompt_builder:
                                    trial_prompt = prompt_builder.build_trial_prompt(trial_with_profile)
                                else:
                                    trial_prompt = trial.get("prompt", f"Trial {trial.get('trial_number', '?')}: Please respond.")
                                _t.sleep(_r.uniform(0.0, 0.2))
                                resp_data = participant.complete_trial(trial_prompt, trial_with_profile)
                                
                                # Call callback immediately
                                if save_callback:
                                    try:
                                        save_callback(resp_data)
                                    except:
                                        pass
                            except Exception as e:
                                logger.error(f"[P{participant.participant_id}] Trial {trial_idx + 1} failed: {e}")
                                raise
                        return participant.participant_id
                    except Exception as e:
                        logger.error(f"[P{participant.participant_id}] Failed: {e}")
                        raise

                pbar = tqdm(total=total_api_calls, desc="Progress", unit="call", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
                
                # Pre-update progress bar for existing responses
                initial_count = sum(len(p.trial_responses) for p in self.participants)
                if initial_count > 0:
                    pbar.update(initial_count)
                    
                try:
                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        futures = {executor.submit(run_for_participant, participant): participant for participant in self.participants}
                        import time
                        last_count = initial_count
                        timeout_counter = 0
                        max_stall_time = 60
                        while any(not f.done() for f in futures):
                            current_count = sum(len(p.trial_responses) for p in self.participants)
                            delta = current_count - last_count
                            if delta > 0:
                                pbar.update(delta)
                                last_count = current_count
                                timeout_counter = 0
                                # Call save callback after each batch of API calls complete
                                if save_callback:
                                    try:
                                        save_callback()
                                    except Exception as e:
                                        logger.warning(f"Save callback failed: {e}")
                            else:
                                timeout_counter += 1
                                if timeout_counter > max_stall_time * 20:
                                    raise TimeoutError(f"Stalled at {current_count}/{total_api_calls} calls")
                            time.sleep(0.05)
                        for future, participant in futures.items():
                            future.result()
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    raise
                finally:
                    pbar.close()

            print("Experiment complete!\n")
        
        return self.aggregate_results()
        
        # Collect all results
        return self.aggregate_results()
    
    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results from all participants for analysis.
        """
        import numpy as np
        
        # Get individual summaries
        summaries = [p.get_summary() for p in self.participants]
        
        # Create a flat list of all individual trial responses across all participants
        # This is expected by many StudyConfig.aggregate_results implementations
        all_individual_data = []
        for p in self.participants:
            all_individual_data.extend(p.trial_responses)
        
        # Calculate group statistics
        conformity_rates = [
            s["conformity_rate"] for s in summaries 
            if s["conformity_rate"] is not None
        ]
        
        if conformity_rates:
            results = {
                "descriptive_statistics": {
                    "conformity_rate": {
                        "experimental": {
                            "n": len(conformity_rates),
                            "mean": float(np.mean(conformity_rates)),
                            "sd": float(np.std(conformity_rates, ddof=1)),
                            "median": float(np.median(conformity_rates)),
                            "min": float(np.min(conformity_rates)),
                            "max": float(np.max(conformity_rates)),
                            "never_conformed": sum(1 for r in conformity_rates if r == 0.0),
                            "always_conformed": sum(1 for r in conformity_rates if r == 1.0)
                        }
                    }
                },
                "inferential_statistics": {},
                "individual_data": all_individual_data,
                "participant_summaries": summaries,
                "raw_responses": [p.trial_responses for p in self.participants]
            }
        else:
            results = {
                "descriptive_statistics": {},
                "inferential_statistics": {},
                "individual_data": all_individual_data,
                "participant_summaries": summaries
            }
        
        return results
