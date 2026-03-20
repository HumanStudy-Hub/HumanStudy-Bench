from typing import Dict, Any, Callable, Union, Optional
import os
import importlib.util
from pathlib import Path

class SystemPromptRegistry:
    """
    Unified registry for LLM participant system prompts.
    Provides a "port" for adding new prompt versions or custom prompt generators.
    """
    _presets: Dict[str, Union[str, Callable[[Dict[str, Any]], str]]] = {}

    @classmethod
    def register(cls, name: str, prompt_handler: Union[str, Callable[[Dict[str, Any]], str]]):
        """
        Register a new system prompt preset.
        
        Args:
            name: Unique name for the preset (e.g., 'v1', 'v2_human')
            prompt_handler: Either a format string or a function that takes a profile dict
                           and returns a prompt string.
        """
        cls._presets[name] = prompt_handler

    @classmethod
    def get_prompt(cls, name: str, profile: Dict[str, Any]) -> str:
        """
        Get the system prompt for a given preset and participant profile.
        """
        # Normalize preset name: convert hyphens to underscores for lookup
        # e.g. "v2-human" matches "v2_human"
        normalized_name = name.replace("-", "_")
        
        # Try normalized name first, then original name
        lookup_name = normalized_name if normalized_name in cls._presets else name
        
        if lookup_name not in cls._presets:
            raise ValueError(f"Unknown system prompt preset: {name}. Available: {list(cls._presets.keys())}")
        
        handler = cls._presets[lookup_name]
        
        if callable(handler):
            return handler(profile)
        
        # Handle template string
        try:
            return handler.format(**profile)
        except KeyError as e:
            # Fallback if a key is missing in profile
            return handler

# --- Auto-Discovery of Custom Methods ---

def _discover_custom_methods():
    """
    Automatically discover and register custom prompt methods from the custom_methods directory.
    
    Scans src/agents/custom_methods/ for Python files containing a generate_prompt function
    and registers them using the filename (without .py) as the preset name.
    """
    # Get the directory path relative to this file
    current_dir = Path(__file__).parent
    custom_methods_dir = current_dir / "custom_methods"
    
    if not custom_methods_dir.exists():
        return
    
    # Scan all .py files in the custom_methods directory
    for file_path in custom_methods_dir.glob("*.py"):
        # Skip __init__.py and template.py
        if file_path.name in ["__init__.py", "template.py"]:
            continue
        
        # Extract method name from filename (without .py extension)
        method_name = file_path.stem
        
        try:
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(method_name, file_path)
            if spec is None or spec.loader is None:
                continue
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if the module has a generate_prompt function
            if hasattr(module, "generate_prompt") and callable(module.generate_prompt):
                # Register the function using the filename as the preset name
                SystemPromptRegistry.register(method_name, module.generate_prompt)
                # Only print if verbose mode is enabled (via environment variable)
                if os.getenv("HS_BENCH_VERBOSE", "").lower() in ("1", "true", "yes"):
                    print(f"  ✓ Registered custom method: {method_name}")
            # Silently skip files without generate_prompt (they might be helper modules)
        except Exception as e:
            # Only print errors, not warnings about missing functions
            print(f"  ✗ Error loading custom method {file_path.name}: {e}")

# Auto-discover custom methods on module import
_discover_custom_methods()

# Register backward compatibility aliases
# "empty" is an alias for "v1_empty" to maintain compatibility with existing code
if "v1_empty" in SystemPromptRegistry._presets and "empty" not in SystemPromptRegistry._presets:
    SystemPromptRegistry.register("empty", SystemPromptRegistry._presets["v1_empty"])
