"""
Generate Generative Agents-style background information using LLM.

This script adapts the Generative Agents background generation method to work with
experimental contexts and demographic information. It uses an LLM (e.g., Gemini Flash)
to generate narrative, semicolon-delimited background descriptions.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

# Try to import Gemini client, fallback to OpenAI-style API
try:
    from legacy.validation_pipeline.utils.gemini_client import GeminiClient
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class AgentBackgroundGenerator:
    """
    Generate Generative Agents-style background information from demographics and experimental context.
    """

    def __init__(
        self,
        model: str = "gemini-flash",
        api_key: Optional[str] = None,
        use_gemini: bool = True
    ):
        """
        Initialize the background generator.

        Args:
            model: Model name. For Gemini: "gemini-flash", "gemini-pro", etc.
                  For OpenAI/OpenRouter: model identifier string
            api_key: API key (if None, reads from environment)
            use_gemini: If True, use Gemini API. If False, use OpenAI/OpenRouter API.
        """
        self.model = model
        self.use_gemini = use_gemini and HAS_GEMINI

        if self.use_gemini:
            # Map friendly names to Gemini model names
            gemini_model_map = {
                "gemini-flash": "models/gemini-1.5-flash-latest",
                "gemini-pro": "models/gemini-1.5-pro-latest",
                "gemini-flash-3": "models/gemini-3-flash-preview",
            }
            actual_model = gemini_model_map.get(model, model)
            self.client = GeminiClient(model=actual_model, api_key=api_key)
        else:
            # Use OpenAI/OpenRouter API
            if not HAS_OPENAI:
                raise ImportError("OpenAI package required. Install with: pip install openai")

            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("API key required. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or pass api_key parameter.")

            # Determine if using OpenRouter (has "/" in model name) or OpenAI
            self.is_openrouter = "/" in model
            if self.is_openrouter:
                self.api_base = "https://openrouter.ai/api/v1"
            else:
                self.api_base = "https://api.openai.com/v1"

            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def generate_background(
        self,
        name: str,
        age: int,
        gender: Optional[str] = None,
        education: Optional[str] = None,
        occupation: Optional[str] = None,
        experimental_context: Optional[str] = None,
        additional_demographics: Optional[Dict[str, Any]] = None,
        relationship_context: Optional[List[Dict[str, Any]]] = None,
        seed_intent: Optional[str] = None
    ) -> str:
        """
        Generate a Generative Agents-style background paragraph.

        Args:
            name: Full name of the agent
            age: Age in years
            gender: Gender (optional)
            education: Education level (e.g., "college student", "high school graduate")
            occupation: Occupation/job (e.g., "pharmacy shopkeeper", "college professor")
            experimental_context: Description of the experimental context/study
            additional_demographics: Additional demographic info (dict with any fields)
            relationship_context: List of relationship dicts with keys:
                - name: Other person's name
                - relationship: Type (e.g., "spouse", "colleague", "neighbor", "friend")
                - details: Additional info about the relationship
            seed_intent: Optional initial intent/goal for the agent

        Returns:
            Semicolon-delimited narrative background paragraph
        """
        # Build the prompt
        prompt = self._build_prompt(
            name=name,
            age=age,
            gender=gender,
            education=education,
            occupation=occupation,
            experimental_context=experimental_context,
            additional_demographics=additional_demographics,
            relationship_context=relationship_context,
            seed_intent=seed_intent
        )

        # Call LLM
        if self.use_gemini:
            response = self.client.generate_content(
                prompt=prompt,
                temperature=1.0,
                max_tokens=2048
            )
        else:
            # OpenAI/OpenRouter API
            messages = [
                {"role": "system", "content": "You are a creative writer who generates believable character backgrounds for psychological experiments."},
                {"role": "user", "content": prompt}
            ]

            response_obj = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1.0,
                max_tokens=2048
            )
            response = response_obj.choices[0].message.content

        # Clean and return
        return self._clean_response(response)

    def _build_prompt(
        self,
        name: str,
        age: int,
        gender: Optional[str],
        education: Optional[str],
        occupation: Optional[str],
        experimental_context: Optional[str],
        additional_demographics: Optional[Dict[str, Any]],
        relationship_context: Optional[List[Dict[str, Any]]],
        seed_intent: Optional[str]
    ) -> str:
        """Build the prompt for LLM generation."""

        prompt_parts = [
            f"Generate a life biography for {name}.",
            "",
            "STYLE:",
            f"- Start with: 'You are {name}.' then '{name} is ...'",
            "- Single paragraph, semicolon-delimited statements",
            "- 5-6 statements about: personality, routines, habits, hobbies, living situation, relationships",
            "- NO experiments, studies, research, trials, or problem scenarios",
            "",
            "DATA:",
            f"- Name: {name}, Age: {age}",
        ]

        if gender:
            prompt_parts.append(f"- Gender: {gender}")
        if education:
            prompt_parts.append(f"- Education: {education}")
        if occupation:
            prompt_parts.append(f"- Occupation: {occupation}")

        prompt_parts.append("")
        prompt_parts.append("EXAMPLE (John Lin):")
        prompt_parts.append('John Lin is a pharmacy shopkeeper at the Willow Market and Pharmacy who loves to help people. He is always looking for ways to make the process of getting medication easier for his customers; John Lin is living with his wife, Mei Lin, who is a college professor, and son, Eddy Lin, who is a student studying music theory; John Lin loves his family very much; John Lin has known the old couple next-door, Sam Moore and Jennifer Moore, for a few years; John Lin thinks Sam Moore is a kind and nice man; John Lin knows his neighbor, Yuriko Yamamoto, well.')

        prompt_parts.append("")
        prompt_parts.append(f"Generate bio for {name} (5-6 statements, pure life only, no experiments).")

        return "\n".join(prompt_parts)

    def _clean_response(self, response: str) -> str:
        """Clean the LLM response to extract just the background paragraph."""
        if not response:
            return ""

        # Remove markdown code blocks if present
        if "```" in response:
            # Extract content between code blocks
            parts = response.split("```")
            if len(parts) >= 3:
                response = parts[1]
                # Remove language identifier if present
                if "\n" in response:
                    lines = response.split("\n")
                    if lines[0].strip() in ["text", "markdown", ""]:
                        response = "\n".join(lines[1:])

        # Remove leading/trailing whitespace
        response = response.strip()

        # Ensure it ends with proper punctuation (not a semicolon if it's the last thing)
        # Actually, semicolons are fine - that's the format we want

        return response

    def generate_backgrounds_batch(
        self,
        profiles: List[Dict[str, Any]],
        experimental_context: Optional[str] = None
    ) -> List[str]:
        """
        Generate backgrounds for multiple participants.

        Args:
            profiles: List of profile dicts, each with name, age, gender, etc.
            experimental_context: Shared experimental context for all participants

        Returns:
            List of background paragraphs
        """
        backgrounds = []
        for profile in profiles:
            background = self.generate_background(
                name=profile.get("name", "Participant"),
                age=profile.get("age", 25),
                gender=profile.get("gender"),
                education=profile.get("education"),
                occupation=profile.get("occupation"),
                experimental_context=experimental_context,
                additional_demographics=profile.get("additional_demographics"),
                relationship_context=profile.get("relationships"),
                seed_intent=profile.get("seed_intent")
            )
            backgrounds.append(background)
        return backgrounds


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Generative Agents-style backgrounds")
    parser.add_argument("--name", type=str, required=True, help="Participant name")
    parser.add_argument("--age", type=int, required=True, help="Age")
    parser.add_argument("--gender", type=str, help="Gender")
    parser.add_argument("--education", type=str, help="Education level")
    parser.add_argument("--occupation", type=str, help="Occupation")
    parser.add_argument("--experimental-context", type=str, help="Experimental context/study description")
    parser.add_argument("--model", type=str, default="gemini-flash", help="LLM model to use")
    parser.add_argument("--api-key", type=str, help="API key (or set GOOGLE_API_KEY/OPENROUTER_API_KEY)")
    parser.add_argument("--use-gemini", action="store_true", default=True, help="Use Gemini API (default)")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI/OpenRouter API instead")
    parser.add_argument("--output", type=str, help="Output file path (JSON)")
    parser.add_argument("--seed-intent", type=str, help="Initial intent/goal for the agent")

    args = parser.parse_args()

    generator = AgentBackgroundGenerator(
        model=args.model,
        api_key=args.api_key,
        use_gemini=args.use_gemini and not args.use_openai
    )

    # Parse relationships if provided as JSON string
    relationships = None
    if hasattr(args, 'relationships'):
        try:
            relationships = json.loads(args.relationships) if args.relationships else None
        except:
            relationships = None

    background = generator.generate_background(
        name=args.name,
        age=args.age,
        gender=args.gender,
        education=args.education,
        occupation=args.occupation,
        experimental_context=args.experimental_context,
        seed_intent=args.seed_intent,
        relationship_context=relationships
    )

    print("Generated Background:")
    print("=" * 80)
    print(background)
    print("=" * 80)

    if args.output:
        output_data = {
            "name": args.name,
            "age": args.age,
            "gender": args.gender,
            "education": args.education,
            "occupation": args.occupation,
            "background": background,
            "experimental_context": args.experimental_context
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
