"""
JSON Generator - Generates compatible JSON files (metadata.json, specification.json, ground_truth.json)
and materials files
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.llm.factory import get_client
from generation_pipeline.utils.pdf_extractor import extract_pdf_text

PDF_TEXT_MAX_CHARS = 400000


class JSONGenerator:
    """Generate JSON files compatible with existing study format"""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "models/gemini-3-flash-preview",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize JSON generator.

        Args:
            provider: One of gemini, openai, anthropic, xai, openrouter
            model: Model name for LLM-based materials generation
            api_key: Optional API key
            api_base: Optional API base URL
        """
        self.client = get_client(provider=provider, model=model, api_key=api_key, api_base=api_base)

    def generate_metadata(
        self,
        extraction_result: Dict[str, Any],
        study_id: str,
        pdf_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate metadata.json using LLM to infer domain/subdomain and structure.

        Args:
            extraction_result: Results from stage2 extraction
            study_id: Study ID (e.g., "study_005")
            pdf_path: Optional path to PDF file for context

        Returns:
            Dictionary matching metadata.json format
        """
        studies = extraction_result.get('studies', [])
        if not studies:
            raise ValueError("No studies found")

        # Collect all scenarios from all sub-studies
        scenarios = []
        for study in studies:
            sub_studies = study.get('sub_studies', [])
            for sub_study in sub_studies:
                sub_id = sub_study.get('sub_study_id', '')
                if sub_id:
                    scenarios.append(sub_id)

        # Use LLM to infer domain/subdomain and keywords
        extraction_summary = json.dumps(extraction_result, indent=2, ensure_ascii=False)

        prompt = f"""You are a metadata generator. Based on the extraction results, generate the domain, subdomain, keywords, and findings for this study.

EXTRACTION RESULTS:
{extraction_summary[:120000]}...

TASK:
Generate a JSON object with:
- domain: The psychological domain (e.g., "social_psychology", "cognitive_psychology", "developmental_psychology")
- subdomain: The specific subdomain (e.g., "social_cognition", "judgment_and_decision_making", "memory")
- keywords: List of relevant keywords (3-5 keywords)
- findings: List of findings extracted from the study, each with:
  - finding_id: Unique identifier (e.g., "F1", "F2")
  - main_hypothesis: The main hypothesis or claim for this finding
  - weight: Default weight of 1.0 (can be adjusted later)
  - tests: List of statistical tests for this finding, each with:
    - test_name: Name of the statistical test
    - weight: Default weight of 1.0 (can be adjusted later)

OUTPUT FORMAT (JSON only):
{{
    "domain": "domain_name",
    "subdomain": "subdomain_name",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "findings": [
        {{
            "finding_id": "F1",
            "main_hypothesis": "Description of the finding/hypothesis",
            "weight": 1.0,
            "tests": [
                {{
                    "test_name": "Name of statistical test",
                    "weight": 1.0
                }}
            ]
        }}
    ]
}}

STRICT REQUIREMENTS:
1. **NO HALLUCINATION**: Only infer metadata based on the provided EXTRACTION RESULTS.
2. **HONESTY**: If the domain or subdomain cannot be clearly determined from the text, use null.
3. **SPECIFICITY**: Use underscores instead of spaces for domain and subdomain names.
4. **FINDINGS**: Extract all findings mentioned in the extraction results. If no findings are explicitly mentioned, use an empty array [].

Generate the JSON:
"""

        try:
            if pdf_path and pdf_path.exists():
                pdf_text = extract_pdf_text(pdf_path, max_chars=PDF_TEXT_MAX_CHARS)
                full_prompt = f"PDF content:\n\n{pdf_text}\n\n{prompt}"
                response = self.client.generate_content(prompt=full_prompt, temperature=0.3)
            else:
                response = self.client.generate_content(prompt=prompt, temperature=0.3)

            if response:
                # Parse LLM response
                llm_metadata = self._parse_json_response(response)
                domain = llm_metadata.get('domain')
                subdomain = llm_metadata.get('subdomain')
                keywords = llm_metadata.get('keywords', [])
                llm_findings = llm_metadata.get('findings', [])
            else:
                domain = None
                subdomain = None
                keywords = []
                llm_findings = []
        except Exception as e:
            print(f"Warning: Error generating metadata with LLM: {e}")
            domain = None
            subdomain = None
            keywords = []
            llm_findings = []

        # Extract keywords from phenomenon if LLM didn't provide
        if not keywords:
            study = studies[0]
            phenomenon = study.get('phenomenon', '')
            if phenomenon:
                keywords = [phenomenon.replace(' ', '_')]

        # Use LLM-extracted findings if available, otherwise extract from extraction_result
        findings = llm_findings if llm_findings else []

        # Fallback: Extract findings from extraction_result if LLM didn't provide them
        if not findings:
            for study in studies:
                sub_studies = study.get('sub_studies', [])
                for sub_study in sub_studies:
                    # Look for findings in human_data or statistical_results
                    human_data = sub_study.get('human_data', {})
                    statistical_results = human_data.get('statistical_results', [])

                    # Group statistical tests by finding/hypothesis
                    finding_map = {}
                    for stat_result in statistical_results:
                        finding_key = stat_result.get('claim', '') or stat_result.get('hypothesis', '') or 'Unknown Finding'
                        if finding_key not in finding_map:
                            finding_map[finding_key] = {
                                "finding_id": f"F{len(findings) + 1}",
                                "main_hypothesis": finding_key,
                                "weight": 1.0,  # Default weight
                                "tests": []
                            }
                        finding_map[finding_key]["tests"].append({
                            "test_name": stat_result.get('test_name', 'Unknown Test'),
                            "weight": 1.0  # Default weight for each test
                        })

                    findings.extend(list(finding_map.values()))

        # If no findings found, create empty findings list
        # (will be populated later from ground_truth.json if available)
        if not findings:
            findings = []

        return {
            "id": study_id,
            "title": extraction_result.get('paper_title', ''),
            "authors": extraction_result.get('paper_authors', []),
            "year": extraction_result.get('paper_year'),
            "domain": domain,
            "subdomain": subdomain,
            "keywords": keywords,
            "difficulty": "medium",  # TODO: Determine from study complexity
            "description": extraction_result.get('paper_abstract', '')[:2000] + "...",
            "scenarios": scenarios,
            "findings": findings
        }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        response_text = response.strip()

        # Remove markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON object
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}

    def generate_specification(self, extraction_result: Dict[str, Any], study_id: str) -> Dict[str, Any]:
        """
        Generate specification.json using LLM to ensure accuracy and avoid defaults.
        """
        extraction_summary = json.dumps(extraction_result, indent=2, ensure_ascii=False)

        prompt = f"""You are an experimental design expert. Based on the extraction results from a research paper, generate a `specification.json` object.

EXTRACTION RESULTS:
{extraction_summary[:120000]}...

TASK:
Generate a JSON object that describes the experimental setup.
ONLY include information explicitly present in the extraction results.

JSON STRUCTURE:
{{
    "study_id": "{study_id}",
    "title": "Paper Title",
    "participants": {{
        "n": 0, // Total N found in paper. 0 if not found.
        "population": "string", // e.g. 'undergraduates', 'MTurk workers'. null if not found.
        "recruitment_source": "string", // null if not found.
        "demographics": {{}}, // Any age, gender, etc. reported. Empty if not found.
        "by_sub_study": {{
            "sub_id": {{ "n": 0, ... }} // Breakdowns if available.
        }}
    }},
    "design": {{
        "type": "string", // e.g. 'Between-Subjects', 'Within-Subjects', 'Mixed'
        "factors": [
            {{
                "name": "Factor Name",
                "levels": ["level1", "level2"],
                "type": "Between-Subjects|Within-Subjects"
            }}
        ]
    }},
    "procedure": {{
        "steps": ["Step 1", "Step 2"] // Extract actual steps from the paper.
    }}
}}

STRICT REQUIREMENTS:
1. **NO HALLUCINATION**: If participant N is not mentioned, use 0. If population is not mentioned, use null. DO NOT guess.
2. **NO DEFAULTS**: Do not assume "undergraduates" or "0.05" or any other common value.
3. **ACCURACY**: Extract factors and levels precisely from the experimental conditions described.

Generate the JSON:
"""
        try:
            response = self.client.generate_content(prompt=prompt, temperature=0.1)
            if response:
                return self._parse_json_response(response)
        except Exception as e:
            print(f"Warning: Error generating specification with LLM: {e}")

        return {
            "study_id": study_id,
            "title": extraction_result.get('paper_title', ''),
            "participants": {"n": 0, "demographics": {}},
            "design": {"factors": []},
            "procedure": {"steps": []}
        }

    def generate_ground_truth(
        self,
        extraction_result: Dict[str, Any],
        study_id: str,
        debug_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate ground_truth.json using LLM to ensure validation criteria
        match the paper's claims and statistical results.
        """
        studies = extraction_result.get('studies', [])
        if not studies:
            raise ValueError("No studies found")

        extraction_summary = json.dumps(extraction_result, indent=2, ensure_ascii=False)

        prompt = f"""You are a scientific data extractor. Your goal is to FAITHFULLY extract ALL validation criteria and statistical results from a research paper based on the provided extraction results.

EXTRACTION RESULTS:
{extraction_summary[:120000]}...

TASK:
Generate a `ground_truth.json` object.

**CRITICAL EXTRACTION REQUIREMENTS (READ CAREFULLY):**

1. **EXTRACT ALL STATISTICAL TESTS - NO EXCEPTIONS**:
   - If the extraction results list 17 item-level t-tests, you MUST output ALL 17 in the JSON.
   - DO NOT summarize. DO NOT select "representative" items. Output EVERY SINGLE numerical finding.
   - The number of items in your `original_data_points.data` MUST match the number of items in the input `item_level_results`.

2. **PRIORITIZE FULL STATISTICS**:
   - For each finding, you MUST first look for standard statistics: t-value, F-value, r, or chi2.
   - If a standard statistic is available, use it (e.g., "t(79) = 2.66").
   - If ONLY a p-value or a proportion is reported (e.g., for Sign tests or descriptive claims), you MUST use the following format: `p=VALUE, n=N` (e.g., "p=0.01, n=92").
   - Strip all non-numeric symbols from the values (e.g., convert "p < .01" to "p=0.01").
   - These values are CRITICAL for calculating the Bayesian Alignment Score (PAS).

3. **HIERARCHICAL GROUPING**:
   - Findings MUST be nested under the correct Study/Experiment they belong to.

4. **DATA TRACEABILITY**:
   - `original_data_points` must show the exact numbers (Means, SDs, Ns, percentages) used to calculate the reported statistics for EVERY item.

5. **NO HALLUCINATION**:
   - Only include numbers and claims explicitly present in the EXTRACTION RESULTS.

JSON STRUCTURE:
{{
    "study_id": "{study_id}",
    "title": "Paper Title",
    "authors": ["Author 1"],
    "year": 19XX,
    "studies": [
        {{
            "study_id": "Study 1",
            "study_name": "Name of the study",
            "findings": [
                {{
                    "finding_id": "F1",
                    "main_hypothesis": "The core psychological claim (e.g., 'People perceive a false consensus').",
                    "statistical_tests": [
                        {{
                            "test_name": "Exact test name (e.g., 'Independent t-test').",
                            "statistical_hypothesis": "Formal comparison (e.g., 'Mean A > Mean B').",
                            "reported_statistics": "Verbatim statistics string (e.g., 't(79) = 5.4, p < .001').",
                            "significance_level": 0.05,
                            "expected_direction": "positive|negative|none"
                        }}
                    ],
                    "original_data_points": {{
                        "description": "Description of data used for these statistics.",
                        "data": {{
                            "item_or_condition_name": {{
                                "mean": 0.0,
                                "sd": 0.0,
                                "n": 0,
                                "percentage": 0.0,
                                "t": 0.0,
                                "other_stat": 0.0
                            }}
                        }}
                    }}
                }}
            ]
        }}
    ],
    "overall_original_results": {{
        "description": "Structured repository of ALL numerical findings.",
        "data_tables": [
            {{
                "table_name": "Table Name",
                "headers": ["Col1", "Col2"],
                "rows": [["Val1", "Val2"]]
            }}
        ],
        "all_raw_data": {{}}
    }}
}}

VERIFICATION CHECKLIST (self-check before output):
- [ ] Count of items in input `item_level_results` == Count of items in your output `data`?
- [ ] Are ALL t-values/F-values preserved and stripped of symbols?
- [ ] Did you avoid summarizing or skipping items?

CRITICAL: Output ONLY valid JSON. Do NOT include comments, markdown code blocks, or any text outside the JSON object.

Generate the JSON:
"""
        try:
            response = self.client.generate_content(prompt=prompt, temperature=0.2)
            if response:
                # Save raw response BEFORE parsing (in case parsing fails)
                debug_path = (Path(debug_dir) if debug_dir else Path(f"data/studies/{study_id}")) / "ground_truth_raw_response.txt"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(str(response), encoding='utf-8')

                parsed = self._parse_json_response(response)
                if parsed and len(parsed) > 0:  # Check for non-empty dict
                    return parsed
                # If parsing failed, raise error with debug file location
                raise ValueError(f"Failed to parse JSON response. Raw response saved to {debug_path}")
        except ValueError:
            raise  # Re-raise ValueError (our custom error)
        except Exception as e:
            print(f"Error generating ground_truth with LLM: {e}")
            raise

    def generate_materials(
        self,
        extraction_result: Dict[str, Any],
        study_dir: Path,
        pdf_path: Optional[Path] = None
    ) -> List[Path]:
        """
        Generate materials files using LLM-based dynamic function generation.

        This method uses LLM to generate a study-specific generate_materials function
        that knows how to extract complete materials from the extraction_result.

        Args:
            extraction_result: Results from stage2 extraction
            study_dir: Study directory path
            pdf_path: Optional path to PDF file for context

        Returns:
            List of generated material file paths
        """
        # Load ground_truth.json if it exists (to ensure label consistency)
        ground_truth = None
        gt_path = study_dir / "ground_truth.json"
        if gt_path.exists():
            try:
                with open(gt_path, 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)
            except Exception:
                pass  # If loading fails, continue without it

        # Use LLM to generate a study-specific materials generation function
        materials_generator_code = self._generate_materials_function_with_llm(
            extraction_result,
            pdf_path,
            ground_truth=ground_truth
        )

        # Execute the generated function
        materials_dir = study_dir / "materials"
        materials_dir.mkdir(parents=True, exist_ok=True)

        # Create a namespace for executing the generated code
        namespace = {
            'json': json,
            'Path': Path,
            'materials_dir': materials_dir,
            'extraction_result': extraction_result,
            'study_dir': study_dir
        }

        try:
            exec(materials_generator_code, namespace)
            # Call the generated function
            if 'generate_materials' in namespace:
                generated_files = namespace['generate_materials'](extraction_result, materials_dir)
            else:
                generated_files = namespace.get('generated_files', [])

            # Ensure all items are Path objects
            generated_files = [Path(f) if not isinstance(f, Path) else f for f in generated_files]

            if not generated_files:
                print(f"Warning: Generated function returned no files, falling back to basic generation")
                generated_files = self._generate_materials_basic(extraction_result, study_dir)
        except Exception as e:
            print(f"Warning: Error executing generated materials function: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic generation
            generated_files = self._generate_materials_basic(extraction_result, study_dir)

        return generated_files

    def _generate_materials_function_with_llm(
        self,
        extraction_result: Dict[str, Any],
        pdf_path: Optional[Path],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Use LLM to generate a study-specific generate_materials function.

        Args:
            extraction_result: Results from stage2 extraction
            pdf_path: Optional path to PDF file

        Returns:
            Python code string for the generated function
        """
        extraction_summary = json.dumps(extraction_result, indent=2, ensure_ascii=False)

        # Include ground_truth in prompt if available (for label consistency)
        gt_context = ""
        if ground_truth:
            # Extract all scenario/item keys from ground_truth for reference
            # Group by study for better organization
            gt_keys_by_study = {}
            all_gt_keys = []

            for study_gt in ground_truth.get("studies", []):
                study_id = study_gt.get("study_id", "")
                study_keys = []

                for finding in study_gt.get("findings", []):
                    data_points = finding.get("original_data_points", {}).get("data", {})
                    if data_points:
                        keys = list(data_points.keys())
                        study_keys.extend(keys)
                        all_gt_keys.extend(keys)

                if study_keys:
                    gt_keys_by_study[study_id] = list(set(study_keys))

            if all_gt_keys:
                # Create a comprehensive list organized by study
                gt_keys_list = []
                for study_id, keys in gt_keys_by_study.items():
                    gt_keys_list.append(f"  {study_id}: {keys}")

                gt_context = f"""
**CRITICAL: GROUND TRUTH KEYS (MUST MATCH EXACTLY)**

The following keys are used in ground_truth.json. You MUST use these EXACT same keys as `metadata.label` for corresponding items:

ALL GROUND TRUTH KEYS (organized by study):
{chr(10).join(gt_keys_list)}

COMPLETE LIST OF ALL KEYS:
{json.dumps(sorted(set(all_gt_keys)), indent=2)}

**MANDATORY REQUIREMENTS:**
1. For EVERY item in materials that corresponds to a key in the list above, you MUST set `metadata.label` to the EXACT key from the list.
2. If an item corresponds to a ground truth key, use that exact key.
3. If an item does NOT correspond to any ground truth key (e.g., background/filler items), you may omit the label or use a descriptive name, but prioritize matching the keys above.
4. The label matching is case-sensitive and must match exactly (after normalization).

**EXAMPLE:**
- If ground truth has key "shy", then materials item should have `"metadata": {{"label": "shy"}}`
- If ground truth has key "hometown_gt_200k", then materials item should have `"metadata": {{"label": "hometown_gt_200k"}}`
- DO NOT use "Shy (not shy)" or "Hometown > 200k" - use the EXACT key from the list above.

"""

        # Build prompt
        # Ensure ground truth context is included BEFORE truncating extraction_summary
        # Truncate extraction_summary to leave room for gt_context
        max_extraction_length = 100000  # Leave room for gt_context
        truncated_extraction = extraction_summary[:max_extraction_length]
        if len(extraction_summary) > max_extraction_length:
            truncated_extraction += "...\n[Truncated for length]"

        prompt = f"""You are a Python code generator. Generate a complete `generate_materials` function that extracts materials for a SIMULATION AGENT into a UNIFIED JSON format.

{gt_context}
EXTRACTION RESULTS:
{truncated_extraction}
TASK:
Generate a Python function that creates standardized JSON files for each sub-study.
1. Each replicable sub-study should have EXACTLY ONE JSON file in the `materials/` directory named `{{sub_study_id}}.json`.
2. The JSON structure for each file MUST follow this schema:
{{
    "sub_study_id": "string (the unique ID for this condition/task)",
    "instructions": "string (the FULL text of scenario or instructions given to the participant)",
    "items": [
        {{
            "id": "string (item number/id)",
            "question": "string (the actual question or stimulus text)",
            "options": ["string"], // Optional: include if the question has fixed choices or Likert scale levels
            "type": "string (e.g., 'multiple_choice', 'open_ended', 'estimation', 'likert')",
            "metadata": {{
                "label": "string" // CRITICAL: If this item has a corresponding key in ground_truth.json, use the EXACT same key as the label
            }} // Optional: any item-specific data (anchors, true values, etc.)
        }}
    ]
}}

REQUIREMENTS:
- UNIFIED FORMAT: All materials MUST be JSON. No .txt files.
- STRICT DIRECTORY: You MUST ONLY write files to the `materials_dir` provided in the namespace.
- NO OVERWRITE: DO NOT attempt to modify or overwrite any files in `generation_pipeline/outputs/` or the parent directory.
- SELF-CONTAINED: Instructions MUST be complete. Reconstruct them if they refer to other experiments.
- ITEMS: If a sub-study has multiple questions/items (e.g., 15 estimation tasks), they MUST all be listed in the "items" array of that sub-study's JSON.
- If a sub-study is just a single scenario with one choice, the "items" array will have one object.
- **LABEL CONSISTENCY (CRITICAL)**: If ground_truth keys are provided above, you MUST use those EXACT keys as `metadata.label` for corresponding items. This ensures automatic matching between materials and ground_truth.

EXAMPLE CODE STRUCTURE:
```python
def generate_materials(extraction_result, materials_dir):
    from pathlib import Path
    import json

    generated_files = []
    studies = extraction_result.get('studies', [])

    for study in studies:
        for sub in study.get('sub_studies', []):
            sub_id = sub.get('sub_study_id', 'unknown')

            # Combine everything into one JSON object per sub-study
            material_data = {{
                "sub_study_id": sub_id,
                "instructions": sub.get('content', ''),
                "items": sub.get('items', [])
            }}

            # Ensure items have standard structure (id, question, etc.)
            # [Logic to transform sub.get('items') into the required schema goes here]

            file_path = materials_dir / f"{{sub_id}}.json"
            file_path.write_text(json.dumps(material_data, indent=2, ensure_ascii=False), encoding='utf-8')
            generated_files.append(file_path)

    return generated_files
```

OUTPUT:
Provide ONLY the complete Python function code. The function should be production-ready.
"""

        # Call LLM (with PDF if available)
        try:
            if pdf_path and pdf_path.exists():
                pdf_text = extract_pdf_text(pdf_path, max_chars=PDF_TEXT_MAX_CHARS)
                full_prompt = f"PDF content:\n\n{pdf_text}\n\n{prompt}"
                response = self.client.generate_content(prompt=full_prompt)
            else:
                response = self.client.generate_content(prompt=prompt)
        except Exception as e:
            raise RuntimeError(f"Error calling LLM API: {e}. Check API key and provider env.")

        if response is None:
            raise ValueError("LLM returned None response. Check API key and network connection.")

        # Extract code from response
        code = self._extract_code_from_response(response)

        return code

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response"""
        response_text = response.strip()

        # Remove markdown code blocks if present
        if '```python' in response_text:
            response_text = response_text.split('```python')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()

        return response_text

    def _generate_materials_basic(
        self,
        extraction_result: Dict[str, Any],
        study_dir: Path
    ) -> List[Path]:
        """
        Basic fallback materials generation (unifying to JSON).

        Args:
            extraction_result: Results from stage2 extraction
            study_dir: Study directory path

        Returns:
            List of generated material file paths
        """
        materials_dir = study_dir / "materials"
        materials_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []
        studies = extraction_result.get('studies', [])

        for study in studies:
            sub_studies = study.get('sub_studies', [])

            for sub_study in sub_studies:
                sub_id = sub_study.get('sub_study_id', '')
                content = sub_study.get('content', '')
                items = sub_study.get('items', [])

                if not sub_id:
                    continue

                # Unify to JSON
                material_data = {
                    "sub_study_id": sub_id,
                    "instructions": content,
                    "items": items
                }

                file_path = materials_dir / f"{sub_id}.json"
                file_path.write_text(
                    json.dumps(material_data, indent=2, ensure_ascii=False),
                    encoding='utf-8'
                )
                generated_files.append(file_path)

        return generated_files

    def inject_gt_keys_into_materials(
        self,
        study_dir: Path,
        material_files: List[Path]
    ) -> Dict[str, Any]:
        """
        Inject gt_key into materials items by matching against ground_truth.json.

        This method:
        1. Loads ground_truth.json
        2. Extracts all scenario/item keys from ground_truth
        3. Matches each item's label to a gt_key
        4. Writes gt_key to item.metadata.gt_key
        5. Reports coverage statistics

        Args:
            study_dir: Study directory path
            material_files: List of material JSON file paths

        Returns:
            Dictionary with coverage statistics:
            {
                "sub_studies": {
                    "sub_study_id": {
                        "gt_keys_total": int,
                        "items_total": int,
                        "matched": int,
                        "missing": int,
                        "missing_items": [item_ids]
                    }
                }
            }
        """
        gt_path = study_dir / "ground_truth.json"
        if not gt_path.exists():
            print(f"Warning: ground_truth.json not found at {gt_path}. Skipping gt_key injection.")
            return {}

        # Load ground truth
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        # Extract all gt_keys organized by sub_study_id
        # First, collect all materials sub_study_ids
        materials_sub_ids = set()
        for material_file in material_files:
            if material_file.exists():
                with open(material_file, 'r', encoding='utf-8') as f:
                    material_data = json.load(f)
                    sub_id = material_data.get("sub_study_id", "")
                    if sub_id:
                        materials_sub_ids.add(sub_id)

        # Build mapping from study labels to sub_study_ids
        # Common patterns for study_001:
        # "Study 1" -> "study_1_hypothetical_stories"
        # "Study 2" -> "study_2_personal_description_items"
        # "Study 3" -> "study_3_sandwich_board_hypothetical"
        gt_keys_by_substudy = {}
        all_gt_keys = set()  # Fallback: all keys if sub-study matching fails

        for study_gt in ground_truth.get("studies", []):
            study_label = study_gt.get("study_id", "")

            # Extract keys from all findings
            all_keys_in_study = []
            for finding in study_gt.get("findings", []):
                data_points = finding.get("original_data_points", {}).get("data", {})
                if data_points:
                    all_keys_in_study.extend(list(data_points.keys()))

            all_gt_keys.update(all_keys_in_study)

            # Try to match study to sub_study_id using common patterns
            matched_sub_ids = []
            study_num_match = re.search(r'Study\s+(\d+)', study_label)

            if study_num_match:
                study_num = study_num_match.group(1)
                # Try common patterns
                patterns = [
                    f"study_{study_num}_",
                    f"_study_{study_num}_",
                    f"study_{study_num}",
                ]
                for sub_id in materials_sub_ids:
                    if any(pattern in sub_id for pattern in patterns):
                        matched_sub_ids.append(sub_id)

            # Also try keyword matching (e.g., "hypothetical" in study_label -> "hypothetical_stories" in sub_id)
            if not matched_sub_ids:
                study_keywords = [w.lower() for w in study_label.split() if len(w) > 4]
                for sub_id in materials_sub_ids:
                    sub_id_lower = sub_id.lower()
                    if any(kw in sub_id_lower for kw in study_keywords):
                        matched_sub_ids.append(sub_id)

            # If we found matches, assign keys to those sub_studies
            if matched_sub_ids:
                for sub_id in matched_sub_ids:
                    if sub_id not in gt_keys_by_substudy:
                        gt_keys_by_substudy[sub_id] = []
                    gt_keys_by_substudy[sub_id].extend(all_keys_in_study)

        # Process each material file
        stats = {}
        total_matched = 0
        total_items = 0
        total_missing = 0

        for material_file in material_files:
            if not material_file.exists():
                continue

            # Load material file
            with open(material_file, 'r', encoding='utf-8') as f:
                material_data = json.load(f)

            sub_study_id = material_data.get("sub_study_id", "")
            items = material_data.get("items", [])

            if not sub_study_id:
                continue

            # Get gt_keys for this sub-study
            gt_keys = gt_keys_by_substudy.get(sub_study_id, [])

            # If not found, try to use all keys (fallback)
            if not gt_keys:
                # Use all unique keys from ground_truth as fallback
                gt_keys = list(all_gt_keys)

            # Initialize stats for this sub-study
            if sub_study_id not in stats:
                stats[sub_study_id] = {
                    "gt_keys_total": len(gt_keys),
                    "items_total": len(items),
                    "matched": 0,
                    "missing": 0,
                    "missing_items": []
                }

            # Process each item
            for item in items:
                total_items += 1
                item_id = item.get("id", "")
                label = item.get("metadata", {}).get("label", "")

                # Skip items without labels (they won't be evaluated against ground_truth)
                if not label:
                    continue

                # Try to find matching gt_key
                gt_key = self._find_gt_key_for_label(label, gt_keys, item_id)

                if gt_key:
                    # Ensure metadata exists
                    if "metadata" not in item:
                        item["metadata"] = {}
                    item["metadata"]["gt_key"] = gt_key
                    stats[sub_study_id]["matched"] += 1
                    total_matched += 1
                else:
                    # Item has label but no matching gt_key
                    # This could mean:
                    # 1. The item doesn't have a corresponding ground_truth entry (not all items are evaluated)
                    # 2. The label format doesn't match ground_truth keys
                    # We'll count it as missing but only fail if we have gt_keys available (meaning we should have matched)
                    stats[sub_study_id]["missing"] += 1
                    total_missing += 1
                    stats[sub_study_id]["missing_items"].append(item_id)

            # Save updated material file
            with open(material_file, 'w', encoding='utf-8') as f:
                json.dump(material_data, f, indent=2, ensure_ascii=False)

        # Print coverage report
        print(f"\n📊 gt_key Coverage Report:")
        print(f"   Total items processed: {total_items}")
        print(f"   Matched: {total_matched}")
        print(f"   Missing: {total_missing}")

        for sub_id, sub_stats in stats.items():
            print(f"\n   {sub_id}:")
            print(f"      Items: {sub_stats['items_total']}, GT Keys: {sub_stats['gt_keys_total']}")
            print(f"      Matched: {sub_stats['matched']}, Missing: {sub_stats['missing']}")
            if sub_stats['missing'] > 0:
                print(f"      Missing items: {', '.join(sub_stats['missing_items'][:5])}")
                if len(sub_stats['missing_items']) > 5:
                    print(f"      ... and {len(sub_stats['missing_items']) - 5} more")

        # Fail only if we have gt_keys available but couldn't match a significant portion
        # Note: Not all items may have corresponding ground_truth entries (e.g., Study 2 has 34 items but only 17 have stats)
        # So we only fail if:
        # 1. We have gt_keys available (meaning there should be matches)
        # 2. The missing rate is very high (>80% of items with labels, or >50% of available gt_keys)
        items_with_labels = sum(s["items_total"] for s in stats.values())
        items_with_gt_keys_available = sum(s["gt_keys_total"] for s in stats.values() if s["gt_keys_total"] > 0)

        if total_missing > 0 and items_with_gt_keys_available > 0:
            # Calculate missing rate relative to items with labels
            missing_rate_vs_items = total_missing / items_with_labels if items_with_labels > 0 else 0
            # Calculate missing rate relative to available gt_keys (more meaningful)
            missing_rate_vs_keys = total_missing / items_with_gt_keys_available if items_with_gt_keys_available > 0 else 0

            # Fail if:
            # - More than 80% of items with labels are missing, OR
            # - More than 50% of available gt_keys couldn't be matched (meaning we're missing most of the evaluable items)
            if missing_rate_vs_items > 0.8 or missing_rate_vs_keys > 0.5:
                raise ValueError(
                    f"gt_key injection failed: {total_missing} items could not be matched to ground_truth keys. "
                    f"Missing rate: {missing_rate_vs_items*100:.1f}% of items with labels, {missing_rate_vs_keys*100:.1f}% of available gt_keys. "
                    f"Please check the mapping between materials labels and ground_truth keys. "
                    f"Note: Some items may not have corresponding ground_truth entries (not all items are evaluated)."
                )
            else:
                print(f"  ⚠️  Warning: {total_missing} items could not be matched. "
                      f"This may be expected if not all items have ground_truth entries "
                      f"(e.g., Study 2 has 34 items but only 17 have statistical results in ground_truth).")

        return stats

    def _find_gt_key_for_label(
        self,
        label: str,
        gt_keys: List[str],
        item_id: str = ""
    ) -> Optional[str]:
        """
        Find the best matching gt_key for a given label.

        Uses multi-step matching:
        1. Exact match (normalized)
        2. Normalized exact match
        3. Token-based similarity (Jaccard)
        4. Keyword matching

        Args:
            label: Item label from materials
            gt_keys: List of available ground truth keys
            item_id: Optional item ID for debugging

        Returns:
            Best matching gt_key or None
        """
        if not label or not gt_keys:
            return None

        # Step 1: Normalize label
        normalized_label = self._normalize_label(label)

        # Step 2: Exact match (normalized)
        if normalized_label in gt_keys:
            return normalized_label

        # Step 3: Try normalized versions of gt_keys
        normalized_gt_map = {self._normalize_label(k): k for k in gt_keys}
        if normalized_label in normalized_gt_map:
            return normalized_gt_map[normalized_label]

        # Step 4: Token-based similarity (Jaccard)
        label_words = set(normalized_label.lower().split())
        best_match = None
        best_score = 0.0

        for gt_key in gt_keys:
            gt_words = set(self._normalize_label(gt_key).lower().split())
            if not label_words or not gt_words:
                continue

            intersection = len(label_words & gt_words)
            union = len(label_words | gt_words)
            score = intersection / union if union > 0 else 0.0

            if score > best_score:
                best_score = score
                best_match = gt_key

        # Require high similarity (>0.7) for token-based match
        if best_score > 0.7:
            return best_match

        # Step 5: Keyword matching (fallback for complex cases)
        # Check if main keywords from label are in gt_key
        for gt_key in gt_keys:
            gt_words = set(self._normalize_label(gt_key).lower().split())
            # If at least 2 significant words match
            significant_words = {w for w in label_words if len(w) > 3}
            if len(significant_words & gt_words) >= 2:
                return gt_key

        return None

    def _normalize_label(self, label: str) -> str:
        """
        Normalize label for matching:
        - Remove parentheses and their content
        - Remove common suffixes (yes/no, on list provided, etc.)
        - Normalize special symbols (more than -> >, less than -> <)
        - Clean whitespace
        """
        if not label:
            return ""

        # Remove parentheses and their content
        normalized = re.sub(r"\s*\([^)]*\)\s*", " ", label)

        # Remove common suffixes
        normalized = re.sub(r"\s+(yes|no|on list provided|don't|do not).*$", "", normalized, flags=re.IGNORECASE)

        # Normalize special symbols
        normalized = re.sub(r"\s+more\s+than\s+", " > ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+less\s+than\s+", " < ", normalized, flags=re.IGNORECASE)

        # Clean whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized
