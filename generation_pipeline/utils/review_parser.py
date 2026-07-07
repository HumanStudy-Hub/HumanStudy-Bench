"""
Review Parser - Parses markdown review files to extract checklist and comments
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class ReviewParser:
    """Parse markdown review files to extract human feedback"""

    @staticmethod
    def parse(md_file: Path) -> Dict[str, Any]:
        """
        Parse markdown review file.

        Args:
            md_file: Path to markdown review file

        Returns:
            Dictionary with parsed information:
            {
                "checklists": [...],  # List of checklist items with status
                "comments": {...},     # Comments by section
                "review_status": {...} # Review status information
            }
        """
        if not md_file.exists():
            raise FileNotFoundError(f"Review file not found: {md_file}")

        content = md_file.read_text(encoding='utf-8')

        return {
            "checklists": ReviewParser._extract_checklists(content),
            "comments": ReviewParser._extract_comments(content),
            "review_status": ReviewParser._extract_review_status(content)
        }

    @staticmethod
    def _extract_checklists(content: str) -> List[Dict[str, Any]]:
        """Extract checklist items from markdown"""
        checklists = []

        # Pattern: - [x] or - [ ] followed by text
        pattern = r'- \[([ x])\] (.+)'

        for match in re.finditer(pattern, content):
            checked = match.group(1).strip() == 'x'
            text = match.group(2).strip()

            # Find which section this checklist belongs to
            section = ReviewParser._find_section(content, match.start())

            checklists.append({
                "checked": checked,
                "text": text,
                "section": section
            })

        return checklists

    @staticmethod
    def _extract_comments(content: str) -> Dict[str, str]:
        """Extract comments from #### Comments: sections"""
        comments = {}

        # Pattern: #### Comments: followed by content until next #### or ---
        pattern = r'#### Comments:\s*\n(.*?)(?=\n####|\n---|$)'

        for match in re.finditer(pattern, content, re.DOTALL):
            comment_text = match.group(1).strip()

            # Find which section this comment belongs to
            section = ReviewParser._find_section(content, match.start())
            comments[section] = comment_text

        return comments

    @staticmethod
    def _extract_review_status(content: str) -> Dict[str, str]:
        """Extract review status section"""
        status = {}

        # Look for Review Status section
        status_match = re.search(r'## Review Status\s*\n(.*?)(?=\n##|$)', content, re.DOTALL)
        if status_match:
            status_text = status_match.group(1)

            # Extract key-value pairs
            for line in status_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().strip('-*').strip()
                    value = value.strip()
                    if key and value:
                        status[key.lower().replace(' ', '_')] = value

        return status

    @staticmethod
    def _find_section(content: str, position: int) -> str:
        """Find which section a position belongs to"""
        # Find the last ### or ## before this position
        before_content = content[:position]

        # Find last section header
        section_match = re.findall(r'^### (.+)$', before_content, re.MULTILINE)
        if section_match:
            return section_match[-1]

        section_match = re.findall(r'^## (.+)$', before_content, re.MULTILINE)
        if section_match:
            return section_match[-1]

        return "unknown"

    @staticmethod
    def get_action(review_status: Dict[str, str]) -> Optional[str]:
        """Extract action from review status"""
        action = review_status.get('action', '').lower()

        if 'continue' in action or 'stage2' in action:
            return 'continue_to_stage2'
        elif 'refine' in action:
            return 'refine'
        elif 'exclude' in action:
            return 'exclude'
        elif 'back' in action:
            return 'back_to_stage1'
        elif 'final' in action:
            return 'continue_to_final'

        return None
