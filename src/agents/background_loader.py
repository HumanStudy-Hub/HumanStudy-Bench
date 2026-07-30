"""
Helper module to load Generative Agents backgrounds for participants.

This module provides utilities to integrate Generative Agents backgrounds
into the participant profile system.
"""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path


class BackgroundLoader:
    """Load stored Generative Agents backgrounds for participants."""

    def __init__(self, storage_dir: Path = None):
        """
        Initialize the background loader.

        Args:
            storage_dir: Directory where backgrounds are stored (default: data/backgrounds)
        """
        if storage_dir is None:
            # Find project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            storage_dir = project_root / "data" / "backgrounds"

        self.storage_dir = Path(storage_dir)

    def load_for_study(
        self,
        study_id: str,
        participant_ids: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Load backgrounds for all participants in a study.

        Args:
            study_id: Study identifier
            participant_ids: List of participant IDs to load (if None, loads all)

        Returns:
            Dict mapping participant_id to background data
        """
        study_dir = self.storage_dir / study_id

        if not study_dir.exists():
            return {}

        backgrounds = {}

        # Load index if exists
        index_path = study_dir / "index.json"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
                available_ids = index.get("participants", [])
        else:
            # Scan directory
            available_ids = []
            for f in study_dir.glob("participant_*.json"):
                try:
                    pid = int(f.stem.split("_")[1])
                    available_ids.append(pid)
                except:
                    continue

        # Filter to requested IDs
        if participant_ids is not None:
            available_ids = [pid for pid in available_ids if pid in participant_ids]

        # Load backgrounds
        for participant_id in available_ids:
            background_path = study_dir / f"participant_{participant_id:04d}.json"
            if background_path.exists():
                try:
                    with open(background_path, 'r', encoding='utf-8') as f:
                        backgrounds[participant_id] = json.load(f)
                except Exception:
                    continue

        return backgrounds

    def load_for_participant(
        self,
        study_id: str,
        participant_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Load background for a single participant.

        Args:
            study_id: Study identifier
            participant_id: Participant ID

        Returns:
            Background data dict or None if not found
        """
        background_path = self.storage_dir / study_id / f"participant_{participant_id:04d}.json"

        if not background_path.exists():
            return None

        try:
            with open(background_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def enrich_profiles(
        self,
        profiles: List[Dict[str, Any]],
        study_id: str
    ) -> List[Dict[str, Any]]:
        """
        Enrich participant profiles with Generative Agents backgrounds.

        Args:
            profiles: List of participant profile dicts
            study_id: Study identifier

        Returns:
            List of enriched profiles with 'generative_background' field
        """
        # Load all backgrounds for this study
        backgrounds = self.load_for_study(study_id)

        # Enrich profiles
        enriched = []
        for profile in profiles:
            participant_id = profile.get('participant_id')

            if participant_id in backgrounds:
                # Add Generative Agents background
                profile = profile.copy()
                profile['generative_background'] = backgrounds[participant_id].get('background', '')
                profile['study_id'] = study_id

            enriched.append(profile)

        return enriched
