"""
Pipeline interface for HumanStudy-Bench generation stages.

Stages:
  1 - Study inventory and simulation eligibility (PDF → stage1 review JSON+MD)
  2 - Study/finding extraction (stage1 + PDF → stage2 review JSON+MD)
  3 - Source/material evidence assembly (stage2 + PDF/OSF → stage3 JSON+MD)
  4 - HumanStudy-Bench package draft (stage3 → metadata/spec/ground_truth/materials)
  5 - HumanStudy-Bench simulation run
"""

__all__ = ["STAGES"]

STAGES = {
    1: "Inventory (study coverage + simulation eligibility)",
    2: "Extraction (study-level findings/effects/statistics)",
    3: "Source/material evidence assembly (study-level fields)",
    4: "HumanStudy-Bench study package draft",
    5: "HumanStudy-Bench simulation",
}
