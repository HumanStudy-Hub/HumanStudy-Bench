"""Effect A+B->C preference-training utilities."""

from .datasets import (
    build_a_rows,
    build_b_control_rows,
    build_b_rows,
    build_c_rows,
)

__all__ = [
    "build_a_rows",
    "build_b_control_rows",
    "build_b_rows",
    "build_c_rows",
]
