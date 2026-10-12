from __future__ import annotations


def priority_group(probability: float) -> str:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"
