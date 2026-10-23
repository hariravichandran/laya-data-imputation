"""Categorical data imputation with a fine-tuned Laya checkpoint.

Each record becomes one Laya state (its observed categorical cells as text) and each missing
cell becomes one choice question over that column's vocabulary. All questions for a record are
answered in a single forward pass; a cell is filled only when Laya's calibrated confidence clears
a threshold, and left blank otherwise.
"""

from .schema import Schema, adult_schema

__all__ = ["Schema", "adult_schema"]
