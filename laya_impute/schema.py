"""Column schema: which columns are context, which are imputation targets, and how to discretise.

Laya reads categorical text, so every column is mapped to a small set of string values before
anything else happens. Numeric columns are only accepted through explicit bin edges or a
zero/nonzero rule; nothing is fed to the model as a raw number.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

@dataclass
class Schema:
    """What to impute and how to read each column.

    columns:     every column kept, in the order they are written into the record text
    targets:     the subset Laya is asked to fill when missing
    bins:        numeric column -> (edges, labels); values are cut into labelled bands
    zero_nonzero: numeric columns reduced to "none" / "some"
    min_freq:    categories rarer than this share of observed cells merge into "Other"
    vocab:       filled by `fit`; target column -> ordered list of allowed values
    """

    columns: List[str]
    targets: List[str]
    bins: Dict[str, tuple] = field(default_factory=dict)
    zero_nonzero: List[str] = field(default_factory=list)
    min_freq: float = 0.003
    other_label: str = "Other"
    vocab: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.targets) - set(self.columns)
        if unknown:
            raise ValueError("targets not in columns: %s" % sorted(unknown))

    # ------------------------------------------------------------------ discretisation
    def discretise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame of categorical strings (or NaN for missing), one column per kept column."""
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            raw = df[col]
            if col in self.bins:
                edges, labels = self.bins[col]
                vals = pd.to_numeric(raw, errors="coerce")
                cut = pd.cut(vals, bins=edges, labels=labels, right=False)
                out[col] = cut.astype(object).where(vals.notna(), None)
            elif col in self.zero_nonzero:
                vals = pd.to_numeric(raw, errors="coerce")
                out[col] = np.where(vals.isna(), None, np.where(vals > 0, "some", "none"))
            else:
                text = raw.astype("string").str.strip()
                text = text.mask(text.isin(["", "?"]))
                out[col] = text.astype(object).where(text.notna(), None)
        return out

    def fit(self, df: pd.DataFrame) -> "Schema":
        """Learn each target's vocabulary from observed cells of `df` (training data only)."""
        cat = self.discretise(df)
        for col in self.targets:
            counts = cat[col].dropna().value_counts()
            keep = counts[counts / counts.sum() >= self.min_freq]
            # Order by frequency so the most common value is option 0; rare values pool into Other.
            values = list(keep.index)
            if len(values) < len(counts):
                values.append(self.other_label)
            self.vocab[col] = values
        return self

    def map_target(self, col: str, value):
        """Map a discretised target value into the fitted vocabulary (rare -> Other, unknown -> None)."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        allowed = self.vocab[col]
        if value in allowed:
            return value
        return self.other_label if self.other_label in allowed else None

    # ------------------------------------------------------------------ persistence
    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Schema":
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        raw["bins"] = {k: (list(v[0]), list(v[1])) for k, v in raw.get("bins", {}).items()}
        return cls(**raw)


def adult_schema() -> Schema:
    """UCI Adult with numeric columns discretised and the three real-gap columns as targets."""
    return Schema(
        columns=["age", "workclass", "education", "marital-status", "occupation",
                 "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "income"],
        targets=["workclass", "occupation", "native-country"],
        bins={
            "age": ([0, 25, 35, 45, 55, 65, 200],
                    ["under 25", "25-34", "35-44", "45-54", "55-64", "65+"]),
            "hours-per-week": ([0, 20, 35, 41, 50, 200],
                               ["under 20", "20-34", "35-40", "41-49", "50+"]),
        },
        zero_nonzero=["capital-gain", "capital-loss"],
    )
