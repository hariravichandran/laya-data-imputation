"""Loading, serialising and masking categorical records for Laya.

A record is written as one line per observed cell (`column: value`). A question is one choice
per missing target cell, with the column's vocabulary as criteria, so the same question set can
be asked at training and at inference time.
"""

import random
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .schema import Schema

def load_adult(path: str) -> pd.DataFrame:
    """Read one UCI Adult file (`adult.data` or `adult.test`) through the Adult preset."""
    from .datasets import PRESETS, read_raw
    return read_raw(PRESETS["adult"], path)


def state_text(cat_row: Dict[str, Optional[str]], columns: Sequence[str], hidden: Iterable[str] = ()) -> str:
    """The Laya state: one `column: value` line per observed, non-hidden cell."""
    hidden = set(hidden)
    lines = ["Census record:"]
    for col in columns:
        if col in hidden:
            continue
        value = cat_row.get(col)
        if value is None or pd.isna(value):
            continue
        lines.append("%s: %s" % (col, value))
    return "\n".join(lines)


def question_for(schema: Schema, col: str) -> Dict:
    """The choice question for one target column, with its fitted vocabulary as criteria."""
    values = schema.vocab[col]
    return {
        "type": "choice",
        "instructions": "Which %s best fits this record?" % col,
        "criteria": {v: "%s = %s" % (col, v) for v in values},
    }


def questions_for(schema: Schema, cols: Sequence[str]) -> Dict[str, Dict]:
    return {col: question_for(schema, col) for col in cols}


def complete_rows(cat: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Rows where every target has an observed (mappable) value, so it can serve as ground truth."""
    mask = pd.Series(True, index=cat.index)
    for col in schema.targets:
        mask &= cat[col].notna()
    return cat[mask]


def training_cases(cat: pd.DataFrame, schema: Schema, views_per_row: int = 2,
                   max_masked: int = 3, seed: int = 0) -> List[Dict]:
    """Masked training items, one JSONL-ready dict per (row, view).

    Each view hides a random non-empty subset of the target columns (at most `max_masked`) and
    asks them together, so the model is trained on the same multi-question, one-pass shape it
    is used with at inference.
    """
    rng = random.Random(seed)
    cases = []
    targets = list(schema.targets)
    rows = complete_rows(cat, schema)
    for _, row in rows.iterrows():
        # Targets are read through the vocabulary too, so a rare value shows up as `Other` in the
        # context text exactly as it does when it is the answer.
        observed = {c: row[c] for c in schema.columns}
        for col in targets:
            observed[col] = schema.map_target(col, observed[col])
        for _ in range(views_per_row):
            k = rng.randint(1, min(max_masked, len(targets)))
            masked = sorted(rng.sample(targets, k))
            cell = {c: (None if c in masked else observed[c]) for c in schema.columns}
            cases.append({
                "state": state_text(cell, schema.columns, hidden=masked),
                "questions": questions_for(schema, masked),
                "expected": {col: observed[col] for col in masked},
            })
    rng.shuffle(cases)
    return cases


def write_jsonl(cases: Iterable[Dict], path: str) -> int:
    import json
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
            count += 1
    return count
