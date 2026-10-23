"""Impute missing categorical cells with a fine-tuned Laya checkpoint.

Each record is answered in one `predict_batch` pass: its observed cells form the state and every
missing target becomes a choice question. A cell is filled only when Laya's calibrated
`answer_confidence` reaches `min_confidence` and the answer is a specific value. The catch-all
`Other` is an abstention, not a fill: it says the record does not point to one of the named
values, so the cell stays blank.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import laya

from .data import questions_for, state_text
from .schema import Schema

ABSTAIN_OTHER = "other-catch-all"
ABSTAIN_LOW_CONF = "low-confidence"
FILLED = "filled"


@dataclass
class Prediction:
    row: int
    column: str
    value: Optional[str]          # the filled value, or None when the cell is left blank
    confidence: float             # Laya's calibrated answer_confidence
    status: str
    probabilities: Dict[str, float]
    choice: str                   # Laya's argmax before any abstention rule


class Imputer:
    def __init__(self, checkpoint_dir: str, min_confidence: float = 0.5, device: Optional[str] = None,
                 batch_size: int = 32):
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.schema = Schema.from_json(os.path.join(checkpoint_dir, "schema.json"))
        self.min_confidence = min_confidence
        self.agent = laya.load(checkpoint_dir, device=device)

    def predict(self, df: pd.DataFrame) -> List[Prediction]:
        """Answer every missing target cell in `df`. Returns one Prediction per missing cell."""
        cat = self.schema.discretise(df)
        groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        for pos in range(len(cat)):
            missing = tuple(c for c in self.schema.targets if cat[c].iloc[pos] is None
                            or pd.isna(cat[c].iloc[pos]))
            if missing:
                groups[missing].append(pos)

        out: List[Prediction] = []
        for missing, positions in groups.items():
            questions = questions_for(self.schema, missing)
            states = []
            for pos in positions:
                row = {c: cat[c].iloc[pos] for c in self.schema.columns}
                for col in self.schema.targets:
                    row[col] = self.schema.map_target(col, row[col])
                states.append(state_text(row, self.schema.columns, hidden=missing))
            # Chunked: one padded batch over a whole table exhausts memory on CPU.
            for start in range(0, len(states), self.batch_size):
                chunk = slice(start, start + self.batch_size)
                results = self.agent.predict_batch(states[chunk], questions)
                for pos, result in zip(positions[chunk], results):
                    for col in missing:
                        answer = result["answers"][col]
                        out.append(self._decide(pos, col, answer))
        return out

    def _decide(self, pos: int, col: str, answer: Dict) -> Prediction:
        probs = answer["probabilities"]
        value = answer["choice"]
        conf = float(answer["answer_confidence"])
        if value == self.schema.other_label:
            status, value = ABSTAIN_OTHER, None
        elif conf < self.min_confidence:
            status, value = ABSTAIN_LOW_CONF, None
        else:
            status = FILLED
        return Prediction(pos, col, value, conf, status, probs, choice=answer["choice"])

    def impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (filled copy of `df`, audit frame with one row per missing target cell)."""
        preds = self.predict(df)
        filled = df.copy()
        audit = []
        for p in preds:
            if p.status == FILLED:
                filled.iat[p.row, filled.columns.get_loc(p.column)] = p.value
            audit.append({"row": p.row, "column": p.column, "value": p.value, "confidence": p.confidence,
                          "status": p.status, "probabilities": json.dumps(p.probabilities)})
        return filled, pd.DataFrame(audit)
