"""Abstention rules of the imputer, with the Laya agent stubbed out (no checkpoint needed)."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from laya_impute.impute import ABSTAIN_LOW_CONF, ABSTAIN_OTHER, FILLED, Imputer  # noqa: E402
from laya_impute.schema import adult_schema  # noqa: E402


class _StubAgent:
    """Returns a fixed answer per column for every state, in the shape `predict_batch` produces."""

    def __init__(self, answers):
        self.answers = answers

    def predict_batch(self, states, questions):
        out = []
        for _ in states:
            out.append({"answers": {col: self.answers[col] for col in questions}})
        return out


def _imputer(answers, min_confidence=0.5):
    imp = Imputer.__new__(Imputer)  # skip laya.load: only the decision code is under test
    imp.schema = adult_schema()
    imp.schema.vocab = {"workclass": ["Private", "Other"], "occupation": ["Sales", "Other"],
                        "native-country": ["United-States", "Other"]}
    imp.min_confidence = min_confidence
    imp.agent = _StubAgent(answers)
    imp.checkpoint_dir = "stub"
    imp.batch_size = 32
    return imp


def _answer(choice, conf, probs=None):
    return {"type": "choice", "choice": choice, "answer_confidence": conf,
            "probabilities": probs or {choice: conf}}


def _frame():
    return pd.DataFrame({
        "age": [30, 40], "workclass": [None, None], "occupation": ["Sales", None],
        "native-country": [None, "Mexico"], "education": ["Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Divorced"], "relationship": ["Own-child", "Unmarried"],
        "race": ["White", "White"], "sex": ["Male", "Female"], "capital-gain": [0, 0],
        "capital-loss": [0, 0], "hours-per-week": [40, 40], "income": ["<=50K", "<=50K"],
    })


def test_confident_specific_answer_fills_the_cell():
    imp = _imputer({"workclass": _answer("Private", 0.9), "occupation": _answer("Sales", 0.9),
                    "native-country": _answer("United-States", 0.9)})
    filled, audit = imp.impute(_frame())
    assert filled.loc[0, "workclass"] == "Private"
    assert filled.loc[0, "native-country"] == "United-States"
    assert filled.loc[1, "native-country"] == "Mexico"  # observed, so never asked
    assert set(audit["status"]) == {FILLED}


def test_low_confidence_leaves_the_cell_blank():
    imp = _imputer({"workclass": _answer("Private", 0.3), "occupation": _answer("Sales", 0.9),
                    "native-country": _answer("United-States", 0.9)})
    filled, audit = imp.impute(_frame())
    assert pd.isna(filled.loc[0, "workclass"])
    row = audit[(audit.row == 0) & (audit.column == "workclass")].iloc[0]
    assert row["status"] == ABSTAIN_LOW_CONF and pd.isna(row["value"])


def test_other_is_an_abstention_even_when_confident():
    imp = _imputer({"workclass": _answer("Other", 0.99), "occupation": _answer("Sales", 0.9),
                    "native-country": _answer("United-States", 0.9)})
    filled, audit = imp.impute(_frame())
    assert pd.isna(filled.loc[0, "workclass"])
    assert ABSTAIN_OTHER in set(audit["status"])


def test_observed_cells_are_never_overwritten():
    imp = _imputer({"workclass": _answer("Private", 0.9), "occupation": _answer("Sales", 0.9),
                    "native-country": _answer("United-States", 0.9)})
    df = _frame()
    filled, audit = imp.impute(df)
    assert filled.loc[0, "occupation"] == "Sales"
    assert list(audit[audit.row == 0]["column"]) == ["workclass", "native-country"]
    assert list(audit[audit.row == 1]["column"]) == ["workclass", "occupation"]
    assert df.loc[0, "occupation"] == "Sales" and pd.isna(df.loc[0, "workclass"])  # input untouched
