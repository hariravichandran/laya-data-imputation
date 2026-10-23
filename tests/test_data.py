"""Data-path checks that need no model: run with `python -m pytest tests/`."""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from laya_impute.data import load_adult, questions_for, state_text, training_cases, write_jsonl  # noqa: E402
from laya_impute.schema import Schema, adult_schema  # noqa: E402

ROWS = [
    "39, Private, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K",
    "50, ?, 83311, Bachelors, 13, Married-civ-spouse, ?, Husband, White, Male, 0, 0, 13, Mexico, <=50K",
    "38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K",
    "53, Self-emp-not-inc, 234721, 11th, 7, Married-civ-spouse, Farming-fishing, Husband, Black, Male, 0, 0, 40, Peru, <=50K",
    "28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Cuba, <=50K",
    "37, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Female, 0, 0, 40, United-States, >50K",
    "49, Private, 160187, 9th, 5, Married-spouse-absent, Other-service, Not-in-family, Black, Female, 0, 0, 16, Jamaica, <=50K",
    "52, Self-emp-not-inc, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 45, United-States, >50K",
]


def _frame(tmp_path):
    path = os.path.join(str(tmp_path), "adult.data")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(ROWS) + "\n")
    return load_adult(path)


def test_load_marks_question_marks_as_missing(tmp_path):
    df = _frame(tmp_path)
    assert len(df) == len(ROWS)
    assert df["workclass"].isna().sum() == 1
    assert df["occupation"].isna().sum() == 1
    assert df.loc[0, "workclass"] == "Private"


def test_discretise_bins_numeric_and_keeps_missing(tmp_path):
    df = _frame(tmp_path)
    schema = adult_schema()
    cat = schema.discretise(df)
    assert cat.loc[0, "age"] == "35-44"          # 39 -> [35, 45)
    assert cat.loc[0, "hours-per-week"] == "35-40"  # 40 -> [35, 41)
    assert cat.loc[0, "capital-gain"] == "some"
    assert cat.loc[1, "capital-gain"] == "none"
    assert cat["workclass"].isna().iloc[1]
    assert "fnlwgt" not in cat.columns and "education-num" not in cat.columns


def test_vocab_pools_rare_values_into_other(tmp_path):
    df = _frame(tmp_path)
    schema = adult_schema()
    schema.min_freq = 0.2  # with 8 rows, values under 20% of observed cells become Other
    schema.fit(df)
    assert schema.vocab["native-country"][-1] == "Other"
    assert "United-States" in schema.vocab["native-country"]
    assert schema.map_target("native-country", "Peru") == "Other"
    assert schema.map_target("native-country", None) is None


def test_schema_round_trips_through_json(tmp_path):
    schema = adult_schema().fit(_frame(tmp_path))
    path = os.path.join(str(tmp_path), "schema.json")
    schema.to_json(path)
    again = Schema.from_json(path)
    assert again.vocab == schema.vocab
    assert again.bins["age"] == schema.bins["age"]
    assert again.targets == schema.targets


def test_state_text_omits_hidden_and_missing_cells():
    cell = {"age": "25-34", "workclass": None, "sex": "Male", "occupation": "Sales"}
    text = state_text(cell, ["age", "workclass", "sex", "occupation"], hidden=["occupation"])
    assert text == "Census record:\nage: 25-34\nsex: Male"


def test_questions_use_fitted_vocabulary(tmp_path):
    schema = adult_schema().fit(_frame(tmp_path))
    qs = questions_for(schema, ["workclass"])
    assert set(qs) == {"workclass"}
    assert qs["workclass"]["type"] == "choice"
    assert list(qs["workclass"]["criteria"]) == schema.vocab["workclass"]


def test_training_cases_mask_targets_and_expect_their_values(tmp_path):
    df = _frame(tmp_path)
    schema = adult_schema().fit(df)
    cat = schema.discretise(df)
    cases = training_cases(cat, schema, views_per_row=3, seed=1)
    assert cases, "expected masked training items"
    for case in cases:
        asked = set(case["questions"])
        assert asked and asked == set(case["expected"])
        assert asked <= set(schema.targets)
        for col in asked:
            # A masked column must not leak into the state, and its answer must be in the vocabulary.
            assert ("\n%s: " % col) not in case["state"]
            assert case["expected"][col] in schema.vocab[col]
            assert list(case["questions"][col]["criteria"]) == schema.vocab[col]


def test_write_jsonl_is_one_case_per_line(tmp_path):
    df = _frame(tmp_path)
    schema = adult_schema().fit(df)
    cases = training_cases(schema.discretise(df), schema, views_per_row=1)
    path = os.path.join(str(tmp_path), "train.jsonl")
    assert write_jsonl(cases, path) == len(cases)
    with open(path, encoding="utf-8") as handle:
        parsed = [json.loads(line) for line in handle]
    assert parsed == json.loads(json.dumps(cases))


def test_test_file_header_and_trailing_period_are_handled(tmp_path):
    # adult.test has a `|1x3 Cross validator` first line and labels like `<=50K.`; both must match
    # the training file, or every held-out record carries a category the model never saw.
    path = os.path.join(str(tmp_path), "adult.test")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("|1x3 Cross validator\n" + ROWS[0].replace("<=50K", "<=50K.") + "\n")
    df = load_adult(path)
    assert len(df) == 1
    assert df.loc[0, "income"] == "<=50K"
