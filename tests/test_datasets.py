"""Preset checks on tiny fixtures: header and header-less files, missing tokens, binning, splits."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from laya_impute.datasets import PRESETS, Preset, make_schema, prepare, read_raw, read_table  # noqa: E402


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def test_headerless_file_with_question_mark_gaps(tmp_path):
    p = Preset(name="t", train_file="t/a.data", columns=["c", "x", "y"], targets=["x"])
    _write(str(tmp_path / "t" / "a.data"), "a,?,p\nb,q,?\n")
    df = read_raw(p, str(tmp_path / "t" / "a.data"))
    assert list(df.columns) == ["c", "x", "y"]
    assert pd.isna(df.loc[0, "x"]) and pd.isna(df.loc[1, "y"])


def test_header_file_with_custom_missing_token(tmp_path):
    p = Preset(name="t", train_file="t/b.csv", columns=["job", "age"], targets=["job"],
               sep=";", header=True, missing=("unknown",), bins={"age": ([0, 30, 200], ["young", "old"])})
    _write(str(tmp_path / "t" / "b.csv"), '"job";"age"\n"admin.";25\n"unknown";41\n')
    df = read_raw(p, str(tmp_path / "t" / "b.csv"))
    assert pd.isna(df.loc[1, "job"]) and df.loc[0, "job"] == "admin."
    schema = make_schema(p)
    cat = schema.discretise(df)
    assert list(cat["age"]) == ["young", "old"]


def test_prepare_splits_when_no_test_file(tmp_path):
    rows = "\n".join("%s,%s" % ("a" if i % 2 else "b", "x" if i % 3 else "y") for i in range(50))
    PRESETS["_tiny"] = Preset(name="_tiny", train_file="tiny/t.data", columns=["k", "v"], targets=["v"])
    try:
        _write(str(tmp_path / "tiny" / "t.data"), rows + "\n")
        paths = prepare("_tiny", str(tmp_path), str(tmp_path / "out"), test_fraction=0.2, seed=0)
        assert len(read_table(paths["train"])) == 40
        assert len(read_table(paths["test"])) == 10
    finally:
        del PRESETS["_tiny"]


def test_every_preset_declares_targets_among_its_columns():
    for name, preset in PRESETS.items():
        cols = [c for c in (preset.columns or []) if c not in preset.drop]
        assert set(preset.targets) <= set(cols), name
        assert preset.source.startswith("https://"), name
