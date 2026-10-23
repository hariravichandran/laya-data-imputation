"""Dataset presets: how to read each raw categorical table and which columns to impute.

Adding a dataset means adding one entry to `PRESETS`: where the raw file is, its column names,
the tokens that mean "missing", which numeric columns to bin, and the imputation targets. `prepare`
turns any preset into a plain train/test CSV pair that the rest of the package reads.

Paths are relative to a data root (default `data/`, which holds one folder per dataset).
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import Schema

INF = float("inf")


@dataclass
class Preset:
    name: str
    train_file: str                       # relative to the data root
    columns: List[str]                    # names for the raw file (header-less files) or header rename
    targets: List[str]                    # columns imputed when missing
    sep: str = ","
    header: bool = False                  # True when the raw file has its own header row
    missing: Tuple[str, ...] = ("?",)     # tokens that mean "missing" in the raw file
    test_file: Optional[str] = None       # a held-out file; otherwise `prepare` splits train_file
    comment: Optional[str] = None         # a line prefix to skip (adult.test's `|1x3` line)
    bins: Dict[str, Tuple[list, list]] = field(default_factory=dict)
    zero_nonzero: List[str] = field(default_factory=list)
    drop: List[str] = field(default_factory=list)
    clean: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    note: str = ""
    source: str = ""


def _strip_trailing_dot(df: pd.DataFrame) -> pd.DataFrame:
    # adult.test labels its income with a trailing period (`<=50K.`); strip it so both files agree.
    df["income"] = df["income"].astype("string").str.rstrip(".")
    return df


ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "income"]

PRESETS: Dict[str, Preset] = {
    "adult": Preset(
        name="adult", train_file="adult/adult.data", test_file="adult/adult.test",
        columns=ADULT_COLUMNS, sep=",", missing=("?",), comment="|",
        targets=["workclass", "occupation", "native-country"],
        drop=["fnlwgt", "education-num"],
        bins={"age": ([0, 25, 35, 45, 55, 65, 200], ["under 25", "25-34", "35-44", "45-54", "55-64", "65+"]),
              "hours-per-week": ([0, 20, 35, 41, 50, 200], ["under 20", "20-34", "35-40", "41-49", "50+"])},
        zero_nonzero=["capital-gain", "capital-loss"], clean=_strip_trailing_dot,
        note="UCI Adult census income; the three columns with real gaps are the targets.",
        source="https://archive.ics.uci.edu/dataset/2/adult (CC BY 4.0)"),

    "mushroom": Preset(
        name="mushroom", train_file="datasets/mushroom/agaricus-lepiota.data",
        columns=["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                 "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
                 "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
                 "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type",
                 "spore-print-color", "population", "habitat"],
        targets=["stalk-root", "spore-print-color", "habitat"],
        note="Mushroom records; stalk-root has about 30% gaps. Values are single letters.",
        source="https://archive.ics.uci.edu/dataset/73/mushroom (CC BY 4.0)"),

    "votes": Preset(
        name="votes", train_file="datasets/votes/house-votes-84.data",
        columns=["party"] + [f"vote-{i:02d}" for i in range(1, 17)],
        targets=["vote-01", "vote-02", "vote-03", "vote-04"],
        note="1984 US House votes (yes/no/abstain); a few votes have many gaps.",
        source="https://archive.ics.uci.edu/dataset/105/congressional+voting+records (CC BY 4.0)"),

    "breast-cancer": Preset(
        name="breast-cancer", train_file="datasets/breast-cancer/breast-cancer.data",
        columns=["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig",
                 "breast", "breast-quad", "irradiat"],
        targets=["node-caps", "breast-quad"],
        note="Ljubljana breast cancer recurrence; node-caps and breast-quad have gaps.",
        source="https://archive.ics.uci.edu/dataset/14/breast+cancer (CC BY 4.0)"),

    "bank": Preset(
        name="bank", train_file="datasets/bank/bank-full.csv", sep=";", header=True,
        columns=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
                 "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"],
        missing=("unknown",),
        targets=["job", "education", "contact"],
        bins={"age": ([0, 30, 40, 50, 60, 200], ["under 30", "30-39", "40-49", "50-59", "60+"]),
              "balance": ([-INF, 0, 500, 2000, 10000, INF], ["negative", "0-499", "500-1999", "2000-9999", "10000+"]),
              "day": ([0, 8, 16, 24, 32], ["1-7", "8-15", "16-23", "24-31"]),
              "duration": ([0, 100, 200, 400, 800, INF], ["under 100s", "100-199s", "200-399s", "400-799s", "800s+"]),
              "campaign": ([0, 2, 4, 8, INF], ["1", "2-3", "4-7", "8+"]),
              "pdays": ([-INF, 0, 30, 90, 400, INF], ["never contacted", "under 30 days", "30-89 days", "90-399 days", "400+ days"]),
              "previous": ([0, 1, 3, INF], ["0", "1-2", "3+"])},
        note="Portuguese bank marketing; 'unknown' is treated as missing.",
        source="https://archive.ics.uci.edu/dataset/222/bank+marketing (CC BY 4.0)"),

    "car": Preset(
        name="car", train_file="datasets/car/car.data",
        columns=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
        targets=["buying", "maint", "safety"],
        missing=(),
        note="Car evaluation; complete data, so every test cell is masked by the protocol.",
        source="https://archive.ics.uci.edu/dataset/19/car+evaluation (CC BY 4.0)"),
}


def read_raw(preset: Preset, path: str) -> pd.DataFrame:
    """Read one raw file with the preset's names, its missing tokens as NaN, and its cleaning."""
    kwargs = dict(sep=preset.sep, skipinitialspace=True, comment=preset.comment)
    # Header files get the preset's names too, so a header only has to be skipped, not trusted.
    df = pd.read_csv(path, names=preset.columns, header=0 if preset.header else None, **kwargs)
    df = df.replace({tok: np.nan for tok in preset.missing})
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("string"):
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    if preset.clean is not None:
        df = preset.clean(df)
    return df


def make_schema(preset: Preset) -> Schema:
    cols = [c for c in (preset.columns or []) if c not in preset.drop]
    return Schema(columns=cols, targets=list(preset.targets), bins=dict(preset.bins),
                  zero_nonzero=list(preset.zero_nonzero))


def prepare(name: str, root: str, out_dir: str, *, test_fraction: float = 0.2, seed: int = 0) -> Dict[str, str]:
    """Write `train.csv` (the training pool) and `test.csv` (held out) for a preset.

    If the preset names its own test file, that file is the held-out set; otherwise the train file is
    split at random with `test_fraction`. Missing cells are written as empty fields. Raw columns in
    `drop` are removed; the header is the preset's column names.
    """
    preset = PRESETS[name]
    train = read_raw(preset, os.path.join(root, preset.train_file))
    if preset.test_file:
        test = read_raw(preset, os.path.join(root, preset.test_file))
    else:
        test = train.sample(frac=test_fraction, random_state=seed)
        train = train.drop(test.index)
    keep = [c for c in train.columns if c not in preset.drop]
    os.makedirs(out_dir, exist_ok=True)
    paths = {"train": os.path.join(out_dir, "train.csv"), "test": os.path.join(out_dir, "test.csv")}
    train[keep].to_csv(paths["train"], index=False)
    test[keep].to_csv(paths["test"], index=False)
    return paths


def read_table(path: str) -> pd.DataFrame:
    """Read a prepared CSV: an empty field is a missing cell, and nothing else is special."""
    return pd.read_csv(path, na_values=[""], keep_default_na=False)
