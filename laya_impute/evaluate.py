"""Held-out evaluation of categorical imputation: Laya against two baselines.

Protocol. Held-out complete rows are the ground truth. For each target column we hide that cell
alone (`single`: the other cells are observed), and we also hide all targets at once (`all`: the
record has to be imputed from its context alone). Each method reports accuracy on every answer
and a coverage/accuracy curve: a cell is filled when the method's confidence clears a threshold,
and abstentions count as misses in the "all answers" accuracy.

Baselines. `majority` predicts the training mode of the column. `naive_bayes` is a categorical
Naive Bayes over the observed cells (`single`) or the context columns (`all`), trained on the
training rows. Both use the same abstention rule, so the curves compare like with like.
"""

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .datasets import PRESETS, make_schema, read_table
from .impute import Imputer
from .schema import Schema

THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9]


def _truth(schema: Schema, cat: pd.DataFrame, col: str) -> pd.Series:
    return cat[col].map(lambda v: schema.map_target(col, v))


class _CategoricalNB:
    """Categorical Naive Bayes with Laplace smoothing; unseen values share one extra code."""

    def __init__(self, features: List[str], schema: Schema, target: str, train: pd.DataFrame):
        self.features = features
        self.encoders = {}
        for f in features:
            values = sorted(v for v in train[f].dropna().unique())
            self.encoders[f] = {v: i for i, v in enumerate(values)}
        self.classes = list(schema.vocab[target])
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        n_cls = len(self.classes)
        y = train[target].map(self.class_index).to_numpy()
        self.log_prior = np.log(np.bincount(y, minlength=n_cls) + 1.0) - math.log(len(y) + n_cls)
        self.log_cond = {}
        for f in features:
            k = len(self.encoders[f]) + 1
            table = np.ones((n_cls, k))           # Laplace smoothing
            codes = self._encode(train[f], f)
            np.add.at(table, (y, codes), 1.0)
            self.log_cond[f] = np.log(table / table.sum(axis=1, keepdims=True))

    def _encode(self, series: pd.Series, f: str) -> np.ndarray:
        enc = self.encoders[f]
        unk = len(enc)
        return np.array([enc.get(v, unk) if isinstance(v, str) else unk for v in series], dtype=int)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        score = np.tile(self.log_prior, (len(frame), 1))
        for f in self.features:
            score += self.log_cond[f][:, self._encode(frame[f], f)].T
        score -= score.max(axis=1, keepdims=True)
        p = np.exp(score)
        return p / p.sum(axis=1, keepdims=True)


def _metrics(records: List[Dict]) -> Dict:
    """`records` hold truth, pred (argmax label), conf. Return accuracy and the threshold curve."""
    n = len(records)
    out = {"n": n, "accuracy": None, "curve": []}
    if n == 0:
        return out
    out["accuracy"] = round(sum(r["pred"] == r["truth"] for r in records) / n, 4)
    for t in THRESHOLDS:
        kept = [r for r in records if r["conf"] >= t and r["pred"] != r["other"]]
        acc = (sum(r["pred"] == r["truth"] for r in kept) / len(kept)) if kept else None
        out["curve"].append({"threshold": t, "coverage": round(len(kept) / n, 4),
                             "accuracy_on_filled": None if acc is None else round(acc, 4)})
    return out


def evaluate(checkpoint_dir: Optional[str], train_csv: str, test_csv: str, n_rows: int = 1000, seed: int = 0,
             device: Optional[str] = None, out_path: Optional[str] = None,
             max_train_rows: Optional[int] = None, dataset: str = "adult") -> Dict:
    """Score Laya (`checkpoint_dir`) and the baselines on held-out rows.

    With `checkpoint_dir=None` only the baselines run, using a vocabulary fitted on the training
    rows, so the baseline numbers can be recorded without a checkpoint.
    `max_train_rows` / `seed` must match the training run, so the baselines see the same rows
    Laya was trained on.
    """
    train_raw = read_table(train_csv)
    test_raw = read_table(test_csv)
    if max_train_rows:
        train_raw = train_raw.sample(n=min(max_train_rows, len(train_raw)), random_state=seed)
    if checkpoint_dir:
        schema = Schema.from_json(os.path.join(checkpoint_dir, "schema.json"))
    else:
        schema = make_schema(PRESETS[dataset]).fit(train_raw)

    # Baseline training data: complete rows of the training file, mapped through the vocabulary.
    train_cat = schema.discretise(train_raw)
    keep = train_cat[schema.targets].notna().all(axis=1)
    train = train_cat[keep].copy()
    for col in schema.targets:
        train[col] = train[col].map(lambda v, c=col: schema.map_target(c, v))
    train = train.dropna(subset=schema.targets)
    context = [c for c in schema.columns if c not in schema.targets]

    test_cat = schema.discretise(test_raw)
    test_complete = test_cat[test_cat[schema.targets].notna().all(axis=1)]
    sample = test_complete.sample(n=min(n_rows, len(test_complete)), random_state=seed)
    sample_raw = test_raw.loc[sample.index]

    # Adult's categorical profiles repeat, so a held-out record can be an exact copy of a training
    # record. Those copies are reported separately: the "novel" subset excludes them.
    def _key(frame):
        return frame[schema.columns].astype(str).agg("|".join, axis=1)
    seen = set(_key(train[schema.columns]))
    novel = ~_key(sample).isin(seen).to_numpy()
    results_overlap = {"sample_rows": int(len(sample)), "novel_rows": int(novel.sum())}

    agent = Imputer(checkpoint_dir, min_confidence=0.0, device=device) if checkpoint_dir else None
    results: Dict = {"checkpoint": checkpoint_dir, "n_rows": int(len(sample)), "seed": seed,
                     "overlap": results_overlap,
                     "vocab": schema.vocab, "protocols": {}}

    for protocol in ("single", "all"):
        per_target = {}
        # "all" hides every target at once, so one prediction pass answers every column.
        shared_preds = None
        if agent is not None and protocol == "all":
            all_raw = sample_raw.copy()
            for t in schema.targets:
                all_raw[t] = np.nan
            shared_preds = agent.predict(all_raw)
        for col in schema.targets:
            truth = _truth(schema, sample, col)
            if protocol == "single":
                masked_raw = sample_raw.copy()
                masked_raw[col] = np.nan
                features = [c for c in schema.columns if c != col]
                preds = agent.predict(masked_raw) if agent is not None else []
            else:
                features = context
                preds = shared_preds or []
            other = schema.other_label
            nb = _CategoricalNB(features, schema, col, train)
            proba = nb.predict_proba(sample[features])
            nb_pred = proba.argmax(axis=1)
            truth_list = [truth.iloc[i] for i in range(len(sample))]

            method_records = {}
            if agent is not None:
                by_pos = {p.row: p for p in preds if p.column == col}
                method_records["laya"] = [{"truth": truth_list[i], "pred": by_pos[i].choice,
                                           "conf": by_pos[i].confidence, "other": other}
                                          for i in range(len(sample))]
            method_records["naive_bayes"] = [{"truth": truth_list[i], "pred": nb.classes[nb_pred[i]],
                                              "conf": float(proba[i].max()), "other": other}
                                             for i in range(len(sample))]
            maj = train[col].value_counts().idxmax()
            method_records["majority"] = [{"truth": truth_list[i], "pred": maj, "conf": 1.0, "other": other}
                                          for i in range(len(sample))]
            print("evaluated %s / %s" % (protocol, col), flush=True)
            per_target[col] = {}
            for name, records in method_records.items():
                per_target[col][name] = {
                    "all": _metrics(records),
                    "novel": _metrics([r for r, keep in zip(records, novel) if keep]),
                }
        results["protocols"][protocol] = per_target

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return results


def format_table(results: Dict) -> str:
    """Markdown tables: accuracy on every held-out answer and on the novel subset, plus the curve."""
    lines = []
    overlap = results.get("overlap", {})
    for protocol, targets in results["protocols"].items():
        lines.append("protocol=%s (n=%d, novel=%s)" % (protocol, results["n_rows"], overlap.get("novel_rows")))
        lines.append("| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for col, methods in targets.items():
            for name in ("laya", "naive_bayes", "majority"):
                if name not in methods:
                    continue
                m = methods[name]["all"]
                novel = methods[name]["novel"]
                c = {round(x["threshold"], 2): x for x in m["curve"]}
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    col, name, _fmt(m["accuracy"]), _fmt(novel["accuracy"]),
                    _fmt(c[0.5]["coverage"]), _fmt(c[0.5]["accuracy_on_filled"]),
                    _fmt(c[0.8]["coverage"]), _fmt(c[0.8]["accuracy_on_filled"])))
        lines.append("")
    return "\n".join(lines)


def _fmt(x) -> str:
    return "-" if x is None else "%.3f" % x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Laya imputation against baselines")
    parser.add_argument("--checkpoint", default=None, help="omit to score the baselines only")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--n-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dataset", default="adult", help="preset name, used only without --checkpoint")
    parser.add_argument("--max-train-rows", type=int, default=None,
                        help="the --max-train-rows used for training (default: all rows)")
    a = parser.parse_args()
    res = evaluate(a.checkpoint, a.train_csv, a.test_csv, a.n_rows, a.seed, a.device, a.out,
                   max_train_rows=a.max_train_rows, dataset=a.dataset)
    print(format_table(res))
