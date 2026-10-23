"""Fine-tune a Laya checkpoint on masked categorical records.

The vocabulary is fitted on the training rows only, written beside the checkpoint as
`schema.json`, and every training and evaluation item is built from it. Laya's own trainer does
the optimisation, calibration and checkpoint writing; this module only prepares the data.
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict

from laya.train import TrainConfig, finetune

from .data import load_adult, training_cases, write_jsonl
from .schema import Schema, adult_schema


def prepare(data_path: str, test_path: str, work_dir: str, *, views_per_row: int, seed: int,
            max_train_rows: int | None, max_eval_items: int = 1000, schema_path: str | None = None):
    """Write train/eval JSONL under `work_dir` and return (schema, train_path, eval_path, counts).

    The eval set is a random sample of `max_eval_items` held-out items: Laya scores it before and
    after training, so its size is what a run pays for twice.
    """
    os.makedirs(work_dir, exist_ok=True)
    train_df = load_adult(data_path)
    test_df = load_adult(test_path) if test_path else None
    if max_train_rows:
        train_df = train_df.sample(n=min(max_train_rows, len(train_df)), random_state=seed)

    if schema_path:
        # Continuing from a checkpoint: its vocabulary must not change, or its options stop matching.
        schema = Schema.from_json(schema_path)
    else:
        schema = adult_schema().fit(train_df)
    train_cat = schema.discretise(train_df)
    train_cases = training_cases(train_cat, schema, views_per_row=views_per_row, seed=seed)

    train_path = os.path.join(work_dir, "train.jsonl")
    n_train = write_jsonl(train_cases, train_path)

    eval_path = None
    n_eval = 0
    if test_df is not None:
        # The test file is disjoint from the training file, so these items are held out.
        test_cat = schema.discretise(test_df)
        eval_cases = training_cases(test_cat, schema, views_per_row=1, seed=seed + 1)
        if len(eval_cases) > max_eval_items:
            eval_cases = random.Random(seed).sample(eval_cases, max_eval_items)
        eval_path = os.path.join(work_dir, "eval.jsonl")
        n_eval = write_jsonl(eval_cases, eval_path)
    return schema, train_path, eval_path, {"train_items": n_train, "eval_items": n_eval}


def resolve_base(base: str) -> str:
    """A local directory, or the cached snapshot of a Hub repo id.

    `laya.train.finetune` asks the Hub about a repo id on every call, which hangs on a machine
    without a working connection; the snapshot already on disk is used directly instead.
    """
    if os.path.isdir(base):
        return base
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(base, local_files_only=True)
    except Exception:
        pass
    # A cache written without a `refs/main` pointer: take the snapshot directory that exists.
    org, _, name = base.partition("/")
    root = os.path.join(os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")),
                        "hub", "models--%s--%s" % (org, name), "snapshots")
    if os.path.isdir(root):
        snaps = sorted(os.listdir(root), key=lambda d: os.path.getmtime(os.path.join(root, d)))
        if snaps:
            return os.path.join(root, snaps[-1])
    return base  # not cached: laya downloads it


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-csv", required=True,
                        help="UCI Adult adult.data (comma-separated, no header, ? for missing)")
    parser.add_argument("--test-csv", required=True, help="UCI Adult adult.test (held out)")
    parser.add_argument("--base", default="convaiinnovations/laya")
    parser.add_argument("--out", required=True, help="output checkpoint directory")
    parser.add_argument("--work-dir", default=None, help="where train/eval JSONL go (default: <out>/data)")
    parser.add_argument("--views-per-row", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512,
                        help="must hold the state plus every option of the widest question; "
                             "smaller values silently drop the widest questions")
    parser.add_argument("--max-eval-items", type=int, default=300)
    parser.add_argument("--schema", default=None,
                        help="reuse this schema.json (e.g. the parent checkpoint's) instead of fitting one")
    parser.add_argument("--loss", default="rlcd", choices=["rlcd", "soft-ce"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    work_dir = args.work_dir or os.path.join(args.out, "data")
    t0 = time.time()
    schema, train_path, eval_path, counts = prepare(
        args.train_csv, args.test_csv, work_dir, views_per_row=args.views_per_row,
        seed=args.seed, max_train_rows=args.max_train_rows, max_eval_items=args.max_eval_items,
        schema_path=args.schema)
    print("prepared", counts, "vocab", {k: len(v) for k, v in schema.vocab.items()}, flush=True)

    config = TrainConfig(epochs=args.epochs, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                         max_len=args.max_len, loss=args.loss, seed=args.seed, eval_data=eval_path)
    summary = finetune(train_path, resolve_base(args.base), args.out, config=config, device=args.device)

    os.makedirs(args.out, exist_ok=True)
    schema.to_json(os.path.join(args.out, "schema.json"))
    with open(os.path.join(args.out, "train_run.json"), "w", encoding="utf-8") as handle:
        json.dump({"base": args.base, "config": asdict(config), "counts": counts,
                   "seconds": round(time.time() - t0, 1)}, handle, indent=2, default=str)
    print("done in %.1f min" % ((time.time() - t0) / 60), flush=True)
    return summary


if __name__ == "__main__":
    main()
