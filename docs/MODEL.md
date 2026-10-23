# Model: laya-data-impute (v2)

The checkpoint itself is not stored in git. It is published on the Hugging Face Hub, and this page
documents what it is, how it was made and how it was evaluated.

**Hub repository:** [hravi/laya-data-impute](https://huggingface.co/hravi/laya-data-impute)
(private at the time of writing; the owner decides when to make it public).

## What it is

A Laya 0.4.13 checkpoint, fine-tuned to answer "which value does this categorical cell take?". The
approach is generic: any categorical table with a preset (see `laya_impute/datasets.py`) can be imputed
the same way. These weights, however, were trained and evaluated on UCI Adult only, on its three
columns with real gaps: `workclass`, `occupation` and `native-country`. Other tables need their own
schema and fine-tune before the checkpoint is used on them.
Every other categorical column and the binned numeric columns are context. The checkpoint folder
contains the weights, the tokenizer and encoder config, `schema.json` (columns, bins and each
target's vocabulary) and `questions.json`.

## How it was made

| step | run record |
|---|---|
| v1: fresh fine-tune of `convaiinnovations/laya`, 2,000 rows (seed 0), 3 epochs | [docs/runs/adult-impute-v1.train_run.json](runs/adult-impute-v1.train_run.json) |
| v2: continued from v1 on 2,000 new rows (seed 1), 2 epochs, same vocabulary | [docs/runs/adult-impute-v2.train_run.json](runs/adult-impute-v2.train_run.json) |

Commands and settings are in [docs/runs/README.md](runs/README.md). Training ran on CPU.

## How it performs

Held-out Adult test rows, one target masked, against majority and Naive Bayes baselines. Full tables,
including the all-targets-masked protocol and the threshold curves, are in
[docs/RESULTS.md](RESULTS.md).

Only Adult has been measured. In short, on Adult: Laya ties the majority baseline on workclass and beats it on native-country only marginally,
and it does not beat Naive Bayes on occupation overall. Its filled cells are more precise on
occupation and workclass, but it covers far fewer cells. It is calibrated, not more accurate.

## Intended use

- Filling gaps in categorical tables where a best guess plus an honest "don't know" is useful. Validate on
  your own table first; this checkpoint is validated on Adult only.
- Choosing a confidence threshold per column on your own validation data.

Not intended for decisions about individuals. Adult contains sex, race and native country as context;
imputed values reproduce the patterns in that census sample.

## Publishing

The checkpoint was uploaded with `scripts/publish_hf.py` from a staged folder that holds the model card.
Git keeps this document, the run records and the results. The Hub repository is created private by default
and made public only once you decide to share it.
