# laya-data-imputation

Fine-tuned [Laya](https://github.com/NandhaKishorM/laya) for **categorical data imputation**.

Laya is a non-autoregressive decision engine: one forward pass answers typed questions about a
state, and every answer carries a calibrated confidence. This project turns that into a
tabular imputer. A record's observed categorical cells become the state, each missing cell becomes
a choice question over that column's vocabulary, and a cell is filled only when Laya is confident
enough. Otherwise the cell is left blank.

It is built on Laya 0.4.13 (Apache-2.0, by Nandha Kishor M). The fine-tuning loop, calibration and
checkpoint format are Laya's; the data path, the imputation rules and the evaluation are this
project's.

## What it does

- **Categorical only.** Numeric columns are binned into labelled bands (age, hours per week) or
  reduced to zero / nonzero (capital gain and loss). Laya never sees a raw number.
- **Targets are explicit.** Only the columns in `Schema.targets` are imputed. Everything else is
  context.
- **One pass per record.** All missing targets of a record are asked as separate questions over
  one state, answered together by `predict_batch`.
- **Calibrated abstention.** A cell is filled when Laya's `answer_confidence` clears
  `--min-confidence`. The catch-all `Other` category is treated as "no specific value" and is
  never written into the data.
- **Auditable.** `--audit` writes every decision: the value, its confidence, its status, and the
  full probability vector over the column's vocabulary.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -q tests/                 # no model needed

scripts/fetch_adult.sh data/adult          # UCI Adult, CC BY 4.0

# Fine-tune on CPU (about two hours at these settings; a GPU is much faster)
python -m laya_impute.train \
    --train-csv data/adult/adult.data --test-csv data/adult/adult.test \
    --out runs/adult-impute-v1 --max-train-rows 2000 --views-per-row 1 \
    --epochs 3 --micro-batch 8 --grad-accum 4 --device cpu

# Impute a CSV with the same column names
python -m laya_impute impute --checkpoint runs/adult-impute-v1 \
    --input data/adult/adult.data --output filled.csv --audit audit.csv --min-confidence 0.5

# Held-out evaluation against baselines
python -m laya_impute.evaluate --checkpoint runs/adult-impute-v1 \
    --train-csv data/adult/adult.data --test-csv data/adult/adult.test \
    --n-rows 1000 --max-train-rows 2000 --out docs/results/laya-adult-v1.json
```

Input CSVs may mark missing cells as blank or `?`. Column names must match the schema.

## Adding your own categorical table

`laya_impute.schema.Schema` describes a table: its columns, which are imputation targets, how
numeric columns are binned, and the rare-value threshold. Build one for your data, then call
`Schema.fit` on the training rows. `laya_impute.schema.adult_schema()` is the worked example.
Fit the vocabulary on training rows only, or the evaluation leaks.

## Design notes

- **Vocabulary.** Each target's vocabulary is the set of values that cover at least 0.3% of the
  observed training cells. Rarer values pool into `Other`, so the choice questions stay small.
- **Training items.** Each training record hides a random subset of its targets (one to three of
  them) and asks them together. Training matches inference, where the record's missing targets are
  asked together.
- **Training budget.** The item budget is set so that the fine-tune spends its optimizer updates
  on the questions that matter. `train.py` sets `max_len` to 512 by default. The state plus the
  options of the widest question must fit inside `max_len`. At 160, Laya dropped 1,833 training
  items as `options_beyond_max_len`, almost all of them occupation questions, so the run would
  have under-trained exactly the hardest column. The skip counts are printed at the start of a run;
  check them.
- **Held-out data.** The model trains on `adult.data`. Evaluation uses `adult.test`. Adult's
  categorical profiles repeat, so some test records exactly match a training record. Every metric
  is reported on all records and on a "novel" subset that excludes those copies.

## Results

See [`docs/results/`](docs/results/) for the raw JSON and [`docs/RESULTS.md`](docs/RESULTS.md) for
the summary table. Numbers are on held-out Adult test rows, with the target cell masked
artificially.

Held-out Adult test rows, one target masked, other cells observed (full tables in
[docs/RESULTS.md](docs/RESULTS.md)):

| target | Laya acc | Naive Bayes acc | majority acc | Laya acc@0.8 (coverage) | NB acc@0.8 (coverage) |
|---|---|---|---|---|---|
| workclass | 0.731 | 0.668 | 0.731 | 0.880 (0.34) | 0.830 (0.46) |
| occupation | 0.301 | 0.323 | 0.140 | 1.000 (0.01) | 0.346 (0.03) |
| native-country | 0.915 | 0.909 | 0.915 | 0.953 (0.93) | 0.954 (0.90) |

In short, Laya does not beat the baselines on raw accuracy. Its confidence is well calibrated, so the
cells it fills are more precise than Naive Bayes on the harder columns, at the cost of much lower
coverage. One seed and 1,000 rows, so treat the gaps as suggestive.

## Limits

- Imputation quality here is bounded by the data. Adult's `occupation` depends on information
  (industry, job title) that the other columns do not carry; no method recovers it reliably.
- The evaluation masks observed cells artificially. Real missingness need not be random, so
  accuracy on real gaps can differ.
- The run is CPU-only and single-seed. Laya's own docs note that takeoff from the chance plateau is
  seed-dependent, so treat one run as a point estimate.
- No model checkpoint is committed. Train one with the command above.

## Credits and license

Apache-2.0 (see [LICENSE](LICENSE)). Laya is by Nandha Kishor M and is also Apache-2.0
([upstream](https://github.com/NandhaKishorM/laya)). UCI Adult is CC BY 4.0 (Kohavi and Becker,
UCI Machine Learning Repository).
