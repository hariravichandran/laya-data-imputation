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

scripts/fetch_datasets.sh                  # raw files for all six presets (see Datasets)

# Fine-tune on CPU (about two hours at these settings; a GPU is much faster)
python -m laya_impute.train --dataset adult --root data --out runs/adult-impute-v1 \
    --max-train-rows 2000 --views-per-row 1 \
    --epochs 3 --micro-batch 8 --grad-accum 4 --device cpu

# Impute a CSV with the same column names
# --input is a CSV with a header row using the schema's column names (see schema.json in the checkpoint)
python -m laya_impute impute --checkpoint runs/adult-impute-v2 \
    --input my_records.csv --output filled.csv --audit audit.csv --min-confidence 0.5

# Held-out evaluation against baselines. The trainer writes the prepared CSVs to <out>/data/prepared/.
python -m laya_impute.evaluate --checkpoint runs/adult-impute-v1 \
    --train-csv runs/adult-impute-v1/data/prepared/train.csv \
    --test-csv runs/adult-impute-v1/data/prepared/test.csv \
    --n-rows 1000 --max-train-rows 2000 --out docs/results/laya-adult-v1.json
```

## Datasets

Six presets are defined in `laya_impute/datasets.py`. Each one names its raw file, its columns, the
token that means "missing", and the columns to impute. All six come from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/) and are licensed CC BY 4.0.

`scripts/fetch_datasets.sh [data_root]` downloads the raw archives into `data/` (the default data root).
It skips any dataset whose files are already present, never overwrites a file, and exits non-zero if an
expected file is missing. `adult` goes to `data/adult/`; the other five go to `data/datasets/<name>/`.

| Preset | What it is | Source | License | Imputation targets | Missing token |
|---|---|---|---|---|---|
| `adult` | UCI Adult census income. Trains on `adult.data`, tests on `adult.test`. | [dataset/2](https://archive.ics.uci.edu/dataset/2/adult) | CC BY 4.0 | `workclass`, `occupation`, `native-country` | `?` |
| `mushroom` | Agaricus-lepiota mushroom records, one letter per value. | [dataset/73](https://archive.ics.uci.edu/dataset/73/mushroom) | CC BY 4.0 | `stalk-root`, `spore-print-color`, `habitat` | `?` |
| `votes` | 1984 US House votes (yes, no, or abstain). | [dataset/105](https://archive.ics.uci.edu/dataset/105/congressional+voting+records) | CC BY 4.0 | `vote-01` to `vote-04` | `?` |
| `breast-cancer` | Ljubljana breast cancer recurrence. | [dataset/14](https://archive.ics.uci.edu/dataset/14/breast+cancer) | CC BY 4.0 | `node-caps`, `breast-quad` | `?` |
| `bank` | Portuguese bank marketing. Semicolon-separated with a header row. | [dataset/222](https://archive.ics.uci.edu/dataset/222/bank+marketing) | CC BY 4.0 | `job`, `education`, `contact` | `unknown` |
| `car` | Car evaluation. Complete data, so no cell is missing in the raw file. | [dataset/19](https://archive.ics.uci.edu/dataset/19/car+evaluation) | CC BY 4.0 | `buying`, `maint`, `safety` | none |

Train and evaluate one preset:

```bash
scripts/fetch_datasets.sh
python -m laya_impute.train --dataset <name> --root data --out runs/<name>-v1 \
    --max-train-rows <rows> --views-per-row <views> --epochs 3 \
    --micro-batch 8 --grad-accum 4 --seed 0 --device cpu
python -m laya_impute.evaluate --checkpoint runs/<name>-v1 \
    --train-csv runs/<name>-v1/data/prepared/train.csv --test-csv runs/<name>-v1/data/prepared/test.csv \
    --n-rows 500 --max-train-rows <rows> --out docs/results/<name>-v1.json
```

To train and evaluate the five presets other than `adult`, each with its own training budget, run
`scripts/run_benchmark.sh`. It runs on CPU, one preset after another, and by default writes checkpoints
under `data/runs/` (pass a second argument to change that).

**Test sets.** `adult` has its own held-out file. The other five have only one file, so `prepare` holds
out a random 20% of the rows (seed 0) as the test set. For `votes` (435 rows) that is about 87 test rows,
and for `breast-cancer` (286 rows) about 57. Their evaluations are noisy: a single wrong answer moves
accuracy by about 1 point on `votes` and about 2 points on `breast-cancer`. Treat differences between Laya
and the baselines on those two as unreliable until they are checked over several seeds.

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

A continued run (v2, two more epochs on new rows) did not change this picture; see the v2 section of
[docs/RESULTS.md](docs/RESULTS.md). The two checkpoints trade coverage against precision differently,
so choose a threshold per column on a validation split.

In short, Laya does not beat the baselines on raw accuracy. Its confidence is well calibrated, so the
cells it fills are more precise than Naive Bayes on the harder columns, at the cost of much lower
coverage. One seed and 1,000 rows, so treat the gaps as suggestive.

## Limits

- Imputation quality here is bounded by the data. Adult's `occupation` depends on information
  (industry, job title) that the other columns do not carry; no method recovers it reliably.
- The evaluation masks observed cells artificially. Real missingness need not be random, so
  accuracy on real gaps can differ.
- Small presets give noisy numbers. `votes` and `breast-cancer` have about 87 and 57 test rows, so
  their per-column accuracies move by a point or more with one answer. Their results are not yet
  comparable to the Adult numbers above.
- The run is CPU-only and single-seed. Laya's own docs note that takeoff from the chance plateau is
  seed-dependent, so treat one run as a point estimate.
- Checkpoints are not committed to git. The Adult v2 checkpoint is published on the Hugging Face Hub
  (see [docs/MODEL.md](docs/MODEL.md)); other checkpoints are rebuilt from the run records in
  [docs/runs/](docs/runs/).

## Credits and license

Apache-2.0 (see [LICENSE](LICENSE)). Laya is by Nandha Kishor M and is also Apache-2.0
([upstream](https://github.com/NandhaKishorM/laya)). UCI Adult is CC BY 4.0 (Kohavi and Becker,
UCI Machine Learning Repository).
