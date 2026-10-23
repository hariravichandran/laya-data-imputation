# Research methodology

This project asks one question: can a fine-tuned Laya model fill missing categorical cells better than
simple statistical baselines, and if so, when? This page records how the models were built, how they
were tested, and what the results can and cannot support. Commands are in [runs/README.md](runs/README.md);
results are in [RESULTS.md](RESULTS.md); the published model is described in [MODEL.md](MODEL.md).

## 1. Task

Given a record with some categorical cells missing, predict the value of each missing target cell.
Each record is a Laya *state*: one `column: value` line per observed cell. Each missing target is a
Laya *choice question* over that column's fitted vocabulary. All missing targets of a record are asked
together, in one forward pass. A cell is filled only when Laya's calibrated confidence reaches a
threshold chosen by the user; the catch-all `Other` value always abstains.

## 2. Data

Each dataset has a preset in `laya_impute/datasets.py` that records its raw file, column names, missing
tokens, numeric bins, imputation targets and source. The datasets were all free and downloaded from the
UCI Machine Learning Repository (CC BY 4.0):

| dataset | rows (train pool / held out) | targets | missing token |
|---|---|---|---|
| Adult | 32,561 / 16,281 (official test file) | workclass, occupation, native-country | `?` |
| Mushroom | 6,499 / 1,625 (random 80/20 split, seed 0) | stalk-root, spore-print-color, habitat | `?` |
| Congressional votes | 348 / 87 (random 80/20 split) | vote-01 to vote-04 | `?` |
| Breast cancer | 229 / 57 (random 80/20 split) | node-caps, breast-quad | `?` |
| Bank marketing | 36,169 / 9,042 (random 80/20 split) | job, education, contact | `unknown` |
| Car evaluation | 1,382 / 346 (random 80/20 split) | buying, maint, safety | none |

Numeric columns are never fed to the model as numbers. They are binned with explicit edges (age, hours
per week, balance, duration and so on) or reduced to zero/nonzero (capital gains and losses). Categorical
vocabularies keep values that cover at least 0.3% of observed training cells; rarer values pool into
`Other`. Vocabularies are fitted on the training pool only.

## 3. Splits and sampling

- **Held-out data.** Where a dataset ships a test file (Adult), that file is the held-out set. Otherwise
  the training pool is split at random (20% held out, seed 0) before anything is fitted. The held-out
  rows are never used for fitting vocabularies, training, calibration or baselines.
- **Training sample.** Each run trains on a random sample of the training pool (seed set per run, 1,000
  to 3,000 rows depending on the dataset budget, or all rows for the small ones). The sample size and seed
  are recorded in each run's `train_run.json`.
- **Training items.** Each sampled row must have every target observed (it is the ground truth). For each
  row, the trainer hides a random non-empty subset of its targets (at most three) and asks them together.
  `views_per_row` controls how many such masked copies each row contributes.

## 4. Training

- Base model: `convaiinnovations/laya` (Laya 0.4.13). Fine-tuned with Laya's own trainer: RLCD loss,
  encoder learning rate 2.5e-5, head learning rate 1e-4, gradient clipping 1.0, maximum length 512.
- Laya holds out a calibration slice (10% of items, at most 400) and fits its confidence temperatures
  on it. The calibration slice is not used for the held-out evaluation.
- Effective batch 32 (micro-batch 8 times gradient accumulation 4). Three epochs per benchmark run; two
  epochs for continued training.
- Continued training (Adult v2) starts from v1 and reuses v1's `schema.json`, so the vocabulary and the
  question options cannot drift between the two checkpoints.
- Training ran on CPU with 32 threads. No hyperparameter search was done; every run uses the same
  settings except the row budget and views per row, which are chosen per dataset to fit the time available.

**A budget mistake we caught.** At `max_len` 160, Laya silently skipped training items whose options
did not fit (`options_beyond_max_len`). On a 2,000-row Adult sample that dropped 1,833 of the items,
almost all occupation questions. The budget was raised to 512, and the skip counts are now checked in
every run's output.

## 5. Evaluation

Each held-out evaluation uses up to 500 complete held-out rows (rows whose targets are all observed).

- **Single-target protocol.** For each target in turn, that cell is masked and the other cells are
  observed. This measures imputation from the rest of the record.
- **All-targets protocol.** All targets are masked at once. This measures imputation from context alone.
- **Methods compared.** Laya, a *majority* baseline (the training mode of the column), and a
  *categorical Naive Bayes* baseline (Laplace smoothing α = 1; unseen values share one extra code). Naive
  Bayes conditions on all observed columns in the single-target protocol and on context columns only in
  the all-targets protocol. It is trained on the same training sample as Laya.
- **Metrics.** Accuracy on every answer (abstentions count as wrong). Coverage (the share of cells filled)
  and accuracy on the filled cells at confidence thresholds 0.5 and 0.8, plus a full curve in the JSON
  output. The same thresholds are applied to every method so the comparison is like-for-like.
- **Novel subset.** Adult's categorical profiles repeat, so some held-out records exactly match a training
  record. Metrics are reported on all records and on a *novel* subset that excludes exact copies.
  Novel-subset sizes in the results: Mushroom 500 of 500 and car 346 of 346 (no repeats), bank 497 of 500,
  votes 62 of 79 and breast cancer 49 of 57. The small datasets lose the most rows to repeats, so their
  novel-subset numbers rest on fewer records.

## 6. Decision rules

- **What gets published.** Only the best checkpoint is published (to the Hugging Face Hub, as a private
  repository until the owner makes it public). Other checkpoints stay local, and their run records are
  committed to git so they can be rebuilt.
- **What "best" means.** For Adult, the best checkpoint is the one with the stronger held-out results at
  the same thresholds, read together with its coverage. Adult v2 was published because it was the best
  Adult checkpoint, not because it beats the baselines: it does not beat Naive Bayes on occupation overall.
  Its model card says so.
- **What is reported regardless.** Every benchmark run is reported, including the ones that trail the
  baselines, so the results table is not a selection of wins.

## 7. Threats to validity

- **One seed per run.** Laya's own documentation notes that takeoff from the chance plateau depends on the
  seed. Differences of one or two points between checkpoints are within noise.
- **Small test sets.** Votes (79 rows) and breast cancer (57 rows) give wide confidence intervals; treat
  their numbers as indicative only.
- **Artificial masking.** Observed cells are masked at random. Real missingness need not be random (for
  example, bank's `unknown` values may depend on the customer's other attributes).
- **Unequal data budgets.** Naive Bayes is trained on the same sample as Laya, but the Adult v2 checkpoint
  saw two samples (about 4,000 rows). The Adult baselines were not retrained on the larger pool, so the
  comparison favours Laya on that one checkpoint.
- **Known evaluation bugs fixed during the project.** A test-file parser bug (header line and trailing
  periods) and a record-key bug on missing context cells. Both were fixed, and the Adult results were
  rerun or confirmed unaffected.

## 8. Reproducing

```bash
scripts/fetch_datasets.sh                       # UCI downloads into data/
scripts/run_benchmark.sh data data/runs         # trains and evaluates every preset
python -m laya_impute.evaluate --checkpoint <dir> --train-csv <dir>/prepared/train.csv \
    --test-csv <dir>/prepared/test.csv --n-rows 500 --max-train-rows <rows> --out docs/results/<name>.json
```

Checkpoints are rebuilt from the recorded run settings. The published checkpoint is on the Hugging Face Hub,
linked from [MODEL.md](MODEL.md).
