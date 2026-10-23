# Training runs

Run records for the checkpoints evaluated in [../RESULTS.md](../RESULTS.md). Checkpoints are not
committed; each record gives the exact command and settings needed to rebuild one.

## adult-impute-v1 (fresh fine-tune of `convaiinnovations/laya`)

- **Command**
  ```bash
  python -m laya_impute.train --train-csv data/adult/adult.data --test-csv data/adult/adult.test \
      --out runs/adult-impute-v1 --max-train-rows 2000 --views-per-row 1 \
      --epochs 3 --micro-batch 8 --grad-accum 4 --seed 0 --device cpu
  ```
- **Data.** 2,000 sampled `adult.data` rows (seed 0) → 3,367 training items after the calibration
  slice (374 items). 318 optimizer updates at effective batch 32. `max_len` 512.
- **Schema.** `schema.json` (kept beside the checkpoint). Vocabulary sizes: workclass 6,
  occupation 14, native-country 11.
- **Time.** 94 minutes on 32 CPU cores.
- **Loss.** Per-epoch mean loss 1.23, 2.65 at one early step of epoch 2 (a spike that recovered
  to 0.29 within a few hundred steps), and 1.19 at the end of epoch 2. Step losses in epoch 3 were
  in the 0.28–1.00 range.
- **In-run eval not used.** Laya's own report flags its 589-item eval set as overlapping training
  data (147 items), so its before/after numbers are not held-out. The held-out results come from
  `laya_impute.evaluate` on 1,000 test rows, in [../RESULTS.md](../RESULTS.md).
- **Skipped items.** None, after raising `max_len` from 160 to 512. At 160 the run dropped 1,833
  items, almost all of them occupation questions; that run was stopped and discarded.

Files: [`adult-impute-v1.train_run.json`](adult-impute-v1.train_run.json) (config and counts),
[`adult-impute-v1.schema.json`](adult-impute-v1.schema.json) (vocabulary).

## adult-impute-v2 (continued from v1 on a new sample)

- **Command**
  ```bash
  python -m laya_impute.train --train-csv data/adult/adult.data --test-csv data/adult/adult.test \
      --base runs/adult-impute-v1 --schema runs/adult-impute-v1/schema.json \
      --out runs/adult-impute-v2 --max-train-rows 2000 --views-per-row 1 \
      --epochs 2 --micro-batch 8 --grad-accum 4 --seed 1 --device cpu
  ```
- **Starting point.** The v1 checkpoint, with its `schema.json` reused so the vocabulary and
  question options are identical to v1. The `--schema` option exists for this.
- **Data.** A different 2,000-row sample of `adult.data` (seed 1) → 3,299 training items after the
  calibration slice (366 items). 208 optimizer updates at effective batch 32, two epochs.
- **Time.** 63 minutes on 32 CPU cores.
- **Loss.** Epoch means 1.15 and 1.16. Individual step losses spiked (2.45 in epoch 2) and recovered;
  a few steps were slightly negative, which the RLCD objective allows because its policy-gradient
  term has zero-mean advantages.
- **Outcome.** Not a clear improvement over v1. See the v2 section of [../RESULTS.md](../RESULTS.md).

Files: [`adult-impute-v2.train_run.json`](adult-impute-v2.train_run.json).
