# Results: categorical imputation on UCI Adult

Single run, seed 0, CPU. Ground truth is held-out `adult.test` rows, with the target cell masked.
Laya was trained on 2,000 sampled `adult.data` rows (3,367 training items after the calibration
slice), three epochs, effective batch 32, 318 optimizer updates. Evaluation uses 1,000 random
complete test rows. "novel" excludes the 190 rows whose full categorical record also appears in
the training sample.

Columns: `acc (all)` counts every answer, with abstentions as misses. `coverage@t` is the share
of cells filled when confidence is at least `t`, and `acc@t` is the accuracy on those filled cells.

## Protocol 1: one target masked, other cells observed

| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |
|---|---|---|---|---|---|---|---|
| workclass | laya | 0.731 | 0.685 | 0.944 | 0.746 | 0.342 | 0.880 |
| workclass | naive_bayes | 0.668 | 0.639 | 0.844 | 0.729 | 0.460 | 0.830 |
| workclass | majority | 0.731 | 0.685 | 1.000 | 0.731 | 1.000 | 0.731 |
| occupation | laya | 0.301 | 0.283 | 0.070 | 0.743 | 0.009 | 1.000 |
| occupation | naive_bayes | 0.323 | 0.294 | 0.338 | 0.453 | 0.026 | 0.346 |
| occupation | majority | 0.140 | 0.142 | 1.000 | 0.140 | 1.000 | 0.140 |
| native-country | laya | 0.915 | 0.895 | 0.988 | 0.922 | 0.927 | 0.953 |
| native-country | naive_bayes | 0.909 | 0.889 | 0.963 | 0.933 | 0.903 | 0.954 |
| native-country | majority | 0.915 | 0.895 | 1.000 | 0.915 | 1.000 | 0.915 |

## Protocol 2: all three targets masked at once

| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |
|---|---|---|---|---|---|---|---|
| workclass | laya | 0.731 | 0.685 | 0.955 | 0.741 | 0.383 | 0.851 |
| workclass | naive_bayes | 0.692 | 0.658 | 0.863 | 0.725 | 0.420 | 0.829 |
| workclass | majority | 0.731 | 0.685 | 1.000 | 0.731 | 1.000 | 0.731 |
| occupation | laya | 0.294 | 0.279 | 0.073 | 0.726 | 0.018 | 0.889 |
| occupation | naive_bayes | 0.325 | 0.301 | 0.306 | 0.448 | 0.019 | 0.368 |
| occupation | majority | 0.140 | 0.142 | 1.000 | 0.140 | 1.000 | 0.140 |
| native-country | laya | 0.915 | 0.895 | 0.983 | 0.926 | 0.932 | 0.952 |
| native-country | naive_bayes | 0.912 | 0.891 | 0.967 | 0.934 | 0.909 | 0.953 |
| native-country | majority | 0.915 | 0.895 | 1.000 | 0.915 | 1.000 | 0.915 |

## What this shows

- **Raw accuracy does not favour Laya.** It ties majority on workclass and native-country, and it
  is slightly below Naive Bayes on occupation (0.301 against 0.323).
- **Laya's advantage is selective precision on the harder columns.** Its confidence is well
  calibrated: on occupation, cells it fills at 0.5 are right 74% of the time, against 45% for Naive
  Bayes. The cost is coverage, which is only 7%. On workclass, at threshold 0.8 it reaches 0.88
  accuracy, against 0.83 for Naive Bayes, but covers 34% of cells against 46%.
- **Native-country is not informative.** The column is about 91% `United-States`, so a majority
  guess is already strong.
- **Uncertainty.** One seed and 1,000 test rows give a standard error of roughly 1.5 points on
  accuracy, so the workclass gap at threshold 0.8 is suggestive rather than settled.

## Reproduce

```bash
python -m laya_impute.evaluate --checkpoint runs/adult-impute-v1 \
    --train-csv data/adult/adult.data --test-csv data/adult/adult.test \
    --n-rows 1000 --max-train-rows 2000 --out docs/results/laya-adult-v1.json
python -m laya_impute.evaluate --train-csv data/adult/adult.data \
    --test-csv data/adult/adult.test --n-rows 1000 --max-train-rows 2000 \
    --out docs/results/baselines.json      # baselines only
```

Raw JSON: `docs/results/laya-adult-v1.json` and `docs/results/baselines.json`.

---

# Continued training: adult-impute-v2

v2 starts from the v1 checkpoint (same `schema.json`, so the vocabulary and options are identical)
and trains two more epochs on a different 2,000-row sample (seed 1). Settings and timing are in
[`runs/README.md`](runs/README.md). Same evaluation, rows and baselines as above.

## Protocol 1: one target masked

| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |
|---|---|---|---|---|---|---|---|
| workclass | laya v2 | 0.731 | 0.685 | 1.000 | 0.731 | 0.236 | 0.864 |
| workclass | laya v1 | 0.731 | 0.685 | 0.944 | 0.746 | 0.342 | 0.880 |
| workclass | naive_bayes | 0.668 | 0.639 | 0.844 | 0.729 | 0.460 | 0.830 |
| occupation | laya v2 | 0.303 | 0.288 | 0.029 | 0.897 | 0.000 | - |
| occupation | laya v1 | 0.301 | 0.283 | 0.070 | 0.743 | 0.009 | 1.000 |
| occupation | naive_bayes | 0.323 | 0.294 | 0.338 | 0.453 | 0.026 | 0.346 |
| native-country | laya v2 | 0.921 | 0.902 | 0.936 | 0.952 | 0.932 | 0.953 |
| native-country | laya v1 | 0.915 | 0.895 | 0.988 | 0.922 | 0.927 | 0.953 |
| native-country | naive_bayes | 0.909 | 0.889 | 0.963 | 0.933 | 0.903 | 0.954 |

## Protocol 2: all three targets masked at once

| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |
|---|---|---|---|---|---|---|---|
| workclass | laya v2 | 0.731 | 0.685 | 1.000 | 0.731 | 0.283 | 0.855 |
| workclass | laya v1 | 0.731 | 0.685 | 0.955 | 0.741 | 0.383 | 0.851 |
| workclass | naive_bayes | 0.692 | 0.658 | 0.863 | 0.725 | 0.420 | 0.829 |
| occupation | laya v2 | 0.304 | 0.289 | 0.034 | 0.912 | 0.000 | - |
| occupation | laya v1 | 0.294 | 0.279 | 0.073 | 0.726 | 0.018 | 0.889 |
| occupation | naive_bayes | 0.325 | 0.301 | 0.306 | 0.448 | 0.019 | 0.368 |
| native-country | laya v2 | 0.922 | 0.904 | 0.936 | 0.952 | 0.935 | 0.952 |
| native-country | laya v1 | 0.915 | 0.895 | 0.983 | 0.926 | 0.932 | 0.952 |
| native-country | naive_bayes | 0.912 | 0.891 | 0.967 | 0.934 | 0.909 | 0.953 |

## What changed, and what did not

- **Continued training did not give a clear improvement.** Overall accuracy moved by at most two
  points. native-country improved slightly.
- **Confidence moved, not accuracy.** On workclass, v2 fills every cell at threshold 0.5, at the same
  accuracy, so its 0.5 threshold is no longer selective. At 0.8, coverage fell from 34% to 24%.
- **On occupation, v2 is more precise but almost silent.** It fills 3% of cells at threshold 0.5
  with 0.90 accuracy, and nothing reaches 0.8.
- **Recommendation.** Keep v1 as the reference checkpoint. Choose a confidence threshold per column
  on a validation split instead of reusing one number across checkpoints, since calibration differs
  between runs.
- **Baselines are unchanged.** They are trained on the seed-0 2,000-row sample, as v1 was. v2 saw
  about 4,000 rows in total, so the baselines are not given the same amount of data. A fair
  comparison would retrain them on the full set.

Raw JSON: `docs/results/laya-adult-v2.json`.
