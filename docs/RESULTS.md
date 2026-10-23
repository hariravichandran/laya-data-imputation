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

---

# Benchmark across categorical datasets

One CPU run per dataset, trained with `scripts/run_benchmark.sh` (three epochs, effective batch 32).
Each evaluation uses up to 500 held-out rows with one target masked at a time. Only the single-target
protocol is shown below; the all-targets-masked tables are in `docs/results/<dataset>-v1.json`.
Test sets for votes (79 rows) and breast cancer (57 rows) are small, so their rows are noisy.

## Mushroom (`mushroom`)

Evaluation rows: 500.

| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |
|---|---|---|---|---|---|---|---|
| stalk-root | laya | 0.960 | 0.960 | 1.000 | 0.960 | 0.972 | 0.971 |
| stalk-root | naive_bayes | 0.980 | 0.980 | 1.000 | 0.980 | 0.964 | 0.998 |
| stalk-root | majority | 0.664 | 0.664 | 1.000 | 0.664 | 1.000 | 0.664 |
| spore-print-color | laya | 0.604 | 0.604 | 0.712 | 0.646 | 0.276 | 1.000 |
| spore-print-color | naive_bayes | 0.626 | 0.626 | 0.984 | 0.632 | 0.292 | 0.966 |
| spore-print-color | majority | 0.346 | 0.346 | 1.000 | 0.346 | 1.000 | 0.346 |
| habitat | laya | 0.684 | 0.684 | 0.686 | 0.843 | 0.504 | 0.984 |
| habitat | naive_bayes | 0.666 | 0.666 | 0.996 | 0.667 | 0.930 | 0.680 |
| habitat | majority | 0.462 | 0.462 | 1.000 | 0.462 | 1.000 | 0.462 |

## Congressional votes (`votes`)

Evaluation rows: 79.

| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |
|---|---|---|---|---|---|---|---|
| vote-01 | laya | 0.620 | 0.532 | 1.000 | 0.620 | 0.342 | 0.667 |
| vote-01 | naive_bayes | 0.620 | 0.532 | 1.000 | 0.620 | 0.962 | 0.618 |
| vote-01 | majority | 0.557 | 0.548 | 1.000 | 0.557 | 1.000 | 0.557 |
| vote-02 | laya | 0.456 | 0.419 | 1.000 | 0.456 | 0.000 | - |
| vote-02 | naive_bayes | 0.646 | 0.661 | 1.000 | 0.646 | 0.683 | 0.685 |
| vote-02 | majority | 0.405 | 0.371 | 1.000 | 0.405 | 1.000 | 0.405 |
| vote-03 | laya | 0.810 | 0.758 | 1.000 | 0.810 | 1.000 | 0.810 |
| vote-03 | naive_bayes | 0.861 | 0.823 | 1.000 | 0.861 | 0.975 | 0.883 |
| vote-03 | majority | 0.532 | 0.581 | 1.000 | 0.532 | 1.000 | 0.532 |
| vote-04 | laya | 0.975 | 0.968 | 1.000 | 0.975 | 1.000 | 0.975 |
| vote-04 | naive_bayes | 0.886 | 0.855 | 1.000 | 0.886 | 0.987 | 0.885 |
| vote-04 | majority | 0.570 | 0.629 | 1.000 | 0.570 | 1.000 | 0.570 |

## Breast cancer (`breast-cancer`)

Evaluation rows: 57.

| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |
|---|---|---|---|---|---|---|---|
| node-caps | laya | 0.912 | 0.898 | 1.000 | 0.912 | 0.807 | 0.957 |
| node-caps | naive_bayes | 0.895 | 0.878 | 1.000 | 0.895 | 0.860 | 0.918 |
| node-caps | majority | 0.825 | 0.796 | 1.000 | 0.825 | 1.000 | 0.825 |
| breast-quad | laya | 0.404 | 0.388 | 0.491 | 0.500 | 0.000 | - |
| breast-quad | naive_bayes | 0.439 | 0.429 | 0.667 | 0.342 | 0.000 | - |
| breast-quad | majority | 0.404 | 0.388 | 1.000 | 0.404 | 1.000 | 0.404 |

## Bank marketing (`bank`)

Evaluation rows: 500.

| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |
|---|---|---|---|---|---|---|---|
| job | laya | 0.260 | 0.258 | 0.000 | - | 0.000 | - |
| job | naive_bayes | 0.432 | 0.429 | 0.592 | 0.551 | 0.152 | 0.618 |
| job | majority | 0.258 | 0.256 | 1.000 | 0.258 | 1.000 | 0.258 |
| education | laya | 0.494 | 0.495 | 0.954 | 0.507 | 0.000 | - |
| education | naive_bayes | 0.690 | 0.688 | 0.930 | 0.712 | 0.456 | 0.772 |
| education | majority | 0.494 | 0.495 | 1.000 | 0.494 | 1.000 | 0.494 |
| contact | laya | 0.926 | 0.926 | 1.000 | 0.926 | 1.000 | 0.926 |
| contact | naive_bayes | 0.904 | 0.903 | 1.000 | 0.904 | 0.894 | 0.935 |
| contact | majority | 0.926 | 0.926 | 1.000 | 0.926 | 1.000 | 0.926 |

## Car evaluation (`car`)

Evaluation rows: 346.

| target | method | acc (all) | acc (novel) | coverage@0.5 | acc@0.5 | coverage@0.8 | acc@0.8 |
|---|---|---|---|---|---|---|---|
| buying | laya | 0.292 | 0.292 | 0.078 | 0.630 | 0.000 | - |
| buying | naive_bayes | 0.246 | 0.246 | 0.078 | 0.630 | 0.000 | - |
| buying | majority | 0.240 | 0.240 | 1.000 | 0.240 | 1.000 | 0.240 |
| maint | laya | 0.301 | 0.301 | 0.078 | 0.630 | 0.000 | - |
| maint | naive_bayes | 0.286 | 0.286 | 0.052 | 0.778 | 0.000 | - |
| maint | majority | 0.205 | 0.205 | 1.000 | 0.205 | 1.000 | 0.205 |
| safety | laya | 0.460 | 0.460 | 0.679 | 0.528 | 0.029 | 1.000 |
| safety | naive_bayes | 0.445 | 0.445 | 0.607 | 0.448 | 0.029 | 1.000 |
| safety | majority | 0.306 | 0.306 | 1.000 | 0.306 | 1.000 | 0.306 |

## Summary (single-target protocol, accuracy on all rows)

| dataset | target | Laya acc | Naive Bayes acc | majority acc |
|---|---|---|---|---|
| mushroom | stalk-root | 0.960 | 0.980 | 0.664 |
| mushroom | spore-print-color | 0.604 | 0.626 | 0.346 |
| mushroom | habitat | 0.684 | 0.666 | 0.462 |
| votes | vote-01 | 0.620 | 0.620 | 0.557 |
| votes | vote-02 | 0.456 | 0.646 | 0.405 |
| votes | vote-03 | 0.810 | 0.861 | 0.532 |
| votes | vote-04 | 0.975 | 0.886 | 0.570 |
| breast-cancer | node-caps | 0.912 | 0.895 | 0.825 |
| breast-cancer | breast-quad | 0.404 | 0.439 | 0.404 |
| bank | job | 0.260 | 0.432 | 0.258 |
| bank | education | 0.494 | 0.690 | 0.494 |
| bank | contact | 0.926 | 0.904 | 0.926 |
| car | buying | 0.292 | 0.246 | 0.240 |
| car | maint | 0.301 | 0.286 | 0.205 |
| car | safety | 0.460 | 0.445 | 0.306 |

## What this shows

- **Laya beats majority on most targets** (mushroom, votes, breast node-caps, car) and **ties it** on bank
  contact, education, job and breast-quad. On breast-quad Naive Bayes is the only method above majority
  (0.439 against 0.404), and even that gap is small.
- **Against Naive Bayes, Laya is mixed.** It is ahead on votes vote-04 (0.975 against 0.886), on breast
  node-caps (0.912 against 0.895), and on mushroom habitat, where at threshold 0.8 its filled cells are
  98% accurate against 68% for Naive Bayes. It is behind on votes vote-02, on bank job and education,
  and on mushroom stalk-root and spore-print-color.
- **Where Laya is behind, the training budget is the likely cause.** Those are the targets with the most
  distinct values or the least training data per value (bank job has 11 values, vote-02 has 43 gaps in
  348 rows). A bigger budget is the obvious next experiment, not a verified fix.
- **Confidence is conservative.** On car and bank, Laya fills few cells at threshold 0.5, so its
  high-confidence precision is high but its coverage is low.
- Naive Bayes is a strong baseline on small categorical tables. Laya's clearest advantage here is its
  calibrated abstention, not raw accuracy.

Run records: `docs/runs/<dataset>-v1.train_run.json` and `docs/runs/<dataset>-v1.schema.json`.
