---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: laya
tags:
  - laya
  - tabular
  - categorical-data
  - data-imputation
datasets:
  - uci-adult
---

# laya-data-impute

Laya fine-tuned to fill missing categorical cells in tables. The approach is generic: each record
becomes a Laya state, each missing column becomes a choice question over that column's vocabulary,
and a cell is filled only when Laya's calibrated confidence reaches the threshold you choose.

**What this checkpoint has been trained and tested on:** UCI Adult only. Its weights are fine-tuned
on census records, and its `schema.json` fixes Adult's columns, bins and vocabularies. The pipeline
supports other categorical tables through dataset presets, but this checkpoint has not been
validated on any other dataset. Other tables need their own schema and their own fine-tune.

Code, dataset presets and evaluation: https://github.com/hariravichandran/laya-data-imputation
Weights: https://huggingface.co/hravi/laya-data-impute (private until made public)

## How it works

- Observed cells become the state, one `column: value` line each.
- All missing targets of a record are asked together, in one forward pass.
- A cell is filled when the calibrated `answer_confidence` clears your threshold. The catch-all
  `Other` value is an abstention, never written into data.

## Usage

```python
import pandas as pd
from laya_impute.impute import Imputer   # from the code repository

imp = Imputer("<downloaded checkpoint folder>", min_confidence=0.5, device="cpu")
filled, audit = imp.impute(pd.read_csv("records.csv", na_values=["?", ""], skipinitialspace=True))
```

Input columns must match the checkpoint's `schema.json`.

## Training

- Base: `convaiinnovations/laya` (Laya 0.4.13).
- v1: three epochs on a 2,000-row sample of Adult's training file (seed 0).
- v2 (this checkpoint): two more epochs on a second 2,000-row sample (seed 1), same vocabulary.
- 208 optimizer updates at effective batch 32, RLCD loss, CPU.

## Held-out evaluation (Adult only)

Adult test rows, one target masked, other cells observed (1,000 rows; "novel" excludes records whose
full categorical profile appears in the training sample).

| target | Laya v2 acc (novel) | Naive Bayes acc (novel) | majority acc (novel) |
|---|---|---|---|
| workclass | 0.731 (0.685) | 0.668 (0.639) | 0.731 (0.685) |
| occupation | 0.303 (0.288) | 0.323 (0.294) | 0.140 (0.142) |
| native-country | 0.921 (0.902) | 0.909 (0.889) | 0.915 (0.895) |

Confidence-threshold behaviour (accuracy on filled cells at threshold 0.5 / 0.8):

- workclass: 0.731 at coverage 100% / 0.864 at 24%
- occupation: 0.897 at coverage 3% / no cells reach 0.8
- native-country: 0.952 at coverage 94% / 0.953 at 93%

Laya does not beat the baselines on raw accuracy. Its filled cells are more precise on the harder
columns, at lower coverage. Single seed; 1,000 test rows, so treat small gaps as noise.

## Limitations

- Validated on Adult only. Performance on other tables is unknown until they are benchmarked.
- Adult is census data with its own biases (sex, race and native country are included as context).
  Do not use imputed values for decisions about individuals.
- Evaluation masks observed cells artificially. Real gaps need not be missing at random.

## License

Apache-2.0 (Laya and the ModernBERT encoder it builds on). Training data: UCI Adult, CC BY 4.0.
