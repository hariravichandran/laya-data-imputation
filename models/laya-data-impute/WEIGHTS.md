# Weights

The checkpoint weights (`model.safetensors`, about 800 MB, plus the tokenizer and encoder config) are stored on
the Hugging Face Hub, not in this repository:

- Hub repository: https://huggingface.co/hravi/laya-data-impute (private until the owner makes it public)

This folder holds everything else that identifies the model: the model card (`README.md`), the vocabularies and
binning (`schema.json`), the question definitions (`questions.json`), Laya's config (`rl_agent_config.json`),
and the training record (`train_run.json`, `train_report.json`). Load it with `laya_impute.impute.Imputer`
after downloading the Hub files into one folder.
