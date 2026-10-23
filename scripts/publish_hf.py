"""Upload a trained checkpoint folder to the Hugging Face Hub.

    python scripts/publish_hf.py --folder <staged checkpoint> --repo-id <user>/<name> [--public]

The repository is created private unless `--public` is given. Log in first with
`huggingface-cli login` or set HF_TOKEN.
"""

import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    if not os.path.isfile(os.path.join(args.folder, "README.md")):
        raise SystemExit("the folder needs a README.md model card")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=args.folder, repo_id=args.repo_id, repo_type="model",
                      commit_message="Upload Laya categorical imputation checkpoint")
    print("https://huggingface.co/%s" % args.repo_id)


if __name__ == "__main__":
    main()
