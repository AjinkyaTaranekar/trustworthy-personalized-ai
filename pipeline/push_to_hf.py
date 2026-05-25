"""
Push a local model directory to HuggingFace.

Usage:
    python push_to_hf.py                                      # uses defaults
    python push_to_hf.py --dir ./models/checkpoint_sft_merged
    python push_to_hf.py --dir ./models/checkpoint_grpo_d_merged --repo AjinkyaTaranekar/trustworthy-ai-grpo-d
"""

import argparse
import os
import sys
from pathlib import Path

HF_USERNAME = "AjinkyaTaranekar"


def derive_repo(checkpoint_dir: str) -> str:
    name = Path(checkpoint_dir).name  # e.g. checkpoint_sft_merged
    slug = name.replace("checkpoint_", "").replace("_merged", "").replace("_", "-")
    return f"{HF_USERNAME}/trustworthy-ai-{slug}"


def main():
    parser = argparse.ArgumentParser(description="Push a local model dir to HuggingFace")
    parser.add_argument("--dir",   default="./models/checkpoint_sft_merged",
                        help="Local model directory to upload")
    parser.add_argument("--repo",  default=None,
                        help="HF repo ID (default: derived from --dir name)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (default: $HF_TOKEN)")
    args = parser.parse_args()

    checkpoint_dir = args.dir
    repo_id = args.repo or derive_repo(checkpoint_dir)

    if not Path(checkpoint_dir).exists():
        print(f"ERROR: directory not found: {checkpoint_dir}")
        sys.exit(1)

    if not args.token:
        print("ERROR: HF_TOKEN not set. Pass --token or set the env var.")
        sys.exit(1)

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    print(f"Uploading {checkpoint_dir} → {repo_id} ...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
