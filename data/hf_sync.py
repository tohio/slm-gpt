"""
data/hf_sync.py: Fast HF Hub Dataset Synchronization Utility for slm-gpt.
Decouples CPU-heavy tokenization and curriculum packing from expensive GPU nodes.
Supports parallel multi-part push, automated dataset card generation, and fast snapshot pull.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Load local environment (.env) for HF_TOKEN
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import HfApi, snapshot_download, create_repo

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import dynamic dataset card generator
try:
    from data.generate_dataset_card import generate_card
except ImportError:
    try:
        from generate_dataset_card import generate_card
    except ImportError:
        generate_card = None


def parse_cli_args() -> Dict[str, Any]:
    kwargs = {
        "action": "pull",              # 'push' or 'pull'
        "repo_id": None,               # e.g., 'tohio/slm-curriculum-10b'
        "stages": "pretrain,sft,dpo",  # comma-separated: 'pretrain', 'sft', 'dpo', or 'all'
        "data_dir": "data",
        "private": True,
        "max_workers": 8,
        "tokens": None,                # '10B' or '80B' (inferred if None)
        "model_target": None,          # '125M', '350M', '1B' (inferred if None)
    }
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            if k in kwargs:
                if k == "private":
                    kwargs[k] = v.lower() in ("true", "1", "yes")
                elif k == "max_workers":
                    kwargs[k] = int(v)
                else:
                    kwargs[k] = v
            else:
                kwargs[k] = v
    return kwargs


def push_dataset(
    repo_id: str,
    data_dir: str = "data",
    stages: List[str] = None,
    private: bool = True,
    tokens: str = None,
    model_target: str = None,
):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set. Add it to your .env or shell environment.")

    api = HfApi(token=token)
    print(f"\n[Hub Push] Target Repository: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

    data_path = Path(data_dir)
    upload_stages = stages or ["pretrain", "sft", "dpo"]

    # 1. Upload stage binary artifacts
    for stage in upload_stages:
        stage_dir = data_path / stage
        if not stage_dir.exists():
            # If data_dir contains the shards directly (e.g. data/slm-curriculum-10b/train_*.bin)
            if stage == "pretrain" and any(data_path.glob("train_*.bin")):
                stage_dir = data_path
            else:
                print(f"  ⚠️ Skipping '{stage}': directory '{stage_dir}' not found.")
                continue

        files = [f for f in stage_dir.iterdir() if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".md")]
        if not files:
            print(f"  ⚠️️ No binary artifacts found in '{stage_dir}'.")
            continue

        total_bytes = sum(f.stat().st_size for f in files)
        print(f"\n[Hub Push] Uploading stage '{stage}' ({len(files)} files, {total_bytes / (1024**2):.1f} MB)...")

        api.upload_folder(
            folder_path=str(stage_dir),
            path_in_repo=stage,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {stage} artifacts ({len(files)} files)",
            allow_patterns=["*.bin", "*.json", "*.jsonl", "*.parquet"],
        )
        print(f"  ✓ Uploaded stage: {stage}")

    # 2. Automatically generate and push the dataset card
    resolved_tokens = tokens
    if not resolved_tokens:
        resolved_tokens = "80B" if "80b" in repo_id.lower() else "10B"

    resolved_target = model_target
    if not resolved_target:
        resolved_target = "350M" if resolved_tokens.upper() == "80B" else "125M"

    readme_path = data_path / "README.md"
    if generate_card is not None:
        print(f"\n[Hub Push] Generating dynamic dataset card ({resolved_tokens}, {resolved_target}) via generate_dataset_card.py...")
        generate_card(tokens=resolved_tokens, model_target=resolved_target, output_path=str(readme_path))
    else:
        print("  ⚠️ generate_card could not be imported; falling back to existing README.md if present.")

    if readme_path.is_file():
        print(f"[Hub Push] Syncing dataset card from '{readme_path}' to repo root...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="docs: sync dynamic dataset card",
        )
        print("  ✓ Dataset card synced to repository root.")
    else:
        print(f"  ⚠️ No dataset card found at '{readme_path}' to upload.")

    print("\n" + "=" * 70)
    print(f"✓ Push complete: https://huggingface.co/datasets/{repo_id}")
    print("=" * 70)


def pull_dataset(repo_id: str, data_dir: str = "data", stages: List[str] = None, max_workers: int = 8):
    token = os.getenv("HF_TOKEN")
    print(f"\n[Hub Pull] Downloading from: {repo_id}")
    target_data_dir = Path(data_dir)
    target_data_dir.mkdir(parents=True, exist_ok=True)

    requested = stages or ["pretrain", "sft", "dpo"]
    patterns = [f"{st}/*" for st in requested] + ["README.md"]

    print(f"[Hub Pull] Target patterns: {patterns}")
    print(f"[Hub Pull] Parallel workers: {max_workers}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_data_dir),
        allow_patterns=patterns,
        token=token,
        max_workers=max_workers,
    )

    print("\n" + "=" * 70)
    print("✓ Dataset artifacts synchronized locally:")
    for st in requested:
        st_dir = target_data_dir / st
        if st_dir.exists():
            files = list(st_dir.glob("*.*"))
            total_size_mb = sum(f.stat().st_size for f in files) / (1024**2)
            print(f"  • {st:<12}: {len(files)} files ({total_size_mb:,.1f} MB) in '{st_dir}'")
    print("=" * 70)


def main():
    args = parse_cli_args()
    if not args["repo_id"]:
        print("Usage:")
        print("  Push: python3 data/hf_sync.py action=push repo_id=<username>/<dataset_name> [stages=pretrain,sft,dpo]")
        print("  Pull: python3 data/hf_sync.py action=pull repo_id=<username>/<dataset_name> [stages=pretrain,sft,dpo]")
        sys.exit(1)

    stages = [s.strip() for s in args["stages"].split(",") if s.strip()]
    if "all" in stages:
        stages = ["pretrain", "sft", "dpo"]

    action = args["action"].lower()
    if action == "push":
        push_dataset(
            repo_id=args["repo_id"],
            data_dir=args["data_dir"],
            stages=stages,
            private=args["private"],
            tokens=args["tokens"],
            model_target=args["model_target"],
        )
    elif action == "pull":
        pull_dataset(
            repo_id=args["repo_id"],
            data_dir=args["data_dir"],
            stages=stages,
            max_workers=args["max_workers"],
        )
    else:
        raise ValueError(f"Unknown action: '{action}'. Must be 'push' or 'pull'.")


if __name__ == "__main__":
    main()
