"""
utils/push_datacards.py: Standalone script to generate and push dataset cards
for tohio/slm-curriculum-10b and tohio/slm-curriculum-80b.
Loads HF_TOKEN directly from repository-root .env using load_dotenv().
"""

import os
from pathlib import Path
import sys
import tempfile

# 1. Resolve repository root and explicitly load .env
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

env_path = REPO_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

from huggingface_hub import HfApi
from data.datacard import create_dataset_card


def push_card(api: HfApi, repo_id: str, card_path: Path):
    print(f"\n[Pushing Card] Uploading to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="docs: add dynamic dataset card with viewer: false",
    )
    print(f"✓ Successfully published README.md to https://huggingface.co/datasets/{repo_id}")


def main():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError(
            f"HF_TOKEN not found. Checked .env at: {env_path.resolve() if env_path.exists() else 'Default environment'}"
        )

    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # =====================================================================
        # 1. 10B Curriculum Card
        # =====================================================================
        stats_10b = {
            "FineWeb-Edu": {
                "upstream": "HuggingFaceFW/fineweb-edu",
                "tokens": 4_833_000_000,
                "status": "100% Unique",
            },
            "DCLM-Edu": {
                "upstream": "HuggingFaceTB/dclm-edu",
                "tokens": 2_517_000_000,
                "status": "100% Unique",
            },
            "The Stack-Edu": {
                "upstream": "HuggingFaceTB/smollm-corpus",
                "tokens": 1_510_000_000,
                "status": "100% Unique",
            },
            "NuminaMath-CoT": {
                "upstream": "AI-MO/NuminaMath-CoT",
                "tokens": 440_000_000,
                "status": "2.00× Repetition Ceiling",
            },
            "OpenMathReasoning": {
                "upstream": "nvidia/OpenMathReasoning",
                "tokens": 400_000_000,
                "status": "2.00× Repetition Ceiling",
            },
            "SLM-Synthetic": {
                "upstream": "tohio/slm-synthetic-pretrain",
                "tokens": 300_000_000,
                "status": "1.20× Repetition",
            },
        }

        out_10b = Path(tmp_dir) / "10b"
        card_10b_path = create_dataset_card(
            output_dir=out_10b,
            dataset_name="slm-curriculum-10b",
            total_tokens=10_000_000_000,
            shard_count=400,
            tokens_per_shard=25_000_000,
            source_stats=stats_10b,
            val_tokens=500_000,
            tokenizer_type="tiktoken",
            vocab_size=50257,
            eot_token_id=50256,
        )
        push_card(api, "tohio/slm-curriculum-10b", card_10b_path)

        # =====================================================================
        # 2. 80B Curriculum Card
        # =====================================================================
        stats_80b = {
            "FineWeb-Edu": {
                "upstream": "HuggingFaceFW/fineweb-edu",
                "tokens": 42_791_000_000,
                "status": "100% Unique",
            },
            "DCLM-Edu": {
                "upstream": "HuggingFaceTB/dclm-edu",
                "tokens": 22_287_000_000,
                "status": "100% Unique",
            },
            "The Stack-Edu": {
                "upstream": "HuggingFaceTB/smollm-corpus",
                "tokens": 13_372_000_000,
                "status": "100% Unique",
            },
            "NuminaMath-CoT": {
                "upstream": "AI-MO/NuminaMath-CoT",
                "tokens": 550_000_000,
                "status": "2.50× Repetition Ceiling",
            },
            "OpenMathReasoning": {
                "upstream": "nvidia/OpenMathReasoning",
                "tokens": 500_000_000,
                "status": "2.50× Repetition Ceiling",
            },
            "SLM-Synthetic": {
                "upstream": "tohio/slm-synthetic-pretrain",
                "tokens": 500_000_000,
                "status": "2.00× Repetition Ceiling",
            },
        }

        out_80b = Path(tmp_dir) / "80b"
        card_80b_path = create_dataset_card(
            output_dir=out_80b,
            dataset_name="slm-curriculum-80b",
            total_tokens=80_000_000_000,
            shard_count=3200,
            tokens_per_shard=25_000_000,
            source_stats=stats_80b,
            val_tokens=500_000,
            tokenizer_type="tiktoken",
            vocab_size=50257,
            eot_token_id=50256,
        )
        push_card(api, "tohio/slm-curriculum-80b", card_80b_path)

    print("\n" + "=" * 70)
    print("✓ Both dataset cards have been successfully pushed.")
    print("=" * 70)


if __name__ == "__main__":
    main()