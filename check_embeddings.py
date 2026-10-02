import os
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

repos = [
    "tohio/slm-125m-sft",
    "tohio/slm-125m-dpo"
]

for repo_id in repos:
    print("=" * 80)
    print(f"CHECKING REPO: {repo_id}")
    print("=" * 80)
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            token=token
        )
    except Exception as e:
        print(f"Failed to download model.safetensors for {repo_id}: {e}")
        continue

    weights = load_file(model_path)
    
    embed_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"

    if embed_key not in weights or head_key not in weights:
        print(f"Missing {embed_key} or {head_key} in {repo_id}!")
        print(f"Available keys (first 10): {list(weights.keys())[:10]}")
        continue

    embed_w = weights[embed_key].float()
    head_w = weights[head_key].float()

    print(f"  • embed_tokens shape: {list(embed_w.shape)}")
    print(f"  • lm_head shape:      {list(head_w.shape)}")

    is_identical = torch.equal(embed_w, head_w)
    max_diff = (embed_w - head_w).abs().max().item()
    mean_diff = (embed_w - head_w).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        embed_w.view(-1), head_w.view(-1), dim=0
    ).item()

    print(f"\nResults for {repo_id}:")
    print(f"  • Exactly identical?      : {is_identical}")
    print(f"  • Max absolute difference : {max_diff:.8e}")
    print(f"  • Mean absolute difference: {mean_diff:.8e}")
    print(f"  • Cosine similarity       : {cos_sim:.6f}")

    if is_identical or max_diff == 0.0:
        print("\n--> VERDICT: TIED (Weights are identical; lm_head was duplicated on save)")
    elif cos_sim > 0.999 and max_diff < 1e-4:
        print("\n--> VERDICT: TIED WITH MINOR FP PRECISION DRIFT")
    else:
        print("\n--> VERDICT: UNTIED (Weights diverged during training; tie_word_embeddings must be False)")
    print()

