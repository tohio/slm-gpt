"""
export/publisher.py: Automated Hugging Face Hub publication for slm-gpt models.
Creates repositories, generates metadata-rich model cards with lineage telemetry,
and uploads artifacts to Hugging Face.
"""

from dataclasses import asdict
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi, create_repo
import torch

from export.convert import convert_checkpoint_to_hf

# Default 4-Source curriculum references for metadata tagging
DEFAULT_DATASET_TAGS = [
    "HuggingFaceFW/fineweb-edu",
    "HuggingFaceTB/cosmopedia-v2",
    "HuggingFaceTB/smollm-corpus",
    "HuggingFaceTB/finemath",
]


def generate_model_card(
    repo_id: str,
    size_tag: str,
    checkpoint_meta: Optional[Dict[str, Any]] = None,
    dataset_tags: Optional[List[str]] = None,
    sources_distribution: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generates a professional Model Card README.md with YAML metadata,
    curriculum distribution tables, and quickstart snippets.
    """
    meta = checkpoint_meta or {}
    tags = dataset_tags or DEFAULT_DATASET_TAGS
    dist = sources_distribution or {
        "FineWeb-Edu (General / Reasoning)": 0.50,
        "Cosmopedia v2 (Synthetic Textbooks)": 0.20,
        "The Stack-Edu / Python-Edu (Code)": 0.15,
        "FineMath (Mathematical Deduction)": 0.15,
    }

    # 1. Build YAML Front-matter
    yaml_lines = [
        "---",
        "language:",
        "- en",
        "license: mit",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- slm",
        "- gpt",
        "- pretraining",
        "- sft",
        "- dpo",
        "- text-generation",
        "datasets:",
    ]
    for d in tags:
        yaml_lines.append(f"- {d}")
    yaml_lines.extend(["---", ""])

    # 2. Build Markdown Body
    body_lines = [
        f"# {repo_id}",
        "",
        f"`{repo_id}` is a small language model ({size_tag} parameters) built from scratch "
        f"and aligned end-to-end using the **slm-gpt** engine.",
        "",
        "## Architecture Highlights",
        "- **Decoder-Only Transformer** with Rotary Position Embeddings (RoPE)",
        "- **Attention:** Grouped-Query Attention (GQA, 3:1 ratio)",
        "- **Activation:** SwiGLU Feed-Forward Network ($d_{ffn} \\approx \\frac{8}{3} d$)",
        "- **Normalization:** Bias-free RMSNorm",
        "- **Weight Tying:** Tied input embedding (`embed_tokens`) and output head (`lm_head`)",
        "- **Hugging Face Native:** Compatible directly with `LlamaForCausalLM` (no `trust_remote_code=True` required)",
        "",
        "## Pre-training Curriculum",
        "The base model was pre-trained across an interleaved multi-source domain mixture:",
        "",
        "| Domain / Family | Target Ratio | Purpose |",
        "| :--- | :--- | :--- |",
    ]

    for source_name, ratio in dist.items():
        pct = f"{ratio * 100:.1f}%" if ratio <= 1.0 else f"{ratio:.1f}%"
        body_lines.append(f"| `{source_name}` | **{pct}** | Structured representation learning |")

    body_lines.extend([
        "",
        "## Alignment Lineage",
        "1. **Pre-training:** Multi-GPU Distributed Data Parallel (DDP) over rank-disjoint memory-mapped token shards.",
        "2. **Supervised Fine-Tuning (SFT):** Full parameter instruction-tuning on ChatML formatted dialogues with prompt loss masking (`ignore_index=-100`).",
        "3. **Direct Preference Optimization (DPO):** Single-stage pairwise preference alignment optimizing chosen vs. rejected generations.",
        "",
        "## Prompt Format (ChatML)",
        "This model adheres strictly to the ChatML template:",
        "",
        "```text",
        "<|im_start|>system",
        "You are a helpful AI assistant.<|im_end|>",
        "<|im_start|>user",
        "Write a Python script to compute the Fibonacci sequence efficiently.<|im_end|>",
        "<|im_start|>assistant",
        "```",
        "",
        "## Quick Start via `transformers`",
        "",
        "```python",
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'model_id = "{repo_id}"',
        "",
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    model_id,",
        "    torch_dtype=torch.bfloat16,",
        '    device_map="auto",',
        ")",
        "",
        "messages = [",
        '    {"role": "system", "content": "You are a concise AI assistant."},',
        '    {"role": "user", "content": "Explain quantum computing in one sentence."},',
        "]",
        "",
        "prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
        'inputs = tokenizer(prompt, return_tensors="pt").to(model.device)',
        "",
        "outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9)",
        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        "```",
        "",
        "*Trained and published autonomously via [slm-gpt](https://github.com).*",
        "",
    ])

    return "\n".join(yaml_lines + body_lines)


def publish_to_hub(
    checkpoint_path: str,
    repo_id: str,
    export_dir: str = "hf_export",
    token: Optional[str] = None,
    private: bool = False,
    tokenizer_type: str = "tiktoken",
    tokenizer_path: Optional[str] = None,
    dataset_tags: Optional[List[str]] = None,
    sources_distribution: Optional[Dict[str, float]] = None,
):
    """
    Converts checkpoint to SafeTensors, builds assets, and pushes directly to Hugging Face Hub.
    """
    api_token = token or os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("Missing Hugging Face API token. Set HF_TOKEN environment variable or pass token=...")

    api = HfApi(token=api_token)

    # 1. Convert Checkpoint to SafeTensors & Configs
    print("=" * 65)
    print(f"--- Exporting Checkpoint for Hugging Face Hub: {repo_id} ---")
    convert_checkpoint_to_hf(
        checkpoint_path=checkpoint_path,
        output_dir=export_dir,
        tokenizer_type=tokenizer_type,
        tokenizer_path=tokenizer_path,
    )

    # 2. Extract Metadata & Generate Model Card
    meta = {}
    size_tag = "125M"
    if os.path.exists(checkpoint_path):
        ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = ckpt_data.get("runtime_meta", {})
        size_tag = meta.get("size_tag", "125M")

    readme_content = generate_model_card(
        repo_id=repo_id,
        size_tag=size_tag,
        checkpoint_meta=meta,
        dataset_tags=dataset_tags,
        sources_distribution=sources_distribution,
    )

    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"✓ Generated Dynamic Model Card: {export_dir}/README.md")

    # 3. Create Repo & Push Folder
    print(f"Creating / verifying repo '{repo_id}' on Hugging Face Hub...")
    create_repo(repo_id, token=api_token, private=private, exist_ok=True)

    print(f"Uploading assets from '{export_dir}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"✓ Model successfully published to https://huggingface.co/{repo_id}")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m export.publisher <checkpoint_path> <repo_id> [private=True/False]")
        sys.exit(1)

    ckpt_file = sys.argv[1]
    hf_repo = sys.argv[2]
    is_private = sys.argv[3].lower() in ("true", "1") if len(sys.argv) > 3 else False
    publish_to_hub(ckpt_file, hf_repo, private=is_private)
