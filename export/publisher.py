"""
export/publisher.py: Automated Hugging Face Hub publication for slm-gpt models.
Dynamically extracts architecture specs, token counts, and evaluation metrics
from model weights and metadata without hardcoding sizes or hyperparameters.
"""

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi, create_repo
import torch

from export.convert import convert_checkpoint_to_hf
from transformer.config import ModelConfig

# Default curriculum sources if not present in runtime_meta
PRETRAIN_DATASETS = [
    "FineWeb-Edu",
    "DCLM-Edu",
    "The Stack-Edu",
    "NuminaMath-CoT",
    "OpenMathReasoning",
    "SLM-Synthetic-Pretrain",
]

DEFAULT_DATASET_TAGS = [
    "HuggingFaceFW/fineweb-edu",
    "mlfoundations/dclm-baseline-1.0",
    "bigcode/the-stack",
    "AI-MO/NuminaMath-CoT",
    "nvidia/OpenMathReasoning",
    "tohio/slm-synthetic-pretrain",
]


def _format_param_count(num_params: int) -> Tuple[str, str]:
    """Formats raw parameter integers into display and tag strings (e.g., '126.8M', '127M')."""
    if num_params >= 1_000_000_000:
        display_str = f"{num_params / 1_000_000_000:.2f}B"
        tag_str = f"{round(num_params / 1_000_000_000)}B"
    elif num_params >= 1_000_000:
        display_str = f"{num_params / 1_000_000:.1f}M"
        tag_str = f"{round(num_params / 1_000_000)}M"
    else:
        display_str = f"{num_params:,}"
        tag_str = "custom"
    return display_str, tag_str


def _format_token_count(total_tokens: Optional[int]) -> str:
    """Formats raw token counts into human-readable strings (e.g., '10B', '25.5B')."""
    if not total_tokens:
        return "multi-billion"
    if total_tokens >= 1_000_000_000:
        val = total_tokens / 1_000_000_000
        return f"{val:.1f}B".replace(".0B", "B")
    if total_tokens >= 1_000_000:
        val = total_tokens / 1_000_000
        return f"{val:.1f}M".replace(".0M", "M")
    return f"{total_tokens:,}"


def generate_model_card(
    repo_id: str,
    config: Optional[ModelConfig] = None,
    num_params: int = 0,
    checkpoint_meta: Optional[Dict[str, Any]] = None,
    eval_metrics: Optional[Dict[str, Any]] = None,
    dataset_tags: Optional[List[str]] = None,
    curriculum_datasets: Optional[List[str]] = None,
) -> str:
    meta = checkpoint_meta or {}
    tags = dataset_tags or DEFAULT_DATASET_TAGS
    datasets_list = curriculum_datasets or PRETRAIN_DATASETS
    repo_lower = repo_id.lower()

    # Dynamic parameter strings
    param_display, param_tag = _format_param_count(num_params)
    size_tag = meta.get("size_tag", param_tag)

    # Dynamic token volume
    raw_tokens = meta.get("total_tokens") or meta.get("tokens_trained")
    token_str = _format_token_count(raw_tokens) if isinstance(raw_tokens, int) else str(raw_tokens or "10B")

    # Architecture telemetry
    n_layers = getattr(config, "n_layers", "N/A")
    d_model = getattr(config, "d_model", getattr(config, "dim", "N/A"))
    d_ffn = getattr(config, "d_ffn", "N/A")
    n_heads = getattr(config, "n_heads", 1)
    n_kv_heads = getattr(config, "n_kv_heads", n_heads)

    if n_heads == n_kv_heads:
        attn_desc = f"Multi-Head Attention (MHA, {n_heads} heads)"
    else:
        attn_desc = f"Grouped-Query Attention (GQA, {n_heads}:{n_kv_heads} ratio)"

    # Resolve sibling repository links
    repo_base_name = re.sub(r"-(base|sft|dpo)$", "", repo_id)
    base_repo_link = f"[`{repo_base_name}-base`](https://huggingface.co/{repo_base_name}-base)"
    sft_repo_link = f"[`{repo_base_name}-sft`](https://huggingface.co/{repo_base_name}-sft)"
    dpo_repo_link = f"[`{repo_base_name}-dpo`](https://huggingface.co/{repo_base_name}-dpo)"

    is_base = "base" in repo_lower
    is_sft = "sft" in repo_lower
    is_dpo = "dpo" in repo_lower

    # 1. YAML Front-matter
    yaml_tags = ["- slm", "- gpt", "- pretraining", "- text-generation"]
    if is_sft or is_dpo:
        yaml_tags.append("- sft")
    if is_dpo:
        yaml_tags.append("- dpo")

    yaml_lines = [
        "---",
        "language:",
        "- en",
        "license: mit",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
    ] + yaml_tags + ["datasets:"]
    for d in tags:
        yaml_lines.append(f"- {d}")
    yaml_lines.extend(["---", ""])

    # 2. Stage-aware narrative & lineage
    if is_base:
        stage_desc = (
            f"`{repo_id}` is a pre-trained base language model ({size_tag}, ~{param_display} parameters) "
            f"trained from scratch on a {token_str} token educational and reasoning curriculum. "
            f"This checkpoint serves as the foundational base and has **not** undergone instruction tuning or alignment."
        )
        lineage_section = [
            "## Pipeline Lineage",
            f"1. **Pre-training (`{repo_id}`):** Completed {token_str} tokens over packed memory-mapped token shards.",
            f"2. **Supervised Fine-Tuning (SFT):** See {sft_repo_link}.",
            f"3. **Direct Preference Optimization (DPO):** See {dpo_repo_link}.",
        ]
        prompt_section = [
            "## Prompt Format (Raw Completion)",
            "As an unaligned base model, this checkpoint performs raw document continuation rather than dialogue:",
            "",
            "```text",
            "The key difference between a compiler and an interpreter is that",
            "```",
        ]
        quickstart_code = [
            "import torch",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'model_id = "{repo_id}"',
            "tokenizer = AutoTokenizer.from_pretrained(model_id)",
            "model = AutoModelForCausalLM.from_pretrained(",
            "    model_id,",
            "    torch_dtype=torch.bfloat16,",
            '    device_map="auto",',
            ")",
            "",
            'prompt = "The key difference between a compiler and an interpreter is that"',
            'inputs = tokenizer(prompt, return_tensors="pt").to(model.device)',
            "outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7, top_p=0.9)",
            "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        ]
    elif is_sft:
        stage_desc = (
            f"`{repo_id}` is an instruction-tuned small language model ({size_tag}, ~{param_display} parameters) "
            f"fine-tuned on multi-turn reasoning and instruction dialogues formatted in ChatML."
        )
        lineage_section = [
            "## Pipeline Lineage",
            f"1. **Pre-training:** Base weights pre-trained on {token_str} tokens ({base_repo_link}).",
            f"2. **Supervised Fine-Tuning (`{repo_id}`):** Full-parameter instruction tuning with loss masking (`ignore_index=-100`).",
            f"3. **Direct Preference Optimization (DPO):** See {dpo_repo_link}.",
        ]
        prompt_section = [
            "## Prompt Format (ChatML)",
            "This model expects standard ChatML formatting:",
            "",
            "```text",
            "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>",
            "<|im_start|>user\nExplain recursion in one sentence.<|im_end|>",
            "<|im_start|>assistant",
            "```",
        ]
        quickstart_code = [
            "import torch",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'model_id = "{repo_id}"',
            "tokenizer = AutoTokenizer.from_pretrained(model_id)",
            "model = AutoModelForCausalLM.from_pretrained(",
            "    model_id,",
            "    torch_dtype=torch.bfloat16,",
            '    device_map="auto",',
            ")",
            "",
            'prompt = "<|im_start|>user\\nExplain recursion in one sentence.<|im_end|>\\n<|im_start|>assistant\\n"',
            'inputs = tokenizer(prompt, return_tensors="pt").to(model.device)',
            "outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7, top_p=0.9)",
            "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        ]
    else:  # DPO
        stage_desc = (
            f"`{repo_id}` is a preference-aligned small language model ({size_tag}, ~{param_display} parameters) "
            f"aligned with Direct Preference Optimization (DPO) on pairwise conversational preferences."
        )
        lineage_section = [
            "## Pipeline Lineage",
            f"1. **Pre-training:** Base weights pre-trained on {token_str} tokens ({base_repo_link}).",
            f"2. **Supervised Fine-Tuning:** SFT instruction baseline ({sft_repo_link}).",
            f"3. **Direct Preference Optimization (`{repo_id}`):** Pairwise preference optimization on normalized token likelihoods.",
        ]
        prompt_section = [
            "## Prompt Format (ChatML)",
            "This model expects standard ChatML formatting:",
            "",
            "```text",
            "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>",
            "<|im_start|>user\nWrite a quicksort function in Python.<|im_end|>",
            "<|im_start|>assistant",
            "```",
        ]
        quickstart_code = [
            "import torch",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'model_id = "{repo_id}"',
            "tokenizer = AutoTokenizer.from_pretrained(model_id)",
            "model = AutoModelForCausalLM.from_pretrained(",
            "    model_id,",
            "    torch_dtype=torch.bfloat16,",
            '    device_map="auto",',
            ")",
            "",
            'prompt = "<|im_start|>user\\nWrite a quicksort function in Python.<|im_end|>\\n<|im_start|>assistant\\n"',
            'inputs = tokenizer(prompt, return_tensors="pt").to(model.device)',
            "outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.9)",
            "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        ]

    # 3. Assemble Body
    body_lines = [
        f"# {repo_id}",
        "",
        stage_desc,
        "",
        "## Architecture Highlights",
        f"- **Parameters:** ~{param_display} ({n_layers} layers, hidden dimension `d_model = {d_model}`)",
        f"- **Attention:** {attn_desc}",
        f"- **Activation:** SwiGLU Feed-Forward Network (`d_ffn = {d_ffn}`)",
        "- **Positional Encoding:** Rotary Position Embeddings (RoPE)",
        "- **Normalization:** Bias-free Pre-LayerNorm / RMSNorm",
        "- **Weight Tying:** Tied input embedding (`embed_tokens`) and output head (`lm_head`)",
        "- **Hugging Face Native:** Directly compatible with `LlamaForCausalLM`",
        "",
        "## Pre-training Curriculum",
        f"The base representation was pre-trained across an interleaved multi-source domain mixture ({token_str} tokens):",
        "",
    ]

    for source_name in datasets_list:
        body_lines.append(f"- `{source_name}`")

    body_lines.extend([""] + lineage_section)

    # 4. Optional Dynamic Benchmarks Table
    if eval_metrics:
        body_lines.extend([
            "",
            "## Evaluation Benchmarks",
            "Evaluated natively with length-normalized log-likelihoods and process-isolated execution:",
            "",
            "| Benchmark | Metric | Result |",
            "| :--- | :--- | :---: |",
        ])
        if "val_loss" in eval_metrics and "val_ppl" in eval_metrics:
            body_lines.append(f"| **Validation Loss / PPL** | Cross-Entropy / Perplexity | **{eval_metrics['val_loss']} / {eval_metrics['val_ppl']}** |")
        if "hellaswag_acc" in eval_metrics:
            body_lines.append(f"| **HellaSwag** | Accuracy (0-shot) | **{eval_metrics['hellaswag_acc']}%** |")
        if "arc_easy_acc" in eval_metrics:
            body_lines.append(f"| **ARC-Easy** | Accuracy (0-shot) | **{eval_metrics['arc_easy_acc']}%** |")
        if "think_adherence" in eval_metrics:
            body_lines.append(f"| **GSM8K** | `<think>` Tag Adherence | **{eval_metrics['think_adherence']}%** |")
        if "gsm8k_acc" in eval_metrics:
            body_lines.append(f"| **GSM8K** | Math Accuracy | **{eval_metrics['gsm8k_acc']}%** |")
        if "syntax_validity" in eval_metrics:
            body_lines.append(f"| **MBPP** | Syntax Validity | **{eval_metrics['syntax_validity']}%** |")
        if "mbpp_pass@1" in eval_metrics:
            body_lines.append(f"| **MBPP** | Pass@1 | **{eval_metrics['mbpp_pass@1']}%** |")
        if "dpo_pref_acc" in eval_metrics:
            body_lines.append(f"| **DPO Alignment** | Preference Accuracy | **{eval_metrics['dpo_pref_acc']}%** |")
        if "mean_margin" in eval_metrics:
            body_lines.append(f"| **DPO Alignment** | Mean Margin | **{eval_metrics['mean_margin']}** |")

    body_lines.extend(
        [""]
        + prompt_section
        + ["", "## Quick Start via `transformers`", "", "```python"]
        + quickstart_code
        + [
            "```",
            "",
            "*Trained and published autonomously via [slm-gpt](https://github.com).*",
            "",
        ]
    )

    return "\n".join(yaml_lines + body_lines)


def publish_to_hub(
    checkpoint_path: str,
    repo_id: str,
    export_dir: str = "hf_export",
    token: Optional[str] = None,
    private: bool = False,
    eval_metrics: Optional[Dict[str, Any]] = None,
    tokenizer_type: str = "tiktoken",
    tokenizer_path: Optional[str] = None,
    dataset_tags: Optional[List[str]] = None,
    curriculum_datasets: Optional[List[str]] = None,
):
    api_token = token or os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("Missing HF_TOKEN. Set it in .env or pass token=...")

    api = HfApi(token=api_token)

    print("=" * 65)
    print(f"--- Exporting Checkpoint for Hugging Face Hub: {repo_id} ---")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt_data.get("runtime_meta", {}) if isinstance(ckpt_data, dict) else {}
    raw_cfg = ckpt_data.get("config") if isinstance(ckpt_data, dict) else None

    if isinstance(raw_cfg, ModelConfig):
        cfg = raw_cfg
    elif isinstance(raw_cfg, dict):
        cfg = ModelConfig(**raw_cfg)
    else:
        cfg = None

    model_state = (
        ckpt_data.get("model_state_dict")
        or ckpt_data.get("model")
        or (ckpt_data if isinstance(ckpt_data, dict) and "state_dict" not in ckpt_data else ckpt_data.get("state_dict", {}))
    )

    # Deduplicate tied output head / embedding weights
    has_wte = any(k.endswith("wte.weight") or k.endswith("embed_tokens.weight") for k in model_state.keys())
    num_params = 0
    if isinstance(model_state, dict):
        for k, v in model_state.items():
            if has_wte and (k.endswith("lm_head.weight") or k == "lm_head.weight"):
                continue
            if hasattr(v, "numel"):
                num_params += v.numel()

    # Auto-load eval_results.json if located in the checkpoint directory
    metrics = eval_metrics
    if metrics is None:
        eval_path = Path(checkpoint_path).parent / "eval_results.json"
        if eval_path.exists():
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                print(f"✓ Automatically loaded eval metrics from: {eval_path}")
            except Exception:
                pass

    # 1. Convert to SafeTensors & HF Artifacts
    convert_checkpoint_to_hf(
        checkpoint_path=checkpoint_path,
        output_dir=export_dir,
        tokenizer_type=tokenizer_type,
        tokenizer_path=tokenizer_path,
    )

    # 2. Generate Dynamic Model Card
    readme_content = generate_model_card(
        repo_id=repo_id,
        config=cfg,
        num_params=num_params,
        checkpoint_meta=meta,
        eval_metrics=metrics,
        dataset_tags=dataset_tags,
        curriculum_datasets=curriculum_datasets,
    )

    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"✓ Generated Dynamic Model Card: {export_dir}/README.md")

    # 3. Upload to Hugging Face Hub
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