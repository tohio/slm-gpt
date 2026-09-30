"""
data/datacard.py: Dynamic Hugging Face Dataset Card Generator.
Adapts metadata and documentation for both tiktoken and CustomBPETokenizer.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union


def _get_size_tag(tokens: int) -> str:
    if tokens < 1_000_000_000:
        return "n<1B"
    if tokens < 10_000_000_000:
        return "1B<n<10B"
    if tokens < 100_000_000_000:
        return "10B<n<100B"
    return "n>100B"


def _build_curriculum_table(source_stats: Dict[str, Dict[str, Any]], total_tokens: int) -> str:
    lines = [
        "| Source | Upstream Repo | Packed Tokens | Share | Repetition / Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for name, data in source_stats.items():
        count = data.get("tokens", 0)
        pct = (count / total_tokens * 100.0) if total_tokens > 0 else 0.0
        repo = data.get("upstream", "Custom")
        status = data.get("status", "100% Unique")
        lines.append(f"| **{name}** | `{repo}` | {count:,} | {pct:.2f}% | {status} |")
    return "\n".join(lines)


def _build_special_tokens_table(special_tokens: Dict[str, int]) -> str:
    lines = [
        "| Special Token | Token ID | Function / Role |",
        "| :--- | :--- | :--- |",
    ]
    role_map = {
        "<|endoftext|>": "Document Delimiter / EOT",
        "<|im_start|>": "ChatML Turn Start",
        "<|im_end|>": "ChatML Turn End",
        "<|pad|>": "Padding Token",
        "<s>": "Sequence Start",
        "</s>": "Sequence End",
        "<|begin_of_text|>": "Beginning of Text",
        "<think>": "Reasoning / CoT Block Start",
        "</think>": "Reasoning / CoT Block End",
        "<tool_call>": "Tool Call Invocation",
        "</tool_call>": "Tool Call Close",
        "<tool_response>": "Tool Execution Response",
        "</tool_response>": "Tool Execution Close",
        "<|fim_prefix|>": "Fill-in-the-Middle Prefix",
        "<|fim_middle|>": "Fill-in-the-Middle Middle",
        "<|fim_suffix|>": "Fill-in-the-Middle Suffix",
        "<|fim_hole|>": "Fill-in-the-Middle Hole",
    }
    for tok, idx in sorted(special_tokens.items(), key=lambda kv: kv[1]):
        role = role_map.get(tok, "Special Token")
        lines.append(f"| `{tok}` | `{idx}` | {role} |")
    return "\n".join(lines)


def create_dataset_card(
    output_dir: Union[str, Path],
    dataset_name: str,
    total_tokens: int,
    shard_count: int,
    tokens_per_shard: int,
    source_stats: Dict[str, Dict[str, Any]],
    val_tokens: int = 500_000,
    tokenizer_type: str = "tiktoken",
    vocab_size: int = 50257,
    eot_token_id: int = 50256,
    special_tokens: Optional[Dict[str, int]] = None,
) -> Path:
    """
    Generates a dynamically tailored Hugging Face README.md dataset card.
    Correctly documents tiktoken vs CustomBPETokenizer contracts and enforces viewer: false.
    """
    out_path = Path(output_dir) / "README.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size_tag = _get_size_tag(total_tokens)
    total_gb = (total_tokens * 2) / (1024**3)
    shard_mb = (tokens_per_shard * 2) / (1024**2)
    curriculum_table = _build_curriculum_table(source_stats, total_tokens)

    reasoning_keys = {"NuminaMath-CoT", "OpenMathReasoning", "SLM-Synthetic-Pretrain", "SLM-Synthetic"}
    reasoning_tokens = sum(
        v.get("tokens", 0) for k, v in source_stats.items() if k in reasoning_keys
    )
    reasoning_pct = (reasoning_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0

    is_custom_bpe = "bpe" in tokenizer_type.lower()
    if is_custom_bpe:
        tok_desc = f"`CustomBPETokenizer` (slm-gpt extended BPE, vocab size {vocab_size:,})"
    else:
        tok_desc = f"`tiktoken` (GPT-2 BPE, vocab size {vocab_size:,})"

    q = "```"

    lines = [
        "---",
        "language:",
        "- en",
        "license: other",
        f"pretty_name: {dataset_name}",
        "size_categories:",
        f"- {size_tag}",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- pretraining",
        "- slm",
        "- curriculum",
        "- uint16",
        "- memmap",
        "viewer: false",
        "---",
        "",
        f"# {dataset_name}",
        "",
        "An interleaved binary pretraining curriculum built for Small Language Models (SLMs).",
        "",
        "The corpus is tokenized, zero-padded, and packed into contiguous 16-bit unsigned integer (`uint16`) binary files, optimized for zero-copy memory mapping (`np.memmap`) across distributed GPU training workers.",
        "",
        "## Dataset Specifications",
        "",
        f"- **Total Packed Tokens:** {total_tokens:,} ({total_tokens / 1e9:.2f}B)",
        f"- **Total Payload Size:** {total_gb:.2f} GB (`uint16`, 2 bytes/token)",
        f"- **Training Shards:** {shard_count:,} files ({tokens_per_shard:,} tokens / {shard_mb:.1f} MB each)",
        f"- **Validation Tokens:** {val_tokens:,} (`val_00000.bin`)",
        f"- **Tokenizer Engine:** {tok_desc}",
        f"- **Delimiter Token ID:** `{eot_token_id}` (`<|endoftext|>`)",
        f"- **Effective Reasoning Density:** ~{reasoning_pct:.2f}%",
        "",
        "---",
        "",
        "## Interleaved Curriculum Breakdown",
        "",
        curriculum_table,
        "",
    ]

    # If Custom BPE, document the special tokens contract
    if is_custom_bpe and special_tokens:
        lines.extend([
            "---",
            "",
            "## Tokenizer Special Tokens Contract",
            "",
            "This curriculum uses the `slm-gpt` extended vocabulary with native support for ChatML, reasoning tags, and code infilling:",
            "",
            _build_special_tokens_table(special_tokens),
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Direct Consumption (PyTorch Zero-Copy)",
        "",
        f"{q}python",
        "import numpy as np",
        "import torch",
        "",
        "# Zero-copy memory map of a shard",
        'shard = np.memmap("data/pretrain/train_00000.bin", dtype=np.uint16, mode="r")',
        "",
        "# Direct slice into PyTorch tensor",
        "seq_len = 2048",
        "batch = torch.from_numpy(shard[:seq_len].astype(np.int64))",
        f"{q}",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Dynamic dataset card written to: {out_path}")
    return out_path