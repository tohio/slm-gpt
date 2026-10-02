import argparse
from pathlib import Path


def generate_card(tokens: str = "10B", model_target: str = "125M", output_path: str = "data/pretrain/README.md") -> None:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    is_10b = tokens.upper() == "10B"
    num_shards = 400 if is_10b else 3200
    token_count = "10,000,000,000" if is_10b else "80,000,000,000"
    repo_id = f"tohio/slm-curriculum-{tokens.lower()}"
    size_category = "10B<n<100B"

    anchor_pct = "9.5%" if is_10b else "1.2%"
    code_pct = "25.0%" if is_10b else "28.0%"
    math_pct = "15.0%" if is_10b else "12.0%"
    web_pct = "50.5%" if is_10b else "58.8%"
    last_shard_idx = f"{num_shards - 1:05d}"
    ticks = chr(96) * 3

    card = f"""---
language:
- en
license: mit
viewer: false
task_categories:
- text-generation
tags:
- pretraining
- curriculum
- synthetic
- code
- reasoning
- slm
size_categories:
- {size_category}
configs:
- config_name: default
  data_files:
  - split: train
    path: "pretrain/train_*.bin"
---

# {repo_id}

Curriculum pretraining corpus containing **{tokens.upper()} tokens** ({token_count} tokens) optimized for Small Language Models (SLMs). Engineered specifically for high overtraining density on the **{model_target}** architecture (~80:1 token-to-parameter ratio).

## 1. Overview & Dataset Architecture

The dataset is serialized as high-throughput, raw binary token shards (`uint16`) packed sequentially to maximize GPU dataloader bandwidth and eliminate runtime tokenization overhead.

* **Total Token Count:** {token_count}
* **Token Encoding:** `uint16` (2 bytes/token, zero-copy `np.memmap` compatible)
* **Shard Size:** 25,000,000 tokens (~50 MB per shard)
* **Total Shards:** {num_shards} shards (`train_00000.bin` to `train_{last_shard_idx}.bin`)
* **Tokenizer:** Tiktoken GPT-2 BPE (`vocab_size = 50,304` with alignment padding)
* **Target Architecture:** {model_target} Decoder-Only Transformer (GQA, RoPE, SwiGLU, RMSNorm)

---

## 2. Curriculum Composition & Repetition Caps

| Domain | Source / Pipeline | Share (%) | Repetition Ceiling | Description |
| :--- | :--- | :---: | :---: | :--- |
| **General Web & Knowledge** | `FineWeb-Edu` + `DCLM-Edu` | {web_pct} | 1.0x (Infinite stream) | Syntactically and educationally filtered expository text. |
| **Clean Repository Code** | `The-Stack-Edu` / Python Repositories | {code_pct} | $\\le$ 1.5x | AST-validated, parseable Python code covering algorithms, typing, and standard libraries. |
| **Reasoning & Math** | OpenMath / Synthetic Reasoning | {math_pct} | $\\le$ 2.0x | Step-by-step verified arithmetic, algebraic proofs, and execution traces. |
| **Synthetic Anchor** | `tohio/slm-synthetic-pretrain` | {anchor_pct} | $\\le$ 2.0x | 5-signal dense anchor regulating high-entropy failure modes. |

### 5-Signal Synthetic Anchor Distribution
The {anchor_pct} synthetic anchor enforces structural convergence across five targeted signals:
1. **Task-Driven Code Synthesis (`task_code`):** Production-grade Python functions with docstrings and type annotations.
2. **Runtime State Tracing (`runtime_trace`):** Step-by-step local variable tracking and state mutation reasoning.
3. **Multi-Step Arithmetic (`arithmetic`):** Pure execution of operations without conversational drift.
4. **Formal Math Verification (`math_verify`):** Correctness auditing and counter-example checks.
5. **Factual Restraint & Calibration (`factual_restraint`):** Strict boundary detection against hallucinated technical claims.

---

## 3. Critical Fixes & Data Integrity Notes

* **Parquet Streaming Schema Fix:** Earlier revisions contained a streaming bug that erroneously ingested hexadecimal SHA-1 string IDs (`blob_id`) as text for code segments. This run completely resolves that defect: code is ingested exclusively from validated, parseable Python syntax trees.
* **Special Token Alignment:** Pre-packed with native boundary handling for `<|endoftext|>` and reserved ChatML/FIM slots without sequence truncation corruption.

---

## 4. Dataloader & Streaming Usage

Load shards directly into PyTorch tensors using zero-copy memory mapping (`np.memmap`):

{ticks}python
import numpy as np
import torch

def load_shard(shard_path: str) -> torch.Tensor:
    # 25,000,000 uint16 tokens = exactly 50,000,000 bytes
    tokens_np = np.memmap(shard_path, dtype=np.uint16, mode="r")
    return torch.from_numpy(tokens_np.astype(np.int64))

# Example: Read shard 0
tokens = load_shard("data/pretrain/train_00000.bin")
print(f"Loaded shard: {{tokens.shape[0]:,}} tokens")
{ticks}

## Citation & Metadata
{ticks}bibtex
@dataset{{slm_curriculum_{tokens.lower()},
  author = {{Ohiokpehai, Tjani}},
  title = {{{repo_id}: Packed Curriculum Pretraining Corpus for Small Language Models}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/datasets/{repo_id}}}
}}
{ticks}
"""
    out_file.write_text(card.strip() + "\n", encoding="utf-8")
    print(f"✓ Generated dataset card at: {out_file} ({tokens.upper()}, {model_target})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset card for SLM curriculum datasets.")
    parser.add_argument("--tokens", type=str, default="10B", help="Token volume: 10B or 80B")
    parser.add_argument("--model_target", type=str, default="125M", help="Target architecture: 125M, 350M, 1B")
    parser.add_argument("--output", type=str, default="data/pretrain/README.md", help="Output path")
    args = parser.parse_args()

    generate_card(tokens=args.tokens, model_target=args.model_target, output_path=args.output)
