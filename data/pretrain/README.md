---
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
- 10B<n<100B
configs:
- config_name: default
  data_files:
  - split: train
    path: "pretrain/train_*.bin"
---

# tohio/slm-curriculum-10b

Curriculum pretraining corpus containing **10B tokens** (10,000,000,000 tokens) optimized for Small Language Models (SLMs). Engineered specifically for high overtraining density on the **125M** architecture (~80:1 token-to-parameter ratio).

## 1. Overview & Dataset Architecture

The dataset is serialized as high-throughput, raw binary token shards (`uint16`) packed sequentially to maximize GPU dataloader bandwidth and eliminate runtime tokenization overhead.

* **Total Token Count:** 10,000,000,000
* **Token Encoding:** `uint16` (2 bytes/token, zero-copy `np.memmap` compatible)
* **Shard Size:** 25,000,000 tokens (~50 MB per shard)
* **Total Shards:** 400 shards (`train_00000.bin` to `train_00399.bin`)
* **Tokenizer:** Tiktoken GPT-2 BPE (`vocab_size = 50,304` with alignment padding)
* **Target Architecture:** 125M Decoder-Only Transformer (GQA, RoPE, SwiGLU, RMSNorm)

---

## 2. Curriculum Composition & Repetition Caps

| Domain | Source / Pipeline | Share (%) | Repetition Ceiling | Description |
| :--- | :--- | :---: | :---: | :--- |
| **General Web & Knowledge** | `FineWeb-Edu` + `DCLM-Edu` | 50.5% | 1.0x (Infinite stream) | Syntactically and educationally filtered expository text. |
| **Clean Repository Code** | `The-Stack-Edu` / Python Repositories | 25.0% | $\le$ 1.5x | AST-validated, parseable Python code covering algorithms, typing, and standard libraries. |
| **Reasoning & Math** | OpenMath / Synthetic Reasoning | 15.0% | $\le$ 2.0x | Step-by-step verified arithmetic, algebraic proofs, and execution traces. |
| **Synthetic Anchor** | `tohio/slm-synthetic-pretrain` | 9.5% | $\le$ 2.0x | 5-signal dense anchor regulating high-entropy failure modes. |

### 5-Signal Synthetic Anchor Distribution
The 9.5% synthetic anchor enforces structural convergence across five targeted signals:
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

```python
import numpy as np
import torch

def load_shard(shard_path: str) -> torch.Tensor:
    # 25,000,000 uint16 tokens = exactly 50,000,000 bytes
    tokens_np = np.memmap(shard_path, dtype=np.uint16, mode="r")
    return torch.from_numpy(tokens_np.astype(np.int64))

# Example: Read shard 0
tokens = load_shard("data/pretrain/train_00000.bin")
print(f"Loaded shard: {tokens.shape[0]:,} tokens")
```

## Citation & Metadata
```bibtex
@dataset{slm_curriculum_10b,
  author = {Ohiokpehai, Tjani},
  title = {tohio/slm-curriculum-10b: Packed Curriculum Pretraining Corpus for Small Language Models},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/tohio/slm-curriculum-10b}
}
```
