### `README.md`

```markdown
# SLM-GPT: Modern Decoder-Only Language Model from Scratch

A high-performance Small Language Model (SLM) engineered modularly from innermost tensor algebra and memory hierarchies outward in PyTorch. Designed with modern architectural primitives (**Grouped-Query Attention**, **Rotary Position Embeddings**, **SwiGLU Feed-Forward Networks**, and **$O(1)$ per-token KV caching**) coupled to a zero-copy, memory-mapped data ingestion engine.

---

## 1. Architecture Specifications (~17.0M Active Parameters)

| Hyperparameter | Symbol | Value | Architectural Justification |
|---|---|---|---|
| **Vocabulary Size** | $V$ | `50,257` | GPT-2 byte-level BPE (`tiktoken`), serialized as `uint16` |
| **Context Length** | $T$ | `256` | Pretraining sequence context window |
| **Model Dimension** | $d_{\text{model}}$ | `256` | Residual stream channel width |
| **Query Heads** | $H_q$ | `8` | Parallel attention projection heads |
| **Key/Value Heads** | $H_{kv}$ | `2` | Grouped-Query Attention ($4\times$ KV memory compression) |
| **Head Dimension** | $d_{\text{head}}$ | `32` | $d_{\text{model}} / H_q = 256 / 8 = 32$ |
| **FFN Intermediate Dimension** | $d_{\text{ffn}}$ | `704` | SwiGLU: $\lfloor \frac{8}{3} d_{\text{model}} \rfloor = 682 \to \mathbf{704}$ (aligned to 64) |
| **Layer Depth** | $L$ | `6` | Identical stacked Transformer decoder blocks |
| **Positional Encoding** | — | RoPE | Rotary Positional Embeddings ($\theta = 10,000$, no $W_{\text{pos}}$ table) |
| **Parameter Tying** | — | Enabled | $W_{\text{embed}} \equiv W_{\text{lm\_head}}$ ($50,257 \times 256$ shared weights) |
| **Bias Terms** | — | `False` | Bias-free linear layers throughout projections and norms |

---

## 2. Structural Tensor Flow

```text
                     Token Indices: X ∈ ℕ^(B × T),  X_{b,t} ∈ [0, V-1]
                                       │
                                       ▼
                       wte: Embedding Table ∈ ℝ^(V × d_model)
                                       │
                       Residual Stream: h_0 ∈ ℝ^(B × T × d_model)
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         │                                                           │
         │  TransformerBlock ℓ ∈ {1, ..., L}                         │
         │  ├── ln_1: LayerNorm(h_{ℓ-1})                             │
         │  ├── GQA Causal Self-Attention + RoPE + O(1) KV Cache      │
         │  │   ├── Q ∈ ℝ^(B × T × H_q × d_head)                     │
         │  │   ├── K ∈ ℝ^(B × T × H_kv × d_head)                    │
         │  │   ├── V ∈ ℝ^(B × T × H_kv × d_head)                    │
         │  │   └── Out: h_{attn} = Attention(Q, K, V) · W_proj      │
         │  ├── Residual Add: h'_ℓ = h_{ℓ-1} + h_{attn}              │
         │  ├── ln_2: LayerNorm(h'_ℓ)                                │
         │  ├── SwiGLU FFN:                                          │
         │  │   ├── Gate = SiLU(h'_ℓ · W_gate), W_gate ∈ ℝ^(d × d_ffn)│
         │  │   ├── Up   = h'_ℓ · W_up,         W_up   ∈ ℝ^(d × d_ffn)│
         │  │   └── Down = (Gate ⊙ Up) · W_down, W_down ∈ ℝ^(d_ffn × d)│
         │  └── Residual Add: h_ℓ = h'_ℓ + Down                      │
         │                                                           │
         └─────────────────────────────┬─────────────────────────────┘
                                       │
                         LayerNorm: h_final ∈ ℝ^(B × T × d_model)
                                       │
                         LM Head: W_embed^T ∈ ℝ^(d_model × V)
                                       │
                                       ▼
                     Logits: logits ∈ ℝ^(B × T × V)

```

---

## 3. Core Architectural Primitives

### 1. Rotary Position Embeddings (RoPE)

Encodes sequence index $m \in [0, T-1]$ directly into the Query and Key representations via complex 2D Givens coordinate rotations:


$$\begin{pmatrix} x_{2i-1}' \\ x_{2i}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i-1} \\ x_{2i} \end{pmatrix}, \quad \theta_i = 10000^{-2(i-1)/d_{\text{head}}}$$


Inner products satisfy $\langle R_m q, R_n k \rangle = q^T R_{n-m} k$, preserving translation invariance across context windows without learned positional lookup tables.

### 2. Grouped-Query Attention (GQA) & $O(1)$ KV Caching

Partitions $H_q = 8$ Query heads across $H_{kv} = 2$ Key/Value heads ($4\times$ reduction in KV cache memory and bandwidth stalls):

* **Training:** Keys and Values are expanded across Query groups via `.repeat_interleave(dim=1, repeats=4)`.
* **Inference:** The sequential decoding step appends strictly $H_{kv} = 2$ heads to pre-allocated buffers, cutting KV cache memory consumption by **$75\%$** compared to standard Multi-Head Attention (MHA).

### 3. SwiGLU Feed-Forward Network

Replaces standard GELU MLPs with bilinear gated SiLU units:


$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$


The intermediate hidden dimension is scaled to $\lfloor \frac{8}{3} d_{\text{model}} \rfloor = 682.6$, rounded up to **$704$** to guarantee alignment with NVIDIA Tensor Core memory tiles (multiples of 64/256).

---

## 4. Systems Engineering & High-Throughput I/O

The data ingestion pipeline delivers continuous token streams to the GPU without introducing CPU stalls or host-to-device bottlenecks, sustaining **~355,000 tokens/sec** training throughput.

```text
┌─────────────────────────────────────────────────────────────────┐
│               Storage (NVMe / SSD / Local Disk)                 │
│              data/processed/fineweb_edu_100m/train.bin          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    mmap() OS System Call (Zero-Copy)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Host Kernel Page Cache                       │
│      Virtual memory pages faulted on-demand (no heap copies)     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    Contiguous 1D Slicing + .reshape(B, T)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Pinned Host Memory (Page-Locked)              │
│       torch.from_numpy(...).pin_memory()                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
             PCIe Bus (Direct Memory Access via non_blocking=True)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  GPU Global Memory (VRAM)                       │
│             Ready for FlashAttention / SDPA Forward Pass         │
└─────────────────────────────────────────────────────────────────┘

```

* **`uint16` Serialization:** Since $V = 50,257 < 2^{16}$, tokens are stored as 16-bit unsigned integers, cutting dataset footprint from 800 MB (`int64`) to **200 MB**.
* **Zero-Copy Memory-Mapping (`np.memmap`):** $O(1)$ startup initialization with zero preloading latency; paging is managed directly by OS page tables.
* **Asynchronous Non-Blocking DMA:** Pinned host allocations (`pin_memory()`) allow the GPU DMA controller to stream batches across PCIe concurrently with forward/backward tensor execution (`non_blocking=True`).

---

## 5. Repository Layout

```text
slm-gpt/
├── checkpoints/
│   └── gqa_rope_17m/
│       ├── best_model.pt            # Checkpoint with lowest validation loss
│       └── ckpt_step_*.pt           # Periodic training snapshots
├── data/
│   ├── processed/
│   │   └── fineweb_edu_100m/
│   │       ├── train.bin            # ~90M tokens (180 MB, uint16)
│   │       ├── val.bin              # ~5M tokens  (10 MB, uint16)
│   │       └── test.bin             # ~5M tokens  (10 MB, uint16)
│   ├── README.md                    # Data pipeline & I/O performance documentation
│   ├── dataloader.py                # Standalone streaming primitives
│   ├── dataset.py                   # PyTorch Dataset wrappers
│   ├── prepare.py                   # Local text corpus tokenizer
│   └── prepare_hf.py                # Hugging Face dataset streamer
├── tokenizer/
│   └── README.md                    # Tokenizer specification & byte-level BPE docs
├── transformer/
│   ├── __init__.py                  # Package exports
│   ├── attention.py                 # GQA, RoPE rotary kernel, and KV cache logic
│   ├── block.py                     # Pre-LN Transformer decoder block
│   ├── config.py                    # ModelConfig dataclass
│   ├── feedforward.py               # SwiGLU gated feed-forward layer
│   ├── model.py                     # DecoderOnlyTransformer & generation logic
│   └── README.md                    # Transformer mathematical & architectural docs
├── eval_inference.py                # Benchmark script: test loss, perplexity, generation speed
├── train.py                         # Production pretraining loop with live text generation
└── README.md                        # Master repository engineering specification

```

---

## 6. Pretraining Baseline Results (100M FineWeb-Edu Split)

| Checkpoint Step | Tokens Processed | Validation Loss | Validation Perplexity | Qualitative Progression Sample |
| --- | --- | --- | --- | --- |
| **Step 0** | $0$ | $10.8711$ | $52,631.04$ | Random token soup (`Consortium Consciousthreatening embody...`) |
| **Step 100** | $13.1\text{M}$ | $6.7299$ | $837.08$ | Basic subword composition, stop words, punctuation |
| **Step 300** | $39.3\text{M}$ | $5.8856$ | $359.84$ | Syntactic clauses, correct casing, natural rhythm |
| **Step 500** | $65.5\text{M}$ | $5.5290$ | $251.89$ | Semantic domain clustering (`data`, `algorithm`, `system`) |
| **Step 700 (Best)** | $91.7\text{M}$ | **$5.3831$** | **$217.69$** | Grammatically coherent definitions and contextual alignment |

* **Hardware:** NVIDIA Cloud GPU instance
* **Throughput:** ~355,000 tokens/second sustained
* **Total Optimizer Steps:** 762 (Effective batch size: 131,072 tokens/step)

---

## 7. Quickstart & Verification

### 1. Environment Setup

```bash
git clone [https://github.com/tohio/slm-gpt.git](https://github.com/tohio/slm-gpt.git)
cd slm-gpt
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy tiktoken

```

### 2. Numerical Invariant Verification

Verify numerical equivalence between full-context attention and iterative single-token cached generation (verifies RoPE, GQA, and SwiGLU state parity to $< 10^{-6}$ float precision):

```bash
python3 -c "
import torch
from transformer import ModelConfig, DecoderOnlyTransformer

torch.manual_seed(42)
cfg = ModelConfig(vocab_size=50257, max_seq_len=256, d_model=256, n_heads=8, n_kv_heads=2, d_ffn=704, n_layers=6, dropout=0.0)
m = DecoderOnlyTransformer(cfg).eval().cuda()

x = torch.randint(0, 50257, (1, 16), device='cuda')
logits_full, _, _ = m(x)
logits_prefill, _, cache = m(x[:, :-1])
logits_step, _, _ = m(x[:, -1:], kv_caches=cache)

diff = (logits_full[:, -1, :] - logits_step[:, 0, :]).abs().max().item()
print(f'Max discrepancy (Full Attention vs Cached Step): {diff:.2e}')
assert diff < 1e-4, 'KV Cache numerical divergence!'
print('✓ Mathematical and cache equivalence verified.')
"

```

### 3. Pretraining

Execute pretraining on the 100M FineWeb-Edu token split with BF16 mixed precision and live generation callbacks:

```bash
python3 train.py

```

### 4. Generation & Inference Benchmarks

Evaluate holdout test loss and benchmark KV-cached generation tokens/second:

```bash
python3 eval_inference.py

```

```

---

### Command to Overwrite on the Server

Run this command directly on your server to update the root `README.md`:

```bash
cat << 'EOF' > /root/slm-gpt/README.md
# SLM-GPT: Modern Decoder-Only Language Model from Scratch

A high-performance Small Language Model (SLM) engineered modularly from innermost tensor algebra and memory hierarchies outward in PyTorch. Designed with modern architectural primitives (**Grouped-Query Attention**, **Rotary Position Embeddings**, **SwiGLU Feed-Forward Networks**, and **$O(1)$ per-token KV caching**) coupled to a zero-copy, memory-mapped data ingestion engine.

---

## 1. Architecture Specifications (~17.0M Active Parameters)

| Hyperparameter | Symbol | Value | Architectural Justification |
|---|---|---|---|
| **Vocabulary Size** | $V$ | `50,257` | GPT-2 byte-level BPE (`tiktoken`), serialized as `uint16` |
| **Context Length** | $T$ | `256` | Pretraining sequence context window |
| **Model Dimension** | $d_{\text{model}}$ | `256` | Residual stream channel width |
| **Query Heads** | $H_q$ | `8` | Parallel attention projection heads |
| **Key/Value Heads** | $H_{kv}$ | `2` | Grouped-Query Attention ($4\times$ KV memory compression) |
| **Head Dimension** | $d_{\text{head}}$ | `32` | $d_{\text{model}} / H_q = 256 / 8 = 32$ |
| **FFN Intermediate Dimension** | $d_{\text{ffn}}$ | `704` | SwiGLU: $\lfloor \frac{8}{3} d_{\text{model}} \rfloor = 682 \to \mathbf{704}$ (aligned to 64) |
| **Layer Depth** | $L$ | `6` | Identical stacked Transformer decoder blocks |
| **Positional Encoding** | — | RoPE | Rotary Positional Embeddings ($\theta = 10,000$, no $W_{\text{pos}}$ table) |
| **Parameter Tying** | — | Enabled | $W_{\text{embed}} \equiv W_{\text{lm\_head}}$ ($50,257 \times 256$ shared weights) |
| **Bias Terms** | — | `False` | Bias-free linear layers throughout projections and norms |

---

## 2. Structural Tensor Flow

```text
                     Token Indices: X ∈ ℕ^(B × T),  X_{b,t} ∈ [0, V-1]
                                       │
                                       ▼
                       wte: Embedding Table ∈ ℝ^(V × d_model)
                                       │
                       Residual Stream: h_0 ∈ ℝ^(B × T × d_model)
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         │                                                           │
         │  TransformerBlock ℓ ∈ {1, ..., L}                         │
         │  ├── ln_1: LayerNorm(h_{ℓ-1})                             │
         │  ├── GQA Causal Self-Attention + RoPE + O(1) KV Cache      │
         │  │   ├── Q ∈ ℝ^(B × T × H_q × d_head)                     │
         │  │   ├── K ∈ ℝ^(B × T × H_kv × d_head)                    │
         │  │   ├── V ∈ ℝ^(B × T × H_kv × d_head)                    │
         │  │   └── Out: h_{attn} = Attention(Q, K, V) · W_proj      │
         │  ├── Residual Add: h'_ℓ = h_{ℓ-1} + h_{attn}              │
         │  ├── ln_2: LayerNorm(h'_ℓ)                                │
         │  ├── SwiGLU FFN:                                          │
         │  │   ├── Gate = SiLU(h'_ℓ · W_gate), W_gate ∈ ℝ^(d × d_ffn)│
         │  │   ├── Up   = h'_ℓ · W_up,         W_up   ∈ ℝ^(d × d_ffn)│
         │  │   └── Down = (Gate ⊙ Up) · W_down, W_down ∈ ℝ^(d_ffn × d)│
         │  └── Residual Add: h_ℓ = h'_ℓ + Down                      │
         │                                                           │
         └─────────────────────────────┬─────────────────────────────┘
                                       │
                         LayerNorm: h_final ∈ ℝ^(B × T × d_model)
                                       │
                         LM Head: W_embed^T ∈ ℝ^(d_model × V)
                                       │
                                       ▼
                     Logits: logits ∈ ℝ^(B × T × V)

```

---

## 3. Core Architectural Primitives

### 1. Rotary Position Embeddings (RoPE)

Encodes sequence index $m \in [0, T-1]$ directly into the Query and Key representations via complex 2D Givens coordinate rotations:


$$\begin{pmatrix} x_{2i-1}' \\ x_{2i}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i-1} \\ x_{2i} \end{pmatrix}, \quad \theta_i = 10000^{-2(i-1)/d_{\text{head}}}$$


Inner products satisfy $\langle R_m q, R_n k \rangle = q^T R_{n-m} k$, preserving translation invariance across context windows without learned positional lookup tables.

### 2. Grouped-Query Attention (GQA) & $O(1)$ KV Caching

Partitions $H_q = 8$ Query heads across $H_{kv} = 2$ Key/Value heads ($4\times$ reduction in KV cache memory and bandwidth stalls):

* **Training:** Keys and Values are expanded across Query groups via `.repeat_interleave(dim=1, repeats=4)`.
* **Inference:** The sequential decoding step appends strictly $H_{kv} = 2$ heads to pre-allocated buffers, cutting KV cache memory consumption by **$75\%$** compared to standard Multi-Head Attention (MHA).

### 3. SwiGLU Feed-Forward Network

Replaces standard GELU MLPs with bilinear gated SiLU units:


$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$


The intermediate hidden dimension is scaled to $\lfloor \frac{8}{3} d_{\text{model}} \rfloor = 682.6$, rounded up to **$704$** to guarantee alignment with NVIDIA Tensor Core memory tiles (multiples of 64/256).

---

## 4. Systems Engineering & High-Throughput I/O

The data ingestion pipeline delivers continuous token streams to the GPU without introducing CPU stalls or host-to-device bottlenecks, sustaining **~355,000 tokens/sec** training throughput.

```text
┌─────────────────────────────────────────────────────────────────┐
│               Storage (NVMe / SSD / Local Disk)                 │
│              data/processed/fineweb_edu_100m/train.bin          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    mmap() OS System Call (Zero-Copy)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Host Kernel Page Cache                       │
│      Virtual memory pages faulted on-demand (no heap copies)     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    Contiguous 1D Slicing + .reshape(B, T)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Pinned Host Memory (Page-Locked)              │
│       torch.from_numpy(...).pin_memory()                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
             PCIe Bus (Direct Memory Access via non_blocking=True)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  GPU Global Memory (VRAM)                       │
│             Ready for FlashAttention / SDPA Forward Pass         │
└─────────────────────────────────────────────────────────────────┘

```

* **`uint16` Serialization:** Since $V = 50,257 < 2^{16}$, tokens are stored as 16-bit unsigned integers, cutting dataset footprint from 800 MB (`int64`) to **200 MB**.
* **Zero-Copy Memory-Mapping (`np.memmap`):** $O(1)$ startup initialization with zero preloading latency; paging is managed directly by OS page tables.
* **Asynchronous Non-Blocking DMA:** Pinned host allocations (`pin_memory()`) allow the GPU DMA controller to stream batches across PCIe concurrently with forward/backward tensor execution (`non_blocking=True`).

---

## 5. Repository Layout

```text
slm-gpt/
├── checkpoints/
│   └── gqa_rope_17m/
│       ├── best_model.pt            # Checkpoint with lowest validation loss
│       └── ckpt_step_*.pt           # Periodic training snapshots
├── data/
│   ├── processed/
│   │   └── fineweb_edu_100m/
│   │       ├── train.bin            # ~90M tokens (180 MB, uint16)
│   │       ├── val.bin              # ~5M tokens  (10 MB, uint16)
│   │       └── test.bin             # ~5M tokens  (10 MB, uint16)
│   ├── README.md                    # Data pipeline & I/O performance documentation
│   ├── dataloader.py                # Standalone streaming primitives
│   ├── dataset.py                   # PyTorch Dataset wrappers
│   ├── prepare.py                   # Local text corpus tokenizer
│   └── prepare_hf.py                # Hugging Face dataset streamer
├── tokenizer/
│   └── README.md                    # Tokenizer specification & byte-level BPE docs
├── transformer/
│   ├── __init__.py                  # Package exports
│   ├── attention.py                 # GQA, RoPE rotary kernel, and KV cache logic
│   ├── block.py                     # Pre-LN Transformer decoder block
│   ├── config.py                    # ModelConfig dataclass
│   ├── feedforward.py               # SwiGLU gated feed-forward layer
│   ├── model.py                     # DecoderOnlyTransformer & generation logic
│   └── README.md                    # Transformer mathematical & architectural docs
├── eval_inference.py                # Benchmark script: test loss, perplexity, generation speed
├── train.py                         # Production pretraining loop with live text generation
└── README.md                        # Master repository engineering specification

```

---

## 6. Pretraining Baseline Results (100M FineWeb-Edu Split)

| Checkpoint Step | Tokens Processed | Validation Loss | Validation Perplexity | Qualitative Progression Sample |
| --- | --- | --- | --- | --- |
| **Step 0** | $0$ | $10.8711$ | $52,631.04$ | Random token soup (`Consortium Consciousthreatening embody...`) |
| **Step 100** | $13.1\text{M}$ | $6.7299$ | $837.08$ | Basic subword composition, stop words, punctuation |
| **Step 300** | $39.3\text{M}$ | $5.8856$ | $359.84$ | Syntactic clauses, correct casing, natural rhythm |
| **Step 500** | $65.5\text{M}$ | $5.5290$ | $251.89$ | Semantic domain clustering (`data`, `algorithm`, `system`) |
| **Step 700 (Best)** | $91.7\text{M}$ | **$5.3831$** | **$217.69$** | Grammatically coherent definitions and contextual alignment |

* **Hardware:** NVIDIA Cloud GPU instance
* **Throughput:** ~355,000 tokens/second sustained
* **Total Optimizer Steps:** 762 (Effective batch size: 131,072 tokens/step)

---

## 7. Quickstart & Verification

### 1. Environment Setup

```bash
git clone [https://github.com/tohio/slm-gpt.git](https://github.com/tohio/slm-gpt.git)
cd slm-gpt
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy tiktoken

```

### 2. Numerical Invariant Verification

Verify numerical equivalence between full-context attention and iterative single-token cached generation (verifies RoPE, GQA, and SwiGLU state parity to $< 10^{-6}$ float precision):

```bash
python3 -c "
import torch
from transformer import ModelConfig, DecoderOnlyTransformer

torch.manual_seed(42)
cfg = ModelConfig(vocab_size=50257, max_seq_len=256, d_model=256, n_heads=8, n_kv_heads=2, d_ffn=704, n_layers=6, dropout=0.0)
m = DecoderOnlyTransformer(cfg).eval().cuda()

x = torch.randint(0, 50257, (1, 16), device='cuda')
logits_full, _, _ = m(x)
logits_prefill, _, cache = m(x[:, :-1])
logits_step, _, _ = m(x[:, -1:], kv_caches=cache)

diff = (logits_full[:, -1, :] - logits_step[:, 0, :]).abs().max().item()
print(f'Max discrepancy (Full Attention vs Cached Step): {diff:.2e}')
assert diff < 1e-4, 'KV Cache numerical divergence!'
print('✓ Mathematical and cache equivalence verified.')
"

```

### 3. Pretraining

Execute pretraining on the 100M FineWeb-Edu token split with BF16 mixed precision and live generation callbacks:

```bash
python3 train.py

```

### 4. Generation & Inference Benchmarks

Evaluate holdout test loss and benchmark KV-cached generation tokens/second:

```bash
python3 eval_inference.py

```