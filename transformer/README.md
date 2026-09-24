### `transformer/README.md`

```markdown
# Transformer Core Architecture (`transformer/`)

This directory contains the inner tensor algebra, attention kernels, feed-forward sublayers, and block assemblies that comprise the decoder-only language model. The architecture is engineered modularly from scratch in PyTorch, departing from legacy GPT-2 designs to implement modern state-of-the-art primitives: **Rotary Position Embeddings (RoPE)**, **Grouped-Query Attention (GQA)**, **SwiGLU Feed-Forward Networks**, and **$O(1)$ per-token KV caching**.

---

## 1. Architectural Topology & Tensor Transformations

The model processes a batch of token sequences through an embedding layer, $L$ identical Transformer decoder blocks arranged in a Pre-LayerNorm configuration, a final LayerNorm, and a weight-tied projection head.

```text
Input Token IDs: X ∈ ℕ^(B × T),  X_{b,t} ∈ [0, V-1]
                   │
                   ▼
       wte: Token Embedding Table ∈ ℝ^(V × d_model)
                   │
       Residual Stream: h_0 ∈ ℝ^(B × T × d_model)
                   │
    ┌──────────────┴──────────────────────────────────────────────────────┐
    │                                                                     │
    │  TransformerBlock ℓ ∈ {1, 2, ..., L}                                │
    │                                                                     │
    │  1. Pre-Attention Normalization:                                    │
    │     h_norm1 = ln_1(h_{ℓ-1})                       ∈ ℝ^(B × T × d)   │
    │                                                                     │
    │  2. Grouped-Query Attention (GQA) + RoPE:                           │
    │     Q = h_norm1 · W_q                             ∈ ℝ^(B × T × H_q × d_head)
    │     K = h_norm1 · W_k                             ∈ ℝ^(B × T × H_kv × d_head)
    │     V = h_norm1 · W_v                             ∈ ℝ^(B × T × H_kv × d_head)
    │     Q_rot = ApplyRoPE(Q, positions)                                 │
    │     K_rot = ApplyRoPE(K, positions)                                 │
    │     [Update & Fetch O(1) Key-Value Cache if in inference]           │
    │     K_exp = repeat_interleave(K_rot, H_q / H_kv)  ∈ ℝ^(B × T × H_q × d_head)
    │     V_exp = repeat_interleave(V,     H_q / H_kv)  ∈ ℝ^(B × T × H_q × d_head)
    │     h_attn = CausalAttention(Q_rot, K_exp, V_exp) · W_proj          │
    │                                                                     │
    │  3. First Residual Connection:                                      │
    │     h'_ℓ = h_{ℓ-1} + Dropout(h_attn)              ∈ ℝ^(B × T × d)   │
    │                                                                     │
    │  4. Pre-FFN Normalization:                                          │
    │     h_norm2 = ln_2(h'_ℓ)                          ∈ ℝ^(B × T × d)   │
    │                                                                     │
    │  5. SwiGLU Gated Feed-Forward Network:                              │
    │     Gate = SiLU(h_norm2 · W_gate)                 ∈ ℝ^(B × T × d_ffn)
    │     Up   = h_norm2 · W_up                         ∈ ℝ^(B × T × d_ffn)
    │     h_ffn = (Gate ⊙ Up) · W_down                  ∈ ℝ^(B × T × d)   │
    │                                                                     │
    │  6. Second Residual Connection:                                     │
    │     h_ℓ = h'_ℓ + Dropout(h_ffn)                    ∈ ℝ^(B × T × d)   │
    │                                                                     │
    └──────────────┬──────────────────────────────────────────────────────┘
                   │
      Final Normalization: h_final = ln_f(h_L)          ∈ ℝ^(B × T × d_model)
                   │
      Language Modeling Head: W_embed^T                 ∈ ℝ^(d_model × V)
                   │
                   ▼
      Output Logits: Z = h_final · W_embed^T            ∈ ℝ^(B × T × V)

```

---

## 2. Mathematical Formulations & Tensor Contracts

### 1. Rotary Position Embeddings (RoPE)

Legacy transformers rely on learned absolute positional embedding tables ($W_{\text{pos}} \in \mathbb{R}^{T_{\text{max}} \times d_{\text{model}}}$) that struggle with sequence length generalization. RoPE incorporates positional awareness into Query and Key vectors via complex coordinate rotation.

For a head vector $x \in \mathbb{R}^{d_{\text{head}}}$ at sequence index $m \in [0, T-1]$:

1. **Frequency Spectrum:**

$$\theta_i = 10000^{-2(i-1)/d_{\text{head}}}, \quad i \in \left\{1, 2, \dots, \frac{d_{\text{head}}}{2}\right\}$$


2. **2D Plane Partition & Givens Rotation:**
Partition $x$ into coordinate pairs $(x_{2i-1}, x_{2i})$. The rotary transformation applies an orthogonal rotation in each 2D sub-plane:

$$\begin{pmatrix} x_{2i-1}' \\ x_{2i}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i-1} \\ x_{2i} \end{pmatrix}$$


3. **Relative Dot-Product Invariance:**
For a query at position $m$ and key at position $n$:

$$\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = (R_{\Theta, m} q)^T (R_{\Theta, n} k) = q^T R_{\Theta, m}^T R_{\Theta, n} k = q^T R_{\Theta, n - m} k$$



Attention scores depend strictly on relative token distance $n - m$, giving the model natural translation invariance across context windows.

---

### 2. Grouped-Query Attention (GQA) & $O(1)$ KV Caching

Multi-Head Attention (MHA) creates independent Key and Value heads for every Query head ($H_q = H_{kv} = 8$), inflating memory footprint and memory bandwidth stalls during autoregressive generation. Multi-Query Attention (MQA) collapses to 1 Key/Value head, harming model capacity.

**Grouped-Query Attention (GQA)** interpolates between MHA and MQA by clustering Query heads into groups that share a smaller set of Key/Value heads:

$$\text{Group Ratio: } G = \frac{H_q}{H_{kv}} = \frac{8}{2} = 4$$

#### Training Forward Contract:

1. Linear projections:

$$Q = X W_q \in \mathbb{R}^{B \times T \times H_q \times d_{\text{head}}}$$


$$K = X W_k \in \mathbb{R}^{B \times T \times H_{kv} \times d_{\text{head}}}$$


$$V = X W_v \in \mathbb{R}^{B \times T \times H_{kv} \times d_{\text{head}}}$$


2. Keys and Values are expanded across groups:

$$K_{\text{exp}} = \text{repeat\_interleave}(K, \text{dim}=1, \text{repeats}=G) \in \mathbb{R}^{B \times H_q \times T \times d_{\text{head}}}$$


$$V_{\text{exp}} = \text{repeat\_interleave}(V, \text{dim}=1, \text{repeats}=G) \in \mathbb{R}^{B \times H_q \times T \times d_{\text{head}}}$$


3. Attention dispatch:

$$\text{Attention}(Q, K_{\text{exp}}, V_{\text{exp}}) = \text{softmax}\left(\frac{Q K_{\text{exp}}^T}{\sqrt{d_{\text{head}}}} + M_{\text{causal}}\right) V_{\text{exp}}$$



#### Inference $O(1)$ KV Caching:

During sequential autoregressive decoding ($T=1$):

* Only the newly sampled token is projected through $W_q, W_k, W_v$.
* $K_{\text{new}}$ and $V_{\text{new}}$ are appended to pre-allocated buffers along the sequence dimension:

$$K_{\text{cache}} \leftarrow [K_{\text{cache}} \,\Vert{}\, K_{\text{new}}] \in \mathbb{R}^{B \times H_{kv} \times S \times d_{\text{head}}}$$


$$V_{\text{cache}} \leftarrow [V_{\text{cache}} \,\Vert{}\, V_{\text{new}}] \in \mathbb{R}^{B \times H_{kv} \times S \times d_{\text{head}}}$$


* Storing only $H_{kv} = 2$ heads reduces cache memory footprint and PCIe/HBM traffic by **$4\times$** compared to standard MHA.

---

### 3. SwiGLU Feed-Forward Networks

Traditional GPT architectures use standard two-layer MLPs with GELU activations:


$$\text{FFN}_{\text{GELU}}(x) = \text{GELU}(x W_{\text{fc}}) W_{\text{proj}}$$

This implementation adopts the **SwiGLU** (Swish-Gated Linear Unit) variant:


$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$


where $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$.

#### Dimension Scaling & GPU Tensor Core Tile Alignment:

Because SwiGLU introduces a third weight matrix ($W_{\text{up}}$), maintaining the standard $4 \times d_{\text{model}}$ hidden dimension would increase parameter count and compute FLOPs by 50%. To preserve parity with standard $4 \times d_{\text{model}}$ MLPs:


$$d_{\text{ffn}} = \left\lfloor \frac{2}{3} \cdot 4 d_{\text{model}} \right\rfloor = \left\lfloor \frac{8}{3} \cdot 256 \right\rfloor = 682.6$$


To ensure optimal tile partitioning on NVIDIA Tensor Cores (which execute GEMM operations on $16 \times 16$ or $32 \times 32$ matrix sub-tiles), the intermediate dimension is rounded up to the nearest multiple of 64:


$$d_{\text{ffn}} = \left( \left\lfloor \frac{682 + 63}{64} \right\rfloor \right) \times 64 = \mathbf{704}$$

---

### 4. Hardware-Aware Attention Kernel Dispatch

`transformer/attention.py` inspects device capabilities at runtime to select the highest-performing available attention kernel:

```text
┌──────────────────────────────────────────────────────────────┐
│                  Runtime GPU Capability Check                │
└──────────────────────────────┬───────────────────────────────┘
                               │
               Is SM90+ (Hopper / Blackwell)?
              ┌────────────────┴────────────────┐
             Yes                                No
              ▼                                 ▼
   [FlashAttention-3]              Is SM80+ (Ampere / Ada)?
                                  ┌─────────────┴─────────────┐
                                 Yes                          No
                                  ▼                           ▼
                       [FlashAttention-2]         [PyTorch Native SDPA]
                                                (cuDNN / Math Fallbacks)

```

1. **FlashAttention-3 (`sm90+`):** Leverages asynchronous Warp-Specialized GEMM pipelines and TMA (Tensor Memory Accelerator) on Hopper/Blackwell hardware.
2. **FlashAttention-2 (`sm80+`):** Splits work along the sequence dimension, minimizing high-bandwidth memory (HBM) read/write traffic via online softmax tiling.
3. **PyTorch Native SDPA (`torch.nn.functional.scaled_dot_product_attention`):** Universal fallback utilizing optimized C++ / cuDNN flash backends across all modern platforms (CUDA, ROCm, MPS, CPU).

---

## 3. Directory File Catalog

```text
transformer/
├── __init__.py         # Package exports (ModelConfig, DecoderOnlyTransformer)
├── config.py           # Model configuration dataclass with integrity assertions
├── attention.py        # GQA implementation, 2D RoPE rotation, and KV cache buffers
├── feedforward.py      # SwiGLU bilinear gated MLP with Tensor Core dimension alignment
├── block.py            # Pre-LN Transformer decoder block binding attention and FFN
└── model.py            # Complete DecoderOnlyTransformer with weight tying & autoregressive .generate()

```

---

## 4. Architectural Verification & Numerical Invariants

To verify that RoPE, GQA, and SwiGLU maintain full numerical equivalence between full-context forward attention and step-by-step $O(1)$ KV-cached decoding:

```bash
python3 -c "
import torch
from transformer import ModelConfig, DecoderOnlyTransformer

torch.manual_seed(42)
cfg = ModelConfig(
    vocab_size=50257,
    max_seq_len=256,
    d_model=256,
    n_heads=8,
    n_kv_heads=2,
    d_ffn=704,
    n_layers=6,
    dropout=0.0,
    bias=False
)
m = DecoderOnlyTransformer(cfg).eval().cuda()

# 1. Structural Invariant Assertions
ffn = m.layers[0].mlp
assert ffn.w_gate.out_features == 704, 'SwiGLU gate dimension mismatch!'
assert ffn.w_up.out_features == 704, 'SwiGLU up dimension mismatch!'
assert ffn.w_down.in_features == 704, 'SwiGLU down dimension mismatch!'

# 2. End-to-End Equivalence: Full Context vs O(1) KV-Cached Forward Step
x = torch.randint(0, 50257, (1, 16), device='cuda')

# Compute full sequence attention
logits_full, _, _ = m(x)

# Compute prefix prefill up to index T-1
logits_prefill, _, cache = m(x[:, :-1])

# Step a single token at index T using cache
logits_step, _, _ = m(x[:, -1:], kv_caches=cache)

# Evaluate maximum absolute difference
discrepancy = (logits_full[:, -1, :] - logits_step[:, 0, :]).abs().max().item()
print(f'Max discrepancy (Full Attention vs Cached Step): {discrepancy:.2e}')
assert discrepancy < 1e-4, 'Numerical divergence detected in KV Cache!'
print('✓ Mathematical equivalence, RoPE rotary state, and KV-cache tracking verified.')
"

```

```

---

### Command to Overwrite on the Server

Run this command directly on your server to write `transformer/README.md`:

```bash
cat << 'EOF' > /root/slm-gpt/transformer/README.md
# Transformer Core Architecture (`transformer/`)

This directory contains the inner tensor algebra, attention kernels, feed-forward sublayers, and block assemblies that comprise the decoder-only language model. The architecture is engineered modularly from scratch in PyTorch, departing from legacy GPT-2 designs to implement modern state-of-the-art primitives: **Rotary Position Embeddings (RoPE)**, **Grouped-Query Attention (GQA)**, **SwiGLU Feed-Forward Networks**, and **$O(1)$ per-token KV caching**.

---

## 1. Architectural Topology & Tensor Transformations

The model processes a batch of token sequences through an embedding layer, $L$ identical Transformer decoder blocks arranged in a Pre-LayerNorm configuration, a final LayerNorm, and a weight-tied projection head.

```text
Input Token IDs: X ∈ ℕ^(B × T),  X_{b,t} ∈ [0, V-1]
                   │
                   ▼
       wte: Token Embedding Table ∈ ℝ^(V × d_model)
                   │
       Residual Stream: h_0 ∈ ℝ^(B × T × d_model)
                   │
    ┌──────────────┴──────────────────────────────────────────────────────┐
    │                                                                     │
    │  TransformerBlock ℓ ∈ {1, 2, ..., L}                                │
    │                                                                     │
    │  1. Pre-Attention Normalization:                                    │
    │     h_norm1 = ln_1(h_{ℓ-1})                       ∈ ℝ^(B × T × d)   │
    │                                                                     │
    │  2. Grouped-Query Attention (GQA) + RoPE:                           │
    │     Q = h_norm1 · W_q                             ∈ ℝ^(B × T × H_q × d_head)
    │     K = h_norm1 · W_k                             ∈ ℝ^(B × T × H_kv × d_head)
    │     V = h_norm1 · W_v                             ∈ ℝ^(B × T × H_kv × d_head)
    │     Q_rot = ApplyRoPE(Q, positions)                                 │
    │     K_rot = ApplyRoPE(K, positions)                                 │
    │     [Update & Fetch O(1) Key-Value Cache if in inference]           │
    │     K_exp = repeat_interleave(K_rot, H_q / H_kv)  ∈ ℝ^(B × T × H_q × d_head)
    │     V_exp = repeat_interleave(V,     H_q / H_kv)  ∈ ℝ^(B × T × H_q × d_head)
    │     h_attn = CausalAttention(Q_rot, K_exp, V_exp) · W_proj          │
    │                                                                     │
    │  3. First Residual Connection:                                      │
    │     h'_ℓ = h_{ℓ-1} + Dropout(h_attn)              ∈ ℝ^(B × T × d)   │
    │                                                                     │
    │  4. Pre-FFN Normalization:                                          │
    │     h_norm2 = ln_2(h'_ℓ)                          ∈ ℝ^(B × T × d)   │
    │                                                                     │
    │  5. SwiGLU Gated Feed-Forward Network:                              │
    │     Gate = SiLU(h_norm2 · W_gate)                 ∈ ℝ^(B × T × d_ffn)
    │     Up   = h_norm2 · W_up                         ∈ ℝ^(B × T × d_ffn)
    │     h_ffn = (Gate ⊙ Up) · W_down                  ∈ ℝ^(B × T × d)   │
    │                                                                     │
    │  6. Second Residual Connection:                                     │
    │     h_ℓ = h'_ℓ + Dropout(h_ffn)                    ∈ ℝ^(B × T × d)   │
    │                                                                     │
    └──────────────┬──────────────────────────────────────────────────────┘
                   │
      Final Normalization: h_final = ln_f(h_L)          ∈ ℝ^(B × T × d_model)
                   │
      Language Modeling Head: W_embed^T                 ∈ ℝ^(d_model × V)
                   │
                   ▼
      Output Logits: Z = h_final · W_embed^T            ∈ ℝ^(B × T × V)

```

---

## 2. Mathematical Formulations & Tensor Contracts

### 1. Rotary Position Embeddings (RoPE)

Legacy transformers rely on learned absolute positional embedding tables ($W_{\text{pos}} \in \mathbb{R}^{T_{\text{max}} \times d_{\text{model}}}$) that struggle with sequence length generalization. RoPE incorporates positional awareness into Query and Key vectors via complex coordinate rotation.

For a head vector $x \in \mathbb{R}^{d_{\text{head}}}$ at sequence index $m \in [0, T-1]$:

1. **Frequency Spectrum:**

$$\theta_i = 10000^{-2(i-1)/d_{\text{head}}}, \quad i \in \left\{1, 2, \dots, \frac{d_{\text{head}}}{2}\right\}$$


2. **2D Plane Partition & Givens Rotation:**
Partition $x$ into coordinate pairs $(x_{2i-1}, x_{2i})$. The rotary transformation applies an orthogonal rotation in each 2D sub-plane:

$$\begin{pmatrix} x_{2i-1}' \\ x_{2i}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i-1} \\ x_{2i} \end{pmatrix}$$


3. **Relative Dot-Product Invariance:**
For a query at position $m$ and key at position $n$:

$$\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = (R_{\Theta, m} q)^T (R_{\Theta, n} k) = q^T R_{\Theta, m}^T R_{\Theta, n} k = q^T R_{\Theta, n - m} k$$



Attention scores depend strictly on relative token distance $n - m$, giving the model natural translation invariance across context windows.

---

### 2. Grouped-Query Attention (GQA) & $O(1)$ KV Caching

Multi-Head Attention (MHA) creates independent Key and Value heads for every Query head ($H_q = H_{kv} = 8$), inflating memory footprint and memory bandwidth stalls during autoregressive generation. Multi-Query Attention (MQA) collapses to 1 Key/Value head, harming model capacity.

**Grouped-Query Attention (GQA)** interpolates between MHA and MQA by clustering Query heads into groups that share a smaller set of Key/Value heads:

$$\text{Group Ratio: } G = \frac{H_q}{H_{kv}} = \frac{8}{2} = 4$$

#### Training Forward Contract:

1. Linear projections:

$$Q = X W_q \in \mathbb{R}^{B \times T \times H_q \times d_{\text{head}}}$$


$$K = X W_k \in \mathbb{R}^{B \times T \times H_{kv} \times d_{\text{head}}}$$


$$V = X W_v \in \mathbb{R}^{B \times T \times H_{kv} \times d_{\text{head}}}$$


2. Keys and Values are expanded across groups:

$$K_{\text{exp}} = \text{repeat\_interleave}(K, \text{dim}=1, \text{repeats}=G) \in \mathbb{R}^{B \times H_q \times T \times d_{\text{head}}}$$


$$V_{\text{exp}} = \text{repeat\_interleave}(V, \text{dim}=1, \text{repeats}=G) \in \mathbb{R}^{B \times H_q \times T \times d_{\text{head}}}$$


3. Attention dispatch:

$$\text{Attention}(Q, K_{\text{exp}}, V_{\text{exp}}) = \text{softmax}\left(\frac{Q K_{\text{exp}}^T}{\sqrt{d_{\text{head}}}} + M_{\text{causal}}\right) V_{\text{exp}}$$



#### Inference $O(1)$ KV Caching:

During sequential autoregressive decoding ($T=1$):

* Only the newly sampled token is projected through $W_q, W_k, W_v$.
* $K_{\text{new}}$ and $V_{\text{new}}$ are appended to pre-allocated buffers along the sequence dimension:

$$K_{\text{cache}} \leftarrow [K_{\text{cache}} \,\Vert{}\, K_{\text{new}}] \in \mathbb{R}^{B \times H_{kv} \times S \times d_{\text{head}}}$$


$$V_{\text{cache}} \leftarrow [V_{\text{cache}} \,\Vert{}\, V_{\text{new}}] \in \mathbb{R}^{B \times H_{kv} \times S \times d_{\text{head}}}$$


* Storing only $H_{kv} = 2$ heads reduces cache memory footprint and PCIe/HBM traffic by **$4\times$** compared to standard MHA.

---

### 3. SwiGLU Feed-Forward Networks

Traditional GPT architectures use standard two-layer MLPs with GELU activations:


$$\text{FFN}_{\text{GELU}}(x) = \text{GELU}(x W_{\text{fc}}) W_{\text{proj}}$$

This implementation adopts the **SwiGLU** (Swish-Gated Linear Unit) variant:


$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$


where $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$.

#### Dimension Scaling & GPU Tensor Core Tile Alignment:

Because SwiGLU introduces a third weight matrix ($W_{\text{up}}$), maintaining the standard $4 \times d_{\text{model}}$ hidden dimension would increase parameter count and compute FLOPs by 50%. To preserve parity with standard $4 \times d_{\text{model}}$ MLPs:


$$d_{\text{ffn}} = \left\lfloor \frac{2}{3} \cdot 4 d_{\text{model}} \right\rfloor = \left\lfloor \frac{8}{3} \cdot 256 \right\rfloor = 682.6$$


To ensure optimal tile partitioning on NVIDIA Tensor Cores (which execute GEMM operations on $16 \times 16$ or $32 \times 32$ matrix sub-tiles), the intermediate dimension is rounded up to the nearest multiple of 64:


$$d_{\text{ffn}} = \left( \left\lfloor \frac{682 + 63}{64} \right\rfloor \right) \times 64 = \mathbf{704}$$

---

### 4. Hardware-Aware Attention Kernel Dispatch

`transformer/attention.py` inspects device capabilities at runtime to select the highest-performing available attention kernel:

```text
┌──────────────────────────────────────────────────────────────┐
│                  Runtime GPU Capability Check                │
└──────────────────────────────┬───────────────────────────────┘
                               │
               Is SM90+ (Hopper / Blackwell)?
              ┌────────────────┴────────────────┐
             Yes                                No
              ▼                                 ▼
   [FlashAttention-3]              Is SM80+ (Ampere / Ada)?
                                  ┌─────────────┴─────────────┐
                                 Yes                          No
                                  ▼                           ▼
                       [FlashAttention-2]         [PyTorch Native SDPA]
                                                (cuDNN / Math Fallbacks)

```

1. **FlashAttention-3 (`sm90+`):** Leverages asynchronous Warp-Specialized GEMM pipelines and TMA (Tensor Memory Accelerator) on Hopper/Blackwell hardware.
2. **FlashAttention-2 (`sm80+`):** Splits work along the sequence dimension, minimizing high-bandwidth memory (HBM) read/write traffic via online softmax tiling.
3. **PyTorch Native SDPA (`torch.nn.functional.scaled_dot_product_attention`):** Universal fallback utilizing optimized C++ / cuDNN flash backends across all modern platforms (CUDA, ROCm, MPS, CPU).

---

## 3. Directory File Catalog

```text
transformer/
├── __init__.py         # Package exports (ModelConfig, DecoderOnlyTransformer)
├── config.py           # Model configuration dataclass with integrity assertions
├── attention.py        # GQA implementation, 2D RoPE rotation, and KV cache logic
├── feedforward.py      # SwiGLU bilinear gated MLP with Tensor Core dimension alignment
├── block.py            # Pre-LN Transformer decoder block binding attention and FFN
└── model.py            # Complete DecoderOnlyTransformer with weight tying & autoregressive .generate()

```

---

## 4. Architectural Verification & Numerical Invariants

To verify that RoPE, GQA, and SwiGLU maintain full numerical equivalence between full-context forward attention and step-by-step $O(1)$ KV-cached decoding:

```bash
python3 -c "
import torch
from transformer import ModelConfig, DecoderOnlyTransformer

torch.manual_seed(42)
cfg = ModelConfig(
    vocab_size=50257,
    max_seq_len=256,
    d_model=256,
    n_heads=8,
    n_kv_heads=2,
    d_ffn=704,
    n_layers=6,
    dropout=0.0,
    bias=False
)
m = DecoderOnlyTransformer(cfg).eval().cuda()

# 1. Structural Invariant Assertions
ffn = m.layers[0].mlp
assert ffn.w_gate.out_features == 704, 'SwiGLU gate dimension mismatch!'
assert ffn.w_up.out_features == 704, 'SwiGLU up dimension mismatch!'
assert ffn.w_down.in_features == 704, 'SwiGLU down dimension mismatch!'

# 2. End-to-End Equivalence: Full Context vs O(1) KV-Cached Forward Step
x = torch.randint(0, 50257, (1, 16), device='cuda')

# Compute full sequence attention
logits_full, _, _ = m(x)

# Compute prefix prefill up to index T-1
logits_prefill, _, cache = m(x[:, :-1])

# Step a single token at index T using cache
logits_step, _, _ = m(x[:, -1:], kv_caches=cache)

# Evaluate maximum absolute difference
discrepancy = (logits_full[:, -1, :] - logits_step[:, 0, :]).abs().max().item()
print(f'Max discrepancy (Full Attention vs Cached Step): {discrepancy:.2e}')
assert discrepancy < 1e-4, 'Numerical divergence detected in KV Cache!'
print('✓ Mathematical equivalence, RoPE rotary state, and KV-cache tracking verified.')
"

```