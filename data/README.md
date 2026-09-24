```markdown
# Data Pipeline & Ingestion Engine (`data/`)

This directory houses the entire data extraction, tokenization, binary serialization, and runtime streaming infrastructure. It is designed to transform raw natural language text into little-endian binary arrays and stream batches to the GPU with zero-copy OS paging and non-blocking PCIe transfers, sustaining over **355,000 tokens/second** without host I/O bottlenecks.

---

## 1. Directory Structure & File Inventory

```text
data/
├── __init__.py                     # Package marker
├── input.txt                       # Raw text sample (Shakespeare / toy reference corpus)
├── prepare.py                      # Local plain-text preprocessing & tokenization script
├── prepare_hf.py                   # Hugging Face remote dataset streaming & tokenization script
├── dataset.py                      # PyTorch Dataset abstractions for custom / in-memory workflows
├── dataloader.py                   # Standalone iterable data streaming abstractions
└── processed/
    └── fineweb_edu_100m/
        ├── train.bin               # Training partition (~90M tokens, ~180 MB)
        ├── val.bin                 # Validation partition (~5M tokens, ~10 MB)
        └── test.bin                # Holdout test partition (~5M tokens, ~10 MB)

```

### Module Responsibilities

* **`prepare.py`**: Reads raw ASCII/UTF-8 files (such as `input.txt`), tokenizes them via `tiktoken`, splits the sequence into train/val sets (typically 90/10), and exports raw flat `.bin` files.
* **`prepare_hf.py`**: Connects directly to Hugging Face (`HuggingFaceFW/fineweb-edu`), streams documents chunk-by-chunk without storing raw Parquet files locally, tokenizes texts, inserts boundary delimiters, and writes partitioned shards into `data/processed/fineweb_edu_100m/`.
* **`dataset.py` & `dataloader.py**`: Provide reusable modular interfaces for loading token arrays, constructing batched tensor views, and managing epoch boundaries.
* **`processed/`**: Holds pre-tokenized binary files ready for zero-copy memory-mapping during pre-training.

---

## 2. Dataset Specification

* **Corpus Source:** FineWeb-Edu (`HuggingFaceFW/fineweb-edu`)
* **Filtering:** High-scoring educational subset curated for factual density, reasoning steps, syntax diversity, and clear grammatical structures.
* **Pretraining Token Budget:** 100,000,000 tokens (100M).
* **Tokenizer Engine:** Byte-level Byte-Pair Encoding (BPE) using OpenAI's `tiktoken` (`gpt2` profile).
* **Vocabulary Size ($V$):** 50,257 unique tokens.
* **Document Delimiter:** `<|endoftext|>` (Token ID `50256`). Appended to the end of every individual document prior to flattening so the model learns cross-document independence without padding overhead.

---

## 3. Storage Format: `uint16` Binary Serialization

Standard tokenizers often write IDs as Python integers, PyTorch `LongTensor` (`int64`, 8 bytes per token), or NumPy `int32` (4 bytes per token). Because the entire vocabulary $V = 50,257$ is strictly smaller than $2^{16} = 65,536$, all valid token IDs fit inside an unsigned 16-bit integer:

$$\text{Token ID} \in [0, 50256] \subset [0, 65535] = [0, 2^{16} - 1]$$

### Comparison of Memory and Disk Bandwidth Requirements

| Encoding Type | Bytes / Token | 100M Token Storage | Read Bandwidth @ 355k tok/s | Cache Behavior |
| --- | --- | --- | --- | --- |
| `int64` (`torch.long`) | 8 bytes | 800.0 MB | 2.84 MB/s | Evicts OS page cache quickly on constrained hosts |
| `int32` (`np.int32`) | 4 bytes | 400.0 MB | 1.42 MB/s | Standard intermediate format; redundant upper 16 bits |
| **`uint16` (`np.uint16`)** | **2 bytes** | **200.0 MB** | **0.71 MB/s** | **Entire dataset fits within standard OS page cache** |

### Key Benefits

1. **50% Disk Footprint Reduction:** Cut storage requirements in half compared to `int32` and by 75% compared to `int64`.
2. **Reduced I/O Pressure:** Halves NVMe read bandwidth demands, preventing storage controller bottlenecks when scaling to distributed multi-GPU training.
3. **RAM Preservation:** The entire 100M token dataset occupies only ~200 MB of physical space, fitting effortlessly inside the OS virtual memory page cache.

---

## 4. Systems Optimizations & High-Throughput I/O

The streaming engine in `train.py` (via `ShardedDataLoader`) delivers continuous batches directly to GPU VRAM without causing host CPU stalls.

```text
┌─────────────────────────────────────────────────────────────────┐
│              Disk Storage (train.bin / val.bin)                 │
│              Contiguous raw little-endian uint16                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    mmap() OS System Call (Zero-Copy)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Host Kernel Page Cache                       │
│        Virtual memory pages faulted on-demand (no heap copies)  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    Contiguous 1D Slice + .reshape(B, T)
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

### 1. Zero-Copy Kernel Paging via `np.memmap`

Traditional approaches load raw files into user-space Python RAM (`open().read()`, `pickle.load()`, or DataFrame conversions). This causes multi-gigabyte memory spikes, long initialization pauses, and garbage collection latency.

`ShardedDataLoader` opens `.bin` files via memory mapping:

```python
self.tokens = np.memmap(filename, dtype=np.uint16, mode="r")

```

* **Instantaneous Initialization ($O(1)$):** The file descriptor is mapped into virtual memory immediately without preloading data bytes into RAM.
* **Kernel-Level Page Fault Handling:** The operating system fetches only the 4 KB physical pages touched by the current batch. Pages no longer needed can be evicted automatically by the OS kernel without runtime overhead.

### 2. Contiguous 1D Slicing with Pointer Offsets

Random sequence sampling across arbitrary token coordinates can cause scattered disk reads and cache thrashing. The loader pulls one contiguous 1D block for the entire batch:

$$\text{Buffer Length} = B \times T + 1$$

For $B = 64$ and $T = 256$, $\text{Buffer Length} = 64 \times 256 + 1 = 16,385$ tokens:

```python
buf_len = B * T + 1
buf = self.tokens[self.current_pos : self.current_pos + buf_len].astype(np.int64)

x_np = buf[:-1].reshape(B, T)
y_np = buf[1:].reshape(B, T)
self.current_pos += B * T

```

* **Linear Sequential I/O:** Reading a single contiguous 16,385-element slice maximizes NVMe hardware read throughput.
* **Offset Autoregressive Alignment:** Target sequence $Y$ is created by shifting inputs $X$ forward by 1 token ($y_t = x_{t+1}$) via a zero-copy pointer slice.

### 3. Pinned (Page-Locked) Memory

Standard host memory is pageable; the OS kernel can relocate or swap pages to disk. Because GPU DMA (Direct Memory Access) controllers cannot safely read from pageable memory without risk of OS relocation, standard `.to("cuda")` calls require PyTorch to first perform an internal copy from pageable host memory to a temporary pinned staging buffer.

By explicitly locking the host pages:

```python
x = torch.from_numpy(x_np).pin_memory()
y = torch.from_numpy(y_np).pin_memory()

```

the GPU DMA controller transfers bytes directly across the PCIe bus, bypassing the CPU staging copy.

### 4. Asynchronous Host-to-Device Streaming (`non_blocking=True`)

Synchronous transfers stall the host CPU until the PCIe bus finishes copying the tensor into VRAM. Passing `non_blocking=True` delegates the transfer to an asynchronous CUDA copy stream:

```python
x = x.to(DEVICE, non_blocking=True)
y = y.to(DEVICE, non_blocking=True)

```

While the GPU Tensor Cores are busy computing the forward and backward passes of micro-step $k$, the PCIe bus is concurrently transferring the data for micro-step $k+1$. This completely hides dataloading latency behind compute execution.

---

## 5. Usage & Verification

### Tokenizing a Local Corpus

```bash
# Tokenizes input.txt and generates train.bin / val.bin in data/
python3 data/prepare.py

```

### Streaming & Processing FineWeb-Edu from Hugging Face

```bash
# Streams 100M tokens from Hugging Face and saves to data/processed/fineweb_edu_100m/
python3 data/prepare_hf.py

```

### Direct Streaming Sanity Check

You can test the loader's integrity, batch shapes, device placement, and autoregressive target alignment with this check:

```python
import torch
import numpy as np
from train import ShardedDataLoader

loader = ShardedDataLoader(
    data_dir="data/processed/fineweb_edu_100m",
    split="train",
    batch_size=64,
    seq_len=256
)

x, y = loader.next_batch()
print(f"Batch X Shape: {x.shape} | Dtype: {x.dtype} | Device: {x.device}")
print(f"Batch Y Shape: {y.shape} | Dtype: {y.dtype} | Device: {y.device}")

# Verify 1-token shift: target y[b, t] must equal input x[b, t + 1]
assert (x[:, 1:] == y[:, :-1]).all(), "Fatal: Autoregressive alignment mismatch!"
print("✓ DataLoader integrity and autoregressive alignment verified successfully.")

```

```

---

### Command to Overwrite on Your Server

Run this command on your server to update `data/README.md`:

```bash
cat << 'EOF' > /root/slm-gpt/data/README.md
# Data Pipeline & Ingestion Engine (`data/`)

This directory houses the entire data extraction, tokenization, binary serialization, and runtime streaming infrastructure. It is designed to transform raw natural language text into little-endian binary arrays and stream batches to the GPU with zero-copy OS paging and non-blocking PCIe transfers, sustaining over **355,000 tokens/second** without host I/O bottlenecks.

---

## 1. Directory Structure & File Inventory

```text
data/
├── __init__.py                     # Package marker
├── input.txt                       # Raw text sample (Shakespeare / toy reference corpus)
├── prepare.py                      # Local plain-text preprocessing & tokenization script
├── prepare_hf.py                   # Hugging Face remote dataset streaming & tokenization script
├── dataset.py                      # PyTorch Dataset abstractions for custom / in-memory workflows
├── dataloader.py                   # Standalone iterable data streaming abstractions
└── processed/
    └── fineweb_edu_100m/
        ├── train.bin               # Training partition (~90M tokens, ~180 MB)
        ├── val.bin                 # Validation partition (~5M tokens, ~10 MB)
        └── test.bin                # Holdout test partition (~5M tokens, ~10 MB)

```

### Module Responsibilities

* **`prepare.py`**: Reads raw ASCII/UTF-8 files (such as `input.txt`), tokenizes them via `tiktoken`, splits the sequence into train/val sets (typically 90/10), and exports raw flat `.bin` files.
* **`prepare_hf.py`**: Connects directly to Hugging Face (`HuggingFaceFW/fineweb-edu`), streams documents chunk-by-chunk without storing raw Parquet files locally, tokenizes texts, inserts boundary delimiters, and writes partitioned shards into `data/processed/fineweb_edu_100m/`.
* **`dataset.py` & `dataloader.py**`: Provide reusable modular interfaces for loading token arrays, constructing batched tensor views, and managing epoch boundaries.
* **`processed/`**: Holds pre-tokenized binary files ready for zero-copy memory-mapping during pre-training.

---

## 2. Dataset Specification

* **Corpus Source:** FineWeb-Edu (`HuggingFaceFW/fineweb-edu`)
* **Filtering:** High-scoring educational subset curated for factual density, reasoning steps, syntax diversity, and clear grammatical structures.
* **Pretraining Token Budget:** 100,000,000 tokens (100M).
* **Tokenizer Engine:** Byte-level Byte-Pair Encoding (BPE) using OpenAI's `tiktoken` (`gpt2` profile).
* **Vocabulary Size ($V$):** 50,257 unique tokens.
* **Document Delimiter:** `<|endoftext|>` (Token ID `50256`). Appended to the end of every individual document prior to flattening so the model learns cross-document independence without padding overhead.

---

## 3. Storage Format: `uint16` Binary Serialization

Standard tokenizers often write IDs as Python integers, PyTorch `LongTensor` (`int64`, 8 bytes per token), or NumPy `int32` (4 bytes per token). Because the entire vocabulary $V = 50,257$ is strictly smaller than $2^{16} = 65,536$, all valid token IDs fit inside an unsigned 16-bit integer:

$$\text{Token ID} \in [0, 50256] \subset [0, 65535] = [0, 2^{16} - 1]$$

### Comparison of Memory and Disk Bandwidth Requirements

| Encoding Type | Bytes / Token | 100M Token Storage | Read Bandwidth @ 355k tok/s | Cache Behavior |
| --- | --- | --- | --- | --- |
| `int64` (`torch.long`) | 8 bytes | 800.0 MB | 2.84 MB/s | Evicts OS page cache quickly on constrained hosts |
| `int32` (`np.int32`) | 4 bytes | 400.0 MB | 1.42 MB/s | Standard intermediate format; redundant upper 16 bits |
| **`uint16` (`np.uint16`)** | **2 bytes** | **200.0 MB** | **0.71 MB/s** | **Entire dataset fits within standard OS page cache** |

### Key Benefits

1. **50% Disk Footprint Reduction:** Cut storage requirements in half compared to `int32` and by 75% compared to `int64`.
2. **Reduced I/O Pressure:** Halves NVMe read bandwidth demands, preventing storage controller bottlenecks when scaling to distributed multi-GPU training.
3. **RAM Preservation:** The entire 100M token dataset occupies only ~200 MB of physical space, fitting effortlessly inside the OS virtual memory page cache.

---

## 4. Systems Optimizations & High-Throughput I/O

The streaming engine in `train.py` (via `ShardedDataLoader`) delivers continuous batches directly to GPU VRAM without causing host CPU stalls.

```text
┌─────────────────────────────────────────────────────────────────┐
│              Disk Storage (train.bin / val.bin)                 │
│              Contiguous raw little-endian uint16                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    mmap() OS System Call (Zero-Copy)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Host Kernel Page Cache                       │
│        Virtual memory pages faulted on-demand (no heap copies)  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    Contiguous 1D Slice + .reshape(B, T)
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

### 1. Zero-Copy Kernel Paging via `np.memmap`

Traditional approaches load raw files into user-space Python RAM (`open().read()`, `pickle.load()`, or DataFrame conversions). This causes multi-gigabyte memory spikes, long initialization pauses, and garbage collection latency.

`ShardedDataLoader` opens `.bin` files via memory mapping:

```python
self.tokens = np.memmap(filename, dtype=np.uint16, mode="r")

```

* **Instantaneous Initialization ($O(1)$):** The file descriptor is mapped into virtual memory immediately without preloading data bytes into RAM.
* **Kernel-Level Page Fault Handling:** The operating system fetches only the 4 KB physical pages touched by the current batch. Pages no longer needed can be evicted automatically by the OS kernel without runtime overhead.

### 2. Contiguous 1D Slicing with Pointer Offsets

Random sequence sampling across arbitrary token coordinates can cause scattered disk reads and cache thrashing. The loader pulls one contiguous 1D block for the entire batch:

$$\text{Buffer Length} = B \times T + 1$$

For $B = 64$ and $T = 256$, $\text{Buffer Length} = 64 \times 256 + 1 = 16,385$ tokens:

```python
buf_len = B * T + 1
buf = self.tokens[self.current_pos : self.current_pos + buf_len].astype(np.int64)

x_np = buf[:-1].reshape(B, T)
y_np = buf[1:].reshape(B, T)
self.current_pos += B * T

```

* **Linear Sequential I/O:** Reading a single contiguous 16,385-element slice maximizes NVMe hardware read throughput.
* **Offset Autoregressive Alignment:** Target sequence $Y$ is created by shifting inputs $X$ forward by 1 token ($y_t = x_{t+1}$) via a zero-copy pointer slice.

### 3. Pinned (Page-Locked) Memory

Standard host memory is pageable; the OS kernel can relocate or swap pages to disk. Because GPU DMA (Direct Memory Access) controllers cannot safely read from pageable memory without risk of OS relocation, standard `.to("cuda")` calls require PyTorch to first perform an internal copy from pageable host memory to a temporary pinned staging buffer.

By explicitly locking the host pages:

```python
x = torch.from_numpy(x_np).pin_memory()
y = torch.from_numpy(y_np).pin_memory()

```

the GPU DMA controller transfers bytes directly across the PCIe bus, bypassing the CPU staging copy.

### 4. Asynchronous Host-to-Device Streaming (`non_blocking=True`)

Synchronous transfers stall the host CPU until the PCIe bus finishes copying the tensor into VRAM. Passing `non_blocking=True` delegates the transfer to an asynchronous CUDA copy stream:

```python
x = x.to(DEVICE, non_blocking=True)
y = y.to(DEVICE, non_blocking=True)

```

While the GPU Tensor Cores are busy computing the forward and backward passes of micro-step $k$, the PCIe bus is concurrently transferring the data for micro-step $k+1$. This completely hides dataloading latency behind compute execution.

---

## 5. Usage & Verification

### Tokenizing a Local Corpus

```bash
# Tokenizes input.txt and generates train.bin / val.bin in data/
python3 data/prepare.py

```

### Streaming & Processing FineWeb-Edu from Hugging Face

```bash
# Streams 100M tokens from Hugging Face and saves to data/processed/fineweb_edu_100m/
python3 data/prepare_hf.py

```

### Direct Streaming Sanity Check

You can test the loader's integrity, batch shapes, device placement, and autoregressive target alignment with this check:

```python
import torch
import numpy as np
from train import ShardedDataLoader

loader = ShardedDataLoader(
    data_dir="data/processed/fineweb_edu_100m",
    split="train",
    batch_size=64,
    seq_len=256
)

x, y = loader.next_batch()
print(f"Batch X Shape: {x.shape} | Dtype: {x.dtype} | Device: {x.device}")
print(f"Batch Y Shape: {y.shape} | Dtype: {y.dtype} | Device: {y.device}")

# Verify 1-token shift: target y[b, t] must equal input x[b, t + 1]
assert (x[:, 1:] == y[:, :-1]).all(), "Fatal: Autoregressive alignment mismatch!"
print("✓ DataLoader integrity and autoregressive alignment verified successfully.")

```