### `tokenizer/README.md`

```markdown
# Tokenizer Engine & Specification (`tokenizer/`)

This directory documents the tokenization architecture, vocabulary structure, byte-level Byte-Pair Encoding (BPE) mechanics, and serialization pipeline. The engine translates variable-length Unicode/UTF-8 character streams into discrete integer coordinate vectors within the vocabulary boundary $V = 50,257$.

---

## 1. Algorithmic Foundation: Byte-Level Byte-Pair Encoding (BBPE)

Standard word-level tokenizers encounter Out-Of-Vocabulary (OOV) failures on rare words, technical terms, or multilingual tokens. Character-level tokenizers solve OOV issues but inflate sequence lengths by $4\times$ to $6\times$, drastically worsening the $O(T^2)$ computational complexity of self-attention.

This project implements **Byte-Level Byte-Pair Encoding (BBPE)** via OpenAI's `tiktoken` (`gpt2` encoding profile):

1. **Base Alphabet of 256 Raw Bytes:**
   The initial vocabulary comprises all individual 8-bit bytes ($0x00$ through $0xFF$). Every valid UTF-8 string is inherently representable as a sequence of bytes. Consequently, the tokenizer possesses an **OOV rate of strictly 0.0%**—every arbitrary byte stream can be encoded and reconstructed without fallback `<unk>` tokens.

2. **Iterative Frequency-Based Merges:**
   Starting from the 256 byte primitives, frequent contiguous byte pairs are iteratively merged over a massive text corpus to construct subword tokens until the vocabulary threshold is reached:
   $$\text{pair}(t_i, t_{i+1}) \longrightarrow t_{\text{merged}}$$

3. **Pre-Tokenization Splitting Regex:**
   To prevent merges across heterogeneous linguistic boundaries (e.g., merging punctuation with letters, or numerals across words), inputs are split by a deterministic regular expression before running BPE merges:
   ```regex
   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

```

* `'s|'t|'re...`: Preserves common English contraction morphology as atomic prefixes/suffixes.
* ` ?\p{L}+`: Matches alphabetic Unicode words, optionally prepended with a single space.
* ` ?\p{N}+`: Isolates numerical runs (preventing number/letter token entanglement).
* ` ?[^\s\p{L}\p{N}]+`: Isolates punctuation sequences and mathematical symbols.
* `\s+(?!\S)|\s+`: Captures trailing whitespace sequences without attaching to subsequent semantic tokens.

---

## 2. Vocabulary Memory Layout & ID Allocations

The model operates over a total vocabulary capacity of $V = 50,257$.

```text
┌─────────────────────────────┬─────────────────────────────┬─────────────────────────────┐
│ Token ID Range              │ Subword Category            │ Concrete Examples           │
├─────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ 0 - 255                     │ Raw UTF-8 Byte Primitives   │ b'\x00', b'a', b'\n', b'\xff'│
│ 256 - 50,255                │ Learned Subwords & Words    │ " the", " algorithm", "Ġin" │
│ 50,256 (Special Token)      │ End-Of-Text (<|endoftext|>) │ Document separator          │
└─────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
Total Vocabulary Cardinality (V): 50,257

```

### 1. The `<|endoftext|>` Sentinel Token (ID `50256`)

* **Role:** Acts as a document boundary marker.
* **Pretraining Injection:** Injected between concatenated documents within the continuous 1D token stream (`train.bin`).
* **Inference Termination:** Signals autoregressive generation to halt decoding prior to reaching `max_seq_len`.
* **Disallowed Special Token Guarding:** Configured in `tiktoken` to permit explicit encoding only when deliberately specified:
```python
enc.encode(text, allowed_special={"<|endoftext|>"})

```



### 2. Numerical Storage Fit (`uint16`)

Because $\max(\text{Token ID}) = 50,256 < 65,535$ ($2^{16} - 1$), token sequences require strictly 2 bytes per integer. This matches the `np.uint16` binary shard storage format documented in `data/README.md`.

---

## 3. Subword Token Mechanics & Invertibility

The tokenizer provides **lossless roundtrip reversibility** (identity mapping):

$$\forall \, s \in \text{Unicode Strings}, \quad \text{decode}(\text{encode}(s)) \equiv s$$

### Space Prepending (`Ġ` Representation)

In GPT-2 BPE, whitespace preceding a word is treated as an intrinsic structural component of the token rather than stripped syntax:

* `"algorithm"` $\to$ Token ID `20199` (`"algorithm"`)
* `" algorithm"` $\to$ Token ID `10688` (`"Ġalgorithm"`)

This distinction allows the transformer's attention heads to model boundary syntax and token transitions without dedicated delimiter channels.

### Multi-Byte UTF-8 Recombination

Characters spanning multiple UTF-8 bytes (e.g., emojis, mathematical symbols, non-Latin scripts) that were not frequent enough in the merge table decompose gracefully into their raw underlying byte tokens:

```python
text = "π"  # UTF-8 hex: 0xCF 0x80 (2 bytes)
# If unmerged, encodes as two individual byte tokens: [Byte(0xCF), Byte(0x80)]
# During decoding, bytes are concatenated and decoded losslessly back into "π"

```

---

## 4. API Interface & Systems Verification

### Standard Encode/Decode Interface

The tokenizer is consumed across `train.py`, `eval_inference.py`, and `prepare_hf.py` via `tiktoken`:

```python
import tiktoken
import numpy as np

class Tokenizer:
    def __init__(self, name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(name)
        self.eot_token = self.enc.eot_token  # 50256
        self.vocab_size = self.enc.n_vocab   # 50257

    def encode(self, text: str, allowed_special: bool = False) -> list[int]:
        if allowed_special:
            return self.enc.encode(text, allowed_special={"<|endoftext|>"})
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)

    def encode_to_uint16(self, text: str) -> np.ndarray:
        tokens = self.encode(text)
        return np.array(tokens, dtype=np.uint16)

```

---

## 5. Verification & Unit Validation Suite

Run this sanity check to verify tokenizer vocabulary boundaries, losslessness, `<|endoftext|>` ID mapping, and `uint16` memory alignment:

```bash
python3 -c "
import tiktoken
import numpy as np

enc = tiktoken.get_encoding('gpt2')

# 1. Cardinality verification
assert enc.n_vocab == 50257, f'Expected 50257 tokens, got {enc.n_vocab}'
assert enc.eot_token == 50256, f'Expected EOT token 50256, got {enc.eot_token}'

# 2. Reversibility & Special token handling
sample_text = 'In computer science, an algorithm is a finite sequence of instructions. <|endoftext|>'
token_ids = enc.encode(sample_text, allowed_special={'<|endoftext|>'})
assert token_ids[-1] == 50256, 'EOT token was not properly encoded to 50256!'
decoded = enc.decode(token_ids)
assert decoded == sample_text, 'Roundtrip losslessness check failed!'

# 3. Memory layout check
arr = np.array(token_ids, dtype=np.uint16)
assert arr.itemsize == 2, 'Array is not 16-bit!'
assert (arr.astype(int) == token_ids).all(), 'Numerical truncation detected when casting to uint16!'

print('✓ Tokenizer specification, vocabulary boundaries, and uint16 compatibility verified.')
"

```

```

---

### Command to Overwrite on the Server

Run this command to create the directory (if needed) and write `tokenizer/README.md`:

```bash
mkdir -p /root/slm-gpt/tokenizer
cat << 'EOF' > /root/slm-gpt/tokenizer/README.md
# Tokenizer Engine & Specification (`tokenizer/`)

This directory documents the tokenization architecture, vocabulary structure, byte-level Byte-Pair Encoding (BPE) mechanics, and serialization pipeline. The engine translates variable-length Unicode/UTF-8 character streams into discrete integer coordinate vectors within the vocabulary boundary $V = 50,257$.

---

## 1. Algorithmic Foundation: Byte-Level Byte-Pair Encoding (BBPE)

Standard word-level tokenizers encounter Out-Of-Vocabulary (OOV) failures on rare words, technical terms, or multilingual tokens. Character-level tokenizers solve OOV issues but inflate sequence lengths by $4\times$ to $6\times$, drastically worsening the $O(T^2)$ computational complexity of self-attention.

This project implements **Byte-Level Byte-Pair Encoding (BBPE)** via OpenAI's `tiktoken` (`gpt2` encoding profile):

1. **Base Alphabet of 256 Raw Bytes:**
   The initial vocabulary comprises all individual 8-bit bytes ($0x00$ through $0xFF$). Every valid UTF-8 string is inherently representable as a sequence of bytes. Consequently, the tokenizer possesses an **OOV rate of strictly 0.0%**—every arbitrary byte stream can be encoded and reconstructed without fallback `<unk>` tokens.

2. **Iterative Frequency-Based Merges:**
   Starting from the 256 byte primitives, frequent contiguous byte pairs are iteratively merged over a massive text corpus to construct subword tokens until the vocabulary threshold is reached:
   $$\text{pair}(t_i, t_{i+1}) \longrightarrow t_{\text{merged}}$$

3. **Pre-Tokenization Splitting Regex:**
   To prevent merges across heterogeneous linguistic boundaries (e.g., merging punctuation with letters, or numerals across words), inputs are split by a deterministic regular expression before running BPE merges:
   ```regex
   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

```

* `'s|'t|'re...`: Preserves common English contraction morphology as atomic prefixes/suffixes.
* ` ?\p{L}+`: Matches alphabetic Unicode words, optionally prepended with a single space.
* ` ?\p{N}+`: Isolates numerical runs (preventing number/letter token entanglement).
* ` ?[^\s\p{L}\p{N}]+`: Isolates punctuation sequences and mathematical symbols.
* `\s+(?!\S)|\s+`: Captures trailing whitespace sequences without attaching to subsequent semantic tokens.

---

## 2. Vocabulary Memory Layout & ID Allocations

The model operates over a total vocabulary capacity of $V = 50,257$.

```text
┌─────────────────────────────┬─────────────────────────────┬─────────────────────────────┐
│ Token ID Range              │ Subword Category            │ Concrete Examples           │
├─────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ 0 - 255                     │ Raw UTF-8 Byte Primitives   │ b'\x00', b'a', b'\n', b'\xff'│
│ 256 - 50,255                │ Learned Subwords & Words    │ " the", " algorithm", "Ġin" │
│ 50,256 (Special Token)      │ End-Of-Text (<|endoftext|>) │ Document separator          │
└─────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
Total Vocabulary Cardinality (V): 50,257

```

### 1. The `<|endoftext|>` Sentinel Token (ID `50256`)

* **Role:** Acts as a document boundary marker.
* **Pretraining Injection:** Injected between concatenated documents within the continuous 1D token stream (`train.bin`).
* **Inference Termination:** Signals autoregressive generation to halt decoding prior to reaching `max_seq_len`.
* **Disallowed Special Token Guarding:** Configured in `tiktoken` to permit explicit encoding only when deliberately specified:
```python
enc.encode(text, allowed_special={"<|endoftext|>"})

```



### 2. Numerical Storage Fit (`uint16`)

Because $\max(\text{Token ID}) = 50,256 < 65,535$ ($2^{16} - 1$), token sequences require strictly 2 bytes per integer. This matches the `np.uint16` binary shard storage format documented in `data/README.md`.

---

## 3. Subword Token Mechanics & Invertibility

The tokenizer provides **lossless roundtrip reversibility** (identity mapping):

$$\forall \, s \in \text{Unicode Strings}, \quad \text{decode}(\text{encode}(s)) \equiv s$$

### Space Prepending (`Ġ` Representation)

In GPT-2 BPE, whitespace preceding a word is treated as an intrinsic structural component of the token rather than stripped syntax:

* `"algorithm"` $\to$ Token ID `20199` (`"algorithm"`)
* `" algorithm"` $\to$ Token ID `10688` (`"Ġalgorithm"`)

This distinction allows the transformer's attention heads to model boundary syntax and token transitions without dedicated delimiter channels.

### Multi-Byte UTF-8 Recombination

Characters spanning multiple UTF-8 bytes (e.g., emojis, mathematical symbols, non-Latin scripts) that were not frequent enough in the merge table decompose gracefully into their raw underlying byte tokens:

```python
text = "π"  # UTF-8 hex: 0xCF 0x80 (2 bytes)
# If unmerged, encodes as two individual byte tokens: [Byte(0xCF), Byte(0x80)]
# During decoding, bytes are concatenated and decoded losslessly back into "π"

```

---

## 4. API Interface & Systems Verification

### Standard Encode/Decode Interface

The tokenizer is consumed across `train.py`, `eval_inference.py`, and `prepare_hf.py` via `tiktoken`:

```python
import tiktoken
import numpy as np

class Tokenizer:
    def __init__(self, name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(name)
        self.eot_token = self.enc.eot_token  # 50256
        self.vocab_size = self.enc.n_vocab   # 50257

    def encode(self, text: str, allowed_special: bool = False) -> list[int]:
        if allowed_special:
            return self.enc.encode(text, allowed_special={"<|endoftext|>"})
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)

    def encode_to_uint16(self, text: str) -> np.ndarray:
        tokens = self.encode(text)
        return np.array(tokens, dtype=np.uint16)

```

---

## 5. Verification & Unit Validation Suite

Run this sanity check to verify tokenizer vocabulary boundaries, losslessness, `<|endoftext|>` ID mapping, and `uint16` memory alignment:

```bash
python3 -c "
import tiktoken
import numpy as np

enc = tiktoken.get_encoding('gpt2')

# 1. Cardinality verification
assert enc.n_vocab == 50257, f'Expected 50257 tokens, got {enc.n_vocab}'
assert enc.eot_token == 50256, f'Expected EOT token 50256, got {enc.eot_token}'

# 2. Reversibility & Special token handling
sample_text = 'In computer science, an algorithm is a finite sequence of instructions. <|endoftext|>'
token_ids = enc.encode(sample_text, allowed_special={'<|endoftext|>'})
assert token_ids[-1] == 50256, 'EOT token was not properly encoded to 50256!'
decoded = enc.decode(token_ids)
assert decoded == sample_text, 'Roundtrip losslessness check failed!'

# 3. Memory layout check
arr = np.array(token_ids, dtype=np.uint16)
assert arr.itemsize == 2, 'Array is not 16-bit!'
assert (arr.astype(int) == token_ids).all(), 'Numerical truncation detected when casting to uint16!'

print('✓ Tokenizer specification, vocabulary boundaries, and uint16 compatibility verified.')
"

```
