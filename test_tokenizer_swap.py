import time
import torch
from tokenizer.factory import get_tokenizer

print("=" * 70)
print("TOKENIZER COMPARISON & VERIFICATION")
print("=" * 70)

# 1. Instantiate both tokenizers
try:
    tok_tiktoken = get_tokenizer("tiktoken")
    print("✓ Loaded 'tiktoken' successfully.")
except Exception as e:
    print(f"✗ Failed to load 'tiktoken': {e}")
    tok_tiktoken = None

try:
    # Pass tokenizer_path if your custom tokenizer requires a model/vocab file
    tok_custom = get_tokenizer("custom")
    print("✓ Loaded 'custom' tokenizer successfully.")
except Exception as e:
    print(f"✗ Failed to load 'custom' tokenizer: {e}")
    tok_custom = None

if not tok_custom:
    print("\nCustom tokenizer could not be initialized. Inspecting factory for available types...")
    exit(1)

tokenizers = {"tiktoken": tok_tiktoken, "custom": tok_custom}

# 2. Check Vocab Size and Special Attributes
print("\n[1] VOCABULARY & ATTRIBUTE AUDIT")
for name, tok in tokenizers.items():
    vocab_sz = getattr(tok, "vocab_size", len(tok) if hasattr(tok, "__len__") else "Unknown")
    eot_id = getattr(tok, "eot_token", getattr(tok, "eos_token_id", None))
    print(f"  • {name:8s} -> vocab_size: {vocab_sz}, eos/eot_token_id: {eot_id}")

# 3. Test Ingestion & Roundtrip Fidelity
test_samples = [
    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        pass",
    "<|im_start|>user\nCalculate 15 * 84.<|im_end|>\n<|im_start|>assistant\n<think>\n15 * 84 = 1260\n</think>\n1260<|im_end|>",
    "<|fim_prefix|>def add(a, b):<|fim_suffix|>    return res<|fim_middle|>\n    res = a + b\n",
]

print("\n[2] ROUNDTRIP & SPECIAL TOKEN AUDIT")
for name, tok in tokenizers.items():
    print(f"\n--- Testing {name} ---")
    for text in test_samples:
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)
        is_exact = (text == decoded)
        print(f"  Length: {len(tokens):3d} tokens | Exact roundtrip: {is_exact}")
        if not is_exact:
            print(f"    Expected: {repr(text)}")
            print(f"    Decoded:  {repr(decoded)}")

# 4. Throughput Benchmark (Critical for 80B Packing)
bench_text = "The quick brown fox jumps over the lazy dog. " * 1000  # ~10,000 words
print("\n[3] THROUGHPUT BENCHMARK (1,000 repetitions)")
for name, tok in tokenizers.items():
    start_t = time.perf_counter()
    n_tokens = 0
    iters = 20
    for _ in range(iters):
        toks = tok.encode(bench_text)
        n_tokens += len(toks)
    elapsed = time.perf_counter() - start_t
    tps = n_tokens / elapsed
    print(f"  • {name:8s}: {tps:,.0f} tokens/sec ({elapsed:.2f}s for {n_tokens:,} tokens)")

print("\n" + "=" * 70)
