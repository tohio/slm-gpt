import os
import time
from pathlib import Path
from tokenizer.factory import get_tokenizer

# 1. Locate custom vocab file
potential_paths = [
    "tokenizer/vocab.json",
    "tokenizer/tokenizer.json",
    "tokenizer/custom_bpe.json",
    "data/tokenizer.json"
]
vocab_path = next((p for p in potential_paths if os.path.exists(p)), None)

print("=" * 70)
print("TOKENIZER COMPARISON & PERFORMANCE BENCHMARK")
print("=" * 70)

# 2. Instantiate Tokenizers
tiktoken_tok = get_tokenizer("tiktoken")
print(f"✓ Tiktoken loaded. Vocab size: {getattr(tiktoken_tok, 'vocab_size', 'Unknown')}")

if not vocab_path:
    print(f"⚠️  No vocab artifact found in default locations: {potential_paths}")
    print("   Please pass the exact path to CustomBPETokenizer.from_file(tokenizer_path).")
    exit(0)

print(f"Found custom vocab at: {vocab_path}")
custom_tok = get_tokenizer("custom", tokenizer_path=vocab_path)
print(f"✓ Custom BPE loaded. Vocab size: {getattr(custom_tok, 'vocab_size', 'Unknown')}")

# 3. Special Tokens & ChatML / Reasoning Roundtrip
samples = [
    "def calculate_total(items: list[dict]) -> float:\n    return sum(item['price'] for item in items)",
    "<|im_start|>user\nCompute 24 * 7.<|im_end|>\n<|im_start|>assistant\n<think>\n24 * 7 = 168\n</think>\n168<|im_end|>",
    "<|fim_prefix|>def add(a, b):<|fim_suffix|>    return out<|fim_middle|>\n    out = a + b\n"
]

print("\n[1] SPECIAL TOKEN & ROUNDTRIP TEST")
for name, tok in [("Tiktoken", tiktoken_tok), ("Custom", custom_tok)]:
    print(f"\n--- {name} ---")
    for text in samples:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        exact = (text == decoded)
        print(f"  Length: {len(ids):3d} tokens | Roundtrip: {exact}")
        if not exact:
            print(f"    Expected: {repr(text)}")
            print(f"    Decoded:  {repr(decoded)}")

# 4. Throughput Benchmark (100k tokens test)
bench_text = "The quick brown fox jumps over the lazy dog. Slm-gpt pretraining tokenization test. " * 500
print("\n[2] ENCODING THROUGHPUT BENCHMARK")
for name, tok in [("Tiktoken", tiktoken_tok), ("Custom", custom_tok)]:
    start = time.perf_counter()
    total_tokens = 0
    iters = 25
    for _ in range(iters):
        ids = tok.encode(bench_text)
        total_tokens += len(ids)
    elapsed = time.perf_counter() - start
    tps = total_tokens / elapsed
    print(f"  • {name:10s}: {tps:,.0f} tokens/sec ({total_tokens:,} tokens in {elapsed:.3f}s)")

print("\n" + "=" * 70)
