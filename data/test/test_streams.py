import os
import sys
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parents[2]))

from tokenizer import PretrainedTiktokenTokenizer


def run_dataset_stream(name: str, config: str = None, text_column: str = "text", samples: int = 5):
    print(f"\n--- Testing Stream: {name} (config: {config}) ---")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    tokenizer = PretrainedTiktokenTokenizer("gpt2")

    ds = load_dataset(name, config, split="train", streaming=True, token=token)
    
    count = 0
    total_tokens = 0
    for row in ds:
        text = row.get(text_column, "")
        if not text.strip():
            continue
        
        token_ids = tokenizer.encode(text)
        total_tokens += len(token_ids)
        snippet = text.replace("\n", " ")[:70]
        print(f"[{count+1}] Tokens: {len(token_ids):4d} | Snippet: \"{snippet}...\"")
        
        count += 1
        if count >= samples:
            break
            
    print(f"✓ {name} stream verified. Sampled {total_tokens:,} tokens across {samples} documents.")


if __name__ == "__main__":
    # 1. TinyStories
    run_dataset_stream("roneneldan/TinyStories", config=None, text_column="text")

    # 2. OpenWebText
    run_dataset_stream("Skylion007/openwebtext", config=None, text_column="text")

    # 3. FineWeb-Edu (Uses the curated 10BT sample subset)
    run_dataset_stream("HuggingFaceFW/fineweb-edu", config="sample-10BT", text_column="text")