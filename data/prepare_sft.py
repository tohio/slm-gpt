"""
data/prepare_sft.py: 4-Pillar Composite SFT Dataset Assembler for slm-gpt.
Harvests Code (Magicoder), Reasoning with <think> (OpenR1-Math), Constraints,
and Dialogue (SmolTalk), performs AST validation, strips preambles/postambles,
and outputs standardized train.jsonl and val.jsonl datasets.
"""

import ast
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Regex rules for iterative preamble/postamble stripping
PREAMBLE_PATTERNS = [
    r"^(?:Sure|Certainly|Of course|Here is|Here's|Below is|Here are)[^:\n]*:\s*",
    r"^(?:Sure|Certainly|Of course|I'd be happy to help|Alright)[,!\.]\s*",
    r"^(?:Here is the python code|Here is the code|Here is the solution)[^:\n]*:\s*",
]

POSTAMBLE_PATTERNS = [
    r"\n*(?:Hope this helps|Let me know if you have any questions|Let me know if you need anything else|Feel free to ask)[^.\n]*\.?$",
    r"\n*(?:Happy coding|Good luck)[\.!]*$",
]


def validate_python_ast(code: str) -> bool:
    """Validates whether a code string is syntactically valid Python."""
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_python_snippet(text: str) -> Optional[str]:
    """Extracts Python code from Markdown code blocks or validates raw text."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        for m in matches:
            if m.strip():
                return m.strip()
    if validate_python_ast(text):
        return text.strip()
    return None


def sanitize_technical_response_preambles(text: str) -> str:
    """Iteratively strips conversational introductory fluff."""
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pat in PREAMBLE_PATTERNS:
            new_text = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
            if new_text != cleaned:
                cleaned = new_text
                changed = True
    return cleaned


def sanitize_technical_response_postambles(text: str) -> str:
    """Iteratively strips conversational sign-offs and pleasantries."""
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pat in POSTAMBLE_PATTERNS:
            new_text = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
            if new_text != cleaned:
                cleaned = new_text
                changed = True
    return cleaned


def harvest_code_samples(budget: int) -> List[Dict[str, Any]]:
    """Harvests Python programming tasks from ise-uiuc/Magicoder-OSS-Instruct-75K."""
    print(f"\n[1/4] Harvesting {budget:,} Code Execution samples (local cache)...")
    samples: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        for row in ds:
            # Handle schema variations: instruction/response vs problem/solution
            prob = (
                row.get("instruction")
                or row.get("problem")
                or row.get("prompt")
                or ""
            ).strip()
            sol = (
                row.get("response")
                or row.get("solution")
                or row.get("answer")
                or ""
            ).strip()

            if not prob or not sol:
                continue

            cleaned_sol = sanitize_technical_response_postambles(
                sanitize_technical_response_preambles(sol)
            )
            snippet = extract_python_snippet(cleaned_sol)
            if snippet and not validate_python_ast(snippet):
                continue

            samples.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert programming assistant that writes clean, correct Python code.",
                        },
                        {"role": "user", "content": prob},
                        {"role": "assistant", "content": cleaned_sol},
                    ]
                }
            )
            if len(samples) >= budget:
                break
    except Exception as e:
        print(f"  ⚠️ Warning: Code harvest encountered an issue: {e}")

    print(f"  ✓ Collected {len(samples):,} code samples.")
    return samples


def harvest_reasoning_samples(budget: int) -> List[Dict[str, Any]]:
    """Harvests step-by-step reasoning samples with <think> tags."""
    print(f"\n[2/4] Harvesting {budget:,} Math/Reasoning samples with <think> (local cache)...")
    samples: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
        for row in ds:
            prob = (row.get("problem") or row.get("question") or "").strip()
            sol = (row.get("solution") or row.get("response") or "").strip()
            if not prob or not sol:
                continue

            # Ensure chain-of-thought formatting
            if "<think>" not in sol:
                sol = f"<think>\nAnalyze the problem and solve systematically.\n</think>\n{sol}"

            samples.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a mathematical reasoning model. Think step-by-step using <think>...</think> tags before providing the final answer.",
                        },
                        {"role": "user", "content": prob},
                        {"role": "assistant", "content": sol},
                    ]
                }
            )
            if len(samples) >= budget:
                break
    except Exception as e:
        print(f"  ⚠️ Warning: Reasoning harvest encountered an issue: {e}")

    print(f"  ✓ Collected {len(samples):,} reasoning samples.")
    return samples


def harvest_constraint_samples(budget: int) -> List[Dict[str, Any]]:
    """Harvests rule-following constraint samples from SmolTalk."""
    print(f"\n[3/4] Harvesting {budget:,} Constraint Task samples (local cache)...")
    samples: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceTB/smoltalk", "smol-constraints", split="train")
        for row in ds:
            msgs = row.get("messages", [])
            if len(msgs) >= 2:
                samples.append({"messages": msgs})
            if len(samples) >= budget:
                break
    except Exception as e:
        print(f"  ⚠️ Warning: Constraints harvest encountered an issue: {e}")

    print(f"  ✓ Collected {len(samples):,} constraint samples.")
    return samples


def harvest_chitchat_samples(budget: int) -> List[Dict[str, Any]]:
    """Harvests multi-turn conversational dialogue from SmolTalk."""
    print(f"\n[4/4] Harvesting {budget:,} Chit-Chat samples (local cache)...")
    samples: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")
        for row in ds:
            msgs = row.get("messages", [])
            if len(msgs) >= 2:
                samples.append({"messages": msgs})
            if len(samples) >= budget:
                break
    except Exception as e:
        print(f"  ⚠️ Warning: Dialogue harvest encountered an issue: {e}")

    print(f"  ✓ Collected {len(samples):,} conversational samples.")
    return samples


def prepare_sft_splits(total_samples: int = 15_000, output_dir: str = "data/sft"):
    """Coordinates composite collection, shuffles, and writes train.jsonl and val.jsonl."""
    print("=" * 70)
    print("   slm-gpt SFT 4-Pillar Dataset Assembler (Local Parquet Engine)")
    print("=" * 70)
    print(f"Target Budget: {total_samples:,} samples")

    os.makedirs(output_dir, exist_ok=True)

    # 4-Pillar distribution: 30% code, 30% reasoning, 25% constraints, 15% chit-chat
    budget_code = int(total_samples * 0.30)
    budget_reasoning = int(total_samples * 0.30)
    budget_constraints = int(total_samples * 0.25)
    budget_chitchat = total_samples - (budget_code + budget_reasoning + budget_constraints)

    dataset: List[Dict[str, Any]] = []
    dataset.extend(harvest_code_samples(budget_code))
    dataset.extend(harvest_reasoning_samples(budget_reasoning))
    dataset.extend(harvest_constraint_samples(budget_constraints))
    dataset.extend(harvest_chitchat_samples(budget_chitchat))

    random.seed(42)
    random.shuffle(dataset)

    val_count = max(50, int(len(dataset) * 0.05))
    train_count = len(dataset) - val_count

    train_data = dataset[:train_count]
    val_data = dataset[train_count:]

    # Canonical paths
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("✓ Successfully generated SFT datasets (Offline):")
    print(f"  Train: {train_path} ({len(train_data):,} samples)")
    print(f"  Val:   {val_path} ({len(val_data):,} samples)")
    print("=" * 70)


if __name__ == "__main__":
    kwargs: Dict[str, Any] = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            if k in ("total_samples", "max_samples", "budget"):
                kwargs["total_samples"] = int(v)
            elif k == "output_dir":
                kwargs["output_dir"] = v
        elif arg.isdigit():
            kwargs["total_samples"] = int(arg)

    prepare_sft_splits(**kwargs)