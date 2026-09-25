"""
eval/evaluate.py: Native 5-Benchmark Evaluation Suite for slm-gpt.
Evaluates:
  1. Perplexity (val_ppl): Exact token-level cross-entropy on packed uint16 shards.
  2. Commonsense QA (hellaswag, arc_easy): Length-normalized log-likelihood ranking.
  3. Step-by-Step Reasoning (gsm8k): Autoregressive generation with <think> tag & answer checking.
  4. Code Execution (mbpp): Python generation validated against unit test assertions.
  5. Alignment Margin (dpo_margin): Chosen vs rejected log-likelihood delta verification.
"""

import ast
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
import numpy as np
import torch
import torch.nn.functional as F

from model.transformer import Transformer
from scaling.config import ModelConfig
from tokenizer.factory import get_tokenizer


# =====================================================================
# Checkpoint Loader & Scoring Helpers
# =====================================================================

def load_checkpoint(ckpt_path: str, device: str = "cuda") -> Tuple[Transformer, ModelConfig, Any]:
    """Loads model weights, configuration, and tokenizer from checkpoint file."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"\n[Loader] Loading weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # Resolve config
    if "config" in state and isinstance(state["config"], ModelConfig):
        cfg = state["config"]
    else:
        # Fallback to 127M architecture default if config object wasn't pickled directly
        cfg = ModelConfig(
            dim=768,
            n_layers=14,
            n_heads=12,
            n_kv_heads=4,
            vocab_size=50277,
            seq_len=2048,
        )

    model = Transformer(cfg).to(device)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)

    model_state = state.get("model", state)
    # Strip torch.compile prefixes if present
    cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    model.load_state_dict(cleaned_state)
    model.eval()

    tokenizer_type = getattr(cfg, "tokenizer_type", "tiktoken")
    tokenizer_path = getattr(cfg, "tokenizer_path", None)
    tokenizer = get_tokenizer(tokenizer_type, tokenizer_path)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Loader] Initialized model: {total_params:,} parameters (dim={cfg.dim}, layers={cfg.n_layers})")

    return model, cfg, tokenizer


@torch.no_grad()
def score_continuation_logprobs(
    model: Transformer,
    prompt_tokens: List[int],
    continuation_tokens: List[int],
    device: str = "cuda",
) -> Tuple[float, float]:
    """
    Computes (sum_logprob, avg_logprob) of continuation_tokens conditioned on prompt_tokens.
    """
    full_seq = prompt_tokens + continuation_tokens
    if len(full_seq) > model.cfg.seq_len:
        # Truncate left of prompt if exceeding context
        overflow = len(full_seq) - model.cfg.seq_len
        prompt_tokens = prompt_tokens[overflow:]
        full_seq = prompt_tokens + continuation_tokens

    x = torch.tensor([full_seq[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_seq[1:]], dtype=torch.long, device=device)

    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    with torch.amp.autocast(device_type=device, dtype=amp_dtype):
        logits = model(x)
        # Log-softmax over vocabulary
        log_probs = F.log_softmax(logits, dim=-1)

    # Slice strictly the target continuation tokens
    prompt_len = len(prompt_tokens)
    target_logits = log_probs[0, prompt_len - 1 :]
    target_tokens = y[0, prompt_len - 1 :]

    token_logprobs = target_logits.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    sum_logprob = token_logprobs.sum().item()
    avg_logprob = token_logprobs.mean().item()

    return sum_logprob, avg_logprob


# =====================================================================
# Task 1: Validation Perplexity (Packed Shard)
# =====================================================================

@torch.no_grad()
def eval_val_ppl(
    model: Transformer,
    val_bin: str,
    max_tokens: int = 2_000_000,
    device: str = "cuda",
) -> Dict[str, float]:
    if not os.path.exists(val_bin):
        print(f"  ⚠️ Validation file '{val_bin}' not found. Skipping.")
        return {"val_loss": float("nan"), "val_ppl": float("nan")}

    data = np.fromfile(val_bin, dtype=np.uint16)
    if len(data) > max_tokens:
        data = data[:max_tokens]

    seq_len = model.cfg.seq_len
    num_chunks = (len(data) - 1) // seq_len
    if num_chunks == 0:
        return {"val_loss": float("nan"), "val_ppl": float("nan")}

    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    total_loss = 0.0
    total_tokens = 0

    for i in range(num_chunks):
        start = i * seq_len
        x = torch.from_numpy(data[start : start + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(data[start + 1 : start + seq_len + 1].astype(np.int64)).unsqueeze(0).to(device)

        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")

        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20.0 else float("inf")
    return {"val_loss": round(avg_loss, 4), "val_ppl": round(ppl, 2)}


# =====================================================================
# Task 2: Log-Likelihood Multiple Choice (HellaSwag & ARC-Easy)
# =====================================================================

@torch.no_grad()
def eval_hellaswag(
    model: Transformer,
    tokenizer: Any,
    limit: int = 500,
    device: str = "cuda",
) -> Dict[str, float]:
    print(f"  • Running HellaSwag ({limit:,} samples)...")
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation")
    except Exception as e:
        print(f"  ⚠️ Error loading HellaSwag: {e}")
        return {"hellaswag_acc": float("nan")}

    correct = 0
    total = 0

    for item in ds:
        if total >= limit:
            break

        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        prompt_tokens = tokenizer.encode(ctx)
        avg_scores = []
        for ending in endings:
            cand_tokens = tokenizer.encode(" " + ending.lstrip())
            _, avg_lp = score_continuation_logprobs(model, prompt_tokens, cand_tokens, device=device)
            avg_scores.append(avg_lp)

        pred_label = int(np.argmax(avg_scores))
        if pred_label == label:
            correct += 1
        total += 1

    acc = (correct / max(1, total)) * 100.0
    return {"hellaswag_acc": round(acc, 2), "samples": total}


@torch.no_grad()
def eval_arc_easy(
    model: Transformer,
    tokenizer: Any,
    limit: int = 500,
    device: str = "cuda",
) -> Dict[str, float]:
    print(f"  • Running ARC-Easy ({limit:,} samples)...")
    try:
        ds = load_dataset("ai2_arc", "ARC-Easy", split="test")
    except Exception as e:
        print(f"  ⚠️ Error loading ARC-Easy: {e}")
        return {"arc_easy_acc": float("nan")}

    correct = 0
    total = 0

    for item in ds:
        if total >= limit:
            break

        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        prompt = f"Question: {question}\nAnswer:"
        prompt_tokens = tokenizer.encode(prompt)

        avg_scores = []
        for choice in choices:
            cand_tokens = tokenizer.encode(" " + choice.strip())
            _, avg_lp = score_continuation_logprobs(model, prompt_tokens, cand_tokens, device=device)
            avg_scores.append(avg_lp)

        pred_idx = int(np.argmax(avg_scores))
        pred_label = labels[pred_idx]

        if str(pred_label).strip().upper() == str(answer_key).strip().upper():
            correct += 1
        total += 1

    acc = (correct / max(1, total)) * 100.0
    return {"arc_easy_acc": round(acc, 2), "samples": total}


# =====================================================================
# Task 3: Reasoning & <think> Verification (GSM8K)
# =====================================================================

@torch.no_grad()
def eval_gsm8k(
    model: Transformer,
    tokenizer: Any,
    limit: int = 100,
    device: str = "cuda",
) -> Dict[str, Any]:
    print(f"  • Running GSM8K with <think> tracking ({limit:,} samples)...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"  ⚠️ Error loading GSM8K: {e}")
        return {"gsm8k_acc": float("nan")}

    correct = 0
    closed_think_count = 0
    total = 0
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    for item in ds:
        if total >= limit:
            break

        q = item["question"]
        gold_match = re.search(r"####\s*(-?[0-9\.,]+)", item["answer"])
        if not gold_match:
            continue
        gold_val = gold_match.group(1).replace(",", "").strip()

        # Aligns with reasoning format trained in Stage 0/1
        prompt = f"Problem:\n{q}\n\nSolution:\n<think>\n"
        input_ids = tokenizer.encode(prompt)
        curr_ids = list(input_ids)

        # Greedy decode up to 384 tokens
        for _ in range(384):
            context_window = curr_ids[-model.cfg.seq_len :]
            x = torch.tensor([context_window], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                logits = model(x)
            next_tok = int(torch.argmax(logits[0, -1, :]).item())
            curr_ids.append(next_tok)
            if next_tok == tokenizer.eot_id:
                break

        completion = tokenizer.decode(curr_ids[len(input_ids) :])

        if "</think>" in completion:
            closed_think_count += 1
            # Extract final answer that follows the closing tag
            after_think = completion.split("</think>")[-1]
            nums = re.findall(r"(-?[0-9]+(?:\.[0-9]+)?)", after_think)
        else:
            # Fallback to any trailing number
            nums = re.findall(r"(-?[0-9]+(?:\.[0-9]+)?)", completion)

        pred_val = nums[-1] if nums else None
        if pred_val and pred_val == gold_val:
            correct += 1

        total += 1

    acc = (correct / max(1, total)) * 100.0
    think_rate = (closed_think_count / max(1, total)) * 100.0
    return {
        "gsm8k_acc": round(acc, 2),
        "think_adherence": round(think_rate, 2),
        "samples": total,
    }


# =====================================================================
# Task 4: Code Generation & Execution (MBPP Sanitized)
# =====================================================================

@torch.no_grad()
def eval_mbpp(
    model: Transformer,
    tokenizer: Any,
    limit: int = 100,
    device: str = "cuda",
) -> Dict[str, Any]:
    print(f"  • Running MBPP Code Execution ({limit:,} samples)...")
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    except Exception:
        try:
            ds = load_dataset("Muennighoff/mbpp", "sanitized", split="test")
        except Exception as e:
            print(f"  ⚠️ Error loading MBPP: {e}")
            return {"mbpp_pass@1": float("nan")}

    passed = 0
    syntax_valid = 0
    total = 0
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    for item in ds:
        if total >= limit:
            break

        task_desc = item["prompt"]
        test_cases = item.get("test_list", [])

        prompt = (
            f"You are an expert programming assistant that writes clean, correct Python code.\n"
            f"Write a Python function to solve this task:\n{task_desc}\n\n```python\n"
        )
        input_ids = tokenizer.encode(prompt)
        curr_ids = list(input_ids)

        for _ in range(256):
            context_window = curr_ids[-model.cfg.seq_len :]
            x = torch.tensor([context_window], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                logits = model(x)
            next_tok = int(torch.argmax(logits[0, -1, :]).item())
            curr_ids.append(next_tok)
            if next_tok == tokenizer.eot_id:
                break

        completion = tokenizer.decode(curr_ids[len(input_ids) :])
        # Extract code body up to markdown fence or end of text
        code_body = completion.split("```")[0].strip()

        # AST syntax check
        try:
            ast.parse(code_body)
            syntax_valid += 1
            is_syntactic = True
        except Exception:
            is_syntactic = False

        # Unit test execution check
        is_pass = False
        if is_syntactic and test_cases:
            try:
                env: Dict[str, Any] = {}
                exec(code_body, env, env)
                all_tests_passed = True
                for test in test_cases:
                    exec(test, env, env)
                is_pass = all_tests_passed
            except Exception:
                is_pass = False

        if is_pass:
            passed += 1
        total += 1

    pass_rate = (passed / max(1, total)) * 100.0
    syntax_rate = (syntax_valid / max(1, total)) * 100.0
    return {
        "mbpp_pass@1": round(pass_rate, 2),
        "syntax_validity": round(syntax_rate, 2),
        "samples": total,
    }


# =====================================================================
# Task 5: DPO Preference Margin & Accuracy Check
# =====================================================================

@torch.no_grad()
def eval_dpo_margin(
    model: Transformer,
    tokenizer: Any,
    dpo_file: str = "data/dpo/preference_pairs.jsonl",
    limit: int = 200,
    device: str = "cuda",
) -> Dict[str, Any]:
    print(f"  • Running DPO Alignment Margin Check ({limit:,} pairs)...")
    if not os.path.exists(dpo_file):
        print(f"  ⚠️ DPO file '{dpo_file}' not found. Skipping.")
        return {"dpo_pref_acc": float("nan"), "mean_margin": float("nan")}

    pairs = []
    with open(dpo_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
                if len(pairs) >= limit:
                    break

    if not pairs:
        return {"dpo_pref_acc": float("nan"), "mean_margin": float("nan")}

    chosen_preferred = 0
    margins = []

    for item in pairs:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        prompt_tokens = tokenizer.encode(prompt)
        chosen_tokens = tokenizer.encode(chosen)
        rejected_tokens = tokenizer.encode(rejected)

        sum_lp_chosen, _ = score_continuation_logprobs(model, prompt_tokens, chosen_tokens, device=device)
        sum_lp_rejected, _ = score_continuation_logprobs(model, prompt_tokens, rejected_tokens, device=device)

        margin = sum_lp_chosen - sum_lp_rejected
        margins.append(margin)

        if sum_lp_chosen > sum_lp_rejected:
            chosen_preferred += 1

    acc = (chosen_preferred / max(1, len(pairs))) * 100.0
    avg_margin = float(np.mean(margins))
    return {
        "dpo_pref_acc": round(acc, 2),
        "mean_margin": round(avg_margin, 4),
        "samples": len(pairs),
    }


# =====================================================================
# Main Orchestrator
# =====================================================================

def run_evaluation(
    ckpt_path: str,
    val_shard: str = "data/pretrain/val_00000.bin",
    dpo_path: str = "data/dpo/preference_pairs.jsonl",
    limit: int = 200,
    tasks: Optional[List[str]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if tasks is None or "all" in tasks:
        tasks = ["val_ppl", "hellaswag", "arc_easy", "gsm8k", "mbpp", "dpo_margin"]

    model, cfg, tokenizer = load_checkpoint(ckpt_path, device=device)

    print("\n" + "=" * 75)
    print("               slm-gpt Comprehensive Evaluation Suite")
    print("=" * 75)
    print(f"Target Checkpoint: {ckpt_path}")
    print(f"Evaluation Device: {device.upper()}")
    print(f"Active Tasks:      {', '.join(tasks)}")
    print(f"Sample Limit:      {limit} per task")
    print("-" * 75)

    results: Dict[str, Any] = {}

    if "val_ppl" in tasks:
        res = eval_val_ppl(model, val_bin=val_shard, device=device)
        results.update(res)
        print(f"  ✓ Val Loss: {res['val_loss']} | Val PPL: {res['val_ppl']}")

    if "hellaswag" in tasks:
        res = eval_hellaswag(model, tokenizer, limit=limit, device=device)
        results.update(res)
        print(f"  ✓ HellaSwag Acc: {res['hellaswag_acc']}%")

    if "arc_easy" in tasks:
        res = eval_arc_easy(model, tokenizer, limit=limit, device=device)
        results.update(res)
        print(f"  ✓ ARC-Easy Acc: {res['arc_easy_acc']}%")

    if "gsm8k" in tasks:
        res = eval_gsm8k(model, tokenizer, limit=min(limit, 100), device=device)
        results.update(res)
        print(f"  ✓ GSM8K Acc: {res['gsm8k_acc']}% | <think> Closing Rate: {res['think_adherence']}%")

    if "mbpp" in tasks:
        res = eval_mbpp(model, tokenizer, limit=min(limit, 100), device=device)
        results.update(res)
        print(f"  ✓ MBPP pass@1: {res['mbpp_pass@1']}% | Syntax Validity: {res['syntax_validity']}%")

    if "dpo_margin" in tasks:
        res = eval_dpo_margin(model, tokenizer, dpo_file=dpo_path, limit=limit, device=device)
        results.update(res)
        print(f"  ✓ DPO Preference Acc: {res['dpo_pref_acc']}% | Mean Margin: {res['mean_margin']}")

    print("=" * 75)
    print("Summary Results Table:")
    for k, v in results.items():
        if k != "samples":
            print(f"  • {k:<25}: {v}")
    print("=" * 75)


if __name__ == "__main__":
    kwargs: Dict[str, Any] = {
        "ckpt_path": "checkpoints/pretrain_127M/pretrain_final.pt",
        "val_shard": "data/pretrain/val_00000.bin",
        "dpo_path": "data/dpo/preference_pairs.jsonl",
        "limit": 200,
        "tasks": None,
    }

    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            if k in ("ckpt", "ckpt_path", "checkpoint"):
                kwargs["ckpt_path"] = v
            elif k in ("val_shard", "val_bin"):
                kwargs["val_shard"] = v
            elif k in ("dpo_path", "dpo_file"):
                kwargs["dpo_path"] = v
            elif k in ("limit", "samples", "n"):
                kwargs["limit"] = int(v)
            elif k == "tasks":
                kwargs["tasks"] = [t.strip() for t in v.split(",")]
            elif k == "device":
                kwargs["device"] = v

    run_evaluation(**kwargs)