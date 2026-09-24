import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from transformer import ModelConfig, DecoderOnlyTransformer
from tokenizer import PretrainedTiktokenTokenizer


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_with_penalties(
    model: DecoderOnlyTransformer,
    idx: torch.Tensor,
    max_new_tokens: int,
    eot_token_id: int,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Temperature scaling
        logits = logits / max(temperature, 1e-5)

        # Top-K
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Top-P (Nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Stop immediately if EOT token is emitted
        if next_token.item() == eot_token_id:
            break

        idx = torch.cat((idx, next_token), dim=1)

    return idx


def extract_corpus_samples(bin_path: Path, tokenizer, num_samples: int = 3, prefix_len: int = 15, ground_truth_len: int = 40):
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    eot_id = tokenizer.encode("<|endoftext|>")[0]
    total_tokens = len(tokens)
    samples = []
    
    attempts = 0
    while len(samples) < num_samples and attempts < 200:
        attempts += 1
        pos = random.randint(0, total_tokens - (prefix_len + ground_truth_len + 5000))
        slice_window = np.array(tokens[pos : pos + 5000])
        eot_locs = np.where(slice_window == eot_id)[0]
        
        if len(eot_locs) > 0:
            start_idx = pos + eot_locs[0] + 1
            if start_idx + prefix_len + ground_truth_len < total_tokens:
                doc_tokens = tokens[start_idx : start_idx + prefix_len + ground_truth_len].tolist()
                if eot_id not in doc_tokens[:prefix_len]:
                    prefix = doc_tokens[:prefix_len]
                    ground_truth = doc_tokens[prefix_len:]
                    samples.append((prefix, ground_truth))
                    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on in-corpus test prompts")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fineweb_edu_chinchilla_17m.pt")
    parser.add_argument("--data_bin", type=str, default="data/processed/fineweb_edu_100m/test.bin")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--prefix_len", type=int, default=15)
    parser.add_argument("--max_gen_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--repetition_penalty", type=float, default=1.25)

    args = parser.parse_args()
    device = resolve_device()

    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint '{checkpoint_path}' on {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = DecoderOnlyTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = PretrainedTiktokenTokenizer("gpt2")
    eot_token_id = tokenizer.encode("<|endoftext|>")[0]

    bin_path = Path(args.data_bin)
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing {bin_path}. Check your data path.")

    corpus_samples = extract_corpus_samples(
        bin_path, tokenizer, num_samples=args.num_samples, prefix_len=args.prefix_len
    )

    print(f"\nExtracted {len(corpus_samples)} holdout documents from {bin_path.name}\n" + "=" * 70)

    for i, (prefix_tokens, ground_truth_tokens) in enumerate(corpus_samples, 1):
        prompt_text = tokenizer.decode(prefix_tokens)
        ground_truth_text = tokenizer.decode(ground_truth_tokens)

        x = torch.tensor([prefix_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            out_tensor = sample_with_penalties(
                model=model,
                idx=x,
                max_new_tokens=args.max_gen_tokens,
                eot_token_id=eot_token_id,
                temperature=args.temperature,
                top_k=40,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

        out_tokens = out_tensor[0].tolist()
        gen_tokens = out_tokens[len(prefix_tokens):]
        if eot_token_id in gen_tokens:
            gen_tokens = gen_tokens[:gen_tokens.index(eot_token_id)]
            
        gen_text = tokenizer.decode(gen_tokens).strip()

        print(f"\n[Test Case {i}]")
        print(f"┌─ PROMPT (from test split):")
        print(f"│  \033[1m\"{prompt_text.strip()}\"\033[0m")
        print(f"├─ MODEL GENERATION:")
        print(f"│  \033[36m{gen_text}\033[0m")
        print(f"└─ GROUND TRUTH CONTINUATION:")
        print(f"   \033[32m{ground_truth_text.strip()[:200]}...\033[0m")
        print("-" * 70)


if __name__ == "__main__":
    main()