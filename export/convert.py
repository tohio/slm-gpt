"""
export/convert.py: Converts slm-gpt PyTorch checkpoints to Hugging Face SafeTensors.
Maps exact internal transformer keys (including fused kv_proj and c_proj)
to standard LlamaForCausalLM schema for zero-code execution.
"""

from dataclasses import asdict
import json
import os
import sys
from typing import Any, Dict, Optional

from safetensors.torch import save_file
import torch

from export.chat_template import write_generation_assets
from tokenizer.factory import get_tokenizer
from transformer.config import ModelConfig


def remap_state_dict_to_hf_llama(
    custom_state_dict: Dict[str, torch.Tensor],
    cfg: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """
    Remaps slm-gpt internal state dict to Hugging Face LlamaForCausalLM naming conventions.
    Extracts k_proj and v_proj from fused kv_proj, and maps c_proj to o_proj.
    """
    hf_dict: Dict[str, torch.Tensor] = {}

    for k, v in custom_state_dict.items():
        tensor = v.contiguous()

        # 1. Global Embeddings & Final Norm
        if k in ("wte.weight", "embed.weight"):
            hf_dict["model.embed_tokens.weight"] = tensor
        elif k == "lm_head.weight":
            hf_dict["lm_head.weight"] = tensor
        elif k in ("ln_f.weight", "norm.weight"):
            hf_dict["model.norm.weight"] = tensor

        # 2. Transformer Layers (layers.X)
        elif k.startswith("layers.") or k.startswith("blocks."):
            parts = k.split(".")
            layer_idx = parts[1]
            submodule = ".".join(parts[2:])

            # Norms
            if submodule in ("ln_1.weight", "attn_norm.weight"):
                hf_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = tensor
            elif submodule in ("ln_2.weight", "ffn_norm.weight"):
                hf_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = tensor

            # Self-Attention
            elif submodule in ("attn.q_proj.weight", "self_attn.q_proj.weight"):
                hf_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = tensor

            # Fused KV projection -> split into k_proj and v_proj
            elif submodule in ("attn.kv_proj.weight", "self_attn.kv_proj.weight"):
                k_w, v_w = torch.chunk(tensor, 2, dim=0)
                hf_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_w.contiguous()
                hf_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_w.contiguous()

            # Separate K and V (if already split)
            elif submodule in ("attn.k_proj.weight", "self_attn.k_proj.weight"):
                hf_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = tensor
            elif submodule in ("attn.v_proj.weight", "self_attn.v_proj.weight"):
                hf_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = tensor

            # Output projection (c_proj -> o_proj)
            elif submodule in ("attn.c_proj.weight", "attn.out_proj.weight", "attn.o_proj.weight"):
                hf_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = tensor

            # SwiGLU MLP (w_gate, w_up, w_down)
            elif submodule in ("mlp.w_gate.weight", "ffn.w_gate.weight"):
                hf_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = tensor
            elif submodule in ("mlp.w_up.weight", "ffn.w_up.weight"):
                hf_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = tensor
            elif submodule in ("mlp.w_down.weight", "ffn.w_down.weight"):
                hf_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = tensor
            else:
                hf_dict[k] = tensor
        else:
            hf_dict[k] = tensor

    # Guarantee lm_head exists (tied embeddings)
    if "lm_head.weight" not in hf_dict and "model.embed_tokens.weight" in hf_dict:
        hf_dict["lm_head.weight"] = hf_dict["model.embed_tokens.weight"]

    return hf_dict


def build_hf_config(cfg: ModelConfig) -> Dict[str, Any]:
    """Constructs a standard Hugging Face LLaMA-compatible config.json."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": cfg.dropout,
        "bos_token_id": 50256,
        "eos_token_id": 50258,
        "hidden_act": "silu",
        "hidden_size": cfg.d_model,
        "initializer_range": 0.02,
        "intermediate_size": cfg.d_ffn,
        "max_position_embeddings": cfg.max_seq_len,
        "model_type": "llama",
        "num_attention_heads": cfg.n_heads,
        "num_hidden_layers": cfg.n_layers,
        "num_key_value_heads": cfg.n_kv_heads,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "rope_theta": getattr(cfg, "rope_theta", 10000.0),
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.44.0",
        "use_cache": True,
        "vocab_size": cfg.vocab_size,
    }


def convert_checkpoint_to_hf(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_type: str = "tiktoken",
    tokenizer_path: Optional[str] = None,
    target_dtype: torch.dtype = torch.bfloat16,
) -> str:
    """
    Main conversion routine.
    Loads checkpoint.pt -> Remaps keys -> Casts to target dtype -> Saves safetensors + configs.
    """
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at '{checkpoint_path}'"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 1. Resolve Config
    cfg_raw = ckpt.get("config")
    if isinstance(cfg_raw, dict):
        cfg = ModelConfig(**cfg_raw)
    elif isinstance(cfg_raw, ModelConfig):
        cfg = cfg_raw
    else:
        raise ValueError("Checkpoint does not contain valid 'config' metadata.")

    # 2. Extract and Remap State Dict
    state_dict = ckpt.get("model_state_dict", ckpt)
    hf_state_dict = remap_state_dict_to_hf_llama(state_dict, cfg)

    # 3. Cast to target dtype (bfloat16)
    for k in list(hf_state_dict.keys()):
        hf_state_dict[k] = hf_state_dict[k].to(dtype=target_dtype)

    # 4. Save SafeTensors
    weights_path = os.path.join(output_dir, "model.safetensors")
    save_file(hf_state_dict, weights_path, metadata={"format": "pt"})
    print(f"✓ Saved SafeTensors weights: {weights_path} ({os.path.getsize(weights_path) / (1024**2):.2f} MB)")

    # 5. Save config.json
    hf_cfg = build_hf_config(cfg)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_cfg, f, indent=2)
    print(f"✓ Saved HF Config: {config_path}")

    # 6. Save Generation & Tokenizer Configs
    meta = ckpt.get("runtime_meta", {})
    tok_t = tokenizer_type or meta.get("tokenizer_type", "tiktoken")
    tok_p = tokenizer_path or meta.get("tokenizer_path", None)
    tokenizer = get_tokenizer(tok_t, tok_p)

    write_generation_assets(output_dir, tokenizer, max_seq_len=cfg.max_seq_len)
    print("✓ Saved Tokenizer & Generation Configs (ChatML Jinja template enabled)")

    return output_dir


if __name__ == "__main__":
    ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/dpo_127M/dpo_final.pt"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "hf_export"
    convert_checkpoint_to_hf(ckpt_arg, out_arg)
