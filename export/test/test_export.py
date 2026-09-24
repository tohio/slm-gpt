"""
export/test/test_export.py: Automated regression tests for SafeTensors export and HF compatibility.
"""

from dataclasses import asdict
import json
import os
import shutil
import tempfile
import pytest
from safetensors import safe_open
import torch

from export.chat_template import CHATML_JINJA_TEMPLATE, build_tokenizer_config
from export.convert import convert_checkpoint_to_hf, remap_state_dict_to_hf_llama
from tokenizer.factory import get_tokenizer
from transformer import DecoderOnlyTransformer, ModelConfig


@pytest.fixture
def temp_export_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_remap_state_dict_llama():
    cfg = ModelConfig(
        vocab_size=1024,
        max_seq_len=128,
        d_model=192,
        n_layers=2,
        n_heads=3,
        n_kv_heads=1,
        d_ffn=512,
        dropout=0.0,
    )
    model = DecoderOnlyTransformer(cfg)
    sd = model.state_dict()

    hf_sd = remap_state_dict_to_hf_llama(sd, cfg)

    # Invariants: standard LLaMA naming
    assert "model.embed_tokens.weight" in hf_sd
    assert "lm_head.weight" in hf_sd
    assert "model.norm.weight" in hf_sd
    assert "model.layers.0.input_layernorm.weight" in hf_sd
    assert "model.layers.0.self_attn.q_proj.weight" in hf_sd
    assert "model.layers.0.self_attn.k_proj.weight" in hf_sd
    assert "model.layers.0.self_attn.v_proj.weight" in hf_sd
    assert "model.layers.0.self_attn.o_proj.weight" in hf_sd
    assert "model.layers.0.mlp.gate_proj.weight" in hf_sd
    assert "model.layers.0.mlp.up_proj.weight" in hf_sd
    assert "model.layers.0.mlp.down_proj.weight" in hf_sd


def test_end_to_end_safetensors_conversion(temp_export_dir):
    cfg = ModelConfig(
        vocab_size=1024,
        max_seq_len=128,
        d_model=192,
        n_layers=2,
        n_heads=3,
        n_kv_heads=1,
        d_ffn=512,
        dropout=0.0,
    )
    model = DecoderOnlyTransformer(cfg)

    ckpt_path = os.path.join(temp_export_dir, "test_ckpt.pt")
    out_dir = os.path.join(temp_export_dir, "hf_output")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "runtime_meta": {"size_tag": "test", "tokenizer_type": "tiktoken"},
        },
        ckpt_path,
    )

    convert_checkpoint_to_hf(ckpt_path, out_dir)

    # Verify generated artifacts
    assert os.path.exists(os.path.join(out_dir, "model.safetensors"))
    assert os.path.exists(os.path.join(out_dir, "config.json"))
    assert os.path.exists(os.path.join(out_dir, "generation_config.json"))
    assert os.path.exists(os.path.join(out_dir, "tokenizer_config.json"))

    # Verify config parameters
    with open(os.path.join(out_dir, "config.json"), "r") as f:
        c = json.load(f)
        assert c["architectures"] == ["LlamaForCausalLM"]
        assert c["model_type"] == "llama"
        assert c["hidden_size"] == 192
        assert c["tie_word_embeddings"] is True

    # Verify SafeTensors integrity
    with safe_open(os.path.join(out_dir, "model.safetensors"), framework="pt") as f:
        keys = f.keys()
        assert "model.embed_tokens.weight" in keys
        tensor = f.get_tensor("model.embed_tokens.weight")
        assert tensor.dtype == torch.bfloat16


def test_chatml_template_generation():
    tok = get_tokenizer("tiktoken")
    cfg = build_tokenizer_config(tok)

    assert cfg["chat_template"] == CHATML_JINJA_TEMPLATE
    assert "<|im_start|>" in cfg["chat_template"]
    assert "<|im_end|>" in cfg["chat_template"]
    assert cfg["eos_token"] == "<|im_end|>"