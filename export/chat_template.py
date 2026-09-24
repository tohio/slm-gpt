"""
export/chat_template.py: ChatML Jinja template and Hugging Face generation configs for slm-gpt.
"""

import json
import os
from typing import Any, Dict

CHATML_JINJA_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def build_generation_config(
    vocab_size: int,
    eos_token_id: int,
    pad_token_id: int,
    bos_token_id: int = 50256,
) -> Dict[str, Any]:
    """Generates standard generation_config.json."""
    return {
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "max_new_tokens": 512,
        "transformers_version": "4.44.0",
    }


def build_tokenizer_config(
    tokenizer,
    model_max_length: int = 2048,
) -> Dict[str, Any]:
    """Builds tokenizer_config.json embedding the standard ChatML Jinja template."""
    return {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            str(tokenizer.eot_id): {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            str(tokenizer.im_start_id): {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            str(tokenizer.im_end_id): {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
        },
        "bos_token": "<|endoftext|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "chat_template": CHATML_JINJA_TEMPLATE,
        "model_max_length": model_max_length,
        "tokenizer_class": "GPT2Tokenizer",
    }


def write_generation_assets(output_dir: str, tokenizer, max_seq_len: int = 2048):
    os.makedirs(output_dir, exist_ok=True)

    gen_cfg = build_generation_config(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.im_end_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    with open(os.path.join(output_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)

    tok_cfg = build_tokenizer_config(tokenizer, model_max_length=max_seq_len)
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2)