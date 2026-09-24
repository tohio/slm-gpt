"""tests/test_sft.py: Unit tests for SFT Dataset and Prompt-Masked Collator."""

import pytest
import torch
from sft.collator import SFTDataCollator
from sft.dataset import IGNORE_INDEX, SFTDataset
from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer


@pytest.fixture
def tokenizer():
    return PretrainedTiktokenTokenizer()


def test_prompt_masking_contract(tokenizer):
    dialogue = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": "Hello!"},
    ]

    dataset = SFTDataset([dialogue], tokenizer=tokenizer, max_seq_len=512)
    sample = dataset[0]

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    assert len(input_ids) == len(
        labels
    ), "Length mismatch between input_ids and labels."

    # Identify where the assistant header '<|im_start|>assistant\n' begins
    assistant_tokens = tokenizer.encode("<|im_start|>assistant\n")
    window_len = len(assistant_tokens)

    asst_header_start = -1
    for i in range(len(input_ids) - window_len + 1):
        if input_ids[i : i + window_len] == assistant_tokens:
            asst_header_start = i
            break

    assert (
        asst_header_start != -1
    ), "Could not find assistant turn in encoded sequence."

    # Verify everything before assistant completion is masked with IGNORE_INDEX
    asst_body_start = asst_header_start + window_len
    for i in range(asst_body_start):
        assert (
            labels[i] == IGNORE_INDEX
        ), f"Token at index {i} was not masked! Token ID: {input_ids[i]}"

    # Verify assistant completion is unmasked
    supervised_labels = labels[asst_body_start:]
    assert all(
        l != IGNORE_INDEX for l in supervised_labels
    ), "Assistant response contains unintended -100 masks."

    # Verify final tokens are <|im_end|> and newline
    assert input_ids[-2] == tokenizer.im_end_id
    assert input_ids[-1] == tokenizer.newline_id
    assert labels[-2] == tokenizer.im_end_id
    assert labels[-1] == tokenizer.newline_id


def test_collator_dynamic_padding_and_alignment():
    features = [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5],
        },
        {
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "labels": [IGNORE_INDEX] * 10 + [11, 12, 13, 14, 15],
        },
    ]

    collator = SFTDataCollator(pad_token_id=50256, pad_to_multiple_of=16)
    batch = collator(features)

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    # Max length is 15 -> rounded to next multiple of 16 is 16
    assert input_ids.shape == (2, 16)
    assert labels.shape == (2, 16)
    assert attention_mask.shape == (2, 16)

    # Check padding positions on first sample (seq_len = 5, pad_len = 11)
    assert (input_ids[0, 5:] == 50256).all()
    assert (labels[0, 5:] == IGNORE_INDEX).all()
    assert (attention_mask[0, 5:] == 0).all()
    assert (attention_mask[0, :5] == 1).all()