"""Builds train/val tensors the same way train_gpt.py does, parameterized by whichever
tokenizer the caller picked instead of a hardcoded USE_BPE flag.
"""
import torch

from . import config


def load_text(path=None):
    with open(path or config.DATA_PATH) as f:
        return f.read()


def build_dataset(text, encode, decode, use_bpe):
    """Returns (train_data, val_data, cpt_train, cpt_val)."""
    if use_bpe:
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data, val_data = data[:n], data[n:]
    else:
        n = int(0.9 * len(text))
        train_data = torch.tensor(encode(text[:n]), dtype=torch.long)
        val_data = torch.tensor(encode(text[n:]), dtype=torch.long)

    cpt_train, cpt_val = 1.0, 1.0
    if use_bpe:
        cpt_train = len(decode(train_data.tolist())) / len(train_data)
        cpt_val = len(decode(val_data.tolist())) / len(val_data)

    return train_data, val_data, cpt_train, cpt_val
