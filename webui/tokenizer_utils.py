"""Wraps the project's existing tokenizer so the web UI can request either BPE or byte-level
encode/decode, matching the branching logic at the top of train_gpt.py exactly.
"""
from BPE.bpe_tokenizer import decoder, encoder, load_tokenizer

from . import config

_bpe_vocab_size = None


def _ensure_bpe_loaded():
    global _bpe_vocab_size
    if _bpe_vocab_size is None:
        _, vocab = load_tokenizer(config.TOKENIZER_PATH)
        _bpe_vocab_size = len(vocab)
    return _bpe_vocab_size


def get_tokenizer(use_bpe):
    """Returns (encode_fn, decode_fn, vocab_size)."""
    if use_bpe:
        vocab_size = _ensure_bpe_loaded()

        def encode(text):
            return [tok for row in encoder(text) for tok in row]

        def decode(ids):
            return decoder([ids])

        return encode, decode, vocab_size

    def encode(text):
        return list(text.encode("utf-8"))

    def decode(ids):
        return bytes(ids).decode("utf-8", errors="replace")

    return encode, decode, 256
