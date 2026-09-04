"""Streaming text generation for the chat UI.

GPTLanguageModel.generate() (in my_gpt.py) only returns the full sequence after every new token
has been produced, with no hook to observe progress. Rather than editing it, this re-implements
the same sampling loop and yields each newly generated chunk of text as it's produced, so the
browser can render a live typewriter effect.
"""
import torch
import torch.nn.functional as F

from . import checkpoints, config, tokenizer_utils


@torch.no_grad()
def generate_stream(filename, location, prompt, max_new_tokens, device=None):
    device = device or config.get_device()
    model, ckpt = checkpoints.load_model(filename, location, device)
    model_config = ckpt["model_config"]
    encode, decode, _ = tokenizer_utils.get_tokenizer(model_config["use_bpe"])

    ids = encode(prompt) if prompt else []
    if not ids:
        ids = [0]  # same empty-context seed train_gpt.py uses when sampling

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    block_size = model_config["block_size"]

    # Decode the whole running sequence each step and diff against what's already been sent,
    # rather than decoding one token at a time -- keeps multi-byte/multi-token characters intact.
    seen_len = len(decode(idx[0].tolist())) if prompt else 0

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        full_text = decode(idx[0].tolist())
        new_text = full_text[seen_len:]
        if new_text:
            seen_len = len(full_text)
            yield new_text
