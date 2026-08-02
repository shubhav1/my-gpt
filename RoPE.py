import torch

def precompute_rope(head_size, block_size):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size)) # (head_size/2), ω_i
    positions = torch.arange(block_size, dtype=torch.float32) # (block_size)
    angles = positions[:, None] * inv_freq[None, :] # (block_size, head_size/2), θ_{pos, i} = pos * ω_i

    cos = torch.cos(angles) # (block_size, head_size/2)
    sin = torch.sin(angles) # (block_size, head_size/2)

    return cos, sin

def apply_rope(x, cos, sin):
    B, T, head_size = x.shape
    cos = cos[:T].to(dtype=x.dtype) # (1, T, head_size/2)
    sin = sin[:T].to(dtype=x.dtype) # (1, T, head_size/2)

    x_even = x[:, :, ::2]
    x_odd = x[:, :, 1::2]

    out = torch.empty_like(x)

    out[:, :, ::2] = x_even * cos - x_odd * sin
    out[:, :, 1::2] = x_even * sin + x_odd * cos

    return out