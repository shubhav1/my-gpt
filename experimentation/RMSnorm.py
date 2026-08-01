import torch
import torch.nn as nn
import time


def bench(module, x, device, n_iters, warmup):
    module = module.to(device)
    for _ in range(warmup):
        y = module(x)
    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        y = module(x)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_iters * 1000  # ms/iter


def bench_bwd(module, x, device, n_iters, warmup):
    module = module.to(device)
    x = x.clone().requires_grad_(True)
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        
        y = module(x)
        y.sum().backward()
    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        y = module(x)
        y.sum().backward()
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_iters * 1000


def run(device):
    print(f"device: {device}, torch version: {torch.__version__}")

    B, T, C = 64, 256, 384  # matching my_gpt size
    n_iters = 200
    warmup = 20 # bc mps is slow/unpredictable on first few iterations

    x = torch.randn(B, T, C, device=device)

    ln_ms = bench(nn.LayerNorm(C), x, device, n_iters, warmup)
    rms_ms = bench(nn.RMSNorm(C), x, device, n_iters, warmup)
    print(f"LayerNorm: {ln_ms:.4f} ms/iter")
    print(f"RMSNorm:   {rms_ms:.4f} ms/iter")
    print(f"ratio (LN/RMS): {ln_ms/rms_ms:.3f}")

    ln_bwd = bench_bwd(nn.LayerNorm(C), x, device, n_iters, warmup)
    rms_bwd = bench_bwd(nn.RMSNorm(C), x, device, n_iters, warmup)
    print(f"\nLayerNorm fwd+bwd: {ln_bwd:.4f} ms/iter")
    print(f"RMSNorm fwd+bwd:   {rms_bwd:.4f} ms/iter")
    print(f"ratio (LN/RMS): {ln_bwd/rms_bwd:.3f}")

    # torch.compile: does Inductor fuse the decomposed RMSNorm ops
    # into a single Metal kernel on MPS?
    compiled_rms = torch.compile(nn.RMSNorm(C).to(device))

    compiled_rms_fwd = bench(compiled_rms, x, device, n_iters, warmup)
    compiled_rms_bwd = bench_bwd(compiled_rms, x, device, n_iters, warmup)

    print(f"\nRMSNorm (compiled): {compiled_rms_fwd:.4f} ms/iter")
    print(f"RMSNorm (compiled) fwd+bwd: {compiled_rms_bwd:.4f} ms/iter")
    print(f"ratio (LN/compiled RMS): {ln_ms/compiled_rms_fwd:.3f}")

    print(f"\nLayerNorm fwd+bwd:        {ln_bwd:.4f} ms/iter")
    print(f"RMSNorm (compiled) fwd+bwd: {compiled_rms_bwd:.4f} ms/iter")
    print(f"ratio (LN/compiled RMS): {ln_bwd/compiled_rms_bwd:.3f}")

    print(f"\ncompile speedup (forward): {rms_ms/compiled_rms_fwd:.3f}x")
    print(f"compile speedup (fwd+bwd): {rms_bwd/compiled_rms_bwd:.3f}x")


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run(device)


"""
Results:
MPS:
device: mps, torch version: 2.13.0
LayerNorm: 0.3833 ms/iter
RMSNorm:   2.5985 ms/iter
ratio (LN/RMS): 0.148

LayerNorm fwd+bwd: 3.3969 ms/iter
RMSNorm fwd+bwd:   9.6377 ms/iter
ratio (LN/RMS): 0.352

RMSNorm (compiled): 0.3987 ms/iter
RMSNorm (compiled) fwd+bwd: 2.5103 ms/iter
ratio (LN/compiled RMS): 0.962

LayerNorm fwd+bwd:        3.3969 ms/iter
RMSNorm (compiled) fwd+bwd: 2.5103 ms/iter
ratio (LN/compiled RMS): 1.353

compile speedup (forward): 6.517x
compile speedup (fwd+bwd): 3.839x


CPU:
device: cpu, torch version: 2.13.0
LayerNorm: 0.8677 ms/iter
RMSNorm:   2.6438 ms/iter
ratio (LN/RMS): 0.328

LayerNorm fwd+bwd: 3.7433 ms/iter
RMSNorm fwd+bwd:   11.9345 ms/iter
ratio (LN/RMS): 0.314

RMSNorm (compiled): 1.0237 ms/iter
RMSNorm (compiled) fwd+bwd: 8.3748 ms/iter
ratio (LN/compiled RMS): 0.848

LayerNorm fwd+bwd:        3.7433 ms/iter
RMSNorm (compiled) fwd+bwd: 8.3748 ms/iter
ratio (LN/compiled RMS): 0.447

compile speedup (forward): 2.583x
compile speedup (fwd+bwd): 1.425x

"""