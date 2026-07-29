# my-gpt
This repository began as a from-scratch implementation of a GPT-style decoder-only transformer. It has since evolved into a place to experiment with different architectural and training decisions. I'm implementing changes one at a time, running controlled ablation studies, and documenting both the results and the reasoning behind them.

Data: `shakespeare.txt` (~1.1M chars). Model code lives in `my_gpt.py`; training entrypoint is `train_gpt.py`. BPE is under `BPE/`.

## Architecture

Decoder-only transformer with causal self-attention, pre-norm residual blocks, and a final LM head. Hyperparameters (as used for the BPE ablation plan in `experimentation/BPE.md`):

| Hyperparameter | Value |
|---|---|
| `batch_size` | 64 |
| `block_size` | 256 |
| `max_iters` | 2000 (ablation target; see Status) |
| `eval_interval` | 200 |
| `learning_rate` | 3e-4 |
| `eval_iters` | 200 |
| `n_embd` | 384 |
| `n_head` | 6 |
| `n_layer` | 6 |
| `dropout` | 0.2 |

Vocab size is 256 for the byte-level baseline, or `256 + num_merges` with BPE (500 merges → 756). This may change based on my experiments, but the ablation plan is to hold all hyperparameters constant except for merge count.


Device: MPS if available, else CPU (`train_gpt.py`).

## Updates

This started as replication of the Attention Is All You Need paper, guided by Andrej Karpathy's zero to hero course. Now, as I learn more, I'm updating the code along the way (and running small experiments sometimes). Here is a live list of updates:

- implemented BPE tokenizer and ran ablation comparing BPE vs UTF-8 byte-level baseline (see `experimentation/BPE.md`)
- implemented bf16 automatic mixed precision in training using torch autocast and ran ablation comparing bf16 vs fp32 (see `experimentation/bf16.md`). not using this one for now, sticking to fp32 until I can run on CUDA with tensor cores.
- switched to prenorm, applying layernorm out of residual stream, before attention and ffwd blocks (see `experimentation/prenorm.md`)
- switched to RMSNorm & removed bias use from all linear layers (see `experimentation/RMSnorm.md`)


## How to run

```bash
# from repo root
uv venv .venv          # or: python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
# loss curves also need matplotlib (not in requirements.txt yet):
uv pip install matplotlib
```

**Train the BPE tokenizer:**

```bash
cd BPE
# shakespeare.txt is at repo root — symlink if needed:
ln -sf ../shakespeare.txt shakespeare.txt
python train_tokenizer.py   # writes BPE/tokenizer.json (500 merges by default)
cd ..
```

**Train the model:**

You can toggle on and off different modes for architectural/training decisions (e.g. BPE, bf16, etc.) and stat tracking (printing bpb, loss curves, etc.) by setting the appropriate flags in `train_gpt.py` as True/False. For example, to use BPE tokenizer as opposed to utf-8 byte-level baseline, set `USE_BPE = True` in `train_gpt.py`.

Then run:

```bash
python train_gpt.py
```