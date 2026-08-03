# my-gpt
This repository began as a from-scratch implementation of a GPT-style decoder-only transformer. It has since evolved into an experimental playground for modern language models. I'm implementing architectural and training changes one at a time, running controlled ablation studies, and documenting both the results and the reasoning behind them.

Data: `shakespeare.txt` (~1.1M chars). Model code lives in `my_gpt.py`; training entrypoint is `train_gpt.py`. BPE is under `BPE/`.

## Architecture

Decoder-only transformer with causal self-attention, pre-norm residual blocks, and a final LM head. Additional architectural updates have been made (e.g., RMSNorm, RoPE, SwiGLU), which are detailed in the Modifications section. These are the most up-to-date hyperparameters I'm using for my experiments:

| Hyperparameter | Value |
|---|---|
| `batch_size` | 64 |
| `block_size` | 256 |
| `max_iters` | 2400 |
| `eval_interval` | 200 |
| `learning_rate` | 3e-4 |
| `eval_iters` | 200 |
| `n_embd` | 300 |
| `n_head` | 3 |
| `n_layer` | 3 |
| `dropout` | 0.2 |

Vocab size is 256 for the byte-level baseline, or `256 + num_merges` with BPE (500 merges → 756).


Device: MPS if available, else CPU (`train_gpt.py`).

## Experiments

Each experiment is documented as a standalone write-up in `experimentation/`.

Every write-up contains:
- Motivation
- Hypothesis
- Experimental setup
- Results
- Limitations
- Follow-up questions

## Modifications explored

This started an implementation of the original transformer, guided by Andrej Karpathy's zero to hero course. As I learned more about modern innovations, I updated the code along the way and ran experiments to understand how they actually affect this model, test my hypotheses, and gain a strong intuition for why these innovations exist and are used. These are the modifications I explored, all of which you can find in the `experimentation/` folder:

- **BPE tokenization:** implemented from-scratch BPE tokenizer for the shakespeare text. compared byte-level and BPE tokenization across multiple merge counts, evaluating convergence speed, bits-per-byte, and generation quality. (see `experimentation/01_BPE.md`)
- **Mixed precision (bf16):** evaluated mixed precision using torch autocast on Apple MPS and investigated why its expected CUDA speedups did not appear on this hardware when compared to fp32 (see `experimentation/02_bf16.md`). not using this one for now, sticking to fp32 until I can run on CUDA with tensor cores.
- **PreNorm:** compared PreNorm and PostNorm to investigate whether normalization placement affected optimization on a shallow transformer. applied layernorm out of residual stream, before attention and ffwd blocks (see `experimentation/03_prenorm.md`)
- ** RMSNorm:** switched to RMSNorm & removed bias use from all linear layers. compared RMSNorm and LayerNorm (see `experimentation/04_RMSnorm.md`, now using torch.compile for RMSNorm layers to speed up training)
- **Feedforward activations:** compared ReLU, GELU, and SwiGLU to study the tradeoff between expressiveness and overfitting. choosing SwiGLU for future runs (see `experimentation/05_activations.md`)
- **Model scaling:** realized how horrendous my overfitting was and investigated causes by reducing model depth and width to better match the dataset (see `experimentation/06_overfitting.md`). ended up reducing number of layers and heads to 3 each, and reducing n_embd from 384 to 300 for future runs
- **RoPE:** implemented RoPE and ran ablation comparing rotary and learned positional embeddings to study their impact on convergence and validation performance (see `experimentation/07_RoPE.md`)
- **Weight decay:** tested a variety of different weight decays to use with AdamW optimizer to explore what would best address overfitting issues (see `experimentation/08_weight_decay.md`)

I might implement gradient accumulation, activation checkpointing, and/or FlashAttention. Besides that, this repo is complete for now! I'm going to move onto different projects to explore beyond what this project can teach me. 


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