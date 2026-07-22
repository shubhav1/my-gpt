# my-gpt

From-scratch nanoGPT-style transformer on Tiny Shakespeare. I'm currently experimenting with BPE tokenization and measuring its effect on training speed, loss, and generation quality.

Data: `shakespeare.txt` (~1.1M chars). Model code lives in `my_gpt.py`; training entrypoint is `train_gpt.py`. BPE is under `BPE/`.

## Architecture

Decoder-only transformer with causal self-attention, pre-norm residual blocks, and a final LM head. Hyperparameters (as used for the BPE ablation plan in `experimentation/BPE.md`):

| Hyperparameter | Value |
|---|---|
| `batch_size` | 64 |
| `block_size` | 256 |
| `max_iters` | 2000 (ablation target; see Status) |
| `eval_interval` | 100 |
| `learning_rate` | 3e-4 |
| `eval_iters` | 200 |
| `n_embd` | 384 |
| `n_head` | 6 |
| `n_layer` | 6 |
| `dropout` | 0.2 |

Vocab size is 256 for the byte-level baseline, or `256 + num_merges` with BPE (500 merges → 756). This may change based on my experiments, but the ablation plan is to hold all hyperparameters constant except for merge count.

Device: MPS if available, else CPU (`train_gpt.py`).

## Tokenization ablation

This is the main thing I'm measuring right now: from-scratch BPE (`BPE/bpe_tokenizer.py`) vs a UTF-8 byte-level baseline (`USE_BPE = False` in `train_gpt.py`). No tiktoken / sentencepiece for the actual tokenizer — `regex` is used only for the GPT-2-style pretoken split. Refer to `experimentation/BPE.md` for more information on the ablation plan.

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

```bash
# in train_gpt.py:
#   USE_BPE = True   # load BPE/tokenizer.json
#   USE_BPE = False  # UTF-8 byte baseline, vocab_size=256
# set max_iters / flags as needed (PRINT_CPT, SAVE_LOSS_CURVE, USE_EVAL_INTERVAL)

python train_gpt.py
```

## Status

WIP. I'm adding pieces as I learn them: baseline transformer first, then BPE, then a proper tokenization ablation with BPB. Not a finished project; expect debug flags, incomplete metrics, and notes in `experimentation/` that aren't cleaned up yet.


notes as i add/change implementation:
- switching to bfloat16 for improved efficiency (hopefully)