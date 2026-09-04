"""Shared paths and default hyperparameters for the web UI.

Run through ../run_webui.py so this directory's parent (the my-gpt project root) is on
sys.path and `import my_gpt` / `from BPE.bpe_tokenizer import ...` resolve.
"""
import os

import torch

WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEBUI_DIR)

CHECKPOINTS_DIR = os.path.join(WEBUI_DIR, "checkpoints")
TEMP_CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, ".tmp")
STATIC_DIR = os.path.join(WEBUI_DIR, "static")
DATA_PATH = os.path.join(PROJECT_ROOT, "shakespeare.txt")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "BPE", "tokenizer.json")

# Per model, at most this many *unsaved* checkpoints are kept at once -- the worst (by val
# loss) get evicted as new ones arrive, so the iteration picker stays short without losing the
# checkpoints most worth looking at. Saved checkpoints are never evicted by this.
MAX_TEMP_CHECKPOINTS_PER_MODEL = 5

# Mirrors the hyperparameters at the top of train_gpt.py.
DEFAULT_TRAIN_CONFIG = {
    "run_name": "webui-run",
    "batch_size": 64,
    "block_size": 256,
    "max_iters": 2000,
    "eval_interval": 200,
    "eval_iters": 200,
    "learning_rate": 3e-4,
    "n_embd": 384,
    "n_head": 3,
    "n_layer": 3,
    "dropout": 0.2,
    "weight_decay": 1.0,
    "use_bpe": True,
    "use_bf16": False,
}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
