"""Saving/loading the model weights produced by the web UI's training runner.

Checkpoints live in one of two places:

- config.TEMP_CHECKPOINTS_DIR ("temp") -- written automatically during training, purely so the
  Chat tab can talk to a run before you've decided whether to keep it. These are wiped whenever
  a browser tab closes (see clear_temp_checkpoints, wired to /api/session/cleanup) and on server
  startup, so they never quietly pile up.
- config.CHECKPOINTS_DIR ("saved") -- permanent. A checkpoint only lands here when the user
  explicitly promotes it via the Save button (promote_to_saved).

Each checkpoint is a single self-contained .pt file: weights + the exact architecture config
needed to reconstruct GPTLanguageModel + the stats at that step. Nothing here touches my_gpt.py;
it only imports GPTLanguageModel to reconstruct a model from a saved state dict.
"""
import os
import shutil
import time

import torch

from my_gpt import GPTLanguageModel

from . import config

os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(config.TEMP_CHECKPOINTS_DIR, exist_ok=True)


def _dir_for(location):
    if location not in ("saved", "temp"):
        raise ValueError(f"unknown checkpoint location: {location!r}")
    return config.CHECKPOINTS_DIR if location == "saved" else config.TEMP_CHECKPOINTS_DIR


def _safe_name(name):
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return cleaned or "run"


def checkpoint_filename(run_name, step):
    return f"{_safe_name(run_name)}_step{step}.pt"


def run_name_exists(run_name):
    """True if any existing checkpoint (saved or temp) already uses this run name -- compared
    after the same sanitization used for filenames, so two names that only differ by characters
    that get stripped (e.g. "My Run!" vs "My Run?") still count as a collision."""
    target = _safe_name(run_name)
    return any(
        _safe_name(c["run_name"]) == target for c in list_checkpoints() if c.get("run_name")
    )


def save_temp_checkpoint(model, model_config, stats, run_name, step, history, sample):
    """Writes a checkpoint to temp storage. Called periodically during training; not durable
    until promote_to_saved() is called on it.

    `history` is a snapshot of the run's event log up to this point and `sample` is text
    generated from these exact weights, both baked into the file -- so a checkpoint is fully
    self-describing (its own loss curve, its own sample) even after the training run that made
    it is long gone from memory.
    """
    filename = checkpoint_filename(run_name, step)
    path = os.path.join(config.TEMP_CHECKPOINTS_DIR, filename)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "run_name": run_name,
            "step": step,
            "stats": stats,
            "history": history,
            "sample": sample,
            "saved_at": time.time(),
        },
        path,
    )
    _evict_worst_temp_checkpoints(run_name)
    return filename


def _evict_worst_temp_checkpoints(run_name, max_keep=None):
    """Keeps at most max_keep unsaved checkpoints for a run, deleting the worst (highest val
    loss) ones first. Later checkpoints aren't always better -- this project's own training runs
    have shown late-run overfitting -- so eviction is by quality, not just age, and only ever
    touches temp storage; anything the user has explicitly saved is untouched."""
    max_keep = max_keep or config.MAX_TEMP_CHECKPOINTS_PER_MODEL
    temp_ckpts = [
        c for c in list_checkpoints() if c["run_name"] == run_name and c["location"] == "temp"
    ]
    if len(temp_ckpts) <= max_keep:
        return

    def val_loss_or_worst(c):
        val_loss = (c["stats"] or {}).get("val_loss")
        return val_loss if val_loss is not None else float("inf")

    temp_ckpts.sort(key=val_loss_or_worst)
    for c in temp_ckpts[max_keep:]:
        fpath = os.path.join(config.TEMP_CHECKPOINTS_DIR, c["filename"])
        if os.path.isfile(fpath):
            os.remove(fpath)
        _model_cache.pop(("temp", c["filename"]), None)


def promote_to_saved(filename):
    """Moves a temp checkpoint into permanent storage. Returns the saved filename (renamed with
    a suffix if a same-named file is already saved, so nothing gets clobbered)."""
    src = os.path.join(config.TEMP_CHECKPOINTS_DIR, filename)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"no unsaved checkpoint named {filename!r}")

    dst = os.path.join(config.CHECKPOINTS_DIR, filename)
    if os.path.exists(dst):
        stem, ext = os.path.splitext(filename)
        dst = os.path.join(config.CHECKPOINTS_DIR, f"{stem}_{int(time.time())}{ext}")

    shutil.move(src, dst)
    saved_filename = os.path.basename(dst)
    _model_cache.pop(("temp", filename), None)
    return saved_filename


def clear_temp_checkpoints():
    """Deletes every not-yet-saved checkpoint. Called on server startup and whenever a tab
    closes -- nothing in temp storage is meant to outlive a session."""
    if os.path.isdir(config.TEMP_CHECKPOINTS_DIR):
        for fname in os.listdir(config.TEMP_CHECKPOINTS_DIR):
            if fname.endswith(".pt"):
                os.remove(os.path.join(config.TEMP_CHECKPOINTS_DIR, fname))

    for key in [k for k in _model_cache if k[0] == "temp"]:
        del _model_cache[key]


def list_checkpoints():
    out = []
    for location in ("saved", "temp"):
        directory = _dir_for(location)
        if not os.path.isdir(directory):
            continue
        for fname in os.listdir(directory):
            if not fname.endswith(".pt"):
                continue
            fpath = os.path.join(directory, fname)
            try:
                # weights_only=False: this file also holds the plain config/stats dicts we wrote
                # ourselves alongside the tensors, and it never leaves this machine.
                ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            except Exception:
                continue
            out.append(
                {
                    "filename": fname,
                    "location": location,
                    "run_name": ckpt.get("run_name"),
                    "step": ckpt.get("step"),
                    "stats": ckpt.get("stats"),
                    "saved_at": ckpt.get("saved_at"),
                    "model_config": ckpt.get("model_config"),
                }
            )

    out.sort(key=lambda c: c["saved_at"] or 0, reverse=True)
    return out


def list_models():
    """Groups list_checkpoints() by run name. Every checkpoint sharing a run name came from the
    same GPTLanguageModel instance, so they're the same model with identical hyperparameters --
    just different points along its training -- not separate models."""
    grouped = {}
    for ckpt in list_checkpoints():
        run_name = ckpt["run_name"]
        entry = grouped.setdefault(
            run_name,
            {"run_name": run_name, "model_config": ckpt["model_config"], "checkpoints": []},
        )
        entry["checkpoints"].append(
            {
                "filename": ckpt["filename"],
                "location": ckpt["location"],
                "step": ckpt["step"],
                "stats": ckpt["stats"],
                "saved_at": ckpt["saved_at"],
            }
        )

    models = []
    for entry in grouped.values():
        entry["checkpoints"].sort(key=lambda c: c["step"] or 0)
        val_losses = [
            (c["stats"] or {}).get("val_loss")
            for c in entry["checkpoints"]
            if (c["stats"] or {}).get("val_loss") is not None
        ]
        entry["saved_count"] = sum(1 for c in entry["checkpoints"] if c["location"] == "saved")
        entry["total_count"] = len(entry["checkpoints"])
        entry["best_val_loss"] = min(val_losses) if val_losses else None
        entry["latest_saved_at"] = max((c["saved_at"] or 0) for c in entry["checkpoints"])
        models.append(entry)

    models.sort(key=lambda m: m["latest_saved_at"], reverse=True)
    return models


def delete_model(run_name):
    """Removes every checkpoint (saved and temp) belonging to this run name. Frees the name up
    for reuse, same as if it had never been trained."""
    removed = 0
    for ckpt in list_checkpoints():
        if ckpt["run_name"] != run_name:
            continue
        fpath = os.path.join(_dir_for(ckpt["location"]), ckpt["filename"])
        if os.path.isfile(fpath):
            os.remove(fpath)
            removed += 1
        _model_cache.pop((ckpt["location"], ckpt["filename"]), None)

    if removed == 0:
        raise FileNotFoundError(f"no model named {run_name!r}")
    return removed


def rename_model(old_run_name, new_run_name):
    """Renames every checkpoint belonging to old_run_name to new_run_name -- updates both the
    run_name recorded inside each file and its on-disk filename (which is derived from the
    name), so the rename is consistent everywhere the name shows up."""
    new_run_name = new_run_name.strip()
    if not new_run_name:
        raise ValueError("new name cannot be empty")
    if _safe_name(new_run_name) != _safe_name(old_run_name) and run_name_exists(new_run_name):
        raise ValueError(f"a model named {new_run_name!r} already exists")

    matches = [c for c in list_checkpoints() if c["run_name"] == old_run_name]
    if not matches:
        raise FileNotFoundError(f"no model named {old_run_name!r}")

    for ckpt in matches:
        directory = _dir_for(ckpt["location"])
        old_path = os.path.join(directory, ckpt["filename"])
        full = torch.load(old_path, map_location="cpu", weights_only=False)
        full["run_name"] = new_run_name

        new_filename = checkpoint_filename(new_run_name, ckpt["step"])
        new_path = os.path.join(directory, new_filename)
        torch.save(full, new_path)
        if new_path != old_path:
            os.remove(old_path)

        _model_cache.pop((ckpt["location"], ckpt["filename"]), None)

    return new_run_name


def get_detail(filename, location):
    """Full record for one checkpoint -- everything list_checkpoints() returns, plus its loss
    history and sample text, minus the weights themselves (not needed for display).

    Checkpoints saved before history/sample existed fall back to empty values instead of
    raising, so older files still open (just without a loss curve or sample to show).
    """
    fpath = os.path.join(_dir_for(location), filename)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"no checkpoint named {filename!r} in {location} storage")

    ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
    return {
        "filename": filename,
        "location": location,
        "run_name": ckpt.get("run_name"),
        "step": ckpt.get("step"),
        "stats": ckpt.get("stats"),
        "saved_at": ckpt.get("saved_at"),
        "model_config": ckpt.get("model_config"),
        "history": ckpt.get("history", []),
        "sample": ckpt.get("sample", ""),
    }


_model_cache = {}


def load_model(filename, location, device):
    cache_key = (location, filename, device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    fpath = os.path.join(_dir_for(location), filename)
    ckpt = torch.load(fpath, map_location=device, weights_only=False)
    mc = ckpt["model_config"]

    model = GPTLanguageModel(
        mc["n_head"], mc["n_embd"], mc["block_size"], mc["dropout"],
        mc["vocab_size"], mc["n_layer"], device,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model_cache[cache_key] = (model, ckpt)
    return model, ckpt
