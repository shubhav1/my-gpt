"""Background training loop that mirrors train_gpt.py step-for-step (same batching, same eval
cadence, same bpb formula), but emits events to subscribers instead of printing to stdout, and
checkpoints the model as it goes. train_gpt.py itself is never imported or modified.
"""
import math
import queue
import threading
import time

import torch

from my_gpt import GPTLanguageModel, estimate_loss, get_batch

from . import checkpoints, config, data_utils, tokenizer_utils

_NUMERIC_INT_FIELDS = ("batch_size", "block_size", "max_iters", "eval_interval", "eval_iters", "n_embd", "n_head", "n_layer")
_NUMERIC_FLOAT_FIELDS = ("learning_rate", "dropout", "weight_decay")


class TrainingRun:
    """Tracks a single in-progress or most-recently-finished training run.

    Only one run is tracked at a time -- this is a local single-user tool, not a job queue.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._subscribers = []

        self.status = "idle"  # idle | running | done | error | stopped
        self.history = []
        self.error_message = None
        self.run_name = None
        self.model_config = None
        self.checkpoint_paths = []

    def is_running(self):
        return self.status == "running"

    def subscribe(self):
        """Registers a new listener queue, pre-filled with everything emitted so far."""
        q = queue.Queue()
        with self._lock:
            for event in self.history:
                q.put(event)
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _emit(self, event):
        with self._lock:
            self.history.append(event)
            for q in self._subscribers:
                q.put(event)

    def start(self, overrides):
        if self.is_running():
            raise RuntimeError("a training run is already in progress")

        cfg = dict(config.DEFAULT_TRAIN_CONFIG)
        cfg.update(overrides or {})
        cfg["run_name"] = str(cfg["run_name"]).strip() or "webui-run"
        for field in _NUMERIC_INT_FIELDS:
            cfg[field] = int(cfg[field])
        for field in _NUMERIC_FLOAT_FIELDS:
            cfg[field] = float(cfg[field])
        cfg["use_bpe"] = bool(cfg["use_bpe"])
        cfg["use_bf16"] = bool(cfg["use_bf16"])

        # Every checkpoint under a run name is treated as the same model with identical
        # hyperparameters -- this system never resumes training, so reusing a name here would
        # always mean a genuinely different (freshly initialized) model wearing the same label.
        if checkpoints.run_name_exists(cfg["run_name"]):
            raise ValueError(
                f"a model named {cfg['run_name']!r} already exists -- choose a different name"
            )

        self.status = "running"
        self.history = []
        self.error_message = None
        self.run_name = cfg["run_name"]
        self.model_config = None
        self.checkpoint_paths = []
        self._stop_event.clear()

        thread = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self, cfg):
        try:
            self._train_loop(cfg)
        except Exception as exc:  # surface training failures to the UI instead of dying silently
            self.status = "error"
            self.error_message = str(exc)
            self._emit({"type": "error", "message": str(exc)})

    def _train_loop(self, cfg):
        device = config.get_device()
        torch.manual_seed(1337)

        text = data_utils.load_text()
        encode, decode, vocab_size = tokenizer_utils.get_tokenizer(cfg["use_bpe"])
        train_data, val_data, cpt_train, cpt_val = data_utils.build_dataset(
            text, encode, decode, cfg["use_bpe"]
        )

        self._emit(
            {
                "type": "start",
                "device": device,
                "cpt_train": cpt_train,
                "cpt_val": cpt_val,
                "vocab_size": vocab_size,
                "config": cfg,
            }
        )

        model = GPTLanguageModel(
            cfg["n_head"], cfg["n_embd"], cfg["block_size"], cfg["dropout"],
            vocab_size, cfg["n_layer"], device,
        ).to(device)

        self.model_config = {
            "n_head": cfg["n_head"],
            "n_embd": cfg["n_embd"],
            "block_size": cfg["block_size"],
            "dropout": cfg["dropout"],
            "vocab_size": vocab_size,
            "n_layer": cfg["n_layer"],
            "use_bpe": cfg["use_bpe"],
        }

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tokens = cfg["batch_size"] * cfg["block_size"] * cfg["max_iters"]
        self._emit(
            {
                "type": "params",
                "params": params,
                "tokens": tokens,
                "tokens_per_param": tokens / params if params else None,
            }
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )

        iter_times = []
        max_iters = cfg["max_iters"]
        eval_interval = cfg["eval_interval"]

        for it in range(max_iters):
            if self._stop_event.is_set():
                self.status = "stopped"
                break

            t0 = time.time()
            xb, yb = get_batch(
                "train", train_data, val_data, cfg["block_size"], cfg["batch_size"], device
            )

            if cfg["use_bf16"]:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xb, yb)
            else:
                logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            t1 = time.time()
            iter_times.append(t1 - t0)

            self._emit(
                {"type": "step", "iter": it + 1, "loss": loss.item(), "ms_per_iter": (t1 - t0) * 1000}
            )

            if it == 0 or (it + 1) % eval_interval == 0 or it == max_iters - 1:
                losses = estimate_loss(
                    model, cfg["eval_iters"], train_data, val_data,
                    cfg["block_size"], cfg["batch_size"], device, cfg["use_bf16"],
                )
                bpb = losses["val"].item() / (cpt_val * math.log(2))
                window = iter_times[-eval_interval:]
                avg_iter_time = sum(window) / len(window)

                self._emit(
                    {
                        "type": "eval",
                        "iter": it + 1,
                        "train_loss": losses["train"].item(),
                        "val_loss": losses["val"].item(),
                        "val_bpb": bpb,
                        "ms_per_iter": avg_iter_time * 1000,
                    }
                )

                stats = {
                    "iter": it + 1,
                    "train_loss": losses["train"].item(),
                    "val_loss": losses["val"].item(),
                    "val_bpb": bpb,
                }

                # A sample from these exact weights, baked into the checkpoint itself so it's
                # unambiguous later which model produced it -- generated every checkpoint rather
                # than only once at the end, since (like estimate_loss) sampling needs eval mode.
                model.eval()
                with torch.no_grad():
                    sample_context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    sample_ids = model.generate(sample_context, max_new_tokens=150)[0].tolist()
                model.train()
                sample_text = decode(sample_ids)

                # Temp storage only -- nothing here is durable until the user hits Save.
                ckpt_filename = checkpoints.save_temp_checkpoint(
                    model, self.model_config, stats, cfg["run_name"], it + 1,
                    list(self.history), sample_text,
                )
                self.checkpoint_paths.append(ckpt_filename)
                self._emit(
                    {"type": "checkpoint", "filename": ckpt_filename, "location": "temp", "iter": it + 1}
                )

        if self.status == "running":
            self.status = "done"

        self._emit({"type": self.status})


run = TrainingRun()
