# my-gpt web UI

A small local control panel for `my-gpt`: start a training run, watch live stats and a loss
curve, and chat with a trained checkpoint — without touching any of the existing training code.

Everything here is new. `train_gpt.py`, `my_gpt.py`, `RoPE.py`, and `BPE/` are untouched; this
just imports them.

## Setup

Flask is the only new dependency. Install it into the project's existing `.venv`:

```sh
uv pip install --python .venv/bin/python -r webui/requirements.txt
```

## Run

From the `my-gpt` project root:

```sh
python run_webui.py
```

Then open http://127.0.0.1:5050.

## Layout

The UI is a sidebar + a main pane, not tabs:

- **Sidebar (left)** — every model you've trained, saved or not, one per checkpoint. "+ Train new
  model" at the top starts a fresh run.
- **"Train new model"** — the config form (defaults mirror the top of `train_gpt.py`), start/stop,
  and a live loss curve/stats while it runs. As soon as a checkpoint is produced it shows up in
  the sidebar — you don't wait for the run to finish to start using a model.
- **Clicking a model in the sidebar** opens it: its stats, its architecture, its own loss curve
  (up to the step it was saved at), a sample generated from those exact weights, and a chat box
  scoped to that one model — so it's never ambiguous which checkpoint produced a given sample or
  which model you're talking to. This is also where you save it, if it's not saved yet.

Each checkpoint is self-contained: the weights, the run's event history up to that point, and a
freshly generated sample are all baked into the same `.pt` file at save time, so opening a model
later doesn't depend on the training run that made it still being in memory (or the server not
having restarted since).

## Saving and discarding checkpoints

- Checkpoints write automatically every `eval_interval` steps (and once more at the end, or
  whenever a run is stopped early), but only to **temp storage** — nothing durable happens until
  you open that model and hit **Save this model**.
- Unsaved ("temp") models are fully usable — stats, loss curve, sample, chat all work — while the
  session is open; the sidebar just tags them "unsaved." You can try a model before deciding
  whether it's worth keeping, and even save an earlier step instead of the last one if it turned
  out to generate better samples.
- Closing the tab (`pagehide`, via `navigator.sendBeacon`) tells the server to wipe everything in
  temp storage. The server also clears temp storage on startup, in case a browser tab was closed
  in a way that never fired the beacon (or the process itself crashed).
- **Caveat**: browsers have no reliable way to distinguish "tab closed" from "page refreshed" —
  both fire the same event — so refreshing the page also discards anything not yet saved. Hit
  Save first if you want to keep it.
- Checkpoints saved before this history/sample tracking existed still open fine — they just show
  an empty loss curve and no sample, since that data was never recorded for them.

## Design notes

- `training_runner.py` re-implements the training loop from `train_gpt.py` (same batching, same
  eval cadence, same bpb formula) inside a background thread that emits events instead of
  printing, so nothing in the original script needed to change.
- `inference.py` re-implements `GPTLanguageModel.generate`'s sampling loop so it can `yield` each
  newly generated chunk of text to the browser instead of only returning after every token has
  been produced — again, without touching `my_gpt.py`.
- Training started from the UI does **not** wrap the model in `torch.compile`, unlike
  `train_gpt.py`. That keeps checkpoint state dicts portable (no `_orig_mod.` prefix headaches
  when loading them back for chat) and keeps "start training" responsive instead of paying a
  compile stall up front. Batching, the eval loop, the optimizer, and the bpb math all match
  `train_gpt.py` exactly.
- Only one training run is tracked at a time — this is a single-user local tool, not a job queue.

## API (for reference)

| Route | Method | What |
|---|---|---|
| `/api/config` | GET | Default hyperparameters |
| `/api/train/start` | POST | Start a run (JSON body overrides defaults) |
| `/api/train/stop` | POST | Stop the current run after the in-flight step |
| `/api/train/stream` | GET (SSE) | Replays history, then streams live training events |
| `/api/train/status` | GET | Current status + full event history, no streaming |
| `/api/checkpoints` | GET | List checkpoints (saved + this session's unsaved ones), lightweight |
| `/api/checkpoints/detail` | GET | Full record for one checkpoint: `?filename=&location=` — stats, architecture, history, sample |
| `/api/checkpoints/save` | POST | Promote a temp checkpoint to permanent storage: `{filename}` |
| `/api/session/cleanup` | POST | Discard all unsaved checkpoints (fired on tab close) |
| `/api/chat` | POST (SSE) | Stream generated text for `{checkpoint, location, prompt, max_new_tokens}` |
