from my_gpt import GPTLanguageModel, get_batch, estimate_loss
import torch
import time

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 1
eval_interval = 200 # how often it prints validation loss
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

if torch.backends.mps.is_available():
    device = 'mps'
    print('using MPS')
else:
    device = 'cpu'
    print('using CPU')

USE_BPE = True

# stat tracking
PRINT_CPT = True
SAVE_LOSS_CURVE = True
USE_EVAL_INTERVAL = False

if not USE_EVAL_INTERVAL:
    eval_interval = max_iters + 1

torch.manual_seed(1337)

# download data
with open("shakespeare.txt") as f:
    text = f.read()

# get tokenizer
if USE_BPE:
    from BPE.bpe_tokenizer import load_tokenizer, encoder, decoder
    merges, vocab = load_tokenizer("BPE/tokenizer.json")
    vocab_size = 256 + len(merges)

    def encode(s):
        return [tok for row in encoder(s) for tok in row]

    def decode(l):
        return decoder([l])
else:
    def encode(s):
        return list(s.encode('utf-8'))

    def decode(l):
        return bytes(l).decode('utf-8', errors='replace')

    vocab_size = 256

# train and test splits
# BPE: split on tokens (after encoding full text)
# byte-level: split on chars, then encode each split
if USE_BPE:
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
else:
    n = int(0.9 * len(text))
    train_text = text[:n]
    val_text = text[n:]
    train_data = torch.tensor(encode(train_text), dtype=torch.long)
    val_data = torch.tensor(encode(val_text), dtype=torch.long)

# CPT on the same split used for training
if USE_BPE and PRINT_CPT:
    cpt_train = len(decode(train_data.tolist())) / len(train_data)
    cpt_val = len(decode(val_data.tolist())) / len(val_data)
    print(f"chars-per-token: train: {cpt_train:.4f}, val: {cpt_val:.4f}")
    cpt_result = {"cpt_train": cpt_train, "cpt_val": cpt_val}


model = GPTLanguageModel(n_head, n_embd, block_size, dropout, vocab_size, n_layer, device)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

iter_times = []

for iter in range(max_iters):
    t0 = time.time()

    # sample a batch of data
    xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    model.lossi.append(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    t1 = time.time()
    iter_times.append(t1-t0)

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device)
        avg_iter_time = sum(iter_times[-eval_interval:]) / len(iter_times[-eval_interval:])
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"ms/iter {avg_iter_time*1000:.2f}")

# saving loss curve
if SAVE_LOSS_CURVE:
    run_name = "mergecount_500"
    m.loss_curve(run_name)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
