import regex as re

def text_processing(text):
    # split by key break points (using the same one as gpt-2)
    gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    split_text = re.findall(gpt2pat, text)

    # raw text -> utf-8 encoding
    encodings = [list(row.encode("utf-8", errors='replace')) for row in split_text]
    return encodings

def find_pairs(tokens):
    pairs = {}
    for row in tokens:
        for pair in zip(row, row[1:]):
            pairs[pair] = pairs.get(pair,0)+1
    return pairs

def merge(tokens, pair, idx):
    newids = []
    for row in tokens:
        i = 0
        temp = []
        while i < len(row):
            if i < len(row) - 1 and row[i] == pair[0] and row[i+1] == pair[1]:
                temp.append(idx)
                i += 2
            else:
                temp.append(row[i])
                i += 1
        newids.append(temp)
    return newids

def training(text, mergecount):
    # encode text
    tokens = text_processing(text)
    store_encodings = tokens.copy()

    # executing merges
    merges = {}
    for i in range(mergecount):
        pairs = find_pairs(tokens)
        highest_pair = max(pairs, key=pairs.get)
        idx = 256 + i
        tokens = merge(tokens, highest_pair, idx)
        merges[highest_pair] = idx
        # print(f"merged {highest_pair} into a new token {idx}")
        if (i+1)%50 == 0:
            print(f"completed {i} merges")
    
    vocab = {i:bytes([i]) for i in range(256)}
    for pair, idx in merges.items():
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    return merges, vocab, store_encodings

def decoder(tokens):
    tokens = b"".join(vocab[idx] for row in tokens for idx in row)
    text = tokens.decode("utf-8", errors="replace")
    return text

def encoder(text):
    tokens = text_processing(text)
    result = []
    for row in tokens:
        while len(row) >= 2:
            stats = find_pairs([row])
            pair = min(stats, key=lambda p:merges.get(p, float("inf")))
            if pair not in merges:
                break
            idx = merges[pair]
            row = merge([row], pair, idx)[0]
        result.append(row)
    return result

import json

def save_tokenizer(merges, vocab, path="tokenizer.json"):
    merges_json = {f"{a},{b}": idx for (a, b), idx in merges.items()}
    vocab_json = {str(idx): list(b) for idx, b in vocab.items()}
    with open(path, "w") as f:
        json.dump({"merges": merges_json, "vocab": vocab_json}, f)

def load_tokenizer(path="tokenizer.json"):
    global merges, vocab
    with open(path, "r") as f:
        data = json.load(f)
    merges = {tuple(int(x) for x in k.split(",")): v for k, v in data["merges"].items()}
    vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
    return merges, vocab


text = "Hello, world! This is a test of the BPE tokenizer. Let's see how it handles this text."
merges, vocab, store_encodings = training(text, mergecount=10)

# print(decoder(encoder("Hello, world! This is a test of the BPE tokenizer. Let's see how it handles this text.")))