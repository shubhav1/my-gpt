# one time vocabulary creation using bpe

from bpe_tokenizer import training, save_tokenizer

merge_count = 500

with open("../shakespeare.txt") as f:
    text = f.read()

# match the same 90/10 split used in train_gpt.py
n = int(0.9 * len(text))
train_text = text[:n]

merges, vocab, store_encodings = training(train_text, merge_count)
save_tokenizer(merges, vocab, "tokenizer.json")
print(f"trained and saved {len(merges)} merges")