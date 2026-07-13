# one time vocabulary creation using bpe

from bpe_tokenizer import training, save_tokenizer

with open("shakespeare.txt") as f:
    text = f.read()

merges, vocab, store_encodings = training(text, mergecount=500)
save_tokenizer(merges, vocab, "tokenizer.json")
print(f"trained and saved {len(merges)} merges")