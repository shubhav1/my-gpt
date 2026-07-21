def calculate_cpt(text, encode_fn):
    # returns chars-per-token compression ratio for a given text and encoder
    tokens = encode_fn(text)
    return len(text) / len(tokens)

def report_cpt(train_text, val_text, encode_fn):
    cpt_train = calculate_cpt(train_text, encode_fn)
    cpt_val = calculate_cpt(val_text, encode_fn)

    print(f"chars-per-token: train: {cpt_train:.4f}, val: {cpt_val:.4f}")
    return {"cpt_train": cpt_train, "cpt_val": cpt_val}