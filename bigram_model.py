import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


# globals ---
context_size = 8
batch_size = 32
emb_dim = 32
train_iters = 5000
eval_iters = 100
lr = 1e-3
# ---

with open("kalevala.txt") as f:
    input_text = f.read()

tokens = sorted(list(set(input_text)))
vocab_size = len(tokens)

char_to_idx = {t:i for i, t in enumerate(tokens)}
idx_to_char = {i:t for i, t in enumerate(tokens)}

encode = lambda text: [char_to_idx[c] for c in text]
decode = lambda idxs: "".join(idx_to_char[idx] for idx in idxs)

# train val split
input_tensor = torch.tensor(encode(input_text), dtype=torch.long)
input_len = len(input_tensor)
split = int(input_len*0.9)
train = input_tensor[:split]
val = input_tensor[split:]


def get_batch(split_str):
    data = train if split_str == "train" else val
    batch_idxs = torch.randint(len(data) - context_size, (batch_size, ))
    xb = torch.stack([data[ix:ix+context_size] for ix in batch_idxs])
    yb = torch.stack([data[ix+1:ix+1+context_size] for ix in batch_idxs])
    return xb, yb

@torch.no_grad
def evaluate(model):
    losses = dict()
    losses = {"train": torch.zeros(eval_iters), "val": torch.zeros(eval_iters)}
    for split in ["train", "val"]:
        for iter in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[split][iter] = loss
    train_loss = losses["train"].mean()
    val_loss = losses["val"].mean()
    return train_loss, val_loss



class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.next_token_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        # B batch size
        # T timestep (context size)
        logits = self.next_token_table(x) # (B, T, emb_dim)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_view = logits.view(B*T, C) # (B*T, vocab_size)
            targets_view = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

    def generate(self, context, max_new_tokens=100):
        # given context generate next token
        # add next token to previous context
        # generate next token based on that, repeat
        for _ in range(max_new_tokens):
            logits, loss = self(context)
            logits = logits[-1, :]
            probs = F.softmax(logits, -1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token))
        return context


model = BigramModel()
optim = AdamW(model.parameters(), lr=lr)



for i in range(train_iters):
    if i % 500 == 0 or i == train_iters-1:
        train_loss, val_loss = evaluate(model)
        print(f"Iter {i} - Train loss: {train_loss:.4}, Validation loss: {val_loss:.4}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optim.zero_grad()
    loss.backward()
    optim.step()

#print(loss)

print(decode(model.generate(torch.tensor(encode("\n"), dtype=torch.long), 200).tolist()))