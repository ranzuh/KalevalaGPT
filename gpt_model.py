import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# globals ---
context_size = 64
batch_size = 64
train_iters = 5000
eval_iters = 100
lr = 1e-2
n_emb = 64
n_heads = 4
n_layers = 3
head_size = 64
dropout = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = "model.pth"
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
input_tensor = torch.tensor(encode(input_text), dtype=torch.long, device=device)
input_len = len(input_tensor)
split = int(input_len*0.9)
train = input_tensor[:split]
val = input_tensor[split:]


def get_batch(split_str):
    data = train if split_str == "train" else val
    batch_idxs = torch.randint(len(data) - context_size, (batch_size, ))
    xb = torch.stack([data[ix:ix+context_size] for ix in batch_idxs])
    yb = torch.stack([data[ix+1:ix+1+context_size] for ix in batch_idxs])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

@torch.no_grad
def evaluate(model):
    model.eval()
    losses = dict()
    losses = {"train": torch.zeros(eval_iters), "val": torch.zeros(eval_iters)}
    for split in ["train", "val"]:
        for iter in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[split][iter] = loss
    train_loss = losses["train"].mean()
    val_loss = losses["val"].mean()
    model.train()
    return train_loss, val_loss


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)

        att_scores = k @ q.transpose(-1, -2) * (n_emb ** -0.5) # (B, T, T)
        masked_att_scores = att_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att_weights = F.softmax(masked_att_scores, -1)
        weighted_values = att_weights @ v # (B, T, T) * (B, T, C) -> (B, T, C)

        return weighted_values
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, embed_dim):
        super().__init__()
        # create n_heads, compute attentions and concatenate them
        self.heads = nn.ModuleList([Head(head_size // n_heads) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outs = []
        for head in self.heads:
            head_outs.append(head(x))
        out = torch.cat(head_outs, dim=-1)
        return self.dropout(self.proj(out))

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size*4),
            nn.ReLU(),
            nn.Linear(input_size*4, input_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, head_size, embed_dim):
        super().__init__()
        self.sa_head = MultiHeadAttention(n_heads, head_size, embed_dim)
        self.ff = MLP(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_emb)
        self.pos_embed = nn.Embedding(context_size, n_emb)
        self.blocks = nn.Sequential(*[
            DecoderBlock(n_heads, head_size, n_emb) for _ in range(n_layers)
        ])
        
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, token_idx, targets=None):
        # B batch size
        # T timestep (context size)
        # C embedding dim - n_emb
        B, T = token_idx.shape

        tok_emb = self.token_embed(token_idx) # (B, T, C)
        pos_emb = self.pos_embed(torch.arange(T, device=device))
        emb = tok_emb + pos_emb # (B, T, C)

        weighted_values = self.blocks(emb)
        logits = self.lm_head(weighted_values)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_view = logits.view(B*T, C) # (B*T, vocab_size)
            targets_view = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

    def generate(self, context, max_new_tokens=1, temperature=0.8):
        # given context generate next token
        # add next token to previous context
        # generate next token based on that, repeat
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            context_cond = context[:, -context_size:]
            logits, loss = self(context_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, -1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=-1)
        return context

if __name__ == "__main__":
    model = GPTModel()
    model.to(device)
    optim = AdamW(model.parameters(), lr=lr)

    for i in tqdm(range(train_iters)):
        if i % 500 == 0 or i == train_iters-1:
            train_loss, val_loss = evaluate(model)
            tqdm.write(f"Iter {i} - Train loss: {train_loss:.4}, Validation loss: {val_loss:.4}")
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()

    print("saving the model to:", save_path)
    torch.save(model.state_dict(), save_path)

    print("generating some sample text:")
    gen_text = model.generate(torch.tensor([[0]], dtype=torch.long, device=device), 200)[0].tolist()
    print(decode(gen_text))
    #open('moremore.txt', 'w').write(decode(model.generate(torch.tensor([[0]], dtype=torch.long, device=device), 10000)[0].tolist()))
