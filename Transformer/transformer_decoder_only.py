import torch
import torch.nn as nn
import torch.nn.functional as F

### Overall Architecture

# Token_Embedding + Pos_Embedding
# Block
    # -----------------------------------------------------
    # Layer Norm                                          |
    # MultiHeadAttention                                  |
    # Residual Connection <--------------------------------
    # -----------------------------------------------------
    # Layer Norm                                          |
    # MLP                                                 |
    # Residual Connection <--------------------------------
# Layer Norm
# Linear Projection

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, block_size, n_heads, head_size):
        super().__init__()

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.linear_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        qkv_proj = self.qkv(x) # (B, T, 3 * n_embd)
        q, k, v = qkv_proj.view(B, T, n_heads, 3 * head_size).split(head_size, dim=-1)

        # q, k, v = (B, T, n_heads, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v = (B, n_heads, T, head_size)

        attention = (q @ k.transpose(-1, -2))*(head_size**(-0.5))

        # causal mask
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        attention = F.softmax(attention, dim=-1) # softmax across each row of attention scores

        attention = attention @ v # (B, n_heads, T, Hs)
        attention = attention.transpose(2, 1).contiguous() # (B, T, n_heads, Hs)
        attention = attention.view(B, T, C)

        out = self.linear_proj(attention)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, block_size, n_heads, head_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.multiheadattention = MultiHeadAttention(n_embd, block_size, n_heads, head_size) 
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd),
                                 nn.ReLU(),
                                 nn.Linear(4 * n_embd, n_embd)
                                )

    def forward(self, x):
        
        x = x + self.multiheadattention(self.ln1(x))
        out = x + self.mlp(self.ln2(x))

        return out

class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_heads, head_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, block_size, n_heads, head_size) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.linear_f = nn.Linear(n_embd, vocab_size) 

    def forward(self, x):
        B, T = x.shape
        x = self.token_embedding(x) + self.pos_embedding(torch.arange(0, T))
        for block in self.blocks:
            x = block(x)
        x = self.lnf(x)
        out = self.linear_f(x)

        return out


if __name__ == '__main__':

    # Hyperparameters
    vocab_size = 5
    block_size = 3
    n_embd = 4
    n_layer = 2
    n_heads = 2
    head_size = 2

    criterion = nn.CrossEntropyLoss()

    # Data
    X = torch.tensor([[1, 2, 3]], dtype=torch.long) # (B, T) idx for each token
    y = torch.tensor([[2, 3, 4]], dtype=torch.long) # (B, T) idx for the next token

    # Initiating Model
    model = TransformerDecoderOnly(vocab_size, block_size, n_embd, n_layer, n_heads, head_size)

    # Forward Pass
    out = model(X) # (B, T, Vocab_Size)

    # Flatten for loss computation
    B, T, v_size = out.shape
    out = out.view(B*T, v_size) 
    targets = y.view(B*T)

    # Compute Loss
    loss = criterion(out, targets)

    print(f"Loss: {loss.item():.4f}")







