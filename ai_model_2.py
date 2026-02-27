import torch
import torch.nn as nn
import math


# -----------------------------
# Input Embedding
# -----------------------------
class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# -----------------------------
# Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        assert d_model % h == 0

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        q = self.w_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        return self.w_o(x)


# -----------------------------
# Feed Forward Network
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# -----------------------------
# Encoder Block
# -----------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, h, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# -----------------------------
# Transformer Encoder
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len,
                 d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.pos = PositionalEncoding(d_model, seq_len, dropout)

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, h, d_ff, dropout)
            for _ in range(N)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# -----------------------------
# Test Run
# -----------------------------
if __name__ == "__main__":
    model = TransformerEncoder(vocab_size=10000, seq_len=20)

    dummy_input = torch.randint(0, 10000, (2, 20))  # (batch=2, seq=20)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # should be (2, 20, 512)