import torch
import torch.nn as nn
import math

'''
B = batch size (no. of sentences)
S = sequence length (length of each sentence)
D = dimension of each vector (length of each word(token))
'''

# 1. input embeds

'''
x enters as a (2,20) tensor, after embedding lookup it becomes (2,20,512)
'''
class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) #create a map from the dict
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        ''' in the attention paper it was cited: In the embedding layers, we multiply those weights by
        √dmodel, for dominance of these embeds over positions
        embedding shape: (batch size, sequence lenght, dimension(of each vector))
        also refered to harvard nlp annotated transformer
        
        before scaling: (B,S,D)
        after scaling: (B,S,D)
        '''


# 2. positional embeds
'''
postional encoding gives a unique address to an token in each sentence
tho it doesnt specify whcih sentence its in due to 'computation efficiency'
for the input shape (batch,seq length, d_model)
the pe is (1,seq length, d_model)
it just gets broadcasted to all batches

segment addressing was in the bert model which i saw
but attention paper positional encoding only answered
where is this token inside its own sequence?
not
which batch sample is this?
'''
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
        ''' using pos,i to generate different frequencies for each position
        lower freq correspoind to lower positions '''
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 3. multihead attention
'''
now we have embeddings with input embeds + positional embeds of shape (B,S,D)
now we have to convert static embeds to context aware embeds
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        assert d_model % h == 0

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h #checking if dimension is divisible by number of heads

        ''' each weight matrix is of size (D,D) , so matrix multiplication of input * weight will be like
        (B,S,D) * (D,D) = (B,S,D)
        training query key value matrices (512,512) '''
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


# 3. feed forward
''' now we have contextually rich embdeds, of shape (B,S,D)
now this ffn processes each token independently'''
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # expand dims to d_ff, u can get more expressive tokens
        self.linear2 = nn.Linear(d_ff, d_model) # project back to orgnal
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU() # adds some non lineariy

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

'''
transformations in this blokc:
 linear1----> (B,S,D) * (D,D_ff) = (B,S,D_ff)
 then relu unit
 linear2----> (B,S,D_ff) * (D_ff,D) = (B,S,D)
 dropout----> (B,S,D)

 accelerator view: 2 matrix multipliers, relu unit
'''


# 4. encoder block
# here we do the full flow input->attention->
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