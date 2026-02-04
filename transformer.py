import torch
import torch.nn as nn
import math
'''
THIS CODE IS THE IMPLEMENTATION OF THE TRANSFORMER MODEL FROM THE "ATTENTION IS ALL YOU NEED" PAPER.

STAGE 0: HIGH LEVEL SUMMARY

A transformer does not understand words.
It only understands 3D tensors.
 ps: tokenization happens outside the neural network.
(batch_size, seq_len, d_model)
Where:
batch_size: number of sequences processed in parallel(sentences, images, etc.)
seq_len: number of tokens in each sequence(words, patches, etc.)
d_model: dimension of the embedding space

x.shape = (2, 5, 512)
Meaning:
	•2 sentences at once
	•each sentence has 5 tokens
	•each token is represented by a 512-dimensional vector

pipeline:
token IDs
   ↓
embedding lookup
   ↓
+ positional encoding
   ↓
Q, K, V projections
   ↓
attention

attention works on tensors, not words: (batch_size, seq_len, d_model)

every block preserves the shape of the tensor, (B,S,D) -> (B,S,D)

Encoder:
    •Sees entire input sentence
	•Uses self-attention only
	•Builds a rich memory representation

Decoder:
    •Generates output one token at a time
	•Uses:
	•	masked self-attention (no future peeking)
	•	cross-attention (queries encoder memory)

A transformer processes batches of token sequences as 3D tensors 
(batch, seq_len, d_model), adds positional information to 
preserve order, and repeatedly refines token representations 
using attention and feedforward blocks while preserving shape
for stable training and deep stacking.

cross attention vs self attention:
self-attention: Q, K, V all come from the same place
cross-attention: Q comes from the decoder, K and V come from the encoder

masked self-attention: prevents attending to future tokens
'''




'''
STAGE 1: INPUT EMBEDDINGS AND POSITIONAL ENCODING

input embeddings:
transformers need to convert token ids to static embeddings

(b,s) -> embedding lookup -> (b,s,d) 
ex: (2,5) -> (2,5,512) gets all token embeddings for 2 sentences of 5 tokens each

this block converts token ids to embeddings
This block converts integer token IDs into trainable 
d_model-dimensional vectors using a lookup table and scales them to match attention dynamics.
'''

class InputEmbeddings(nn.Module): # this is a pytorch module, which will hold weights and support backprop
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__() # d_model: embeds ka dimension,vocab_size: number of unique tokens in dictionary
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        #the above line declares a trainable matrix of shape (vocab_size, d_model) for getting embeddings of each token
        # this matrix is initialized randomly and learned during training
    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

'''
end to end flow of this block:
intput: x = [[12, 4, 9]]
process:
12 → embedding_table[12]
4  → embedding_table[4]
9  → embedding_table[9]
output:
(batch, seq_len, d_model)*√d_model
'''


'''
STAGE 2 - POSITIONAL ENCODING
[I, love, you]
[you, love, I]
so positions matter
the idea: each position gets a unique waveform added to the embedding
why not just give each position a unique vector?
1. generalization to longer sequences
2. smooth gradients
3. extrapolation to unseen positions
This block adds unique positional encodings to token embeddings based on sine 
and cosine functions to inject order information into the model.


positional encoding must enable the model to learn:
token at position i is k steps away from token at position j

so we encode positions using sine and cosine waves of different frequencies
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
pos = token position (0 to seq_len-1)
i = dimension index (0 to d_model-1)
d_model = embedding dimension (512, 1024, etc.)

for each token u get a d_model-dimensional vector(512-dim, 1024-dim, etc.)
positional encoding also produces a (batch, seq_len, d_model) tensor
index i refers to the index within the d_model dimension

small i: slow osscillations
large i: fast osscillations
[ sin(slow), cos(slow), sin(medium), cos(medium), sin(fast), cos(fast), ... ]

from input embeds we get (no. of sentences, no. of tokens, no. of dims per token)
so, 
input: (batch, seq_len, d_model)
output: (batch, seq_len, d_model) + positional encodings(batch, seq_len, d_model)
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
    pe = torch.zeros(seq_len, d_model)#for every position, we will have a d_model-dimensional positional encoding
    position = torch.arange(0, seq_len).unsqueeze(1) # shape (seq_len, 1), this is pos
    div_term = torch.exp(
    torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    ) # implements 1/10000^(2i/d_model)
    pe[:, 0::2] = torch.sin(position * div_term) # even indices
    pe[:, 1::2] = torch.cos(position * div_term) # odd indices
    pe = pe.unsqueeze(0)
    self.register_buffer("pe", pe)

    def forward(self, x):
        # finally add positional encodings to input embeddings
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) #pe no need training since it has formula defined above
        return self.dropout(x) # for regularization of input embeds + positional encodings
    
'''
STAGE 3: MULTI-HEAD ATTENTION MECHANISM

'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)