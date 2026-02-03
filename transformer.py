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
'''