# Transformer

In the transformer_decoder_only python file I have written the core architecture for decoder-only Transformer.

## Overall Architecture
```
 - Token_Embedding + Pos_Embedding
 - Block
   - ----------------------
   - LayerNorm            |
   - MultiHeadAttention   |
   - Residual Connection <-
   - ----------------------
   - LayerNorm            |
   - MLP                  |
   - Residual Connection <-
 - LayerNorm
 - Linear Projection
```
## Key Points

There are many interesting aspects of the Transformer to understand but here are some of the key ones in my opinion:

 - Self-Attention
   - Self-Attention means that the representation of each token can communicate with every other token. Using Query, Key and Values matrices it is able to learn what information is relevant to that token and how to incorporate that into its representation. It is very difficult to know exactly what's happening but one self-attention head learns a certain relationship between the token and its context. Like this there can be many relationships (MultiHeadAttention).
 - MultiHeadAttention
   - MultiHeadAttention allows the model to learn multiple relationships between a token and its context. This is important in areas such as Language since there are many layers to the meaning of a word in its context. Further, MultiHeadAttention is a highly parallelizable operation thanks to the appropriate arranged of QKV matrices (large linear matrix) and taking advantage of tensor maniputation using .view() and .transpose().
   - After MultiHeadAttention there is a linear projection layer so that the MLP doesn't just receive the segregated inputs from each head but instead they are combined in some way.
 - Causal Masking
   - Causal Masking is fundamental to how a transformer works in the context of Language Language Models (LLMs). Using matrix tricks ('tril' -> torch.tril()) we can tell the model to ignore the attention computed for tokens in the future. By doing this in a triangular fashion we can simultaneously create mutliple training example from one set of tokens.
 - MLP
   - The MLP after MultiHeadAttention allows the model to 'think' about the context-aware representations of the tokens. Very important to note that the MLP acts on each token representation separately, there is no communication between them. So we can think of the attention as the 'communication' and the MLP as the 'thinking'.
