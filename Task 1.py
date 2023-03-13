"""
Task1: implementing forward pass of Transformer model
"""

import math
import numpy as np

# inpout is a 2D array, instead of a 1D array
def drop_out(x, drop_prob=0.1):
    if drop_prob == 0:
        return x
    keep_prob = 1 - drop_prob
    mask = np.random.binomial(1, keep_prob, size=x.shape)
    return x * mask / keep_prob

def activation(x):
    return np.maximum(0, x)

def softmax(x):
    """
    x: (batch_size, seq_len, hidden_size)
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=-1, keepdims=True)

class PositionalEncoding():
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout
        self.pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = np.sin(np.outer(position, div_term))
        self.pe[:, 1::2] = np.cos(np.outer(position, div_term))


    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return drop_out(x, drop_prob=self.dropout)

def init_weights(dim_in, dim_out):  # Glorot normal
    sd = np.sqrt(2.0 / (dim_in + dim_out))
    W = np.random.normal(loc=0, scale=sd, size=(dim_in, dim_out))
    b = np.zeros(dim_out)
    return W, b

def skip_connections(x, y):
  return x+y

# feed_forward
def feed_forward(x, d_model, d_hidden, dropout=0.1):
    W_1, b_1 = init_weights(d_model, d_hidden)
    W_2, b_2 = init_weights(d_hidden, d_model)
    x = np.matmul(x, W_1)+b_1
    x = activation(x)
    x = np.matmul(x, W_2)+b_2
    x = drop_out(x, dropout)
    return x

def layer_normalize(x, eps=1e-6):
    # layer normalization
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def linear_layer(x, input, output, dropout=0.1):
    W_1, b_1 = init_weights(input, output)
    x = np.matmul(x, W_1) + b_1
    x = drop_out(x, dropout)
    return x

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]
    scores = np.matmul(query, key.T) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, 1e-9)
    p_attn = softmax(scores)
    if dropout is not None:
        p_attn = drop_out(p_attn)
    return np.dot(p_attn, value)

def multi_head_attention(query, key, value, head=8, mask=None, dropout=0.1):
    # "Concat" using a view and apply a final linear. 
    for i in range(head):
        x = attention(query[i], key[i], value[i], mask=mask, dropout=dropout)
        
        if i == 0:
            output = x
        else:
            output = np.concatenate((output, x), axis=1)
    output = np.matmul(output, W_O_encoder)    
    return output

def encoder(x, n_layers=6):
    x = W_embedding_encoder[x]
    x += position_encoding.forward(x)
    x = drop_out(x, 0.1)  # apply dropout to the sums of the embeddings and the positional encodings

    for i in range(n_layers):
        # get query, key, value
        query_encoder = [np.matmul(x, W_K_encoder[i]) for i in range(head)]
        key_encoder = [np.matmul(x, W_Q_encoder[i]) for i in range(head)]
        value_encoder = [np.matmul(x, W_V_encoder[i]) for i in range(head)]
        self_attenion = multi_head_attention(query_encoder, key_encoder, value_encoder, mask=None, dropout=0.1)
        x += skip_connections(x,
                              self_attenion)  # apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        x = layer_normalize(x)

        # feed forward
        x += skip_connections(x, drop_out(feed_forward(x, d_model, d_hidden=1024, dropout=0.1)))
        x = layer_normalize(x)
    return x

def decoder(x, y, n_layers=6, mask=None):
    x = W_embedding_encoder[x]
    x += position_encoding.forward(x)
    x = drop_out(x, 0.1)  # apply dropout to the sums of the embeddings and the positional encodings

    for i in range(n_layers):
        # get query, key, value for decoder attention
        query_decoder = [np.matmul(x, W_K_decoder[i]) for i in range(head)]
        key_decoder = [np.matmul(x, W_Q_decoder[i]) for i in range(head)]
        value_decoder = [np.matmul(x, W_V_decoder[i]) for i in range(head)]
        # decoder attention
        self_attenion = multi_head_attention(query_decoder, key_decoder, value_decoder, mask=mask, dropout=0.1)
        x += skip_connections(x, self_attenion)  # apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        x = layer_normalize(x)

        # get query, key, value for encoder-decoder attention
        query_decoder = [np.matmul(x, W_Q_encoder_decoder[i]) for i in range(head)] # from decoder attention
        key_decoder = [np.matmul(y, W_K_encoder_decoder[i]) for i in range(head)] # from encoder attention
        value_decoder = [np.matmul(y, W_V_encoder_decoder[i]) for i in range(head)] # from encoder attention
        # encoder-decoder attention
        attenion = multi_head_attention(query_decoder, key_decoder, value_decoder, mask=None, dropout=0.1)
        x += skip_connections(x, attenion)  # apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        x = layer_normalize(x)

        # feed forward
        x += skip_connections(x, drop_out(feed_forward(x, d_model, d_hidden=d_hidden, dropout=0.1)))
        x = layer_normalize(x)

    # flatten 2D matrix to 1D vector
    x = x.flatten()
    # linear layer
    x = linear_layer(x, len(x), vocab_size)
    output = softmax(x)

    return output

# initialization
forward_pass_array_input = np.array([1, 40, 50, 60, 17, 12]) # input for forward pass
forward_pass_array_output = np.array([12, 48, 50, 63]) # output for forward pass
mask = np.tril(np.ones((4, 4)), k=1).astype('uint8') # mask the last 2 words
d_model = 512
vocab_size = 1000
d_attention = 64
d_hidden = 2048
head = 8
attention_layers = 6
position_encoding = PositionalEncoding(d_model=d_model, max_len=6)

# initialize weights for encoder_attention
W_embedding_encoder = init_weights(vocab_size, d_model)[0]
W_K_encoder = [init_weights(d_model, d_attention)[0]] * head
W_Q_encoder = [init_weights(d_model, d_attention)[0]] * head
W_V_encoder = [init_weights(d_model, d_attention)[0]] * head
W_O_encoder = init_weights(d_attention * head, d_model)[0]

# initialize weights for decoder_attention and encoder-decoder_attention
W_embedding_decoder = init_weights(vocab_size, d_model)[0]
W_K_decoder = [init_weights(d_model, d_attention)[0]] * head
W_Q_decoder = [init_weights(d_model, d_attention)[0]] * head
W_V_decoder = [init_weights(d_model, d_attention)[0]] * head
W_O_decoder = init_weights(d_attention * head, d_model)[0]

W_K_encoder_decoder = [init_weights(d_model, d_attention)[0]] * head
W_Q_encoder_decoder = [init_weights(d_model, d_attention)[0]] * head
W_V_encoder_decoder = [init_weights(d_model, d_attention)[0]] * head
W_O_encoder_decoder = init_weights(d_attention * head, d_model)[0]

y = encoder(forward_pass_array_input, n_layers=attention_layers) # output of encoder
out_prob = decoder(forward_pass_array_output, y, n_layers=attention_layers, mask=mask) #
print(out_prob.shape)

