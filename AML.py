## Task 1.11
#Implement the decoder attention.

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
#matplotlib inline


# inpout is a 2D array, instead of a 1D array
def dropout(x, drop_prob):
    """
    x: (batch_size, seq_len, hidden_size)
    """
    if drop_prob == 0:
        return x
    keep_prob = 1 - drop_prob
    mask = np.random.binomial(1, keep_prob, size=x.shape)
    return x * mask / keep_prob

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

        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #self.register_buffer('pe', pe) # why?

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return dropout(x, drop_prob=self.dropout)




def init_weights(self, n_in, n_out, h=0):  # Glorot normal
    sd = np.sqrt(2.0 / (n_in + n_out))
    
    for i in range(n_in):
        for j in range(n_out): 
          for hi in range(h):
             x = np.float32(np.normal(0.0, sd))
             self.weights[i,j,hi] = x
    return self.weights




def skip_connections(x, y):
  if x.size(1) != y.size(1):
    y = y[:, :x.size(1)]
  return np.concatenate((x,y),axis=0)

# feed_forward

def feed_forward(x, d_model, d_ff, dropout=0.1):
    """
    x: (batch_size, seq_len, hidden_size)
    """
    W_1 = init_weights(d_model, d_ff)
    W_2 = init_weights(d_ff, d_model)
    x = np.matmul(x, W_1)
    x = np.matmul(x, W_2)
    x = dropout(x, dropout)
    return x


# You will test your implementation on a single array:
import numpy as np
forward_pass_array = np.array(([101, 400, 500, 600, 107, 102], [101, 400, 500, 600, 107, 102]))
d_model = forward_pass_array.shape[1]
# plt.figure(figsize=(15, 5))
# pe = PositionalEncoding(forward_pass_array.shape[0], 0)
# y = pe.forward(Variable(torch.zeros(1, 100, 20)))
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d"%p for p in [4,5,6,7]])

# layer normalization
def layer_normalize(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    
    return (x - mean) / (std + eps)


def attention(query, key, value, mask=None, dropout=None):
    """
    query: (batch_size, seq_len, hidden_size)
    key: (batch_size, seq_len, hidden_size)
    value: (batch_size, seq_len, hidden_size)
    mask: (batch_size, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = np.matmul(query, key.T) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return np.dot(p_attn, value), p_attn

for i in range(h):
        query = np.matmul(input, W_k[i])
        key = np.matmul(input, W_q[i])
        value = np.matmul(input, W_v[i])

def multi_head_attention(query, key, value, mask=None, dropout=None):
    """
    key: (seq_len, hidden_size)
    value: (seq_len, hidden_size)
    mask: (seq_len, seq_len)
    """

    # "Concat" using a view and apply a final linear. 
    for i in range(h):
        x, _ = attention(query[:,:,i], key[:,:,i], value[:,:,i], mask=mask, dropout=dropout)
        
        if i == 0:
            output = x
        else:
            output = np.concatenate((output, x),axis=0)
    output = np.matmul(output, W_O[i])    

    return output



# initialize weights for encoder_attention
N = 8
d = 3
h = 2
dic_dim = 10000
W_k = init_weights(forward_pass_array.shape[1], d, h) # 6x3x2
W_q = init_weights(forward_pass_array.shape[1], d, h) # 6x3x2
W_v = init_weights(forward_pass_array.shape[1], d, h) # 6x3x2

W_O = init_weights(d*h, forward_pass_array.shape[1]) # 2x18x6
W_linear = init_weights(input, dic_dim) # 6x6

def encoder_attention(input):
    input += PositionalEncoding(input.shape[0], 0)
    input = dropout(input, 0.1) # apply dropout to the sums of the embeddings and the positional encodings

    # get query, key, value
    query = np.matmul(input, W_k) # 6x3x2
    key = np.matmul(input, W_q) # 6x3x2
    value = np.matmul(input, W_v) # 6x3x2
    
    for i in range(N):  
         self_attenion = multi_head_attention(query, key, value, mask=None, dropout=dropout)
         input += skip_connections(input, self_attenion, 0.1)  #apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
         input = layer_normalize(input)
    
         # feed forward
         input += skip_connections(input, dropout(feed_forward(input), 0.1)) 
         input = layer_normalize(input)
    return input

def encoder(input, N):
    input += PositionalEncoding(input.shape[0], 0)
    input = dropout(input, 0.1) # apply dropout to the sums of the embeddings and the positional encodings
    
    for i in range(N):
       input += skip_connections(input, dropout(encoder_attention(input), 0.1))  #apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
       input = layer_normalize(input)

       # feed forward
       input += skip_connections(input, dropout(feed_forward(input), 0.1)) 
       input = layer_normalize(input)
    return input





def decoder_attention(input,h, mask=matrix_mask):
    # to be defined
    # masked attention
    # get query, key, value
    query = np.matmul(input, W_k) # 6x3x2
    key = np.matmul(input, W_q) # 6x3x2
    value = np.matmul(input, W_v) # 6x3x2
    
    for i in range(N):  
         self_attenion = multi_head_attention(query, key, value, mask=None, dropout=dropout)
         input += skip_connections(input, self_attenion, 0.1)  #apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
         input = layer_normalize(input)
    
         # feed forward
         input += skip_connections(input, dropout(feed_forward(input), 0.1)) 
         input = layer_normalize(input)
    return input

def encoder_decoder_attention(input, encoder_output):
    # get the keys, queries, and values
        # matrix multiplication and scaling
        # get query, key, value
    query = np.matmul(encoder_output W_k) # 6x3x2
    key = np.matmul(encoder_output, W_q) # 6x3x2
    value = np.matmul(input, W_v) # 6x3x2    


def linear_layer(input):

    # linear layer
    input = np.matmul(input, W_linear)

    return input

def decoder(input,encoder_output, N):
    input += PositionalEncoding(input.shape[0], 0)
    input = dropout(input, 0.1) # apply dropout to the sums of the embeddings and the positional encodings

    for i in range(N):
       # decoder attention
       input += skip_connections(input, dropout(decoder_attention(input), 0.1))  #apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
       input = layer_normalize(input)

       # encoder-decoder attention
       input += skip_connections(input, dropout(encoder_decoder_attention(input, encoder_output), 0.1)) 
       input = layer_normalize(input)

       # feed forward
       input += skip_connections(input, dropout(feed_forward(input), 0.1)) 
       input = layer_normalize(input)
       
    # flatten 2D matrix to 1D vector
    input = input.flatten()
    # linear layer
    input = linear_layer(input)
    output = softmax(input)

    return output   



# mask the padding
