# gpt: generative pretraining transformer; 
# recreating chatgpt from scratch 

# Attention is all you need: 
# train a transformer based model, character level

# input: with a given amount of data, we would like the transformer able to predict the next character after a given sequence of characters. 
# this model doesnt do anything yet because the only important part is that the nn.embedding, to create the look up table, but it is not actually a transformer. The model perform the same operation in both training and evaluation mode. 

# we do not have the dropoff layer (i.e. temporarily remove/deactivate a certain % of neurons in neural network layer during each iteraction of the training process)
# no batch normalization layer (i.e. normalize the input to the layer to have a mean of 0 and a standard deviation of 1)

# Some model will hv different behaviour in inferencing and evaluation time, such as:
# inference time: the duration it takes for a trainde machine learning model to process new input data and generate an output or prediction. 

# dropsout: regularization technique to prevent overfitting
# randomly drop out some of the nodes in the neural network layer during each iteraction of the training process
# because a random subset of the nodes are not active, the model is ended up training an ensemble of sub networks
# when all the nodes are enable, the model merge back into the original ensemble

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from datetime import datetime


# Use MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# hyperparameters
batch_size = 32  # increased for GPU
block_size = 128 # increased context window
max_iters = 30000  # Quick test
eval_interval = 3000
learning_rate = 3e-4 # the self attention block cant tolerate very high learning rate


eval_iters = 200  # increased for better evaluation
n_embd = 128     # increased embedding dimension
dropout = 0.2
n_head = 4
n_layer = 5     


torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # when we load the data, we move it to the device (cpu or gpu)
    return x, y

@torch.no_grad() # context manager to tell pytorch that everythings happened in this function: we will not call backward on the tensors in this function.
# such that pytorch will not track the gradients of the tensors in this function and can be a lot more efficient with its memory usage because it doesnt call backward.

def estimate_loss():
    """
    estimate_loss used to evaluate how well the model is performing on both the training and validation datasets.
    It temporarily sets the model to evaluation mode (which disables certain layers like dropout), then repeatedly samples batches from both the training and validation splits.
    For each batch, it computes the loss (a measure of how far the model's predictions are from the actual targets) and stores these losses.
    After collecting losses for a number of batches (eval_iters), it averages them to get a representative loss value for each split.
    Finally, it returns a dictionary containing the average loss for both the training and validation sets, and switches the model back to training mode.
    
    Example:
    # The model outputs logits of shape (B, T, C), where:
    #   B = batch size
    #   T = Time sequence length (block_size)
    #   C = number of classes (vocab_size)
    # For each position in the sequence, the logits represent the (unnormalized) log-probabilities for each possible next character.
    # The targets tensor contains the correct next character indices for each position.
    
    The loss is computed using cross-entropy:
    For each position, the cross-entropy loss is:
         -log(softmax(logits)[target]): this is the cross-entropy loss for the target character at the current position.
    The total loss is the mean over all positions in the batch.
    
    Example:
    x, y = get_batch('train')
    logits, loss = model(x, y)
    print("Loss:", loss.item())
    
    this approach reduce the noise of the loss, by averaging the losses over the eval_iters.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # average the losses over the eval_iters
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # we create a tril matrix to mask the future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # make sure they dont communicate with the past
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # randomly prevent some of the nodes from communicating 
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # this is the projection layer that projects the output of the heads to the embedding dimension
        self.dropout = nn.Dropout(dropout)
        
        
        
    def forward(self, x): 
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # linear transformation to the embedding dimension
        out = self.proj(out)
        return out 
    # dim = -1 refers to the last dimension of a tensor
    # In the context of torch.cat([h(x) for h in self.heads], dim=-1):
    # - Each head outputs a tensor of shape (B, T , head_size)
    # - We want to concatenate these outputs along the channel/feature dimension
    # - dim=-1 means concatenate along the last dimension (the head_size dimension)
    # - This results in a tensor of shape (B, T, num_heads * head_size)
    # - For example, if we have 4 heads each with head_size=8, the concatenated result
    #   will have shape (B, T, 32) where 32 = 4 * 8
    # - This is the standard way to combine multiple attention heads in parallel
    # concatenate the heads and project the result to the embedding dimension over the channel dime 
    
    # create multiple independent communication channels -> gather datas -> attention heads -> more expressive model 
    
    
    
# we want to implement computation per node: 
class FeedForward(nn.Module):
    """ 
    FeedForward vs Forward:
    
    FeedForward: A neural network layer that applies linear transformations followed by non-linear activations.
    - Expands input from n_embd to 4*n_embd dimensions
    - Applies ReLU activation for non-linearity
    - Projects back down to n_embd dimensions
    - Includes dropout for regularization
    
    Forward: The method that defines how data flows through the layer during computation.
    - Takes input tensor x and passes it through the network
    - Returns the transformed output
    - Called automatically during model execution
    
    Functionality: FeedForward enables each token to perform independent computation
    after communication (attention), allowing the model to process and transform
    the information gathered from other tokens.
    
    Once they received all the information, they need to think independently and in parallel
    """

    def __init__(self, n_embd):
        """
        The FeedForward layer expands the input from n_embd to 4*n_embd dimensions, applies ReLU activation for non-linearity, projects back down to n_embd dimensions, and includes dropout for regularization.
        
        It is 4 because the dimensionality of the input and output is d_model = 512, and the inner layer has dimensionality d_ff = 2048
        
        they are multipliers to the embedding dimension, and 4 is the default value for the inner layer dimension in the transformer paper.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # First linear layer: expands from embedding dimension to embedding dimension
            nn.ReLU(),                  # ReLU activation function: introduces non-linearity and sets negative values to 0
            nn.Linear(4 * n_embd, n_embd),  # Second linear layer: projects back down to original embedding dimension
            nn.Dropout(dropout) # Dropout layer: randomly sets a certain percentage of neurons to 0 during training to prevent overfitting
        )

    # all the tokens here do the computation in parallel & independently
    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    """ 
    Transformer block: communication followed by computation 
    
    - The transformer block alternates between communication (self-attention) and computation (feedforward).
    - This design mimics the transformer architecture where blocks are grouped and duplicated to create
    the full transformer model. Each block allows tokens to communicate with each other through
    self-attention, then perform independent computation through feedforward layers. Except for the cross-attention layer.
    
    layer normalization: normalize the input to the layer to have a mean of 0 and a standard deviation of 1
    
    we apply the layer norm before the self-attention and feedforward layers
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # fork off from the residual pathway and perform some computation and then project back to the residual pathway
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    
        
# super simple bigram model with attention
class BigramLanguageModel(nn.Module):
    # no need to pass the vocab size because we are using the same vocab size for the input and output

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # n_embd is the number of embedding dimensions, intermediate representation of the token: 32 directional embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # block_size is the maximum context length for predictions, so we need to create a lookup table for the position embeddings
        # self.sa_head = MultiHeadAttention(4, n_embd // 4) # self-attention head
        # self.ffwd = FeedForward(n_embd) # feedforward layer
        # self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer to project the embedding to the vocab size
        # # n_embd // 4 is the number of heads, and n_embd is the number of embedding dimensions
        # # 4 communication channels -> 8 dimensional vectors, and that concatenate the heads and project the result to the embedding dimension over the channel dime -> 32 i.e. 4 heads of 8 - dimensional self-attention (similar to group convolution
        
        # assume we have 5 blocks in the transformer
        # self.blocks =  nn.sequential(
        #     Block(n_embd, n_head=n_head),
        #     Block(n_embd, n_head=n_head), 
        #  for n times -> n layer
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # it is for unpacking the list of blocks into a sequence of modules
        
        
        # n_embd // 4 is the number of heads, and n_embd is the number of embedding dimensions
        # 4 communication channels -> 8 dimensional vectors, and that concatenate the heads and project the result to the embedding dimension over the channel dime -> 32 i.e. 4 heads of 8 - dimensional self-attention (similar to group convolution
        
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer to project the embedding to the vocab size
        
        # n_embd // 4 is the number of heads, and n_embd is the number of embedding dimensions
        # 4 communication channels -> 8 dimensional vectors, and that concatenate the heads and project the result to the embedding dimension over the channel dime -> 32 i.e. 4 heads of 8 - dimensional self-attention (similar to group convolution)
        
        
    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # Handle variable sequence lengths during generation by using modulo
        pos_emb = self.position_embedding_table(torch.arange(T, device=device) % block_size) 
        # (T,C): used to add the position information to the token embeddings
        # pos_emb gets broadcasted across the batch dimension (B) when added to tok_emb
        # This means the same position embeddings are applied to all sequences in the batch
        # x holds the token identities and the position information
        x = tok_emb + pos_emb # (B,T,C)
        
        # apply transformer blocks
        x = self.blocks(x) # (B,T,C)
        
        # apply final layer norm
        x = self.ln_f(x) # (B,T,C)
        
        # project to vocab size for next token prediction
        logits = self.lm_head(x) # (B,T,vocab_size)
        

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens, such that we never use more than block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictionsndex
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # we create a tril matrix to mask the future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # make sure they dont communicate with the past
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
        
        
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("Sample of the model's output:")
generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"generated_text_{timestamp}.txt"
output_path = os.path.join(output_dir, output_filename)

# Save generated text to file
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("Generated Text from GPT Model\n")
    f.write("=" * 50 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model parameters: batch_size={batch_size}, block_size={block_size}, n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}\n")
    f.write("=" * 50 + "\n\n")
    f.write(generated_text)

print(f"Generated text saved to: {output_path}")

# Save the trained model
model_save_path = 'trained_gpt_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'vocab_size': vocab_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'dropout': dropout
    },
    'chars': chars,
    'stoi': stoi,
    'itos': itos
}, model_save_path)
print(f"Model saved to {model_save_path}")
