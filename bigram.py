# gpt: generative pretraining transformer; 
# recreating chatgpt from scratch 
# author: Matthew Ng @ust, {ctngah}@connect.ust.hk

# Attention is all you need: 
# train a transformer based model, character level

# input: with a given amount of data, we would like the transformer able to predict the next character after a given sequence of characters. 
# this model doesnt do anything yet because the only important part is that the nn.embedding, to create the look up table, but it is not actually a transformer. The model perform the same operation in both training and evaluation mode. 

# we do not have the dropoff layer (i.e. temporarily remove/deactivate a certain % of neurons in neural network layer during each iteraction of the training process)
# no batch normalization layer (i.e. normalize the input to the layer to have a mean of 0 and a standard deviation of 1)

# Some model will hv different behaviour in inferencing and evaluation time, such as:
# inference time: the duration it takes for a trainde machine learning model to process new input data and generate an output or prediction. 

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

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

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
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
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))