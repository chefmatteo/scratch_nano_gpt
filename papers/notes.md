`ps aux | grep ollama`: check if running
- The command consists of two parts, connected by a pipe (|):
ps aux: This command lists all running processes on the system.
a: Shows processes for all users.
u: Displays detailed, user-oriented output.
x: Includes processes that are not attached to a specific terminal.
grep ollama: The grep command filters the output from ps aux, searching for any lines that contain the text "ollama". This is how you find the process related to the Ollama service. 


`pkill -f ollama`: stop server

## To Do - Transformer Exercises

### EX1: The n-dimensional tensor mastery challenge
Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension (answer is in nanoGPT).

### EX2: Train the GPT on your own dataset of choice!
What other data could be fun to blabber on about? 

**Advanced suggestion**: Train a GPT to do addition of two numbers, i.e. a+b=c. You may find it helpful to predict the digits of c in reverse order, as the typical addition algorithm (that you're hoping it learns) would proceed right to left too. You may want to modify the data loader to simply serve random problems and skip the generation of train.bin, val.bin. You may want to mask out the loss at the input positions of a+b that just specify the problem using y=-1 in the targets (see CrossEntropyLoss ignore_index). Does your Transformer learn to add? 

**Swole doge project**: Build a calculator clone in GPT, for all of +-*/. Not an easy problem. You may need Chain of Thought traces.

### EX3: Pretraining and Fine-tuning
Find a dataset that is very large, so large that you can't see a gap between train and val loss. Pretrain the transformer on this data, then initialize with that model and finetune it on tiny shakespeare with a smaller number of steps and lower learning rate. Can you obtain a lower validation loss by the use of pretraining?

### EX4: Research and Implementation
Read some transformer papers and implement one additional feature or change that people seem to use. Does it improve the performance of your GPT?


`#### Residual Connections: `
The Formula: 
In traditional feedforward neural networks, data flows through each layer sequentially: The output of a layer is the input for the next layer.

Residual connection provides another path for data to reach latter parts of the neural network by skipping some layers. Consider a sequence of layers, layer i to layer i + n, and let F be the function represented by these layers. Denote the input for layer i by x. In the traditional feedforward setting, x will simply go through these layers one by one, and the outcome of layer i + n is F(x). A residual connection that bypasses these layers typically works as follows:


Figure 1. Residual Block. Created by the author.
The residual connection first applies identity mapping to x, then it performs element-wise addition F(x) + x. In literature, the whole architecture that takes an input x and produces output F(x) + x is usually called a residual block or a building block. Quite often, a residual block will also include an activation function such as ReLU applied to F(x) + x.
![Residual Connections](images/residual_connections.png)

The main reason for emphasizing the seemingly superfluous identity mapping in the figure above is that it serves as a placeholder for more complicated functions if needed. For example, the element-wise addition F(x) + x makes sense only if F(x) and x have the same dimensions. If their dimensions are different, we can replace the identity mapping with a linear transformation (i.e. multiplication by a matrix W), and perform F(x) + Wx instead.

In general, multiple residual blocks, which may be of the same or different architectures, are used within the whole neural network.

How does it help training deep neural networks
For feedforward neural networks, training a deep network is usually very difficult, due to problems such as exploding gradients and vanishing gradients. On the other hand, the training process of a neural network with residual connections is empirically shown to converge much more easily, even if the network has several hundreds layers. Like many techniques in deep learning, we still do not fully understand all the details about residual connection. However, we do have some interesting theories that are supported by strong experimental results.

Behaving Like Ensembles of Shallow Neural Networks
For feedforward neural networks, as we have mentioned above, the input will go through each layer of the network sequentially. More technically speaking, the input goes through a single path that has length equal to the number of layers. On the other hand, networks with residual connections consist of many paths of varying lengths.

![Residual Connections 2](images/unraveled_view_of_residual_connections.png)

Figure 2. Unraveled View of Residual Connection. Created by the author.
As an example, consider a network with 3 residual blocks. We can try to expand the formula of this network step by step:


```
x₃ = H(x₂) + x₂
   = H(G(x₁) + x₁) + G(x₁) + x₁
   = H(G(F(x₀) + x₀) + F(x₀) + x₀) + G(F(x₀) + x₀) + F(x₀) + x₀
```
There are 2³ = 8 terms of the input x₀ that contribute to the output x₃. Hence, we can also view this network as a collection of 8 paths, of lengths 0, 1, 2 and 3, as illustrated in the figure above.

Under this “unraveled view”,  shows that networks with residual connections behave like ensembles of networks that do not strongly depend on each other. Moreover, most of the gradient during gradient descent training comes from short paths. In other words, residual connection does not resolve the exploding or vanishing gradient problems. Rather, it avoids those problems by having shallow networks in the “ensembles”.

> Residual connections create a unique architectural pattern where the network can branch off from a main pathway, perform computations, and then merge back into the original pathway.

**Gradient Distribution Through Addition:**
The addition operation in residual connections has a crucial mathematical property: it distributes gradients equally to both of its input branches. When backpropagation occurs, the gradient flowing through an addition node splits evenly between the two paths that fed into it. This means that both the residual pathway (the identity mapping) and the transformed pathway (the learned function F(x)) receive the same gradient signal.

**Gradient Flow and the "Superhighway" Effect:**
The loss signal can propagate through every addition node in the network, creating a direct path from the final supervision signal all the way back to the input layer. This creates what's often called a "gradient superhighway" - an unimpeded pathway that allows gradients to flow directly from the output to the input without being blocked or severely diminished by intermediate layers.

**Initialization Strategy and Early Training:**
During the initial stages of training, the residual blocks are typically initialized in a way that makes them contribute very little to the overall computation. 
- In the beginning, the network essentially behaves like a shallow network, with most of the signal flowing through the residual pathway. As training progresses, the residual blocks gradually learn to contribute more meaningful transformations while maintaining the gradient flow benefits of the residual connections. 
- At initialization, we can go directly from supervision to the input - gradient flow is unimpeded, and blocks gradually kick in over time.


#### `out = torch.cat([h(x) for h in self.heads]`
This line of code `out = torch.cat([h(x) for h in self.heads], dim=-1)` is a very common pattern in PyTorch, especially when implementing structures like Multi-Head Attention. Its core purpose is: to concatenate the output results of multiple parallel processing branches ("heads") along the last dimension, forming a comprehensive output tensor.

Below I will explain each part of this code in detail through an example.

🧠 Code Breakdown

1.  `self.heads`: This is usually an `nn.ModuleList` containing multiple parallel neural network layers (each layer is a "head"). For example, in multi-head attention, each `h` might be an independent linear layer (`nn.Linear`) used to project the input to different subspaces.
2.  `[h(x) for h in self.heads]`: This is a list comprehension. It iterates through each head `h` in `self.heads` and passes the same input `x` to each head for computation. This produces a list where each element is an output tensor from a head processing `x`.
3.  `torch.cat(..., dim=-1)`: This is a concatenation operation. It concatenates all output tensors in the list along the specified dimension (here `dim=-1`, which is the last dimension). `dim=-1` is a common notation that makes the code more robust when tensor dimensions change, as it always selects the last dimension.

📚 Concrete Example: A Simple Multi-Head Processing Module

Suppose we have an input tensor `x` with shape `(batch_size, sequence_length, feature_dim)`. We want to process it independently with 4 different linear layers (i.e., 4 "heads"), then concatenate each head's output along the feature dimension.

Output result:
```
Input shape: torch.Size([2, 3, 8])
Output shape: torch.Size([2, 3, 64]) # Because 4 heads * 16 dimensions per head = 64 dimensions
```

🖼️ Process Diagram

Let's imagine a simple case where `batch_size=1` and `sequence_length=1`, so we can focus on the feature dimension changes:
•   Input `x`: Shape `[1, 1, 8]`, can be viewed as an 8-dimensional feature vector.

•   Each head (h): Is an `nn.Linear(8, 16)`. It receives 8-dimensional input and outputs 16-dimensional results.

•   4 heads: Each head independently processes the input, producing 4 output tensors with shape `[1, 1, 16]`.

•   `torch.cat(..., dim=-1)`: Concatenates these 4 tensors with shape `[1, 1, 16]` along the last dimension (feature dimension), finally producing an output tensor with shape `[1, 1, 64]`.

```
      Input x
    [8-dimensional features]
      ↓
      ↓
┌──────┴──────┐
│             │
h₁ (Linear)   h₂ (Linear)   h₃ (Linear)   h₄ (Linear)
(8->16)       (8->16)       (8->16)       (8->16)
│             │             │             │
Output₁       Output₂       Output₃       Output₄
(16-dim)      (16-dim)      (16-dim)      (16-dim)
└──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │
       └─────────────┼─────────────┘
                     │
             torch.cat(dim=-1)
                     ↓
            Final output out (64-dim)
```

💡 Summary and Key Points

•   **Purpose**: This line of code implements parallel processing and feature fusion. Multiple heads can extract features from different angles or different subspaces of the input, and the concatenation operation aggregates these different feature representations together, forming richer and more powerful comprehensive representations.

•   **Dimension Requirements**: To successfully use `torch.cat`, all tensors must have identical dimensions except for the concatenation dimension (dim). In this example, each head's output must have the same batch dimension and sequence length dimension to concatenate along the feature dimension.

•   **Typical Applications**: The most common application is the multi-head self-attention mechanism in Transformers. Each attention head calculates attention weights for different positions in the input sequence, and finally all heads' outputs are concatenated and then integrated through a linear layer.

Hope this example helps you thoroughly understand this line of code!