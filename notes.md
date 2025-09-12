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