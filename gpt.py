import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from melodyPlay import NOTE_FREQUENCIES, play_melody
from scipy.stats import entropy
import numpy as np
from nltk.util import ngrams
from collections import Counter

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 192
n_head = 4
n_layer = 2
dropout = 0.2
model_save_path = "gpt_melody_model.pt" 
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('inputMelodiesAugmented.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text.split())))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[token] for token in s.split()] # encoder: take a note, output a list of integers
decode = lambda l: ' '.join([itos[i] for i in l]) # decoder: take a list of integers, output a note

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
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
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
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
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
    
def calculate_perplexity(loss):
    return torch.exp(loss.clone().detach())

# Baseline comparison
def generate_random_melody(length):
    notes_with_octaves = list(NOTE_FREQUENCIES.keys())
    melody = random.choices(notes_with_octaves, k=length)
    return ' '.join(melody)

model = GPTLanguageModel().to(device)
if os.path.exists(model_save_path):
    print(f"Loading model from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.eval()
else:

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            perplexity = calculate_perplexity(losses['val'])
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, perplexity {perplexity: .4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_sequence = decode(model.generate(context, max_new_tokens=100)[0].tolist())
print("Generated Melody:", generated_sequence)

# def validate_and_filter_sequence(sequence):
#     valid_tokens = set(NOTE_FREQUENCIES.keys())
#     tokens = sequence.split()
#     filtered_tokens = [token for token in tokens if token in valid_tokens]
#     return filtered_tokens

# filtered_sequence = validate_and_filter_sequence(generated_sequence)

# filtered_sequence_str = " ".join(filtered_sequence)
play_melody(generated_sequence, "generated_melody")

#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

baseline_melody = generate_random_melody(100)
print("Baseline Melody:", baseline_melody)
play_melody(baseline_melody, "baseline_melody")

# Objective Evaluation: Fidelity

def get_note_distribution(sequence, vocab, epsilon=1e-8):
    """Calculate note distribution as normalized frequency to see how closely the distribution
    of notes in generated melody matches the training dataset."""
    distribution = {note: sequence.count(note) for note in vocab}
    total = sum(distribution.values())
    return np.array([(count + epsilon) / (total + epsilon*len(vocab)) for count in distribution.values()])

training_distribution = get_note_distribution(decode(train_data.tolist()).split(), list(stoi.keys()))
generated_distribution = get_note_distribution(generated_sequence.split(), list(stoi.keys()))
kl_divergence = entropy(training_distribution, generated_distribution)
print("KL Divergence (Lower is Better):", kl_divergence)

def calculate_transition_matrix(sequence, vocab, epsilon=1e-8):
    """Calculate transition probabilities between notes. This approach evaluates how closely the 
    probabilities of transitioning between notes in generated melody aligns with those in the dataset."""
    vocab_size = len(vocab)
    matrix = np.zeros((vocab_size, vocab_size))

    for i in range(len(sequence) - 1):
        current_note = stoi[sequence[i]]
        next_note = stoi[sequence[i + 1]]
        matrix[current_note, next_note] += 1

    # Normalize rows to get probabilities
    matrix = matrix + epsilon
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix

training_matrix = calculate_transition_matrix(decode(train_data.tolist()).split(), list(stoi.keys()))
generated_matrix = calculate_transition_matrix(generated_sequence.split(), list(stoi.keys()))
mse = np.mean((training_matrix - generated_matrix) ** 2)
print("Transition Matrix MSE (Lower is Better):", mse)

def calculate_ngram_overlap(sequence, reference_sequence, n=4):
    """Calculate n-gram overlap between two sequences. It measures how many short subsequences(n-grams)
    in the generated melody overlap with those in the dataset."""
    generated_ngrams = Counter(ngrams(sequence, n))
    reference_ngrams = Counter(ngrams(reference_sequence, n))
    overlap = sum((generated_ngrams & reference_ngrams).values())
    total = sum(reference_ngrams.values())
    return overlap / total if total > 0 else 0

reference_sequence = decode(train_data.tolist()).split()
ngram_overlap = calculate_ngram_overlap(generated_sequence.split(), reference_sequence, n=4)
print("N-Gram Overlap (Higher is Better):", ngram_overlap)