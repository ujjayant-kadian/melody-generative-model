#######################################
# gpt.py - Two-Dimensional GPT + Metrics
#######################################
import os
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from math import inf
from collections import Counter
from nltk.util import ngrams
from scipy.stats import entropy
from melodyPlay import play_melody

#######################################
# Hyperparameters
#######################################
batch_size = 64
block_size = 256
max_iters = 2000
eval_interval = 200
eval_iters = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model params
n_embd = 256
n_head = 8
n_layer = 4
dropout = 0.1
model_save_path = "gpt_melody_model.pt"
random_seed = 1337
torch.manual_seed(random_seed)

#######################################
# 1) Read dataset + Build vocab
#######################################
with open('inputMelodiesAugmented.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read().strip()

lines = raw_text.split('\n')

# We'll collect tokens for a global set if needed, but main focus is pitch/rhythm vocab
all_token_pairs = []  # (pitch, rhythm)
pitch_set = set()
rhythm_set = set()

def extract_pitch_rhythm(token):
    """
    If token == 'R', treat as (pitch='R', rhythm='REST'),
    else (pitch=token, rhythm='NOTE').
    """
    if token == 'R':
        return ('R', 'REST')
    else:
        return (token, 'NOTE')

for line in lines:
    tokens = line.split()
    for tk in tokens:
        p, r = extract_pitch_rhythm(tk)
        pitch_set.add(p)
        rhythm_set.add(r)
        all_token_pairs.append((p, r))

pitch_vocab = sorted(list(pitch_set))
rhythm_vocab = sorted(list(rhythm_set))

pitch_stoi = {p: i for i, p in enumerate(pitch_vocab)}
pitch_itos = {i: p for p, i in pitch_stoi.items()}
rhythm_stoi = {r: i for i, r in enumerate(rhythm_vocab)}
rhythm_itos = {i: r for r, i in rhythm_stoi.items()}

pitch_vocab_size = len(pitch_vocab)
rhythm_vocab_size = len(rhythm_vocab)
print("Pitch vocab size:", pitch_vocab_size)
print("Rhythm vocab size:", rhythm_vocab_size)

# Encode the entire dataset
pitch_ids = []
rhythm_ids = []
for (p, r) in all_token_pairs:
    p_id = pitch_stoi[p]
    r_id = rhythm_stoi[r]
    pitch_ids.append(p_id)
    rhythm_ids.append(r_id)

pitch_ids = torch.tensor(pitch_ids, dtype=torch.long)
rhythm_ids = torch.tensor(rhythm_ids, dtype=torch.long)
data_size = len(pitch_ids)
print("Data size (# tokens):", data_size)

# Train/Val split
split_idx = int(0.9 * data_size)
pitch_train = pitch_ids[:split_idx]
pitch_val = pitch_ids[split_idx:]
rhythm_train = rhythm_ids[:split_idx]
rhythm_val = rhythm_ids[split_idx:]

def get_batch(split):
    if split == 'train':
        data_pitch = pitch_train
        data_rhythm = rhythm_train
    else:
        data_pitch = pitch_val
        data_rhythm = rhythm_val
    ix = torch.randint(len(data_pitch) - block_size, (batch_size,))
    x_pitch = torch.stack([data_pitch[i : i + block_size] for i in ix])
    x_rhythm = torch.stack([data_rhythm[i : i + block_size] for i in ix])
    y_pitch = torch.stack([data_pitch[i + 1 : i + block_size + 1] for i in ix])
    y_rhythm = torch.stack([data_rhythm[i + 1 : i + block_size + 1] for i in ix])
    x_pitch, x_rhythm = x_pitch.to(device), x_rhythm.to(device)
    y_pitch, y_rhythm = y_pitch.to(device), y_rhythm.to(device)
    return x_pitch, x_rhythm, y_pitch, y_rhythm

#######################################
# 2) Specialized GPT Implementation
#######################################
class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """2-layer feed-forward."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: self-attention + feed-forward."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

class SpecializedGPT(nn.Module):
    def __init__(self,
                 pitch_vocab_size,
                 rhythm_vocab_size,
                 n_embd,
                 n_head,
                 n_layer,
                 block_size,
                 dropout,
                 device):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        self.device = device
        
        # Embeddings
        self.pitch_embedding_table = nn.Embedding(pitch_vocab_size, n_embd)
        self.rhythm_embedding_table = nn.Embedding(rhythm_vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)

        # Separate output heads
        self.pitch_head   = nn.Linear(n_embd, pitch_vocab_size)
        self.rhythm_head  = nn.Linear(n_embd, rhythm_vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, pitch_idx, rhythm_idx, targets=None):
        B, T = pitch_idx.shape
        pitch_emb = self.pitch_embedding_table(pitch_idx)      # (B,T,n_embd)
        rhythm_emb = self.rhythm_embedding_table(rhythm_idx)   # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,n_embd)

        x = pitch_emb + rhythm_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits_pitch = self.pitch_head(x)     # (B,T,pitch_vocab_size)
        logits_rhythm = self.rhythm_head(x)   # (B,T,rhythm_vocab_size)

        loss = None
        if targets is not None:
            pitch_targets, rhythm_targets = targets
            pitch_loss = F.cross_entropy(logits_pitch.view(-1, pitch_vocab_size),
                                         pitch_targets.view(-1))
            rhythm_loss = F.cross_entropy(logits_rhythm.view(-1, rhythm_vocab_size),
                                          rhythm_targets.view(-1))
            loss = (pitch_loss + rhythm_loss) / 2.0

        return (logits_pitch, logits_rhythm), loss

    def generate(self, pitch_context, rhythm_context, max_new_tokens=100):
        for _ in range(max_new_tokens):
            pitch_cond = pitch_context[:, -self.block_size:]
            rhythm_cond = rhythm_context[:, -self.block_size:]
            (logits_pitch, logits_rhythm), _ = self.forward(pitch_cond, rhythm_cond)
            logits_pitch   = logits_pitch[:, -1, :]   # (B, pitch_vocab_size)
            logits_rhythm  = logits_rhythm[:, -1, :]  # (B, rhythm_vocab_size)
            probs_pitch    = F.softmax(logits_pitch, dim=-1)
            probs_rhythm   = F.softmax(logits_rhythm, dim=-1)
            pitch_next     = torch.multinomial(probs_pitch,  num_samples=1)
            rhythm_next    = torch.multinomial(probs_rhythm, num_samples=1)
            pitch_context  = torch.cat((pitch_context, pitch_next), dim=1)
            rhythm_context = torch.cat((rhythm_context, rhythm_next), dim=1)
        return pitch_context, rhythm_context

#######################################
# 3) Setup, Train or Load
#######################################
model = SpecializedGPT(
    pitch_vocab_size,
    rhythm_vocab_size,
    n_embd,
    n_head,
    n_layer,
    block_size,
    dropout,
    device
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            xP, xR, yP, yR = get_batch(split)
            _, loss = model(xP, xR, (yP, yR))
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        perplexity = torch.exp(torch.tensor(avg_loss))
        out[split] = {'loss': avg_loss, 'perplexity': perplexity.item()}
    model.train()
    return out

if os.path.exists(model_save_path):
    print(f"Loading existing model from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
else:
    print(f"Model param count: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            metrics = estimate_loss()
            train_loss = metrics['train']['loss']
            val_loss = metrics['val']['loss']
            train_ppl = metrics['train']['perplexity']
            val_ppl = metrics['val']['perplexity']
            print(f"Step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, perplexity {val_ppl:.2f}")
        xP, xR, yP, yR = get_batch('train')
        (_, _), loss = model(xP, xR, (yP, yR))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

#######################################
# 4) Generate a Sample
#######################################
model.eval()
start_pitch = torch.tensor([[pitch_stoi['R']]], dtype=torch.long, device=device)
start_rhythm = torch.tensor([[rhythm_stoi['REST']]], dtype=torch.long, device=device)

gen_pitch, gen_rhythm = model.generate(start_pitch, start_rhythm, max_new_tokens=100)

def decode_pitch_rhythm(p_id, r_id):
    p_token = pitch_itos[p_id]
    r_token = rhythm_itos[r_id]
    if r_token == 'REST':
        return 'R'
    else:
        return p_token

# Flatten out the first (and only) batch
generated_sequence_ids = list(zip(gen_pitch[0].tolist(), gen_rhythm[0].tolist()))
generated_sequence = [decode_pitch_rhythm(p_id, r_id) for (p_id, r_id) in generated_sequence_ids]

print("Generated Melody:", ' '.join(generated_sequence))
play_melody(' '.join(generated_sequence), "generated_melody")

###########################################
# 8) Baseline Model: Random Melody
###########################################
def generate_random_melody(length=100):
    """
    Generate a random melody from the global_tokens (any token).
    or purely from pitch_vocab, if you want only pitches + rests.
    """
    # If you want purely pitch-based random melody:
    pitch_list = list(pitch_stoi.keys())
    melody = [random.choice(pitch_list) for _ in range(length)]
    return ' '.join(melody)

baseline_melody = generate_random_melody(100)
print("Baseline Melody:", baseline_melody)
play_melody(baseline_melody, "baseline_melody")

#######################################
# 5) Adapted Fidelity Metrics
#    (Pitch & Rhythm separately)
#######################################
def compute_distribution_counts(ids, vocab_size):
    counts = np.zeros(vocab_size, dtype=np.float64)
    for idx in ids:
        counts[idx] += 1
    return counts

def compute_normalized_distribution(ids, vocab_size, epsilon=1e-8):
    counts = compute_distribution_counts(ids, vocab_size)
    total = np.sum(counts)
    dist = (counts + epsilon) / (total + epsilon * vocab_size)
    return dist

def kl_divergence(real_ids, gen_ids, vocab_size):
    """
    Compare distribution of real_ids vs. gen_ids (pitch or rhythm).
    """
    real_dist = compute_normalized_distribution(real_ids, vocab_size)
    gen_dist  = compute_normalized_distribution(gen_ids, vocab_size)
    return entropy(real_dist, gen_dist)

def calculate_transition_matrix(ids, vocab_size, epsilon=1e-8):
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for i in range(len(ids) - 1):
        curr_id = ids[i]
        next_id = ids[i + 1]
        matrix[curr_id, next_id] += 1
    matrix += epsilon
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix

def transition_matrix_mse(real_ids, gen_ids, vocab_size):
    real_matrix = calculate_transition_matrix(real_ids, vocab_size)
    gen_matrix  = calculate_transition_matrix(gen_ids, vocab_size)
    return np.mean((real_matrix - gen_matrix)**2)

def ngram_overlap(ref_ids, gen_ids, n=4):
    ref_ngrams = Counter(list(ngrams(ref_ids, n)))
    gen_ngrams = Counter(list(ngrams(gen_ids, n)))
    overlap_count = sum((ref_ngrams & gen_ngrams).values())
    total = sum(ref_ngrams.values())
    return overlap_count / total if total > 0 else 0

############################################
# 6) Evaluate on pitch and rhythm separately
############################################
# We'll treat the training set as "real data".
# (Or you might have a separate "test set".)
real_pitch_ids_np = pitch_train.cpu().numpy()
real_rhythm_ids_np = rhythm_train.cpu().numpy()

# The generated pitch/rhythm:
generated_pitch_ids_np = np.array(gen_pitch[0].tolist(), dtype=np.int32)
generated_rhythm_ids_np = np.array(gen_rhythm[0].tolist(), dtype=np.int32)

# 6a) KL Divergence
pitch_kl = kl_divergence(real_pitch_ids_np, generated_pitch_ids_np, pitch_vocab_size)
rhythm_kl = kl_divergence(real_rhythm_ids_np, generated_rhythm_ids_np, rhythm_vocab_size)
print(f"[Pitch] KL Divergence: {pitch_kl:.4f}")
print(f"[Rhythm] KL Divergence: {rhythm_kl:.4f}")

# 6b) Transition Matrix MSE
pitch_trans_mse = transition_matrix_mse(real_pitch_ids_np, generated_pitch_ids_np, pitch_vocab_size)
rhythm_trans_mse = transition_matrix_mse(real_rhythm_ids_np, generated_rhythm_ids_np, rhythm_vocab_size)
print(f"[Pitch] Transition Matrix MSE: {pitch_trans_mse:.6f}")
print(f"[Rhythm] Transition Matrix MSE: {rhythm_trans_mse:.6f}")

# 6c) N-gram Overlap (4-gram by default)
pitch_ngram_overlap = ngram_overlap(real_pitch_ids_np, generated_pitch_ids_np, n=4)
rhythm_ngram_overlap = ngram_overlap(real_rhythm_ids_np, generated_rhythm_ids_np, n=4)
print(f"[Pitch] 4-gram Overlap: {pitch_ngram_overlap:.4f}")
print(f"[Rhythm] 4-gram Overlap: {rhythm_ngram_overlap:.4f}")
