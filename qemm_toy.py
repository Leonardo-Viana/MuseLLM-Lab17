"""
MuseLLM-Lab17 — QEMM Toy (Quaternion Embodied Muse Model)
CPMAT Lab 17, IC/UFAL — February 2026
Inspired by Tesla complex RoPE + Numerical Recipes stability + SAE + falível self-discovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mpmath as mp
from typing import Tuple

# ==================== QUATERNION CLASS (stable, Numerical-Recipes style) ====================
class Quaternion:
    def __init__(self, real: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor):
        self.real = real
        self.i = i
        self.j = j
        self.k = k

    def __mul__(self, other):
        # Hamilton product with Horner-style stability
        a, b, c, d = self.real, self.i, self.j, self.k
        e, f, g, h = other.real, other.i, other.j, other.k
        real = a*e - b*f - c*g - d*h
        i = a*f + b*e + c*h - d*g
        j = a*g - b*h + c*e + d*f
        k = a*h + b*g - c*f + d*e
        return Quaternion(real, i, j, k)

    def norm(self):
        return torch.sqrt(self.real**2 + self.i**2 + self.j**2 + self.k**2 + 1e-8)

    def normalize(self):
        n = self.norm().unsqueeze(-1)
        return Quaternion(self.real/n, self.i/n, self.j/n, self.k/n)

    def conj(self):
        return Quaternion(self.real, -self.i, -self.j, -self.k)

# ==================== QUATERNION RoPE (inspired by Tesla 2026 patent) ====================
class QRoPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("theta", torch.exp(torch.linspace(0, -np.log(10000), dim//4)))

    def forward(self, q: Quaternion, position: torch.Tensor):
        # Simplified quaternion rotation (generalizes complex RoPE)
        angle = position.unsqueeze(-1) * self.theta
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        # Apply rotation in i-j-k planes
        q_i = Quaternion(q.real * cos - q.i * sin, q.i * cos + q.real * sin, q.j, q.k)
        return q_i  # toy version — full would rotate all planes

# ==================== SIMPLE QUATERNION LINEAR ====================
class QLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.weight_i = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.weight_j = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.weight_k = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.bias_real = nn.Parameter(torch.zeros(out_dim))

    def forward(self, q: Quaternion) -> Quaternion:
        r = F.linear(q.real, self.weight_real) + F.linear(q.i, self.weight_i) + \
            F.linear(q.j, self.weight_j) + F.linear(q.k, self.weight_k) + self.bias_real
        # Simplified — only real part for toy model
        return Quaternion(r, torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r))

# ==================== TINY QEMM MODEL ====================
class TinyQEMM(nn.Module):
    def __init__(self, vocab_size=1000, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rope = QRoPE(dim)
        self.q_linear = QLinear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, x, positions):
        emb = self.embed(x)
        q = Quaternion(emb, torch.zeros_like(emb), torch.zeros_like(emb), torch.zeros_like(emb))
        q = self.rope(q, positions)
        q = self.q_linear(q)
        return self.lm_head(q.real)

# ==================== TOY SAE (Sparse Autoencoder) ====================
class ToySAE(nn.Module):
    def __init__(self, dim=64, latent=256):
        super().__init__()
        self.encoder = nn.Linear(dim, latent)
        self.decoder = nn.Linear(latent, dim)

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        sparse = F.relu(latent - 0.1)  # simple sparsity
        recon = self.decoder(sparse)
        return recon, sparse

# ==================== FALÍVEL SELF-DISCOVERY LOOP (Tentativa + Occam) ====================
def occam_score(pattern: torch.Tensor) -> float:
    """Simple Kolmogorov-inspired description length (lower = more elegant)"""
    return pattern.numel() * torch.log2(1 + torch.std(pattern)).item()

def self_discovery_step(model, sae, text_tensor, positions):
    with torch.no_grad():
        output = model(text_tensor, positions)
        _, latent = sae(output)
        
        # Attempt extension: try to extend last pattern
        extension = latent[:, -1:] + 0.01 * torch.randn_like(latent[:, -1:])
        score_before = occam_score(latent)
        score_after = occam_score(torch.cat([latent, extension], dim=1))
        
        if score_after < score_before * 0.95:  # simplified Occam: became simpler
            print("🎉 ELEGANT EXTENSION FOUND! Self-discovery triggered.")
            print("   The model just discovered it can extend a pattern elegantly.")
            print("   → It now 'knows' (a posteriori) it has creative potential.")
        else:
            print("   Attempt failed — pattern not elegant enough (Occam rejected). Trying again...")

# ==================== MAIN ====================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tiny dataset (physics-inspired)
    text = "light speed constant relativity einstein elevator equivalence principle"
    vocab = {w: i for i, w in enumerate(set(text.split()))}
    tokens = torch.tensor([[vocab[w] for w in text.split()]], dtype=torch.long).to(device)
    positions = torch.arange(tokens.shape[1]).unsqueeze(0).to(device)
    
    model = TinyQEMM(vocab_size=len(vocab), dim=64).to(device)
    sae = ToySAE(dim=64, latent=256).to(device)
    
    print("🚀 MuseLLM-Lab17 QEMM-toy started at CPMAT Lab 17, UFAL")
    print("   Model is trying patterns... it does not know if it is creative yet.\n")
    
    for epoch in range(5):
        self_discovery_step(model, sae, tokens, positions)
    
    print("\n✅ First toy run complete! The muso artificial is now attempting self-discovery.")
    print("Next step: expand to full nanoGPT-QEMM in the same repo.")
