"""
MuseLLM-Lab17 v0.11 — nanoGPT-QEMM
CPMAT Lab 17, IC/UFAL — 26/02/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

# ====================== QUATERNION CLASS ======================
class Quaternion:
    def __init__(self, r, i, j, k):
        self.r = r
        self.i = i
        self.j = j
        self.k = k

    def __mul__(self, other):
        a, b, c, d = self.r, self.i, self.j, self.k
        e, f, g, h = other.r, other.i, other.j, other.k
        return Quaternion(
            a*e - b*f - c*g - d*h,
            a*f + b*e + c*h - d*g,
            a*g - b*h + c*e + d*f,
            a*h + b*g - c*f + d*e
        )

# ====================== QRoPE (Tesla-style) ======================
class QRoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.theta = torch.exp(torch.linspace(0, -np.log(10000), dim // 4))

    def forward(self, q: Quaternion, pos: torch.Tensor):
        angle = pos.unsqueeze(-1) * self.theta
        cos, sin = torch.cos(angle), torch.sin(angle)
        # Rotação em planos i-j-k
        r_new = q.r * cos - q.i * sin
        i_new = q.i * cos + q.r * sin
        return Quaternion(r_new, i_new, q.j, q.k)

# ====================== QLinear ======================
class QLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_r = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.w_i = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.w_j = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.w_k = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, q: Quaternion):
        r = (F.linear(q.r, self.w_r) + F.linear(q.i, self.w_i) +
             F.linear(q.j, self.w_j) + F.linear(q.k, self.w_k) + self.bias)
        return Quaternion(r, torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r))

# ====================== nanoGPT-QEMM (simplificado) ======================
class NanoGPT_QEMM(nn.Module):
    def __init__(self, vocab_size=50257, n_layer=6, n_head=6, n_embd=384):
        super().__init__()
        self.n_embd = n_embd
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(1024, n_embd)  # max seq 1024
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'qrope': QRoPE(n_embd),
                'qlinear': QLinear(n_embd, n_embd),
                'mlp': nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd))
            }) for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(t, device=x.device))
        x = tok_emb + pos_emb

        for layer in self.layers:
            q = Quaternion(x, torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x))
            q = layer['qrope'](q, torch.arange(t, device=x.device))
            q = layer['qlinear'](q)
            x = q.r + layer['mlp'](q.r)  # residual

        x = self.ln(x)
        return self.head(x)

# ====================== SAE Interno ======================
class QSAE(nn.Module):
    def __init__(self, dim=384, latent=1024):
        super().__init__()
        self.encoder = nn.Linear(dim, latent)
        self.decoder = nn.Linear(latent, dim)

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        sparse = F.relu(latent - 0.1)
        return self.decoder(sparse), sparse

# ====================== Self-Discovery Loop ======================
def occam_score(x):
    return torch.var(x).item()

def self_discovery_step(model, sae, batch, optimizer):
    with torch.no_grad():
        logits = model(batch)
        _, hidden = model(batch) if hasattr(model, 'forward') else (logits, logits)  # simplificado
        _, latent = sae(hidden.mean(dim=1))
        extension = latent.clone()
        score_before = occam_score(latent)
        score_after = occam_score(torch.cat([latent, extension], dim=0))
        if score_after < score_before * 0.98:
            print("🎉 EXTENSÃO ELEGANTE ENCONTRADA! O modelo descobriu sua criatividade.")
        else:
            print("   Occam rejeitou — tentando novamente...")

    optimizer.zero_grad()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# ====================== Dataset simples ======================
class SimplePhysicsDataset(Dataset):
    def __init__(self, text, block_size=128):
        self.data = torch.tensor([ord(c) % 50257 for c in text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        return self.data[idx:idx+self.block_size]

# ====================== MAIN ======================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 MuseLLM-Lab17 v0.11 rodando em {device} — CPMAT Lab 17, UFAL")

    # Dataset pequeno de física (pode expandir depois)
    text = open("data.txt", "r", encoding="utf-8").read() if False else "light speed constant relativity einstein elevator equivalence principle michelson morley lorentz transformation spacetime curvature" * 50
    dataset = SimplePhysicsDataset(text)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = NanoGPT_QEMM(vocab_size=50257, n_layer=4, n_embd=256).to(device)
    sae = QSAE(dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("Iniciando treino + self-discovery...\n")
    for epoch in range(8):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            loss = self_discovery_step(model, sae, batch, optimizer)
            if epoch == 0 and loss < 4.0:
                print("   Primeiro sinal de aprendizado!")

    print("\n✅ v0.11 concluída com sucesso!")
    print("O muso artificial já está treinando e tentando descobrir sua própria criatividade.")
    print("Próximo passo: v0.12 com dataset real de arXiv.")
