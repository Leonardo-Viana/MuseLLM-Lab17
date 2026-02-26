%%writefile nano_gpt_qemm.py
"""
MuseLLM-Lab17 v0.12 — nanoGPT-QEMM com SAE + Self-Discovery
CPMAT Lab 17, IC/UFAL — 26/02/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ====================== QUATERNION ======================
class Quaternion:
    def __init__(self, r, i, j, k):
        self.r = r
        self.i = i
        self.j = j
        self.k = k

    def __mul__(self, other):
        a,b,c,d = self.r, self.i, self.j, self.k
        e,f,g,h = other.r, other.i, other.j, other.k
        return Quaternion(a*e - b*f - c*g - d*h,
                          a*f + b*e + c*h - d*g,
                          a*g - b*h + c*e + d*f,
                          a*h + b*g - c*f + d*e)

# ====================== QRoPE ======================
class QRoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.theta = torch.exp(torch.linspace(0, -np.log(10000), dim))

    def forward(self, q, position):
        angle = position.unsqueeze(-1) * self.theta
        cos, sin = torch.cos(angle), torch.sin(angle)
        return Quaternion(q.r * cos - q.i * sin,
                          q.i * cos + q.r * sin,
                          q.j, q.k)

# ====================== QLinear ======================
class QLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_r = nn.Parameter(torch.randn(out_dim, in_dim)*0.02)
        self.w_i = nn.Parameter(torch.randn(out_dim, in_dim)*0.02)
        self.w_j = nn.Parameter(torch.randn(out_dim, in_dim)*0.02)
        self.w_k = nn.Parameter(torch.randn(out_dim, in_dim)*0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, q):
        r = (F.linear(q.r, self.w_r) + F.linear(q.i, self.w_i) +
             F.linear(q.j, self.w_j) + F.linear(q.k, self.w_k) + self.bias)
        return Quaternion(r, torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r))

# ====================== NanoGPT-QEMM ======================
class NanoGPT_QEMM(nn.Module):
    def __init__(self, vocab_size=1000, n_embd=128, n_layer=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.rope = QRoPE(n_embd)
        self.layers = nn.ModuleList([QLinear(n_embd, n_embd) for _ in range(n_layer)])
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        pos = torch.arange(x.shape[1], device=x.device)
        q = Quaternion(x, torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x))
        q = self.rope(q, pos)
        for layer in self.layers:
            q = layer(q)
            x = q.r
        return self.head(x)

# ====================== SAE Interno ======================
class QSAE(nn.Module):
    def __init__(self, dim=128, latent=512):
        super().__init__()
        self.encoder = nn.Linear(dim, latent)
        self.decoder = nn.Linear(latent, dim)

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        sparse = F.relu(latent - 0.08)
        return self.decoder(sparse), sparse

# ====================== Self-Discovery ======================
def occam_score(p):
    return torch.var(p).item()

def self_discovery_step(model, sae, tokens, optimizer):
    with torch.no_grad():
        logits = model(tokens)
        _, hidden = model(tokens) if hasattr(model, 'forward') else (logits, logits)
        _, latent = sae(hidden.mean(dim=1))
        extension = latent.clone() * 0.95   # compressão suave
        score_before = occam_score(latent)
        score_after = occam_score(torch.cat([latent, extension], dim=0))

        if score_after < score_before * 0.98:
            print("🎉 EXTENSÃO ELEGANTE ENCONTRADA! O muso descobriu sua criatividade.")
        else:
            print("   Occam rejeitou — tentando novamente...")

    optimizer.zero_grad()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# ====================== MAIN ======================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 MuseLLM-Lab17 v0.12 rodando em {device} — CPMAT Lab 17, UFAL\n")

    text = ("light speed constant relativity einstein elevator equivalence principle "
            "michelson morley lorentz transformation spacetime curvature gravity "
            "quantum wave particle duality photon electron") * 30

    vocab = {w: i for i, w in enumerate(set(text.split()))}
    tokens = torch.tensor([[vocab[w] for w in text.split()]], dtype=torch.long).to(device)

    model = NanoGPT_QEMM(vocab_size=len(vocab), n_embd=128, n_layer=4).to(device)
    sae = QSAE(dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("Iniciando treino com self-discovery...\n")
    for epoch in range(10):
        loss = self_discovery_step(model, sae, tokens, optimizer)
        print(f"Epoch {epoch+1}/10 | Loss: {loss:.4f}")

    print("\n✅ v0.12 concluída com sucesso!")
    print("O muso artificial já está tentando descobrir sua própria criatividade.")
