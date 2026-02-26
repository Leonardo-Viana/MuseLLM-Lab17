# MuseLLM-Lab17 v0.11 — nanoGPT-QEMM

**Quaternion Embodied Muse Model baseado em nanoGPT**  
CPMAT Lab 17, IC/UFAL, Maceió — 26 de fevereiro de 2026

Primeira versão funcional completa do projeto.  
Inclui:
- RoPE quaternion (generalização da técnica complexa usada no Tesla FSD/Optimus)
- Camadas quaternion-aware
- SAE interno para descoberta espontânea de padrões
- Loop de auto-descoberta estética + Navalha de Occam emergente
- Treino em textos científicos (física, relatividade, etc.)

### Como rodar no Colab (gratuito)
```python
!git clone https://github.com/Leonardo-Viana/MuseLLM-Lab17.git
%cd MuseLLM-Lab17
!pip install -r requirements.txt
!python nano_gpt_qemm.py# MuseLLM-Lab17

