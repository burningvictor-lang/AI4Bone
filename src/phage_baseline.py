# -*- coding: utf-8 -*-
"""
GOAI baseline - Module B: phage lysin candidate ranking (pipeline skeleton).
Only depends on numpy. Sequences in lysin_candidates.txt are SYNTHETIC
placeholders to demonstrate the pipeline; replace with real candidates
mined from phagesDB/NCBI (DeepLysin-style) later.
Run: python phage_baseline.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FASTA = os.path.join(HERE, "lysin_candidates.txt")

def load_fasta(path):
    seqs, cur = [], None
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].strip(); seqs.append([cur, ""])
            else:
                seqs[-1][1] += line
    return seqs

seqs = load_fasta(FASTA)
print(f"loaded {len(seqs)} candidate sequences (synthetic demo)")

BASIC = set("KRH"); HYDRO = set("AILMFVWYPCG")

def features(aa):
    n = len(aa)
    if n == 0:
        return np.zeros(4)
    k = sum(1 for c in aa if c in BASIC) / n
    h = sum(1 for c in aa if c in HYDRO) / n
    # simple heuristics used only for the demo ranking
    return np.array([n, k, h, np.sin(n / 50.0)])

names, F = [], []
for name, aa in seqs:
    names.append(name)
    F.append(features(aa))
F = np.array(F)

# demo score: longer + more basic (cationic lysins) rank higher
score = F[:, 0] / 100.0 + F[:, 1] * 2.0
order = np.argsort(-score)
print("\n--- ranked candidates (synthetic demo scoring) ---")
for i in order:
    print(f"  {names[i]:45s} len={int(F[i,0]):4d} basic_frac={F[i,1]:.2f}  score={score[i]:.3f}")

print("\nskeleton OK. Next: real data pipeline = ")
print("  1) mine lysin candidates from uncharacterized phages (DeepLysin-style)")
print("  2) embed sequences with ESM protein language model")
print("  3) train host-range / activity predictor and output ranked candidates")

