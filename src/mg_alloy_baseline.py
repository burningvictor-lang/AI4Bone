# -*- coding: utf-8 -*-
"""
GOAI baseline - Module A: Mg alloy property prediction + reverse design (demo).
Only depends on numpy -> runs anywhere, fully reproducible (SEED fixed).

Data: mg_alloy_sample.csv (PLACEHOLDER sample to demonstrate the pipeline).
Replace with the real open dataset: Zenodo record 17672235
  "Accelerating the Design of Resorbable Magnesium Alloys" (410 samples).
Run: python mg_alloy_baseline.py
"""
import os, csv, random
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CSV = os.path.join(DATA, "DatasetMg_imputed.csv")
if not os.path.exists(CSV):
    CSV = os.path.join(DATA, "mg_alloy_sample.csv")
    print("NOTE: Zenodo DatasetMg_imputed.csv not found - using sample demo data")
FEATURES = ["Zn", "Ca", "Sr", "Mn"]
TARGETS = ["UTS", "YieldStrength", "Elongation", "DegradationRate"]

def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

rows = load(CSV)
X = np.array([[r[k] for k in FEATURES] for r in rows], dtype=float)
Y = {t: np.array([r[t] for r in rows], dtype=float) for t in TARGETS}
n = len(rows)
print(f"loaded {n} rows; features={FEATURES}")

# ridge regression with intercept: center X AND y (features standardized)
def fit_model(Xtr, ytr, lam=1.0):
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0) + 1e-9
    Xs = (Xtr - mu) / sd
    yc = ytr - ytr.mean()
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ yc)
    return {"w": w, "mu": mu, "sd": sd, "ymean": float(ytr.mean())}

def predict(X, m):
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["ymean"]

def metrics(y, pred):
    mae = float(np.mean(np.abs(pred - y)))
    r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    return mae, r2

# ---- hold-out evaluation (train-only standardization, fixed SEED) ----
idx = list(range(n)); random.shuffle(idx)
k = max(2, int(n * 0.25))
te, tr = idx[:k], idx[k:]
print("\n--- prediction results (held-out %d of %d rows) ---" % (len(te), n))
full_models = {}
for t in TARGETS:
    y = Y[t]
    m = fit_model(X[tr], y[tr])
    pred = predict(X[te], m)
    mae, r2 = metrics(y[te], pred)
    print(f"{t:>15s}: MAE={mae:7.2f}  R2={r2:6.3f}")
    full_models[t] = fit_model(X, y)   # full-data model for reverse design

# ---- permutation importance on UTS (full-data model) ----
print("\n--- permutation importance (UTS, full-data model) ---")
m = full_models["UTS"]
base = metrics(Y["UTS"], predict(X, m))[1]
for j, name in enumerate(FEATURES):
    Xp = X.copy(); np.random.shuffle(Xp[:, j])
    r2p = metrics(Y["UTS"], predict(Xp, m))[1]
    print(f"  {name}: R2_drop={base - r2p:.4f}")

# ---- reverse design: grid search constrained by clinical requirement ----
print("\n--- reverse design demo: maximize UTS with DegradationRate <= 0.40 mm/y ---")
best = None
for zn in np.arange(1.0, 5.01, 0.5):
    for ca in np.arange(0.2, 1.01, 0.2):
        for sr in np.arange(0.1, 0.71, 0.2):
            for mn in np.arange(0.1, 0.51, 0.2):
                x = np.array([zn, ca, sr, mn])
                p = {t: float(predict(x[None, :], full_models[t])[0]) for t in TARGETS}
                if p["DegradationRate"] <= 0.40 and (best is None or p["UTS"] > best[0]["UTS"]):
                    best = (p, x)
if best:
    p, x = best
    print(f"  recommended composition (wt%): Zn={x[0]:.1f}, Ca={x[1]:.1f}, Sr={x[2]:.1f}, Mn={x[3]:.1f}")
    print(f"  predicted UTS={p['UTS']:.1f} MPa, degradation={p['DegradationRate']:.3f} mm/y, "
          f"YS={p['YieldStrength']:.1f} MPa, Elong={p['Elongation']:.1f}%")
else:
    print("  no composition met the constraint in the sampled grid")

print("\nbaseline OK (demo data). Next: swap in Zenodo 410-sample dataset, add SHAP + NSGA-II.")
