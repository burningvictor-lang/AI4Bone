# -*- coding: utf-8 -*-
"""
GOAI baseline - Module A-1: alloy-agnostic design targets + Mg alloy property prediction.
Only depends on numpy/stdlib -> runs anywhere, fully reproducible (SEED=42).

Two layers:
  1) Alloy scenario layer: data/materials_lib.json holds reference material parameters
     for WE43 Mg and Zn-based alloys. The same structure-design pipeline can be applied
     to each (Mg or Zn) by swapping material parameters -- no new dataset needed here.
  2) Mg composition-property layer: ridge models on Mg composition data
     (Zenodo record 17672235, 410 samples when present; else data/mg_alloy_sample.csv).
     The Mg dataset covers Mg compositions only -- zinc composition modeling needs a
     separate curated dataset and is deferred to the semi-final round.

Run:
  python mg_alloy_baseline.py                      # default alloy: we43_mg
  python mg_alloy_baseline.py --alloy zn_1mg
  python mg_alloy_baseline.py --list-alloys
"""
import os, csv, json, random, argparse
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
LIB = os.path.join(DATA, "materials_lib.json")

# ---------- 1) alloy scenario layer ----------
with open(LIB, encoding="utf-8-sig") as f:
    MATERIALS = json.load(f)["materials"]
SITES = json.load(open(LIB, encoding="utf-8-sig"))["sites"]

def alloy_lines(names=None):
    names = names or list(MATERIALS)
    out = []
    for n in names:
        m = MATERIALS[n]
        out.append(f"  {n:10s} | {m['family'].upper():3s} | E0={m['E_MPa']:6.0f} MPa | "
                   f"YS0={m['yield_MPa']:5.0f} MPa | corr~={m['corrosion_mm_per_year']:.2f} mm/y | "
                   f"deg_scale={m['deg_scale_days']:.0f} d")
    return "\n".join(out)

def design_targets(name, site):
    m = MATERIALS[name]; s = SITES[site]
    return s["E_MPa"], s["T_min_days"], s["T_max_days"], s["note"]

# ---------- 2) Mg composition-property layer ----------
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

def run_mg_pipeline():
    rows = load(CSV)
    X = np.array([[r[k] for k in FEATURES] for r in rows], dtype=float)
    Y = {t: np.array([r[t] for r in rows], dtype=float) for t in TARGETS}
    n = len(rows)
    print(f"\n[Mg composition-property] loaded {n} rows; features={FEATURES}")

    idx = list(range(n)); random.shuffle(idx)
    k = max(2, int(n * 0.25))
    te, tr = idx[:k], idx[k:]
    print("--- prediction results (held-out %d of %d rows) ---" % (len(te), n))
    full_models = {}
    for t in TARGETS:
        y = Y[t]
        m = fit_model(X[tr], y[tr])
        pred = predict(X[te], m)
        mae, r2 = metrics(y[te], pred)
        print(f"{t:>15s}: MAE={mae:7.2f}  R2={r2:6.3f}")
        full_models[t] = fit_model(X, y)

    print("--- permutation importance (UTS, full-data model) ---")
    m = full_models["UTS"]
    base = metrics(Y["UTS"], predict(X, m))[1]
    for j, name in enumerate(FEATURES):
        Xp = X.copy(); np.random.shuffle(Xp[:, j])
        r2p = metrics(Y["UTS"], predict(Xp, m))[1]
        print(f"  {name}: R2_drop={base - r2p:.4f}")

    print("--- reverse design demo: maximize UTS with DegradationRate <= 0.40 mm/y ---")
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

def main():
    ap = argparse.ArgumentParser(description="Module A-1: alloy scenarios + Mg composition-property demo")
    ap.add_argument("--alloy", default="we43_mg", choices=list(MATERIALS),
                    help="primary alloy scenario (used for design-target table)")
    ap.add_argument("--list-alloys", action="store_true", help="list alloys in materials_lib.json and exit")
    args = ap.parse_args()

    print("=== Alloy scenario layer (materials_lib.json) ===")
    print(alloy_lines())
    if args.list_alloys:
        return
    print(f"\n=== Design targets for primary alloy: {args.alloy} ===")
    for site in SITES:
        E, t0, t1, note = design_targets(args.alloy, site)
        print(f"  {site:14s}: target E={E:6.0f} MPa, T in [{t0:.0f},{t1:.0f}] days | {note}")

    run_mg_pipeline()
    print("\nNote: Mg dataset covers Mg compositions only. Zinc composition modeling needs a "
          "separate curated dataset (deferred to semi-final). Structure design itself is "
          "alloy-agnostic via materials_lib.json + mcts_structure_design.py --material.")
    print("Next: swap in Zenodo 410-sample dataset, add SHAP + NSGA-II.")

if __name__ == "__main__":
    main()
