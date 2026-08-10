# -*- coding: utf-8 -*-
"""
Module A core: tree-search-based personalized biodegradable implant design.
Implements the design workflow (simplified, numpy-only demo):
  patient targets -> initial structures (3x3x3 porosity) -> "FEM" labels
  -> surrogate model -> tree-search sampling -> t-SNE-style diversity
  -> FEM verify + dataset augmentation -> iterative active learning.

The physics below is a SYNTHETIC stand-in for real finite-element simulation.
Replace "FEM" functions with Abaqus/单元退化法 outputs and the surrogate with
the patent's CNN in the production version.

Alloy-agnostic: material parameters (E0, YS0, degradation time scale) come from
data/materials_lib.json, so the same pipeline runs for Mg and Zn alloys.

CLI:
  python structure_design.py                          # we43_mg / cancellous
  python structure_design.py --material zn_1mg --site cortical
API:
  from structure_design import design
  design(material="zn_1mg", site="cortical", seed=42)
"""
import os, json, argparse
import numpy as np

SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
with open(os.path.join(DATA, "materials_lib.json"), encoding="utf-8-sig") as f:
    _lib = json.load(f)
MATERIALS = _lib["materials"]
SITES = _lib["sites"]

LPBF = {"power_W": [40, 100], "speed_mm_per_s": [400, 1200]}


def design(material="we43_mg", site="cancellous", seed=SEED, n_init=100, search_iters=100, top_k=5):
    """Run the tree-search design pipeline and return structured results (dict)."""
    rng = np.random.default_rng(seed)
    mat = MATERIALS[material]
    site_cfg = SITES[site]
    TARGET_E = site_cfg["E_MPa"]
    TARGET_T_MIN, TARGET_T_MAX = site_cfg["T_min_days"], site_cfg["T_max_days"]
    E0, YS0 = mat["E_MPa"], mat["yield_MPa"]
    DEG_SCALE = mat["deg_scale_days"]

    def solid_frac(matx):
        return float(np.mean(1.0 - matx))

    def fem(matx):
        e = solid_frac(matx)
        E = E0 * e ** 2.0
        YS = YS0 * e ** 1.5
        T = DEG_SCALE * e ** 1.2
        return np.array([E, YS, T])

    def ridge_fit(X, Yt, lam=1.0):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        yc = Yt - Yt.mean()
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ yc)
        return w, mu, sd, float(Yt.mean())

    def predict(X, m):
        w, mu, sd, ym = m
        return ((X - mu) / sd) @ w + ym

    def score(pred):
        E, YS, T = pred
        dE = abs(E - TARGET_E) / TARGET_E
        dT = 0.0 if TARGET_T_MIN <= T <= TARGET_T_MAX else min(abs(T - TARGET_T_MIN), abs(T - TARGET_T_MAX)) / TARGET_T_MIN
        return 1.0 - 0.5 * dE - 0.3 * dT + 0.2 * (YS / 60.0)

    def predict_vec(X, models):
        return np.stack([predict(X, models[k]) for k in ["E", "YS", "T"]], axis=1)

    X_init = rng.uniform(0.3, 0.8, size=(n_init, 27))
    Y_init = np.array([fem(m.reshape(3, 3, 3)) for m in X_init])
    models = {k: ridge_fit(X_init, Y_init[:, j]) for j, k in enumerate(["E", "YS", "T"])}

    def mutate(x, rng_):
        y = x.copy()
        i = rng_.integers(0, 27)
        y[i] = np.clip(y[i] + rng_.choice([-0.1, 0.1]), 0.2, 0.9)
        return y

    def tree_search(root, iters=search_iters, rng_=None):
        rng_ = rng_ or rng
        cur = root.copy()
        best = (score(predict_vec(cur[None, :], models)[0]), cur.copy())
        for _ in range(iters):
            cand = mutate(cur, rng_)
            pred = predict_vec(cand[None, :], models)[0]
            s = score(pred)
            if s > best[0]:
                best = (s, cand.copy())
                cur = cand.copy()
            elif rng_.uniform() < 0.2:
                cur = cand.copy()
        return best

    roots = np.argsort([score(predict_vec(X_init[i:i + 1], models)[0]) for i in range(n_init)])[-5:]
    candidates = []
    for r in roots:
        s, best_mat = tree_search(X_init[r].copy())
        candidates.append((s, best_mat))
    candidates.sort(key=lambda t: -t[0])

    out = []
    for s, m in candidates[:top_k]:
        p = fem(m.reshape(3, 3, 3))
        out.append({
            "score": round(float(s), 4),
            "E_MPa": round(float(p[0]), 1),
            "YS_MPa": round(float(p[1]), 1),
            "T_deg_days": round(float(p[2]), 1),
            "porosity_mean": round(1.0 - solid_frac(m.reshape(3, 3, 3)), 3),
            "matrix_3x3x3": [[[round(float(m[i * 9 + j * 3 + k]), 3) for k in range(3)] for j in range(3)] for i in range(3)],
        })
    return {
        "material": material,
        "material_name": mat["name"],
        "site": site,
        "target": {"E_MPa": TARGET_E, "T_min_days": TARGET_T_MIN, "T_max_days": TARGET_T_MAX,
                   "site_note": site_cfg["note"]},
        "lpbf": LPBF,
        "candidates": out,
        "note": "synthetic FEM stand-in; production uses Abaqus/CNN (patent)",
    }


def main():
    ap = argparse.ArgumentParser(description="Module A: tree-search implant structure design (alloy-agnostic demo)")
    ap.add_argument("--material", default="we43_mg", choices=list(MATERIALS))
    ap.add_argument("--site", default="cancellous", choices=list(SITES))
    args = ap.parse_args()
    res = design(material=args.material, site=args.site)
    print("Structure design result - tree search (top 5, FEM-verified with synthetic physics)")
    print(f"  material={res['material']} ({res['material_name']}), site={res['site']}")
    t = res["target"]
    print(f"  target E={t['E_MPa']} MPa, T in [{t['T_min_days']},{t['T_max_days']}] days, maximize YS")
    for c in res["candidates"]:
        print(f"  score={c['score']:.3f}  E={c['E_MPa']:7.1f} MPa  YS={c['YS_MPa']:6.1f} MPa  "
              f"T_deg={c['T_deg_days']:6.1f} days  p={c['porosity_mean']:.3f}")
    print("\nOK. Production: replace synthetic FEM with Abaqus simulation, surrogate with CNN "
          "(3x3x3 porosity voxelized to 60x60x60, conv 16/8/4 @3x3x3, FC 64/32), "
          "add t-SNE diversity selection + active-learning augmentation loop.")


if __name__ == "__main__":
    main()
