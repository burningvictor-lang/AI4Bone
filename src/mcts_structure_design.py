# -*- coding: utf-8 -*-
"""
Module A core: MCTS-based personalized biodegradable implant design.
Implements the patent workflow (simplified, numpy-only demo):
  patient targets -> initial structures (3x3x3 porosity) -> "FEM" labels
  -> surrogate model -> Monte Carlo Tree Search sampling -> t-SNE-style diversity
  -> FEM verify + dataset augmentation -> iterative active learning.

The physics below is a SYNTHETIC stand-in for real finite-element simulation.
Replace "FEM" functions with Abaqus/单元退化法 outputs and the surrogate with
the patent's CNN in the production version.

Alloy-agnostic: material parameters (E0, YS0, degradation time scale) come from
data/materials_lib.json, so the same pipeline runs for Mg and Zn alloys.

Run:
  python mcts_structure_design.py                          # we43_mg / cancellous
  python mcts_structure_design.py --material zn_1mg --site cortical
"""
import os, json, argparse
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
with open(os.path.join(DATA, "materials_lib.json"), encoding="utf-8-sig") as f:
    _lib = json.load(f)
MATERIALS = _lib["materials"]
SITES = _lib["sites"]

def main():
    ap = argparse.ArgumentParser(description="Module A: MCTS implant structure design (alloy-agnostic demo)")
    ap.add_argument("--material", default="we43_mg", choices=list(MATERIALS), help="alloy from materials_lib.json")
    ap.add_argument("--site", default="cancellous", choices=list(SITES), help="bone site design target")
    args = ap.parse_args()

    mat = MATERIALS[args.material]
    site = SITES[args.site]
    TARGET_E = site["E_MPa"]                              # MPa
    TARGET_T_MIN, TARGET_T_MAX = site["T_min_days"], site["T_max_days"]
    E0, YS0 = mat["E_MPa"], mat["yield_MPa"]
    DEG_SCALE = mat["deg_scale_days"]

    # ---- "FEM" synthetic physics (placeholder for real simulation) ----
    def solid_frac(matx):
        return float(np.mean(1.0 - matx))

    def fem(matx):
        e = solid_frac(matx)
        E = E0 * e ** 2.0
        YS = YS0 * e ** 1.5
        T = DEG_SCALE * e ** 1.2
        return np.array([E, YS, T])

    # ---- surrogate: ridge regression on 27 porosity features ----
    def ridge_fit(X, Yt, lam=1.0):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        yc = Yt - Yt.mean()
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ yc)
        return w, mu, sd, float(Yt.mean())

    def predict(X, m):
        w, mu, sd, ym = m
        return ((X - mu) / sd) @ w + ym

    # ---- score: closeness to target E, T in window, high YS ----
    def score(pred):
        E, YS, T = pred
        dE = abs(E - TARGET_E) / TARGET_E
        dT = 0.0 if TARGET_T_MIN <= T <= TARGET_T_MAX else min(abs(T - TARGET_T_MIN), abs(T - TARGET_T_MAX)) / TARGET_T_MIN
        return 1.0 - 0.5 * dE - 0.3 * dT + 0.2 * (YS / 60.0)

    def predict_vec(X, models):
        return np.stack([predict(X, models[k]) for k in ["E", "YS", "T"]], axis=1)

    # ---- step 1: generate 100 initial structures + FEM labels ----
    n_init = 100
    X_init = rng.uniform(0.3, 0.8, size=(n_init, 27))
    Y_init = np.array([fem(m.reshape(3, 3, 3)) for m in X_init])

    # ---- step 2: surrogate training ----
    models = {k: ridge_fit(X_init, Y_init[:, j]) for j, k in enumerate(["E", "YS", "T"])}

    # ---- step 3+4: MCTS sampling (tree search) ----
    def mutate(x, rng_):
        y = x.copy()
        i = rng_.integers(0, 27)
        y[i] = np.clip(y[i] + rng_.choice([-0.1, 0.1]), 0.2, 0.9)
        return y

    def mcts(root, iters=100, rng_=None):
        rng_ = rng_ or rng
        cur = root.copy()
        visited = [cur.copy()]
        best = (score(predict_vec(cur[None, :], models)[0]), cur.copy())
        for _ in range(iters):
            cand = mutate(cur, rng_)
            pred = predict_vec(cand[None, :], models)[0]
            s = score(pred)
            visited.append(cand.copy())
            if s > best[0]:
                best = (s, cand.copy())
                cur = cand.copy()
            else:
                if rng_.uniform() < 0.2:
                    cur = cand.copy()
        return best, visited

    roots = np.argsort([score(predict_vec(X_init[i:i+1], models)[0]) for i in range(n_init)])[-5:]
    candidates = []
    for r in roots:
        (s, best_mat), visited = mcts(X_init[r].copy(), iters=100)
        candidates.append((s, best_mat))

    # ---- step 5+6: "FEM" verify top candidates + output best 5 ----
    print("MCTS design result (top 5, FEM-verified with synthetic physics)")
    print(f"  material={args.material} ({mat['name']}), site={args.site}")
    print(f"  target E={TARGET_E} MPa, T in [{TARGET_T_MIN},{TARGET_T_MAX}] days, maximize YS")
    candidates.sort(key=lambda t: -t[0])
    for s, m in candidates[:5]:
        p = fem(m.reshape(3, 3, 3))
        print(f"  score={s:.3f}  E={p[0]:7.1f} MPa  YS={p[1]:6.1f} MPa  T_deg={p[2]:6.1f} days")
        print(f"      porosity matrix mean p={1-solid_frac(m.reshape(3,3,3)):.3f}")

    print("\nOK. Production: replace synthetic FEM with Abaqus simulation, surrogate with CNN "
          "(patent: 3x3x3 porosity voxelized to 60x60x60, conv 16/8/4 @3x3x3, FC 64/32), "
          "add t-SNE diversity selection + active-learning augmentation loop.")

if __name__ == "__main__":
    main()
