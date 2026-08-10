# -*- coding: utf-8 -*-
"""
GOAI baseline - Module A-1: alloy-agnostic design targets + Mg alloy property prediction.
Only depends on numpy/stdlib -> runs anywhere, fully reproducible (SEED=42).

Two layers:
  1) Alloy scenario layer: data/materials_lib.json holds reference material parameters
     for WE43 Mg and Zn-based alloys. The same structure-design pipeline can be applied
     to each (Mg or Zn) by swapping material parameters -- no new dataset needed here.
  2) Mg composition-property layer: ridge models on Mg composition data
     (Zenodo record 17672235, DatasetMg_imputed.csv, 600 rows). Schema-aware:
       * real schema (28 cols): 20 composition + 5 process + YS_MPa/UTS_MPa/El_%
       * demo schema (mg_alloy_sample.csv): Zn/Ca/Sr/Mn -> UTS/YieldStrength/Elongation/DegradationRate
     Cleaning: feature blanks -> column median; rows with missing targets are dropped.

Run:
  python mg_alloy_baseline.py                      # default alloy: we43_mg
  python mg_alloy_baseline.py --alloy zn_1mg
  python mg_alloy_baseline.py --list-alloys
"""
import os, csv, json, random, argparse
import numpy as np
try:
    from sklearn.ensemble import RandomForestRegressor
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

SEED = 42
random.seed(SEED); np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
LIB = os.path.join(DATA, "materials_lib.json")

# ---------- 1) alloy scenario layer ----------
with open(LIB, encoding="utf-8-sig") as f:
    _lib = json.load(f)
MATERIALS = _lib["materials"]
SITES = _lib["sites"]

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
DEMO = os.path.join(DATA, "mg_alloy_sample.csv")

PROC_COLS = ["Extr_Temp_K", "Extr_Speed_m/min", "Extr_Ratio", "Pre_Temp_K", "Pre_Time_h"]
TARGETS_REAL = ["YS_MPa", "UTS_MPa", "El_%"]
TARGETS_DEMO = ["UTS", "YieldStrength", "Elongation", "DegradationRate"]

def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")

def build_matrix(rows):
    """rows: list of dict; returns (X, Y dict, FEATURES, TARGETS, schema_name, meta)."""
    cols = list(rows[0].keys())
    if "DegradationRate" in cols:
        FEATURES = ["Zn", "Ca", "Sr", "Mn"]
        TARGETS = TARGETS_DEMO
        schema = "demo"
    else:
        FEATURES = [c for c in cols if c not in set(TARGETS_REAL)]
        TARGETS = TARGETS_REAL
        schema = "real"

    # numeric matrix with NaN for blanks
    raw = {k: np.array([to_float(r[k]) for r in rows], dtype=float) for k in FEATURES + TARGETS}
    # drop rows missing any target
    mask = np.ones(len(rows), dtype=bool)
    for t in TARGETS:
        mask &= ~np.isnan(raw[t])
    keep = np.where(mask)[0]
    # feature blanks -> column median (computed on kept rows)
    X = np.column_stack([raw[f][keep] for f in FEATURES])
    for j in range(X.shape[1]):
        med = np.nanmedian(X[:, j])
        X[np.isnan(X[:, j]), j] = med
    Y = {t: raw[t][keep] for t in TARGETS}
    meta = {"total": len(rows), "kept": int(mask.sum()), "dropped": int((~mask).sum())}
    return X, Y, FEATURES, TARGETS, schema, meta

def fit_model(Xtr, ytr, lam=1.0):
    if HAVE_SKLEARN:
        rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                   random_state=SEED, n_jobs=-1)
        rf.fit(Xtr, ytr)
        return {"kind": "rf", "rf": rf}
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0) + 1e-9
    Xs = (Xtr - mu) / sd
    yc = ytr - ytr.mean()
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ yc)
    return {"kind": "ridge", "w": w, "mu": mu, "sd": sd, "ymean": float(ytr.mean())}

def predict(X, m):
    if m["kind"] == "rf":
        return m["rf"].predict(X)
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["ymean"]

def metrics(y, pred):
    mae = float(np.mean(np.abs(pred - y)))
    r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    return mae, r2

def run_mg_pipeline():
    path = CSV if os.path.exists(CSV) else DEMO
    if not os.path.exists(CSV):
        print("NOTE: Zenodo DatasetMg_imputed.csv not found - using sample demo data")
    rows = load_csv(path)
    X, Y, FEATURES, TARGETS, schema, meta = build_matrix(rows)
    n = meta["kept"]
    model_kind = "RandomForest" if HAVE_SKLEARN else "ridge (no sklearn)"
    print(f"\n[Mg composition-property] schema={schema} | model={model_kind} | "
          f"total={meta['total']} | kept={n} | dropped={meta['dropped']} | features={len(FEATURES)}")

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

    metric = "UTS_MPa" if schema == "real" else "UTS"
    print(f"--- permutation importance ({metric}, full-data model) ---")
    m = full_models[metric]
    base = metrics(Y[metric], predict(X, m))[1]
    drops = []
    for j, name in enumerate(FEATURES):
        Xp = X.copy(); np.random.shuffle(Xp[:, j])
        r2p = metrics(Y[metric], predict(Xp, m))[1]
        drops.append((base - r2p, name))
    drops.sort(reverse=True)
    for d, name in drops[:8]:
        print(f"  {name}: R2_drop={d:.4f}")

    # ---- reverse design ----
    if schema == "real":
        constraint, op, val, best_col = "El_%", "ge", 10.0, "UTS_MPa"
        proc_mean = {c: float(np.nanmedian(X[:, FEATURES.index(c)])) for c in PROC_COLS}
        grid_cols = [c for c in ["Zn", "Ca", "Mn", "Sr"] if c in FEATURES]
        others = [c for c in FEATURES if c not in grid_cols and c not in PROC_COLS]
        print(f"--- reverse design demo: maximize UTS_MPa with El_% >= 10% (process params at median) ---")
    else:
        constraint, op, val, best_col = "DegradationRate", "le", 0.40, "UTS"
        grid_cols = ["Zn", "Ca", "Sr", "Mn"]
        others = []
        proc_mean = {}
        print(f"--- reverse design demo: maximize UTS with DegradationRate <= 0.40 mm/y ---")

    pct5 = np.percentile(X, 5, axis=0); pct95 = np.percentile(X, 95, axis=0)
    grids = []
    for gc in grid_cols:
        lo = float(pct5[FEATURES.index(gc)]); hi = float(pct95[FEATURES.index(gc)])
        step = 0.5 if gc == "Zn" else 0.2
        grids.append((gc, np.arange(lo, hi + 1e-9, step)))
    best = None
    for combo in np.ndindex(*(len(g[1]) for g in grids)):
        x = np.zeros(len(FEATURES))
        for j, (gc, vals) in enumerate(grids):
            x[FEATURES.index(gc)] = vals[combo[j]]
        for oc in others:
            x[FEATURES.index(oc)] = 0.0
        for pc, v in proc_mean.items():
            x[FEATURES.index(pc)] = v
        p = {t: float(predict(x[None, :], full_models[t])[0]) for t in TARGETS}
        ok = p[constraint] >= val if op == "ge" else p[constraint] <= val
        if ok and (best is None or p[best_col] > best[0][best_col]):
            best = (p, x)
    if best:
        p, x = best
        max_obs = float(np.max(Y[best_col]))
        if p[best_col] > 1.5 * max_obs:
            print(f"  [warn] predicted {best_col}={p[best_col]:.1f} MPa exceeds observed max "
                  f"{max_obs:.1f} MPa -> linear baseline extrapolation, reference only (use GBM/RF in production)")
        comp = {gc: float(x[FEATURES.index(gc)]) for gc in grid_cols}
        print(f"  recommended composition: " + ", ".join(f"{gc}={v:.1f}" for gc, v in comp.items()) + " (wt%)")
        if schema == "real":
            print(f"  predicted UTS={p['UTS_MPa']:.1f} MPa, YS={p['YS_MPa']:.1f} MPa, El={p['El_%']:.1f}%  "
                  f"[process: " + ", ".join(f"{pc}={proc_mean[pc]:.0f}" for pc in PROC_COLS) + "]")
        else:
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
          "alloy-agnostic via materials_lib.json + structure_design.py --material.")
    print("Next: add SHAP + NSGA-II.")

if __name__ == "__main__":
    main()
