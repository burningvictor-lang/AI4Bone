# -*- coding: utf-8 -*-
"""Patient-specific biodegradable metal implant structure design demo.

Workflow:
  patient geometry -> defect envelope -> zonal porosity targets ->
  surrogate-guided tree search -> synthetic FEM verification -> LPBF plan.

The physics functions are transparent synthetic stand-ins for finite-element
simulation. Production use should replace them with validated FEM results and
a trained 3D surrogate model.
"""
import argparse
import json
import math
import os

import numpy as np

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

with open(os.path.join(DATA, "materials_lib.json"), encoding="utf-8-sig") as f:
    _lib = json.load(f)

MATERIALS = _lib["materials"]
SITES = _lib["sites"]
LPBF = {
    "power_W": [40, 100],
    "speed_mm_per_s": [400, 1200],
    "note": "筛选性工艺窗口；正式打印前需按设备、粉末与试样完成参数标定。",
}


def _clamp(value, low, high):
    return max(low, min(high, value))


def _patient_plan(defect_length_mm, canal_diameter_mm, cortical_ratio, ideal_porosity):
    defect_length_mm = _clamp(float(defect_length_mm), 10.0, 120.0)
    canal_diameter_mm = _clamp(float(canal_diameter_mm), 6.0, 40.0)
    cortical_ratio = _clamp(float(cortical_ratio), 0.10, 0.80)

    interface_p = _clamp(ideal_porosity - 0.08 * cortical_ratio, 0.30, 0.78)
    core_p = _clamp(ideal_porosity + 0.08 * (1.0 - cortical_ratio), 0.42, 0.88)
    transition_p = (interface_p + core_p) / 2.0
    envelope_volume = math.pi * (canal_diameter_mm / 2.0) ** 2 * defect_length_mm

    zones = [
        {
            "name": "近端界面区",
            "length_fraction": 0.20,
            "porosity_target": round(interface_p, 3),
            "pore_size_um": [420, 650],
            "strut_size_um": [350, 520],
            "design_role": "提高界面承载与初始稳定性",
        },
        {
            "name": "缺损核心区",
            "length_fraction": 0.60,
            "porosity_target": round(core_p, 3),
            "pore_size_um": [700, 1100],
            "strut_size_um": [260, 420],
            "design_role": "为骨长入、传质与降解产物扩散保留空间",
        },
        {
            "name": "远端界面区",
            "length_fraction": 0.20,
            "porosity_target": round(transition_p, 3),
            "pore_size_um": [480, 720],
            "strut_size_um": [320, 500],
            "design_role": "平滑过渡刚度，降低局部应力集中",
        },
    ]
    return {
        "input_mode": "demo_parameters",
        "defect_length_mm": round(defect_length_mm, 1),
        "canal_diameter_mm": round(canal_diameter_mm, 1),
        "cortical_ratio": round(cortical_ratio, 2),
        "estimated_envelope_volume_mm3": round(envelope_volume, 1),
        "zones": zones,
        "strategy": "单一合金体系内进行孔隙率、孔径和杆径的空间梯度设计；多材料打印仅作为后续验证方向。",
    }


def design(
    material="we43_mg",
    site="cancellous",
    seed=SEED,
    defect_length_mm=40.0,
    canal_diameter_mm=18.0,
    cortical_ratio=0.35,
    n_init=120,
    search_iters=120,
    top_k=5,
):
    """Return patient-specific zonal candidates and a printing plan."""
    rng = np.random.default_rng(seed)
    mat = MATERIALS[material]
    site_cfg = SITES[site]

    target_e = float(site_cfg["E_MPa"])
    target_t_min = float(site_cfg["T_min_days"])
    target_t_max = float(site_cfg["T_max_days"])
    e0 = float(mat["E_MPa"])
    ys0 = float(mat["yield_MPa"])
    deg_scale = float(mat["deg_scale_days"])

    ideal_porosity = _clamp(1.0 - math.sqrt(target_e / e0), 0.35, 0.85)
    patient = _patient_plan(
        defect_length_mm, canal_diameter_mm, cortical_ratio, ideal_porosity
    )
    layer_targets = np.array(
        [zone["porosity_target"] for zone in patient["zones"]], dtype=float
    )
    target_vector = np.repeat(layer_targets, 9)

    def solid_frac(matrix):
        return float(np.mean(1.0 - matrix))

    def synthetic_fem(matrix):
        sf = solid_frac(matrix)
        layer_means = matrix.mean(axis=(1, 2))
        transition = float(np.mean(np.abs(np.diff(layer_means))))
        e_mod = e0 * sf**2.0 * (1.0 - 0.08 * transition)
        yield_strength = ys0 * sf**1.5 * (1.0 - 0.12 * transition)
        degradation_days = deg_scale * sf**0.825
        return np.array([e_mod, yield_strength, degradation_days])

    def ridge_fit(x, y, lam=1.0):
        mu, sd = x.mean(0), x.std(0) + 1e-9
        xs = (x - mu) / sd
        yc = y - y.mean()
        weights = np.linalg.solve(
            xs.T @ xs + lam * np.eye(xs.shape[1]), xs.T @ yc
        )
        return weights, mu, sd, float(y.mean())

    def predict(x, model):
        weights, mu, sd, y_mean = model
        return ((x - mu) / sd) @ weights + y_mean

    def predict_all(x, models):
        return np.stack(
            [predict(x, models[key]) for key in ("E", "YS", "T")], axis=1
        )

    def score(prediction, vector):
        e_mod, yield_strength, degradation_days = prediction
        e_error = abs(e_mod - target_e) / target_e
        if target_t_min <= degradation_days <= target_t_max:
            t_error = 0.0
        else:
            t_error = min(
                abs(degradation_days - target_t_min),
                abs(degradation_days - target_t_max),
            ) / target_t_min
        zone_error = float(np.mean(np.abs(vector - target_vector)))
        return (
            1.0
            - 0.45 * e_error
            - 0.25 * t_error
            - 0.15 * zone_error
            + 0.15 * (yield_strength / max(ys0, 1.0))
        )

    x_init = np.clip(
        target_vector
        + rng.normal(0.0, 0.10, size=(n_init, 27))
        + rng.uniform(-0.04, 0.04, size=(n_init, 27)),
        0.20,
        0.90,
    )
    y_init = np.array(
        [synthetic_fem(vector.reshape(3, 3, 3)) for vector in x_init]
    )
    models = {
        key: ridge_fit(x_init, y_init[:, idx])
        for idx, key in enumerate(("E", "YS", "T"))
    }

    def mutate(vector):
        candidate = vector.copy()
        idx = int(rng.integers(0, 27))
        candidate[idx] = np.clip(
            candidate[idx] + rng.choice([-0.08, -0.04, 0.04, 0.08]),
            0.20,
            0.90,
        )
        return candidate

    def tree_search(root):
        current = root.copy()
        best = (score(predict_all(current[None, :], models)[0], current), current)
        for _ in range(search_iters):
            candidate = mutate(current)
            prediction = predict_all(candidate[None, :], models)[0]
            candidate_score = score(prediction, candidate)
            if candidate_score > best[0]:
                best = (candidate_score, candidate.copy())
                current = candidate
            elif rng.uniform() < 0.18:
                current = candidate
        return best

    initial_scores = [
        score(predict_all(x_init[i : i + 1], models)[0], x_init[i])
        for i in range(n_init)
    ]
    roots = np.argsort(initial_scores)[-8:]
    candidates = [tree_search(x_init[idx]) for idx in roots]
    candidates.sort(key=lambda item: -item[0])

    output = []
    for candidate_score, vector in candidates[:top_k]:
        matrix = vector.reshape(3, 3, 3)
        prediction = synthetic_fem(matrix)
        output.append(
            {
                "score": round(float(candidate_score), 4),
                "E_MPa": round(float(prediction[0]), 1),
                "YS_MPa": round(float(prediction[1]), 1),
                "T_deg_days": round(float(prediction[2]), 1),
                "porosity_mean": round(float(matrix.mean()), 3),
                "layer_porosity": [
                    round(float(value), 3)
                    for value in matrix.mean(axis=(1, 2))
                ],
                "matrix_3x3x3": [
                    [
                        [round(float(matrix[i, j, k]), 3) for k in range(3)]
                        for j in range(3)
                    ]
                    for i in range(3)
                ],
            }
        )

    return {
        "material": material,
        "material_name": mat["name"],
        "site": site,
        "target": {
            "E_MPa": target_e,
            "T_min_days": target_t_min,
            "T_max_days": target_t_max,
            "site_note": site_cfg["note"],
        },
        "patient_geometry": patient,
        "imaging_pipeline": [
            "DICOM/CT 导入与质量控制",
            "骨组织及缺损区域分割",
            "缺损包络与皮质/松质骨分区",
            "空间梯度结构参数生成",
            "有限元与可制造性联合筛选",
        ],
        "lpbf": LPBF,
        "candidates": output,
        "note": "当前为合成物理演示；正式版需接入真实患者影像、有限元结果和打印验证数据。",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Patient-specific biodegradable implant structure design demo"
    )
    parser.add_argument("--material", default="we43_mg", choices=list(MATERIALS))
    parser.add_argument("--site", default="cancellous", choices=list(SITES))
    parser.add_argument("--defect-length", type=float, default=40.0)
    parser.add_argument("--canal-diameter", type=float, default=18.0)
    parser.add_argument("--cortical-ratio", type=float, default=0.35)
    args = parser.parse_args()

    result = design(
        material=args.material,
        site=args.site,
        defect_length_mm=args.defect_length,
        canal_diameter_mm=args.canal_diameter,
        cortical_ratio=args.cortical_ratio,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
