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


def _mechanics_summary(candidate, patient, site, target_e, target_t_mid):
    """Build a transparent mechanics/FEM summary for the top candidate."""
    site_loads = {"cancellous": 350.0, "cortical": 1200.0, "periarticular": 800.0}
    load_n = site_loads.get(site, 500.0)
    diameter = patient["canal_diameter_mm"]
    length = patient["defect_length_mm"]
    porosity = candidate["porosity_mean"]
    solid_fraction = max(1.0 - porosity, 0.08)
    area_mm2 = math.pi * (diameter / 2.0) ** 2
    effective_area = area_mm2 * solid_fraction

    anisotropy_ratio = round(0.55 + 0.35 * patient["cortical_ratio"], 3)
    stress_concentration = 1.8 + 0.7 * (1.0 - patient["cortical_ratio"])
    max_stress = load_n / max(effective_area, 1.0) * stress_concentration
    displacement = load_n * length / max(candidate["E_MPa"] * effective_area, 1.0)
    safety_factor = candidate["YS_MPa"] / max(max_stress, 0.1)
    stress_shielding = abs(candidate["E_MPa"] - target_e) / max(target_e, 1.0)

    rows, cols = 6, 9
    raw_map = []
    for row in range(rows):
        values = []
        for col in range(cols):
            hotspot = math.exp(-(((row - 2.3) / 1.7) ** 2 + ((col - 5.8) / 2.2) ** 2))
            interface = 0.22 * math.exp(-((col - 1.0) / 1.4) ** 2)
            values.append(0.18 + 0.74 * hotspot + interface + 0.025 * row)
        raw_map.append(values)
    raw_max = max(max(row) for row in raw_map)
    stress_map = [
        [round(max_stress * value / raw_max, 2) for value in row]
        for row in raw_map
    ]

    target_days = target_t_mid
    time_days = [0, 30, 60, 90, 120, 180]
    implant_support = []
    tissue_support = []
    combined_support = []
    for day in time_days:
        implant = 1.35 * load_n * math.exp(-0.70 * day / max(candidate["T_deg_days"], 30.0))
        tissue = 1.10 * load_n / (
            1.0 + math.exp(-(day - 0.55 * target_days) / max(0.16 * target_days, 8.0))
        )
        implant_support.append(round(implant, 1))
        tissue_support.append(round(tissue, 1))
        combined_support.append(round(implant + tissue, 1))

    return {
        "model_basis": [
            "CT 灰度/体积分数用于描述局部骨量空间变化",
            "结构张量用于描述松质骨主方向与各向异性",
            "植入物采用近端—核心—远端连续梯度参数",
        ],
        "boundary_conditions": {
            "load_case": "轴向压缩 + 界面弯曲敏感性筛查",
            "load_N": load_n,
            "constraint": "远端截面固定，近端沿解剖轴施加载荷",
            "contact": "骨—植入物界面采用绑定接触演示；正式版进行摩擦敏感性分析",
        },
        "outputs": {
            "max_von_mises_MPa": round(max_stress, 2),
            "max_displacement_mm": round(displacement, 3),
            "safety_factor": round(safety_factor, 2),
            "stress_shielding_ratio": round(stress_shielding, 3),
            "anisotropy_ratio_transverse_to_axial": anisotropy_ratio,
        },
        "stress_map_MPa": stress_map,
        "time_series": {
            "days": time_days,
            "implant_support_N": implant_support,
            "tissue_support_N": tissue_support,
            "combined_support_N": combined_support,
            "required_support_N": [load_n for _ in time_days],
        },
        "acceptance": {
            "stress_below_yield": max_stress < candidate["YS_MPa"],
            "safety_factor_ge_1_5": safety_factor >= 1.5,
            "combined_support_maintained": min(combined_support) >= load_n,
        },
        "note": "当前云图与时间序列为合成力学演示；正式版需用患者网格、真实载荷边界和经验证材料本构替换。",
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

    mechanics = _mechanics_summary(
        output[0], patient, site, target_e,
        0.5 * (target_t_min + target_t_max),
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
        "mechanics_model": mechanics,
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
