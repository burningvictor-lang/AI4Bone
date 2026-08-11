"""AI4Bone web demo backend."""
import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from structure_design import design, LPBF, MATERIALS, SITES  # noqa: E402

app = FastAPI(title="AI4Bone Design API", version="0.2.0")


@app.get("/api/meta")
def meta():
    return {
        "alloys": {
            key: {
                "name": value["name"],
                "family": value["family"],
                "E_MPa": value["E_MPa"],
                "yield_MPa": value["yield_MPa"],
                "corrosion_mm_per_year": value["corrosion_mm_per_year"],
            }
            for key, value in MATERIALS.items()
        },
        "sites": SITES,
        "lpbf": LPBF,
    }


@app.post("/api/design")
def run_design(payload: dict):
    material = payload.get("material", "we43_mg")
    site = payload.get("site", "cancellous")
    if material not in MATERIALS:
        return {"error": f"unknown material: {material}", "materials": list(MATERIALS)}
    if site not in SITES:
        return {"error": f"unknown site: {site}", "sites": list(SITES)}

    try:
        return design(
            material=material,
            site=site,
            seed=int(payload.get("seed", 42)),
            defect_length_mm=float(payload.get("defect_length_mm", 40)),
            canal_diameter_mm=float(payload.get("canal_diameter_mm", 18)),
            cortical_ratio=float(payload.get("cortical_ratio", 0.35)),
        )
    except (TypeError, ValueError) as exc:
        return {"error": f"invalid patient geometry: {exc}"}


app.mount(
    "/",
    StaticFiles(directory=os.path.join(HERE, "static"), html=True),
    name="static",
)
