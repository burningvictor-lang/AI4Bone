"""AI4Bone - Web Demo backend (FastAPI).
Run: uvicorn web.main:app --host 127.0.0.1 --port 8765
"""
import os, sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from mcts_structure_design import design, MATERIALS, SITES, LPBF  # noqa: E402

app = FastAPI(title="AI4Bone Design API", version="0.1.0")


@app.get("/api/meta")
def meta():
    return {
        "alloys": {k: {"name": v["name"], "family": v["family"], "E_MPa": v["E_MPa"],
                       "yield_MPa": v["yield_MPa"], "corrosion_mm_per_year": v["corrosion_mm_per_year"]}
                   for k, v in MATERIALS.items()},
        "sites": {k: v for k, v in SITES.items()},
        "lpbf": LPBF,
    }


@app.post("/api/design")
def run_design(payload: dict):
    material = payload.get("material", "we43_mg")
    site = payload.get("site", "cancellous")
    seed = int(payload.get("seed", 42))
    if material not in MATERIALS:
        return {"error": f"unknown material: {material}", "materials": list(MATERIALS)}
    if site not in SITES:
        return {"error": f"unknown site: {site}", "sites": list(SITES)}
    return design(material=material, site=site, seed=seed)


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
