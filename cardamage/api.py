"""HTTP wrapper: uvicorn cardamage.api:app   (CARDAMAGE_MODE=demo|prod)

POST /analyze  (multipart, field `files`, one or more images) -> AnalysisResult JSON.
GET  /schema   JSON Schema of that payload (the contract the JEV side codes against).
GET  /healthz

Demo mode only (none of this exists in prod): `/` demo website, `/api/info`, `/api/samples`,
`/samples/*`, and `/analyze` additionally writes annotated images served under `/demo/...`.
In prod mode no image is ever written and the renderer is never imported.
"""
from __future__ import annotations

import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .schema import SCHEMA_VERSION, AnalysisResult

MAX_FILES = 12
MAX_BYTES = 25 * 1024 * 1024  # per image

_ROOT = Path(__file__).resolve().parents[1]
_WEB = Path(__file__).resolve().parent / "web"
_SAMPLES = _ROOT / "samples"
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_state: dict = {}
_DEMO_DIR = Path(tempfile.gettempdir()) / "cardamage_demo"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .pipeline import DamageAnalyzer
    _state["settings"] = s = Settings.from_env()
    _state["analyzer"] = DamageAnalyzer(s)  # load weights once per process
    yield
    _state.clear()


app = FastAPI(title="car-damage", lifespan=lifespan)
_DEMO = Settings.from_env().mode == "demo"


@app.get("/healthz")
def healthz():
    return {"ok": "analyzer" in _state, "mode": _state["settings"].mode if _state else None}


@app.get("/schema")
def schema():
    return {"schema_version": SCHEMA_VERSION, "schema": AnalysisResult.model_json_schema()}


@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"at most {MAX_FILES} images per request")
    imgs, ids = [], []
    for n, f in enumerate(files):
        data = await f.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename!r} exceeds {MAX_BYTES // 2**20} MB")
        if not data:
            raise HTTPException(422, f"{f.filename!r} is empty")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(422, f"{f.filename!r} is not a decodable image")
        imgs.append(img)
        ids.append(Path(f.filename or f"image_{n}").stem)
    s: Settings = _state["settings"]
    result, arrs = await run_in_threadpool(_state["analyzer"].analyze, imgs, ids)
    body = result.model_dump(mode="json")
    if s.mode == "demo":
        from .render import annotated_names, write_demo_outputs  # never imported in prod
        run = uuid.uuid4().hex[:8]
        write_demo_outputs(result, arrs, _DEMO_DIR / run)
        names = annotated_names([r.image_id for r in result.images])
        body["demo"] = {"report": f"/demo/{run}/report.html",
                        "annotated": [f"/demo/{run}/annotated/{quote(n)}" for n in names]}
    return body


if _DEMO:
    _DEMO_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/demo", StaticFiles(directory=_DEMO_DIR), name="demo")
    if _SAMPLES.is_dir():
        app.mount("/samples", StaticFiles(directory=_SAMPLES), name="samples")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_WEB / "index.html")

    @app.get("/api/info")
    def info():
        s: Settings = _state["settings"]
        a = _state["analyzer"]
        return {"mode": s.mode, "schema_version": SCHEMA_VERSION, "device": a.device,
                "damage_model": a.damage.info.model_dump(mode="json"),
                "licence": a.damage_entry.license,
                "vehicle_filter": bool(a.vehicle),
                "max_files": MAX_FILES}

    @app.get("/api/samples")
    def samples():
        files = sorted(p.name for p in _SAMPLES.iterdir() if p.suffix.lower() in _IMG_EXT) if _SAMPLES.is_dir() else []
        return {"samples": [f"/samples/{n}" for n in files]}
