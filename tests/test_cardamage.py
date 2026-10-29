import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "s25.jpg"
sys.path.insert(0, str(ROOT))

from cardamage.geometry import overlap_fraction, poly_area, simplify  # noqa: E402
from cardamage.schema import AnalysisResult  # noqa: E402


def test_poly_area_unit_square():
    assert poly_area([(0, 0), (1, 0), (1, 1), (0, 1)]) == pytest.approx(1.0)


def test_overlap_fraction():
    inner = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    outer = [(0.0, 0.0), (0.2, 0.0), (0.2, 1.0), (0.0, 1.0)]
    assert overlap_fraction(inner, outer) == pytest.approx(0.5, abs=0.05)
    assert overlap_fraction(inner, inner) == pytest.approx(1.0, abs=0.01)


def test_simplify_keeps_shape():
    circle = [(0.5 + 0.2 * __import__("math").cos(t / 50 * 6.2832), 0.5 + 0.2 * __import__("math").sin(t / 50 * 6.2832))
              for t in range(50)]
    s = simplify(circle, 0.002)
    assert 3 <= len(s) <= len(circle)
    assert poly_area(s) == pytest.approx(poly_area(circle), rel=0.05)


def _run(mode, out):
    env = {**os.environ, "YOLO_OFFLINE": "1"}
    subprocess.run([sys.executable, "-m", "cardamage", "run", "--mode", mode, "--input", str(SAMPLE),
                    "--out", str(out)], cwd=ROOT, check=True, env=env, capture_output=True)


@pytest.mark.skipif(not (ROOT / "models" / "abdullahg7_v8s_seg").exists(), reason="weights not converted")
def test_prod_emits_json_only_demo_emits_both(tmp_path):
    _run("prod", tmp_path / "prod")
    _run("demo", tmp_path / "demo")
    assert sorted(p.name for p in (tmp_path / "prod").iterdir()) == ["result.json"]
    demo = sorted(p.name for p in (tmp_path / "demo").iterdir())
    assert demo == ["annotated", "report.html", "result.json"]
    assert (tmp_path / "demo" / "annotated" / "s25.jpg").stat().st_size > 10_000
    # same JEV payload either way (modulo mode + timing)
    a = json.loads((tmp_path / "prod" / "result.json").read_text())
    b = json.loads((tmp_path / "demo" / "result.json").read_text())
    assert a["mode"] == "prod" and b["mode"] == "demo"
    for r in (a, b):
        AnalysisResult.model_validate(r)
        for im in r["images"]:
            im.pop("inference_ms")
        r.pop("mode")
    assert a == b
    assert a["images"][0]["damages"], "expected damage on the close-up sample"
