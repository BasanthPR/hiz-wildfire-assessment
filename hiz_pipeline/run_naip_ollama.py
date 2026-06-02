"""
run_naip_ollama.py
HIZ-VLM Pipeline — NAIP Public Imagery Inference via Ollama

Runs Qwen2.5-VL on 78 NAIP tiles (60 cm GSD) using the Ollama server.
Results saved to results/naip_qwen25vl/.

Usage:
    # Start Ollama first:
    OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 ollama serve &
    python3 ~/hiz_pipeline/run_naip_ollama.py [--workers 4]
"""

import argparse
import base64
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

PIPELINE_DIR  = Path.home() / "hiz_pipeline"
TILES_DIR     = PIPELINE_DIR / "tiles_naip"
MANIFEST_CSV  = TILES_DIR / "tile_manifest_naip.csv"
RESULTS_DIR   = PIPELINE_DIR / "results" / "naip_qwen25vl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "qwen2.5vl:7b"
MAX_TOKENS    = 512
TEMPERATURE   = 0.1
REQUEST_TIMEOUT = 180

SYSTEM_MSG = (
    "You are a wildfire defensible space compliance assessor evaluating "
    "aerial imagery of California residential properties against PRC 4291 "
    "and IBHS Wildfire Prepared Home standards. "
    "Return valid JSON only — no markdown, no preamble."
)

NAIP_PREAMBLE = (
    "NOTE: This image is from NAIP (National Agriculture Imagery Program) "
    "public aerial photography at approximately 60 cm ground sampling distance (GSD). "
    "Fine structural details visible in drone imagery may not be resolved. "
    "Focus detection on objects resolvable at this scale: vehicles (>2 m), "
    "large woodpiles (>1 m), storage sheds (>2 m), and dense dry vegetation. "
    "Apply the same regulatory standards but note reduced confidence for "
    "objects smaller than 2–3 pixels.\n\n"
)

COMPACT_EXAMPLE = """\
[EXAMPLE — Propane tank, Zone_0, VIOLATION]
PERCEIVE: Cylindrical silver object, ~70cm diameter.
LOCATE: ~4ft from structure wall → Zone_0.
RETRIEVE: PRC4291 §M: 10ft bare mineral soil + 10ft no flammable veg.
ASSESS: VIOLATION. Dry grass present within clearance zone.
{"object_class":"propane_tank","bounding_box_pixels":[45,210,73,238],\
"zone":"Zone_0","compliance_status":"VIOLATION","confidence":"HIGH",\
"severity":"CRITICAL","cot_trace":"PERCEIVE→propane tank. \
LOCATE→Zone_0. RETRIEVE→PRC4291§M clearance. ASSESS→VIOLATION."}
"""

ZONE_REG: dict[str, str] = {
    "Zone_0": (
        "Zone_0 (0–5ft from structure) — HIGHEST RISK:\n"
        "• No combustible objects (vehicles, propane tanks, wood piles, trash cans)\n"
        "• PRC 4291 §B: bare mineral soil or non-combustible ground cover only"
    ),
    "Zone_1": (
        "Zone_1 (5–30ft from structure) — HIGH RISK:\n"
        "• Vehicles: no parking within 5ft of structure\n"
        "• Propane tanks: 10ft clearance from structure minimum"
    ),
    "Zone_2": (
        "Zone_2 (30–100ft from structure) — MODERATE RISK:\n"
        "• Vehicles: no combustible material stored underneath\n"
        "• Spacing: CAL FIRE LE-100a §6 — trees ≥10ft crown separation"
    ),
}


def build_messages(tile_path: str, zone: str, parcel_id: str,
                   site: str, gsd_cm: float) -> list[dict]:
    reg = ZONE_REG.get(zone, ZONE_REG["Zone_2"])
    user_text = (
        f"{NAIP_PREAMBLE}"
        f"{COMPACT_EXAMPLE}\n"
        f"[REGULATORY CONTEXT — {zone}]\n{reg}\n\n"
        f"Inspect this {zone} NAIP tile for: vehicles/cars/trucks, "
        f"trash cans/garbage bins, propane tanks.\n"
        f"Parcel: {parcel_id} | Site: {site} | GSD: {gsd_cm:.1f} cm/px\n\n"
        f"Return JSON:\n"
        f"  detections: [{{object_class, bounding_box_pixels, zone, "
        f"compliance_status, confidence, severity, cot_trace}}]\n"
        f"  overall_parcel_risk: LOW/MEDIUM/HIGH/CRITICAL\n"
        f"  aerial_limitations: [list]"
    )
    with open(tile_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode("ascii")
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_text, "images": [img_b64]},
    ]


def parse_json(raw: str):
    raw = raw.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    s = raw.find("{")
    if s == -1:
        return None
    for e in range(len(raw), s, -1):
        try:
            return json.loads(raw[s:e])
        except json.JSONDecodeError:
            continue
    return None


def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"  Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
        print(f"  Ollama OK — model {OLLAMA_MODEL} ready")
    except requests.ConnectionError:
        print(
            "  ERROR: Ollama not running.\n"
            "  Start with: OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 ollama serve &"
        )
        sys.exit(1)


def infer_tile(row: dict) -> dict:
    tile_path = row["tile_path"]
    zone      = row["zone"]
    parcel_id = row.get("parcel_id", Path(tile_path).stem)
    site      = row.get("site", "naip")
    gsd_cm    = float(row.get("gsd_cm", 60.0))

    t0 = time.time()
    try:
        messages = build_messages(tile_path, zone, parcel_id, site, gsd_cm)
        payload  = {
            "model":   OLLAMA_MODEL,
            "messages": messages,
            "stream":  False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "num_ctx": 2048},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw_text = resp.json()["message"]["content"]
        elapsed  = time.time() - t0

        parsed = parse_json(raw_text)
        if parsed is None:
            return {"tile_path": tile_path, "parcel_id": parcel_id, "zone": zone,
                    "parse_failed": True, "raw_response": raw_text[:500],
                    "elapsed_s": round(elapsed, 2), "model": OLLAMA_MODEL, "backend": "ollama"}

        parsed["tile_path"]    = tile_path
        parsed["parcel_id"]    = parcel_id
        parsed["zone"]         = zone
        parsed["parse_failed"] = False
        parsed["elapsed_s"]    = round(elapsed, 2)
        parsed["model"]        = OLLAMA_MODEL
        parsed["backend"]      = "ollama"
        return parsed

    except Exception as exc:
        return {"tile_path": tile_path, "parcel_id": parcel_id, "zone": zone,
                "parse_failed": True, "error": str(exc),
                "elapsed_s": round(time.time() - t0, 2), "model": OLLAMA_MODEL, "backend": "ollama"}


def result_path(row: dict) -> Path:
    return RESULTS_DIR / f"{Path(row['tile_path']).stem}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    check_ollama()

    if not MANIFEST_CSV.exists():
        sys.exit(f"NAIP manifest not found: {MANIFEST_CSV}")

    all_tiles = []
    with open(MANIFEST_CSV) as f:
        for row in csv.DictReader(f):
            all_tiles.append(row)

    done = {result_path(r) for r in all_tiles if result_path(r).exists()}
    pending = [r for r in all_tiles if not result_path(r).exists()]

    print(f"NAIP tiles: {len(all_tiles)} total | {len(done)} done | {len(pending)} pending")
    print(f"Results dir: {RESULTS_DIR}")

    if not pending:
        print("All NAIP tiles processed.")
        return

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(infer_tile, row): row for row in pending}
        with tqdm(total=len(pending), desc="NAIP Qwen2.5-VL") as bar:
            for fut in as_completed(futures):
                result = fut.result()
                out_path = result_path(futures[fut])
                out_path.write_text(json.dumps(result, indent=2))
                if result.get("parse_failed"):
                    fail += 1
                else:
                    ok += 1
                bar.set_postfix(ok=ok, fail=fail)
                bar.update(1)

    print(f"\nDone. ok={ok}, fail={fail}")
    print(f"Results: {RESULTS_DIR}")
    print("Next: python3 ~/hiz_pipeline/aggregate_naip_results.py && python3 ~/hiz_pipeline/evaluate.py")


if __name__ == "__main__":
    main()
