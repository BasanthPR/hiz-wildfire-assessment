"""
run_ablation_ollama.py
HIZ-VLM Pipeline — Ablation Study via Ollama (no Graph-RAG)

Re-runs Qwen2.5-VL on a 30% random sample of tiles using the PLAIN prompt
(no regulatory context injection), to measure the impact of Graph-RAG.

Results saved to results/qwen25vl_ablation/.

Usage:
    # Start Ollama first:
    OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 ollama serve &
    python3 ~/hiz_pipeline/run_ablation_ollama.py [--workers 4]
"""

import argparse
import base64
import csv
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

PIPELINE_DIR    = Path.home() / "hiz_pipeline"
TILES_DIR       = PIPELINE_DIR / "tiles"
_FILTERED       = TILES_DIR / "tile_manifest_filtered.csv"
_FULL           = TILES_DIR / "tile_manifest.csv"
MANIFEST_CSV    = _FILTERED if _FILTERED.exists() else _FULL
RESULTS_DIR     = PIPELINE_DIR / "results" / "qwen25vl_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "qwen2.5vl:7b"
MAX_TOKENS      = 512
TEMPERATURE     = 0.1
REQUEST_TIMEOUT = 180
SAMPLE_FRACTION = 0.30
RANDOM_SEED     = 42

SYSTEM_MSG = (
    "You are a wildfire defensible space compliance assessor evaluating "
    "drone imagery of California residential properties. "
    "Return valid JSON only — no markdown, no preamble."
)

# Plain prompt — no regulatory context, no few-shot example
PLAIN_USER_TEMPLATE = (
    "Inspect this aerial drone tile for wildfire safety compliance violations.\n"
    "Look for: vehicles, trash cans, propane tanks.\n"
    "Zone: {zone} | Parcel: {parcel_id}\n\n"
    "Return JSON:\n"
    "  detections: [{{object_class, bounding_box_pixels, zone, "
    "compliance_status (COMPLIANT/VIOLATION/UNCERTAIN), confidence (LOW/MEDIUM/HIGH), "
    "severity (LOW/MEDIUM/HIGH/CRITICAL), cot_trace}}]\n"
    "  overall_parcel_risk: LOW/MEDIUM/HIGH/CRITICAL\n"
    "  aerial_limitations: [list of what you could not determine from aerial view]"
)


def build_messages(tile_path: str, zone: str, parcel_id: str) -> list[dict]:
    user_text = PLAIN_USER_TEMPLATE.format(zone=zone, parcel_id=parcel_id)
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
    parcel_id = row.get("parcel_id", "unknown")

    t0 = time.time()
    try:
        messages = build_messages(tile_path, zone, parcel_id)
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
                    "parse_failed": True, "ablation": True,
                    "raw_response": raw_text[:500], "elapsed_s": round(elapsed, 2),
                    "model": OLLAMA_MODEL, "backend": "ollama"}

        parsed["tile_path"]    = tile_path
        parsed["parcel_id"]    = parcel_id
        parsed["zone"]         = zone
        parsed["parse_failed"] = False
        parsed["ablation"]     = True
        parsed["elapsed_s"]    = round(elapsed, 2)
        parsed["model"]        = OLLAMA_MODEL
        parsed["backend"]      = "ollama"
        return parsed

    except Exception as exc:
        return {"tile_path": tile_path, "parcel_id": parcel_id, "zone": zone,
                "parse_failed": True, "ablation": True, "error": str(exc),
                "elapsed_s": round(time.time() - t0, 2), "model": OLLAMA_MODEL, "backend": "ollama"}


def result_path(row: dict) -> Path:
    return RESULTS_DIR / f"{Path(row['tile_path']).stem}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    check_ollama()

    all_tiles = []
    with open(MANIFEST_CSV) as f:
        for row in csv.DictReader(f):
            all_tiles.append(row)

    random.seed(RANDOM_SEED)
    n = max(1, int(len(all_tiles) * SAMPLE_FRACTION))
    sample = random.sample(all_tiles, n)
    print(f"Ablation sample: {n} tiles ({SAMPLE_FRACTION*100:.0f}% of {len(all_tiles)})")

    done    = {result_path(r) for r in sample if result_path(r).exists()}
    pending = [r for r in sample if not result_path(r).exists()]
    print(f"  Already done: {len(done)} | Pending: {len(pending)}")

    if not pending:
        print("All ablation tiles processed.")
        return

    est_min = len(pending) * 25 / args.workers / 60
    print(f"  Est. time: {est_min:.0f} min with {args.workers} workers")
    print(f"  Results dir: {RESULTS_DIR}")
    print("  Starting...")

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(infer_tile, row): row for row in pending}
        with tqdm(total=len(pending), desc="Ablation (no Graph-RAG)") as bar:
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
    print("Next: python3 ~/hiz_pipeline/evaluate.py")


if __name__ == "__main__":
    main()
