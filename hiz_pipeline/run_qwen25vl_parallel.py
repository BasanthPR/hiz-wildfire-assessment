"""
run_qwen25vl_parallel.py
HIZ-VLM Pipeline — Parallel Inference via Ollama HTTP API

Uses Ollama's continuous-batching server to run multiple tile inferences
concurrently on the same GPU, cutting the 14h single-threaded run to ~4-5h.

Architecture:
  - Ollama server handles GPU scheduling + KV-cache reuse (no model reload)
  - ThreadPoolExecutor sends N simultaneous HTTP requests (default N=4)
  - Shortened prompts (~700 tokens vs 1417): 1 example, zone-specific context
  - Resume-safe: skips tiles already in results/qwen25vl_parallel/

Usage:
    # Start Ollama server first (if not already running):
    OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 ollama serve &

    python3 ~/hiz_pipeline/run_qwen25vl_parallel.py [--workers 4] [--dry-run]

PRIVACY: LOCAL INFERENCE ONLY — all data stays on-device.
"""

import argparse
import base64
import csv
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

PIPELINE_DIR  = Path.home() / "hiz_pipeline"
TILES_DIR     = PIPELINE_DIR / "tiles"
RESULTS_DIR   = PIPELINE_DIR / "results" / "qwen25vl_parallel"
_FILTERED     = TILES_DIR / "tile_manifest_filtered.csv"
_FULL         = TILES_DIR / "tile_manifest.csv"
MANIFEST_CSV  = _FILTERED if _FILTERED.exists() else _FULL

OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "qwen2.5vl:7b"
MAX_TOKENS    = 512
TEMPERATURE   = 0.1
REQUEST_TIMEOUT = 180    # seconds per tile (generous for GPU queue wait)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PIPELINE_DIR))


# ── Prompt helpers ─────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are a wildfire defensible space compliance assessor evaluating "
    "drone imagery of California residential properties against PRC 4291 "
    "and IBHS Wildfire Prepared Home standards. "
    "Return valid JSON only — no markdown, no preamble."
)

# Single compact example instead of 3 (saves ~500 tokens)
COMPACT_EXAMPLE = """\
[EXAMPLE — Propane tank, Zone_0, VIOLATION]
PERCEIVE: Cylindrical silver object, ~70cm diameter.
LOCATE: ~4ft from structure wall → Zone_0.
RETRIEVE: PRC4291 §M: 10ft bare mineral soil + 10ft no flammable veg.
ASSESS: VIOLATION. Dry grass present within clearance zone.
{"object_class":"propane_tank","bounding_box_pixels":[45,210,73,238],\
"zone":"Zone_0","compliance_status":"VIOLATION","confidence":"HIGH",\
"severity":"CRITICAL","cot_trace":"PERCEIVE→propane tank ~70cm. \
LOCATE→Zone_0 ~4ft. RETRIEVE→PRC4291§M clearance. ASSESS→VIOLATION dry grass in zone."}
"""

# Zone-specific regulatory summaries (avoids injecting all-zone context)
ZONE_REG: dict[str, str] = {
    "Zone_0": (
        "Zone_0 (0–5ft from structure) — HIGHEST RISK:\n"
        "• No combustible objects (vehicles, propane tanks, wood piles, trash cans)\n"
        "• PRC 4291 §B: bare mineral soil or non-combustible ground cover only\n"
        "• Propane tanks: 10ft clearance + additional 10ft no flammable vegetation\n"
        "• Any violation here is CRITICAL priority."
    ),
    "Zone_1": (
        "Zone_1 (5–30ft from structure) — HIGH RISK:\n"
        "• Vehicles: no parking within 5ft of structure (IBHS WPH §7.3)\n"
        "• Trash cans: must have tight-fitting lid and be stored away from vents\n"
        "• Propane tanks: 10ft clearance from structure minimum\n"
        "• Wood piles: min 10ft from structure, elevated off ground (PRC 4291 §G)\n"
        "• Grass/vegetation: ≤4 inches height for Zone_1"
    ),
    "Zone_2": (
        "Zone_2 (30–100ft from structure) — MODERATE RISK:\n"
        "• Vehicles: no combustible material stored underneath\n"
        "• Trash cans: acceptable if lidded, not directly adjacent to structure\n"
        "• Wood piles: should not create continuous fuel ladder to structure\n"
        "• Spacing requirements: CAL FIRE LE-100a §6 — trees ≥10ft crown separation"
    ),
}


def build_ollama_messages(tile_path: str, zone: str, parcel_id: str,
                          site: str, gsd_cm: float) -> list[dict]:
    """
    Build chat messages list for Ollama API (image as base64).
    Compact prompt: ~700 tokens vs 1417 in the full version.
    """
    reg = ZONE_REG.get(zone, ZONE_REG["Zone_2"])

    user_text = (
        f"{COMPACT_EXAMPLE}\n"
        f"[REGULATORY CONTEXT — {zone}]\n{reg}\n\n"
        f"Inspect this {zone} drone tile for: vehicles/cars/trucks, "
        f"trash cans/garbage bins, propane tanks.\n"
        f"Parcel: {parcel_id} | Site: {site} | GSD: {gsd_cm:.1f} cm/px\n\n"
        f"Return JSON:\n"
        f"  detections: [{{object_class, bounding_box_pixels, zone, "
        f"compliance_status, confidence, severity, cot_trace}}]\n"
        f"  overall_parcel_risk: LOW/MEDIUM/HIGH/CRITICAL\n"
        f"  aerial_limitations: [list]"
    )

    # Read image → base64
    with open(tile_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode("ascii")

    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_text, "images": [img_b64]},
    ]


# ── JSON parsing ───────────────────────────────────────────────────────────────

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


# ── Ollama health check ────────────────────────────────────────────────────────

def check_ollama():
    """Verify Ollama server is running and model is loaded."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"  Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
        print(f"  Ollama OK — model {OLLAMA_MODEL} ready")
    except requests.ConnectionError:
        print(
            "  ERROR: Ollama server not running.\n"
            "  Start it with:\n"
            "    OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 ollama serve &\n"
            "  Then re-run this script."
        )
        sys.exit(1)


# ── Per-tile worker ────────────────────────────────────────────────────────────

def infer_tile(row: dict) -> dict:
    """
    Send one tile to Ollama and return a result dict.
    Runs in a worker thread — no shared mutable state.
    """
    tile_path = row["tile_path"]
    zone      = row["zone"]
    parcel_id = row.get("parcel_id", "unknown")
    site      = row.get("site", "unknown")
    gsd_cm    = float(row.get("gsd_cm", 2.5))

    t0 = time.time()
    try:
        messages = build_ollama_messages(tile_path, zone, parcel_id, site, gsd_cm)

        payload = {
            "model":   OLLAMA_MODEL,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature":  TEMPERATURE,
                "num_predict":  MAX_TOKENS,
                "num_ctx":      2048,
            },
        }

        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw_text = resp.json()["message"]["content"]
        elapsed  = time.time() - t0

        parsed = parse_json(raw_text)
        if parsed is None:
            return {
                "tile_path":    tile_path,
                "parcel_id":    parcel_id,
                "zone":         zone,
                "parse_failed": True,
                "raw_response": raw_text[:500],
                "elapsed_s":    round(elapsed, 2),
                "model":        OLLAMA_MODEL,
                "backend":      "ollama",
            }

        parsed["tile_path"]    = tile_path
        parsed["parcel_id"]    = parcel_id
        parsed["zone"]         = zone
        parsed["parse_failed"] = False
        parsed["elapsed_s"]    = round(elapsed, 2)
        parsed["model"]        = OLLAMA_MODEL
        parsed["backend"]      = "ollama"
        return parsed

    except Exception as exc:
        return {
            "tile_path":    tile_path,
            "parcel_id":    parcel_id,
            "zone":         zone,
            "parse_failed": True,
            "error":        str(exc),
            "elapsed_s":    round(time.time() - t0, 2),
            "model":        OLLAMA_MODEL,
            "backend":      "ollama",
        }


def result_path(row: dict) -> Path:
    tile_id = Path(row["tile_path"]).stem
    return RESULTS_DIR / f"{tile_id}.json"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",  type=int, default=4,
                        help="Parallel workers (default 4; Ollama queues excess)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print plan without running inference")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Cap number of tiles (for testing)")
    args = parser.parse_args()

    if not MANIFEST_CSV.exists():
        sys.exit(f"Manifest not found: {MANIFEST_CSV}")

    rows = list(csv.DictReader(open(MANIFEST_CSV)))
    print(f"Manifest: {MANIFEST_CSV.name} — {len(rows)} tiles")

    # Resume: skip tiles already processed
    pending = [r for r in rows if not result_path(r).exists()]
    done    = len(rows) - len(pending)
    print(f"  Already done : {done}")
    print(f"  Pending      : {len(pending)}")

    if args.max_tiles:
        pending = pending[:args.max_tiles]
        print(f"  Capped at    : {args.max_tiles}")

    if args.dry_run or not pending:
        if not pending:
            print("All tiles complete.")
        return

    check_ollama()

    # Estimate time
    est_s_per_tile = 6.0   # Ollama continuous batching + 4 workers → ~6s/tile
    est_total_min  = len(pending) * est_s_per_tile / 60 / args.workers
    print(
        f"\n  Workers       : {args.workers}"
        f"\n  Est. time     : {est_total_min:.0f} min "
        f"({est_total_min/60:.1f} h) with {args.workers} workers"
        f"\n  Results dir   : {RESULTS_DIR}"
        f"\n  Starting...\n"
    )

    t_start  = time.time()
    success  = 0
    failures = 0
    parse_ok = 0

    # Warm up model with first tile before flooding with parallel requests
    warmup = pending[:1]
    print("Warming up model (first tile)...")
    r0 = infer_tile(warmup[0])
    with open(result_path(warmup[0]), "w") as f:
        json.dump(r0, f)
    warmup_elapsed = r0.get("elapsed_s", 0)
    print(f"  Warm tile: {warmup_elapsed:.1f}s  parse_failed={r0.get('parse_failed')}")
    success += 1
    if not r0.get("parse_failed"):
        parse_ok += 1

    remaining = pending[1:]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(infer_tile, row): row for row in remaining}

        with tqdm(total=len(remaining), unit="tile", desc="Inference") as pbar:
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    result = fut.result()
                    with open(result_path(row), "w") as f:
                        json.dump(result, f)
                    success += 1
                    if not result.get("parse_failed"):
                        parse_ok += 1
                    else:
                        failures += 1
                    pbar.set_postfix(
                        ok=parse_ok,
                        fail=failures,
                        s=f"{result.get('elapsed_s', 0):.1f}s",
                    )
                except Exception as exc:
                    failures += 1
                    tqdm.write(f"  THREAD ERROR {Path(row['tile_path']).name}: {exc}")
                finally:
                    pbar.update(1)

    elapsed_total = time.time() - t_start
    total_processed = success + failures
    avg_s = elapsed_total / max(total_processed, 1)

    print(f"\n{'='*55}")
    print(f"  Parallel inference complete")
    print(f"  Total tiles       : {total_processed}")
    print(f"  Parse success     : {parse_ok}  ({parse_ok/max(total_processed,1)*100:.0f}%)")
    print(f"  Parse failures    : {failures}")
    print(f"  Wall time         : {elapsed_total/60:.1f} min")
    print(f"  Effective speed   : {avg_s:.1f} s/tile (wall) | "
          f"{avg_s * args.workers:.1f} s/tile (GPU)")
    print(f"  Results dir       : {RESULTS_DIR}")
    print(f"{'='*55}")
    print(f"\nNext: python3 ~/hiz_pipeline/evaluate.py")


if __name__ == "__main__":
    main()
