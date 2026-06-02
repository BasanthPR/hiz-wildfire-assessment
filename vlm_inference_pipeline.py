"""
vlm_inference_pipeline.py
=========================
CLIP zero-shot classification pipeline for wildfire HIZ defensible-space.

Uses openai/clip-vit-large-patch14-336 (already cached) for fast (~0.5s/chip)
object detection across 33 HIZ object classes. No autoregressive generation
needed — cosine similarity between chip embeddings and per-class text prompts.

Inputs
------
  Drone orthomosaics         ~/hiz_data/henri/
  CLIP model                 openai/clip-vit-large-patch14-336 (HF cache)
  Graph-RAG lookup           AI for HIZ/graph_rag_lookup.py

Processing per parcel
---------------------
  1. Load 4-band GeoTIFF (RGB + CHM in Band 4)
  2. Normalize RGB float32 → uint8 (p2/p98 per-band stretch)
  3. Chip into 512×512 px patches at 50% overlap (256 px stride)
  4. Skip chips with >60% NoData/blank pixels
  5. For each valid chip:
       a. Run CLIP zero-shot over 33 object classes (MPS/CPU)
       b. Threshold cosine similarity to determine detected objects
       c. Assign zone via CHM heuristic (centre mean > 1.5 m → Zone_0)
       d. Query graph_rag_lookup for each detected object × zone
       e. Record compliance findings
  6. Aggregate per-parcel and write outputs

CLIP text prompt templates (per class)
---------------------------------------
  "aerial drone photo of [object] in a residential yard"
  "overhead view of [object] near a house"
  "satellite image showing [object]"
  → mean of 3 prompts per class for robustness

Detection threshold
-------------------
  A class is detected when its max cosine similarity across all prompts
  exceeds CLIP_THRESHOLD (default 0.24, calibrated for aerial imagery).

Zone assignment
---------------
  Band 4 (CHM) centre 128×128 px mean > 1.5 m → Zone_0 (near structure).
  Otherwise Zone_1 (defensible-space zone within parcel clip).

Privacy
-------
  Drone orthomosaic data is LOCAL INFERENCE ONLY. No images sent to external APIs.

Outputs (all in AI for HIZ/)
-----------------------------
  vlm_inference_results.json     — full per-chip records
  vlm_inference_summary.xlsx     — per-parcel, per-chip, compliance findings
  vlm_inference_log.txt          — progress + timing log

Run
---
  cd ~/Documents/HIZ/AI\ for\ HIZ
  python3 vlm_inference_pipeline.py [--parcels N] [--resume]
"""

import os
import sys
from pathlib import Path
import json
import time
import argparse
import logging
import warnings
import datetime
warnings.filterwarnings("ignore")

import numpy as np
import rasterio
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
HENRI_DIR = str(Path.home() / "hiz_data" / "henri")
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"

sys.path.insert(0, BASE_DIR)
from graph_rag_lookup import get_regulatory_context

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_JSON = os.path.join(BASE_DIR, "vlm_inference_results.json")
OUT_XLSX = os.path.join(BASE_DIR, "vlm_inference_summary.xlsx")
OUT_LOG  = os.path.join(BASE_DIR, "vlm_inference_log.txt")

# ── Constants ─────────────────────────────────────────────────────────────────
CHIP_SIZE        = 512
STRIDE           = 256
NODATA_VAL       = 3.4e+38
NODATA_TOL       = 0.6          # skip chip if >60% zero/blank
UNIFORM_TOL      = 0.6
CHM_STRUCT_THRESH = 1.5         # m, centre CHM → Zone_0
CHM_BAND         = 4
CLIP_THRESHOLD   = 0.285        # cosine similarity threshold for detection (calibrated for aerial imagery)

# 33 object classes from lab taxonomy
OBJECT_CLASSES = [
    "woodpile", "furniture", "car", "rv", "above_ground_pool_or_hot_tub",
    "play_set", "pergola_gazebo", "garbage_bin", "boat", "propane",
    "storage_shed", "clutter", "planters", "fuel_breaks", "irrigation",
    "driveway", "welcome_mat", "address_sign", "fuel_or_flame_wick", "hoses",
    "broom", "ladder", "portable_gas_pump", "curtains", "lights",
    "live_herb", "live_shrub", "live_tree", "dead_vegetation", "mulch",
    "deck_patio", "fence", "bbq_grill",
]

# Human-readable labels for CLIP prompts
OBJECT_DISPLAY = {c: c.replace("_", " ") for c in OBJECT_CLASSES}
OBJECT_DISPLAY.update({
    "above_ground_pool_or_hot_tub": "above ground pool or hot tub",
    "fuel_or_flame_wick": "fuel or flame wick",
    "deck_patio": "deck or patio",
    "pergola_gazebo": "pergola or gazebo",
    "bbq_grill": "bbq grill or barbecue",
    "dead_vegetation": "dead vegetation or dead plants",
    "live_herb": "lawn grass or herbs",
    "live_shrub": "shrubs or bushes",
    "live_tree": "trees",
    "garbage_bin": "garbage bin or trash can",
    "storage_shed": "storage shed or outbuilding",
    "play_set": "play set or swing set",
    "fuel_breaks": "fuel break or cleared strip",
    "address_sign": "address sign or house number",
    "portable_gas_pump": "portable gas pump or generator",
})

# Multiple text prompt templates per class for robustness
PROMPT_TEMPLATES = [
    "aerial drone photo of {} in a residential yard",
    "overhead view of {} near a house",
    "nadir view showing {} from above",
]


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(OUT_LOG, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Raster utilities
# ══════════════════════════════════════════════════════════════════════════════

def normalize_band(arr: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    valid = arr[np.isfinite(arr) & (arr < NODATA_VAL * 0.9)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(valid, p_low), np.percentile(valid, p_high)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo), 0, 1).multiply(255).astype(np.uint8) \
        if False else (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def load_parcel(tif_path: str):
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        r   = src.read(1).astype(np.float32)
        g   = src.read(2).astype(np.float32)
        b   = src.read(3).astype(np.float32)
        # CHM is band 4; synthesized from building footprints for Tahoe Donner parcels
        # for Tahoe area clips. Fall back to zeros if the file has only 3 bands.
        if src.count >= CHM_BAND:
            chm = src.read(CHM_BAND).astype(np.float32)
        else:
            chm = np.zeros((src.height, src.width), dtype=np.float32)
        gsd_m  = abs(src.transform.a)
        gsd_cm = gsd_m * 100

    nd = NODATA_VAL * 0.9
    for arr in (r, g, b):
        arr[arr > nd] = np.nan
    chm[chm > nd] = np.nan
    chm[chm < -10] = np.nan

    rgb = np.stack([normalize_band(r), normalize_band(g), normalize_band(b)], axis=-1)
    return rgb, chm, gsd_cm, meta


def chip_generator(rgb: np.ndarray, chm: np.ndarray):
    H, W = rgb.shape[:2]
    rows = list(range(0, max(H - CHIP_SIZE + 1, 1), STRIDE))
    cols = list(range(0, max(W - CHIP_SIZE + 1, 1), STRIDE))

    for r in rows:
        for c in cols:
            r_end = min(r + CHIP_SIZE, H)
            c_end = min(c + CHIP_SIZE, W)
            rgb_c = rgb[r:r_end, c:c_end]
            chm_c = chm[r:r_end, c:c_end]

            pad_h = CHIP_SIZE - rgb_c.shape[0]
            pad_w = CHIP_SIZE - rgb_c.shape[1]
            if pad_h > 0 or pad_w > 0:
                rgb_c = np.pad(rgb_c, ((0, pad_h), (0, pad_w), (0, 0)))
                chm_c = np.pad(chm_c, ((0, pad_h), (0, pad_w)), constant_values=np.nan)

            # Skip mostly-blank chips
            zero_frac = np.mean(rgb_c.sum(axis=-1) == 0)
            if zero_frac > NODATA_TOL:
                continue
            if np.mean(rgb_c == rgb_c[0, 0]) > UNIFORM_TOL:
                continue

            yield r, c, rgb_c, chm_c


def assign_zone(chm_chip: np.ndarray) -> str:
    cy, cx = CHIP_SIZE // 2, CHIP_SIZE // 2
    centre = chm_chip[cy - 64:cy + 64, cx - 64:cx + 64]
    valid  = centre[np.isfinite(centre)]
    if valid.size == 0:
        return "Zone_1"
    return "Zone_0" if valid.mean() > CHM_STRUCT_THRESH else "Zone_1"


# ══════════════════════════════════════════════════════════════════════════════
# CLIP inference
# ══════════════════════════════════════════════════════════════════════════════

def build_text_prompts() -> tuple[list[str], list[str]]:
    """
    Returns (all_prompt_strings, class_labels_aligned_to_prompts).
    Each class gets len(PROMPT_TEMPLATES) prompts.
    """
    prompts, labels = [], []
    for cls in OBJECT_CLASSES:
        disp = OBJECT_DISPLAY.get(cls, cls.replace("_", " "))
        for tmpl in PROMPT_TEMPLATES:
            prompts.append(tmpl.format(disp))
            labels.append(cls)
    return prompts, labels


def load_clip(device: str):
    log.info(f"Loading CLIP ({CLIP_MODEL_ID}) on {device} …")
    t0 = time.time()
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    log.info(f"CLIP loaded in {time.time()-t0:.1f}s")
    return model, proc


def precompute_text_embeddings(
    clip_model: CLIPModel,
    clip_proc: CLIPProcessor,
    prompts: list[str],
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """Pre-compute and normalise all text embeddings once."""
    all_embs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = clip_proc(text=batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = clip_model.get_text_features(**inputs)
            # transformers 5.x may return ModelOutput; unwrap if needed
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb)
    return torch.cat(all_embs, dim=0)   # (N_prompts, D)


def classify_chip(
    chip_rgb: np.ndarray,
    clip_model: CLIPModel,
    clip_proc: CLIPProcessor,
    text_embs: torch.Tensor,       # (N_prompts, D), normalised
    prompt_labels: list[str],      # class label for each prompt row
    device: str,
    threshold: float = CLIP_THRESHOLD,
) -> tuple[list[str], dict[str, float]]:
    """
    Returns (detected_classes, {class: max_similarity}).
    """
    pil_img = Image.fromarray(chip_rgb)
    inputs  = clip_proc(images=pil_img, return_tensors="pt")
    inputs  = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)
        if hasattr(img_emb, "pooler_output"):
            img_emb = img_emb.pooler_output
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)   # (1, D)

    # Cosine similarity: (1, D) @ (D, N_prompts) → (1, N_prompts)
    sims = (img_emb @ text_embs.T).squeeze(0).cpu().numpy()      # (N_prompts,)

    # Aggregate: max similarity across prompt variants per class
    class_scores: dict[str, float] = {}
    for sim, cls in zip(sims, prompt_labels):
        if cls not in class_scores or sim > class_scores[cls]:
            class_scores[cls] = float(sim)

    detected = [cls for cls, s in class_scores.items() if s >= threshold]
    return detected, class_scores


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0, "UNKNOWN": 0}

def _max_severity(findings: list[dict]) -> str:
    if not findings:
        return "NONE"
    return max(
        (f.get("severity", "NONE") for f in findings),
        key=lambda s: SEVERITY_ORDER.get(s, 0),
        default="NONE",
    )


def _save_json(results: list[dict]):
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)


def get_tif_paths() -> list[str]:
    return sorted(
        os.path.join(HENRI_DIR, f)
        for f in os.listdir(HENRI_DIR)
        if f.endswith("cliptoparcel.tif")  # skip raw area tiles (r04c08.tif etc.)
    )


def run_pipeline(parcels_limit: int = None, resume: bool = True, threshold: float = CLIP_THRESHOLD):
    log.info("=" * 70)
    log.info("VLM INFERENCE PIPELINE (CLIP) — HIZ WILDFIRE DEFENSIBLE SPACE")
    log.info(f"Started: {datetime.datetime.now().isoformat()}")
    log.info(f"CLIP threshold: {threshold}")
    log.info("=" * 70)

    # ── Resume ────────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    done_parcels: set[str] = set()
    if resume and os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = []
        done_parcels = {r["parcel_id"] for r in all_results}
        log.info(f"[RESUME] {len(all_results)} chip records, {len(done_parcels)} parcels done")

    # ── TIF list ──────────────────────────────────────────────────────────────
    tif_paths = get_tif_paths()
    if parcels_limit:
        tif_paths = tif_paths[:parcels_limit]
    log.info(f"{len(tif_paths)} parcel TIFs in {HENRI_DIR}")

    pending = [p for p in tif_paths
               if os.path.basename(p).replace(".tif", "") not in done_parcels]
    log.info(f"Pending: {len(pending)} parcels")
    if not pending:
        log.info("Nothing to do.")
        write_outputs(all_results)
        return

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clip_model, clip_proc = load_clip(device)

    # Pre-compute text embeddings
    all_prompts, prompt_labels = build_text_prompts()
    log.info(f"Pre-computing {len(all_prompts)} text embeddings ({len(OBJECT_CLASSES)} classes × {len(PROMPT_TEMPLATES)} templates) …")
    t0 = time.time()
    text_embs = precompute_text_embeddings(clip_model, clip_proc, all_prompts, device)
    log.info(f"Text embeddings ready in {time.time()-t0:.2f}s  shape={tuple(text_embs.shape)}")

    # ── Process parcels ───────────────────────────────────────────────────────
    for tif_path in pending:
        parcel_id = os.path.basename(tif_path).replace(".tif", "")
        site      = parcel_id[:3]
        log.info(f"\n{'─'*60}")
        log.info(f"Parcel: {parcel_id}  site={site}")

        try:
            rgb, chm, gsd_cm, _ = load_parcel(tif_path)
        except Exception as exc:
            log.error(f"  Load failed: {exc}")
            continue

        H, W = rgb.shape[:2]
        log.info(f"  {W}×{H} px  GSD={gsd_cm:.2f} cm")

        chips_ok = 0
        chips_skip = 0
        t_parcel = time.time()

        for row_off, col_off, chip_rgb, chip_chm in chip_generator(rgb, chm):
            zone = assign_zone(chip_chm)
            t_chip = time.time()

            try:
                detected, scores = classify_chip(
                    chip_rgb, clip_model, clip_proc,
                    text_embs, prompt_labels, device, threshold
                )
            except Exception as exc:
                log.warning(f"  chip ({row_off},{col_off}) error: {exc}")
                chips_skip += 1
                continue

            elapsed_chip = time.time() - t_chip

            # Top-5 scores for diagnostics
            top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
            top5_str = ", ".join(f"{c}={s:.3f}" for c, s in top5)

            # Graph-RAG compliance lookup
            compliance_findings = []
            for obj in detected:
                try:
                    ctx = get_regulatory_context(obj, zone)
                    compliance_findings.append({
                        "object_class"      : obj,
                        "zone"              : zone,
                        "clip_score"        : round(scores.get(obj, 0), 4),
                        "aerial_detectable" : ctx["aerial_detectable"],
                        "violations"        : ctx["violations"],
                        "ibhs_count"        : len(ctx["ibhs_requirements"]),
                        "prc_count"         : len(ctx["prc_sections"]),
                        "top_violation"     : ctx["violations"][0]["violation_id"] if ctx["violations"] else None,
                        "severity"          : ctx["violations"][0]["severity"] if ctx["violations"] else "NONE",
                    })
                except Exception as exc:
                    log.warning(f"  graph-rag lookup failed for {obj}: {exc}")

            chip_record = {
                "parcel_id"          : parcel_id,
                "site"               : site,
                "gsd_cm"             : round(gsd_cm, 3),
                "row_offset"         : row_off,
                "col_offset"         : col_off,
                "chip_size_px"       : CHIP_SIZE,
                "zone"               : zone,
                "detected_objects"   : detected,
                "clip_top5"          : top5_str,
                "compliance_findings": compliance_findings,
                "has_violations"     : any(f["violations"] for f in compliance_findings),
                "max_severity"       : _max_severity(compliance_findings),
                "chip_time_s"        : round(elapsed_chip, 3),
                "timestamp"          : datetime.datetime.now().isoformat(),
            }
            all_results.append(chip_record)
            chips_ok += 1

            if chips_ok % 20 == 0:
                log.info(f"  … {chips_ok} chips  top5=[{top5_str}]")
                _save_json(all_results)

        parcel_time = time.time() - t_parcel
        log.info(f"  Done: {chips_ok} chips in {parcel_time:.1f}s  ({parcel_time/max(chips_ok,1):.2f}s/chip)  {chips_skip} skipped")
        _save_json(all_results)

    # ── Final outputs ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info(f"Pipeline complete — {len(all_results)} chip records")
    write_outputs(all_results)


def write_outputs(results: list[dict]):
    _save_json(results)
    log.info(f"  ✓  {OUT_JSON}")

    try:
        import pandas as pd

        # Sheet 1: per-parcel summary
        parcels: dict[str, dict] = {}
        for r in results:
            pid = r["parcel_id"]
            parcels.setdefault(pid, {
                "parcel_id": pid, "site": r["site"], "gsd_cm": r["gsd_cm"],
                "chips": 0, "chips_with_violations": 0,
                "all_objects": set(), "sev_vals": [],
            })
            p = parcels[pid]
            p["chips"] += 1
            if r["has_violations"]:
                p["chips_with_violations"] += 1
            p["all_objects"].update(r["detected_objects"])
            p["sev_vals"].append(SEVERITY_ORDER.get(r["max_severity"], 0))

        parcel_rows = []
        for pid, p in parcels.items():
            sev_val = max(p["sev_vals"]) if p["sev_vals"] else 0
            sev_str = next((k for k, v in SEVERITY_ORDER.items() if v == sev_val), "NONE")
            parcel_rows.append({
                "parcel_id"            : p["parcel_id"],
                "site"                 : p["site"],
                "gsd_cm"               : p["gsd_cm"],
                "total_chips"          : p["chips"],
                "chips_with_violations": p["chips_with_violations"],
                "violation_rate"       : round(p["chips_with_violations"] / p["chips"], 3) if p["chips"] else 0,
                "unique_objects"       : ", ".join(sorted(p["all_objects"])),
                "max_severity"         : sev_str,
            })
        df_parcels = pd.DataFrame(parcel_rows)

        # Sheet 2: per-chip
        chip_rows = []
        for r in results:
            chip_rows.append({
                "parcel_id"       : r["parcel_id"],
                "site"            : r["site"],
                "gsd_cm"          : r["gsd_cm"],
                "row_offset"      : r["row_offset"],
                "col_offset"      : r["col_offset"],
                "zone"            : r["zone"],
                "detected_objects": ", ".join(r["detected_objects"]),
                "clip_top5"       : r.get("clip_top5", ""),
                "has_violations"  : r["has_violations"],
                "max_severity"    : r["max_severity"],
                "chip_time_s"     : r.get("chip_time_s", ""),
            })
        df_chips = pd.DataFrame(chip_rows)

        # Sheet 3: compliance findings
        finding_rows = []
        for r in results:
            for f in r["compliance_findings"]:
                finding_rows.append({
                    "parcel_id"       : r["parcel_id"],
                    "site"            : r["site"],
                    "chip_row"        : r["row_offset"],
                    "chip_col"        : r["col_offset"],
                    "zone"            : r["zone"],
                    "object_class"    : f["object_class"],
                    "clip_score"      : f.get("clip_score", ""),
                    "aerial_detectable": f["aerial_detectable"],
                    "ibhs_count"      : f["ibhs_count"],
                    "prc_count"       : f["prc_count"],
                    "top_violation"   : f["top_violation"],
                    "severity"        : f["severity"],
                })
        df_findings = pd.DataFrame(finding_rows)

        # Sheet 4: CLIP score heatmap data (per-class mean score per parcel)
        score_rows = []
        for r in results:
            row = {"parcel_id": r["parcel_id"], "site": r["site"]}
            # clip_top5 is a string — we need per-class scores; add all detected objects
            for f in r["compliance_findings"]:
                row[f["object_class"]] = f.get("clip_score", "")
            score_rows.append(row)
        df_scores = pd.DataFrame(score_rows)

        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
            df_parcels.to_excel(writer,  sheet_name="Per-Parcel Summary",  index=False)
            df_chips.to_excel(writer,    sheet_name="Per-Chip Detail",     index=False)
            df_findings.to_excel(writer, sheet_name="Compliance Findings",  index=False)
            df_scores.to_excel(writer,   sheet_name="CLIP Scores",          index=False)

        log.info(f"  ✓  {OUT_XLSX}")
        log.info(f"     Parcels={len(df_parcels)} Chips={len(df_chips)} Findings={len(df_findings)}")

    except Exception as exc:
        log.error(f"  XLSX write failed: {exc}")

    log.info("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP zero-shot VLM inference for HIZ defensible-space"
    )
    parser.add_argument("--parcels",   type=int,   default=None,         help="Limit to first N parcels")
    parser.add_argument("--no-resume", action="store_true",              help="Start fresh")
    parser.add_argument("--threshold", type=float, default=CLIP_THRESHOLD, help="CLIP detection threshold (default 0.24)")
    args = parser.parse_args()

    run_pipeline(
        parcels_limit=args.parcels,
        resume=not args.no_resume,
        threshold=args.threshold,
    )
