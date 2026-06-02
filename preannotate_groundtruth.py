"""
preannotate_groundtruth.py
==========================
OWLv2 open-vocabulary object detection pre-annotation for ground truth creation.

Uses google/owlv2-base-patch16-ensemble (pure transformers, MPS-native) to detect
the 33 HIZ object classes in Henri's 45 parcel orthomosaics.  Outputs are ready
to import into Label Studio for human verification.

Why OWLv2, not GroundingDINO:
  GroundingDINO requires a CUDA-compiled C++ extension (deformable attention).
  This extension will not build on Apple Silicon without a CUDA toolchain.
  OWLv2 is pure Python / PyTorch and runs correctly on MPS at ~1.2 s/chip.

Processing per parcel
---------------------
  1. Load 4-band GeoTIFF (RGB + CHM in Band 4)
  2. Normalize RGB float32 → uint8 (p2/p98 per-band stretch)
  3. Tile at 960×960 px with 50% overlap (480 px stride)
  4. Skip tiles with >60% blank pixels
  5. Run OWLv2 with 33 class text prompts → bounding boxes + scores
  6. NMS across overlapping tiles (IoU threshold 0.5)
  7. Save tile PNG + COCO JSON for Label Studio

Outputs  (all in  AI for HIZ/preannotations/)
----------------------------------------------
  images/        — PNG tiles (960×960), named <parcel>_r<row>_c<col>.png
  annotations/   — per-parcel COCO JSON   <parcel>_coco.json
  ground_truth_coco.json — merged COCO JSON (all parcels, all tiles)
  preannotation_log.txt

Run
---
  /opt/miniconda3/bin/python3 preannotate_groundtruth.py [--parcels N] [--threshold 0.12] [--no-resume]

Privacy
-------
  Henri data is LOCAL INFERENCE ONLY. No images sent to external APIs.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
PARCEL_DIR = Path("/Users/basanthyajman/hiz_data/henri")
OUT_DIR    = BASE_DIR / "preannotations"
IMG_DIR    = OUT_DIR / "images"
ANN_DIR    = OUT_DIR / "annotations"
LOG_FILE   = OUT_DIR / "preannotation_log.txt"
MERGED_OUT = OUT_DIR / "ground_truth_coco.json"
RESUME_FILE= OUT_DIR / ".preannotation_done"

# ── Constants ─────────────────────────────────────────────────────────────────
CHIP_SIZE   = 960           # OWLv2 native resolution
STRIDE      = 480           # 50% overlap
BLANK_THRESH= 0.60          # skip chip if >60% pixels are near-black
OWL_THRESH  = 0.12          # confidence threshold for OWLv2 (lower than CLIP; bboxes are noisier)
NMS_IOU     = 0.50          # IoU threshold for NMS across overlapping tiles
MODEL_ID    = "google/owlv2-base-patch16-ensemble"

# 33 object classes from lab taxonomy (same as CLIP pipeline)
OBJECT_CLASSES = [
    "woodpile", "furniture", "car", "rv", "above_ground_pool_or_hot_tub",
    "play_set", "pergola_gazebo", "garbage_bin", "boat", "propane",
    "storage_shed", "clutter", "planters", "fuel_breaks", "irrigation",
    "driveway", "welcome_mat", "address_sign", "fuel_or_flame_wick", "hoses",
    "broom", "ladder", "portable_gas_pump", "curtains", "lights",
    "live_herb", "live_shrub", "live_tree", "dead_vegetation", "mulch",
    "deck_patio", "fence", "bbq_grill",
]

OBJECT_DISPLAY = {c: c.replace("_", " ") for c in OBJECT_CLASSES}
OBJECT_DISPLAY.update({
    "above_ground_pool_or_hot_tub": "above ground pool or hot tub",
    "fuel_or_flame_wick"          : "fuel or flame wick",
    "deck_patio"                  : "deck or patio",
    "pergola_gazebo"              : "pergola or gazebo",
    "bbq_grill"                   : "bbq grill",
    "live_herb"                   : "live herb or groundcover",
    "live_shrub"                  : "live shrub",
    "live_tree"                   : "live tree",
    "dead_vegetation"             : "dead vegetation or dry grass",
    "fuel_breaks"                 : "fuel break or gravel strip",
    "portable_gas_pump"           : "portable gas pump or generator",
    "deck_patio"                  : "wooden deck or patio",
})

# Build text prompt list — one string per class, aerial-specific
TEXT_PROMPTS = [
    f"aerial view of {OBJECT_DISPLAY[c]} in a residential yard"
    for c in OBJECT_CLASSES
]

# ── Logging ───────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)
ANN_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_band(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Stretch float32 band to uint8 using percentile clipping."""
    lo, hi = np.nanpercentile(arr, p_low), np.nanpercentile(arr, p_high)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def load_parcel(tif_path: Path):
    """Load GeoTIFF. Returns (rgb_uint8 HxWx3, chm_float32 HxW)."""
    import rasterio
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        b1 = src.read(1).astype(np.float32)
        b2 = src.read(2).astype(np.float32)
        b3 = src.read(3).astype(np.float32)
        chm = src.read(4).astype(np.float32) if src.count >= 4 else np.zeros_like(b1)
        nodata = src.nodata or 0.0
    # Zero-out nodata
    for band in (b1, b2, b3, chm):
        band[band == nodata] = 0.0
    rgb = np.stack([normalize_band(b) for b in (b1, b2, b3)], axis=-1)  # HxWx3 uint8
    gsd_cm = abs(meta["transform"][0]) * 100  # metres → cm
    return rgb, chm, gsd_cm, meta


def tile_generator(rgb: np.ndarray, chm: np.ndarray):
    """Yield (row_off, col_off, chip_rgb HxWx3, chip_chm HxW) for 960×960 tiles."""
    H, W = rgb.shape[:2]
    rows = list(range(0, max(H - CHIP_SIZE + 1, 1), STRIDE)) + ([H - CHIP_SIZE] if H > CHIP_SIZE else [])
    cols = list(range(0, max(W - CHIP_SIZE + 1, 1), STRIDE)) + ([W - CHIP_SIZE] if W > CHIP_SIZE else [])
    seen = set()
    for r in rows:
        r = max(0, min(r, H - CHIP_SIZE)) if H >= CHIP_SIZE else 0
        for c in cols:
            c = max(0, min(c, W - CHIP_SIZE)) if W >= CHIP_SIZE else 0
            if (r, c) in seen:
                continue
            seen.add((r, c))
            h = min(CHIP_SIZE, H - r)
            w = min(CHIP_SIZE, W - c)
            chip = np.zeros((CHIP_SIZE, CHIP_SIZE, 3), dtype=np.uint8)
            chip[:h, :w] = rgb[r:r+h, c:c+w]
            cchm = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
            cchm[:h, :w] = chm[r:r+h, c:c+w]
            yield r, c, chip, cchm


def is_blank(chip: np.ndarray, threshold: float = BLANK_THRESH) -> bool:
    blank_px = np.all(chip < 8, axis=-1).sum()
    return blank_px / chip.size * 3 > threshold


def iou(boxA, boxB):
    """IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter)


def nms(detections: list, iou_thresh: float = NMS_IOU) -> list:
    """
    Non-maximum suppression across detections.
    Each detection: {"bbox_parcel": [x1,y1,x2,y2], "score": float, "label_idx": int}
    """
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    suppressed = [False] * len(detections)
    for i, d in enumerate(detections):
        if suppressed[i]:
            continue
        kept.append(d)
        for j in range(i + 1, len(detections)):
            if not suppressed[j] and d["label_idx"] == detections[j]["label_idx"]:
                if iou(d["bbox_parcel"], detections[j]["bbox_parcel"]) > iou_thresh:
                    suppressed[j] = True
    return kept


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device: str):
    log.info(f"Loading {MODEL_ID} …")
    t0 = time.time()
    proc  = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    log.info(f"Loaded in {time.time()-t0:.1f}s on {device}")
    return model, proc


# ── Inference on one tile ─────────────────────────────────────────────────────

@torch.no_grad()
def detect_tile(
    chip_rgb: np.ndarray,
    row_off: int,
    col_off: int,
    model,
    proc,
    device: str,
    threshold: float,
) -> list:
    """
    Run OWLv2 on one tile.  Returns list of raw detections with parcel-level coords.
    Each: {"bbox_parcel": [x1,y1,x2,y2], "score": float, "label_idx": int, "label": str,
           "tile_row": row_off, "tile_col": col_off}
    """
    img = Image.fromarray(chip_rgb)
    h, w = chip_rgb.shape[:2]

    inputs = proc(text=[TEXT_PROMPTS], images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)

    target_sizes = torch.tensor([[h, w]], device=device)
    results = proc.post_process_grounded_object_detection(
        outputs=out, threshold=threshold, target_sizes=target_sizes
    )[0]

    boxes  = results["boxes"].cpu().numpy()   # N×4  x1y1x2y2 in tile coords
    scores = results["scores"].cpu().numpy()  # N
    labels = results["labels"].cpu().numpy()  # N  (index into TEXT_PROMPTS)

    dets = []
    for box, score, label_idx in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        # Convert tile-local coords → parcel-level coords
        dets.append({
            "bbox_parcel": [
                float(x1 + col_off), float(y1 + row_off),
                float(x2 + col_off), float(y2 + row_off),
            ],
            "score"     : float(score),
            "label_idx" : int(label_idx),
            "label"     : OBJECT_CLASSES[int(label_idx)],
            "tile_row"  : row_off,
            "tile_col"  : col_off,
        })
    return dets


# ── COCO helpers ──────────────────────────────────────────────────────────────

def build_coco_categories():
    return [{"id": i + 1, "name": cls, "supercategory": "hiz_object"}
            for i, cls in enumerate(OBJECT_CLASSES)]

CAT_NAME_TO_ID = {cls: i + 1 for i, cls in enumerate(OBJECT_CLASSES)}


def write_parcel_coco(
    parcel_id: str,
    tile_records: list,  # list of {"png_path", "width", "height", "detections":[...]}
    out_path: Path,
):
    images, annotations = [], []
    img_id  = 1
    ann_id  = 1
    for tile in tile_records:
        images.append({
            "id"       : img_id,
            "file_name": tile["png_path"],
            "width"    : tile["width"],
            "height"   : tile["height"],
            "parcel_id": parcel_id,
            "tile_row" : tile["tile_row"],
            "tile_col" : tile["tile_col"],
        })
        for det in tile["detections"]:
            # bbox is tile-local [x1,y1,x2,y2] → COCO [x,y,w,h]
            x1, y1, x2, y2 = det["bbox_tile"]
            w = x2 - x1
            h_box = y2 - y1
            annotations.append({
                "id"         : ann_id,
                "image_id"   : img_id,
                "category_id": CAT_NAME_TO_ID[det["label"]],
                "bbox"       : [float(x1), float(y1), float(w), float(h_box)],
                "area"       : float(w * h_box),
                "score"      : float(det["score"]),
                "iscrowd"    : 0,
            })
            ann_id += 1
        img_id += 1

    coco = {
        "info"       : {"description": f"HIZ pre-annotations — {parcel_id}", "version": "1.0"},
        "categories" : build_coco_categories(),
        "images"     : images,
        "annotations": annotations,
    }
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)
    return coco


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(parcels_limit: int, threshold: float, resume: bool):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"Device: {device}  |  threshold: {threshold}  |  model: {MODEL_ID}")

    model, proc = load_model(device)

    # Discover parcels
    tif_files = sorted(PARCEL_DIR.glob("*cliptoparcel.tif"))  # skip raw area tiles
    if parcels_limit:
        tif_files = tif_files[:parcels_limit]

    # Resume tracking
    done_parcels = set()
    if resume and RESUME_FILE.exists():
        done_parcels = set(RESUME_FILE.read_text().splitlines())
        log.info(f"Resuming: {len(done_parcels)} parcels already done")

    all_coco_images      = []
    all_coco_annotations = []
    global_img_id = 1
    global_ann_id = 1

    for tif_idx, tif_path in enumerate(tif_files):
        parcel_id = tif_path.stem
        if parcel_id in done_parcels:
            log.info(f"[{tif_idx+1}/{len(tif_files)}] {parcel_id} — skipped (done)")
            continue

        log.info(f"\n{'─'*60}")
        log.info(f"[{tif_idx+1}/{len(tif_files)}] Parcel: {parcel_id}")
        t_parcel = time.time()

        try:
            rgb, chm, gsd_cm, meta = load_parcel(tif_path)
        except Exception as e:
            log.error(f"  Load failed: {e}")
            continue

        H, W = rgb.shape[:2]
        log.info(f"  Raster: {W}×{H} px  GSD={gsd_cm:.2f} cm")

        # Collect all detections (parcel-level coords) across tiles
        all_parcel_dets: list = []
        tile_records: list = []
        n_tiles = 0
        n_blank = 0

        for row_off, col_off, chip_rgb, chip_chm in tile_generator(rgb, chm):
            if is_blank(chip_rgb):
                n_blank += 1
                continue

            n_tiles += 1
            t0 = time.time()
            tile_dets = detect_tile(chip_rgb, row_off, col_off, model, proc, device, threshold)
            elapsed = time.time() - t0

            # Save tile PNG
            tile_name = f"{parcel_id}_r{row_off:05d}_c{col_off:05d}.png"
            tile_path = IMG_DIR / tile_name
            Image.fromarray(chip_rgb).save(tile_path, optimize=False)

            # Tile-local bboxes for COCO (reset origin to 0,0 for the tile image)
            tile_local_dets = []
            for d in tile_dets:
                x1p, y1p, x2p, y2p = d["bbox_parcel"]
                tile_local_dets.append({
                    "bbox_tile": [x1p - col_off, y1p - row_off, x2p - col_off, y2p - row_off],
                    "label"    : d["label"],
                    "score"    : d["score"],
                    "label_idx": d["label_idx"],
                })

            tile_records.append({
                "png_path" : str(tile_path),
                "width"    : chip_rgb.shape[1],
                "height"   : chip_rgb.shape[0],
                "tile_row" : row_off,
                "tile_col" : col_off,
                "detections": tile_local_dets,
            })

            all_parcel_dets.extend(tile_dets)
            log.info(f"  Tile r={row_off} c={col_off}: {len(tile_dets)} dets in {elapsed:.2f}s")

        # NMS across the whole parcel
        post_nms = nms(all_parcel_dets, iou_thresh=NMS_IOU)
        log.info(f"  Tiles: {n_tiles} valid, {n_blank} blank  |  "
                 f"Raw dets: {len(all_parcel_dets)}  →  After NMS: {len(post_nms)}")

        # Write per-parcel tile-level COCO (for Label Studio import)
        parcel_coco_path = ANN_DIR / f"{parcel_id}_coco.json"
        parcel_coco = write_parcel_coco(parcel_id, tile_records, parcel_coco_path)

        # Also write NMS-filtered parcel-level boxes (for spatial analysis)
        nms_path = ANN_DIR / f"{parcel_id}_nms.json"
        with open(nms_path, "w") as f:
            json.dump({
                "parcel_id": parcel_id,
                "n_tiles"  : n_tiles,
                "n_raw"    : len(all_parcel_dets),
                "n_nms"    : len(post_nms),
                "detections": post_nms,
            }, f, indent=2)

        # Accumulate into merged COCO (renumber IDs)
        img_id_remap = {}
        for img in parcel_coco["images"]:
            new_id = global_img_id
            img_id_remap[img["id"]] = new_id
            merged_img = dict(img)
            merged_img["id"] = new_id
            all_coco_images.append(merged_img)
            global_img_id += 1

        for ann in parcel_coco["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = global_ann_id
            new_ann["image_id"] = img_id_remap[ann["image_id"]]
            all_coco_annotations.append(new_ann)
            global_ann_id += 1

        parcel_time = time.time() - t_parcel
        log.info(f"  Parcel done in {parcel_time:.1f}s  |  COCO: {parcel_coco_path}")

        # Mark done
        with open(RESUME_FILE, "a") as f:
            f.write(f"{parcel_id}\n")
        done_parcels.add(parcel_id)

    # Write merged COCO JSON — always rebuild from ALL per-parcel files
    # so resume runs don't lose previously processed parcels.
    all_coco_images_full, all_coco_annotations_full = [], []
    gid, aid = 1, 1
    for coco_path in sorted(ANN_DIR.glob("*_coco.json")):
        with open(coco_path) as f:
            pc = json.load(f)
        remap = {}
        for img in pc["images"]:
            remap[img["id"]] = gid
            merged_img = dict(img); merged_img["id"] = gid
            all_coco_images_full.append(merged_img); gid += 1
        for ann in pc["annotations"]:
            a = dict(ann); a["id"] = aid; a["image_id"] = remap[ann["image_id"]]
            all_coco_annotations_full.append(a); aid += 1

    merged = {
        "info"       : {"description": "HIZ pre-annotations — all parcels", "version": "1.0"},
        "categories" : build_coco_categories(),
        "images"     : all_coco_images_full,
        "annotations": all_coco_annotations_full,
    }
    with open(MERGED_OUT, "w") as f:
        json.dump(merged, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"DONE  |  {len(done_parcels)} parcels processed this run  |  "
             f"{len(all_coco_images_full)} total tiles  "
             f"|  {len(all_coco_annotations_full)} total pre-annotations")
    log.info(f"Merged COCO: {MERGED_OUT}")
    log.info(f"Images dir : {IMG_DIR}")
    log.info(f"Per-parcel : {ANN_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parcels", type=int, default=0, help="Limit to first N parcels (0=all)")
    parser.add_argument("--threshold", type=float, default=OWL_THRESH)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    main(
        parcels_limit=args.parcels,
        threshold=args.threshold,
        resume=not args.no_resume,
    )
