"""
preprocess_naip.py
HIZ-VLM Pipeline — NAIP Public Imagery Preprocessing

Tiles NAIP GeoTIFF scenes into 512-pixel patches, detects building
footprints with Grounding DINO (or falls back to center-crop), assigns
zone labels from pixel distances, and writes a tile manifest identical
in schema to the drone tile manifest so run_qwen25vl_naip.py can consume it.

Key differences from preprocess.py (drone orthomosaic data):
  - No CHM band — NAIP is 4-band (RGBI) or 3-band (RGB)
  - GSD ≈ 60 cm (vs 1–3 cm) → zone radii in pixels are much smaller
  - Building detection via Grounding DINO text prompt "building . house . roof"

Outputs:
  ~/hiz_pipeline/tiles_naip/  — PNG tiles + tile_manifest_naip.csv

Usage:
    python3 ~/hiz_pipeline/preprocess_naip.py [--site fel]
    python3 ~/hiz_pipeline/preprocess_naip.py           # all sites
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

NAIP_DIR   = Path.home() / "hiz_data" / "naip"
TILES_DIR  = Path.home() / "hiz_pipeline" / "tiles_naip"
MANIFEST   = TILES_DIR / "tile_manifest_naip.csv"
TILES_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE   = 512
OVERLAP     = 64     # px overlap between tiles to avoid edge effects
NAIP_GSD_CM = 60.0   # NAIP 2022 nominal resolution

# Zone distances in feet → pixels at 60cm GSD
FT_TO_M  = 0.3048
M_TO_PX  = 1.0 / (NAIP_GSD_CM / 100.0)   # 1 px = 0.60 m → 1.67 px/m
ZONE_RADII_PX = {
    "Zone_0": int(5   * FT_TO_M * M_TO_PX),   #  ~2.5 px — single ring
    "Zone_1": int(30  * FT_TO_M * M_TO_PX),   # ~15 px
    "Zone_2": int(100 * FT_TO_M * M_TO_PX),   # ~51 px
}

MANIFEST_HEADER = [
    "tile_path", "parcel_id", "site", "row", "col",
    "zone", "gsd_cm", "chm_mean_in_tile", "chm_max_in_tile",
    "tile_size_px", "row_start_px", "col_start_px",
]


# ── Building detection (Grounding DINO or fallback) ────────────────────────────

_GDINO_MODEL = None
_GDINO_PROC  = None

def load_gdino():
    """Load Grounding DINO once and cache globally."""
    global _GDINO_MODEL, _GDINO_PROC
    if _GDINO_MODEL is not None:
        return _GDINO_MODEL, _GDINO_PROC
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print("  Loading Grounding DINO (once)...")
        _GDINO_PROC  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        _GDINO_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).eval()
        print("  Grounding DINO ready.")
    except Exception as e:
        print(f"  Grounding DINO unavailable ({e}), using center-crop fallback.")
    return _GDINO_MODEL, _GDINO_PROC


def detect_building_gdino(img_rgb: np.ndarray):
    """
    Returns (x0, y0, x1, y1) in pixels using Grounding DINO.
    Falls back to center-third crop if DINO is unavailable.
    Model is loaded once and reused across all scenes.
    """
    import torch
    model, proc = load_gdino()

    if model is not None and proc is not None:
        try:
            pil = Image.fromarray(img_rgb)
            inputs = proc(
                images=pil,
                text="building . house . roof .",
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = model(**inputs)

            results = proc.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=0.35,
                text_threshold=0.25,
                target_sizes=[pil.size[::-1]],
            )[0]

            boxes = results["boxes"].tolist()
            if boxes:
                scores = results["scores"].tolist()
                best   = max(range(len(scores)), key=lambda i: scores[i])
                x0, y0, x1, y1 = [int(v) for v in boxes[best]]
                return x0, y0, x1, y1
        except Exception:
            pass

    # Fallback: center third
    h, w = img_rgb.shape[:2]
    return w // 3, h // 3, 2 * w // 3, 2 * h // 3


def zone_for_tile(row_start, col_start, cx, cy, h, w):
    """
    Assign zone to a tile based on its closest corner distance to building centroid.
    Tiles that entirely surround Zone_0 get Zone_1, etc.
    """
    tile_cx = col_start + TILE_SIZE / 2
    tile_cy = row_start + TILE_SIZE / 2
    d = ((tile_cx - cx) ** 2 + (tile_cy - cy) ** 2) ** 0.5

    if d <= ZONE_RADII_PX["Zone_0"] + TILE_SIZE / 2:
        return "Zone_0"
    if d <= ZONE_RADII_PX["Zone_1"] + TILE_SIZE / 2:
        return "Zone_1"
    if d <= ZONE_RADII_PX["Zone_2"] + TILE_SIZE / 2:
        return "Zone_2"
    return None   # outside all zones → skip


# ── Read NAIP GeoTIFF ─────────────────────────────────────────────────────────

def read_naip_rgb(tif_path: Path) -> np.ndarray:
    """Return H×W×3 uint8 array (RGB) from NAIP GeoTIFF."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            # NAIP band order: R G B [NIR]
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)
    except Exception:
        # Fallback via PIL if rasterio struggles
        img = Image.open(tif_path).convert("RGB")
        return np.array(img)

    def norm(band):
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        if hi == lo:
            return np.zeros_like(band, dtype=np.uint8)
        return np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    return np.stack([norm(r), norm(g), norm(b)], axis=2)


# ── Process one NAIP scene ────────────────────────────────────────────────────

def process_scene(tif_path: Path, site: str) -> list[dict]:
    """Tile one NAIP GeoTIFF. Returns list of manifest rows."""
    scene_id = tif_path.stem
    print(f"  Scene: {scene_id}")

    img = read_naip_rgb(tif_path)
    if img is None:
        print(f"    Could not read {tif_path.name}")
        return []

    h, w = img.shape[:2]
    print(f"    Size: {w}×{h} px  ({w * NAIP_GSD_CM / 100:.0f}m × {h * NAIP_GSD_CM / 100:.0f}m)")

    x0, y0, x1, y1 = detect_building_gdino(img)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    print(f"    Building centroid: ({cx:.0f}, {cy:.0f})")

    # Save a full scene preview
    preview_path = TILES_DIR / f"{scene_id}_preview.jpg"
    Image.fromarray(img).save(preview_path, quality=85)

    rows   = []
    stride = TILE_SIZE - OVERLAP
    r_idx  = 0
    for row_start in range(0, h - TILE_SIZE + 1, stride):
        c_idx = 0
        for col_start in range(0, w - TILE_SIZE + 1, stride):
            zone = zone_for_tile(row_start, col_start, cx, cy, h, w)
            if zone is None:
                c_idx += 1
                continue

            patch = img[row_start:row_start + TILE_SIZE,
                        col_start:col_start + TILE_SIZE]
            tile_fname = f"{scene_id}_tile_{r_idx:03d}_{c_idx:03d}.png"
            tile_path  = TILES_DIR / tile_fname
            Image.fromarray(patch).save(tile_path)

            rows.append({
                "tile_path":       str(tile_path),
                "parcel_id":       scene_id,
                "site":            site,
                "row":             r_idx,
                "col":             c_idx,
                "zone":            zone,
                "gsd_cm":          NAIP_GSD_CM,
                "chm_mean_in_tile": 0.0,
                "chm_max_in_tile":  0.0,
                "tile_size_px":    TILE_SIZE,
                "row_start_px":    row_start,
                "col_start_px":    col_start,
            })
            c_idx += 1
        r_idx += 1

    print(f"    Tiles written: {len(rows)}")
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tile NAIP public imagery for HIZ VLM inference"
    )
    parser.add_argument(
        "--site", help="Process only this site (default: all)"
    )
    args = parser.parse_args()

    all_sites = [args.site] if args.site else [
        d.name for d in sorted(NAIP_DIR.iterdir()) if d.is_dir()
    ]
    if not all_sites:
        sys.exit(f"No NAIP data in {NAIP_DIR}. Run download_public_imagery.py first.")

    all_rows = []
    for site in all_sites:
        site_dir = NAIP_DIR / site
        tifs = sorted(site_dir.glob("*_image.tif")) + sorted(site_dir.glob("*.tif"))
        if not tifs:
            print(f"[{site}] No GeoTIFF files — skipping")
            continue
        print(f"\n[{site}] {len(tifs)} GeoTIFF(s)")
        for tif in tifs:
            rows = process_scene(tif, site)
            all_rows.extend(rows)

    if not all_rows:
        print("No tiles generated. Check that download_public_imagery.py ran successfully.")
        return

    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nManifest written: {MANIFEST}")
    print(f"Total NAIP tiles: {len(all_rows)}")
    print("Next: python3 ~/hiz_pipeline/run_qwen25vl_naip.py")


if __name__ == "__main__":
    main()
