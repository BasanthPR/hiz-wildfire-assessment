"""
preprocess.py
HIZ-VLM Pipeline — Step 4: Dataset Preprocessing
Processes all 45 consented parcel GeoTIFFs:
  4a. Read & validate
  4b. Normalize RGB to uint8
  4c. Extract & process CHM Band 4
  4d. Detect building footprint + derive zone buffers
  4e. Tile into 512x512 patches with zone labels
  4f. Save per-parcel metadata JSON

Usage:
    python3 ~/hiz_pipeline/preprocess.py
"""

import os
import sys
import json
import csv
import glob
import warnings
import traceback
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
HENRI_DIR   = Path.home() / "hiz_data" / "henri"
TILES_DIR   = Path.home() / "hiz_pipeline" / "tiles"
MANIFEST_CSV = TILES_DIR / "tile_manifest.csv"

# ─── Constants ────────────────────────────────────────────────────────────────
NODATA_VAL          = 3.4e+38
NODATA_THRESHOLD    = 3.0e+38    # values above this are treated as nodata
CHM_MAX_HEIGHT_M    = 60.0       # clamp for normalisation
CHM_BUILDING_MIN_M  = 2.5        # height threshold for structure detection
CHM_BUILDING_MAX_M  = 15.0       # upper bound (excludes tall trees)

# Zone distances in feet → convert to metres
ZONE_0_FT = 5.0
ZONE_1_FT = 30.0
ZONE_2_FT = 100.0
FT_TO_M   = 0.3048

TILE_SIZE_DEFAULT   = 512
TILE_SIZE_SUBPX     = 1024      # for par017 (GSD < 1 cm)
MAX_NODATA_FRACTION = 0.80      # skip tiles where >80% pixels are nodata

# Zone overlay colours (BGR for cv2, then converted back)
ZONE_COLOURS = {
    "Zone_0": (255, 0,   0),    # red
    "Zone_1": (255, 128, 0),    # orange
    "Zone_2": (255, 215, 0),    # yellow
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def percentile_stretch(band: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Clip to [lo, hi] percentile, scale to [0, 255] uint8."""
    valid = band[band < NODATA_THRESHOLD]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    p_lo = np.percentile(valid, lo)
    p_hi = np.percentile(valid, hi)
    if p_hi == p_lo:
        return np.zeros_like(band, dtype=np.uint8)
    clipped = np.clip(band, p_lo, p_hi)
    scaled = (clipped - p_lo) / (p_hi - p_lo) * 255.0
    return scaled.astype(np.uint8)


def detect_building_bbox(chm: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Threshold CHM to find main structure bounding box.
    Returns (row_min, col_min, row_max, col_max) in pixel coords,
    or None if no structure found.
    """
    mask = (chm >= CHM_BUILDING_MIN_M) & (chm <= CHM_BUILDING_MAX_M)
    mask_uint8 = mask.astype(np.uint8) * 255

    # Morphological close to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Find connected components, keep largest
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n_labels < 2:
        return None

    # stats[0] is background; find largest non-background component
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x  = stats[largest, cv2.CC_STAT_LEFT]
    y  = stats[largest, cv2.CC_STAT_TOP]
    w  = stats[largest, cv2.CC_STAT_WIDTH]
    h  = stats[largest, cv2.CC_STAT_HEIGHT]
    return (y, x, y + h, x + w)   # (row_min, col_min, row_max, col_max)


def expand_bbox(bbox: tuple, expand_px: int, img_h: int, img_w: int) -> tuple:
    r0, c0, r1, c1 = bbox
    return (
        max(0,     r0 - expand_px),
        max(0,     c0 - expand_px),
        min(img_h, r1 + expand_px),
        min(img_w, c1 + expand_px),
    )


def build_zone_masks(building_bbox, gsd_m, img_h, img_w):
    """
    Returns dict: {zone_name: bool mask array}
    Each mask is True inside that zone (annular region).
    Zone_0: 0-5ft ring, Zone_1: 5-30ft ring, Zone_2: 30-100ft ring
    """
    if building_bbox is None:
        # If no building detected, all pixels go to Zone_2
        zone2 = np.ones((img_h, img_w), dtype=bool)
        return {"Zone_0": np.zeros_like(zone2),
                "Zone_1": np.zeros_like(zone2),
                "Zone_2": zone2}

    px_z0 = int(ZONE_0_FT * FT_TO_M / gsd_m) if gsd_m > 0 else 0
    px_z1 = int(ZONE_1_FT * FT_TO_M / gsd_m) if gsd_m > 0 else 0
    px_z2 = int(ZONE_2_FT * FT_TO_M / gsd_m) if gsd_m > 0 else 0

    bb_z0 = expand_bbox(building_bbox, px_z0, img_h, img_w)
    bb_z1 = expand_bbox(building_bbox, px_z1, img_h, img_w)
    bb_z2 = expand_bbox(building_bbox, px_z2, img_h, img_w)

    def bbox_mask(bb):
        m = np.zeros((img_h, img_w), dtype=bool)
        m[bb[0]:bb[2], bb[1]:bb[3]] = True
        return m

    mask_z0 = bbox_mask(bb_z0)
    mask_z1 = bbox_mask(bb_z1)
    mask_z2 = bbox_mask(bb_z2)

    return {
        "Zone_0": mask_z0,
        "Zone_1": mask_z1 & ~mask_z0,
        "Zone_2": mask_z2 & ~mask_z1,
    }


def tile_zone_label(tile_r, tile_c, tile_h, tile_w, zone_masks: dict) -> str:
    """Determine zone of tile center pixel."""
    cy = tile_r + tile_h // 2
    cx = tile_c + tile_w // 2
    for zone in ("Zone_0", "Zone_1", "Zone_2"):
        m = zone_masks[zone]
        if cy < m.shape[0] and cx < m.shape[1] and m[cy, cx]:
            return zone
    return "Zone_2"


def draw_zone_overlay(rgb_img: Image.Image,
                      building_bbox,
                      zone_masks: dict,
                      gsd_m: float) -> Image.Image:
    """Draw zone boundary outlines on the RGB image."""
    img = rgb_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw zone bounding boxes as outlines (approximation for display)
    if building_bbox is not None:
        img_h, img_w = img.size[1], img.size[0]
        for zone, (ft, colour) in [
            ("Zone_0", (ZONE_0_FT,  (255,  0,   0,  180))),
            ("Zone_1", (ZONE_1_FT,  (255, 128,  0,  180))),
            ("Zone_2", (ZONE_2_FT,  (255, 215,  0,  180))),
        ]:
            expand_px = int(ft * FT_TO_M / gsd_m) if gsd_m > 0 else 0
            bb = expand_bbox(building_bbox, expand_px, img_h, img_w)
            # PIL uses (x0, y0, x1, y1), our bbox is (row0, col0, row1, col1)
            draw.rectangle(
                [bb[1], bb[0], bb[3], bb[2]],
                outline=colour, width=2
            )

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def nodata_fraction(arr: np.ndarray) -> float:
    return float(np.mean(arr >= NODATA_THRESHOLD))


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_parcel(tif_path: Path) -> dict | None:
    """Process a single parcel GeoTIFF. Returns tile records or None on failure."""
    parcel_id = tif_path.stem.replace("_4b_cliptoparcel", "")
    site = parcel_id[:3]

    try:
        with rasterio.open(tif_path) as ds:
            if ds.count < 4:
                raise ValueError(f"Expected 4 bands, got {ds.count}")

            transform = ds.transform
            gsd_m  = abs(transform.a)       # metres/pixel from transform
            gsd_cm = gsd_m * 100.0

            img_w, img_h = ds.width, ds.height

            r_raw = ds.read(1).astype(np.float32)
            g_raw = ds.read(2).astype(np.float32)
            b_raw = ds.read(3).astype(np.float32)
            chm   = ds.read(4).astype(np.float32)

        # ── 4b. Normalise RGB ────────────────────────────────────────────────
        r_u8 = percentile_stretch(r_raw)
        g_u8 = percentile_stretch(g_raw)
        b_u8 = percentile_stretch(b_raw)
        rgb_arr = np.stack([r_u8, g_u8, b_u8], axis=-1)
        rgb_img = Image.fromarray(rgb_arr, mode="RGB")
        rgb_full_path = TILES_DIR / f"{parcel_id}_rgb_full.png"
        rgb_img.save(str(rgb_full_path))

        # ── 4c. CHM processing ───────────────────────────────────────────────
        chm_clean = np.clip(chm, 0.0, CHM_MAX_HEIGHT_M)
        chm_uint8 = (chm_clean / CHM_MAX_HEIGHT_M * 255.0).astype(np.uint8)
        chm_png_path = TILES_DIR / f"{parcel_id}_chm.png"
        Image.fromarray(chm_uint8).save(str(chm_png_path))
        chm_npy_path = TILES_DIR / f"{parcel_id}_chm.npy"
        np.save(str(chm_npy_path), chm_clean)

        # ── 4d. Building detection + zone buffers ────────────────────────────
        building_bbox = detect_building_bbox(chm_clean)
        zone_masks = build_zone_masks(building_bbox, gsd_m, img_h, img_w)

        # Zone pixel radii for metadata
        px_z0 = int(ZONE_0_FT * FT_TO_M / gsd_m) if gsd_m > 0 else 0
        px_z1 = int(ZONE_1_FT * FT_TO_M / gsd_m) if gsd_m > 0 else 0
        px_z2 = int(ZONE_2_FT * FT_TO_M / gsd_m) if gsd_m > 0 else 0

        zone_annotated = draw_zone_overlay(rgb_img, building_bbox, zone_masks, gsd_m)
        zones_path = TILES_DIR / f"{parcel_id}_rgb_zones.png"
        zone_annotated.save(str(zones_path))

        # Zone masks as .npz
        masks_path = TILES_DIR / f"{parcel_id}_zone_masks.npz"
        np.savez_compressed(
            str(masks_path),
            Zone_0=zone_masks["Zone_0"],
            Zone_1=zone_masks["Zone_1"],
            Zone_2=zone_masks["Zone_2"],
        )

        # ── 4e. Tiling ───────────────────────────────────────────────────────
        tile_size   = TILE_SIZE_SUBPX if parcel_id == "par017" else TILE_SIZE_DEFAULT
        stride      = tile_size // 2
        rgb_np      = np.array(rgb_img, dtype=np.uint8)   # H x W x 3
        # Use r_raw as nodata indicator (original float32 with 3.4e38 values)
        nodata_ref  = r_raw

        tile_records = []
        row_idx = 0
        for row_start in range(0, img_h - tile_size + 1, stride):
            col_idx = 0
            for col_start in range(0, img_w - tile_size + 1, stride):
                tile_nd = nodata_ref[row_start:row_start + tile_size,
                                     col_start:col_start + tile_size]
                if nodata_fraction(tile_nd) > MAX_NODATA_FRACTION:
                    col_idx += 1
                    continue

                tile_rgb = rgb_np[row_start:row_start + tile_size,
                                  col_start:col_start + tile_size]
                tile_chm = chm_clean[row_start:row_start + tile_size,
                                     col_start:col_start + tile_size]

                zone = tile_zone_label(row_start, col_start,
                                       tile_size, tile_size, zone_masks)

                tile_name = f"{parcel_id}_tile_{row_idx:03d}_{col_idx:03d}.png"
                tile_path = TILES_DIR / tile_name
                Image.fromarray(tile_rgb).save(str(tile_path))

                tile_records.append({
                    "tile_path":        str(tile_path),
                    "parcel_id":        parcel_id,
                    "site":             site,
                    "row":              row_idx,
                    "col":              col_idx,
                    "zone":             zone,
                    "gsd_cm":           round(gsd_cm, 3),
                    "chm_mean_in_tile": round(float(tile_chm.mean()), 3),
                    "chm_max_in_tile":  round(float(tile_chm.max()),  3),
                    "tile_size_px":     tile_size,
                    "row_start_px":     row_start,
                    "col_start_px":     col_start,
                })
                col_idx += 1
            row_idx += 1

        # ── 4f. Per-parcel metadata JSON ─────────────────────────────────────
        # Convert all numpy scalars to native Python types for JSON serialisation
        def to_py(v):
            if hasattr(v, "item"):   # numpy scalar → Python scalar
                return v.item()
            if isinstance(v, (list, tuple)):
                return [to_py(x) for x in v]
            return v

        meta = {
            "parcel_id":         parcel_id,
            "site":              site,
            "gsd_cm":            round(gsd_cm, 3),
            "dims_px":           [int(img_w), int(img_h)],
            "n_tiles":           len(tile_records),
            "building_bbox_px":  to_py(list(building_bbox)) if building_bbox else None,
            "zone_pixel_radii":  {"Zone_0": int(px_z0), "Zone_1": int(px_z1),
                                  "Zone_2": int(px_z2)},
            "chm_range":         [round(float(chm_clean.min()), 3),
                                  round(float(chm_clean.max()), 3)],
            "tile_size_px":      tile_size,
        }
        meta_path = TILES_DIR / f"{parcel_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return tile_records

    except Exception:
        print(f"\n  [ERROR] {parcel_id}: {traceback.format_exc()}")
        return None


def main():
    tif_files = sorted([
        p for p in HENRI_DIR.glob("*.tif")
        if not p.name.endswith(".ovr")
    ])

    if not tif_files:
        sys.exit(f"No GeoTIFF files found in {HENRI_DIR}. "
                 f"Check that drone orthomosaic data is at {HENRI_DIR}.")

    print(f"Found {len(tif_files)} GeoTIFF files in {HENRI_DIR}")
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    all_tile_records: list[dict] = []
    failed_parcels: list[str] = []
    site_tile_counts: dict[str, int] = {}

    manifest_fields = [
        "tile_path", "parcel_id", "site", "row", "col", "zone",
        "gsd_cm", "chm_mean_in_tile", "chm_max_in_tile",
        "tile_size_px", "row_start_px", "col_start_px",
    ]

    with open(MANIFEST_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=manifest_fields)
        writer.writeheader()

        for tif_path in tqdm(tif_files, desc="Preprocessing parcels", unit="parcel"):
            records = process_parcel(tif_path)
            parcel_id = tif_path.stem.replace("_4b_cliptoparcel", "")

            if records is None:
                failed_parcels.append(parcel_id)
            else:
                for rec in records:
                    writer.writerow({k: rec[k] for k in manifest_fields})
                all_tile_records.extend(records)
                site = parcel_id[:3]
                site_tile_counts[site] = site_tile_counts.get(site, 0) + len(records)

    # ── Summary ───────────────────────────────────────────────────────────────
    zone_counts: dict[str, int] = {}
    for rec in all_tile_records:
        z = rec["zone"]
        zone_counts[z] = zone_counts.get(z, 0) + 1

    print("\n" + "=" * 60)
    print("  PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Total parcels processed : {len(tif_files) - len(failed_parcels)}")
    print(f"  Failed parcels          : {len(failed_parcels)}")
    if failed_parcels:
        for fp in failed_parcels:
            print(f"    - {fp}")
    print(f"  Total tiles generated   : {len(all_tile_records)}")
    print()
    print("  Tiles per site:")
    for site, count in sorted(site_tile_counts.items()):
        print(f"    {site}: {count}")
    print()
    print("  Tiles per zone:")
    for zone in ("Zone_0", "Zone_1", "Zone_2"):
        print(f"    {zone}: {zone_counts.get(zone, 0)}")
    print(f"\n  Tile manifest saved to: {MANIFEST_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
