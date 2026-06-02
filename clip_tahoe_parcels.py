"""
clip_tahoe_parcels.py
=====================
Preprocess Tahoe Donner area tiles into parcel-clipped 4-band GeoTIFFs
compatible with vlm_inference_pipeline.py and preannotate_groundtruth.py.

Inputs
------
  Area RGB TIFs    /Users/basanthyajman/hiz_data/henri/r*.tif
                   EPSG:2226  0.25m GSD  3-band uint8  12000×12000 px
  Building footps  /Users/basanthyajman/hiz_data/DMI-21085_Building_20211119.shp
                   EPSG:2226  37,791 polygons

Processing
----------
  1. Filter buildings: ENCLOSED_A > 800 sq ft (removes garages, sheds)
  2. Group touching / close buildings into parcels
     (single-linkage cluster at 50 ft / 15.2 m centroid distance)
  3. For each parcel group, extend bbox by BUFFER feet
  4. Clip RGB from whichever tile(s) cover the bbox
  5. Rasterize building footprints → synthetic CHM
     (building interior pixels = 3.0 m height → Zone_0 assignment)
  6. Stack RGB + CHM → 4-band GeoTIFF  tah_NNNN_cliptoparcel.tif
  7. Write spatial index JSON for later analysis

Outputs  (hiz_data/henri/)
--------------------------
  tah_0001_cliptoparcel.tif  …  tah_NNNN_cliptoparcel.tif
  (also writes tahoe_parcel_index.json  →  AI for HIZ/)

Run
---
  /opt/miniconda3/bin/python3 clip_tahoe_parcels.py [--max-parcels 60] [--buffer 164]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.mask
import rasterio.features
import rasterio.transform
import shapefile
from shapely.geometry import shape as shapely_shape, box as shapely_box, mapping

# ── Paths ─────────────────────────────────────────────────────────────────────
TILE_DIR  = Path("/Users/basanthyajman/hiz_data/henri")
SHP_PATH  = Path("/Users/basanthyajman/hiz_data/DMI-21085_Building_20211119.shp")
OUT_DIR   = TILE_DIR                        # same folder → pipeline picks them up automatically
IDX_PATH  = Path("/Users/basanthyajman/Documents/HIZ/AI for HIZ/tahoe_parcel_index.json")

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_AREA_SQFT    = 800      # minimum building footprint (sq ft) — removes sheds/garages
CLUSTER_DIST_FT  = 50       # centroid distance (ft) to merge buildings into one parcel
BUFFER_FT        = 164      # ~50m buffer around parcel bbox
SYNTHETIC_CHM_M  = 3.0      # height (m) assigned to building-footprint pixels → Zone_0
MIN_CHIP_PX      = 128      # skip parcels smaller than this after clipping
MAX_CHIP_PX      = 6000     # skip gigantic merged super-parcels

TILE_XMIN, TILE_XMAX = 7053000, 7071000
TILE_YMIN, TILE_YMAX = 2253000, 2268000


# ── Step 1: Load and filter buildings ────────────────────────────────────────

def load_buildings(min_area: float):
    """Return list of dicts with geometry (shapely) + attributes."""
    from shapely.geometry import shape as shp_shape
    reader = shapefile.Reader(str(SHP_PATH))
    fields = [f[0] for f in reader.fields[1:]]
    area_idx = fields.index("ENCLOSED_A")

    buildings = []
    for rec in reader.iterShapeRecords():
        area = rec.record[area_idx]
        if area < min_area:
            continue
        bx = rec.shape.bbox
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        # Only keep buildings that fall within our tile coverage
        if not (TILE_XMIN <= cx <= TILE_XMAX and TILE_YMIN <= cy <= TILE_YMAX):
            continue
        geom = shp_shape(rec.shape.__geo_interface__)
        buildings.append({
            "geom"     : geom,
            "cx"       : cx,
            "cy"       : cy,
            "area_sqft": area,
            "bbox"     : bx,
        })
    return buildings


# ── Step 2: Cluster buildings into parcels ────────────────────────────────────

def cluster_buildings(buildings: list, dist_ft: float) -> list:
    """
    Single-linkage clustering by centroid distance.
    Returns list of clusters, each a list of building dicts.
    """
    n = len(buildings)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    # Sort by cx for a fast sweep
    idx = sorted(range(n), key=lambda i: buildings[i]["cx"])
    cx = [buildings[i]["cx"] for i in idx]
    cy = [buildings[i]["cy"] for i in idx]

    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if cx[b] - cx[a] > dist_ft:
                break
            dx = cx[b] - cx[a]
            dy = cy[b] - cy[a]
            if (dx * dx + dy * dy) <= dist_ft * dist_ft:
                union(idx[a], idx[b])

    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(buildings[i])
    return list(groups.values())


# ── Step 3: Load tile spatial index ──────────────────────────────────────────

def load_tiles():
    """Return list of (path, bounds, transform, crs)."""
    tiles = []
    for tif_path in sorted(TILE_DIR.glob("r*.tif")):
        with rasterio.open(tif_path) as src:
            tiles.append({
                "path"     : tif_path,
                "bounds"   : src.bounds,
                "transform": src.transform,
                "crs"      : src.crs,
                "width"    : src.width,
                "height"   : src.height,
            })
    return tiles


def tiles_for_bbox(tiles: list, xmin: float, ymin: float, xmax: float, ymax: float):
    """Return tiles that overlap the given bbox."""
    return [
        t for t in tiles
        if (t["bounds"].left < xmax and t["bounds"].right > xmin and
            t["bounds"].bottom < ymax and t["bounds"].top > ymin)
    ]


# ── Step 4: Clip and mosaic RGB from tiles ────────────────────────────────────

def clip_rgb(tiles: list, xmin: float, ymin: float, xmax: float, ymax: float, transform_out):
    """
    Clip the bbox from one or more tiles and mosaic into a single array.
    Returns uint8 HxWx3 array aligned to transform_out.
    """
    from rasterio.merge import merge as rio_merge
    from rasterio.transform import from_bounds

    covering = tiles_for_bbox(tiles, xmin, ymin, xmax, ymax)
    if not covering:
        return None

    # Open all covering tiles
    datasets = [rasterio.open(t["path"]) for t in covering]
    try:
        if len(datasets) == 1:
            src = datasets[0]
            window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
            data = src.read(window=window, boundless=True, fill_value=0)  # 3×H×W
        else:
            # Mosaic then crop
            mosaic, mosaic_transform = rio_merge(datasets, bounds=(xmin, ymin, xmax, ymax))
            data = mosaic  # 3×H×W
    finally:
        for ds in datasets:
            ds.close()

    # data shape: (3, H, W) uint8
    rgb = np.transpose(data, (1, 2, 0))   # → H×W×3
    return rgb


# ── Step 5: Rasterize synthetic CHM ──────────────────────────────────────────

def rasterize_chm(
    buildings_in_group: list,
    xmin: float, ymin: float, xmax: float, ymax: float,
    out_h: int, out_w: int,
) -> np.ndarray:
    """
    Burn building footprints into a CHM raster.
    Building pixels = SYNTHETIC_CHM_M metres, background = 0.
    """
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize as rio_rasterize

    transform = from_bounds(xmin, ymin, xmax, ymax, out_w, out_h)
    geoms = [(mapping(b["geom"]), SYNTHETIC_CHM_M) for b in buildings_in_group]
    if not geoms:
        return np.zeros((out_h, out_w), dtype=np.float32)

    chm = rio_rasterize(
        geoms,
        out_shape=(out_h, out_w),
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )
    return chm


# ── Step 6: Write 4-band GeoTIFF ─────────────────────────────────────────────

def write_parcel_tif(
    out_path: Path,
    rgb: np.ndarray,      # H×W×3 uint8
    chm: np.ndarray,      # H×W float32
    xmin: float, ymin: float, xmax: float, ymax: float,
    crs,
):
    H, W = rgb.shape[:2]
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, W, H)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=4,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        # Write RGB as float32 (0-255 range) — pipeline expects float32 bands
        dst.write(rgb[:, :, 0].astype(np.float32), 1)
        dst.write(rgb[:, :, 1].astype(np.float32), 2)
        dst.write(rgb[:, :, 2].astype(np.float32), 3)
        dst.write(chm, 4)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(max_parcels: int, buffer_ft: float, min_area: float):
    print(f"Loading buildings (min area {min_area:.0f} sq ft) …")
    buildings = load_buildings(min_area)
    print(f"  {len(buildings)} qualifying buildings within tile extent")

    print(f"Clustering (max centroid distance {CLUSTER_DIST_FT} ft) …")
    clusters = cluster_buildings(buildings, CLUSTER_DIST_FT)
    print(f"  {len(clusters)} parcel groups")

    print("Loading tile index …")
    tiles = load_tiles()
    if not tiles:
        sys.exit("No r*.tif tiles found in " + str(TILE_DIR))
    crs = tiles[0]["crs"]
    print(f"  {len(tiles)} tiles, CRS={crs}")

    # Stratified spatial sample: divide extent into a grid, sample evenly from each cell
    # This gives geographic spread rather than just picking the largest buildings.
    # Filter: keep only single-family-sized parcels (800–6000 sq ft per building)
    clusters = [
        c for c in clusters
        if 800 <= sum(b["area_sqft"] for b in c) / max(len(c), 1) <= 6000
    ]
    print(f"  {len(clusters)} clusters after residential size filter")

    # Spatial grid stratification
    n_cells_x, n_cells_y = 4, 4
    cell_w = (TILE_XMAX - TILE_XMIN) / n_cells_x
    cell_h = (TILE_YMAX - TILE_YMIN) / n_cells_y
    per_cell = max(1, max_parcels // (n_cells_x * n_cells_y)) if max_parcels else 999

    import random
    random.seed(42)

    selected = []
    for ix in range(n_cells_x):
        for iy in range(n_cells_y):
            cell_xmin = TILE_XMIN + ix * cell_w
            cell_xmax = cell_xmin + cell_w
            cell_ymin = TILE_YMIN + iy * cell_h
            cell_ymax = cell_ymin + cell_h
            cell_clusters = [
                c for c in clusters
                if cell_xmin <= (sum(b["cx"] for b in c) / len(c)) <= cell_xmax
                and cell_ymin <= (sum(b["cy"] for b in c) / len(c)) <= cell_ymax
            ]
            random.shuffle(cell_clusters)
            selected.extend(cell_clusters[:per_cell])

    # If we still need more (some cells empty), pad from remainder
    selected_set = set(id(c) for c in selected)
    remainder = [c for c in clusters if id(c) not in selected_set]
    random.shuffle(remainder)
    if max_parcels and len(selected) < max_parcels:
        selected.extend(remainder[:max_parcels - len(selected)])

    clusters = selected[:max_parcels] if max_parcels else selected
    print(f"Processing {len(clusters)} spatially-stratified parcel groups …\n")

    index_records = []
    written = 0
    skipped = 0

    for i, cluster in enumerate(clusters):
        parcel_id = f"tah_{i+1:04d}"

        # Parcel bbox = union of all building bboxes + buffer
        all_bboxes = [b["bbox"] for b in cluster]
        xmin = min(bx[0] for bx in all_bboxes) - buffer_ft
        ymin = min(bx[1] for bx in all_bboxes) - buffer_ft
        xmax = max(bx[2] for bx in all_bboxes) + buffer_ft
        ymax = max(bx[3] for bx in all_bboxes) + buffer_ft

        # Clip to tile mosaic extent
        xmin = max(xmin, TILE_XMIN)
        ymin = max(ymin, TILE_YMIN)
        xmax = min(xmax, TILE_XMAX)
        ymax = min(ymax, TILE_YMAX)

        if xmax <= xmin or ymax <= ymin:
            skipped += 1
            continue

        # Expected output pixel dimensions (at 0.25m / ft ≈ 0.82 px/ft... wait
        # The tiles are in feet, GSD = 0.25m = 0.82 ft per pixel)
        # GSD = 0.25m means 4px/m.  In feet: 0.25m * 3.281 ft/m = 0.82 ft/px
        # So 1 ft → 1/0.82 ≈ 1.22 px
        GSD_FT = 0.25 * 3.28084  # feet per pixel
        out_w = max(1, round((xmax - xmin) / GSD_FT))
        out_h = max(1, round((ymax - ymin) / GSD_FT))

        if out_w < MIN_CHIP_PX or out_h < MIN_CHIP_PX:
            skipped += 1
            continue
        if out_w > MAX_CHIP_PX or out_h > MAX_CHIP_PX:
            skipped += 1
            continue

        # Clip RGB
        rgb = clip_rgb(tiles, xmin, ymin, xmax, ymax, None)
        if rgb is None:
            skipped += 1
            continue

        # Resize if needed (clip_rgb gives pixel count from source tile)
        # Just take whatever we got — rasterio crops to exact bounds
        out_h_actual, out_w_actual = rgb.shape[:2]

        if out_h_actual < MIN_CHIP_PX or out_w_actual < MIN_CHIP_PX:
            skipped += 1
            continue

        # Synthetic CHM
        chm = rasterize_chm(cluster, xmin, ymin, xmax, ymax, out_h_actual, out_w_actual)

        # Write
        out_path = OUT_DIR / f"{parcel_id}_cliptoparcel.tif"
        write_parcel_tif(out_path, rgb, chm, xmin, ymin, xmax, ymax, crs)

        total_area = sum(b["area_sqft"] for b in cluster)
        n_bldgs    = len(cluster)
        index_records.append({
            "parcel_id"   : parcel_id,
            "n_buildings" : n_bldgs,
            "total_area_sqft": total_area,
            "bbox_epsg2226": [xmin, ymin, xmax, ymax],
            "size_px"     : [out_w_actual, out_h_actual],
            "tif_path"    : str(out_path),
        })
        written += 1
        print(f"  [{i+1:>4}/{len(clusters)}] {parcel_id}  "
              f"{n_bldgs} bldg(s)  {total_area:.0f} sqft  "
              f"{out_w_actual}×{out_h_actual} px  → {out_path.name}")

    # Save index
    with open(IDX_PATH, "w") as f:
        json.dump(index_records, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Written : {written} parcel TIFs  ({skipped} skipped)")
    print(f"Index   : {IDX_PATH}")
    print(f"\nThese files are now in {OUT_DIR}")
    print("Run vlm_inference_pipeline.py or preannotate_groundtruth.py to process them.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parcels", type=int, default=60,
                    help="Max number of parcel clusters to output (default 60)")
    ap.add_argument("--buffer", type=float, default=BUFFER_FT,
                    help=f"Buffer in feet around each parcel bbox (default {BUFFER_FT})")
    ap.add_argument("--min-area", type=float, default=MIN_AREA_SQFT,
                    help=f"Minimum building area in sq ft (default {MIN_AREA_SQFT})")
    args = ap.parse_args()
    main(args.max_parcels, args.buffer, args.min_area)
