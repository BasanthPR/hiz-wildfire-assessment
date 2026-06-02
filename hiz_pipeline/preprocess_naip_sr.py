"""
preprocess_naip_sr.py
HIZ-VLM Pipeline — 4× Upscaling of NAIP tiles

Upscales 60 cm NAIP tiles 4× → ~15 cm GSD using Lanczos interpolation.
(Real-ESRGAN incompatible with Python 3.13 / basicsr setup.py bug.)

Output: ~/hiz_pipeline/tiles_naip_sr/  +  tile_manifest_naip_sr.csv

Scientific purpose: test whether upscaled public imagery closes the
detection gap with 1.62 cm drone imagery (Novelty Claim N5 ablation).
Manuscript framing: "4× bicubic-equivalent upscaling (Lanczos)" —
a valid baseline; Real-ESRGAN AI-SR can be added when basicsr is
ported to Python 3.13.

Usage:
    python3 ~/hiz_pipeline/preprocess_naip_sr.py
"""

import csv
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PIPELINE_DIR  = Path.home() / "hiz_pipeline"
TILES_SRC_DIR = PIPELINE_DIR / "tiles_naip"
TILES_DST_DIR = PIPELINE_DIR / "tiles_naip_sr"
MANIFEST_SRC  = TILES_SRC_DIR / "tile_manifest_naip.csv"
MANIFEST_DST  = TILES_DST_DIR / "tile_manifest_naip_sr.csv"
SCALE         = 4   # 60 cm → 15 cm GSD
TILE_SIZE     = 512 # input px; output 2048×2048


def upscale_tile(src_path: Path, dst_path: Path) -> bool:
    try:
        img = Image.open(src_path).convert("RGB")
        w, h = img.size
        upscaled = img.resize((w * SCALE, h * SCALE), Image.LANCZOS)
        upscaled.save(dst_path, format="PNG")
        return True
    except Exception as e:
        print(f"  Failed {src_path.name}: {e}")
        return False


def main():
    if not MANIFEST_SRC.exists():
        sys.exit(f"NAIP manifest not found: {MANIFEST_SRC}\nRun preprocess_naip.py first.")

    TILES_DST_DIR.mkdir(parents=True, exist_ok=True)

    tiles = list(csv.DictReader(open(MANIFEST_SRC)))
    print(f"Upscaling {len(tiles)} NAIP tiles 4× (Lanczos): 60 cm → 15 cm GSD")

    done = {p.stem for p in TILES_DST_DIR.glob("*_sr.png")}
    pending = [r for r in tiles if Path(r["tile_path"]).stem + "_sr" not in done]
    if done:
        print(f"Resuming: {len(done)} done, {len(pending)} remaining.")
    if not pending:
        print("All tiles already upscaled.")
    else:
        errors = 0
        for row in tqdm(pending, desc="Lanczos 4×", unit="tile"):
            src = Path(row["tile_path"])
            dst = TILES_DST_DIR / (src.stem + "_sr.png")
            if not upscale_tile(src, dst):
                errors += 1
        print(f"Upscaling complete. Errors: {errors}/{len(pending)}")

    # Write SR manifest
    all_tiles = list(csv.DictReader(open(MANIFEST_SRC)))
    sr_rows = []
    for row in all_tiles:
        src = Path(row["tile_path"])
        dst = TILES_DST_DIR / (src.stem + "_sr.png")
        if dst.exists():
            r = dict(row)
            r["tile_path"] = str(dst)
            r["gsd_cm"]    = str(round(float(row["gsd_cm"]) / SCALE, 2))
            sr_rows.append(r)

    if sr_rows:
        with open(MANIFEST_DST, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sr_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sr_rows)
        print(f"SR manifest written: {MANIFEST_DST} ({len(sr_rows)} tiles)")
        print(f"GSD: 60 cm → 15 cm | Tile: {TILE_SIZE}px → {TILE_SIZE*SCALE}px")
    else:
        print("No SR tiles found.")


if __name__ == "__main__":
    main()
