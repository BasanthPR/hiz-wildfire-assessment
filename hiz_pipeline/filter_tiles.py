"""
filter_tiles.py
HIZ-VLM Pipeline — Tile Pre-Filter

Uses CLIP zero-shot classification to remove tiles that are pure
vegetation/forest with no visible residential structure.  Eliminates
~60% of Zone_1/Zone_2 tiles before the expensive VLM inference step.

Rules:
  Zone_0  → always KEEP (0-5 ft from building — highest risk, always relevant)
  Zone_1/2 → KEEP if CLIP structural score >= VEG_THRESHOLD
             SKIP  if vegetation score > VEG_THRESHOLD  (dense canopy, no property)

Output: ~/hiz_pipeline/tiles/tile_manifest_filtered.csv

Usage:
    python3 ~/hiz_pipeline/filter_tiles.py [--threshold 0.70]
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PIPELINE_DIR   = Path.home() / "hiz_pipeline"
TILES_DIR      = PIPELINE_DIR / "tiles"
MANIFEST_IN    = TILES_DIR / "tile_manifest.csv"
MANIFEST_OUT   = TILES_DIR / "tile_manifest_filtered.csv"
CLIP_MODEL_ID  = "openai/clip-vit-large-patch14-336"
VEG_THRESHOLD  = 0.70    # CLIP vegetation score above this → skip tile
BATCH_SIZE     = 32

STRUCTURAL_TEXT = (
    "aerial view of residential property with rooftop driveway "
    "vehicles and structures"
)
VEGETATION_TEXT = (
    "dense tree canopy forest aerial view no buildings no structures "
    "vegetation only"
)


def load_clip():
    print(f"Loading CLIP ({CLIP_MODEL_ID}) ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).eval()
    proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"  CLIP on {device}")
    return model, proc, device


def classify_batch(images, model, proc, device):
    """
    Returns vegetation probability for each image in the batch.
    Higher = more likely pure vegetation, lower = more likely structural.
    """
    texts = [STRUCTURAL_TEXT, VEGETATION_TEXT]
    inputs = proc(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    # logits_per_image: (B, 2) — col 0=structural, col 1=vegetation
    probs = out.logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs[:, 1]   # vegetation scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=VEG_THRESHOLD,
                        help=f"Vegetation score threshold (default {VEG_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts without writing manifest")
    args = parser.parse_args()
    thresh = args.threshold

    if not MANIFEST_IN.exists():
        sys.exit(f"Manifest not found: {MANIFEST_IN}. Run preprocess.py first.")

    rows = list(csv.DictReader(open(MANIFEST_IN)))
    print(f"Loaded {len(rows)} tiles from manifest.")

    # Split: Zone_0 always kept, Zone_1/2 go through CLIP filter
    z0_rows = [r for r in rows if r["zone"] == "Zone_0"]
    other   = [r for r in rows if r["zone"] != "Zone_0"]
    print(f"  Zone_0 (force-keep): {len(z0_rows)}")
    print(f"  Zone_1/Zone_2 (CLIP filter): {len(other)}")

    model, proc, device = load_clip()

    kept, skipped = [], []
    batch_rows, batch_imgs = [], []

    def flush_batch():
        if not batch_imgs:
            return
        scores = classify_batch(batch_imgs, model, proc, device)
        for row, score in zip(batch_rows, scores):
            row["_veg_score"] = f"{score:.4f}"
            if score >= thresh:
                skipped.append(row)
            else:
                kept.append(row)
        batch_rows.clear()
        batch_imgs.clear()

    for r in tqdm(other, desc="CLIP filter", unit="tile"):
        try:
            img = Image.open(r["tile_path"]).convert("RGB")
        except Exception:
            kept.append(r)  # keep on read error
            continue
        batch_rows.append(r)
        batch_imgs.append(img)
        if len(batch_imgs) >= BATCH_SIZE:
            flush_batch()
    flush_batch()

    # Free CLIP
    del model, proc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    total_kept = len(z0_rows) + len(kept)
    print(f"\n{'='*55}")
    print(f"  Filter results (threshold={thresh})")
    print(f"  Zone_0 force-kept : {len(z0_rows)}")
    print(f"  Zone_1/2 kept     : {len(kept)}")
    print(f"  Zone_1/2 skipped  : {len(skipped)}")
    print(f"  TOTAL kept        : {total_kept} / {len(rows)} "
          f"({total_kept/len(rows)*100:.0f}%)")
    print(f"  Tiles eliminated  : {len(skipped)} "
          f"({len(skipped)/len(rows)*100:.0f}%)")

    # Zone breakdown of skipped
    from collections import Counter
    skip_zones = Counter(r["zone"] for r in skipped)
    for z, n in sorted(skip_zones.items()):
        print(f"    {z} skipped: {n}")
    print(f"{'='*55}")

    if args.dry_run:
        print("Dry run — no manifest written.")
        return

    # Write filtered manifest
    all_kept = z0_rows + kept
    fieldnames = list(rows[0].keys())
    with open(MANIFEST_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_kept)

    print(f"\nFiltered manifest: {MANIFEST_OUT}")
    print(f"Skipped tile list: "
          f"{TILES_DIR / 'tile_manifest_skipped.csv'}")

    # Also save skipped list for audit
    with open(TILES_DIR / "tile_manifest_skipped.csv", "w", newline="") as f:
        writer = csv.DictWriter(f,
                                fieldnames=list(rows[0].keys()) + ["_veg_score"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(skipped)

    print("\nNext: python3 ~/hiz_pipeline/run_qwen25vl.py")


if __name__ == "__main__":
    main()
