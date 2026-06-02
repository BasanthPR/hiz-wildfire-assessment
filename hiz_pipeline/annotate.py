"""
annotate.py
HIZ-VLM Pipeline — Step 9: Bounding Box Annotation
Produces final annotated images for each parcel × model combination.

Usage:
    python3 ~/hiz_pipeline/annotate.py
"""

import json
import sys
import csv
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path.home() / "hiz_pipeline"
TILES_DIR    = PIPELINE_DIR / "tiles"
ANNOTATED    = PIPELINE_DIR / "results" / "annotated"
MANIFEST_CSV = TILES_DIR / "tile_manifest.csv"
ANNOTATED.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen25vl": PIPELINE_DIR / "results" / "qwen25vl",
    "geochat":  PIPELINE_DIR / "results" / "geochat",
    "internvl": PIPELINE_DIR / "results" / "internvl",
}

# ─── Colours ──────────────────────────────────────────────────────────────────
STATUS_COLOURS = {
    "VIOLATION":  (220,  50,  50, 220),
    "COMPLIANT":  ( 50, 200,  50, 220),
    "UNCERTAIN":  (230, 200,  20, 220),
    "UNKNOWN":    (180, 180, 180, 220),
}

ZONE_OUTLINE_COLOURS = {
    "Zone_0": (255,  50,  50, 180),
    "Zone_1": (255, 160,  30, 180),
    "Zone_2": (230, 215,  30, 180),
}

SEVERITY_ICONS = {
    "CRITICAL": "!!",
    "HIGH":     "! ",
    "MEDIUM":   "~ ",
    "NONE":     "  ",
    "REVIEW":   "? ",
}


def load_font(size: int = 12):
    """Try to load a monospace font; fall back to PIL default."""
    candidates = [
        "/System/Library/Fonts/Supplemental/CourierNewPSMT.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_SM = load_font(11)
FONT_MD = load_font(13)


def tile_bbox_to_full_image(
    det_bbox: list,
    row_start_px: int,
    col_start_px: int,
) -> list[int]:
    """
    Convert tile-local [x0, y0, x1, y1] to full-image pixel coordinates.
    tile_bbox uses (x=col, y=row) convention (PIL / image convention).
    """
    if len(det_bbox) < 4:
        return []
    x0, y0, x1, y1 = det_bbox
    return [
        col_start_px + x0,
        row_start_px + y0,
        col_start_px + x1,
        row_start_px + y1,
    ]


def annotate_parcel(parcel_id: str, model_name: str, results_dir: Path) -> bool:
    """
    Produce one annotated image for a parcel + model combination.
    Returns True on success.
    """
    # Load zone-annotated base image
    base_path = TILES_DIR / f"{parcel_id}_rgb_zones.png"
    if not base_path.exists():
        base_path = TILES_DIR / f"{parcel_id}_rgb_full.png"
    if not base_path.exists():
        return False

    base_img = Image.open(base_path).convert("RGBA")
    overlay  = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)

    # Load per-parcel summary
    summary_path = results_dir / f"{parcel_id}_summary.json"
    if not summary_path.exists():
        return False
    with open(summary_path) as f:
        summary = json.load(f)

    # Load tile manifest for row/col pixel offsets
    tile_offsets: dict[str, dict] = {}
    if MANIFEST_CSV.exists():
        with open(MANIFEST_CSV) as f:
            for row in csv.DictReader(f):
                if row["parcel_id"] == parcel_id:
                    key = f"{row['row']}_{row['col']}"
                    tile_offsets[key] = {
                        "row_start_px": int(row["row_start_px"]),
                        "col_start_px": int(row["col_start_px"]),
                    }

    # Gather detections from all per-tile JSON files
    all_detections: list[dict] = []
    for tile_json in sorted(results_dir.glob(f"{parcel_id}_r*_c*.json")):
        try:
            with open(tile_json) as f:
                tile_data = json.load(f)
            row_i = str(tile_data.get("row", "0"))
            col_i = str(tile_data.get("col", "0"))
            offset = tile_offsets.get(f"{row_i}_{col_i}", {})

            for det in tile_data.get("result", {}).get("detections", []):
                det["_row_start_px"] = offset.get("row_start_px", 0)
                det["_col_start_px"] = offset.get("col_start_px", 0)
                all_detections.append(det)
        except Exception:
            continue

    # Draw detections
    for det in all_detections:
        bbox_local = det.get("bounding_box_pixels", [])
        r_start    = det.get("_row_start_px", 0)
        c_start    = det.get("_col_start_px", 0)
        full_bbox  = tile_bbox_to_full_image(bbox_local, r_start, c_start)

        if len(full_bbox) < 4:
            continue

        x0, y0, x1, y1 = full_bbox
        status    = det.get("compliance_status", "UNKNOWN")
        obj_class = det.get("object_class", "unknown")
        zone      = det.get("zone", "?")
        confidence = det.get("confidence", "?")
        severity  = det.get("severity", "NONE")
        icon      = SEVERITY_ICONS.get(severity, "  ")

        colour = STATUS_COLOURS.get(status, STATUS_COLOURS["UNKNOWN"])

        # Bounding box
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)

        # Label background
        label_text = f"{icon}{obj_class} | {zone} | {confidence}"
        try:
            text_w = draw.textlength(label_text, font=FONT_SM)
        except Exception:
            text_w = len(label_text) * 7

        lbl_x0 = x0
        lbl_y0 = max(0, y0 - 18)
        lbl_x1 = x0 + int(text_w) + 6
        lbl_y1 = y0

        label_bg = (*colour[:3], 200)
        draw.rectangle([lbl_x0, lbl_y0, lbl_x1, lbl_y1], fill=label_bg)
        draw.text((lbl_x0 + 3, lbl_y0 + 2), label_text,
                  fill=(255, 255, 255, 255), font=FONT_SM)

        # Small severity badge in top-right corner of box
        badge_text = severity[:4]
        badge_x = x1 - 35
        badge_y = y0 + 3
        draw.rectangle([badge_x, badge_y, badge_x+32, badge_y+14],
                       fill=colour)
        draw.text((badge_x+2, badge_y+2), badge_text,
                  fill=(255, 255, 255, 255), font=FONT_SM)

    # Parcel info header
    risk_score = summary.get("score", "?")
    risk_label = summary.get("risk_label", "?")
    n_viol     = summary.get("violation_count", "?")
    meta_path  = TILES_DIR / f"{parcel_id}_meta.json"
    gsd_cm     = "?"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        gsd_cm = meta.get("gsd_cm", "?")

    header_lines = [
        f"Parcel: {parcel_id}  |  Model: {model_name}  |  GSD: {gsd_cm} cm/px",
        f"Risk Score: {risk_score}  |  Risk Level: {risk_label}  |  Violations: {n_viol}",
        f"PRIVACY: LOCAL USE ONLY — Henri consented parcels (SJSU WIRC 2024)",
    ]
    for i, line in enumerate(header_lines):
        y_pos = 8 + i * 18
        try:
            tw = draw.textlength(line, font=FONT_MD)
        except Exception:
            tw = len(line) * 8
        draw.rectangle([5, y_pos - 2, 10 + int(tw), y_pos + 16],
                       fill=(0, 0, 0, 160))
        draw.text((8, y_pos), line, fill=(255, 255, 255, 230), font=FONT_MD)

    combined = Image.alpha_composite(base_img, overlay).convert("RGB")
    out_path  = ANNOTATED / f"{parcel_id}_{model_name}_annotated.png"
    combined.save(str(out_path))
    return True


def main():
    if not MANIFEST_CSV.exists():
        sys.exit(f"Manifest not found: {MANIFEST_CSV}. Run preprocess.py first.")

    parcel_ids: list[str] = []
    seen = set()
    with open(MANIFEST_CSV) as f:
        for row in csv.DictReader(f):
            pid = row["parcel_id"]
            if pid not in seen:
                parcel_ids.append(pid)
                seen.add(pid)

    print(f"Annotating {len(parcel_ids)} parcels × {len(MODELS)} models...")

    success = 0
    skipped = 0
    for model_name, results_dir in MODELS.items():
        for parcel_id in tqdm(parcel_ids, desc=f"Annotating ({model_name})",
                              unit="parcel"):
            ok = annotate_parcel(parcel_id, model_name, results_dir)
            if ok:
                success += 1
            else:
                skipped += 1

    print(f"\nAnnotation complete.")
    print(f"  Saved  : {success} images → {ANNOTATED}")
    print(f"  Skipped: {skipped} (missing results — run inference first)")


if __name__ == "__main__":
    main()
