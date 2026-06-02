"""
aggregate_parallel_results.py
Converts qwen25vl_parallel per-tile JSONs into per-parcel summary files
that evaluate.py expects in results/qwen25vl/.

Run: python3 ~/hiz_pipeline/aggregate_parallel_results.py
"""

import json
from collections import defaultdict
from pathlib import Path

PIPELINE_DIR = Path.home() / "hiz_pipeline"
SRC_DIR      = PIPELINE_DIR / "results" / "qwen25vl_parallel"
DST_DIR      = PIPELINE_DIR / "results" / "qwen25vl"

RISK_SCORE = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
SCORE_RISK = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "UNKNOWN"}

DST_DIR.mkdir(parents=True, exist_ok=True)


def parcel_from_filename(fp: Path) -> str:
    """Extract parcel_id from filename like fel008_tile_000_005.json"""
    # Fallback: use embedded parcel_id
    return None


def main():
    tile_files = sorted(SRC_DIR.glob("*.json"))
    print(f"Loading {len(tile_files)} tile files from {SRC_DIR} ...")

    # Group tiles by parcel
    parcel_tiles: dict[str, list[dict]] = defaultdict(list)
    for fp in tile_files:
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        parcel_id = data.get("parcel_id")
        if not parcel_id:
            # Derive from filename: fel008_tile_000_005 -> fel008
            parts = fp.stem.split("_tile_")
            parcel_id = parts[0] if parts else fp.stem
        parcel_tiles[parcel_id].append(data)

        # Also write tile in the naming format evaluate.py expects (*_r*_c*.json)
        # Derive row/col from filename: fel008_tile_ROW_COL.json
        tile_path_str = data.get("tile_path", "")
        tile_path = Path(tile_path_str) if tile_path_str else None
        if tile_path and tile_path.exists():
            stem = tile_path.stem  # e.g. fel008_tile_001_005
            parts = stem.split("_tile_")
            if len(parts) == 2:
                rc = parts[1].split("_")
                if len(rc) == 2:
                    row, col = rc
                    dst_tile = DST_DIR / f"{parcel_id}_r{int(row)+1}_c{int(col)+1}.json"
                    if not dst_tile.exists():
                        # Write in format evaluate.py load_tile_results expects
                        tile_out = {
                            "tile_path": tile_path_str,
                            "parcel_id": parcel_id,
                            "zone": data.get("zone", "Zone_2"),
                            "gsd_cm": "unknown",
                            "row": str(int(row)+1),
                            "col": str(int(col)+1),
                            "model": data.get("model", "Qwen2.5-VL-7B-Instruct"),
                            "result": {
                                "detections": data.get("detections", []),
                                "overall_parcel_risk": data.get("overall_parcel_risk", "UNKNOWN"),
                                "aerial_limitations": data.get("aerial_limitations", []),
                            },
                        }
                        dst_tile.write_text(json.dumps(tile_out, indent=2))

    print(f"Found {len(parcel_tiles)} parcels.")

    # Build per-parcel summaries
    summaries_written = 0
    for parcel_id, tiles in parcel_tiles.items():
        all_detections = []
        all_risks = []
        parse_fails = 0

        for tile in tiles:
            dets = tile.get("detections", [])
            all_detections.extend(dets)
            risk_str = tile.get("overall_parcel_risk", "UNKNOWN")
            all_risks.append(RISK_SCORE.get(risk_str, 0))
            if tile.get("parse_failed"):
                parse_fails += 1

        violations = [d for d in all_detections
                      if d.get("compliance_status") == "VIOLATION"]
        max_risk_score = max(all_risks) if all_risks else 0
        risk_label = SCORE_RISK.get(max_risk_score, "UNKNOWN")

        # Compute numeric score: violations + critical severity bonus
        score = len(violations)
        for v in violations:
            if v.get("severity") == "CRITICAL":
                score += 2
            elif v.get("severity") == "HIGH":
                score += 1

        summary = {
            "parcel_id":        parcel_id,
            "tile_count":       len(tiles),
            "detection_count":  len(all_detections),
            "violation_count":  len(violations),
            "score":            score,
            "risk_label":       risk_label,
            "parse_fails":      parse_fails,
            "detections":       all_detections,
            "violations":       violations,
        }

        out_path = DST_DIR / f"{parcel_id}_summary.json"
        out_path.write_text(json.dumps(summary, indent=2))
        summaries_written += 1

    print(f"Wrote {summaries_written} parcel summary files to {DST_DIR}")
    print("Next: python3 ~/hiz_pipeline/evaluate.py")


if __name__ == "__main__":
    main()
