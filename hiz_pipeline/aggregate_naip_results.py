"""
aggregate_naip_results.py
Builds per-parcel summary files from naip_qwen25vl per-tile JSONs
so evaluate.py can load them via load_parcel_summaries("naip_qwen25vl").

Run: python3 ~/hiz_pipeline/aggregate_naip_results.py
"""

import json
from collections import defaultdict
from pathlib import Path

PIPELINE_DIR = Path.home() / "hiz_pipeline"
SRC_DIR      = PIPELINE_DIR / "results" / "naip_qwen25vl"
DST_DIR      = SRC_DIR  # summaries go in the same folder

RISK_SCORE = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
SCORE_RISK = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "UNKNOWN"}


def main():
    tile_files = sorted(f for f in SRC_DIR.glob("*.json") if "_summary" not in f.name)
    print(f"Loading {len(tile_files)} NAIP tile files ...")

    parcel_tiles: dict[str, list[dict]] = defaultdict(list)
    for fp in tile_files:
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        parcel_id = data.get("parcel_id", fp.stem)
        parcel_tiles[parcel_id].append(data)

    print(f"Found {len(parcel_tiles)} NAIP parcels/scenes.")

    for parcel_id, tiles in parcel_tiles.items():
        all_detections, all_risks, parse_fails = [], [], 0
        for tile in tiles:
            all_detections.extend(tile.get("detections", []))
            all_risks.append(RISK_SCORE.get(tile.get("overall_parcel_risk", "UNKNOWN"), 0))
            if tile.get("parse_failed"):
                parse_fails += 1

        violations = [d for d in all_detections if d.get("compliance_status") == "VIOLATION"]
        max_risk   = max(all_risks) if all_risks else 0
        score      = len(violations)
        for v in violations:
            if v.get("severity") == "CRITICAL":
                score += 2
            elif v.get("severity") == "HIGH":
                score += 1

        summary = {
            "parcel_id":       parcel_id,
            "tile_count":      len(tiles),
            "detection_count": len(all_detections),
            "violation_count": len(violations),
            "score":           score,
            "risk_label":      SCORE_RISK.get(max_risk, "UNKNOWN"),
            "parse_fails":     parse_fails,
            "detections":      all_detections,
            "violations":      violations,
        }
        (DST_DIR / f"{parcel_id}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote {len(parcel_tiles)} NAIP summary files to {DST_DIR}")
    print("Next: python3 ~/hiz_pipeline/evaluate.py")


if __name__ == "__main__":
    main()
