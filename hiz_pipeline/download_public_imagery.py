"""
download_public_imagery.py
HIZ-VLM Pipeline — Public Imagery Acquisition

Downloads NAIP (National Agriculture Imagery Program) tiles for the 5 HIZ
research sites from Microsoft Planetary Computer (free, no auth required).
NAIP 2022 CA: 0.6 m GSD true-color orthomosaic.

Sites:
  fel  — Felton, Santa Cruz County
  red  — Redwood Estates, Santa Cruz County
  sar  — Saratoga, Santa Clara County
  par  — Paradise, Butte County (Camp Fire WUI)
  tah  — South Lake Tahoe, El Dorado County

Outputs: ~/hiz_data/naip/{site}/  — GeoTIFF tiles, one per NAIP scene cell

Usage:
    python3 ~/hiz_pipeline/download_public_imagery.py [--site fel]
    python3 ~/hiz_pipeline/download_public_imagery.py           # all sites
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

# ── Site bounding boxes (WGS-84: min_lon, min_lat, max_lon, max_lat) ──────────
# Bounding boxes derived from study-site parcel centroids.
SITE_BBOXES = {
    "fel": [-122.08, 37.03, -122.02, 37.09],   # Felton, Santa Cruz Mtns
    "red": [-122.04, 37.05, -121.98, 37.11],   # Redwood Estates
    "sar": [-122.08, 37.23, -122.01, 37.29],   # Saratoga foothills
    "par": [-121.64, 39.74, -121.56, 39.81],   # Paradise / Camp Fire WUI
    "tah": [-120.08, 38.92, -120.00, 38.98],   # South Lake Tahoe WUI
}

NAIP_OUTPUT = Path.home() / "hiz_data" / "naip"
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_COLLECTION = "naip"
TARGET_YEAR = "2022"


def stac_search(bbox, year=TARGET_YEAR):
    """Search Planetary Computer NAIP catalog for items covering bbox."""
    endpoint = f"{PC_STAC_URL}/search"
    payload = {
        "collections": [NAIP_COLLECTION],
        "bbox": bbox,
        "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
        "limit": 10,
        "query": {"naip:state": {"eq": "ca"}},
    }
    resp = requests.post(endpoint, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("features", [])


def sign_url(href: str) -> str:
    """Get a signed (time-limited) download URL from Planetary Computer."""
    sign_endpoint = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
    resp = requests.get(sign_endpoint, params={"href": href}, timeout=15)
    if resp.ok:
        return resp.json().get("href", href)
    return href


def download_file(url: str, dest: Path, label: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {label} → {dest.name} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        dl = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                dl += len(chunk)
                if total:
                    pct = dl / total * 100
                    print(f"\r    {pct:5.1f}%  {dl/1e6:.1f}/{total/1e6:.1f} MB",
                          end="", flush=True)
    print(f"\r    Done ({dl/1e6:.1f} MB)            ")


def process_site(site: str):
    bbox = SITE_BBOXES[site]
    out_dir = NAIP_OUTPUT / site
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "stac_items.json"

    print(f"\n{'─'*55}")
    print(f"Site: {site.upper()} | bbox: {bbox}")

    items = stac_search(bbox)
    if not items:
        # Try 2020 as fallback
        items = stac_search(bbox, year="2020")
        if not items:
            print(f"  No NAIP items found for {site}. Check bounding box.")
            return

    print(f"  Found {len(items)} NAIP scene(s)")
    json.dump(items, open(meta_path, "w"), indent=2)

    for item in items:
        item_id = item.get("id", "unknown")
        assets  = item.get("assets", {})

        # Prefer 'image' asset (the raw GeoTIFF); fallback to 'thumbnail'
        for asset_key in ("image", "rendered_preview", "thumbnail"):
            asset = assets.get(asset_key)
            if asset:
                href = asset.get("href", "")
                if not href:
                    continue
                # Sign the URL
                signed = sign_url(href)
                ext = ".tif" if "image" in asset_key else ".jpg"
                dest = out_dir / f"{item_id}_{asset_key}{ext}"
                try:
                    download_file(signed, dest, f"{item_id}/{asset_key}")
                except Exception as e:
                    print(f"  Download failed: {e}")
                if asset_key == "image":
                    break   # only need the full GeoTIFF


def main():
    parser = argparse.ArgumentParser(
        description="Download NAIP public imagery for HIZ research sites"
    )
    parser.add_argument(
        "--site", choices=list(SITE_BBOXES.keys()),
        help="Download only this site (default: all)"
    )
    parser.add_argument(
        "--year", default=TARGET_YEAR, help="NAIP year (default: 2022)"
    )
    args = parser.parse_args()

    sites = [args.site] if args.site else list(SITE_BBOXES.keys())
    print(f"NAIP Public Imagery Download")
    print(f"Sites: {', '.join(sites)} | Year: {args.year}")
    print(f"Output: {NAIP_OUTPUT}")

    for site in sites:
        try:
            process_site(site)
        except Exception as e:
            print(f"  ERROR processing {site}: {e}")

    print(f"\nDone. Files in {NAIP_OUTPUT}")
    print("Next: python3 ~/hiz_pipeline/preprocess_naip.py")


if __name__ == "__main__":
    main()
