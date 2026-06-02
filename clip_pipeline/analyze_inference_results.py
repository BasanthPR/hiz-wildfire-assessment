"""
analyze_inference_results.py
=============================
Manuscript-quality analysis of CLIP inference results.

Reads   vlm_inference_results.json
Writes  vlm_inference_analysis.txt   — prose summary for manuscript
        vlm_inference_figures.xlsx   — all tables ready for paper

Key analyses
------------
1. Overall detection and violation rates
2. Aerial-detectable vs. ground-only violation split (Graph-RAG driven)
3. Per-zone violation distributions (Zone_0 vs Zone_1)
4. Per-site risk profiles across 5 study areas
5. Object class frequency ranked by IBHS severity
6. IBHS requirement coverage across detected objects

Run
---
  python3 clip_pipeline/analyze_inference_results.py
"""

import json, os, sys
import pandas as pd
import numpy as np
from collections import defaultdict

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR  = os.path.dirname(BASE_DIR)
KG_DIR    = os.path.join(REPO_DIR, "knowledge_graph")
sys.path.insert(0, KG_DIR)
sys.path.insert(0, BASE_DIR)
from graph_rag_lookup import get_regulatory_context
from vlm_inference_pipeline import OBJECT_CLASSES

IN_JSON   = os.path.join(BASE_DIR, "vlm_inference_results.json")
OUT_TXT   = os.path.join(BASE_DIR, "vlm_inference_analysis.txt")
OUT_XLSX  = os.path.join(BASE_DIR, "vlm_inference_figures.xlsx")

SITES = {
    "fel": "Felton (SCU)",
    "par": "Paradise",
    "red": "Red Zone",
    "sar": "Santa Rosa",
    "tah": "Tahoe Donner",
}

SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0, "UNKNOWN": 0}

# ── Load results ──────────────────────────────────────────────────────────────
with open(IN_JSON) as f:
    chips = json.load(f)

print(f"Loaded {len(chips)} chip records")

# ── Pre-fetch Graph-RAG context for all 33 classes × 2 zones ─────────────────
# (avoids redundant lookups in the loop below)
rag_cache = {}
for obj in OBJECT_CLASSES:
    for zone in ("Zone_0", "Zone_1"):
        rag_cache[(obj, zone)] = get_regulatory_context(obj, zone)

# ── Build flat findings table ──────────────────────────────────────────────────
rows = []
for chip in chips:
    parcel  = chip["parcel_id"]
    site    = chip["site"]
    zone    = chip["zone"]
    row_off = chip["row_offset"]
    col_off = chip["col_offset"]
    for obj in chip["detected_objects"]:
        ctx = rag_cache.get((obj, zone), {})
        aerial = ctx.get("aerial_detectable", False)
        viols  = ctx.get("violations", [])
        ibhs   = ctx.get("ibhs_requirements", [])
        sev    = viols[0]["severity"] if viols else "NONE"
        rows.append({
            "parcel_id"        : parcel,
            "site"             : site,
            "site_name"        : SITES.get(site, site),
            "zone"             : zone,
            "row_off"          : row_off,
            "col_off"          : col_off,
            "object_class"     : obj,
            "aerial_detectable": bool(aerial),
            "severity"         : sev,
            "severity_val"     : SEVERITY_ORDER.get(sev, 0),
            "n_ibhs_reqs"      : len(ibhs),
            "has_violation"    : bool(viols),
        })

df = pd.DataFrame(rows)
print(f"Findings rows: {len(df)}")

# ── Analysis 1: Overall stats ─────────────────────────────────────────────────
total_chips   = len(chips)
det_chips     = sum(1 for c in chips if c["detected_objects"])
viol_chips    = sum(1 for c in chips if c["has_violations"])
total_finds   = len(df)
aerial_finds  = df["aerial_detectable"].sum()
ground_finds  = (~df["aerial_detectable"]).sum()

# ── Analysis 2: Per-zone breakdown ───────────────────────────────────────────
zone_stats = df.groupby("zone").agg(
    findings        = ("object_class", "count"),
    aerial_pct      = ("aerial_detectable", lambda x: 100 * x.mean()),
    critical_count  = ("severity", lambda x: (x == "CRITICAL").sum()),
    high_count      = ("severity", lambda x: (x == "HIGH").sum()),
    unique_objects  = ("object_class", "nunique"),
).reset_index()

# Zone chip counts
zone_chip_counts = pd.Series({z: sum(1 for c in chips if c["zone"] == z) for z in ("Zone_0", "Zone_1")})

# ── Analysis 3: Per-site risk profile ────────────────────────────────────────
site_chip_counts = pd.Series({s: sum(1 for c in chips if c["site"] == s) for s in SITES})
site_viol_counts = pd.Series({s: sum(1 for c in chips if c["site"] == s and c["has_violations"]) for s in SITES})

site_stats = df.groupby("site").agg(
    findings        = ("object_class", "count"),
    aerial_pct      = ("aerial_detectable", lambda x: 100 * x.mean()),
    critical_count  = ("severity", lambda x: (x == "CRITICAL").sum()),
    high_count      = ("severity", lambda x: (x == "HIGH").sum()),
    unique_objects  = ("object_class", "nunique"),
).reset_index()
site_stats["site_name"]     = site_stats["site"].map(SITES)
site_stats["total_chips"]   = site_stats["site"].map(site_chip_counts)
site_stats["viol_chips"]    = site_stats["site"].map(site_viol_counts)
site_stats["violation_rate"]= (site_stats["viol_chips"] / site_stats["total_chips"] * 100).round(1)
site_stats["finds_per_chip"]= (site_stats["findings"] / site_stats["total_chips"]).round(2)
site_stats = site_stats.sort_values("violation_rate", ascending=False)

# ── Analysis 4: Object class frequency table ──────────────────────────────────
obj_stats = df.groupby("object_class").agg(
    chip_count      = ("object_class", "count"),
    aerial_detectable = ("aerial_detectable", "first"),
    severity        = ("severity", "first"),
    n_ibhs_reqs     = ("n_ibhs_reqs", "first"),
    sites           = ("site", lambda x: ", ".join(sorted(x.unique()))),
).reset_index()
obj_stats["chip_pct"] = (obj_stats["chip_count"] / total_chips * 100).round(1)
obj_stats["severity_val"] = obj_stats["severity"].map(SEVERITY_ORDER)
obj_stats = obj_stats.sort_values(["severity_val", "chip_count"], ascending=[False, False])

# ── Analysis 5: Aerial vs. ground-only violation split per site ───────────────
aerial_site = df[df["aerial_detectable"]].groupby("site")["object_class"].count().rename("aerial_findings")
ground_site = df[~df["aerial_detectable"]].groupby("site")["object_class"].count().rename("ground_findings")
aerial_ground = pd.concat([aerial_site, ground_site], axis=1).fillna(0).astype(int)
aerial_ground["aerial_pct"] = (aerial_ground["aerial_findings"] /
                                (aerial_ground["aerial_findings"] + aerial_ground["ground_findings"]) * 100).round(1)
aerial_ground["site_name"] = aerial_ground.index.map(SITES)

# ── Analysis 6: Severity distribution per zone ────────────────────────────────
sev_zone = df.groupby(["zone", "severity"]).size().unstack(fill_value=0)
for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
    if sev not in sev_zone.columns:
        sev_zone[sev] = 0
sev_zone = sev_zone[["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]]

# ── Write analysis text ───────────────────────────────────────────────────────
lines = []
w = lines.append

w("=" * 72)
w("HIZ WILDFIRE AERIAL INFERENCE — RESULTS ANALYSIS")
w(f"Dataset: 45 parcels, 5 sites | Chips: {total_chips:,} | Threshold: 0.285")
w(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
w("=" * 72)

w("")
w("── 1. OVERALL DETECTION SUMMARY ─────────────────────────────────────────")
w(f"  Total 512×512 chips processed : {total_chips:,}")
w(f"  Chips with ≥1 CLIP detection  : {det_chips:,}  ({100*det_chips/total_chips:.1f}%)")
w(f"  Chips with ≥1 violation       : {viol_chips:,}  ({100*viol_chips/total_chips:.1f}%)")
w(f"  Total object-level findings   : {total_finds:,}")
w(f"  Aerial-detectable findings    : {aerial_finds:,}  ({100*aerial_finds/total_finds:.1f}%)")
w(f"  Ground-inspection-only        : {ground_finds:,}  ({100*ground_finds/total_finds:.1f}%)")
w(f"  Mean detections per chip      : {total_finds/total_chips:.2f}")

w("")
w("── 2. AERIAL vs. GROUND-ONLY VIOLATION SPLIT ────────────────────────────")
w(f"  Of {total_finds:,} total object detections with regulatory implications:")
w(f"    {aerial_finds:,} ({100*aerial_finds/total_finds:.1f}%) are aerial-detectable")
w(f"      → can be assessed from drone/satellite imagery alone")
w(f"    {ground_finds:,} ({100*ground_finds/total_finds:.1f}%) require ground inspection")
w(f"      → not reliably visible from nadir view")
w("")
w("  Per-site aerial detection fraction:")
for _, row in aerial_ground.iterrows():
    w(f"    {row['site_name']:20s}  aerial={row['aerial_findings']:4.0f}  "
      f"ground={row['ground_findings']:4.0f}  aerial%={row['aerial_pct']:.1f}%")

w("")
w("── 3. ZONE DISTRIBUTION ─────────────────────────────────────────────────")
for _, row in zone_stats.iterrows():
    zchips = zone_chip_counts.get(row["zone"], 1)
    w(f"  {row['zone']}:")
    w(f"    Chips processed  : {zchips:,}")
    w(f"    Findings         : {row['findings']:,}  ({row['findings']/zchips:.2f}/chip)")
    w(f"    Aerial-detectable: {row['aerial_pct']:.1f}%")
    w(f"    CRITICAL findings: {row['critical_count']:,}")
    w(f"    HIGH findings    : {row['high_count']:,}")

w("")
w("── 4. PER-SITE RISK PROFILE ─────────────────────────────────────────────")
w(f"  {'Site':<22} {'Chips':>6} {'Viol%':>6} {'Finds/chip':>10} {'Aerial%':>8} {'CRIT':>6} {'Classes':>8}")
w(f"  {'-'*70}")
for _, row in site_stats.iterrows():
    w(f"  {row['site_name']:<22} {row['total_chips']:>6} {row['violation_rate']:>5.1f}% "
      f"{row['finds_per_chip']:>10.2f} {row['aerial_pct']:>7.1f}% "
      f"{row['critical_count']:>6} {row['unique_objects']:>8}")

w("")
w("── 5. TOP DETECTED OBJECT CLASSES (by IBHS severity then frequency) ─────")
w(f"  {'Object Class':<32} {'Chips':>6} {'%':>5} {'Aerial':>7} {'Severity':>10} {'IBHS reqs':>10}")
w(f"  {'-'*75}")
for _, row in obj_stats.head(20).iterrows():
    aer = "YES" if row["aerial_detectable"] else "no"
    w(f"  {row['object_class']:<32} {row['chip_count']:>6} {row['chip_pct']:>4.1f}% "
      f"{aer:>7} {row['severity']:>10} {row['n_ibhs_reqs']:>10}")

w("")
w("── 6. SEVERITY DISTRIBUTION BY ZONE ─────────────────────────────────────")
for zone in sev_zone.index:
    row = sev_zone.loc[zone]
    total = row.sum()
    w(f"  {zone}: CRITICAL={row.get('CRITICAL',0)} ({100*row.get('CRITICAL',0)/max(total,1):.0f}%)  "
      f"HIGH={row.get('HIGH',0)} ({100*row.get('HIGH',0)/max(total,1):.0f}%)  "
      f"MEDIUM={row.get('MEDIUM',0)} ({100*row.get('MEDIUM',0)/max(total,1):.0f}%)")

w("")
w("── 7. MANUSCRIPT-RELEVANT CLAIMS ────────────────────────────────────────")
aerial_pct = 100 * aerial_finds / total_finds
top_aerial  = obj_stats[obj_stats["aerial_detectable"]].head(3)["object_class"].tolist()
top_ground  = obj_stats[~obj_stats["aerial_detectable"]].head(3)["object_class"].tolist()
top_site    = site_stats.iloc[0]["site_name"]
top_viol    = site_stats.iloc[0]["violation_rate"]
w(f"  N5 (aerial detectability): {aerial_pct:.1f}% of violation findings are aerial-detectable,")
w(f"      supporting automated compliance screening from drone imagery.")
w(f"      Top aerial-detectable classes: {', '.join(top_aerial)}")
w(f"      Ground-only classes include:  {', '.join(top_ground)}")
w(f"  Risk ranking: {top_site} shows highest violation rate ({top_viol:.1f}%),")
top_classes = obj_stats.head(3)["object_class"].tolist()
w(f"      driven primarily by {', '.join(top_classes)}.")

w("")
w("=" * 72)

report = "\n".join(lines)
with open(OUT_TXT, "w") as f:
    f.write(report)
print(report)

# ── Write figures Excel ───────────────────────────────────────────────────────
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:

    # Table 1: Site summary
    t1 = site_stats[["site_name", "total_chips", "viol_chips", "violation_rate",
                      "finds_per_chip", "aerial_pct", "critical_count", "unique_objects"]]
    t1.columns = ["Site", "Chips", "Viol Chips", "Viol Rate %",
                  "Findings/chip", "Aerial %", "CRITICAL", "Unique Classes"]
    t1.to_excel(xw, sheet_name="T1 Site Summary", index=False)

    # Table 2: Object class frequency
    t2 = obj_stats[["object_class", "chip_count", "chip_pct", "aerial_detectable",
                     "severity", "n_ibhs_reqs", "sites"]]
    t2.columns = ["Object Class", "Chip Count", "Chip %", "Aerial Detectable",
                  "Max Severity", "IBHS Reqs", "Sites Detected"]
    t2.to_excel(xw, sheet_name="T2 Object Classes", index=False)

    # Table 3: Zone breakdown
    t3 = zone_stats.copy()
    t3["total_chips"] = t3["zone"].map(zone_chip_counts)
    t3["finds_per_chip"] = (t3["findings"] / t3["total_chips"]).round(2)
    t3.columns = ["Zone", "Findings", "Aerial %", "CRITICAL", "HIGH",
                  "Unique Classes", "Total Chips", "Findings/Chip"]
    t3.to_excel(xw, sheet_name="T3 Zone Breakdown", index=False)

    # Table 4: Aerial vs ground per site
    t4 = aerial_ground.reset_index()
    t4.columns = ["Site", "Aerial Findings", "Ground Findings", "Aerial %", "Site Name"]
    t4.to_excel(xw, sheet_name="T4 Aerial vs Ground", index=False)

    # Table 5: Severity by zone
    sev_zone.reset_index().to_excel(xw, sheet_name="T5 Severity by Zone", index=False)

    # Table 6: Raw chip-level summary
    chip_df = pd.DataFrame([{
        "parcel_id"      : c["parcel_id"],
        "site"           : c["site"],
        "zone"           : c["zone"],
        "row_offset"     : c["row_offset"],
        "col_offset"     : c["col_offset"],
        "detected_objects": ", ".join(c["detected_objects"]),
        "n_detected"     : len(c["detected_objects"]),
        "has_violations" : c["has_violations"],
        "max_severity"   : c["max_severity"],
        "clip_top5"      : c.get("clip_top5", ""),
    } for c in chips])
    chip_df.to_excel(xw, sheet_name="T6 All Chips", index=False)

print(f"\n✓  {OUT_TXT}")
print(f"✓  {OUT_XLSX}  ({len(xw.sheets) if hasattr(xw,'sheets') else 6} sheets)")
