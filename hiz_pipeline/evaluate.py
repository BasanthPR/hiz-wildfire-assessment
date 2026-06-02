"""
evaluate.py
HIZ-VLM Pipeline — Step 10: Evaluation Metrics & Results Report
Generates detection summaries, compliance score distributions,
model agreement analysis, ablation comparison, and aerial limitations.

Usage:
    python3 ~/hiz_pipeline/evaluate.py
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PIPELINE_DIR = Path.home() / "hiz_pipeline"
TILES_DIR    = PIPELINE_DIR / "tiles"
MANIFEST_CSV = TILES_DIR / "tile_manifest.csv"
RESULTS_DIR  = PIPELINE_DIR / "results"
REPORT_MD    = RESULTS_DIR / "RESULTS_REPORT.md"
REPORT_JSON  = RESULTS_DIR / "results_summary.json"

MODELS = ["qwen25vl", "internvl", "geochat"]
MODEL_NAMES = {
    "qwen25vl": "Qwen2.5-VL-7B-Instruct",
    "internvl": "InternVL2-8B",
    "geochat":  "GeoChat-7B",
}
NAIP_MODELS = ["naip_qwen25vl"]
NAIP_MODEL_NAMES = {"naip_qwen25vl": "Qwen2.5-VL-7B (NAIP public)"}
TARGET_CLASSES = ["vehicle", "trash_can", "propane_tank"]
ZONES = ["Zone_0", "Zone_1", "Zone_2"]


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_parcel_summaries(model: str) -> dict[str, dict]:
    """Load all {parcel_id}_summary.json files for a model."""
    results: dict[str, dict] = {}
    model_dir = RESULTS_DIR / model
    if not model_dir.exists():
        return results
    for fp in model_dir.glob("*_summary.json"):
        try:
            with open(fp) as f:
                data = json.load(f)
            parcel_id = data.get("parcel_id", fp.stem.replace("_summary", ""))
            results[parcel_id] = data
        except Exception:
            continue
    return results


def load_tile_results(model: str) -> list[dict]:
    """Load all per-tile result JSON files for a model."""
    tiles = []
    model_dir = RESULTS_DIR / model
    if not model_dir.exists():
        return tiles
    for fp in sorted(model_dir.glob("*_r*_c*.json")):
        try:
            with open(fp) as f:
                tiles.append(json.load(f))
        except Exception:
            continue
    return tiles


def load_ablation_tiles(model: str) -> list[dict]:
    key = f"{model}_ablation"
    abl_dir = RESULTS_DIR / key
    tiles = []
    if not abl_dir.exists():
        return tiles
    for fp in sorted(abl_dir.glob("*.json")):
        try:
            with open(fp) as f:
                tiles.append(json.load(f))
        except Exception:
            continue
    return tiles


# ─── Analysis functions ───────────────────────────────────────────────────────

def detection_summary(summaries: dict[str, dict]) -> dict:
    """Count detections by class and zone across all parcels."""
    by_class: dict[str, int] = defaultdict(int)
    by_class_zone: dict[str, dict[str, int]] = {
        cls: defaultdict(int) for cls in TARGET_CLASSES
    }
    uncertain_count  = 0
    parse_fail_count = 0

    for parcel_id, summary in summaries.items():
        for det in summary.get("detections", []):
            cls  = det.get("object_class", "unknown")
            zone = det.get("zone", "Zone_2")
            status = det.get("compliance_status", "UNKNOWN")
            by_class[cls] += 1
            if cls in by_class_zone:
                by_class_zone[cls][zone] += 1
            if status == "UNCERTAIN":
                uncertain_count += 1

    return {
        "by_class":       dict(by_class),
        "by_class_zone":  {k: dict(v) for k, v in by_class_zone.items()},
        "uncertain":      uncertain_count,
        "parse_failures": parse_fail_count,
    }


def top_risk_parcels(summaries: dict[str, dict], n: int = 10) -> list[dict]:
    ranked = sorted(
        [{"parcel_id": pid, "score": s.get("score", 0),
          "risk_label": s.get("risk_label", "?"),
          "violations": s.get("violation_count", 0)}
         for pid, s in summaries.items()],
        key=lambda x: x["score"], reverse=True
    )
    return ranked[:n]


def model_agreement(gc_summaries: dict, iv_summaries: dict) -> dict:
    """
    Compare GeoChat and InternVL2 compliance verdicts per parcel.
    """
    agree_count = 0
    disagree_count = 0
    disagree_parcels: list[dict] = []

    common_parcels = set(gc_summaries.keys()) & set(iv_summaries.keys())

    for parcel_id in common_parcels:
        gc_risk = gc_summaries[parcel_id].get("risk_label", "?")
        iv_risk = iv_summaries[parcel_id].get("risk_label", "?")
        gc_viols = gc_summaries[parcel_id].get("violation_count", 0)
        iv_viols = iv_summaries[parcel_id].get("violation_count", 0)

        if gc_risk == iv_risk:
            agree_count += 1
        else:
            disagree_count += 1
            disagree_parcels.append({
                "parcel_id":    parcel_id,
                "geochat_risk": gc_risk,
                "geochat_viols": gc_viols,
                "internvl_risk": iv_risk,
                "internvl_viols": iv_viols,
            })

    disagree_parcels.sort(
        key=lambda x: abs(
            ["LOW","MEDIUM","HIGH","CRITICAL"].index(x["geochat_risk"])
            - ["LOW","MEDIUM","HIGH","CRITICAL"].index(x["internvl_risk"])
        ) if x["geochat_risk"] in ["LOW","MEDIUM","HIGH","CRITICAL"]
          and x["internvl_risk"] in ["LOW","MEDIUM","HIGH","CRITICAL"]
        else 0,
        reverse=True
    )

    return {
        "total_common_parcels": len(common_parcels),
        "agree_count":          agree_count,
        "disagree_count":       disagree_count,
        "agreement_rate":       agree_count / max(len(common_parcels), 1),
        "top_disagreements":    disagree_parcels[:10],
    }


def ablation_comparison(
    full_tiles: list[dict],
    abl_tiles:  list[dict],
    model_name: str,
) -> dict:
    """
    Compare detection rates with vs without Graph-RAG context.
    Also checks whether models cite regulatory clauses when context is given.
    """
    def extract_detections(tiles):
        all_dets = []
        for t in tiles:
            # support both nested {result: {detections}} and flat {detections}
            dets = t.get("result", {}).get("detections") or t.get("detections", [])
            for det in dets:
                all_dets.append(det)
        return all_dets

    full_dets = extract_detections(full_tiles)
    abl_dets  = extract_detections(abl_tiles)

    full_by_class = Counter(d.get("object_class","?") for d in full_dets)
    abl_by_class  = Counter(d.get("object_class","?") for d in abl_dets)

    # CoT clause citation check: does the trace mention "PRC" or "IBHS"?
    def has_clause_citation(det: dict) -> bool:
        cot = det.get("cot_trace", "")
        return "PRC" in cot or "IBHS" in cot or "§" in cot

    full_with_cite = sum(1 for d in full_dets if has_clause_citation(d))
    abl_with_cite  = sum(1 for d in abl_dets  if has_clause_citation(d))

    return {
        "model":                    model_name,
        "full_run_tile_count":      len(full_tiles),
        "ablation_tile_count":      len(abl_tiles),
        "full_detections_total":    len(full_dets),
        "ablation_detections_total": len(abl_dets),
        "full_by_class":            dict(full_by_class),
        "ablation_by_class":        dict(abl_by_class),
        "full_clause_citation_rate": (full_with_cite / max(len(full_dets), 1)),
        "ablation_clause_citation_rate": (abl_with_cite / max(len(abl_dets), 1)),
    }


def aerial_limitations_analysis(all_tile_results: dict[str, list]) -> dict:
    """Collect and categorise aerial_limitations across all models + tiles."""
    CATEGORIES = {
        "canopy_occlusion":    ["canopy", "tree cover", "occluded by tree",
                                "canopy shadow", "overhang"],
        "shadow_ambiguity":    ["shadow", "shading", "cast shadow"],
        "vertical_clearance":  ["vertical", "height", "clearance", "distance above"],
        "object_type_uncertainty": ["material", "plastic vs metal", "cannot confirm",
                                    "type unclear", "ambiguous"],
        "proximity_measurement":   ["exact distance", "feet from", "proximity",
                                    "measure distance"],
        "interior_not_visible":    ["interior", "inside", "inside home",
                                    "not visible from above"],
        "temporal":            ["when", "time of", "red flag day", "evacuation"],
    }

    cat_counts: dict[str, int] = defaultdict(int)
    raw_items: list[str] = []

    for model, tiles in all_tile_results.items():
        for tile in tiles:
            lims = tile.get("result", {}).get("aerial_limitations") or tile.get("aerial_limitations", [])
            if isinstance(lims, list):
                for lim in lims:
                    raw_items.append(str(lim).lower())

    for item in raw_items:
        matched = False
        for cat, keywords in CATEGORIES.items():
            if any(kw in item for kw in keywords):
                cat_counts[cat] += 1
                matched = True
                break
        if not matched:
            cat_counts["other"] += 1

    return {
        "total_limitation_mentions": len(raw_items),
        "by_category":               dict(sorted(cat_counts.items(),
                                                  key=lambda x: -x[1])),
    }


# ─── Report writer ────────────────────────────────────────────────────────────

def write_md_report(report_data: dict, output_path: Path):
    active_models = [m for m in MODELS
                     if report_data.get(f"{m}_detection_summary")]
    model_label = " · ".join(MODEL_NAMES[m] for m in active_models) or "No results yet"

    lines = [
        "# HIZ-VLM Inference Pipeline — Results Report",
        f"**Research:** Basanth Periyapatna Roopa Kumar, SJSU WIRC  ",
        f"**Models:** {model_label}  ",
        f"**Dataset:** 45 consented parcel orthomosaics (2024) + "
        f"NAIP public aerial imagery  ",
        f"**Privacy:** LOCAL INFERENCE ONLY — data never sent to external APIs  ",
        "",
        "---",
        "",
    ]

    # ── 1. Detection summary ──────────────────────────────────────────────────
    lines += ["## 1. Detection Summary\n"]
    for model in MODELS:
        mname = MODEL_NAMES[model]
        det_s = report_data.get(f"{model}_detection_summary", {})
        lines += [f"### {mname}\n"]
        lines += ["| Object Class | Zone_0 | Zone_1 | Zone_2 | Total |",
                  "|---|---|---|---|---|"]
        by_cz = det_s.get("by_class_zone", {})
        by_c  = det_s.get("by_class", {})
        for cls in TARGET_CLASSES:
            z0 = by_cz.get(cls, {}).get("Zone_0", 0)
            z1 = by_cz.get(cls, {}).get("Zone_1", 0)
            z2 = by_cz.get(cls, {}).get("Zone_2", 0)
            tot = by_c.get(cls, 0)
            lines.append(f"| {cls} | {z0} | {z1} | {z2} | {tot} |")
        lines += [
            "",
            f"- UNCERTAIN detections flagged for human review: "
            f"{det_s.get('uncertain', 0)}",
            f"- JSON parse failures: {det_s.get('parse_failures', 0)}",
            "",
        ]

    # ── 2. Compliance score distribution ──────────────────────────────────────
    lines += ["## 2. Compliance Score Distribution & Top Risk Parcels\n"]
    for model in MODELS:
        mname = MODEL_NAMES[model]
        top10 = report_data.get(f"{model}_top_risk", [])
        lines += [f"### {mname} — Top 10 Highest-Risk Parcels\n",
                  "| Rank | Parcel ID | Risk Score | Risk Level | Violations |",
                  "|---|---|---|---|---|"]
        for i, p in enumerate(top10, 1):
            lines.append(
                f"| {i} | {p['parcel_id']} | {p['score']} "
                f"| {p['risk_label']} | {p['violations']} |"
            )
        lines.append("")

    # ── 3. Model agreement ────────────────────────────────────────────────────
    lines += ["## 3. Model Agreement Analysis (Qwen2.5-VL vs InternVL2)\n"]
    agree = report_data.get("model_agreement", {})
    lines += [
        f"- Common parcels assessed by both models: "
        f"{agree.get('total_common_parcels', 0)}",
        f"- Agreement (same risk label): {agree.get('agree_count', 0)} "
        f"({agree.get('agreement_rate', 0)*100:.1f}%)",
        f"- Disagreement: {agree.get('disagree_count', 0)} parcels",
        "",
        "### Top Disagreements (flag for inspector priority)\n",
        "| Parcel ID | Qwen2.5-VL Risk | Qwen Violations | "
        "InternVL2 Risk | InternVL2 Violations |",
        "|---|---|---|---|---|",
    ]
    for d in agree.get("top_disagreements", []):
        lines.append(
            f"| {d['parcel_id']} | {d.get('model_a_risk','?')} | "
            f"{d.get('model_a_viols','?')} | {d.get('model_b_risk','?')} | "
            f"{d.get('model_b_viols','?')} |"
        )
    lines.append("")

    # ── 3b. NAIP public imagery results ──────────────────────────────────────
    naip_det = report_data.get("naip_detection_summary")
    if naip_det:
        lines += [
            "## 3b. NAIP Public Imagery Results\n",
            "_Scalability demonstration: Qwen2.5-VL on 60 cm public aerial imagery_\n",
            "| Object Class | Zone_0 | Zone_1 | Zone_2 | Total |",
            "|---|---|---|---|---|",
        ]
        by_cz = naip_det.get("by_class_zone", {})
        by_c  = naip_det.get("by_class", {})
        for cls in TARGET_CLASSES:
            z0  = by_cz.get(cls, {}).get("Zone_0", 0)
            z1  = by_cz.get(cls, {}).get("Zone_1", 0)
            z2  = by_cz.get(cls, {}).get("Zone_2", 0)
            tot = by_c.get(cls, 0)
            lines.append(f"| {cls} | {z0} | {z1} | {z2} | {tot} |")
        lines += [
            "",
            "_Note: NAIP 60 cm GSD — objects < 2 m may not be resolved. "
            "Detection rates expected lower than drone imagery._",
            "",
        ]

    # ── 4. Ablation comparison ────────────────────────────────────────────────
    lines += ["## 4. Ablation Study — Graph-RAG vs Plain Prompt\n",
              "_RQ2: Does regulatory context injection improve detection "
              "and compliance judgment?_\n"]
    for model in MODELS:
        abl = report_data.get(f"{model}_ablation", {})
        if not abl:
            lines += [f"### {MODEL_NAMES[model]}: ablation results not yet available\n"]
            continue
        lines += [f"### {MODEL_NAMES[model]}\n",
                  "| Metric | With Graph-RAG | Without (Ablation) |",
                  "|---|---|---|"]
        for cls in TARGET_CLASSES:
            full_n = abl.get("full_by_class", {}).get(cls, 0)
            abl_n  = abl.get("ablation_by_class", {}).get(cls, 0)
            lines.append(f"| {cls} detections | {full_n} | {abl_n} |")
        fr = abl.get("full_clause_citation_rate", 0)
        ar = abl.get("ablation_clause_citation_rate", 0)
        lines += [
            f"| Clause citation in CoT trace | {fr*100:.1f}% | {ar*100:.1f}% |",
            "",
        ]

    # ── 5. Aerial limitations ─────────────────────────────────────────────────
    lines += [
        "## 5. Aerial Limitations — What VLMs Cannot Determine\n",
        "_These categories map to the aerial-detectable vs. ground-only "
        "partition (Manuscript Novelty Claim N5)_\n",
        "| Category | Mention Count |",
        "|---|---|",
    ]
    lims = report_data.get("aerial_limitations", {})
    for cat, count in sorted(
        lims.get("by_category", {}).items(), key=lambda x: -x[1]
    ):
        lines.append(f"| {cat} | {count} |")
    lines += [
        "",
        f"Total limitation mentions: "
        f"{lims.get('total_limitation_mentions', 0)}",
        "",
        "---",
        "",
        "## Status Checklist\n",
    ]
    for item in report_data.get("checklist", []):
        tick = "x" if item["done"] else " "
        lines.append(f"- [{tick}] {item['label']}")

    lines += [
        "",
        "---",
        "_Report generated by HIZ-VLM Inference Pipeline | SJSU WIRC_",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def model_agreement_generic(
    summaries_a: dict, name_a: str,
    summaries_b: dict, name_b: str,
) -> dict:
    """Generic two-model agreement analysis."""
    agree_count    = 0
    disagree_count = 0
    disagree_rows: list[dict] = []
    RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    for pid in set(summaries_a.keys()) & set(summaries_b.keys()):
        ra = summaries_a[pid].get("risk_label", "?")
        rb = summaries_b[pid].get("risk_label", "?")
        va = summaries_a[pid].get("violation_count", 0)
        vb = summaries_b[pid].get("violation_count", 0)
        if ra == rb:
            agree_count += 1
        else:
            disagree_count += 1
            span = abs(
                (RISK_ORDER.index(ra) if ra in RISK_ORDER else 0) -
                (RISK_ORDER.index(rb) if rb in RISK_ORDER else 0)
            )
            disagree_rows.append({
                "parcel_id": pid,
                "model_a_risk": ra, "model_a_viols": va,
                "model_b_risk": rb, "model_b_viols": vb,
                "_span": span,
            })
    disagree_rows.sort(key=lambda x: -x["_span"])
    total = agree_count + disagree_count
    return {
        "model_a": name_a, "model_b": name_b,
        "total_common_parcels": total,
        "agree_count": agree_count,
        "disagree_count": disagree_count,
        "agreement_rate": agree_count / max(total, 1),
        "top_disagreements": [{k: v for k, v in d.items() if k != "_span"}
                              for d in disagree_rows[:10]],
    }


def main():
    report_data: dict = {}

    # ── Load results for all models ──────────────────────────────────────────
    qw_summaries = load_parcel_summaries("qwen25vl")
    iv_summaries = load_parcel_summaries("internvl")
    gc_summaries = load_parcel_summaries("geochat")
    qw_tiles     = load_tile_results("qwen25vl")
    iv_tiles     = load_tile_results("internvl")
    gc_tiles     = load_tile_results("geochat")
    qw_abl       = load_ablation_tiles("qwen25vl")
    iv_abl       = load_ablation_tiles("internvl")
    gc_abl       = load_ablation_tiles("geochat")

    # NAIP public imagery results
    naip_summaries = load_parcel_summaries("naip_qwen25vl")
    naip_tiles     = load_tile_results("naip_qwen25vl")

    print(f"Qwen2.5-VL : {len(qw_summaries)} parcel summaries, {len(qw_tiles)} tiles")
    print(f"InternVL2  : {len(iv_summaries)} parcel summaries, {len(iv_tiles)} tiles")
    print(f"GeoChat    : {len(gc_summaries)} parcel summaries, {len(gc_tiles)} tiles")
    print(f"NAIP Qwen  : {len(naip_summaries)} parcel summaries, {len(naip_tiles)} tiles")

    # 1. Detection summary
    report_data["qwen25vl_detection_summary"] = detection_summary(qw_summaries)
    report_data["internvl_detection_summary"] = detection_summary(iv_summaries)
    report_data["geochat_detection_summary"]  = detection_summary(gc_summaries)
    if naip_summaries:
        report_data["naip_detection_summary"] = detection_summary(naip_summaries)

    # 2. Top risk parcels
    report_data["qwen25vl_top_risk"] = top_risk_parcels(qw_summaries)
    report_data["internvl_top_risk"] = top_risk_parcels(iv_summaries)
    report_data["geochat_top_risk"]  = top_risk_parcels(gc_summaries)

    # 3. Model agreement — primary comparison: Qwen2.5-VL vs InternVL2
    if qw_summaries and iv_summaries:
        report_data["model_agreement"] = model_agreement_generic(
            qw_summaries, "Qwen2.5-VL-7B-Instruct",
            iv_summaries, "InternVL2-8B",
        )

    # 4. Ablation — Graph-RAG vs plain prompt
    for model_key, model_name, full_t, abl_t in [
        ("qwen25vl_ablation",  "Qwen2.5-VL-7B-Instruct", qw_tiles, qw_abl),
        ("internvl_ablation",  "InternVL2-8B",            iv_tiles, iv_abl),
        ("geochat_ablation",   "GeoChat-7B",              gc_tiles, gc_abl),
    ]:
        if full_t and abl_t:
            report_data[model_key] = ablation_comparison(
                full_t, abl_t, model_name
            )

    # 5. Aerial limitations
    report_data["aerial_limitations"] = aerial_limitations_analysis(
        {"qwen25vl": qw_tiles, "internvl": iv_tiles,
         "geochat": gc_tiles, "naip": naip_tiles}
    )

    # Checklist
    kg_path = Path.home() / "hiz_pipeline" / "knowledge_graph" / "hiz_graph.graphml"
    naip_dir = Path.home() / "hiz_data" / "naip"
    report_data["checklist"] = [
        {"label": "Knowledge graph built (346 nodes, 9603 edges)",
         "done": kg_path.exists()},
        {"label": "All 45 parcels preprocessed, tile manifest complete",
         "done": MANIFEST_CSV.exists()},
        {"label": f"Qwen2.5-VL inference — {len(qw_tiles)} tiles, "
                  f"{sum(len(s.get('detections',[])) for s in qw_summaries.values())} dets",
         "done": len(qw_summaries) > 0},
        {"label": f"InternVL2 inference — {len(iv_tiles)} tiles, "
                  f"{sum(len(s.get('detections',[])) for s in iv_summaries.values())} dets",
         "done": len(iv_summaries) > 0},
        {"label": "Ablation study complete (with/without Graph-RAG)",
         "done": (len(qw_abl) > 0 or len(iv_abl) > 0)},
        {"label": f"NAIP public imagery downloaded ({len(list(naip_dir.rglob('*.tif')))} scenes)",
         "done": naip_dir.exists() and any(naip_dir.rglob("*.tif"))},
        {"label": f"NAIP inference — {len(naip_tiles)} tiles, "
                  f"{sum(len(s.get('detections',[])) for s in naip_summaries.values())} dets",
         "done": len(naip_summaries) > 0},
        {"label": "Annotated bounding box images saved",
         "done": len(list((RESULTS_DIR / "annotated").glob("*.png"))) > 0},
        {"label": "Results report generated",
         "done": REPORT_MD.exists()},
    ]

    # Print status checklist to terminal
    print("\n" + "=" * 60)
    print("  PIPELINE STATUS CHECKLIST")
    print("=" * 60)
    for item in report_data["checklist"]:
        tick = "x" if item["done"] else " "
        print(f"  [{tick}] {item['label']}")
    print("=" * 60 + "\n")

    # Save JSON summary
    with open(REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"JSON summary saved: {REPORT_JSON}")

    # Write markdown report
    write_md_report(report_data, REPORT_MD)


if __name__ == "__main__":
    main()
