"""
graph_rag_lookup.py
====================
RAG-ready lookup for the HIZ Wildfire Knowledge Graph.
Usage:
    from graph_rag_lookup import get_regulatory_context
    ctx = get_regulatory_context("propane", "Zone_0")
    print(ctx["vlm_prompt"])
"""

import json, os

_BASE = os.path.dirname(os.path.abspath(__file__))
_NODES_F = os.path.join(_BASE, "knowledge_graph_nodes.json")
_EDGES_F = os.path.join(_BASE, "knowledge_graph_edges.json")

_nodes, _edges = None, None

def _load():
    global _nodes, _edges
    if _nodes is None:
        with open(_NODES_F) as f:
            raw = json.load(f)
            _nodes = {n["id"]: n for n in raw}
    if _edges is None:
        with open(_EDGES_F) as f:
            _edges = json.load(f)

# Alias resolver
_ALIASES = {
    "firewood": "woodpile", "wood pile": "woodpile", "wood stack": "woodpile",
    "propane tank": "propane", "lpg": "propane", "gas tank": "propane",
    "trash bin": "garbage_bin", "garbage can": "garbage_bin", "recycle bin": "garbage_bin",
    "hot tub": "above_ground_pool_or_hot_tub", "spa": "above_ground_pool_or_hot_tub",
    "shed": "storage_shed", "outbuilding": "storage_shed",
    "patio furniture": "patio_furniture", "chair": "patio_furniture",
    "deck": "deck_patio", "patio": "deck_patio",
    "pergola": "pergola_gazebo", "gazebo": "pergola_gazebo", "carport": "pergola_gazebo",
    "dead veg": "dead_vegetation", "debris": "dead_vegetation",
    "herb": "live_herb", "grass": "live_herb", "weed": "live_herb",
    "shrub": "live_shrub", "bush": "live_shrub",
    "tree": "live_tree",
    "bbq": "bbq_grill", "grill": "bbq_grill",
    "wood chip mulch": "mulch", "bark mulch": "mulch",
    "vehicle": "car", "automobile": "car",
    "playset": "play_set", "swing set": "play_set",
    "door mat": "welcome_mat",
    "potted plant": "planters", "planter": "planters",
}

ZONE_RANGES = {
    "Zone_0": "0–5 ft (Immediate Noncombustible Zone)",
    "Zone_1": "5–30 ft (Intermediate Defensible Space Zone)",
    "Zone_2": "30–100 ft (Extended Zone)",
}


def _resolve(object_class: str) -> str:
    return _ALIASES.get(object_class.lower().strip(), object_class.lower().strip().replace(" ","_"))


def get_regulatory_context(object_class: str, zone: str) -> dict:
    """
    Given a detected object and its HIZ zone, returns:
      - All applicable PRC 4291 sections (with full text)
      - All applicable IBHS requirements (with full text)
      - Associated violation types and severities
      - Aerial detectability flag
      - Suggested VLM prompt fragment for this object+zone combo
    """
    _load()
    obj_id = _resolve(object_class)
    zone_norm = zone.strip()

    # Collect outgoing edges from this object
    prc_regs, ibhs_reqs, violations = [], [], []
    for e in _edges:
        if e["source"] != obj_id:
            continue
        tgt = _nodes.get(e["target"], {})
        rel = e.get("relation","")
        if rel == "governed_by" and tgt.get("source") == "PRC_4291":
            if tgt.get("zone_scope","ALL") in (zone_norm, "ALL"):
                prc_regs.append({
                    "section_id"  : e["target"],
                    "full_text"   : tgt.get("full_text",""),
                    "zone_scope"  : tgt.get("zone_scope",""),
                    "confidence"  : e.get("confidence", 0.4),
                })
        elif rel == "subject_to" and tgt.get("source") == "IBHS_WFPH":
            if tgt.get("zone","ALL") in (zone_norm, "ALL"):
                ibhs_reqs.append({
                    "ibhs_id"           : e["target"],
                    "full_text"         : tgt.get("full_text",""),
                    "severity"          : tgt.get("severity",""),
                    "aerial_detectable" : tgt.get("aerial_detectable",""),
                    "confidence"        : e.get("confidence", 0.4),
                })
        elif rel == "can_cause":
            violations.append({
                "violation_id" : e["target"],
                "severity"     : e.get("severity", tgt.get("severity","")),
                "detection"    : tgt.get("detection_method","aerial"),
            })

    # Sort by confidence
    prc_regs  = sorted(prc_regs,  key=lambda x: -x["confidence"])
    ibhs_reqs = sorted(ibhs_reqs, key=lambda x: -x["confidence"])

    aerial_flag = any(r.get("aerial_detectable") in (True, "True") for r in ibhs_reqs)
    severity    = violations[0]["severity"] if violations else "UNKNOWN"
    zone_range  = ZONE_RANGES.get(zone_norm, zone_norm)

    # Build VLM prompt fragment
    prc_snippet  = prc_regs[0]["full_text"][:150]  if prc_regs  else "No specific PRC 4291 section matched."
    ibhs_snippet = ibhs_reqs[0]["full_text"][:150] if ibhs_reqs else "No specific IBHS requirement matched."
    viol_str     = violations[0]["violation_id"]   if violations else "General_compliance_required"

    obj_display = object_class.replace("_"," ").title()
    vlm_prompt = (
        f"A {obj_display} detected in {zone_norm} ({zone_range}). "
        f"Applicable regulation: {prc_snippet}. "
        f"IBHS requirement: {ibhs_snippet}. "
        f"Compliance assessment required: {viol_str}. "
        f"Severity if non-compliant: {severity}."
    )

    return {
        "object_class"      : obj_id,
        "zone"              : zone_norm,
        "zone_range"        : zone_range,
        "prc_sections"      : prc_regs,
        "ibhs_requirements" : ibhs_reqs,
        "violations"        : violations,
        "aerial_detectable" : aerial_flag,
        "vlm_prompt"        : vlm_prompt,
    }


if __name__ == "__main__":
    import pprint
    test_cases = [
        ("propane",        "Zone_0"),
        ("woodpile",       "Zone_1"),
        ("car",            "Zone_1"),
        ("mulch",          "Zone_0"),
        ("garbage_bin",    "Zone_0"),
        ("storage_shed",   "Zone_1"),
        ("live_shrub",     "Zone_0"),
        ("dead_vegetation","Zone_1"),
    ]
    for obj, zone in test_cases:
        print(f"\n{'='*60}")
        ctx = get_regulatory_context(obj, zone)
        print(f"Object: {ctx['object_class']} | Zone: {ctx['zone']}")
        print(f"VLM Prompt:\n  {ctx['vlm_prompt']}")
        print(f"Aerial detectable: {ctx['aerial_detectable']}")
        print(f"PRC sections matched: {len(ctx['prc_sections'])}")
        print(f"IBHS reqs matched:   {len(ctx['ibhs_requirements'])}")
        print(f"Violations:          {len(ctx['violations'])}")
