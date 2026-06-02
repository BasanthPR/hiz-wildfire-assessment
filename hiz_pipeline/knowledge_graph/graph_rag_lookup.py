"""
graph_rag_lookup.py
HIZ-VLM Pipeline — Graph-RAG Lookup Module
Loads the persisted knowledge graph and provides context retrieval
for prompt injection into GeoChat / InternVL2.

Usage:
    from knowledge_graph.graph_rag_lookup import get_regulatory_context
    ctx = get_regulatory_context("propane_tank", "Zone_0")
"""

import os
import networkx as nx
from pathlib import Path
from typing import Optional

_GRAPH_PATH = Path.home() / "hiz_pipeline" / "knowledge_graph" / "hiz_graph.graphml"
_G: Optional[nx.DiGraph] = None

# ─── Alias table (canonical name → list of aliases) ──────────────────────────
# Kept in-memory so we don't need to serialise list nodes in graphml
ALIASES: dict[str, list[str]] = {
    "propane_tank":  ["propane", "lpg", "gas tank", "bbq tank",
                      "liquid propane", "cylinder"],
    "trash_can":     ["garbage bin", "waste bin", "recycle bin",
                      "plastic bin", "garbage can", "trash bin"],
    "vehicle":       ["car", "truck", "van", "pickup", "suv",
                      "automobile", "rv", "boat"],
    "wood_pile":     ["firewood", "woodpile", "lumber stack",
                      "log pile", "stacked wood"],
    "storage_shed":  ["shed", "outbuilding", "structure"],
}

ZONE_WEIGHTS: dict[str, int] = {
    "Zone_0": 3,
    "Zone_1": 2,
    "Zone_2": 1,
}

SEVERITY_SCORES: dict[str, int] = {
    "CRITICAL": 3,
    "HIGH":     2,
    "MEDIUM":   1,
    "LOW":      0,
    "NONE":     0,
}


def _load_graph() -> nx.DiGraph:
    global _G
    if _G is None:
        if not _GRAPH_PATH.exists():
            raise FileNotFoundError(
                f"Knowledge graph not found at {_GRAPH_PATH}. "
                "Run build_graph.py first."
            )
        _G = nx.read_graphml(str(_GRAPH_PATH))
    return _G


def resolve_alias(raw_label: str) -> str:
    """
    Map a raw detection label (e.g. 'car', 'bbq tank') to a canonical
    object class (e.g. 'vehicle', 'propane_tank').
    Returns the input unchanged if no match is found.
    """
    label_lower = raw_label.lower().strip()
    # Exact match first
    if label_lower in ALIASES:
        return label_lower
    # Alias match
    for canonical, aliases in ALIASES.items():
        if label_lower in aliases or label_lower == canonical:
            return canonical
    # Partial substring match (last resort)
    for canonical, aliases in ALIASES.items():
        if any(alias in label_lower or label_lower in alias for alias in aliases):
            return canonical
        if canonical.replace("_", " ") in label_lower:
            return canonical
    return raw_label  # unresolved — caller decides how to handle


def get_regulatory_context(object_class: str, zone: str) -> dict:
    """
    Retrieve all applicable regulatory rules for an object in a given zone.

    Parameters
    ----------
    object_class : str
        Raw or canonical object label.
    zone : str
        One of "Zone_0", "Zone_1", "Zone_2".

    Returns
    -------
    dict with keys:
        canonical_class   : resolved canonical object class
        zone              : zone string
        zone_weight       : int (3 / 2 / 1)
        regulations       : list of dicts {clause, text, source}
        potential_violations : list of dicts {name, severity, description}
        severity          : worst-case severity string for this zone
        severity_score    : int
        formatted_context : ready-to-inject string for prompt
    """
    G = _load_graph()

    canonical = resolve_alias(object_class)
    zone_weight = ZONE_WEIGHTS.get(zone, 1)

    # Retrieve regulation nodes
    regulations = []
    if canonical in G:
        for neighbor in G.successors(canonical):
            ndata = G.nodes[neighbor]
            if ndata.get("node_type") == "regulation":
                regulations.append({
                    "clause":  ndata.get("clause", neighbor),
                    "text":    ndata.get("text", ""),
                    "source":  ndata.get("source", ""),
                })

    # Retrieve violation nodes
    violations = []
    worst_severity = "NONE"
    worst_score = 0
    if canonical in G:
        for neighbor in G.successors(canonical):
            ndata = G.nodes[neighbor]
            if ndata.get("node_type") == "violation":
                sev = ndata.get("severity", "MEDIUM")
                violations.append({
                    "name":        neighbor,
                    "severity":    sev,
                    "description": ndata.get("description", ""),
                })
                score = SEVERITY_SCORES.get(sev, 1)
                if score > worst_score:
                    worst_score = score
                    worst_severity = sev

    # Build formatted context string for prompt injection
    lines = [
        f"REGULATORY CONTEXT — {canonical.upper().replace('_', ' ')} in {zone}",
        f"Zone weight: {zone_weight} (Zone 0=3 critical, Zone 1=2, Zone 2=1)",
        "",
        "Applicable regulations:",
    ]
    for reg in regulations:
        lines.append(f"  [{reg['clause']}] {reg['text']}")
    if not regulations:
        lines.append("  (No specific regulations found for this object class)")

    lines += ["", "Potential violations:"]
    for viol in violations:
        lines.append(
            f"  [{viol['severity']}] {viol['name']}: {viol['description']}"
        )
    if not violations:
        lines.append("  (No pre-defined violations for this class/zone combo)")

    lines += [
        "",
        f"Worst-case severity for this detection: {worst_severity} "
        f"(score {worst_score})",
    ]

    return {
        "canonical_class":        canonical,
        "zone":                   zone,
        "zone_weight":            zone_weight,
        "regulations":            regulations,
        "potential_violations":   violations,
        "severity":               worst_severity,
        "severity_score":         worst_score,
        "formatted_context":      "\n".join(lines),
    }


def get_all_contexts_for_prompt(zone: str) -> str:
    """
    Build a combined regulatory context string for all three target
    object classes (propane_tank, trash_can, vehicle) for use in the
    standard VLM prompt.
    """
    target_classes = ["propane_tank", "trash_can", "vehicle"]
    sections = []
    for cls in target_classes:
        ctx = get_regulatory_context(cls, zone)
        sections.append(ctx["formatted_context"])
        sections.append("")
    return "\n".join(sections)


def compute_risk_score(detections: list[dict]) -> dict:
    """
    Compute a parcel-level risk score from a list of detection dicts.
    Each dict must have: compliance_status, object_class, zone, severity.

    Returns: {score, risk_label, violation_count, detection_breakdown}
    """
    score = 0
    violation_count = 0
    breakdown: dict[str, int] = {}

    for det in detections:
        status    = det.get("compliance_status", "UNKNOWN")
        obj_class = det.get("object_class", "unknown")
        zone      = det.get("zone", "Zone_2")
        severity  = det.get("severity", "NONE")

        if status == "VIOLATION":
            zw  = ZONE_WEIGHTS.get(zone, 1)
            sv  = SEVERITY_SCORES.get(severity, 1)
            score += zw * sv
            violation_count += 1
            key = obj_class
            breakdown[key] = breakdown.get(key, 0) + 1

    if score == 0:
        risk_label = "LOW"
    elif score <= 4:
        risk_label = "MEDIUM"
    elif score <= 9:
        risk_label = "HIGH"
    else:
        risk_label = "CRITICAL"

    return {
        "score":               score,
        "risk_label":          risk_label,
        "violation_count":     violation_count,
        "detection_breakdown": breakdown,
    }


# ─── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing alias resolution:")
    tests = ["car", "bbq tank", "garbage can", "shed", "log pile", "unknown_obj"]
    for t in tests:
        print(f"  '{t}' → '{resolve_alias(t)}'")

    print("\nRegulatory context for propane_tank in Zone_0:")
    ctx = get_regulatory_context("propane_tank", "Zone_0")
    print(ctx["formatted_context"])

    print("\nRegulatory context for vehicle in Zone_1:")
    ctx = get_regulatory_context("car", "Zone_1")
    print(ctx["formatted_context"])
