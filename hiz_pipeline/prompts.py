"""
prompts.py
HIZ-VLM Pipeline — Step 5: Geo-CoT Prompt Construction
Builds structured few-shot prompts for GeoChat and InternVL2.

Usage:
    from prompts import build_prompt, self_consistency_vote
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "hiz_pipeline"))
from knowledge_graph.graph_rag_lookup import get_all_contexts_for_prompt

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM MESSAGE
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = (
    "You are a wildfire defensible space compliance assessor evaluating "
    "drone imagery of residential properties in California WUI communities "
    "against PRC 4291 and IBHS Wildfire Prepared Home standards. "
    "Work through each step before producing any verdict. "
    "Cite the specific regulatory clause. "
    "Return your answer as valid JSON only — no markdown, no preamble."
)

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES (Geo-CoT framework)
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "description": "Example 1 — Propane tank, Zone 0, VIOLATION",
        "cot": (
            "PERCEIVE: Cylindrical silver object at lower-left quadrant of tile. "
            "Diameter ~28px at GSD 2.5cm/px = ~70cm real diameter. Shadow length "
            "~14px = ~35cm height consistent with BBQ propane tank.\n"
            "LOCATE: Object is approximately 4 feet from the south wall of the "
            "structure based on the red zone boundary line. Zone 0 (0-5 ft).\n"
            "RETRIEVE: PRC 4291 Section M requires 10ft bare mineral soil around "
            "propane tank plus additional 10ft no flammable vegetation. "
            "Dry grass is visible within the required clearance area.\n"
            "ASSESS: VIOLATION. Zone weight x3. Confidence: HIGH. "
            "Recommend inspector priority visit."
        ),
        "output": json.dumps({
            "object_class": "propane_tank",
            "bounding_box_pixels": [45, 210, 73, 238],
            "zone": "Zone_0",
            "compliance_status": "VIOLATION",
            "confidence": "HIGH",
            "severity": "CRITICAL",
            "cot_trace": (
                "PERCEIVE: Cylindrical silver object ~70cm real diameter. "
                "LOCATE: ~4ft from structure, Zone_0. "
                "RETRIEVE: PRC4291 §M — 10ft mineral soil + 10ft no flammable veg. "
                "ASSESS: Clearance requirement violated. Dry grass present within zone."
            ),
        }),
    },
    {
        "description": "Example 2 — Vehicle, Zone 1, COMPLIANT",
        "cot": (
            "PERCEIVE: Dark sedan at upper-right of tile. Clear rectangular vehicle "
            "profile, visible shadow consistent with midday flight.\n"
            "LOCATE: Approximately 18 feet from nearest structure wall per orange "
            "zone boundary. Zone 1 (5-30 ft).\n"
            "RETRIEVE: IBHS recommends no vehicle within 5ft of structure. "
            "At 18ft this requirement is met.\n"
            "ASSESS: COMPLIANT. Confidence: HIGH."
        ),
        "output": json.dumps({
            "object_class": "vehicle",
            "bounding_box_pixels": [380, 42, 498, 118],
            "zone": "Zone_1",
            "compliance_status": "COMPLIANT",
            "confidence": "HIGH",
            "severity": "NONE",
            "cot_trace": (
                "PERCEIVE: Dark sedan, rectangular profile. "
                "LOCATE: ~18ft from nearest structure, Zone_1. "
                "RETRIEVE: IBHS Zone 0 Vehicle — no parking within 5ft. "
                "ASSESS: 18ft exceeds 5ft minimum. COMPLIANT."
            ),
        }),
    },
    {
        "description": "Example 3 — Ambiguous object, UNCERTAIN",
        "cot": (
            "PERCEIVE: Cylindrical object partially occluded by tree canopy shadow "
            "at tile center. Diameter unclear due to shadow. Could be propane tank "
            "or water storage container.\n"
            "LOCATE: Within Zone 0 based on red boundary.\n"
            "RETRIEVE: Cannot confirm object class with sufficient confidence.\n"
            "ASSESS: UNCERTAIN. Human review required. Confidence: LOW."
        ),
        "output": json.dumps({
            "object_class": "unknown_cylinder",
            "bounding_box_pixels": [200, 180, 240, 220],
            "zone": "Zone_0",
            "compliance_status": "UNCERTAIN",
            "confidence": "LOW",
            "severity": "REVIEW",
            "cot_trace": (
                "PERCEIVE: Cylindrical object, partially occluded. "
                "LOCATE: Zone_0 (inside red boundary). "
                "RETRIEVE: Cannot confirm propane_tank vs water_storage. "
                "ASSESS: Insufficient confidence. Flag for human review."
            ),
        }),
    },
]


def _format_few_shot_block() -> str:
    lines = ["--- FEW-SHOT EXAMPLES ---"]
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"\n[{ex['description']}]")
        lines.append("Chain-of-thought reasoning:")
        lines.append(ex["cot"])
        lines.append("JSON output:")
        lines.append(ex["output"])
    lines.append("\n--- END EXAMPLES ---\n")
    return "\n".join(lines)


FEW_SHOT_BLOCK = _format_few_shot_block()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(
    tile_path: str,
    parcel_meta: dict,
    zone: str,
    use_graph_rag: bool = True,
) -> dict:
    """
    Build a structured prompt dict for a single tile.

    Parameters
    ----------
    tile_path    : absolute path to the tile PNG
    parcel_meta  : dict loaded from {parcel_id}_meta.json
    zone         : zone label for this tile ("Zone_0" / "Zone_1" / "Zone_2")
    use_graph_rag: if False, builds the plain ablation prompt (Step 8)

    Returns
    -------
    dict with keys: system, regulatory_context, few_shot, user, tile_path
    """
    parcel_id = parcel_meta.get("parcel_id", "unknown")
    site      = parcel_meta.get("site", "unknown")
    gsd_cm    = parcel_meta.get("gsd_cm", 0.0)

    # Extract CHM stats for this specific tile if stored in manifest
    chm_mean = parcel_meta.get("chm_mean_in_tile", 0.0)
    chm_max  = parcel_meta.get("chm_max_in_tile", 0.0)

    if use_graph_rag:
        reg_context = get_all_contexts_for_prompt(zone)
        user_message = (
            f"Inspect this drone tile for the following combustible objects: "
            f"[vehicles / cars / trucks], [trash cans / garbage bins], "
            f"[propane tanks].\n\n"
            f"The tile is from {zone} of parcel {parcel_id} in site '{site}'.\n"
            f"GSD is {gsd_cm:.1f} cm/pixel.\n"
            f"CHM mean height in this tile: {chm_mean:.2f}m, "
            f"max: {chm_max:.2f}m.\n\n"
            f"Return valid JSON (no markdown, no preamble) with fields:\n"
            f"  detections: list of objects found, each with:\n"
            f"    object_class, bounding_box_pixels [x0,y0,x1,y1],\n"
            f"    zone, compliance_status (VIOLATION/COMPLIANT/UNCERTAIN),\n"
            f"    confidence (HIGH/MEDIUM/LOW),\n"
            f"    severity (CRITICAL/HIGH/MEDIUM/NONE/REVIEW),\n"
            f"    cot_trace (PERCEIVE→LOCATE→RETRIEVE→ASSESS)\n"
            f"  overall_parcel_risk: LOW/MEDIUM/HIGH/CRITICAL\n"
            f"  aerial_limitations: list of things you cannot determine "
            f"from this aerial view\n\n"
            f"[Tile image attached]"
        )
    else:
        # Ablation: plain prompt, no regulatory context
        reg_context = None
        user_message = (
            "Look at this aerial drone image. "
            "Identify any vehicles, trash cans, or propane tanks you can see. "
            "For each object, describe its location and approximate distance "
            "from the nearest structure. "
            "Return JSON with a detections list."
        )

    return {
        "system":              SYSTEM_MESSAGE,
        "regulatory_context":  reg_context,
        "few_shot":            FEW_SHOT_BLOCK if use_graph_rag else None,
        "user":                user_message,
        "tile_path":           tile_path,
        "parcel_id":           parcel_id,
        "zone":                zone,
    }


def format_full_prompt_text(prompt_dict: dict) -> str:
    """
    Flatten the prompt dict into a single string for models that
    take a plain text input (no chat template).
    """
    parts = []
    parts.append(f"[SYSTEM]\n{prompt_dict['system']}\n")

    if prompt_dict.get("regulatory_context"):
        parts.append(
            f"[REGULATORY CONTEXT]\n{prompt_dict['regulatory_context']}\n"
        )

    if prompt_dict.get("few_shot"):
        parts.append(f"[EXAMPLES]\n{prompt_dict['few_shot']}\n")

    parts.append(f"[USER]\n{prompt_dict['user']}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-CONSISTENCY VOTING
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_detection(det: dict) -> dict:
    """Ensure a detection dict has all expected fields."""
    return {
        "object_class":        det.get("object_class", "unknown"),
        "bounding_box_pixels": det.get("bounding_box_pixels", []),
        "zone":                det.get("zone", "Zone_2"),
        "compliance_status":   det.get("compliance_status", "UNCERTAIN"),
        "confidence":          det.get("confidence", "LOW"),
        "severity":            det.get("severity", "NONE"),
        "cot_trace":           det.get("cot_trace", ""),
    }


def self_consistency_vote(responses: list[dict]) -> dict:
    """
    Aggregate multiple VLM responses to the same tile via majority vote.

    Parameters
    ----------
    responses : list of parsed JSON dicts (each from a single inference call)

    Returns
    -------
    dict with keys:
        voted_detections   : list of merged detection dicts (majority verdict)
        all_agree          : bool — True if all responses agree on every detection
        disagreement_flags : list of detection indices where responses disagree
        vote_details       : raw vote tallies per detection per field
    """
    if not responses:
        return {
            "voted_detections": [],
            "all_agree": True,
            "disagreement_flags": [],
            "vote_details": [],
        }

    if len(responses) == 1:
        dets = responses[0].get("detections", [])
        return {
            "voted_detections": [_normalize_detection(d) for d in dets],
            "all_agree": True,
            "disagreement_flags": [],
            "vote_details": [],
        }

    # Align detections across responses by object_class (simplified matching)
    # Collect all unique object classes mentioned across all responses
    all_classes: list[str] = []
    for resp in responses:
        for det in resp.get("detections", []):
            cls = det.get("object_class", "unknown")
            if cls not in all_classes:
                all_classes.append(cls)

    voted_detections = []
    disagreement_flags = []
    vote_details = []

    for cls in all_classes:
        # Gather detections for this class across responses
        class_dets = []
        for resp in responses:
            matching = [d for d in resp.get("detections", [])
                        if d.get("object_class") == cls]
            class_dets.append(matching[0] if matching else None)

        # Vote on compliance_status
        statuses = [d["compliance_status"] if d else "UNCERTAIN"
                    for d in class_dets]
        from collections import Counter
        status_votes = Counter(statuses)
        majority_status = status_votes.most_common(1)[0][0]
        all_agree_status = len(status_votes) == 1

        # Vote on severity
        severities = [d.get("severity", "NONE") if d else "NONE"
                      for d in class_dets]
        sev_votes = Counter(severities)
        majority_severity = sev_votes.most_common(1)[0][0]

        # Vote on confidence — take lowest if any disagrees
        confidences = [d.get("confidence", "LOW") if d else "LOW"
                       for d in class_dets]
        conf_order = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
        min_conf = min(confidences, key=lambda c: conf_order.get(c, 0))

        # If all three disagree, mark UNCERTAIN
        if not all_agree_status and len(status_votes) == len(responses):
            majority_status = "UNCERTAIN"
            min_conf = "LOW"
            disagreement_flags.append(cls)

        # Use bounding box from the first non-None response
        best_det = next((d for d in class_dets if d is not None), {})
        bbox = best_det.get("bounding_box_pixels", [])
        zone = best_det.get("zone", "Zone_2")
        cot  = best_det.get("cot_trace", "")

        voted_detections.append({
            "object_class":       cls,
            "bounding_box_pixels": bbox,
            "zone":               zone,
            "compliance_status":  majority_status,
            "confidence":         min_conf,
            "severity":           majority_severity,
            "cot_trace":          cot,
        })

        vote_details.append({
            "object_class":   cls,
            "status_votes":   dict(status_votes),
            "severity_votes": dict(sev_votes),
        })

    all_agree = len(disagreement_flags) == 0
    return {
        "voted_detections":   voted_detections,
        "all_agree":          all_agree,
        "disagreement_flags": disagreement_flags,
        "vote_details":       vote_details,
    }
