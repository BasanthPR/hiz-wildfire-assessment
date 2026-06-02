"""
build_graph.py
HIZ-VLM Pipeline — Step 3: Knowledge Graph Construction
Encodes PRC 4291 and IBHS Wildfire Prepared Home standards as a
NetworkX DiGraph for Graph-RAG context injection.

Usage:
    python3 ~/hiz_pipeline/knowledge_graph/build_graph.py

Outputs:
    ~/hiz_pipeline/knowledge_graph/hiz_graph.graphml
    (graph_rag_lookup.py is a companion module — import separately)
"""

import os
import networkx as nx
from pathlib import Path

OUTPUT_DIR = Path.home() / "hiz_pipeline" / "knowledge_graph"
GRAPH_PATH = OUTPUT_DIR / "hiz_graph.graphml"

# ─────────────────────────────────────────────────────────────────────────────
# NODE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

OBJECT_NODES = {
    "propane_tank": {
        "node_type": "object",
        "aliases": "propane,lpg,gas tank,bbq tank,liquid propane,cylinder",
        "description": "Propane or LPG storage tank on residential property",
    },
    "trash_can": {
        "node_type": "object",
        "aliases": "garbage bin,waste bin,recycle bin,plastic bin,garbage can,trash bin",
        "description": "Residential waste or recycling container",
    },
    "vehicle": {
        "node_type": "object",
        "aliases": "car,truck,van,pickup,suv,automobile,rv,boat",
        "description": "Motorized vehicle parked on property",
    },
    "wood_pile": {
        "node_type": "object",
        "aliases": "firewood,woodpile,lumber stack,log pile,stacked wood",
        "description": "Stack of firewood or lumber stored outdoors",
    },
    "storage_shed": {
        "node_type": "object",
        "aliases": "shed,outbuilding,structure",
        "description": "Accessory storage structure on property",
    },
}

REGULATION_NODES = {
    # ── Propane tank regulations ──────────────────────────────────────────
    "PRC4291_M": {
        "node_type": "regulation",
        "source": "PRC 4291 Section M",
        "text": "10ft bare mineral soil clearance required around propane tank",
        "clause": "PRC 4291 §M",
    },
    "PRC4291_M_ext": {
        "node_type": "regulation",
        "source": "PRC 4291 Section M Extended",
        "text": "Additional 10ft no flammable vegetation beyond the mineral soil ring",
        "clause": "PRC 4291 §M (extended)",
    },
    "IBHS_Z1_propane": {
        "node_type": "regulation",
        "source": "IBHS Wildfire Prepared Home",
        "text": (
            "If >30ft from home is not possible, minimum 10ft clearance "
            "plus enclosed 4 sides with concrete block construction"
        ),
        "clause": "IBHS Zone 1 Propane",
    },
    # ── Vehicle regulations ───────────────────────────────────────────────
    "IBHS_Z0_vehicle": {
        "node_type": "regulation",
        "source": "IBHS Wildfire Prepared Home",
        "text": "Do not park vehicles within 5ft of any structure",
        "clause": "IBHS Zone 0 Vehicle",
    },
    "IBHS_redflag": {
        "node_type": "regulation",
        "source": "IBHS Wildfire Prepared Home",
        "text": (
            "On red flag warning days, point vehicle toward exit "
            "and keep keys inside for rapid evacuation"
        ),
        "clause": "IBHS Red Flag Vehicle Protocol",
    },
    # ── Trash can regulations ─────────────────────────────────────────────
    "IBHS_Z0_trash_plastic": {
        "node_type": "regulation",
        "source": "IBHS Wildfire Prepared Home",
        "text": (
            "Plastic trash cans in Zone 0 (0-5ft) are a combustible hazard "
            "and must be removed or replaced with metal containers"
        ),
        "clause": "IBHS Zone 0 Trash (Plastic)",
    },
    "IBHS_Z0_trash_metal": {
        "node_type": "regulation",
        "source": "IBHS Wildfire Prepared Home",
        "text": "Metal trash cans are acceptable in Zone 0",
        "clause": "IBHS Zone 0 Trash (Metal)",
    },
    # ── Wood pile regulations ─────────────────────────────────────────────
    "PRC4291_G": {
        "node_type": "regulation",
        "source": "PRC 4291 Section G",
        "text": (
            "Relocate exposed wood pile outside Zone 1 unless fully covered "
            "with fire-resistant material"
        ),
        "clause": "PRC 4291 §G",
    },
    "IBHS_woodpile": {
        "node_type": "regulation",
        "source": "IBHS Wildfire Prepared Home",
        "text": (
            "Store firewood at least 30ft from structure; "
            "maintain 10ft bare mineral soil clearance around the pile"
        ),
        "clause": "IBHS Firewood Storage",
    },
    # ── Storage shed regulations ──────────────────────────────────────────
    "IBHS_Z1_shed": {
        "node_type": "regulation",
        "source": "IBHS Wildfire Prepared Home",
        "text": (
            "Place accessory structures at least 10ft from the primary home; "
            "maintain 5ft noncombustible buffer around each structure"
        ),
        "clause": "IBHS Zone 1 Shed",
    },
}

ZONE_NODES = {
    "Zone_0": {
        "node_type": "zone",
        "description": "Immediate zone: 0-5 ft from structure",
        "feet_min": 0,
        "feet_max": 5,
        "weight": 3,
    },
    "Zone_1": {
        "node_type": "zone",
        "description": "Intermediate zone: 5-30 ft from structure",
        "feet_min": 5,
        "feet_max": 30,
        "weight": 2,
    },
    "Zone_2": {
        "node_type": "zone",
        "description": "Extended zone: 30-100 ft from structure",
        "feet_min": 30,
        "feet_max": 100,
        "weight": 1,
    },
}

VIOLATION_NODES = {
    "Propane_in_Zone0": {
        "node_type": "violation",
        "severity": "CRITICAL",
        "description": "Propane tank within 5ft of structure",
    },
    "Vehicle_within_5ft": {
        "node_type": "violation",
        "severity": "HIGH",
        "description": "Vehicle parked within 5ft of structure",
    },
    "Plastic_trash_Zone0": {
        "node_type": "violation",
        "severity": "CRITICAL",
        "description": "Plastic trash container in Zone 0",
    },
    "Woodpile_Zone1_exposed": {
        "node_type": "violation",
        "severity": "HIGH",
        "description": "Exposed wood pile within Zone 1",
    },
    "Shed_too_close": {
        "node_type": "violation",
        "severity": "MEDIUM",
        "description": "Storage shed less than 10ft from primary structure",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# EDGE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

EDGES = [
    # propane_tank → regulations
    ("propane_tank", "PRC4291_M",        {"rel": "governed_by"}),
    ("propane_tank", "PRC4291_M_ext",    {"rel": "governed_by"}),
    ("propane_tank", "IBHS_Z1_propane",  {"rel": "governed_by"}),
    # propane_tank → violation
    ("propane_tank", "Propane_in_Zone0", {"rel": "can_cause"}),
    ("Propane_in_Zone0", "Zone_0",       {"rel": "occurs_in"}),

    # vehicle → regulations
    ("vehicle", "IBHS_Z0_vehicle",       {"rel": "governed_by"}),
    ("vehicle", "IBHS_redflag",          {"rel": "governed_by"}),
    # vehicle → violation
    ("vehicle", "Vehicle_within_5ft",    {"rel": "can_cause"}),
    ("Vehicle_within_5ft", "Zone_0",     {"rel": "occurs_in"}),

    # trash_can → regulations
    ("trash_can", "IBHS_Z0_trash_plastic", {"rel": "governed_by"}),
    ("trash_can", "IBHS_Z0_trash_metal",   {"rel": "governed_by"}),
    # trash_can → violation
    ("trash_can", "Plastic_trash_Zone0",   {"rel": "can_cause"}),
    ("Plastic_trash_Zone0", "Zone_0",      {"rel": "occurs_in"}),

    # wood_pile → regulations
    ("wood_pile", "PRC4291_G",             {"rel": "governed_by"}),
    ("wood_pile", "IBHS_woodpile",         {"rel": "governed_by"}),
    # wood_pile → violation
    ("wood_pile", "Woodpile_Zone1_exposed",{"rel": "can_cause"}),
    ("Woodpile_Zone1_exposed", "Zone_1",   {"rel": "occurs_in"}),

    # storage_shed → regulations
    ("storage_shed", "IBHS_Z1_shed",       {"rel": "governed_by"}),
    # storage_shed → violation
    ("storage_shed", "Shed_too_close",     {"rel": "can_cause"}),
    ("Shed_too_close", "Zone_1",           {"rel": "occurs_in"}),

    # Zone relationships
    ("Zone_0", "Zone_1", {"rel": "contained_in"}),
    ("Zone_1", "Zone_2", {"rel": "contained_in"}),
]


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    for node_id, attrs in OBJECT_NODES.items():
        G.add_node(node_id, **attrs)

    for node_id, attrs in REGULATION_NODES.items():
        G.add_node(node_id, **attrs)

    for node_id, attrs in ZONE_NODES.items():
        G.add_node(node_id, **{k: str(v) if not isinstance(v, str) else v
                               for k, v in attrs.items()})

    for node_id, attrs in VIOLATION_NODES.items():
        G.add_node(node_id, **attrs)

    for src, dst, attrs in EDGES:
        G.add_edge(src, dst, **attrs)

    return G


def print_coverage_table(G: nx.DiGraph):
    print("\n" + "=" * 72)
    print("  KNOWLEDGE GRAPH — REGULATORY COVERAGE TABLE")
    print("=" * 72)
    print(f"{'Object Class':<18} {'Regulations':<35} {'Potential Violations'}")
    print("-" * 72)

    for obj in OBJECT_NODES:
        regs = [
            n for n in G.successors(obj)
            if G.nodes[n].get("node_type") == "regulation"
        ]
        viols = [
            n for n in G.successors(obj)
            if G.nodes[n].get("node_type") == "violation"
        ]
        reg_names = [REGULATION_NODES[r]["clause"] for r in regs if r in REGULATION_NODES]
        viol_names = [f"{v} ({VIOLATION_NODES[v]['severity']})"
                      for v in viols if v in VIOLATION_NODES]

        print(f"\n  {obj}")
        for r in reg_names:
            print(f"    ← {r}")
        for v in viol_names:
            print(f"    ! Violation: {v}")

    print("\n" + "=" * 72)
    print(f"  Total nodes : {G.number_of_nodes()}")
    print(f"  Total edges : {G.number_of_edges()}")
    print(f"  Object nodes     : {sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'object')}")
    print(f"  Regulation nodes : {sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'regulation')}")
    print(f"  Zone nodes       : {sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'zone')}")
    print(f"  Violation nodes  : {sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'violation')}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    print("Building HIZ Regulatory Knowledge Graph...")

    G = build_graph()
    print_coverage_table(G)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(GRAPH_PATH))
    print(f"Graph saved to: {GRAPH_PATH}")
    print("Done.")
