"""
ingest_calfire_le100a.py
========================
Incrementally updates the HIZ Wildfire Knowledge Graph with requirements
extracted from the CAL FIRE LE-100a (08/23) Notice of Defensible Space
Inspection form.

Run AFTER build_knowledge_graph.py has already produced:
  - knowledge_graph_nodes.json
  - knowledge_graph_edges.json
  - knowledge_graph.graphml

Outputs regenerated:
  - knowledge_graph_nodes.json      (updated)
  - knowledge_graph_edges.json      (updated)
  - knowledge_graph.graphml         (updated)
  - knowledge_graph_static.png      (updated)
  - knowledge_graph_interactive.html (updated)
  - knowledge_graph_edge_coverage.xlsx (updated)
  - aerial_detectability_partition.xlsx (updated)
  - graph_validation_report.txt     (updated)
  - calfire_le100a_requirements.json (new)
"""

import os
import re
import json
import time
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IN = {
    "nodes_json" : os.path.join(BASE_DIR, "knowledge_graph_nodes.json"),
    "edges_json" : os.path.join(BASE_DIR, "knowledge_graph_edges.json"),
    "graphml"    : os.path.join(BASE_DIR, "knowledge_graph.graphml"),
}

OUT = {
    "calfire_json"  : os.path.join(BASE_DIR, "calfire_le100a_requirements.json"),
    "nodes_json"    : os.path.join(BASE_DIR, "knowledge_graph_nodes.json"),
    "edges_json"    : os.path.join(BASE_DIR, "knowledge_graph_edges.json"),
    "graphml"       : os.path.join(BASE_DIR, "knowledge_graph.graphml"),
    "static_png"    : os.path.join(BASE_DIR, "knowledge_graph_static.png"),
    "interactive"   : os.path.join(BASE_DIR, "knowledge_graph_interactive.html"),
    "edge_xlsx"     : os.path.join(BASE_DIR, "knowledge_graph_edge_coverage.xlsx"),
    "detect_xlsx"   : os.path.join(BASE_DIR, "aerial_detectability_partition.xlsx"),
    "validation"    : os.path.join(BASE_DIR, "graph_validation_report.txt"),
}

print("=" * 70)
print("CALFIRE LE-100A INGESTION — HIZ KNOWLEDGE GRAPH UPDATE")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD EXISTING GRAPH
# ══════════════════════════════════════════════════════════════════════
print("\n[1] Loading existing knowledge graph …")

with open(IN["nodes_json"]) as f:
    raw_nodes = json.load(f)
with open(IN["edges_json"]) as f:
    raw_edges = json.load(f)

G = nx.DiGraph()

for nd in raw_nodes:
    nid = nd.pop("id")
    # restore bool fields
    for k, v in nd.items():
        if v == "True":
            nd[k] = True
        elif v == "False":
            nd[k] = False
    G.add_node(nid, **nd)

for ed in raw_edges:
    src = ed.pop("source")
    tgt = ed.pop("target")
    G.add_edge(src, tgt, **ed)

print(f"  Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — DEFINE CALFIRE LE-100A REQUIREMENTS
# ══════════════════════════════════════════════════════════════════════
print("\n[2] Defining CAL FIRE LE-100a requirements …")

CALFIRE_REQUIREMENTS = [
    # ── Zone 1 items (labeled A–G on form, apply within 30 ft = Zone_1) ──
    {
        "calfire_id"        : "CALFIRE_Z1_A",
        "zone"              : "Zone_1",
        "item_label"        : "A",
        "requirement_text"  : "Remove all branches within 10 feet of any chimney or stovepipe outlet.",
        "object_mentions"   : [],
        "aerial_detectable" : False,
        "requirement_type"  : "removal",
        "severity"          : "HIGH",
    },
    {
        "calfire_id"        : "CALFIRE_Z1_B",
        "zone"              : "Zone_0",
        "item_label"        : "B",
        "requirement_text"  : "Remove leaves, needles or other vegetation on roofs, gutters, decks, porches, stairways, etc.",
        "object_mentions"   : ["debris", "deck_patio"],
        "aerial_detectable" : True,
        "requirement_type"  : "removal",
        "severity"          : "CRITICAL",
    },
    {
        "calfire_id"        : "CALFIRE_Z1_C",
        "zone"              : "Zone_1",
        "item_label"        : "C",
        "requirement_text"  : "Remove dead tree or shrub branches that overhang roofs, below or adjacent to windows, or which are adjacent to wall surfaces.",
        "object_mentions"   : ["dead_vegetation", "live_tree", "live_shrub"],
        "aerial_detectable" : True,
        "requirement_type"  : "removal",
        "severity"          : "HIGH",
    },
    {
        "calfire_id"        : "CALFIRE_Z1_D",
        "zone"              : "Zone_1",
        "item_label"        : "D",
        "requirement_text"  : "Remove all dead and dying grass, plants, shrubs, trees, branches, leaves, weeds and needles.",
        "object_mentions"   : ["dead_vegetation", "live_herb", "live_shrub", "live_tree", "debris"],
        "aerial_detectable" : True,
        "requirement_type"  : "removal",
        "severity"          : "HIGH",
    },
    {
        "calfire_id"        : "CALFIRE_Z1_E",
        "zone"              : "Zone_1",
        "item_label"        : "E",
        "requirement_text"  : "Remove or separate fuels to maintain spacing between vegetation to interrupt fire's path. Prune limbs; separate plants and ground cover.",
        "object_mentions"   : ["live_shrub", "live_tree", "live_herb"],
        "aerial_detectable" : True,
        "requirement_type"  : "clearance",
        "severity"          : "HIGH",
    },
    {
        "calfire_id"        : "CALFIRE_Z1_F",
        "zone"              : "Zone_1",
        "item_label"        : "F",
        "requirement_text"  : "Remove flammable vegetation and items that could catch fire which are adjacent to, or below, combustible decks, balconies, and stairs.",
        "object_mentions"   : ["deck_patio", "vegetation", "debris"],
        "aerial_detectable" : True,
        "requirement_type"  : "removal",
        "severity"          : "HIGH",
    },
    {
        "calfire_id"        : "CALFIRE_Z1_G",
        "zone"              : "Zone_1",
        "item_label"        : "G",
        "requirement_text"  : "Relocate exposed wood piles outside of Zone 1 (30 ft) unless completely covered in a fire-resistant material.",
        "object_mentions"   : ["woodpile"],
        "aerial_detectable" : True,
        "requirement_type"  : "spacing",
        "severity"          : "HIGH",
    },
    # ── Zone 2 items (labeled H–O on form, apply within 30–100 ft = Zone_2) ──
    {
        "calfire_id"        : "CALFIRE_Z2_H",
        "zone"              : "Zone_2",
        "item_label"        : "H",
        "requirement_text"  : "Cut annual grasses and forbs down to a maximum height of 4 inches.",
        "object_mentions"   : ["live_herb"],
        "aerial_detectable" : True,
        "requirement_type"  : "clearance",
        "severity"          : "MEDIUM",
    },
    {
        "calfire_id"        : "CALFIRE_Z2_I",
        "zone"              : "Zone_2",
        "item_label"        : "I",
        "requirement_text"  : "Remove fuels to create proper horizontal and vertical spacing among shrubs and trees, and remove lower tree limbs (ladder fuels).",
        "object_mentions"   : ["live_tree", "live_shrub"],
        "aerial_detectable" : True,
        "requirement_type"  : "clearance",
        "severity"          : "MEDIUM",
    },
    {
        "calfire_id"        : "CALFIRE_Z2_J",
        "zone"              : "Zone_2",
        "item_label"        : "J",
        "requirement_text"  : "All exposed wood piles must have a minimum of 10 feet clearance, down to bare mineral soil, in all directions.",
        "object_mentions"   : ["woodpile"],
        "aerial_detectable" : True,
        "requirement_type"  : "spacing",
        "severity"          : "MEDIUM",
    },
    {
        "calfire_id"        : "CALFIRE_Z2_K",
        "zone"              : "Zone_2",
        "item_label"        : "K",
        "requirement_text"  : "Remove all dead and dying trees, branches, shrubs, or other plants, and surface debris. Loose surface litter (fallen leaves, needles, twigs, bark, cones) shall be permitted to a depth of 3 inches.",
        "object_mentions"   : ["dead_vegetation", "live_tree", "debris"],
        "aerial_detectable" : True,
        "requirement_type"  : "removal",
        "severity"          : "MEDIUM",
    },
    {
        "calfire_id"        : "CALFIRE_Z2_L",
        "zone"              : "Zone_2",
        "item_label"        : "L",
        "requirement_text"  : "Logs or stumps embedded in the soil must be removed or isolated from other vegetation.",
        "object_mentions"   : ["dead_vegetation", "debris"],
        "aerial_detectable" : True,
        "requirement_type"  : "removal",
        "severity"          : "MEDIUM",
    },
    {
        "calfire_id"        : "CALFIRE_Z2_M",
        "zone"              : "Zone_2",
        "item_label"        : "M",
        "requirement_text"  : "Outbuildings and Liquid Propane Gas (LPG) storage tanks shall have 10 feet of clearance to bare mineral soil and no flammable vegetation for an additional 10 feet around their exterior.",
        "object_mentions"   : ["storage_shed", "propane"],
        "aerial_detectable" : True,
        "requirement_type"  : "spacing",
        "severity"          : "HIGH",
    },
    {
        "calfire_id"        : "CALFIRE_Z2_N",
        "zone"              : "ALL",
        "item_label"        : "N",
        "requirement_text"  : "Address numbers shall be displayed in contrasting colors (4 inch minimum size) and readable from the street or access road.",
        "object_mentions"   : ["address_sign"],
        "aerial_detectable" : True,
        "requirement_type"  : "structural",
        "severity"          : "LOW",
    },
    {
        "calfire_id"        : "CALFIRE_Z2_O",
        "zone"              : "ALL",
        "item_label"        : "O",
        "requirement_text"  : "Equip chimney or stovepipe openings with a metal screen having openings between 3/8 inch and 1/2 inch.",
        "object_mentions"   : [],
        "aerial_detectable" : False,
        "requirement_type"  : "structural",
        "severity"          : "HIGH",
    },
    # ── Zone 0 / Home Hardening items ──
    {
        "calfire_id"        : "CALFIRE_Z0_001",
        "zone"              : "Zone_0",
        "item_label"        : "Z0-Ember",
        "requirement_text"  : "Eliminate flammable materials and vegetation in the 0–5 foot ember-resistant zone. This zone is critical for protecting homes during wildfires.",
        "object_mentions"   : ["vegetation", "mulch", "debris", "live_herb"],
        "aerial_detectable" : True,
        "requirement_type"  : "removal",
        "severity"          : "CRITICAL",
    },
    {
        "calfire_id"        : "CALFIRE_Z0_002",
        "zone"              : "Zone_0",
        "item_label"        : "Z0-Roof",
        "requirement_text"  : "Roof shall be the highest priority for ignition-resistant materials (Class A roofing). Roof and eaves/soffits shall use ignition-resistant construction.",
        "object_mentions"   : [],
        "aerial_detectable" : False,
        "requirement_type"  : "material",
        "severity"          : "CRITICAL",
    },
    {
        "calfire_id"        : "CALFIRE_Z0_003",
        "zone"              : "Zone_0",
        "item_label"        : "Z0-Vents",
        "requirement_text"  : "Cover and protect all vent openings to prevent ember intrusion. Vents shall be covered with fine mesh screening.",
        "object_mentions"   : [],
        "aerial_detectable" : False,
        "requirement_type"  : "structural",
        "severity"          : "CRITICAL",
    },
    {
        "calfire_id"        : "CALFIRE_Z0_004",
        "zone"              : "Zone_0",
        "item_label"        : "Z0-Windows",
        "requirement_text"  : "Install dual-paned windows and protect against blow-outs during wildfire.",
        "object_mentions"   : [],
        "aerial_detectable" : False,
        "requirement_type"  : "structural",
        "severity"          : "CRITICAL",
    },
    {
        "calfire_id"        : "CALFIRE_Z0_005",
        "zone"              : "Zone_0",
        "item_label"        : "Z0-Gutters",
        "requirement_text"  : "Screen or enclose rain gutters to prevent vegetative debris accumulation that could ignite from embers.",
        "object_mentions"   : ["debris"],
        "aerial_detectable" : False,
        "requirement_type"  : "structural",
        "severity"          : "CRITICAL",
    },
    {
        "calfire_id"        : "CALFIRE_Z0_006",
        "zone"              : "Zone_0",
        "item_label"        : "Z0-Chimney",
        "requirement_text"  : "Cover chimney outlets with non-combustible screens to prevent ember escape or entry.",
        "object_mentions"   : [],
        "aerial_detectable" : False,
        "requirement_type"  : "structural",
        "severity"          : "CRITICAL",
    },
    {
        "calfire_id"        : "CALFIRE_Z0_007",
        "zone"              : "ALL",
        "item_label"        : "Z0-Driveway",
        "requirement_text"  : "Ensure driveway access to your home complies with local fire codes to allow emergency vehicle access.",
        "object_mentions"   : ["driveway"],
        "aerial_detectable" : True,
        "requirement_type"  : "structural",
        "severity"          : "MEDIUM",
    },
    {
        "calfire_id"        : "CALFIRE_Z0_008",
        "zone"              : "ALL",
        "item_label"        : "Z0-Water",
        "requirement_text"  : "Have multiple garden hoses that are long enough to reach all areas of your home for fire suppression.",
        "object_mentions"   : ["hoses"],
        "aerial_detectable" : False,
        "requirement_type"  : "structural",
        "severity"          : "MEDIUM",
    },
]

print(f"  Defined {len(CALFIRE_REQUIREMENTS)} CAL FIRE LE-100a requirements")

with open(OUT["calfire_json"], "w") as f:
    json.dump(CALFIRE_REQUIREMENTS, f, indent=2)
print(f"  Saved → {OUT['calfire_json']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — ADD CALFIRE REGULATION NODES TO GRAPH
# ══════════════════════════════════════════════════════════════════════
print("\n[3] Adding CAL FIRE LE-100a nodes …")

existing_node_ids = set(G.nodes())
added_nodes = 0

for req in CALFIRE_REQUIREMENTS:
    nid = req["calfire_id"]
    if nid in existing_node_ids:
        print(f"  Skipping existing node: {nid}")
        continue
    G.add_node(
        nid,
        type="regulation",
        source="CALFIRE_LE100A",
        full_text=req["requirement_text"],
        zone=req["zone"],
        item_label=req["item_label"],
        requirement_type=req["requirement_type"],
        severity=req["severity"],
        aerial_detectable=req["aerial_detectable"],
    )
    added_nodes += 1

print(f"  Added {added_nodes} new regulation nodes")


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — WIRE EDGES
# ══════════════════════════════════════════════════════════════════════
print("\n[4] Adding edges …")

ZONES = ["Zone_0", "Zone_1", "Zone_2"]
added_edges = 0

def add_edge_once(G, u, v, **attrs):
    global added_edges
    if not G.has_edge(u, v):
        G.add_edge(u, v, **attrs)
        added_edges += 1

# zone → calfire_req: covered_by
for req in CALFIRE_REQUIREMENTS:
    zone = req["zone"]
    target_zones = ZONES if zone == "ALL" else [zone]
    for z in target_zones:
        if z in G:
            add_edge_once(G, z, req["calfire_id"], relation="covered_by")

# object → calfire_req: subject_to
OBJECT_ALIASES = {
    "vegetation" : ["live_herb", "live_shrub", "live_tree", "dead_vegetation"],
    "woodpile"   : ["woodpile"],
    "debris"     : ["debris", "dead_vegetation"],
    "deck_patio" : ["deck_patio"],
    "storage_shed": ["storage_shed"],
    "propane"    : ["propane"],
    "address_sign": ["address_sign"],
    "driveway"   : ["driveway"],
    "hoses"      : ["hoses"],
    "mulch"      : ["mulch"],
}

def calc_confidence(obj_id, obj_text, req_text):
    t = req_text.lower()
    name_norm = obj_id.replace("_", " ")
    if name_norm in t:
        return 1.0
    if obj_text and any(a.lower() in t for a in obj_text.split(", ")):
        return 0.8
    categories = ["combustible", "vegetation", "flammable", "fuel", "dead"]
    if any(c in t for c in categories):
        return 0.6
    return 0.4

for req in CALFIRE_REQUIREMENTS:
    req_id = req["calfire_id"]
    for obj_mention in req["object_mentions"]:
        # Map generic aliases to actual graph node names
        resolved = OBJECT_ALIASES.get(obj_mention, [obj_mention])
        for obj_id in resolved:
            if obj_id in G:
                obj_attrs = G.nodes[obj_id]
                aliases_str = obj_attrs.get("aliases", "")
                conf = calc_confidence(obj_id, aliases_str, req["requirement_text"])
                add_edge_once(G, obj_id, req_id, relation="subject_to", confidence=conf)

# calfire_req → violation: defines
CALFIRE_VIOLATION_MAP = {
    "CALFIRE_Z1_B" : ["Roof_debris_present", "Gutter_not_cleared", "Debris_accumulation"],
    "CALFIRE_Z1_C" : ["Dead_vegetation_present"],
    "CALFIRE_Z1_D" : ["Dead_vegetation_present", "Debris_accumulation"],
    "CALFIRE_Z1_E" : ["Tree_crown_overlap", "Vegetation_within_Zone0"],
    "CALFIRE_Z1_F" : ["Combustible_within_Zone0", "Debris_accumulation"],
    "CALFIRE_Z1_G" : ["Firewood_within_30ft"],
    "CALFIRE_Z2_H" : ["Dead_vegetation_present"],
    "CALFIRE_Z2_I" : ["Tree_crown_overlap"],
    "CALFIRE_Z2_J" : ["Firewood_within_30ft"],
    "CALFIRE_Z2_K" : ["Dead_vegetation_present", "Debris_accumulation"],
    "CALFIRE_Z2_L" : ["Debris_accumulation"],
    "CALFIRE_Z2_M" : ["Shed_too_close_to_home", "Propane_within_5ft"],
    "CALFIRE_Z0_001": ["Vegetation_within_Zone0", "Combustible_within_Zone0", "Mulch_within_Zone0"],
    "CALFIRE_Z0_005": ["Gutter_not_cleared", "Roof_debris_present"],
}

for calfire_id, viols in CALFIRE_VIOLATION_MAP.items():
    if calfire_id in G:
        for v in viols:
            if v in G:
                add_edge_once(G, calfire_id, v, relation="defines")

print(f"  Added {added_edges} new edges")
print(f"  Graph now: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — EXPORT UPDATED GRAPH
# ══════════════════════════════════════════════════════════════════════
print("\n[5] Exporting updated graph …")

# GraphML
G_export = G.copy()
for n, d in G_export.nodes(data=True):
    for k, v in list(d.items()):
        if isinstance(v, (list, dict, bool)):
            G_export.nodes[n][k] = str(v)
for u, v, d in G_export.edges(data=True):
    for k, val in list(d.items()):
        if isinstance(val, (list, dict, bool)):
            G_export[u][v][k] = str(val)

nx.write_graphml(G_export, OUT["graphml"])
print(f"  GraphML → {OUT['graphml']}")

# Nodes JSON
nodes_data = [{"id": n, **dict(d)} for n, d in G.nodes(data=True)]
with open(OUT["nodes_json"], "w") as f:
    json.dump(nodes_data, f, indent=2, default=str)
print(f"  Nodes JSON → {OUT['nodes_json']}")

# Edges JSON
edges_data = [{"source": u, "target": v, **dict(d)} for u, v, d in G.edges(data=True)]
with open(OUT["edges_json"], "w") as f:
    json.dump(edges_data, f, indent=2, default=str)
print(f"  Edges JSON → {OUT['edges_json']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — UPDATE AERIAL DETECTABILITY PARTITION
# ══════════════════════════════════════════════════════════════════════
print("\n[6] Updating aerial detectability partition …")

# Collect all regulation nodes (IBHS + CALFIRE)
all_regs = []
for n, d in G.nodes(data=True):
    if d.get("type") == "regulation":
        src = d.get("source", "")
        if src in ("IBHS_WFPH", "CALFIRE_LE100A"):
            all_regs.append({
                "reg_id"            : n,
                "source"            : src,
                "zone"              : d.get("zone", d.get("zone_scope", "ALL")),
                "requirement_text"  : d.get("full_text", "")[:200],
                "aerial_detectable" : d.get("aerial_detectable", False),
                "severity"          : d.get("severity", "—"),
            })

df_detect = pd.DataFrame(all_regs)
total = len(df_detect)
aerial_pct = df_detect["aerial_detectable"].apply(lambda x: x in (True, "True")).mean() * 100
print(f"  Total reqs (IBHS + CALFIRE): {total}")
print(f"  Aerial detectable          : {aerial_pct:.1f}%")

df_detect.to_excel(OUT["detect_xlsx"], index=False)
print(f"  Saved → {OUT['detect_xlsx']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 7 — UPDATE EDGE COVERAGE TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n[7] Updating edge coverage table …")

SEV_ORDER = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3,"—":4}
edge_rows = []

for obj_node in [n for n,d in G.nodes(data=True) if d.get("type")=="object"]:
    obj_attrs = G.nodes[obj_node]
    obj_zone  = obj_attrs.get("typical_zone","—")

    prc_edges    = [(v,d) for u,v,d in G.out_edges(obj_node, data=True) if d.get("relation")=="governed_by"]
    ibhs_edges   = [(v,d) for u,v,d in G.out_edges(obj_node, data=True) if d.get("relation")=="subject_to"
                    and G.nodes.get(v,{}).get("source")=="IBHS_WFPH"]
    calfire_edges= [(v,d) for u,v,d in G.out_edges(obj_node, data=True) if d.get("relation")=="subject_to"
                    and G.nodes.get(v,{}).get("source")=="CALFIRE_LE100A"]
    viol_edges   = [(v,d) for u,v,d in G.out_edges(obj_node, data=True) if d.get("relation")=="can_cause"]

    all_reg_edges = prc_edges + ibhs_edges + calfire_edges
    if not all_reg_edges and not viol_edges:
        edge_rows.append({
            "object_class":"[NO EDGES] "+obj_node,"zone":obj_zone,
            "prc_section":"—","ibhs_requirement":"—","calfire_requirement":"—",
            "violation_type":"—","severity":"—","aerial_detectable":"—",
            "confidence_score":"—","source":"—"
        })
        continue

    prc_list     = prc_edges     if prc_edges     else [("—", {})]
    ibhs_list    = ibhs_edges    if ibhs_edges    else [("—", {})]
    calfire_list = calfire_edges if calfire_edges else [("—", {})]

    for (prc_id, prc_d) in prc_list[:2]:
        for (ibhs_id, ibhs_d) in ibhs_list[:2]:
            for (cal_id, cal_d) in calfire_list[:2]:
                viols = [v for v,_ in viol_edges]
                viol_str = "; ".join(viols[:2]) if viols else "—"
                sev_list = [d.get("severity","—") for _,d in viol_edges]
                sev = min(sev_list, key=lambda s: SEV_ORDER.get(s,99)) if sev_list else "—"
                # aerial from ibhs or calfire
                aerial = "—"
                if ibhs_id != "—":
                    aerial = G.nodes.get(ibhs_id,{}).get("aerial_detectable","—")
                if aerial == "—" and cal_id != "—":
                    aerial = G.nodes.get(cal_id,{}).get("aerial_detectable","—")
                conf = cal_d.get("confidence", ibhs_d.get("confidence", prc_d.get("confidence","—")))
                source_parts = []
                if prc_id  != "—": source_parts.append("PRC_4291")
                if ibhs_id != "—": source_parts.append("IBHS_WFPH")
                if cal_id  != "—": source_parts.append("CALFIRE_LE100A")
                edge_rows.append({
                    "object_class"       : obj_node,
                    "zone"               : obj_zone,
                    "prc_section"        : prc_id,
                    "ibhs_requirement"   : ibhs_id,
                    "calfire_requirement": cal_id,
                    "violation_type"     : viol_str,
                    "severity"           : sev,
                    "aerial_detectable"  : aerial,
                    "confidence_score"   : round(conf, 2) if isinstance(conf, float) else conf,
                    "source"             : " + ".join(source_parts) if source_parts else "—",
                })

df_edges = pd.DataFrame(edge_rows)
df_edges["_sev_ord"]  = df_edges["severity"].map(lambda s: SEV_ORDER.get(s,99))
df_edges["_zone_ord"] = df_edges["zone"].map({"Zone_0":0,"Zone_1":1,"Zone_2":2,"ALL":3,"—":4})
df_edges = df_edges.sort_values(["_sev_ord","_zone_ord"]).drop(columns=["_sev_ord","_zone_ord"])
df_edges.to_excel(OUT["edge_xlsx"], index=False)
print(f"  {len(df_edges)} rows → {OUT['edge_xlsx']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 8 — REGENERATE VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════
print("\n[8] Regenerating visualizations …")

NODE_COLORS = {
    "object"    : "#4A90D9",
    "zone"      : "#F5A623",
    "violation" : "#9B59B6",
}
REG_COLORS = {
    "PRC_4291"      : "#E74C3C",
    "IBHS_WFPH"     : "#27AE60",
    "CALFIRE_LE100A": "#E67E22",   # orange — new source
}

def node_color(nid, attrs):
    t = attrs.get("type","")
    if t == "object":    return NODE_COLORS["object"]
    if t == "zone":      return NODE_COLORS["zone"]
    if t == "violation": return NODE_COLORS["violation"]
    return REG_COLORS.get(attrs.get("source",""), "#95A5A6")

# Build subgraph for viz: all zones, violations, objects, key regs
key_nodes = set()
for n, d in G.nodes(data=True):
    t = d.get("type","")
    if t in ("zone","violation","object"):
        key_nodes.add(n)
    elif t == "regulation":
        key_nodes.add(n)

Gsub = G.subgraph(key_nodes).copy()
degrees = dict(Gsub.degree())
max_deg = max(degrees.values()) if degrees else 1
node_sizes = [200 + 800*(degrees.get(n,0)/max_deg) for n in Gsub.nodes()]
colors = [node_color(n, Gsub.nodes[n]) for n in Gsub.nodes()]

fig, ax = plt.subplots(figsize=(32, 24))
ax.set_facecolor("#0d1117")
fig.patch.set_facecolor("#0d1117")
pos = nx.spring_layout(Gsub, k=2.5, iterations=60, seed=42)

nx.draw_networkx_edges(Gsub, pos, ax=ax,
                        alpha=0.20, edge_color="#aaaaaa",
                        arrows=True, arrowsize=8,
                        connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(Gsub, pos, ax=ax,
                        node_color=colors, node_size=node_sizes, alpha=0.9)
nx.draw_networkx_labels(Gsub, pos, ax=ax,
                         font_size=4.5, font_color="white", font_weight="bold")

legend_elements = [
    mpatches.Patch(color=NODE_COLORS["object"],        label="Object"),
    mpatches.Patch(color=NODE_COLORS["zone"],          label="HIZ Zone"),
    mpatches.Patch(color=REG_COLORS["PRC_4291"],       label="PRC 4291"),
    mpatches.Patch(color=REG_COLORS["IBHS_WFPH"],      label="IBHS WFPH"),
    mpatches.Patch(color=REG_COLORS["CALFIRE_LE100A"], label="CAL FIRE LE-100a"),
    mpatches.Patch(color=NODE_COLORS["violation"],     label="Violation"),
]
ax.legend(handles=legend_elements, loc="upper left",
          facecolor="#1a1a2e", labelcolor="white", fontsize=10)
ax.set_title("HIZ Wildfire Knowledge Graph (+ CAL FIRE LE-100a)", color="white",
             fontsize=16, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT["static_png"], dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Static PNG → {OUT['static_png']}")

# Interactive HTML
net = Network(height="900px", width="100%", bgcolor="#0d1117",
              font_color="white", directed=True)
net.set_options("""
{
  "physics": {"solver":"forceAtlas2Based","forceAtlas2Based":{"springLength":120}},
  "interaction": {"hover":true,"tooltipDelay":100},
  "edges": {"smooth":{"type":"curvedCW","roundness":0.2},
             "arrows":{"to":{"enabled":true,"scaleFactor":0.5}},
             "color":{"inherit":"from"}},
  "nodes": {"borderWidth":1.5}
}
""")

for nid, attrs in Gsub.nodes(data=True):
    color = node_color(nid, attrs)
    deg = degrees.get(nid, 0)
    size = 10 + deg * 2
    tip_lines = [f"<b>{nid}</b>", f"Type: {attrs.get('type','')}",
                 f"Source: {attrs.get('source','')}"]
    for k, v in attrs.items():
        if k not in ("type","source"):
            tip_lines.append(f"{k}: {str(v)[:80]}")
    net.add_node(nid, label=nid, color=color, size=size, title="<br>".join(tip_lines))

for u, v, d in Gsub.edges(data=True):
    rel = d.get("relation","")
    conf = d.get("confidence","")
    tip = f"<b>{rel}</b>"
    if conf: tip += f"<br>confidence: {conf}"
    net.add_edge(u, v, title=tip, label=rel if rel else "")

net.write_html(OUT["interactive"])
print(f"  Interactive HTML → {OUT['interactive']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 9 — REGENERATE VALIDATION REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n[9] Regenerating validation report …")

report_lines = []
def rprint(line=""):
    print(line)
    report_lines.append(line)

rprint("=" * 70)
rprint("HIZ WILDFIRE KNOWLEDGE GRAPH — VALIDATION REPORT (POST LE-100a UPDATE)")
rprint(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
rprint("=" * 70)

rprint("\n── NODE COUNTS ──")
type_counts = {}
for n, d in G.nodes(data=True):
    t = d.get("type","unknown")
    if t == "regulation":
        src = d.get("source","unknown")
        t = f"regulation:{src}"
    type_counts[t] = type_counts.get(t,0) + 1
for t, c in sorted(type_counts.items()):
    rprint(f"  {t:45s}: {c}")
rprint(f"  {'TOTAL':45s}: {G.number_of_nodes()}")

rprint("\n── EDGE COUNTS BY RELATION ──")
rel_counts = {}
for u, v, d in G.edges(data=True):
    r = d.get("relation","unknown")
    rel_counts[r] = rel_counts.get(r,0) + 1
for r, c in sorted(rel_counts.items(), key=lambda x: -x[1]):
    rprint(f"  {r:45s}: {c}")
rprint(f"  {'TOTAL':45s}: {G.number_of_edges()}")

rprint("\n── CALFIRE LE-100A REQUIREMENTS ADDED ──")
calfire_nodes = [(n,d) for n,d in G.nodes(data=True) if d.get("source")=="CALFIRE_LE100A"]
rprint(f"  Total: {len(calfire_nodes)} requirements")
for nid, d in sorted(calfire_nodes, key=lambda x: x[0]):
    zone = d.get("zone","—")
    sev  = d.get("severity","—")
    label= d.get("item_label","")
    rprint(f"  [{zone:6s}/{sev:8s}] {nid:25s}  (Item {label})")

rprint("\n── NEW CALFIRE → VIOLATION CONNECTIONS ──")
for nid, d in sorted(calfire_nodes, key=lambda x: x[0]):
    viols = [v for _,v,ed in G.out_edges(nid, data=True) if ed.get("relation")=="defines"]
    if viols:
        rprint(f"  {nid:25s} → {', '.join(viols)}")

rprint("\n── OBJECTS GAINING NEW CALFIRE EDGES ──")
for obj_node in sorted([n for n,d in G.nodes(data=True) if d.get("type")=="object"]):
    calfire_reqs = [v for _,v,ed in G.out_edges(obj_node, data=True)
                    if ed.get("relation")=="subject_to"
                    and G.nodes.get(v,{}).get("source")=="CALFIRE_LE100A"]
    if calfire_reqs:
        rprint(f"  {obj_node:35s}: {len(calfire_reqs)} CALFIRE reqs")

rprint("\n── REGULATORY SOURCE COVERAGE PER OBJECT ──")
for obj_node in sorted([n for n,d in G.nodes(data=True) if d.get("type")=="object"]):
    sources = set()
    for _,v,ed in G.out_edges(obj_node, data=True):
        src = G.nodes.get(v,{}).get("source","")
        if src: sources.add(src)
    if sources:
        rprint(f"  {obj_node:35s}: {', '.join(sorted(sources))}")

rprint("\n── VIOLATIONS PER ZONE ──")
for z in ["Zone_0","Zone_1","Zone_2"]:
    viols = [u for u,v,d in G.in_edges(z, data=True) if d.get("relation")=="occurs_in"]
    rprint(f"  {z}: {len(viols)} violations")

rprint("\n── OUTPUT FILES ──")
for k, path in OUT.items():
    exists = "✓" if os.path.exists(path) else "✗ MISSING"
    rprint(f"  {exists}  {os.path.basename(path)}")

rprint("\n" + "="*70)

with open(OUT["validation"], "w") as f:
    f.write("\n".join(report_lines))
print(f"\n  Validation report → {OUT['validation']}")

print("\n" + "="*70)
print("CALFIRE LE-100A INGESTION COMPLETE")
print("="*70)
