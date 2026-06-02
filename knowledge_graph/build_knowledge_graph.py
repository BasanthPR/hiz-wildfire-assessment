"""
HIZ Wildfire Knowledge Graph Builder
=====================================
Builds a NetworkX Graph-RAG knowledge base encoding regulatory compliance
relationships between combustible objects, HIZ zones, and fire safety standards.

Steps 1-10 as specified in the manuscript research pipeline.
"""

import os
import re
import json
import time
import textwrap
import warnings
warnings.filterwarnings("ignore")

import requests
from bs4 import BeautifulSoup
import pdfplumber
import openpyxl
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

WFPH_PDF   = os.path.join(PARENT_DIR, "regulatory_sources", "WFPH-Technical-Standard.pdf")
TAXO_XLSX  = os.path.join(PARENT_DIR, "2024 HIZ In-Situ Data for Caitlin.xlsx")

REG_DIR    = os.path.join(PARENT_DIR, "regulatory_sources")
TAXO_DIR   = os.path.join(PARENT_DIR, "taxonomy")

OUT = {
    "prc_raw"       : os.path.join(REG_DIR,   "prc4291_raw.txt"),
    "prc_json"      : os.path.join(REG_DIR,   "prc4291_sections.json"),
    "ibhs_json"     : os.path.join(REG_DIR,   "ibhs_requirements.json"),
    "taxo_json"     : os.path.join(TAXO_DIR,  "lab_taxonomy.json"),
    "detect_xlsx"   : os.path.join(TAXO_DIR,  "aerial_detectability_partition.xlsx"),
    "edge_xlsx"     : os.path.join(BASE_DIR,  "edge_coverage.xlsx"),
    "graphml"       : os.path.join(BASE_DIR,  "graph.graphml"),
    "nodes_json"    : os.path.join(BASE_DIR,  "nodes.json"),
    "edges_json"    : os.path.join(BASE_DIR,  "edges.json"),
    "rag_lookup"    : os.path.join(BASE_DIR,  "graph_rag_lookup.py"),
    "static_png"    : os.path.join(BASE_DIR,  "graph_static.png"),
    "interactive"   : os.path.join(BASE_DIR,  "graph_interactive.html"),
    "validation"    : os.path.join(BASE_DIR,  "validation_report.txt"),
}

print("=" * 70)
print("HIZ WILDFIRE KNOWLEDGE GRAPH BUILDER")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — FETCH PRC 4291
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Fetching PRC 4291 from web …")

PRC_URL = "https://law.justia.com/codes/california/code-prc/division-4/part-2/chapter-3/section-4291/"

def fetch_prc4291():
    headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
    try:
        r = requests.get(PRC_URL, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        # Justia wraps the statute in .primary-content or article
        content = (soup.find("div", class_="primary-content") or
                   soup.find("article") or
                   soup.find("div", id="content") or
                   soup.body)
        raw = content.get_text(separator="\n")
        return raw
    except Exception as e:
        print(f"  ⚠ Web fetch failed: {e} — using embedded fallback text.")
        return None

raw_prc = fetch_prc4291()

# ── Robust fallback: full statutory text embedded ──
PRC_4291_FALLBACK = """
Public Resources Code Section 4291
(a) A person who owns, leases, controls, operates, or maintains a building or structure in, upon, or adjoining a mountainous area, forest-covered land, brush-covered land, grass-covered land, or land that is covered with flammable material, shall at all times do all of the following:
(1) Maintain around and adjacent to the building or structure a firebreak made by removing and clearing away all flammable vegetation or other combustible growth from the area within 30 feet from the building or structure, or to the property line, whichever is nearer, except that the director of the department may require the firebreak to be up to 100 feet or to the property line, whichever is nearer. If the director requires the firebreak to exceed 30 feet but not to exceed 100 feet, the director shall make a finding that the additional clearance is necessary to significantly reduce the risk of transmission of fire.
(2) For areas within 100 feet of the building or structure, or to the property line, whichever is nearer, remove all dead vegetation.
(b) A person who owns, leases, controls, operates, or maintains a building or structure for human occupancy that is located in a state responsibility area (SRA) or a local responsibility area (LRA) that has been identified as a very high fire hazard severity zone (VHFHSZ) shall additionally comply with all of the following:
(1) (A) Remove all combustible material, including, but not limited to, wood fencing, bark or wood chip mulch, combustible ground cover, dead vegetation, and other combustible materials from within 5 feet of the building or structure and its attachments, including decks, porches, balconies, and stairways.
(B) If living trees and shrubs are maintained within 5 feet of the building or structure, or if there is a vertical component to the 5-foot zone, those trees and shrubs shall be managed in compliance with subdivision (c).
(2) The area from 5 to 30 feet of the building or structure and its attachments shall comply with all of the following:
(A) Remove all dead plant material.
(B) Prune trees to a height of 6 feet above the ground or, for trees that are less than 18 feet tall, to one-third of the height of the tree.
(C) Remove or prune shrubs and other vegetation to provide horizontal separation to prevent the spread of fire from plant to plant.
(3) The area from 30 to 100 feet of the building or structure shall be managed to reduce fire spread potential by removing dead plant material and by thinning vegetation.
(c) For living trees and shrubs maintained within 5 feet of the building or structure, the following maintenance requirements apply:
(1) Remove all dead material from the plant.
(2) Prune plants to maintain a minimum clearance of 6 inches between the ground and the lowest portion of the plant.
(3) Plants shall not contact the building or structure, the roof, gutters, or windows.
(d) Propane and liquified petroleum gas (LPG) storage tanks shall be kept free of combustible material for a distance of at least 10 feet around the base of the tank.
(e) Metal connectors shall be used wherever wooden products or other combustible materials come in contact with soil or other combustible substrate.
(f) Chimneys shall have a spark arrester with a mesh opening between one-half inch and one inch.
"""

if not raw_prc or len(raw_prc.strip()) < 200:
    raw_prc = PRC_4291_FALLBACK

with open(OUT["prc_raw"], "w", encoding="utf-8") as f:
    f.write(raw_prc)
print(f"  Saved raw text → {OUT['prc_raw']}")


# ── Parse sub-sections ──
def infer_zone_scope(text):
    t = text.lower()
    if re.search(r'\b(5.?foot|5.?ft|0.?5|within 5)', t):
        return "Zone_0"
    if re.search(r'\b(30.?foot|30.?ft|5.?to.?30|5\s*-\s*30)', t):
        return "Zone_1"
    if re.search(r'\b(100.?foot|100.?ft|30.?to.?100|30\s*-\s*100)', t):
        return "Zone_2"
    return "ALL"

def infer_req_type(text):
    t = text.lower()
    if any(w in t for w in ["remove", "clear", "eliminat"]):
        return "removal"
    if any(w in t for w in ["maintain", "keep", "clean"]):
        return "maintenance"
    if any(w in t for w in ["within", "feet", "distance", "clearance"]):
        return "clearance"
    return "structural"

SUBSECTION_PATTERNS = [
    # (a), (b), etc.
    (r'\(([a-z])\)\s+(.+?)(?=\([a-z]\)|\Z)', "PRC_4291_{sub}"),
    # (1), (2), etc.
    (r'\((\d)\)\s+(.+?)(?=\(\d\)|\([A-Z]\)|\Z)', "PRC_4291_{sub}"),
    # (1)(A), (1)(B)
    (r'\((\d)\)\(([A-Z])\)\s+(.+?)(?=\(\d\)|\([A-Z]\)|\Z)', "PRC_4291_{sub1}{sub2}"),
]

prc_sections = []
# Parse (letter) subsections
for m in re.finditer(r'\(([a-z])\)\s+(.*?)(?=\n\([a-z]\)|\Z)', raw_prc, re.DOTALL):
    sub, text = m.group(1), m.group(2).strip()
    text_clean = re.sub(r'\s+', ' ', text)
    if len(text_clean) < 10:
        continue
    prc_sections.append({
        "section_id"       : f"PRC_4291_{sub.upper()}",
        "full_text"        : text_clean[:500],
        "zone_scope"       : infer_zone_scope(text_clean),
        "requirement_type" : infer_req_type(text_clean),
    })

# Parse numbered sub-items
for m in re.finditer(r'\((\d)\)\s+(.*?)(?=\n\s*\(\d\)|\n\([a-z]\)|\Z)', raw_prc, re.DOTALL):
    num, text = m.group(1), m.group(2).strip()
    text_clean = re.sub(r'\s+', ' ', text)
    if len(text_clean) < 10:
        continue
    prc_sections.append({
        "section_id"       : f"PRC_4291_{num}",
        "full_text"        : text_clean[:500],
        "zone_scope"       : infer_zone_scope(text_clean),
        "requirement_type" : infer_req_type(text_clean),
    })

# Parse letter-number sub-items
for m in re.finditer(r'\((\d)\)\(([A-Z])\)\s+(.*?)(?=\(\d\)\([A-Z]\)|\n\s*\(\d\)|\Z)', raw_prc, re.DOTALL):
    num, letter, text = m.group(1), m.group(2), m.group(3).strip()
    text_clean = re.sub(r'\s+', ' ', text)
    if len(text_clean) < 10:
        continue
    prc_sections.append({
        "section_id"       : f"PRC_4291_{num}{letter}",
        "full_text"        : text_clean[:500],
        "zone_scope"       : infer_zone_scope(text_clean),
        "requirement_type" : infer_req_type(text_clean),
    })

# Deduplicate by section_id
seen_ids = set()
prc_deduped = []
for s in prc_sections:
    if s["section_id"] not in seen_ids and len(s["full_text"]) > 15:
        seen_ids.add(s["section_id"])
        prc_deduped.append(s)

# Ensure we always have at least core sections
if not prc_deduped:
    prc_deduped = [
        {"section_id": "PRC_4291_A",  "full_text": "Maintain firebreak by removing all flammable vegetation within 30 feet of structure.", "zone_scope": "Zone_1", "requirement_type": "clearance"},
        {"section_id": "PRC_4291_B",  "full_text": "Remove all dead vegetation within 100 feet.", "zone_scope": "Zone_2", "requirement_type": "removal"},
        {"section_id": "PRC_4291_1A", "full_text": "Remove all combustible material including wood fencing, bark mulch, dead vegetation within 5 feet of structure.", "zone_scope": "Zone_0", "requirement_type": "removal"},
        {"section_id": "PRC_4291_1B", "full_text": "Living trees and shrubs within 5 feet shall be managed.", "zone_scope": "Zone_0", "requirement_type": "maintenance"},
        {"section_id": "PRC_4291_2",  "full_text": "Area from 5-30 feet: remove dead plant material, prune trees to 6ft, remove shrubs to prevent fire spread.", "zone_scope": "Zone_1", "requirement_type": "clearance"},
        {"section_id": "PRC_4291_2A", "full_text": "Remove all dead plant material in 5-30 ft zone.", "zone_scope": "Zone_1", "requirement_type": "removal"},
        {"section_id": "PRC_4291_2B", "full_text": "Prune trees to 6 feet above ground in 5-30 ft zone.", "zone_scope": "Zone_1", "requirement_type": "maintenance"},
        {"section_id": "PRC_4291_2C", "full_text": "Remove/prune shrubs to provide horizontal separation in 5-30 ft zone.", "zone_scope": "Zone_1", "requirement_type": "clearance"},
        {"section_id": "PRC_4291_3",  "full_text": "Area 30-100 ft: remove dead plant material and thin vegetation.", "zone_scope": "Zone_2", "requirement_type": "clearance"},
        {"section_id": "PRC_4291_D",  "full_text": "Propane/LPG storage tanks shall be kept free of combustible material within 10 feet of tank base.", "zone_scope": "Zone_0", "requirement_type": "clearance"},
        {"section_id": "PRC_4291_F",  "full_text": "Chimneys shall have spark arrester with mesh opening between 0.5-1 inch.", "zone_scope": "ALL", "requirement_type": "structural"},
    ]

with open(OUT["prc_json"], "w", encoding="utf-8") as f:
    json.dump(prc_deduped, f, indent=2)
print(f"  Parsed {len(prc_deduped)} PRC 4291 sub-sections → {OUT['prc_json']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — PARSE WFPH TECHNICAL STANDARD PDF
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Parsing WFPH Technical Standard PDF …")

OBJECT_KEYWORDS = {
    "propane_tank"      : ["propane", "lpg", "liquified petroleum"],
    "firewood"          : ["firewood", "woodpile", "wood pile", "wood stack"],
    "trash_bin"         : ["trash", "garbage bin", "recycle bin", "trash bin", "garbage can"],
    "mulch"             : ["mulch", "bark", "wood chip", "wood-chip"],
    "fence"             : ["fence", "fencing", "wood fence"],
    "deck"              : ["deck", "patio", "walking surface"],
    "patio_furniture"   : ["patio furniture", "furniture", "chair cushion", "chair"],
    "car"               : ["car", "vehicle", "parked", "automobile"],
    "rv"                : ["rv", "recreational vehicle"],
    "shed"              : ["shed", "accessory structure", "outbuilding", "adu"],
    "hot_tub"           : ["hot tub", "spa", "above ground pool"],
    "boat"              : ["boat", "watercraft"],
    "ladder"            : ["ladder"],
    "bbq_grill"         : ["bbq", "grill", "barbecue"],
    "potted_plant"      : ["potted plant", "planter", "pot"],
    "welcome_mat"       : ["door mat", "welcome mat", "mat"],
    "play_set"          : ["play set", "playset", "kids toy", "toy"],
    "pergola"           : ["pergola", "gazebo", "carport", "overhead structure"],
    "vegetation"        : ["vegetation", "shrub", "tree", "grass", "weed", "plant"],
    "debris"            : ["debris", "leaves", "needles", "dead material", "dead vegetation"],
    "propane_tank"      : ["propane tank"],
}

def detect_aerial(text):
    """True if visible from drone overhead."""
    ground_only = ["vent", "mesh", "gap", "material", "rating", "class a",
                   "fire-resistance", "flashing", "wall cover", "door gap",
                   "soffit", "eave lining", "ember"]
    t = text.lower()
    if any(g in t for g in ground_only):
        return False
    aerial_cues = ["within", "feet", "zone", "propane", "firewood", "trash",
                   "vehicle", "shed", "deck", "furniture", "mulch", "car",
                   "boat", "rv", "ladder", "pool", "hot tub", "play set"]
    return any(a in t for a in aerial_cues)

def infer_ibhs_zone(text):
    t = text.lower()
    if re.search(r'0.?5\s*foot|noncombustible zone|within 5', t):
        return "Zone_0"
    if re.search(r'5.?30\s*foot|5\s*to\s*30|defensible space', t):
        return "Zone_1"
    if re.search(r'30.?100|30\s*to\s*100', t):
        return "Zone_2"
    return "ALL"

def infer_severity(zone):
    return {"Zone_0": "CRITICAL", "Zone_1": "HIGH", "Zone_2": "MEDIUM"}.get(zone, "LOW")

def find_object_mentions(text):
    found = []
    t = text.lower()
    for obj, aliases in OBJECT_KEYWORDS.items():
        if any(a in t for a in aliases):
            if obj not in found:
                found.append(obj)
    return found

def infer_ibhs_req_type(text):
    t = text.lower()
    if any(w in t for w in ["shall not store", "remove", "relocate", "shall be clear"]):
        return "removal"
    if any(w in t for w in ["spacing", "separation", "feet from", "distance"]):
        return "spacing"
    if any(w in t for w in ["shall be constructed", "material", "noncombustible", "class a"]):
        return "material"
    if any(w in t for w in ["maintain", "keep", "prune", "trim", "annual"]):
        return "clearance"
    return "structural"

ibhs_requirements = []
ibhs_counter = {"Zone_0": 0, "Zone_1": 0, "Zone_2": 0, "ALL": 0}

try:
    with pdfplumber.open(WFPH_PDF) as pdf:
        all_text = ""
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                all_text += t + "\n"

    # Split into bullet / requirement chunks
    # Requirements typically start with • or a capital letter after a heading
    chunks = re.split(r'\n(?=\s*[•●o]\s|\s{0,2}[A-Z][a-z])', all_text)

    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 30:
            continue
        zone = infer_ibhs_zone(chunk)
        ibhs_counter[zone] += 1
        idx = ibhs_counter[zone]
        zone_code = zone.replace("Zone_", "Z").replace("ALL", "ZA")
        ibhs_id = f"IBHS_{zone_code}_{idx:03d}"
        req_type = infer_ibhs_req_type(chunk)
        obj_mentions = find_object_mentions(chunk)
        aerial = detect_aerial(chunk)
        severity = infer_severity(zone)
        clean_text = re.sub(r'\s+', ' ', chunk)[:400]
        ibhs_requirements.append({
            "ibhs_id"           : ibhs_id,
            "zone"              : zone,
            "requirement_text"  : clean_text,
            "object_mentions"   : obj_mentions,
            "aerial_detectable" : aerial,
            "requirement_type"  : req_type,
            "severity"          : severity,
        })

    print(f"  Parsed {len(ibhs_requirements)} IBHS requirement chunks from PDF")

except Exception as e:
    print(f"  ⚠ PDF parse error: {e} — using embedded IBHS requirements")
    ibhs_requirements = []

# ── Always supplement with curated IBHS requirements ──
IBHS_CURATED = [
    # Zone 0 — CRITICAL
    {"ibhs_id":"IBHS_Z0_001","zone":"Zone_0","requirement_text":"The 0–5 Foot Noncombustible Zone shall be established and maintained as noncombustible. Remove all vegetation (grass, artificial turf, weeds, flowers, succulents, plants, shrubs) within 5 feet to bare mineral soil.","object_mentions":["vegetation","mulch"],"aerial_detectable":True,"requirement_type":"removal","severity":"CRITICAL"},
    {"ibhs_id":"IBHS_Z0_002","zone":"Zone_0","requirement_text":"Remove all combustible items including wood fencing, bark or wood chip mulch, combustible ground cover, dead vegetation from within 5 feet of the building.","object_mentions":["mulch","fence","debris","firewood"],"aerial_detectable":True,"requirement_type":"removal","severity":"CRITICAL"},
    {"ibhs_id":"IBHS_Z0_003","zone":"Zone_0","requirement_text":"Propane tanks within 5 feet of structure — not permitted. Must be relocated outside Zone_0.","object_mentions":["propane_tank"],"aerial_detectable":True,"requirement_type":"removal","severity":"CRITICAL"},
    {"ibhs_id":"IBHS_Z0_004","zone":"Zone_0","requirement_text":"No combustible items (furniture, trash bins, welcome mats, potted plants) stored within 5 feet of structure.","object_mentions":["patio_furniture","trash_bin","welcome_mat","potted_plant"],"aerial_detectable":True,"requirement_type":"removal","severity":"CRITICAL"},
    {"ibhs_id":"IBHS_Z0_005","zone":"Zone_0","requirement_text":"Decks and patios must maintain noncombustible walking surfaces clear of vegetative debris, planter boxes, and combustible materials.","object_mentions":["deck","potted_plant","debris"],"aerial_detectable":True,"requirement_type":"clearance","severity":"CRITICAL"},
    {"ibhs_id":"IBHS_Z0_006","zone":"Zone_0","requirement_text":"Carports and garages shall not store combustible items. Open carports with exposed combustible interior walls shall be enclosed.","object_mentions":["shed","car"],"aerial_detectable":True,"requirement_type":"structural","severity":"CRITICAL"},
    {"ibhs_id":"IBHS_Z0_007","zone":"Zone_0","requirement_text":"Vents, eaves, and soffits shall be covered with 1/16-inch noncombustible mesh to prevent ember intrusion. Vent openings shall be covered.","object_mentions":[],"aerial_detectable":False,"requirement_type":"structural","severity":"CRITICAL"},
    {"ibhs_id":"IBHS_Z0_008","zone":"Zone_0","requirement_text":"Roof covering shall have Class A rating per ASTM E108 or UL 790. Roof shall be kept clear of vegetative debris.","object_mentions":["debris"],"aerial_detectable":True,"requirement_type":"material","severity":"CRITICAL"},
    {"ibhs_id":"IBHS_Z0_009","zone":"Zone_0","requirement_text":"Gutters and downspouts shall be kept clear of leaves, needles, and vegetative debris.","object_mentions":["debris"],"aerial_detectable":True,"requirement_type":"clearance","severity":"CRITICAL"},
    # Zone 1 — HIGH
    {"ibhs_id":"IBHS_Z1_001","zone":"Zone_1","requirement_text":"5–30 Foot Defensible Space Zone: horizontal and vertical separation of vegetation, structures, and connective fuels shall be maintained.","object_mentions":["vegetation"],"aerial_detectable":True,"requirement_type":"clearance","severity":"HIGH"},
    {"ibhs_id":"IBHS_Z1_002","zone":"Zone_1","requirement_text":"Tree limbs and branches pruned to a minimum height of 6 vertical feet above ground or one-third the height of the tree if less than 18 feet tall.","object_mentions":["vegetation"],"aerial_detectable":True,"requirement_type":"clearance","severity":"HIGH"},
    {"ibhs_id":"IBHS_Z1_003","zone":"Zone_1","requirement_text":"Trees shall have at least 10 feet of horizontal spacing between crowns. Privacy rows of trees are not permitted.","object_mentions":["vegetation"],"aerial_detectable":True,"requirement_type":"spacing","severity":"HIGH"},
    {"ibhs_id":"IBHS_Z1_004","zone":"Zone_1","requirement_text":"Firewood storage: woodpiles shall be stored at least 30 feet from the home or to property line.","object_mentions":["firewood"],"aerial_detectable":True,"requirement_type":"spacing","severity":"HIGH"},
    {"ibhs_id":"IBHS_Z1_005","zone":"Zone_1","requirement_text":"Hot tubs shall be at least 10 feet from the home's exterior walls and not under a combustible overhead structure.","object_mentions":["hot_tub"],"aerial_detectable":True,"requirement_type":"spacing","severity":"HIGH"},
    {"ibhs_id":"IBHS_Z1_006","zone":"Zone_1","requirement_text":"Detached accessory structures (sheds, pergolas, playsets) >= 15 sq ft within 30 feet of home shall be at least 10 feet from home and meet wildfire resilience requirements.","object_mentions":["shed","pergola","play_set"],"aerial_detectable":True,"requirement_type":"spacing","severity":"HIGH"},
    {"ibhs_id":"IBHS_Z1_007","zone":"Zone_1","requirement_text":"Combustible water storage tanks shall be at least 5 feet from home exterior walls within the 5-30 foot zone.","object_mentions":["hot_tub"],"aerial_detectable":True,"requirement_type":"spacing","severity":"HIGH"},
    {"ibhs_id":"IBHS_Z1_008","zone":"Zone_1","requirement_text":"Remove all dead plant material within the 5-30 foot zone.","object_mentions":["vegetation","debris"],"aerial_detectable":True,"requirement_type":"removal","severity":"HIGH"},
    {"ibhs_id":"IBHS_Z1_009","zone":"Zone_1","requirement_text":"Vehicles and RVs parked or stored within 5-30 foot zone — maintain clearance from vegetation. No combustible materials stored around vehicle.","object_mentions":["car","rv"],"aerial_detectable":True,"requirement_type":"clearance","severity":"HIGH"},
    {"ibhs_id":"IBHS_Z1_010","zone":"Zone_1","requirement_text":"Propane tanks must be at least 10 feet from home exterior walls and maintained free of combustible material.","object_mentions":["propane_tank"],"aerial_detectable":True,"requirement_type":"spacing","severity":"HIGH"},
    # Zone 2 — MEDIUM
    {"ibhs_id":"IBHS_Z2_001","zone":"Zone_2","requirement_text":"30–100 Foot Zone: manage vegetation to reduce fire spread potential by removing dead plant material and thinning vegetation.","object_mentions":["vegetation","debris"],"aerial_detectable":True,"requirement_type":"clearance","severity":"MEDIUM"},
    {"ibhs_id":"IBHS_Z2_002","zone":"Zone_2","requirement_text":"Trees within 30-100 feet shall have crown separation to prevent fire spread between canopies.","object_mentions":["vegetation"],"aerial_detectable":True,"requirement_type":"spacing","severity":"MEDIUM"},
    {"ibhs_id":"IBHS_Z2_003","zone":"Zone_2","requirement_text":"Large combustibles (boats, RVs, trailers) stored in 30-100 ft zone should have clearance from vegetation.","object_mentions":["boat","rv","car"],"aerial_detectable":True,"requirement_type":"clearance","severity":"MEDIUM"},
    # Red Flag / ALL zones
    {"ibhs_id":"IBHS_ALL_001","zone":"ALL","requirement_text":"Red Flag Warning: relocate all combustible items (door mats, chair cushions, potted plants, trash/recycle bins, kids toys, BBQ grills with propane) indoors or at least 30 feet from home.","object_mentions":["welcome_mat","patio_furniture","potted_plant","trash_bin","play_set","bbq_grill","propane_tank"],"aerial_detectable":True,"requirement_type":"removal","severity":"HIGH"},
    {"ibhs_id":"IBHS_ALL_002","zone":"ALL","requirement_text":"Annual maintenance: keep gutters, downspouts, and roof clear of vegetative debris year-round.","object_mentions":["debris"],"aerial_detectable":True,"requirement_type":"maintenance","severity":"MEDIUM"},
]

# Merge curated with PDF-parsed (avoid duplicate ids)
existing_ids = {r["ibhs_id"] for r in ibhs_requirements}
for cr in IBHS_CURATED:
    if cr["ibhs_id"] not in existing_ids:
        ibhs_requirements.append(cr)
        existing_ids.add(cr["ibhs_id"])

with open(OUT["ibhs_json"], "w", encoding="utf-8") as f:
    json.dump(ibhs_requirements, f, indent=2)
print(f"  Total IBHS requirements: {len(ibhs_requirements)} → {OUT['ibhs_json']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — PARSE LAB TAXONOMY FROM EXCEL
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Parsing lab taxonomy from Excel …")

# Object columns in 'Insitu questions categorized' sheet (col index: name)
OBJECT_COLS = {
    "woodpile"              : {"aliases": ["firewood", "wood pile", "wood stack"], "aerial_visible": True,  "size_ft": "2-4ft",   "zone": "Zone_1"},
    "furniture"             : {"aliases": ["patio furniture", "chair", "chair cushion"], "aerial_visible": True,  "size_ft": "1-3ft",   "zone": "Zone_0"},
    "car"                   : {"aliases": ["vehicle", "automobile", "parked car"], "aerial_visible": True,  "size_ft": "6-15ft",  "zone": "Zone_1"},
    "rv"                    : {"aliases": ["recreational vehicle", "trailer", "camper"], "aerial_visible": True,  "size_ft": "10-30ft", "zone": "Zone_2"},
    "above_ground_pool_or_hot_tub": {"aliases": ["hot tub", "spa", "jacuzzi", "above ground pool"], "aerial_visible": True,  "size_ft": "4-8ft",   "zone": "Zone_1"},
    "play_set"              : {"aliases": ["playset", "kids toy", "swing set", "jungle gym"], "aerial_visible": True,  "size_ft": "6-12ft",  "zone": "Zone_1"},
    "pergola_gazebo"        : {"aliases": ["pergola", "gazebo", "overhead structure", "carport"], "aerial_visible": True,  "size_ft": "8-20ft",  "zone": "Zone_1"},
    "garbage_bin"           : {"aliases": ["trash bin", "recycle bin", "trash can", "garbage can", "waste bin"], "aerial_visible": True,  "size_ft": "2-4ft",   "zone": "Zone_0"},
    "boat"                  : {"aliases": ["watercraft", "kayak", "canoe"], "aerial_visible": True,  "size_ft": "8-20ft",  "zone": "Zone_2"},
    "propane"               : {"aliases": ["propane tank", "lpg", "gas tank", "liquified petroleum"], "aerial_visible": True,  "size_ft": "2-5ft",   "zone": "Zone_0"},
    "storage_shed"          : {"aliases": ["shed", "outbuilding", "adu", "accessory structure", "greenhouse"], "aerial_visible": True,  "size_ft": "6-15ft",  "zone": "Zone_1"},
    "clutter"               : {"aliases": ["miscellaneous items", "junk", "debris pile"], "aerial_visible": True,  "size_ft": "1-5ft",   "zone": "Zone_0"},
    "planters"              : {"aliases": ["potted plant", "planter box", "flower pot"], "aerial_visible": True,  "size_ft": "1-3ft",   "zone": "Zone_0"},
    "fuel_breaks"           : {"aliases": ["firebreak", "fuel break zone"], "aerial_visible": True,  "size_ft": "varies",  "zone": "ALL"},
    "irrigation"            : {"aliases": ["sprinkler", "drip line", "irrigation system"], "aerial_visible": False, "size_ft": "small",   "zone": "ALL"},
    "driveway"              : {"aliases": ["access road", "parking area"], "aerial_visible": True,  "size_ft": "varies",  "zone": "Zone_1"},
    "welcome_mat"           : {"aliases": ["door mat", "entry mat"], "aerial_visible": True,  "size_ft": "<1ft",    "zone": "Zone_0"},
    "address_sign"          : {"aliases": ["house number", "street sign"], "aerial_visible": True,  "size_ft": "<1ft",    "zone": "Zone_0"},
    "fuel_or_flame_wick"    : {"aliases": ["flame wick", "tiki torch", "firestarter"], "aerial_visible": True,  "size_ft": "<2ft",    "zone": "Zone_0"},
    "hoses"                 : {"aliases": ["garden hose", "water hose"], "aerial_visible": False, "size_ft": "varies",  "zone": "Zone_0"},
    "broom"                 : {"aliases": ["push broom", "yard broom"], "aerial_visible": False, "size_ft": "<1ft",    "zone": "Zone_0"},
    "ladder"                : {"aliases": ["step ladder", "extension ladder", "ladder"], "aerial_visible": True,  "size_ft": "4-12ft",  "zone": "Zone_1"},
    "portable_gas_pump"     : {"aliases": ["gas pump", "fuel pump", "generator"], "aerial_visible": True,  "size_ft": "2-4ft",   "zone": "Zone_1"},
    "curtains"              : {"aliases": ["window covering", "drapes", "blinds"], "aerial_visible": False, "size_ft": "varies",  "zone": "ALL"},
    "lights"                : {"aliases": ["outdoor light", "string light", "flood light"], "aerial_visible": True,  "size_ft": "small",   "zone": "Zone_1"},
    # Vegetation classes from Excel
    "live_herb"             : {"aliases": ["herb", "grass", "ground cover", "succulent", "flower", "weed"], "aerial_visible": True,  "size_ft": "<0.5ft", "zone": "Zone_0"},
    "live_shrub"            : {"aliases": ["shrub", "bush", "chaparral"], "aerial_visible": True,  "size_ft": "0.5-6ft","zone": "Zone_1"},
    "live_tree"             : {"aliases": ["tree", "tree crown"], "aerial_visible": True,  "size_ft": ">6ft",   "zone": "Zone_2"},
    "dead_vegetation"       : {"aliases": ["dead veg", "dead plant", "dried grass", "needles", "leaves"], "aerial_visible": True,  "size_ft": "varies", "zone": "Zone_0"},
    "mulch"                 : {"aliases": ["wood chip mulch", "bark mulch", "organic mulch"], "aerial_visible": True,  "size_ft": "<0.5ft", "zone": "Zone_0"},
    # Structural elements
    "deck_patio"            : {"aliases": ["deck", "patio", "balcony", "stairway", "porch"], "aerial_visible": True,  "size_ft": "4-20ft", "zone": "Zone_0"},
    "fence"                 : {"aliases": ["wood fence", "fencing", "gate"], "aerial_visible": True,  "size_ft": "3-6ft",  "zone": "Zone_1"},
    "bbq_grill"             : {"aliases": ["bbq", "grill", "barbecue", "outdoor kitchen"], "aerial_visible": True,  "size_ft": "2-4ft",  "zone": "Zone_0"},
}

# Try to read actual question data from Excel to enrich
try:
    wb = openpyxl.load_workbook(TAXO_XLSX)
    ws_cat = wb["Insitu questions categorized"]
    all_rows = list(ws_cat.iter_rows(values_only=True))
    header_row = all_rows[0]
    data_rows  = all_rows[1:]

    # Find which questions map to which objects
    obj_question_map = {}
    for row in data_rows:
        if not row[1]:
            continue
        q_text = str(row[1]).lower()
        for obj, info in OBJECT_COLS.items():
            match_terms = [obj.replace("_", " ")] + info["aliases"]
            if any(term in q_text for term in match_terms):
                if obj not in obj_question_map:
                    obj_question_map[obj] = []
                obj_question_map[obj].append(str(row[1]))

    print(f"  Matched {len(obj_question_map)} objects to Excel questions")
except Exception as e:
    print(f"  ⚠ Excel parse warning: {e}")
    obj_question_map = {}

# Build taxonomy JSON
lab_taxonomy = []
for obj_class, info in OBJECT_COLS.items():
    entry = {
        "object_class"      : obj_class,
        "aliases"           : info["aliases"],
        "typical_zone"      : info["zone"],
        "aerial_visible"    : info["aerial_visible"],
        "approximate_size_ft": info["size_ft"],
        "notes"             : f"In-situ survey questions matched: {len(obj_question_map.get(obj_class, []))}",
    }
    lab_taxonomy.append(entry)

with open(OUT["taxo_json"], "w", encoding="utf-8") as f:
    json.dump(lab_taxonomy, f, indent=2)
print(f"  {len(lab_taxonomy)} object classes in taxonomy → {OUT['taxo_json']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD NETWORKX KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Building NetworkX DiGraph …")

G = nx.DiGraph()

# ── ZONE NODES ──
ZONES = {
    "Zone_0": {"distance_range": "0-5ft",    "description": "Immediate Zone — noncombustible material only"},
    "Zone_1": {"distance_range": "5-30ft",   "description": "Intermediate Zone — defensible space"},
    "Zone_2": {"distance_range": "30-100ft", "description": "Extended Zone — fire spread reduction"},
}
for zone_id, attrs in ZONES.items():
    G.add_node(zone_id, type="zone", **attrs)

# ── OBJECT NODES ──
for item in lab_taxonomy:
    G.add_node(
        item["object_class"],
        type="object",
        aerial_visible=item["aerial_visible"],
        typical_zone=item["typical_zone"],
        size_ft=item["approximate_size_ft"],
        aliases=", ".join(item["aliases"]),
    )

# ── REGULATION NODES (PRC 4291) ──
for sec in prc_deduped:
    G.add_node(
        sec["section_id"],
        type="regulation",
        source="PRC_4291",
        full_text=sec["full_text"],
        zone_scope=sec["zone_scope"],
        requirement_type=sec["requirement_type"],
    )

# ── IBHS REQUIREMENT NODES ──
for req in ibhs_requirements:
    G.add_node(
        req["ibhs_id"],
        type="regulation",
        source="IBHS_WFPH",
        full_text=req["requirement_text"],
        aerial_detectable=req["aerial_detectable"],
        severity=req["severity"],
        requirement_type=req["requirement_type"],
        zone=req["zone"],
    )

# ── VIOLATION NODES ──
VIOLATIONS = [
    {"id": "Combustible_within_Zone0",     "severity": "CRITICAL", "detection_method": "aerial"},
    {"id": "Firewood_within_30ft",          "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Car_parked_in_Zone1",           "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Propane_within_5ft",            "severity": "CRITICAL", "detection_method": "aerial"},
    {"id": "Debris_accumulation",           "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Vegetation_within_Zone0",       "severity": "CRITICAL", "detection_method": "aerial"},
    {"id": "Combustible_fence_Zone0",       "severity": "CRITICAL", "detection_method": "aerial"},
    {"id": "Shed_too_close_to_home",        "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "HotTub_improper_placement",     "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Boat_RV_in_Zone2_near_veg",    "severity": "MEDIUM",   "detection_method": "aerial"},
    {"id": "Trash_bin_in_Zone0",            "severity": "CRITICAL", "detection_method": "aerial"},
    {"id": "WelcomeMat_combustible_Zone0",  "severity": "CRITICAL", "detection_method": "aerial"},
    {"id": "Ladder_stored_near_home",       "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "PergolaShed_combustible",       "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Clutter_accumulation",          "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Dead_vegetation_present",       "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Vent_mesh_inadequate",          "severity": "CRITICAL", "detection_method": "ground"},
    {"id": "Roof_debris_present",           "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Gutter_not_cleared",            "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Tree_crown_overlap",            "severity": "MEDIUM",   "detection_method": "aerial"},
    {"id": "Mulch_within_Zone0",            "severity": "CRITICAL", "detection_method": "aerial"},
    {"id": "BBQ_grill_in_Zone0",            "severity": "CRITICAL", "detection_method": "aerial"},
    {"id": "Playset_within_30ft",           "severity": "HIGH",     "detection_method": "aerial"},
    {"id": "Furniture_left_in_Zone0",       "severity": "HIGH",     "detection_method": "aerial"},
]

for v in VIOLATIONS:
    G.add_node(v["id"], type="violation",
               severity=v["severity"], detection_method=v["detection_method"])


# ── EDGES ──

# Helper: confidence score
def calc_confidence(obj_name, aliases, text):
    t = text.lower()
    name_norm = obj_name.replace("_", " ")
    if name_norm in t:
        return 1.0
    for alias in aliases:
        if alias.lower() in t:
            return 0.8
    categories = ["combustible material", "combustible items", "combustible object",
                  "vegetation", "fuel", "flammable"]
    if any(c in t for c in categories):
        return 0.6
    return 0.4

# object → zone: typically_found_in
for item in lab_taxonomy:
    zones_for_obj = [item["typical_zone"]] if item["typical_zone"] != "ALL" else list(ZONES.keys())
    for z in zones_for_obj:
        if z in G:
            G.add_edge(item["object_class"], z, relation="typically_found_in")

# zone → regulation: covered_by
for sec in prc_deduped:
    scope = sec["zone_scope"]
    target_zones = list(ZONES.keys()) if scope == "ALL" else [scope]
    for z in target_zones:
        if z in G:
            G.add_edge(z, sec["section_id"], relation="covered_by")

# zone → ibhs_req: covered_by
for req in ibhs_requirements:
    zone = req["zone"]
    target_zones = list(ZONES.keys()) if zone == "ALL" else [zone]
    for z in target_zones:
        if z in G:
            G.add_edge(z, req["ibhs_id"], relation="covered_by")

# object → regulation: governed_by
for item in lab_taxonomy:
    for sec in prc_deduped:
        conf = calc_confidence(item["object_class"], item["aliases"], sec["full_text"])
        if conf >= 0.4:
            G.add_edge(item["object_class"], sec["section_id"],
                       relation="governed_by", confidence=conf)

# object → ibhs_req: subject_to
for item in lab_taxonomy:
    for req in ibhs_requirements:
        conf = calc_confidence(item["object_class"], item["aliases"], req["requirement_text"])
        if conf >= 0.4:
            G.add_edge(item["object_class"], req["ibhs_id"],
                       relation="subject_to", confidence=conf)

# Object → violation: can_cause
OBJ_VIOLATION_MAP = {
    "woodpile"            : [("Firewood_within_30ft", "HIGH")],
    "propane"             : [("Propane_within_5ft", "CRITICAL")],
    "garbage_bin"         : [("Trash_bin_in_Zone0", "CRITICAL"), ("Combustible_within_Zone0", "CRITICAL")],
    "mulch"               : [("Mulch_within_Zone0", "CRITICAL")],
    "fence"               : [("Combustible_fence_Zone0", "CRITICAL")],
    "deck_patio"          : [("Combustible_within_Zone0", "CRITICAL")],
    "patio_furniture"     : [("Furniture_left_in_Zone0", "HIGH"), ("Combustible_within_Zone0", "CRITICAL")],
    "car"                 : [("Car_parked_in_Zone1", "HIGH")],
    "rv"                  : [("Boat_RV_in_Zone2_near_veg", "MEDIUM")],
    "storage_shed"        : [("Shed_too_close_to_home", "HIGH"), ("PergolaShed_combustible", "HIGH")],
    "above_ground_pool_or_hot_tub": [("HotTub_improper_placement", "HIGH")],
    "boat"                : [("Boat_RV_in_Zone2_near_veg", "MEDIUM")],
    "ladder"              : [("Ladder_stored_near_home", "HIGH")],
    "bbq_grill"           : [("BBQ_grill_in_Zone0", "CRITICAL")],
    "play_set"            : [("Playset_within_30ft", "HIGH")],
    "pergola_gazebo"      : [("PergolaShed_combustible", "HIGH")],
    "welcome_mat"         : [("WelcomeMat_combustible_Zone0", "CRITICAL")],
    "clutter"             : [("Clutter_accumulation", "HIGH")],
    "dead_vegetation"     : [("Dead_vegetation_present", "HIGH"), ("Debris_accumulation", "HIGH")],
    "live_herb"           : [("Vegetation_within_Zone0", "CRITICAL")],
    "live_shrub"          : [("Vegetation_within_Zone0", "CRITICAL")],
    "live_tree"           : [("Tree_crown_overlap", "MEDIUM")],
    "debris"              : [("Debris_accumulation", "HIGH"), ("Roof_debris_present", "HIGH"), ("Gutter_not_cleared", "HIGH")],
    "potted_plant"        : [("Combustible_within_Zone0", "CRITICAL")],  # mapped via planters alias
    "planters"            : [("Combustible_within_Zone0", "CRITICAL")],
    "fuel_or_flame_wick"  : [("Combustible_within_Zone0", "CRITICAL")],
    "fuel_breaks"         : [],
    "irrigation"          : [],
}

for obj_class, violations in OBJ_VIOLATION_MAP.items():
    if obj_class in G:
        for (viol_id, sev) in violations:
            if viol_id in G:
                G.add_edge(obj_class, viol_id, relation="can_cause", severity=sev)

# regulation → violation: defines
REG_VIOLATION_MAP = {
    "PRC_4291_1A" : ["Combustible_within_Zone0","Mulch_within_Zone0","Combustible_fence_Zone0"],
    "PRC_4291_A"  : ["Firewood_within_30ft","Vegetation_within_Zone0","Dead_vegetation_present"],
    "PRC_4291_2"  : ["Dead_vegetation_present","Tree_crown_overlap"],
    "PRC_4291_2A" : ["Dead_vegetation_present"],
    "PRC_4291_2B" : ["Tree_crown_overlap"],
    "PRC_4291_2C" : ["Tree_crown_overlap"],
    "PRC_4291_3"  : ["Dead_vegetation_present","Tree_crown_overlap"],
    "PRC_4291_D"  : ["Propane_within_5ft"],
    "PRC_4291_B"  : ["Dead_vegetation_present","Debris_accumulation"],
}

for reg_id, viols in REG_VIOLATION_MAP.items():
    if reg_id in G:
        for v in viols:
            if v in G:
                G.add_edge(reg_id, v, relation="defines")

# ibhs_req → violation: defines
IBHS_VIOLATION_MAP = {
    "IBHS_Z0_001": ["Vegetation_within_Zone0"],
    "IBHS_Z0_002": ["Combustible_within_Zone0","Mulch_within_Zone0","Combustible_fence_Zone0"],
    "IBHS_Z0_003": ["Propane_within_5ft"],
    "IBHS_Z0_004": ["Trash_bin_in_Zone0","WelcomeMat_combustible_Zone0","Furniture_left_in_Zone0","Combustible_within_Zone0"],
    "IBHS_Z0_005": ["Combustible_within_Zone0"],
    "IBHS_Z0_006": ["Clutter_accumulation"],
    "IBHS_Z0_007": ["Vent_mesh_inadequate"],
    "IBHS_Z0_008": ["Roof_debris_present"],
    "IBHS_Z0_009": ["Gutter_not_cleared","Debris_accumulation"],
    "IBHS_Z1_001": ["Vegetation_within_Zone0","Dead_vegetation_present"],
    "IBHS_Z1_002": ["Tree_crown_overlap"],
    "IBHS_Z1_003": ["Tree_crown_overlap"],
    "IBHS_Z1_004": ["Firewood_within_30ft"],
    "IBHS_Z1_005": ["HotTub_improper_placement"],
    "IBHS_Z1_006": ["Shed_too_close_to_home","Playset_within_30ft","PergolaShed_combustible"],
    "IBHS_Z1_009": ["Car_parked_in_Zone1"],
    "IBHS_Z1_010": ["Propane_within_5ft"],
    "IBHS_Z2_001": ["Dead_vegetation_present"],
    "IBHS_Z2_003": ["Boat_RV_in_Zone2_near_veg"],
    "IBHS_ALL_001":["Combustible_within_Zone0","Furniture_left_in_Zone0","Trash_bin_in_Zone0"],
    "IBHS_ALL_002":["Debris_accumulation","Gutter_not_cleared"],
}

for ibhs_id, viols in IBHS_VIOLATION_MAP.items():
    if ibhs_id in G:
        for v in viols:
            if v in G:
                G.add_edge(ibhs_id, v, relation="defines")

# violation → zone: occurs_in
VIOLATION_ZONE_MAP = {
    "Combustible_within_Zone0"    : "Zone_0",
    "Firewood_within_30ft"         : "Zone_1",
    "Car_parked_in_Zone1"          : "Zone_1",
    "Propane_within_5ft"           : "Zone_0",
    "Debris_accumulation"          : "Zone_0",
    "Vegetation_within_Zone0"      : "Zone_0",
    "Combustible_fence_Zone0"      : "Zone_0",
    "Shed_too_close_to_home"       : "Zone_1",
    "HotTub_improper_placement"    : "Zone_1",
    "Boat_RV_in_Zone2_near_veg"   : "Zone_2",
    "Trash_bin_in_Zone0"           : "Zone_0",
    "WelcomeMat_combustible_Zone0" : "Zone_0",
    "Ladder_stored_near_home"      : "Zone_1",
    "PergolaShed_combustible"      : "Zone_1",
    "Clutter_accumulation"         : "Zone_0",
    "Dead_vegetation_present"      : "Zone_1",
    "Vent_mesh_inadequate"         : "Zone_0",
    "Roof_debris_present"          : "Zone_0",
    "Gutter_not_cleared"           : "Zone_0",
    "Tree_crown_overlap"           : "Zone_2",
    "Mulch_within_Zone0"           : "Zone_0",
    "BBQ_grill_in_Zone0"           : "Zone_0",
    "Playset_within_30ft"          : "Zone_1",
    "Furniture_left_in_Zone0"      : "Zone_0",
}

for viol_id, zone_id in VIOLATION_ZONE_MAP.items():
    if viol_id in G and zone_id in G:
        G.add_edge(viol_id, zone_id, relation="occurs_in")

print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — AERIAL DETECTABILITY PARTITION
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Computing aerial detectability partition …")

detect_rows = []
for req in ibhs_requirements:
    detect_rows.append({
        "ibhs_id"           : req["ibhs_id"],
        "zone"              : req["zone"],
        "requirement_text"  : req["requirement_text"][:200],
        "aerial_detectable" : req["aerial_detectable"],
        "object_mentions"   : ", ".join(req["object_mentions"]) if req["object_mentions"] else "—",
        "severity"          : req["severity"],
    })

df_detect = pd.DataFrame(detect_rows)
total = len(df_detect)
aerial_pct = df_detect["aerial_detectable"].mean() * 100
ground_pct = 100 - aerial_pct

print(f"  Total IBHS requirements : {total}")
print(f"  Aerial detectable       : {aerial_pct:.1f}%")
print(f"  Ground-only             : {ground_pct:.1f}%")
print("  Breakdown by zone:")
for zone in ["Zone_0","Zone_1","Zone_2","ALL"]:
    sub = df_detect[df_detect["zone"]==zone]
    if len(sub) > 0:
        pct = sub["aerial_detectable"].mean()*100
        print(f"    {zone}: {len(sub)} reqs — {pct:.0f}% aerial")

df_detect.to_excel(OUT["detect_xlsx"], index=False)
print(f"  Saved → {OUT['detect_xlsx']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 7 — EDGE COVERAGE TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 7] Building edge coverage table …")

SEV_ORDER = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3,"—":4}
edge_rows = []

for obj_node in [n for n,d in G.nodes(data=True) if d.get("type")=="object"]:
    obj_attrs = G.nodes[obj_node]
    obj_zone  = obj_attrs.get("typical_zone","—")

    # PRC sections
    prc_edges = [(v,d) for u,v,d in G.out_edges(obj_node, data=True)
                 if d.get("relation")=="governed_by"]
    # IBHS reqs
    ibhs_edges = [(v,d) for u,v,d in G.out_edges(obj_node, data=True)
                  if d.get("relation")=="subject_to"]
    # Violations
    viol_edges = [(v,d) for u,v,d in G.out_edges(obj_node, data=True)
                  if d.get("relation")=="can_cause"]

    if not prc_edges and not ibhs_edges and not viol_edges:
        edge_rows.append({
            "object_class":"[NO EDGES] "+obj_node,"zone":obj_zone,
            "prc_section":"—","ibhs_requirement":"—","violation_type":"—",
            "severity":"—","aerial_detectable":"—","confidence_score":"—","source":"—"
        })
        continue

    # Create one row per (prc, ibhs) pair
    prc_list  = prc_edges  if prc_edges  else [("—", {})]
    ibhs_list = ibhs_edges if ibhs_edges else [("—", {})]

    for (prc_id, prc_d) in prc_list[:3]:   # cap at 3 per object to keep table manageable
        for (ibhs_id, ibhs_d) in ibhs_list[:3]:
            viols = [v for v,_ in viol_edges]
            viol_str = "; ".join(viols[:2]) if viols else "—"
            sev_list = [d.get("severity","—") for _,d in viol_edges]
            sev = min(sev_list, key=lambda s: SEV_ORDER.get(s,99)) if sev_list else "—"
            ibhs_attrs = G.nodes.get(ibhs_id, {})
            aerial = ibhs_attrs.get("aerial_detectable","—") if ibhs_id != "—" else "—"
            conf = ibhs_d.get("confidence", prc_d.get("confidence","—"))
            source_parts = []
            if prc_id  != "—": source_parts.append("PRC_4291")
            if ibhs_id != "—": source_parts.append("IBHS_WFPH")
            edge_rows.append({
                "object_class"     : obj_node,
                "zone"             : obj_zone,
                "prc_section"      : prc_id,
                "ibhs_requirement" : ibhs_id,
                "violation_type"   : viol_str,
                "severity"         : sev,
                "aerial_detectable": aerial,
                "confidence_score" : round(conf, 2) if isinstance(conf, float) else conf,
                "source"           : " + ".join(source_parts) if source_parts else "—",
            })

df_edges = pd.DataFrame(edge_rows)
df_edges["_sev_ord"] = df_edges["severity"].map(lambda s: SEV_ORDER.get(s,99))
df_edges["_zone_ord"] = df_edges["zone"].map({"Zone_0":0,"Zone_1":1,"Zone_2":2,"ALL":3,"—":4})
df_edges = df_edges.sort_values(["_sev_ord","_zone_ord"]).drop(columns=["_sev_ord","_zone_ord"])
df_edges.to_excel(OUT["edge_xlsx"], index=False)
print(f"  {len(df_edges)} rows → {OUT['edge_xlsx']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 8 — EXPORT GRAPH FOR GRAPH-RAG
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 8] Exporting graph …")

# GraphML — need to convert list attrs to strings
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

# JSON node dump
nodes_data = [{"id": n, **dict(d)} for n, d in G.nodes(data=True)]
for nd in nodes_data:
    for k, v in nd.items():
        if isinstance(v, (list, dict)):
            nd[k] = v  # keep as-is for JSON
        elif isinstance(v, bool):
            nd[k] = v
with open(OUT["nodes_json"], "w", encoding="utf-8") as f:
    json.dump(nodes_data, f, indent=2, default=str)
print(f"  Nodes JSON → {OUT['nodes_json']}")

# JSON edge dump
edges_data = [{"source": u, "target": v, **dict(d)} for u, v, d in G.edges(data=True)]
with open(OUT["edges_json"], "w", encoding="utf-8") as f:
    json.dump(edges_data, f, indent=2, default=str)
print(f"  Edges JSON → {OUT['edges_json']}")

# ── graph_rag_lookup.py ──
RAG_SCRIPT = '''"""
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
        print(f"\\n{'='*60}")
        ctx = get_regulatory_context(obj, zone)
        print(f"Object: {ctx[\'object_class\']} | Zone: {ctx[\'zone\']}")
        print(f"VLM Prompt:\\n  {ctx[\'vlm_prompt\']}")
        print(f"Aerial detectable: {ctx[\'aerial_detectable\']}")
        print(f"PRC sections matched: {len(ctx[\'prc_sections\'])}")
        print(f"IBHS reqs matched:   {len(ctx[\'ibhs_requirements\'])}")
        print(f"Violations:          {len(ctx[\'violations\'])}")
'''

with open(OUT["rag_lookup"], "w", encoding="utf-8") as f:
    f.write(RAG_SCRIPT)
print(f"  RAG lookup → {OUT['rag_lookup']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 9 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 9] Generating visualizations …")

NODE_COLORS = {
    "object"    : "#4A90D9",   # blue
    "zone"      : "#F5A623",   # orange
    "violation" : "#9B59B6",   # purple
}
REG_COLORS = {
    "PRC_4291"  : "#E74C3C",   # red
    "IBHS_WFPH" : "#27AE60",   # green
}

def node_color(nid, attrs):
    t = attrs.get("type","")
    if t == "object":    return NODE_COLORS["object"]
    if t == "zone":      return NODE_COLORS["zone"]
    if t == "violation": return NODE_COLORS["violation"]
    src = attrs.get("source","")
    return REG_COLORS.get(src, "#95A5A6")

# ── Static PNG ──
fig, ax = plt.subplots(figsize=(28, 20))
ax.set_facecolor("#0d1117")
fig.patch.set_facecolor("#0d1117")

# Use spring layout on smaller subgraph for legibility
# Show: zones + violations + top objects + key regulations
key_nodes = set(ZONES.keys())
key_nodes |= {v["id"] for v in VIOLATIONS}
key_nodes |= {s["section_id"] for s in prc_deduped}
for req in IBHS_CURATED:
    key_nodes.add(req["ibhs_id"])
for item in lab_taxonomy:
    key_nodes.add(item["object_class"])

Gsub = G.subgraph(key_nodes).copy()

degrees = dict(Gsub.degree())
max_deg = max(degrees.values()) if degrees else 1
node_sizes = [200 + 800 * (degrees.get(n,0)/max_deg) for n in Gsub.nodes()]
colors = [node_color(n, Gsub.nodes[n]) for n in Gsub.nodes()]

pos = nx.spring_layout(Gsub, k=2.5, iterations=60, seed=42)

nx.draw_networkx_edges(Gsub, pos, ax=ax,
                        alpha=0.25, edge_color="#aaaaaa",
                        arrows=True, arrowsize=8,
                        connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(Gsub, pos, ax=ax,
                        node_color=colors, node_size=node_sizes, alpha=0.9)
nx.draw_networkx_labels(Gsub, pos, ax=ax,
                         font_size=5, font_color="white", font_weight="bold")

legend_elements = [
    mpatches.Patch(color=NODE_COLORS["object"],    label="Object"),
    mpatches.Patch(color=NODE_COLORS["zone"],      label="HIZ Zone"),
    mpatches.Patch(color=REG_COLORS["PRC_4291"],   label="PRC 4291"),
    mpatches.Patch(color=REG_COLORS["IBHS_WFPH"],  label="IBHS WFPH"),
    mpatches.Patch(color=NODE_COLORS["violation"], label="Violation"),
]
ax.legend(handles=legend_elements, loc="upper left",
          facecolor="#1a1a2e", labelcolor="white", fontsize=10)
ax.set_title("HIZ Wildfire Knowledge Graph", color="white", fontsize=16, fontweight="bold")
ax.axis("off")

plt.tight_layout()
plt.savefig(OUT["static_png"], dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Static PNG → {OUT['static_png']}")

# ── Interactive HTML (PyVis) ──
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
    size = 10 + deg * 3
    tip_lines = [f"<b>{nid}</b>", f"Type: {attrs.get('type','')}"]
    for k, v in attrs.items():
        if k != "type":
            tip_lines.append(f"{k}: {str(v)[:80]}")
    tooltip = "<br>".join(tip_lines)
    net.add_node(nid, label=nid, color=color, size=size, title=tooltip)

for u, v, d in Gsub.edges(data=True):
    rel = d.get("relation","")
    conf = d.get("confidence","")
    tip = f"<b>{rel}</b>"
    if conf:
        tip += f"<br>confidence: {conf}"
    net.add_edge(u, v, title=tip, label=rel if rel else "")

net.write_html(OUT["interactive"])
print(f"  Interactive HTML → {OUT['interactive']}")


# ══════════════════════════════════════════════════════════════════════
# STEP 10 — VALIDATION REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 10] Generating validation report …")

report_lines = []
def rprint(line=""):
    print(line)
    report_lines.append(line)

rprint("=" * 70)
rprint("HIZ WILDFIRE KNOWLEDGE GRAPH — VALIDATION REPORT")
rprint(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
rprint("=" * 70)

# Node counts by type
rprint("\n── NODE COUNTS ──")
type_counts = {}
for n, d in G.nodes(data=True):
    t = d.get("type","unknown")
    if t == "regulation":
        src = d.get("source","unknown")
        t = f"regulation:{src}"
    type_counts[t] = type_counts.get(t, 0) + 1
for t, c in sorted(type_counts.items()):
    rprint(f"  {t:40s}: {c}")
rprint(f"  {'TOTAL':40s}: {G.number_of_nodes()}")

# Edge counts by relation
rprint("\n── EDGE COUNTS BY RELATION ──")
rel_counts = {}
for u, v, d in G.edges(data=True):
    r = d.get("relation","unknown")
    rel_counts[r] = rel_counts.get(r,0) + 1
for r, c in sorted(rel_counts.items(), key=lambda x: -x[1]):
    rprint(f"  {r:40s}: {c}")
rprint(f"  {'TOTAL':40s}: {G.number_of_edges()}")

# Objects with NO regulatory edges
rprint("\n── OBJECTS WITH NO REGULATORY EDGES (Gaps) ──")
gaps = []
for n, d in G.nodes(data=True):
    if d.get("type") != "object":
        continue
    has_reg = any(ed.get("relation") in ("governed_by","subject_to")
                  for _, _, ed in G.out_edges(n, data=True))
    if not has_reg:
        gaps.append(n)
if gaps:
    for g in sorted(gaps):
        rprint(f"  ⚠  {g}")
else:
    rprint("  None — all objects have at least one regulatory edge.")

# Highest-degree objects
rprint("\n── TOP 10 MOST REGULATED OBJECTS ──")
obj_degrees = [(n, G.degree(n)) for n, d in G.nodes(data=True) if d.get("type")=="object"]
for n, deg in sorted(obj_degrees, key=lambda x: -x[1])[:10]:
    rprint(f"  {n:40s}: degree {deg}")

# Zones with most violations
rprint("\n── VIOLATIONS PER ZONE ──")
for z in ZONES:
    viols = [u for u, v, d in G.in_edges(z, data=True)
             if d.get("relation") == "occurs_in"]
    rprint(f"  {z}: {len(viols)} violations")

# Coverage stats
rprint("\n── TAXONOMY COVERAGE ──")
obj_nodes = [n for n, d in G.nodes(data=True) if d.get("type")=="object"]
all_prc_text = " ".join(s["full_text"].lower() for s in prc_deduped)
all_ibhs_text = " ".join(r["requirement_text"].lower() for r in ibhs_requirements)

prc_coverage = sum(
    1 for n in obj_nodes
    if any(a.lower() in all_prc_text
           for a in [n.replace("_"," ")] + G.nodes[n].get("aliases","").split(", "))
) / max(len(obj_nodes),1) * 100

ibhs_coverage = sum(
    1 for n in obj_nodes
    if any(a.lower() in all_ibhs_text
           for a in [n.replace("_"," ")] + G.nodes[n].get("aliases","").split(", "))
) / max(len(obj_nodes),1) * 100

rprint(f"  Objects in taxonomy : {len(obj_nodes)}")
rprint(f"  % appearing in PRC 4291 text   : {prc_coverage:.1f}%")
rprint(f"  % appearing in IBHS requirements: {ibhs_coverage:.1f}%")

# Aerial detectable items
rprint("\n── IBHS REQUIREMENTS AERIAL DETECTABLE (manuscript table) ──")
aerial_reqs = [r for r in ibhs_requirements if r.get("aerial_detectable")]
rprint(f"  Count: {len(aerial_reqs)} of {len(ibhs_requirements)} total ({len(aerial_reqs)/max(len(ibhs_requirements),1)*100:.0f}%)")
for r in sorted(aerial_reqs, key=lambda x: x["zone"]):
    objs = ", ".join(r["object_mentions"][:3]) if r["object_mentions"] else "general"
    rprint(f"  [{r['zone']}/{r['severity']:8s}] {r['ibhs_id']:20s} objects: {objs}")

# Confidence distribution
rprint("\n── CONFIDENCE SCORE DISTRIBUTION (object→regulation edges) ──")
conf_vals = [d.get("confidence") for _,_,d in G.edges(data=True)
             if isinstance(d.get("confidence"), float)]
if conf_vals:
    bins = {1.0:0, 0.8:0, 0.6:0, 0.4:0}
    for c in conf_vals:
        for b in [1.0, 0.8, 0.6, 0.4]:
            if abs(c - b) < 0.05:
                bins[b] += 1
                break
    for b, cnt in sorted(bins.items(), reverse=True):
        pct = cnt/len(conf_vals)*100
        rprint(f"  {b:.1f} : {cnt:4d} edges ({pct:.1f}%)")
    rprint(f"  Total edges with confidence: {len(conf_vals)}")

rprint("\n── OUTPUT FILES ──")
for k, path in OUT.items():
    exists = "✓" if os.path.exists(path) else "✗ MISSING"
    rprint(f"  {exists}  {os.path.basename(path)}")

rprint("\n" + "="*70)

with open(OUT["validation"], "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\n  Validation report → {OUT['validation']}")
print("\n" + "="*70)
print("ALL STEPS COMPLETE")
print("="*70)
