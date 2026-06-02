# Wildfire HIZ Defensible-Space Assessment: Graph-RAG and VLM Inference Pipeline

This repository contains the research codebase for an AI-assisted wildfire home-ignition zone (HIZ) assessment system developed at the Wildfire Interdisciplinary Research Center (WIRC), San Jose State University. The work addresses a core challenge in wildland-urban interface (WUI) fire management: at scale, human inspection of residential defensible space is neither fast enough nor consistent enough to keep pace with annual regulatory compliance cycles across fire-prone communities.

The approach encodes three major bodies of fire-safety regulation into a structured knowledge graph, then queries that graph automatically during vision-language model (VLM) inference over drone orthomosaic imagery. The result is a per-parcel compliance report that identifies which objects are present, which regulations they implicate, and which violations they represent, without requiring a human inspector to be physically present.


## Research Context

California's Public Resources Code Section 4291 and its derivative instruments, the IBHS Wildfire Prepared Home Standard, and CAL FIRE's LE-100a defensible-space inspection form collectively define what constitutes a compliant residential parcel in a State Responsibility Area (SRA) or Very High Fire Hazard Severity Zone (VHFHSZ). These regulations operate at three concentric clearance zones radiating outward from a structure:

- Zone 0 (0 to 5 ft): the immediate noncombustible zone, where no combustible material of any kind is permitted
- Zone 1 (5 to 30 ft): the intermediate defensible-space zone, governing vegetation spacing, firewood storage, accessory structures, and combustible objects
- Zone 2 (30 to 100 ft): the extended fuel-reduction zone, focused on vegetation thinning and ladder-fuel removal

Aerial drone imagery, particularly 4-band ortho-mosaics with an embedded canopy height model (CHM) in Band 4, provides a nadir view of the parcel that captures most of the object categories regulated in Zone 0 and Zone 1 with sufficient spatial resolution at a GSD of approximately 5 cm per pixel. This project formalizes that observational alignment into a machine-readable compliance-checking framework.


## System Architecture

The pipeline proceeds through four major stages.

### Stage 1: Knowledge Graph Construction

`build_knowledge_graph.py` constructs a directed NetworkX knowledge graph that encodes the regulatory landscape as a typed, attributed graph. The graph contains four node types:

- **Object nodes** (33 classes): residential objects drawn from the lab taxonomy, including vegetation categories (live herb, live shrub, live tree, dead vegetation, mulch), combustible storage (woodpile, propane tank, garbage bin, BBQ grill, welcome mat, planters), vehicles and large equipment (car, RV, boat), structural elements (deck/patio, fence, storage shed, pergola/gazebo, ladder, play set), and miscellaneous items
- **Zone nodes** (3): Zone_0, Zone_1, Zone_2, each carrying its distance range and descriptive label
- **Regulation nodes** (variable): one node per sub-section parsed from PRC 4291 (lettered and numbered sub-items), one node per IBHS Wildfire Prepared Home requirement, and one node per CAL FIRE LE-100a inspection item
- **Violation nodes** (24): named compliance failures that can be detected from aerial imagery, each carrying a severity level (CRITICAL, HIGH, MEDIUM, LOW) and a detection method (aerial or ground-only)

Edges encode seven relation types: `typically_found_in` (object to zone), `covered_by` (zone to regulation), `governed_by` (object to PRC section), `subject_to` (object to IBHS or CALFIRE requirement), `can_cause` (object to violation), `defines` (regulation to violation), and `occurs_in` (violation to zone).

Confidence scores on `governed_by` and `subject_to` edges are assigned by lexical matching between the object name and its registered aliases against the regulatory requirement text:

- 1.0 if the object name appears verbatim in the text
- 0.8 if any registered alias matches
- 0.6 if the text contains a generic fuel category ("combustible material", "vegetation", etc.)
- 0.4 as a floor for all object-regulation pairs within the same zone

The graph produced after the base build and the CAL FIRE LE-100a incremental ingestion step contains 346 nodes and 9,603 edges.

### Stage 2: CAL FIRE LE-100a Incremental Ingestion

`ingest_calfire_le100a.py` extends the knowledge graph with requirements extracted from the CAL FIRE LE-100a (08/23) Notice of Defensible Space Inspection form. This script is designed to run after `build_knowledge_graph.py` and updates the serialized graph artifacts in place. It adds 22 new regulation nodes (items A through O on the form plus seven Zone 0 home-hardening items) and wires them to the existing object and violation nodes using the same edge schema as the base build.

### Stage 3: Graph-RAG Lookup Module

`graph_rag_lookup.py` exposes a single public function, `get_regulatory_context(object_class, zone)`, that takes a detected object class and a zone label and returns a dictionary containing all applicable PRC 4291 sections, all applicable IBHS requirements, all associated violations with severity levels, an aerial detectability flag, and a pre-formatted natural-language prompt fragment suitable for passing to an autoregressive vision-language model.

The function resolves common aliases (e.g., "firewood" to "woodpile", "propane tank" to "propane", "hot tub" to "above_ground_pool_or_hot_tub") before querying the graph, so downstream callers do not need to normalize object names themselves.

### Stage 4: CLIP Zero-Shot Inference Pipeline

`vlm_inference_pipeline.py` implements the primary inference loop. It processes 4-band parcel GeoTIFFs using CLIP ViT-L/14-336 (openai/clip-vit-large-patch14-336) in a sliding-window fashion:

1. The RGB bands are linearly stretched to uint8 using the 2nd and 98th percentile of valid pixel values per band
2. The CHM (Band 4) is retained in float32 for zone assignment
3. 512 x 512 pixel chips are extracted at a 256 pixel stride (50% overlap)
4. Chips where more than 60% of pixels are blank or uniform are skipped
5. For each valid chip, the centre 128 x 128 pixel region of the CHM is examined: a mean height above 1.5 m indicates proximity to a structure and triggers Zone_0 assignment; otherwise Zone_1 is assigned
6. CLIP text embeddings for all 33 classes are pre-computed once per session using three prompt templates per class ("aerial drone photo of ...", "overhead view of ... near a house", "nadir view showing ... from above") and their mean cosine similarity defines the per-class score
7. Classes whose maximum cosine similarity across prompt variants exceeds the calibrated threshold of 0.285 are recorded as detected
8. For each detected object, `get_regulatory_context` is called and the compliance finding is appended to the chip record
9. Results are periodically flushed to disk to support resume on interruption

The pipeline runs on Apple Silicon MPS where available and falls back to CPU.

### Supplementary Scripts

`clip_tahoe_parcels.py` preprocesses the Tahoe Donner area tiles: it filters buildings by footprint area (minimum 800 sq ft to exclude sheds and garages), clusters touching or nearby buildings into parcels using single-linkage at a 50 ft centroid distance, clips the RGB tile to each parcel bounding box with a 164 ft (50 m) buffer, rasterizes building footprints to produce a synthetic CHM, and stacks the result into a 4-band GeoTIFF compatible with the inference pipeline.

`preannotate_groundtruth.py` implements an OWLv2-based (google/owlv2-base-patch16-ensemble) bounding-box pre-annotation pass over the parcel orthomosaics. OWLv2 was selected over GroundingDINO because GroundingDINO requires a CUDA-compiled C++ extension for deformable attention that does not build on Apple Silicon without a CUDA toolchain; OWLv2 is pure Python and PyTorch and runs correctly on MPS. The script outputs per-tile PNG images and COCO-format JSON annotation files ready for import into Label Studio for human verification.

`analyze_inference_results.py` reads the saved inference results and produces manuscript-quality analysis tables: overall detection and violation rates, aerial-detectable versus ground-only violation split driven by the graph's `aerial_detectable` flags, per-zone violation distributions, per-site risk profiles across the five study areas (Felton, Paradise, Red Zone, Santa Rosa, Tahoe Donner), object class frequency ranked by IBHS severity, and IBHS requirement coverage across detected objects.

`launch_labelstudio.py` provides a local Label Studio launch helper for the annotation workflow.


## Study Sites

The inference pipeline has been applied to drone orthomosaics from five study areas in California:

| Site code | Location | Context |
|-----------|----------|---------|
| fel | Felton (Santa Cruz Mountains) | SCU Lightning Complex burn scar |
| par | Paradise (Butte County) | 2018 Camp Fire destruction zone |
| red | Red Zone (Shasta/Trinity area) | High fire-hazard severity designation |
| sar | Santa Rosa (Coffey Park area) | 2017 Tubbs Fire area |
| tah | Tahoe Donner (Nevada County) | SRA, active defensible-space compliance program |


## Repository Structure

```
.
├── build_knowledge_graph.py       # Knowledge graph construction (Steps 1-10)
├── ingest_calfire_le100a.py       # CAL FIRE LE-100a incremental ingestion
├── graph_rag_lookup.py            # Graph-RAG query API
├── vlm_inference_pipeline.py      # CLIP zero-shot inference loop
├── clip_tahoe_parcels.py          # Tahoe parcel clipping preprocessor
├── preannotate_groundtruth.py     # OWLv2 pre-annotation for Label Studio
├── analyze_inference_results.py   # Manuscript analysis of inference output
├── launch_labelstudio.py          # Label Studio annotation session launcher
│
├── knowledge_graph_nodes.json     # Serialized graph nodes (346 nodes)
├── knowledge_graph_edges.json     # Serialized graph edges (9,603 edges)
├── knowledge_graph.graphml        # GraphML export for Gephi or Cytoscape
├── knowledge_graph_static.png     # Static visualization (spring layout)
├── knowledge_graph_interactive.html # Interactive PyVis visualization
├── knowledge_graph_edge_coverage.xlsx # Object x regulation coverage table
│
├── calfire_le100a_requirements.json  # CAL FIRE LE-100a inspection items
├── ibhs_requirements.json            # IBHS Wildfire Prepared Home requirements
├── prc4291_sections.json             # PRC 4291 parsed sub-sections
├── prc4291_raw.txt                   # PRC 4291 full statutory text
├── lab_taxonomy.json                 # 33-class object taxonomy with aliases
├── aerial_detectability_partition.xlsx # Aerial vs. ground-only requirement split
│
├── graph_validation_report.txt     # Validation report (node/edge counts, gaps)
├── knowledge_graph_explanation.txt # Narrative explanation of the graph
│
├── lib/                            # Third-party JS (vis.js, tom-select)
│
├── requirements.txt
└── README.md
```


## Excluded from This Repository

The following files are excluded because they constitute private research data, copyrighted technical standards, or parcel-level geospatial data that is not redistributable under this repository's terms:

- Raw in-situ field survey data (2024 HIZ In-Situ Data for Caitlin.xlsx)
- Drone orthomosaic GeoTIFFs (hiz_data/henri/ - approximately 45 parcels, ~20 GB)
- Pre-annotation imagery tiles (preannotations/images/)
- COCO ground-truth annotations (preannotations/annotations/, ground_truth_coco.json)
- Inference output files (vlm_inference_results.json, vlm_inference_summary.xlsx)
- Manuscript drafts
- Copyrighted PDF standards (IBHS WFPH Technical Standard, CAL FIRE LE-100a form)
- Parcel spatial index (tahoe_parcel_index.json)


## Setup and Reproducibility

### Environment

The codebase was developed and validated on Apple Silicon (M-series) macOS with Python 3.13 and Miniconda. A CUDA-equipped Linux machine is also supported; the pipeline will automatically use MPS (Apple) or CPU as available.

```bash
conda create -n hiz python=3.13
conda activate hiz
pip install -r requirements.txt
```

### Running the Knowledge Graph Build

```bash
python build_knowledge_graph.py
python ingest_calfire_le100a.py
```

`build_knowledge_graph.py` fetches PRC 4291 text from Justia, parses the IBHS WFPH Technical Standard PDF (if present as `WFPH-Technical-Standard.pdf`), loads the lab taxonomy from the in-situ survey workbook, constructs the NetworkX DiGraph, and writes all output files. The IBHS curated requirements embedded in the script serve as a fallback if the PDF is absent. `ingest_calfire_le100a.py` extends the graph with CAL FIRE LE-100a items.

### Running the Graph-RAG Lookup

```python
from graph_rag_lookup import get_regulatory_context

ctx = get_regulatory_context("propane", "Zone_0")
print(ctx["vlm_prompt"])
print(ctx["violations"])
```

The module loads the serialized node and edge JSON files lazily on the first call and caches them in memory for subsequent queries.

### Running the CLIP Inference Pipeline

The pipeline expects parcel GeoTIFFs in a directory configured as `HENRI_DIR` inside the script (default: `/Users/basanthyajman/hiz_data/henri`). Each file must follow the `*cliptoparcel.tif` naming convention and carry RGB in Bands 1-3 and a CHM in Band 4.

```bash
python vlm_inference_pipeline.py
python vlm_inference_pipeline.py --parcels 10          # limit to first 10 for testing
python vlm_inference_pipeline.py --no-resume            # start from scratch
python vlm_inference_pipeline.py --threshold 0.30       # adjust detection sensitivity
```

Results are written incrementally to `vlm_inference_results.json` and a summary Excel workbook.

### Preprocessing Tahoe Donner Parcels

If working with the raw Tahoe Donner area tiles and building footprint shapefile:

```bash
python clip_tahoe_parcels.py --max-parcels 60 --buffer 164
```

This writes parcel-clipped 4-band GeoTIFFs to the Henri data directory alongside the existing parcel clips from other sites.

### Generating Ground-Truth Pre-Annotations

```bash
python preannotate_groundtruth.py --parcels 10 --threshold 0.12
```

This creates PNG chips and COCO JSON files under `preannotations/` for import into Label Studio.

### Analyzing Inference Results

```bash
python analyze_inference_results.py
```

This produces `vlm_inference_analysis.txt` (a prose summary) and `vlm_inference_figures.xlsx` (all tables in manuscript-ready format).


## Knowledge Graph: Key Statistics

The graph encodes the regulatory overlap across three authoritative sources:

| Source | Node type | Count |
|--------|-----------|-------|
| Lab taxonomy | Object | 33 |
| PRC 4291 | Regulation | 14 |
| IBHS WFPH | Regulation | 25 |
| CAL FIRE LE-100a | Regulation | 22 |
| HIZ zones | Zone | 3 |
| Violation types | Violation | 24 |

Edge distribution by relation:

| Relation | Count |
|----------|-------|
| subject_to | ~4,800 |
| governed_by | ~3,400 |
| covered_by | ~200 |
| can_cause | ~90 |
| defines | ~80 |
| typically_found_in | ~50 |
| occurs_in | ~24 |

Approximately 88% of IBHS and CAL FIRE requirements are classified as aerially detectable, meaning a drone nadir view provides sufficient information to assess compliance without ground-level inspection. This figure forms a central claim of the manuscript.


## Extending the Graph

The graph is designed for incremental extension. To add a new regulatory source:

1. Define a list of requirement dictionaries following the schema in `ingest_calfire_le100a.py` (fields: `calfire_id`, `zone`, `requirement_text`, `object_mentions`, `aerial_detectable`, `requirement_type`, `severity`)
2. Load the existing graph from the serialized JSON files
3. Add regulation nodes and wire edges using the established relation types
4. Export the updated graph back to JSON and GraphML

To add a new object class:

1. Add an entry to the `OBJECT_COLS` dictionary in `build_knowledge_graph.py` with aliases, typical zone, aerial visibility, and approximate size
2. Add the class to `OBJECT_CLASSES` in `vlm_inference_pipeline.py` and `preannotate_groundtruth.py`
3. Optionally add a human-readable display label to `OBJECT_DISPLAY`
4. Rebuild the graph

The alias resolver in `graph_rag_lookup.py` should also be updated to map common alternative names to the canonical class label.


## Open Questions and Next Steps

The following are active areas of investigation and natural entry points for new contributors:

1. **Ground-truth validation**: The OWLv2 pre-annotations in the preannotations directory need human review and correction in Label Studio before any precision/recall analysis can be computed for the CLIP pipeline. The priority queue is Zone 0 objects (propane, mulch, garbage bin, welcome mat, furniture) where false positives have the highest cost in a compliance context.

2. **CHM zone assignment refinement**: The current Zone_0/Zone_1 split is a simple threshold on the centre-chip CHM mean. A more principled assignment would use the building footprint polygons directly to define the exact radial zones for each parcel.

3. **GeoChat evaluation**: The original research proposal envisioned evaluating GeoChat-7B (a vision-language model fine-tuned on remote sensing imagery) as an alternative to CLIP's zero-shot classification. The VLM prompt fragments produced by `get_regulatory_context` were designed to feed directly into that evaluation.

4. **Multi-source regulation integration**: The graph currently covers PRC 4291, IBHS WFPH, and CAL FIRE LE-100a. The Insurance Institute for Business and Home Safety (IBHS) has published additional community-level standards; the NFPA 1144 standard represents another major regulatory layer that could be ingested using the same incremental architecture.

5. **Temporal change detection**: A single-epoch inference pass produces a static compliance snapshot. Pairing two ortho-mosaics from different seasons or years and differencing the violation maps would reveal whether household compliance is improving or degrading over time.


## Citation

If you use this codebase or the knowledge graph artifacts in published work, please cite the associated manuscript (in preparation) and this repository.


## Contact

Wildfire Interdisciplinary Research Center (WIRC)
San Jose State University
