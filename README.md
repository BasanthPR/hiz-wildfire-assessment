# AI-Assisted Wildfire HIZ Defensible-Space Assessment

This repository contains the complete research codebase for an AI-assisted wildfire home-ignition zone (HIZ) compliance assessment system developed at the Wildfire Interdisciplinary Research Center (WIRC), San Jose State University. The work addresses the practical challenge of scaling defensible-space inspection across fire-prone California communities: human inspection is neither fast enough nor consistent enough to cover the hundreds of thousands of parcels that require annual review in State Responsibility Areas and Very High Fire Hazard Severity Zones.

The system combines a regulatory knowledge graph, encoding three major fire-safety regulatory instruments into a structured, queryable graph, with two complementary vision inference pipelines: a fast CLIP zero-shot classifier for large-scale screening and an autoregressive VLM pipeline with structured chain-of-thought reasoning for deep compliance assessment and inspector triage.


## Project Overview

### Regulatory Framework

California's defensible-space law operates through three concentric zones around any structure in a fire-hazard area:

- **Zone 0 (0 to 5 ft)**: immediate noncombustible zone; no combustible material of any kind is permitted
- **Zone 1 (5 to 30 ft)**: intermediate defensible-space zone; governs vegetation spacing, firewood, accessory structures, and combustible objects
- **Zone 2 (30 to 100 ft)**: extended fuel-reduction zone; focuses on vegetation thinning and ladder-fuel removal

Three regulatory instruments define specific compliance requirements across these zones: Public Resources Code Section 4291 (PRC 4291), the IBHS Wildfire Prepared Home Technical Standard (IBHS WFPH), and the CAL FIRE LE-100a Notice of Defensible Space Inspection form. All three are encoded in the knowledge graph.


### System Components

The repository is organized into two top-level components, each with its own detailed documentation.

**Root directory: Knowledge graph construction and CLIP zero-shot pipeline**

The root-level scripts build a directed knowledge graph (346 nodes, 9,603 edges) linking 33 residential object classes to their governing regulations and potential violation types, then run a CLIP ViT-L/14-336 zero-shot classification pipeline over 4-band drone ortho-mosaics to identify which objects are present in each zone and generate structured compliance findings.

**`hiz_pipeline/` subdirectory: Autoregressive VLM inference pipeline**

The `hiz_pipeline/` scripts implement a Geo-CoT (geographic chain-of-thought) prompting framework for autoregressive VLMs (Qwen2.5-VL-7B-Instruct, InternVL2-8B, GeoChat-7B), include a Graph-RAG retrieval step that injects zone-specific regulatory text into the system prompt, and provide ablation infrastructure for comparing the Graph-RAG condition against a plain-prompt baseline. A scalability extension applies the same pipeline to freely available NAIP 60 cm public aerial imagery.


## Study Sites

Five WUI communities in California, covering a range of fire history, climate regime, and regulatory compliance context:

| Site code | Location | Context |
|---|---|---|
| fel | Felton (Santa Cruz Mountains) | SCU Lightning Complex burn scar |
| par | Paradise (Butte County) | 2018 Camp Fire destruction zone |
| red | Red Zone (Shasta-Trinity area) | High fire-hazard severity designation |
| sar | Santa Rosa (Coffey Park area) | 2017 Tubbs Fire area |
| tah | Tahoe Donner (Nevada County) | SRA, active defensible-space compliance program |


## Repository Structure

```
.
├── README.md                          # This file — project-level overview
│
├── build_knowledge_graph.py           # Knowledge graph construction (Steps 1-10)
├── ingest_calfire_le100a.py           # CAL FIRE LE-100a incremental ingestion
├── graph_rag_lookup.py                # Graph-RAG query API
├── vlm_inference_pipeline.py          # CLIP zero-shot inference loop (33 classes)
├── clip_tahoe_parcels.py              # Tahoe parcel clipping preprocessor
├── preannotate_groundtruth.py         # OWLv2 pre-annotation for Label Studio
├── analyze_inference_results.py       # Manuscript analysis of CLIP inference output
├── launch_labelstudio.py              # Label Studio session launcher
│
├── knowledge_graph_nodes.json         # Serialized graph nodes (346 nodes)
├── knowledge_graph_edges.json         # Serialized graph edges (9,603 edges)
├── knowledge_graph.graphml            # GraphML export for Gephi / Cytoscape
├── knowledge_graph_static.png         # Static visualization
├── knowledge_graph_interactive.html   # Interactive PyVis visualization
├── knowledge_graph_edge_coverage.xlsx # Object x regulation coverage table
│
├── calfire_le100a_requirements.json   # CAL FIRE LE-100a inspection items
├── ibhs_requirements.json             # IBHS Wildfire Prepared Home requirements
├── prc4291_sections.json              # PRC 4291 parsed sub-sections
├── prc4291_raw.txt                    # PRC 4291 full statutory text
├── lab_taxonomy.json                  # 33-class object taxonomy with aliases
├── aerial_detectability_partition.xlsx # Aerial vs. ground-only requirement split
│
├── graph_validation_report.txt        # Validation report (node/edge counts, gaps)
├── knowledge_graph_explanation.txt    # Narrative explanation of the graph structure
│
├── lib/                               # Third-party JS (vis.js, tom-select)
├── requirements.txt                   # Python dependencies
│
└── hiz_pipeline/                      # Autoregressive VLM inference pipeline
    ├── README.md                      # hiz_pipeline component documentation
    ├── preprocess.py                  # Tile drone orthos, build manifest
    ├── preprocess_naip.py             # Tile NAIP public imagery
    ├── preprocess_naip_sr.py          # Tile NAIP super-resolution imagery
    ├── download_public_imagery.py     # Fetch NAIP tiles from Planetary Computer
    ├── filter_tiles.py                # Filter manifest to information-rich tiles
    ├── prompts.py                     # Geo-CoT prompt builder, self-consistency voter
    ├── run_qwen25vl.py                # Qwen2.5-VL-7B-Instruct inference
    ├── run_qwen25vl_parallel.py       # Parallel shard runner
    ├── run_qwen25vl_naip.py           # Qwen2.5-VL on NAIP imagery
    ├── run_qwen25vl_naip_sr.py        # Qwen2.5-VL on NAIP super-resolution
    ├── run_geochat.py                 # GeoChat-7B inference
    ├── run_internvl.py                # InternVL2-8B inference
    ├── run_naip_ollama.py             # Ollama-based local LLM on NAIP
    ├── run_ablation.py                # Ablation: plain-prompt (no Graph-RAG)
    ├── run_ablation_ollama.py         # Ablation via Ollama runner
    ├── aggregate_parallel_results.py  # Merge parallel inference shards
    ├── aggregate_naip_results.py      # Merge NAIP inference results
    ├── annotate.py                    # Overlay bounding boxes on tile PNGs
    ├── evaluate.py                    # Generate results report
    ├── run_full_pipeline.sh           # Master sequential runner
    ├── setup_env.sh                   # One-time environment setup
    ├── knowledge_graph/               # Focused regulatory graph (5-class)
    │   ├── build_graph.py
    │   ├── graph_rag_lookup.py
    │   └── hiz_graph.graphml
    └── results/
        ├── RESULTS_REPORT.md          # Generated results report (all models)
        └── results_summary.json       # Aggregate statistics
```


## Knowledge Graph: Summary

The root-level knowledge graph encodes the full regulatory landscape for the 33-class object taxonomy:

| Source | Node type | Count |
|---|---|---|
| Lab taxonomy | Object | 33 |
| PRC 4291 | Regulation | 14 |
| IBHS WFPH | Regulation | 25 |
| CAL FIRE LE-100a | Regulation | 22 |
| HIZ zones | Zone | 3 |
| Violation types | Violation | 24 |

Approximately 88% of IBHS and CAL FIRE requirements are classified as aerially detectable, establishing the theoretical ceiling for what any drone-based inspection system can assess without ground-level follow-up.


## CLIP Pipeline: Key Results

Zero-shot detection with CLIP ViT-L/14-336 over 45 parcel ortho-mosaics from the five study sites. Detection threshold calibrated at cosine similarity 0.285 for aerial nadir imagery.

- 33 object classes from the lab taxonomy
- 512 x 512 px chips at 50% overlap per parcel
- Zone assigned per chip from CHM Band 4 centre-patch heuristic
- Per-chip compliance findings generated via Graph-RAG lookup


## VLM Pipeline: Key Results

Qwen2.5-VL-7B-Instruct with Geo-CoT prompting and Graph-RAG context injection over the same parcel set.

| Object class | Total detections | Zone 0 |
|---|---|---|
| Propane tank | 494 | 283 |
| Trash can | 687 | 134 |
| Vehicle | 266 | 27 |

**Ablation (Graph-RAG vs plain prompt):** Graph-RAG condition increases trash can recall 6.4x and raises regulatory clause citation rate in chain-of-thought traces from 0% to 91.6%.


## Excluded from This Repository

The following are not committed because they constitute private research data, copyrighted standards, or large binary artifacts:

- Raw drone ortho-mosaics (45 parcels, approximately 20 GB, consented private data — available under data-sharing agreement)
- Drone tile images for VLM inference (hiz_pipeline/tiles/)
- NAIP tile images (hiz_pipeline/tiles_naip/, hiz_pipeline/tiles_naip_sr/)
- Per-tile inference result JSONs for all models
- Annotated tile images with detection overlays
- Pre-annotation ground truth imagery and COCO labels (preannotations/)
- In-situ field survey data
- Manuscript drafts
- Copyrighted technical standards (IBHS WFPH, CAL FIRE LE-100a form PDFs)


## Setup

### Knowledge Graph and CLIP Pipeline

```bash
conda create -n hiz python=3.13 && conda activate hiz
pip install -r requirements.txt
python build_knowledge_graph.py
python ingest_calfire_le100a.py
python vlm_inference_pipeline.py
```

### Autoregressive VLM Pipeline

```bash
bash ~/hiz_pipeline/setup_env.sh
source ~/hiz_venv/bin/activate
bash ~/hiz_pipeline/run_full_pipeline.sh
```

Full setup and usage instructions for each component are in:
- Root `requirements.txt` and the detailed inline README below
- `hiz_pipeline/README.md` for the VLM inference pipeline

---

## Detailed Documentation: Knowledge Graph and CLIP Pipeline

### Knowledge Graph Construction

`build_knowledge_graph.py` constructs a directed NetworkX knowledge graph that encodes the regulatory landscape as a typed, attributed graph. The graph contains four node types:

- **Object nodes** (33 classes): residential objects drawn from the lab taxonomy, including vegetation categories (live herb, live shrub, live tree, dead vegetation, mulch), combustible storage (woodpile, propane, garbage bin, BBQ grill, welcome mat, planters), vehicles and large equipment (car, RV, boat), structural elements (deck/patio, fence, storage shed, pergola/gazebo, ladder, play set), and miscellaneous items
- **Zone nodes** (3): Zone_0, Zone_1, Zone_2, each carrying its distance range and descriptive label
- **Regulation nodes** (variable): one node per sub-section parsed from PRC 4291, one node per IBHS requirement, and one node per CAL FIRE LE-100a inspection item
- **Violation nodes** (24): named compliance failures detectable from aerial imagery, each carrying a severity level and a detection method flag

Edges encode seven relation types: `typically_found_in`, `covered_by`, `governed_by`, `subject_to`, `can_cause`, `defines`, and `occurs_in`. Confidence scores on object-to-regulation edges are assigned by lexical matching (1.0 verbatim, 0.8 alias match, 0.6 generic fuel category, 0.4 floor).

`ingest_calfire_le100a.py` extends the graph with 22 CAL FIRE LE-100a items and runs after the base build.

### Graph-RAG Lookup

```python
from graph_rag_lookup import get_regulatory_context

ctx = get_regulatory_context("propane", "Zone_0")
print(ctx["vlm_prompt"])     # pre-formatted natural language prompt fragment
print(ctx["violations"])     # applicable violation types with severity
print(ctx["prc_sections"])   # matched PRC 4291 sub-sections
```

### CLIP Inference Pipeline

```bash
python vlm_inference_pipeline.py                     # run on all parcels
python vlm_inference_pipeline.py --parcels 10        # limit to first 10
python vlm_inference_pipeline.py --threshold 0.30    # adjust detection threshold
```

### OWLv2 Pre-Annotation for Ground Truth

```bash
python preannotate_groundtruth.py --parcels 10 --threshold 0.12
```

Produces PNG chips and COCO JSON files under `preannotations/` for import into Label Studio.

### Results Analysis

```bash
python analyze_inference_results.py
```

Writes `vlm_inference_analysis.txt` and `vlm_inference_figures.xlsx` with manuscript-ready tables.


## Open Questions and Contribution Opportunities

1. **Ground-truth annotation**: the Label Studio annotation workflow is active. Priority classes are Zone 0 objects (propane, mulch, garbage bin, welcome mat, furniture) where false positives carry the highest cost. OWLv2 pre-annotations are in `preannotations/` locally.

2. **InternVL2 and GeoChat calibration**: both models currently produce zero detections. This is a prompt format issue requiring per-model calibration of the JSON output specification, not a model capability limitation.

3. **CHM zone assignment**: the current Zone_0/Zone_1 assignment uses a simple CHM centre-patch threshold. Using building footprint polygons to define exact radial zones per parcel would improve precision.

4. **GeoChat evaluation**: the research proposal envisioned GeoChat-7B (fine-tuned on remote sensing imagery) as the primary model; the VLM prompt fragments produced by `get_regulatory_context` were designed for this evaluation.

5. **Super-resolution NAIP**: `hiz_pipeline/preprocess_naip_sr.py` and `hiz_pipeline/run_qwen25vl_naip_sr.py` implement upscaling before inference. Whether super-resolution meaningfully improves detection at 60 cm GSD is pending evaluation.

6. **Multi-source regulation integration**: NFPA 1144 and community-level IBHS standards are the next regulatory layers for graph ingestion.

7. **Temporal change detection**: pairing two ortho-mosaics from different years and differencing the violation maps would reveal whether compliance is improving or degrading over time.


## Citation

If you use this codebase or the knowledge graph artifacts in published work, please cite the associated manuscript (in preparation) and this repository.


## Contact

Wildfire Interdisciplinary Research Center (WIRC)
San Jose State University
