# Wildfire HIZ Defensible-Space Assessment: Autoregressive VLM Inference Pipeline

This repository contains the autoregressive vision-language model (VLM) inference pipeline developed at the Wildfire Interdisciplinary Research Center (WIRC), San Jose State University, for AI-assisted wildfire home-ignition zone (HIZ) defensible-space compliance assessment. It is the companion to the CLIP zero-shot classification pipeline hosted in the AI-for-HIZ repository, and the two share the same regulatory knowledge graph architecture and five-site study area.

The central research question addressed here is: can a locally deployed, open-source VLM, operating with regulatory context injected through a Graph-RAG retrieval step, reason about wildfire compliance in aerial drone imagery well enough to produce citation-grounded assessments suitable for inspector triage, and to what extent does the regulatory context injection improve detection and compliance judgment relative to a plain-prompt baseline?


## Research Context

Autoregressive VLMs differ fundamentally from CLIP-based classifiers in the kind of output they produce. Where CLIP returns cosine similarity scores over predefined classes, a model such as Qwen2.5-VL-7B-Instruct returns natural language that can include spatial reasoning, regulatory citation, uncertainty acknowledgment, and bounding box coordinates. This affords a richer compliance record but introduces new failure modes: JSON hallucination, verbose non-compliance with the output format specification, and tendency to over-count when prompted without regulatory grounding.

This pipeline addresses those failure modes through three mechanisms:

1. **Geo-CoT prompting**: a structured chain-of-thought template that requires the model to reason sequentially through four stages (PERCEIVE, LOCATE, RETRIEVE, ASSESS) before producing a JSON verdict. This is analogous to the scratchpad technique used in mathematical reasoning but adapted to the spatial and regulatory domain.

2. **Graph-RAG context injection**: at inference time, the knowledge graph (shared with the CLIP pipeline) is queried by zone to retrieve the precise PRC 4291 sub-section and IBHS requirement text applicable to that spatial region. This text is injected into the system prompt, requiring the model to cite specific clauses rather than produce generic compliance language.

3. **Self-consistency voting**: for the ablation evaluation, the same tile is submitted to the model multiple times and responses are aggregated by majority vote on compliance_status and severity fields, with disagreement flagged for human review.


## System Architecture

### Pipeline Overview

The pipeline is organized as a numbered sequence of idempotent steps, coordinated by `run_full_pipeline.sh`. Each step writes its output before the next step reads it, so a crash at any point can be resumed without re-running earlier stages.

```
Step 0   setup_env.sh          Environment and dependency setup
Step 1   preprocess.py          Tile drone orthomosaics, write tile manifest
Step 2   download_public_imagery.py  Fetch NAIP 60cm public imagery (5 sites)
Step 3   preprocess_naip.py     Tile NAIP imagery
Step 3b  preprocess_naip_sr.py  Tile NAIP + super-resolution upscaled tiles
Step 4   run_qwen25vl.py        Qwen2.5-VL-7B-Instruct inference on drone tiles
Step 5   run_ablation.py        Ablation: same inference without Graph-RAG context
Step 6   run_qwen25vl_naip.py   Qwen2.5-VL on NAIP public imagery
Step 7a  run_geochat.py         GeoChat-7B inference (baseline comparison)
Step 7b  run_internvl.py        InternVL2-8B inference (strong baseline)
Step 7c  run_geochat.py ...     Ablation for GeoChat and InternVL2
Step 8   filter_tiles.py        Filter manifest to non-blank, information-rich tiles
Step 9   aggregate_parallel_results.py   Merge parallel inference shards
Step 10  evaluate.py            Generate results report and summary JSON
```

### Core Modules

**`preprocess.py`** loads 4-band parcel GeoTIFFs from the drone orthomosaic dataset, stretches each RGB band using the 2nd and 98th percentile, chips the image at 500 x 500 pixel tiles with 25% overlap, annotates each chip with a zone label derived from the CHM band, draws coloured zone-boundary circles on the tile for VLM spatial reference, and writes the tile PNG alongside a tile manifest CSV and a per-parcel metadata JSON.

**`prompts.py`** implements the Geo-CoT prompting framework. The system message establishes the model's role as a defensible-space compliance assessor with instructions to cite specific regulatory clauses. The few-shot block provides three demonstrations of the PERCEIVE / LOCATE / RETRIEVE / ASSESS reasoning trace: one VIOLATION case (propane tank in Zone 0), one COMPLIANT case (vehicle at 18 ft in Zone 1), and one UNCERTAIN case (partially occluded cylindrical object). The `build_prompt` function assembles the system message, regulatory context retrieved from the knowledge graph, the few-shot block, and the user instruction into a structured prompt dictionary. The `self_consistency_vote` function aggregates multiple model responses by majority vote on the compliance_status, severity, and confidence fields.

**`knowledge_graph/build_graph.py`** and **`knowledge_graph/graph_rag_lookup.py`** implement a focused, streamlined version of the regulatory knowledge graph covering five object classes (propane_tank, trash_can, vehicle, wood_pile, storage_shed) and their applicable PRC 4291 and IBHS regulations. This is a lighter complement to the comprehensive 33-class graph in the AI-for-HIZ repository; the two share the same schema but differ in scope to match the target detection classes of each inference pipeline. `get_all_contexts_for_prompt(zone)` returns a formatted regulatory context string for the given zone, ready for direct inclusion in the system prompt.

**`run_qwen25vl.py`** implements the primary inference loop for Qwen2.5-VL-7B-Instruct. The script attempts BitsAndBytes 4-bit quantized loading (~4.5 GB on MPS), falls back to 8-bit (~8 GB), and further falls back to float16 on CPU. Tiles are processed in order from the manifest, results are saved per-tile to the results/qwen25vl/ directory as JSON, and the script supports resumption by checking which tiles have already been processed. After each tile, MPS memory is flushed every five tiles to prevent fragmentation on Apple Silicon.

**`run_ablation.py`** runs the same inference loop but calls `build_prompt` with `use_graph_rag=False`, which strips the regulatory context and few-shot block and replaces them with a plain natural-language instruction. The comparison between the full (Graph-RAG + Geo-CoT) and ablation (plain-prompt) runs constitutes the primary ablation study.

**`run_geochat.py`** and **`run_internvl.py`** implement inference for GeoChat-7B (MBZUAI/geochat-7B) and InternVL2-8B (OpenGVLab/InternVL2-8B) respectively, using the same prompt structure and output schema. GeoChat is loaded from a local source clone of the MBZUAI GeoChat repository (required because it uses a non-standard chat template). InternVL2-8B uses the standard transformers loading path.

**`run_qwen25vl_naip.py`** and **`preprocess_naip.py`** extend the pipeline to NAIP 60 cm public aerial imagery as a scalability demonstration. NAIP imagery covers the same five study sites and is freely available through the Microsoft Planetary Computer STAC API. The NAIP tiles are processed identically to the drone tiles except that a synthetic CHM is not available, so all tiles are assigned to Zone_1 for compliance assessment.

**`filter_tiles.py`** reads the tile manifest and filters out blank tiles (more than 70% uniform pixels) and tiles dominated by vegetation canopy with no visible structure. The filtered manifest is used by the inference scripts to focus computation on information-rich tiles.

**`annotate.py`** overlays detected bounding boxes, class labels, compliance verdicts, and severity colours on the tile PNG for visual inspection.

**`evaluate.py`** reads all saved inference results, computes detection summaries by class and zone for each model, generates compliance risk scores per parcel (weighted sum of violation severity levels), identifies the top 10 highest-risk parcels, compares model agreement across Qwen2.5-VL and InternVL2, reports the ablation comparison, and writes the RESULTS_REPORT.md and results_summary.json.


## Key Results

Results from the completed pipeline runs are summarised below. Raw per-tile inference results are not included in this repository; only the aggregate summary is committed.

### Primary Model: Qwen2.5-VL-7B-Instruct

| Object Class | Zone 0 | Zone 1 | Zone 2 | Total |
|---|---|---|---|---|
| Propane tank | 283 | 39 | 172 | 494 |
| Trash can | 134 | 93 | 460 | 687 |
| Vehicle | 27 | 40 | 199 | 266 |

Regulatory clause citation rate in chain-of-thought traces: 91.6% (full pipeline, with Graph-RAG).

### Ablation Study: Graph-RAG vs Plain Prompt (Qwen2.5-VL)

| Object class | With Graph-RAG | Without Graph-RAG |
|---|---|---|
| Vehicle detections | 295 | 173 |
| Trash can detections | 720 | 113 |
| Propane tank detections | 516 | 101 |
| Clause citation rate | 91.6% | 0.0% |

The Graph-RAG condition substantially increases detection recall across all three classes, with the largest gain on trash cans (6.4x). The plain-prompt baseline produces no regulatory citations in any chain-of-thought trace, confirming that clause citation is driven entirely by the injected context rather than by model memorisation.

### NAIP Scalability Demonstration (Qwen2.5-VL, 60 cm GSD)

| Object class | Total detections |
|---|---|
| Propane tank | 9 |
| Vehicle | 4 |
| Trash can | 0 |

The lower detection rate on NAIP imagery relative to the drone dataset is expected given the 60 cm versus 5 cm GSD difference. Propane tanks, being large reflective cylinders, remain detectable at 60 cm; small objects such as trash cans do not.

### Aerial Limitation Analysis

The model explicitly reports 3,065 limitation mentions across the full inference run, breaking down as: general/other (2,495), shadow ambiguity (356), canopy occlusion (172), inability to measure proximity (31), and inability to assess vertical clearance (11). These categories map directly to the aerial-detectable versus ground-only partition established in the knowledge graph.


## Study Sites

The same five sites as the AI-for-HIZ CLIP pipeline:

| Code | Location | Notes |
|------|----------|-------|
| fel | Felton (Santa Cruz Mountains) | SCU Lightning Complex burn scar |
| par | Paradise (Butte County) | 2018 Camp Fire zone |
| red | Red Zone (Shasta-Trinity) | High FHSZ designation |
| sar | Santa Rosa (Coffey Park) | 2017 Tubbs Fire area |
| tah | Tahoe Donner (Nevada County) | Active defensible-space program |


## Repository Structure

```
.
├── preprocess.py                  # Tile drone orthomosaics, build manifest
├── preprocess_naip.py             # Tile NAIP public imagery
├── preprocess_naip_sr.py          # Tile NAIP super-resolution imagery
├── download_public_imagery.py     # Fetch NAIP tiles from Planetary Computer
├── filter_tiles.py                # Filter manifest to information-rich tiles
├── prompts.py                     # Geo-CoT prompt builder and self-consistency voter
├── run_qwen25vl.py                # Qwen2.5-VL-7B-Instruct inference (primary model)
├── run_qwen25vl_parallel.py       # Parallel shard runner for Qwen2.5-VL
├── run_qwen25vl_naip.py           # Qwen2.5-VL on NAIP imagery
├── run_qwen25vl_naip_sr.py        # Qwen2.5-VL on NAIP super-resolution imagery
├── run_geochat.py                 # GeoChat-7B inference
├── run_internvl.py                # InternVL2-8B inference
├── run_naip_ollama.py             # Ollama-based local LLM inference on NAIP
├── run_ablation.py                # Ablation: plain-prompt inference (no Graph-RAG)
├── run_ablation_ollama.py         # Ablation via Ollama runner
├── aggregate_parallel_results.py  # Merge parallel inference shards
├── aggregate_naip_results.py      # Merge NAIP inference results
├── annotate.py                    # Overlay bounding boxes on tile PNGs
├── evaluate.py                    # Generate RESULTS_REPORT.md and summary JSON
├── run_full_pipeline.sh           # Master sequential runner
├── setup_env.sh                   # One-time environment and dependency setup
│
├── knowledge_graph/
│   ├── build_graph.py             # Build focused 5-class regulatory graph
│   ├── graph_rag_lookup.py        # Graph-RAG query API (used by prompts.py)
│   └── hiz_graph.graphml          # Serialised graph (GraphML)
│
└── results/
    ├── RESULTS_REPORT.md          # Generated report (all models, all metrics)
    └── results_summary.json       # Aggregate statistics (safe to share)
```


## Excluded from This Repository

The following are excluded because they contain private parcel imagery or large per-tile inference records:

- Drone tile images (tiles/ directory, ~45 parcels, thousands of PNGs)
- NAIP tile images (tiles_naip/, tiles_naip_sr/)
- Per-tile inference result JSONs (results/qwen25vl/, results/geochat/, results/internvl/, results/qwen25vl_ablation/, results/naip_qwen25vl/, etc.)
- Annotated tile images with detection overlays (results/annotated/)
- Pipeline run logs (*.log files)


## Setup and Reproducibility

### Environment

Developed and validated on Apple Silicon (M4) macOS with 16 GB unified memory, Python 3.11, and a Miniconda installation. A Linux machine with CUDA is also supported; the inference scripts detect MPS, CUDA, and CPU automatically.

```bash
bash ~/hiz_pipeline/setup_env.sh
source ~/hiz_venv/bin/activate
```

### Model Download

The pipeline requires three model families. Download them separately before running inference:

```bash
# Qwen2.5-VL-7B-Instruct (primary model, ~15 GB)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir ~/hiz_data/models/qwen25vl-7b

# InternVL2-8B (comparison baseline, ~17 GB)
huggingface-cli download OpenGVLab/InternVL2-8B \
    --local-dir ~/hiz_data/models/internvl2-8b

# GeoChat-7B (remote sensing specialist, ~14 GB, requires source install)
cd ~/hiz_data
git clone https://github.com/mbzuai-oryx/GeoChat.git
cd GeoChat && pip install -e .
huggingface-cli download MBZUAI/geochat-7B \
    --local-dir ~/hiz_data/models/geochat-7b
```

### Running the Full Pipeline

```bash
bash ~/hiz_pipeline/run_full_pipeline.sh
```

To resume from a specific step after a crash, set the appropriate `SKIP_*` flags inside `run_full_pipeline.sh` or run individual scripts directly.

### Preprocessing Drone Orthomosaic Data

If the tile manifest does not exist, preprocess the drone orthomosaics first:

```bash
python3 ~/hiz_pipeline/preprocess.py
```

The script expects 4-band parcel GeoTIFFs (RGB + CHM) in `~/hiz_data/henri/`. This step produces tile PNGs in `~/hiz_pipeline/tiles/` and a manifest CSV.

### Running Inference Directly

```bash
# Qwen2.5-VL with Graph-RAG (full pipeline)
python3 ~/hiz_pipeline/run_qwen25vl.py

# Ablation (no regulatory context)
python3 ~/hiz_pipeline/run_ablation.py --model qwen25vl

# NAIP public imagery
python3 ~/hiz_pipeline/run_qwen25vl_naip.py
```

### Generating the Results Report

```bash
python3 ~/hiz_pipeline/evaluate.py
```

This reads all saved inference result JSONs from `results/` and writes `RESULTS_REPORT.md` and `results_summary.json`.


## Relationship to the AI-for-HIZ Repository

This repository and the AI-for-HIZ repository address the same research problem from two different modelling paradigms.

The AI-for-HIZ repository implements CLIP zero-shot classification: it processes the same drone tiles using cosine similarity between image chip embeddings and per-class text prompts, achieving rapid throughput (~0.5 s per chip on MPS) across a 33-class taxonomy. No natural language output is produced; the output is a structured compliance record derived entirely from the knowledge graph.

This repository implements autoregressive VLM inference: each tile is processed using a generative model (Qwen2.5-VL, InternVL2, or GeoChat) that produces natural language reasoning before emitting a structured JSON verdict. Throughput is lower (~15 s per tile on MPS at 4-bit quantization) but the output includes spatial references, uncertainty flags, and regulatory citations that are directly useful for inspector triage.

The two repositories share the same five study sites, the same zone taxonomy (Zone 0 / Zone 1 / Zone 2), the same regulatory knowledge graph schema, and the same ground-truth annotation framework (Label Studio, OWLv2 pre-annotations). The CLIP pipeline is designed for high-throughput screening across large parcel inventories; the VLM pipeline is designed for deep compliance reasoning on a priority subset.


## Open Questions and Next Steps

1. **Ground-truth evaluation**: neither pipeline has been quantitatively evaluated against human-verified ground truth. The Label Studio annotation workflow is in progress. Priority classes for ground truth are propane tanks, trash cans, and vehicles in Zone 0 and Zone 1, where false positives carry the highest cost in a compliance triage context.

2. **InternVL2 and GeoChat calibration**: both models currently produce zero detections in the results report. This is a prompt format issue, not a model capability issue. The InternVL2 chat template and the GeoChat system message format differ from Qwen2.5-VL and require separate calibration of the JSON output specification.

3. **Super-resolution NAIP integration**: `preprocess_naip_sr.py` and `run_qwen25vl_naip_sr.py` implement a super-resolution preprocessing step to upscale NAIP 60 cm imagery before inference. The evaluation of whether super-resolution meaningfully increases detection rates at this GSD is pending.

4. **Parallel inference scaling**: `run_qwen25vl_parallel.py` shards the tile manifest across multiple processes for faster throughput on machines with multiple GPU devices or for CPU parallelism. The aggregation step is handled by `aggregate_parallel_results.py`. This path has been tested with Ollama as the inference backend and is partially validated.

5. **Multi-model ensemble**: the self-consistency voting function in `prompts.py` is designed to aggregate responses from multiple model runs or multiple models. A two-model ensemble of Qwen2.5-VL and InternVL2 is the natural next step once InternVL2 calibration is complete.


## Citation

If you use this codebase in published work, please cite the associated manuscript (in preparation) and this repository.


## Contact

Wildfire Interdisciplinary Research Center (WIRC)
San Jose State University
