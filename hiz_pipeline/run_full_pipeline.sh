#!/bin/bash
# run_full_pipeline.sh
# HIZ-VLM Pipeline — Master runner for Qwen2.5-VL + NAIP extension
#
# Runs all steps sequentially. Each step is idempotent (safe to re-run).
# Edit SKIP_* flags to start from a specific step after a crash.
#
# Usage:
#   bash ~/hiz_pipeline/run_full_pipeline.sh
#   bash ~/hiz_pipeline/run_full_pipeline.sh 2>&1 | tee ~/hiz_pipeline/run.log

set -e
PY=/opt/miniconda3/bin/python3
PIPE=~/hiz_pipeline

echo "========================================================"
echo "  HIZ-VLM FULL PIPELINE"
echo "  Model: Qwen2.5-VL-7B-Instruct (primary open-source)"
echo "  Date:  $(date)"
echo "========================================================"

# ── 0. Install/verify packages ────────────────────────────────────────────────
echo ""
echo "[0] Verifying required packages..."
$PY -c "import transformers, torch, PIL, tqdm, psutil, accelerate, bitsandbytes" \
    && echo "  Core packages OK" \
    || { echo "  Installing missing packages..."; \
         /opt/miniconda3/bin/pip install -q transformers accelerate bitsandbytes \
             huggingface_hub torch torchvision pillow tqdm psutil; }

# Install pystac_client + planetary_computer for NAIP download
$PY -c "import pystac_client" 2>/dev/null \
    || /opt/miniconda3/bin/pip install -q pystac-client planetary-computer requests

# Install rasterio for GeoTIFF reading
$PY -c "import rasterio" 2>/dev/null \
    || /opt/miniconda3/bin/pip install -q rasterio

echo "  All packages ready."

# ── 1. Preprocess drone orthomosaic data (skip if manifest exists) ─────────────
echo ""
echo "[1] Checking tile manifest..."
if [ -f "$PIPE/tiles/tile_manifest.csv" ]; then
    N=$(tail -n +2 "$PIPE/tiles/tile_manifest.csv" | wc -l | tr -d ' ')
    echo "  Manifest exists: $N tiles. Skipping preprocess."
else
    echo "  Running preprocess.py..."
    $PY "$PIPE/preprocess.py"
fi

# ── 2. Download NAIP public imagery ───────────────────────────────────────────
echo ""
echo "[2] Downloading NAIP public imagery (5 sites)..."
$PY "$PIPE/download_public_imagery.py" 2>&1 | grep -v "^$" || true

# ── 3. Tile NAIP imagery ──────────────────────────────────────────────────────
echo ""
echo "[3] Tiling NAIP imagery..."
if [ -f "$PIPE/tiles_naip/tile_manifest_naip.csv" ]; then
    N=$(tail -n +2 "$PIPE/tiles_naip/tile_manifest_naip.csv" | wc -l | tr -d ' ')
    echo "  NAIP manifest exists: $N tiles. Skipping preprocess_naip."
else
    $PY "$PIPE/preprocess_naip.py"
fi

# ── 4. Qwen2.5-VL inference on drone tiles ────────────────────────────────────
echo ""
echo "[4] Running Qwen2.5-VL inference on drone tiles (3628 tiles)..."
echo "  NOTE: This will take ~3-8 hours depending on quantization mode."
echo "  Progress is saved per-tile — safe to interrupt and resume."
$PY "$PIPE/run_qwen25vl.py"

# ── 5. Ablation study (no Graph-RAG) ─────────────────────────────────────────
echo ""
echo "[5] Running ablation study (Qwen2.5-VL without Graph-RAG)..."
$PY "$PIPE/run_ablation.py" --model qwen25vl

# ── 6. Qwen2.5-VL on NAIP public imagery ─────────────────────────────────────
echo ""
echo "[6] Running Qwen2.5-VL inference on NAIP public imagery..."
$PY "$PIPE/run_qwen25vl_naip.py"

# ── 7. (Optional) InternVL2-8B inference ─────────────────────────────────────
echo ""
echo "[7] InternVL2-8B inference (optional — downloads ~17 GB)..."
if [ -d ~/hiz_data/models/internvl2-8b ] && \
   [ "$(ls ~/hiz_data/models/internvl2-8b/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    $PY "$PIPE/run_internvl.py"
    $PY "$PIPE/run_ablation.py" --model internvl
else
    echo "  InternVL2-8B not downloaded. Skipping (run separately if needed)."
    echo "  To download: huggingface-cli download OpenGVLab/InternVL2-8B \\"
    echo "               --local-dir ~/hiz_data/models/internvl2-8b"
fi

# ── 8. Evaluation report ──────────────────────────────────────────────────────
echo ""
echo "[8] Generating evaluation report..."
$PY "$PIPE/evaluate.py"

echo ""
echo "========================================================"
echo "  PIPELINE COMPLETE"
echo "  Report: $PIPE/results/RESULTS_REPORT.md"
echo "  JSON:   $PIPE/results/results_summary.json"
echo "========================================================"
