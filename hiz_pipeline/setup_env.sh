#!/bin/bash
# setup_env.sh
# HIZ-VLM Pipeline — Step 1: Environment Setup
# Run once to create the venv and install all dependencies.
#
# Usage:
#   bash ~/hiz_pipeline/setup_env.sh

set -e
echo "========================================================"
echo "  HIZ-VLM Pipeline — Environment Setup"
echo "  Apple M4 | 16 GB RAM | macOS"
echo "========================================================"

# ── Create venv ───────────────────────────────────────────────────────────────
if [ ! -d "$HOME/hiz_venv" ]; then
    echo "Creating Python 3.11 virtual environment at ~/hiz_venv..."
    python3 -m venv ~/hiz_venv
else
    echo "~/hiz_venv already exists — skipping creation."
fi

source ~/hiz_venv/bin/activate
echo "Activated: $(which python3)"

# ── Core packages ──────────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

echo "Installing PyTorch (MPS backend)..."
pip install torch torchvision torchaudio --quiet

echo "Installing MLX..."
pip install mlx mlx-lm --quiet

echo "Installing HuggingFace stack..."
pip install transformers accelerate huggingface_hub bitsandbytes --quiet

echo "Installing geospatial + image packages..."
pip install rasterio shapely numpy pillow opencv-python --quiet

echo "Installing utilities..."
pip install networkx tqdm rich psutil --quiet

echo "Installing model-specific packages..."
pip install sentencepiece protobuf einops timm --quiet

echo "Installing torchvision transforms..."
pip install torchvision --quiet

# ── GeoChat from source ────────────────────────────────────────────────────────
mkdir -p ~/hiz_data
if [ ! -d "$HOME/hiz_data/GeoChat" ]; then
    echo ""
    echo "Cloning GeoChat repository..."
    cd ~/hiz_data
    git clone https://github.com/mbzuai-oryx/GeoChat.git
    cd GeoChat
    pip install -e . --quiet
    echo "GeoChat installed from source."
else
    echo "GeoChat repo already exists at ~/hiz_data/GeoChat."
fi

# ── Working directories ────────────────────────────────────────────────────────
mkdir -p ~/hiz_pipeline/{tiles,results/{geochat,internvl,geochat_ablation,internvl_ablation,annotated},annotations,knowledge_graph}
mkdir -p ~/hiz_data/{henri,models/{geochat-7b,internvl2-8b}}

echo ""
echo "========================================================"
echo "  Setup complete."
echo "  Next steps:"
echo "  1. Download models (Step 2):"
echo "     huggingface-cli download MBZUAI/geochat-7B \\"
echo "       --local-dir ~/hiz_data/models/geochat-7b \\"
echo "       --local-dir-use-symlinks False"
echo ""
echo "     huggingface-cli download OpenGVLab/InternVL2-8B \\"
echo "       --local-dir ~/hiz_data/models/internvl2-8b \\"
echo "       --local-dir-use-symlinks False"
echo ""
echo "  2. Activate the venv before running any pipeline script:"
echo "     source ~/hiz_venv/bin/activate"
echo "========================================================"
