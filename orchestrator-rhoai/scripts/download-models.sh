#!/usr/bin/env bash
set -euo pipefail

#
# Download model weights from HuggingFace.
# Requires: huggingface-cli (pip install huggingface_hub)
#

MODELS_DIR="${MODELS_DIR:-./models}"
HF_TOKEN="${HF_TOKEN:-}"

echo "============================================="
echo " orchestrator-rhoai — Model Downloader"
echo " Target directory: $MODELS_DIR"
echo "============================================="
echo ""

if ! command -v huggingface-cli &>/dev/null; then
    echo "Error: huggingface-cli not found."
    echo "Install with: pip install huggingface_hub"
    exit 1
fi

mkdir -p "$MODELS_DIR"

download_model() {
    local repo="$1"
    local dest="$MODELS_DIR/$(echo "$repo" | tr '/' '_')"

    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  [SKIP] $repo (already downloaded)"
        return
    fi

    echo "  [DOWNLOADING] $repo → $dest"
    if [ -n "$HF_TOKEN" ]; then
        huggingface-cli download "$repo" --local-dir "$dest" --token "$HF_TOKEN"
    else
        huggingface-cli download "$repo" --local-dir "$dest"
    fi
    echo "  [DONE] $repo"
}

echo "Tier 1 Models (required for minimal deployment):"
download_model "nvidia/Nemotron-Orchestrator-8B"
download_model "Qwen/Qwen3-32B"
echo ""

echo "Tier 2 Models (standard deployment):"
download_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
download_model "Qwen/Qwen2.5-Coder-32B-Instruct"
echo ""

echo "Tier 3 Models (full deployment):"
download_model "meta-llama/Llama-3.3-70B-Instruct"
download_model "Qwen/Qwen2.5-Math-72B-Instruct"
download_model "Qwen/Qwen2.5-Math-7B-Instruct"
echo ""

echo "Training Base Model:"
download_model "Qwen/Qwen3-8B"
echo ""

echo "Retrieval Index:"
download_model "multi-train/index"
echo ""

echo "============================================="
echo " Download complete."
echo " Models are in: $MODELS_DIR"
echo "============================================="
