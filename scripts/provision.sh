#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────
# provision.sh — run ON a RunPod cloud GPU instance
# Usage (single GPU): bash provision.sh qwen3 gemma3 gemma4 mistral phi4 steuerllm
# Usage (dual GPU):  bash provision.sh deepseek qwen3 gemma3 gemma4 mistral phi4 steuerllm
#
# Models with deepseek → GPU0 gets deepseek (10K), GPU1 gets rest (30K)
# Without deepseek    → single sequential run (30K each)
# ──────────────────────────────────────────────────

REPO_DIR="$HOME/german-legal-ai-detection"
MODELS=("$@")

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: bash provision.sh <model1> [model2 ...]"
    echo "  Single-GPU:  bash provision.sh qwen3 gemma3 gemma4 mistral phi4 steuerllm"
    echo "  Dual-GPU:    bash provision.sh deepseek qwen3 gemma3 gemma4 mistral phi4 steuerllm"
    exit 1
fi

echo "=== Step 1: Install system deps ==="
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git git-lfs wget unzip curl

echo "=== Step 2: Install Ollama (if missing) ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama server if not running
ollama serve &>/tmp/ollama.log &
sleep 2

echo "=== Step 3: Setup Python venv ==="
cd "$REPO_DIR"
python3 -m venv venv
source venv/bin/activate
pip install requests tqdm -q

echo "=== Step 4: Pull models ==="
HAS_DEEPSEEK=false
declare -a PULL_LIST
for model_key in "${MODELS[@]}"; do
    case "$model_key" in
        deepseek)  PULL_LIST+=("deepseek-r1:70b"); HAS_DEEPSEEK=true ;;
        qwen3)     PULL_LIST+=("qwen3:30b") ;;
        gemma3)    PULL_LIST+=("gemma3:27b") ;;
        gemma4)    PULL_LIST+=("gemma4:12b") ;;
        mistral)   PULL_LIST+=("mistral-small:24b") ;;
        phi4)      PULL_LIST+=("phi4:14b") ;;
        steuerllm) echo "Pulling steuerllm:28b from HuggingFace..." ; PULL_LIST+=("steuerllm:28b") ;;
        *)         echo "Unknown model: $model_key" ;;
    esac
done

echo "Pulling ${#PULL_LIST[@]} models..."
for model in "${PULL_LIST[@]}"; do
    echo "  Pulling $model ..."
    ollama pull "$model" &
    sleep 3
done
wait
echo "All models pulled."

echo "=== Step 5: Run generation ==="
# Increase Ollama parallel processing for throughput
export OLLAMA_NUM_PARALLEL=3

if [ "$HAS_DEEPSEEK" = true ] && [ ${#MODELS[@]} -gt 1 ]; then
    # Dual GPU: deepseek on GPU0, rest on GPU1
    GPU0_MODELS=()
    GPU1_MODELS=()
    for model_key in "${MODELS[@]}"; do
        if [ "$model_key" = "deepseek" ]; then
            GPU0_MODELS+=("$model_key")
        else
            GPU1_MODELS+=("$model_key")
        fi
    done

    echo "Starting deepseek on GPU0 (10K x 2 temps)..."
    CUDA_VISIBLE_DEVICES=0 python3 main.py --generate --models "${GPU0_MODELS[@]}" --count 10000 --temps 0.3 0.7 &
    PID0=$!

    echo "Starting others on GPU1 (30K x 2 temps)..."
    CUDA_VISIBLE_DEVICES=1 python3 main.py --generate --models "${GPU1_MODELS[@]}" --count 30000 --temps 0.3 0.7 &
    PID1=$!

    wait $PID0 $PID1
else
    # Single GPU: run all sequentially at 30K each
    echo "Running all models sequentially (30K x 2 temps each)..."
    python3 main.py --generate --models "${MODELS[@]}" --count 30000 --temps 0.3 0.7
fi

echo "=== Step 6: Package results ==="
cd "$REPO_DIR"
zip -r ai_generated.zip data/ai_generated/
echo "Done! Download results:"
echo "  scp -P <port> root@<pod-ip>:$REPO_DIR/ai_generated.zip ./"
