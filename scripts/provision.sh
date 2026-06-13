#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────
# provision.sh — run ON the cloud GPU instance
# Usage: bash provision.sh <model1> <model2> ...
#   e.g. bash provision.sh deepseek qwen3 phi4 gemma4 llama3.1
# ──────────────────────────────────────────────────

REPO_DIR="$HOME/german-legal-ai-detection"
MODELS=("$@")

echo "=== Step 1: Install system deps ==="
sudo apt-get update -qq
sudo apt-get install -y -qq curl python3 python3-pip

echo "=== Step 2: Install Ollama ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== Step 3: Create project dir ==="
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

echo "=== Step 4: Pull Ollama models in parallel ==="
# Kill any running ollama server first
ollama serve &>/dev/null &
sleep 2
for model in "${MODELS[@]}"; do
    ollama_name=""
    case "$model" in
        qwen2.5)   ollama_name="qwen2.5:7b" ;;
        qwen3)     ollama_name="qwen3:14b" ;;
        gemma3)    ollama_name="gemma3:12b" ;;
        gemma4)    ollama_name="gemma4:12b" ;;
        mistral)   ollama_name="mistral" ;;
        llama3.1)  ollama_name="llama3.1:8b" ;;
        deepseek)  ollama_name="deepseek-r1:7b" ;;
        phi4)      ollama_name="phi4:14b" ;;
        *)
            echo "Unknown model: $model"
            exit 1
            ;;
    esac
    echo "Pulling $ollama_name..."
    ollama pull "$ollama_name" &
done
wait
echo "All models pulled."

echo "=== Step 5: Wait for project files to be uploaded ==="
echo "Upload the bundle now from your local machine:"
echo "  scp bundle.zip ubuntu@<HOST>:$REPO_DIR/"
echo ""
echo "Then press Enter to continue..."
read -r

echo "=== Step 6: Extract bundle ==="
unzip -o bundle.zip -d "$REPO_DIR"
rm bundle.zip

echo "=== Step 7: Install Python deps ==="
pip install requests tqdm -q

echo "=== Step 8: Run generation ==="
# Generate for all models at temp 0.3 and 0.7
for temp in 0.3 0.7; do
    python3 main.py --generate --models "${MODELS[@]}" --temps "$temp"
done

echo "=== Step 9: Package results ==="
cd "$REPO_DIR"
zip -r ai_generated.zip data/ai_generated/
echo "Done! Download results:"
echo "  scp ubuntu@<HOST>:$REPO_DIR/ai_generated.zip ./"
