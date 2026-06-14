#!/usr/bin/env bash
# ──────────────────────────────────────────────────
# deploy_cloud.sh — prepare and upload project to RunPod
#
# Option A: git clone (recommended for RunPod)
#   1. Push your branch to GitHub
#   2. SSH into pod and run:
#      git clone https://github.com/YOUR_USER/german-legal-ai-detection.git
#      cd german-legal-ai-detection
#      bash scripts/provision.sh <models...>
#
# Option B: scp bundle (for direct TCP access)
#   bash scripts/deploy_cloud.sh root@<ip> -p <port> [models]
#
# Examples:
#   bash scripts/deploy_cloud.sh "root@64.247.201.34 -p 13746" deepseek
# ──────────────────────────────────────────────────
set -euo pipefail

HOST="${1:-}"
MODELS="${2:-qwen3 gemma3 gemma4 mistral deepseek phi4 steuerllm}"

if [ -z "$HOST" ]; then
    echo "Usage: bash scripts/deploy_cloud.sh <user@host> [models]"
    echo ""
    echo "For RunPod git clone (recommended):"
    echo "  git clone https://github.com/YOUR_USER/german-legal-ai-detection.git"
    echo "  cd german-legal-ai-detection"
    echo "  bash scripts/provision.sh $MODELS"
    echo ""
    echo "For direct scp (need direct TCP port):"
    echo "  bash scripts/deploy_cloud.sh 'root@64.247.201.34 -p 13746' 'deepseek qwen3 gemma3 gemma4 mistral phi4 steuerllm'"
    exit 1
fi

BUNDLE="/tmp/gen_bundle_$(date +%s)"
echo "=== Creating bundle at $BUNDLE ==="
mkdir -p "$BUNDLE/data/topics" "$BUNDLE/scripts" "$BUNDLE/utils"

cp main.py config.py "$BUNDLE/"
cp scripts/generate_ai.py scripts/extract_topics.py scripts/provision.sh "$BUNDLE/scripts/"
cp utils/mining.py "$BUNDLE/utils/"
cp data/topics/all_topics.jsonl "$BUNDLE/data/topics/"

cd /tmp
ZIP_NAME="gen_bundle.zip"
zip -r "$ZIP_NAME" "$(basename "$BUNDLE")" > /dev/null
BUNDLE_ZIP="/tmp/$ZIP_NAME"
echo "Bundle created: $(du -h "$BUNDLE_ZIP" | cut -f1)"

echo "=== Uploading to $HOST ==="
scp "$BUNDLE_ZIP" "$HOST:~/"

echo "=== Connect and run ==="
echo "ssh $HOST"
echo "cd ~ && unzip gen_bundle.zip && cd gen_bundle_*"
echo "bash scripts/provision.sh $MODELS"

rm -rf "$BUNDLE" "$BUNDLE_ZIP"
echo "Local bundle cleaned up."
