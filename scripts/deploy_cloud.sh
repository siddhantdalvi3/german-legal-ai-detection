#!/usr/bin/env bash
# ──────────────────────────────────────────────────
# deploy_cloud.sh — run on your LOCAL machine
# Packages the project and uploads to a cloud GPU
# ──────────────────────────────────────────────────
set -euo pipefail

HOST="${1:-}"
MODELS="${2:-deepseek qwen3 phi4 gemma4 llama3.1}"

if [ -z "$HOST" ]; then
    echo "Usage: bash scripts/deploy_cloud.sh <user@host> [models]"
    echo "  e.g. bash scripts/deploy_cloud.sh ubuntu@123.45.67.89 'deepseek qwen3 phi4 gemma4 llama3.1'"
    echo ""
    echo "Local bundle size: ~10 MB (topics + scripts only)"
    exit 1
fi

BUNDLE="/tmp/gen_bundle_$(date +%s)"
echo "=== Creating bundle at $BUNDLE ==="
mkdir -p "$BUNDLE/data/topics" "$BUNDLE/scripts" "$BUNDLE/utils"

# Copy only what generation needs (no raw data)
cp main.py config.py "$BUNDLE/"
cp scripts/generate_ai.py scripts/extract_topics.py "$BUNDLE/scripts/"
cp scripts/provision.sh "$BUNDLE/"
cp utils/mining.py "$BUNDLE/utils/"
cp data/topics/all_topics.jsonl "$BUNDLE/data/topics/"

# Verify topic count
TOPIC_COUNT=$(wc -l < "$BUNDLE/data/topics/all_topics.jsonl")
echo "Bundle contains $TOPIC_COUNT topics"

# Zip it
cd /tmp
ZIP_NAME="gen_bundle.zip"
zip -r "$ZIP_NAME" "$(basename "$BUNDLE")" > /dev/null
BUNDLE_ZIP="/tmp/$ZIP_NAME"
echo "Bundle created: $(du -h "$BUNDLE_ZIP" | cut -f1)"

echo ""
echo "=== Uploading to $HOST ==="
scp "$BUNDLE_ZIP" "$HOST:~/"

echo ""
echo "=== Connect and run ==="
echo "ssh $HOST"
echo "cd ~ && unzip gen_bundle.zip && cd gen_bundle_*"
echo "bash provision.sh $MODELS"
echo ""
echo "After it finishes (it will pause for the scp step):"
echo "  scp $HOST:~/gen_bundle_*/ai_generated.zip ./"
echo "  unzip ai_generated.zip -d data/"
echo ""

# Cleanup
rm -rf "$BUNDLE" "$BUNDLE_ZIP"
echo "Local bundle cleaned up."
