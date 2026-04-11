#!/bin/bash
# ─────────────────────────────────────────────
#  setup_ollama.sh
#  One-click setup: Ollama + Phi-3-mini
#  Run: bash setup_ollama.sh
# ─────────────────────────────────────────────

echo "═══════════════════════════════════════════"
echo "  Smart Security — Ollama + Phi-3 Setup"
echo "═══════════════════════════════════════════"

# 1. Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "[1/3] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[1/3] Ollama already installed ✅"
fi

# 2. Start Ollama server (background)
echo "[2/3] Starting Ollama server..."
ollama serve &
sleep 3

# 3. Pull Phi-3-mini
echo "[3/3] Downloading Phi-3-mini (2.3GB, one-time)..."
echo "      This is the lightest capable model for CPU inference."
ollama pull phi3:mini

echo ""
echo "✅ Done! Phi-3-mini ready for use."
echo ""
echo "Test it: ollama run phi3:mini 'Hello!'"
echo "Then:    python main.py --placeholder"
