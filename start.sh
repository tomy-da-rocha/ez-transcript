#!/usr/bin/env bash
set -e

echo "============================================"
echo "  EZ Transcript — Transcription locale IA"
echo "============================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ] 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERREUR] Python 3.10+ est requis mais non trouvé."
    echo "Installez Python depuis https://www.python.org/downloads/"
    exit 1
fi

echo "[INFO] Python trouvé : $($PYTHON --version)"

# Check FFmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "[AVERTISSEMENT] FFmpeg non trouvé. Installez-le :"
    echo "  Ubuntu/Debian : sudo apt install ffmpeg"
    echo "  macOS : brew install ffmpeg"
    echo ""
fi

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "[INFO] Création de l'environnement virtuel..."
    $PYTHON -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "[INFO] Vérification des dépendances..."
pip install -r requirements.txt --quiet 2>/dev/null || pip install -r requirements.txt

# Detect GPU and install CUDA libs if needed
echo ""
echo "[INFO] Détection du matériel..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r name vram; do
        echo "  GPU: $name (${vram} Mo VRAM)"
    done
    # Install CUDA runtime libraries if not already working
    if python -c "import ctypes; ctypes.CDLL('libcublas.so.12')" &>/dev/null; then
        echo "  CUDA: opérationnel"
    else
        echo "[INFO] Installation des bibliothèques CUDA..."
        if pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 --quiet 2>/dev/null; then
            echo "[INFO] CUDA installé avec succès."
        else
            echo "[AVERTISSEMENT] Impossible d'installer CUDA — le mode CPU sera utilisé."
        fi
    fi
else
    echo "  Pas de GPU NVIDIA détecté — mode CPU"
fi

PORT=${PORT:-7860}

echo ""
echo "[INFO] Démarrage du serveur sur http://localhost:$PORT"
echo "[INFO] Appuyez sur Ctrl+C pour arrêter."
echo ""

# Open browser (best effort)
(sleep 2 && {
    if command -v xdg-open &>/dev/null; then
        xdg-open "http://localhost:$PORT" 2>/dev/null
    elif command -v open &>/dev/null; then
        open "http://localhost:$PORT" 2>/dev/null
    fi
}) &

# Run server
python -m uvicorn backend.main:app --host 0.0.0.0 --port "$PORT" --no-access-log
