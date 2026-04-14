#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo "  Aria Personal Agent v7"
echo "  HuggingFace + llama-cpp-python backend"
echo ""

if ! command -v python3 &>/dev/null; then
    echo "  [ERROR] Python 3 not found. Install from https://python.org"
    read -p "Press Enter to exit..."
    exit 1
fi

if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo "  [Setup] llama-cpp-python not installed — running setup..."
    python3 setup_v7.py
fi

pip3 install -r requirements.txt --quiet 2>/dev/null

echo "  [Launch] Dashboard → http://localhost:8000"
echo "  [Launch] HF Models → http://localhost:8000/hf"
echo ""

# Open browser after 3 seconds
(sleep 3 && open http://localhost:8000) &

python3 main.py "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "  [ERROR] Aria exited. Try: python3 setup_v7.py"
    read -p "Press Enter to exit..."
fi
