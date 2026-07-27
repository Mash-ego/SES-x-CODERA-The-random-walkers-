#!/usr/bin/env bash
# Sets up a self-contained virtual environment for this project.
# Run once: bash setup.sh
set -e

cd "$(dirname "$0")"

echo "Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required libraries..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "To run the analysis:"
echo "  source .venv/bin/activate"
echo "  python multi_predictor_analysis.py"
