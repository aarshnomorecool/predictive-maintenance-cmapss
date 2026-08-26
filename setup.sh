#!/usr/bin/env bash
# One-shot environment setup for macOS and Linux.
#
#   bash setup.sh
#
# Creates a virtual environment, installs dependencies, and verifies the data
# foundation. Safe to re-run: an existing .venv is reused, not rebuilt.

set -euo pipefail
cd "$(dirname "$0")"

echo
echo "=== Predictive Maintenance : setup ==="
echo

# --- Python check ------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "Python not found on PATH. Install Python 3.10 or newer first." >&2
    exit 1
fi
echo "Python $($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])') detected"

# --- Virtual environment -----------------------------------------------------
if [ -d ".venv" ]; then
    echo "Reusing existing .venv"
else
    echo "Creating .venv ..."
    "$PY" -m venv .venv
fi

VENV_PY=".venv/bin/python"

# --- Dependencies ------------------------------------------------------------
echo "Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip --quiet

echo "Installing requirements (this takes a few minutes, TensorFlow is large) ..."
"$VENV_PY" -m pip install -r requirements.txt --quiet
echo "Dependencies installed"

# --- Verify ------------------------------------------------------------------
echo
echo "Verifying the data foundation ..."
"$VENV_PY" verify_foundation.py

cat <<'DONE'

=== Setup complete ===

Activate the environment:
    source .venv/bin/activate

Then train the models:
    python -m src.train_classifier

Then launch the dashboard:
    streamlit run dashboard/app.py

DONE
