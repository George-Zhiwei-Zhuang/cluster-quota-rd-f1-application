#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/f1_replication_mpl}"
cd "$PROJECT_DIR"

if [[ "${REFRESH_RAW:-0}" == "1" ]]; then
  "$PYTHON_BIN" code/00_download/00_download_jolpica.py --force
else
  "$PYTHON_BIN" code/00_download/00_download_jolpica.py
fi
"$PYTHON_BIN" code/01_clean/01_clean_jolpica.py
"$PYTHON_BIN" code/02_construct/02_construct_analysis.py
"$PYTHON_BIN" code/03_analyze/03_analyze.py
"$PYTHON_BIN" code/04_tables/04_make_tables.py
"$PYTHON_BIN" code/05_figures/05_make_figures.py
"$PYTHON_BIN" code/06_diagnostics/06_validate_package.py
echo "Replication complete. See output/tables, output/figures, and output/logs."
