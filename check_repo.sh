#!/usr/bin/env bash
set -u

LOG="outputs/repo_health_check.log"
mkdir -p outputs

echo "LayerWise-QC repo health check" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Repo: $(pwd)" | tee -a "$LOG"
echo "Git branch: $(git branch --show-current 2>/dev/null || echo 'no git')" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

run_check () {
  local name="$1"
  shift
  echo "" | tee -a "$LOG"
  echo "===== $name =====" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "✅ PASS: $name" | tee -a "$LOG"
  else
    echo "❌ FAIL: $name | exit code $status" | tee -a "$LOG"
  fi
  return 0
}

run_check "Python version" python --version
run_check "Pip version" python -m pip --version
run_check "Install package in editable mode" python -m pip install -e .
run_check "Install required report dependency tabulate" python -m pip install tabulate
run_check "Pytest" python -m pytest -q

run_check "Import core package" python - <<'PY'
import am_defect_detection
print("Imported:", am_defect_detection.__file__)
PY

run_check "Check important files exist" python - <<'PY'
from pathlib import Path

required = [
    "app/streamlit_app.py",
    "src/am_defect_detection/physics_features.py",
    "src/am_defect_detection/data_manifest.py",
    "src/am_defect_detection/data.py",
    "src/am_defect_detection/feature_table.py",
    "src/am_defect_detection/literature_data.py",
    "src/am_defect_detection/inference.py",
    "src/am_defect_detection/reporting.py",
    "scripts/validate_manifest.py",
    "scripts/build_feature_table.py",
    "scripts/build_literature_dataset.py",
    "scripts/train_tabular_baselines.py",
    "scripts/generate_validation_report.py",
    "data/demo_samples/manifest.csv",
    "data/literature/raw_literature_records_template.csv",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    print("Missing files:")
    for p in missing:
        print(" -", p)
    raise SystemExit(1)
print("All important files exist.")
PY

run_check "Check spot-size symbols exist" bash -lc "grep -R \"spot_size_um\\|power_density_w_mm2\\|beam_area_mm2\" -n src app scripts tests | head -80"

run_check "Validate demo manifest" python scripts/validate_manifest.py \
  --manifest data/demo_samples/manifest.csv \
  --root data/demo_samples

run_check "Build feature table" python scripts/build_feature_table.py \
  --manifest data/demo_samples/manifest.csv \
  --root data/demo_samples \
  --out outputs/features.csv

run_check "Inspect feature table columns" python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("outputs/features.csv")
if not p.exists():
    raise SystemExit("outputs/features.csv not found")

df = pd.read_csv(p)
print("features shape:", df.shape)
needed = [
    "spot_size_um",
    "spot_size_mm",
    "beam_radius_mm",
    "beam_area_mm2",
    "power_density_w_mm2",
]
missing = [c for c in needed if c not in df.columns]
print("spot/beam columns:", [c for c in df.columns if "spot" in c or "beam" in c or "power_density" in c])
if missing:
    raise SystemExit(f"Missing required feature columns: {missing}")
print(df[needed].head())
PY

run_check "Build literature dataset" python scripts/build_literature_dataset.py \
  --input data/literature/raw_literature_records_template.csv \
  --out outputs/literature_features.csv \
  --manifest-out outputs/literature_manifest.csv

run_check "Train tabular baselines" python scripts/train_tabular_baselines.py \
  --features outputs/features.csv \
  --out outputs/tabular \
  --target class_name \
  --task classification

run_check "Check tabular outputs" python - <<'PY'
from pathlib import Path
for p in [
    "outputs/tabular/metrics.json",
    "outputs/tabular/predictions.csv",
    "outputs/tabular/feature_importance.csv",
    "outputs/tabular/leaderboard.csv",
]:
    print(p, "exists:", Path(p).exists())
    if not Path(p).exists():
        raise SystemExit(f"Missing {p}")
PY

run_check "Generate validation report" python scripts/generate_validation_report.py \
  --metrics outputs/tabular/metrics.json \
  --predictions outputs/tabular/predictions.csv \
  --manifest data/demo_samples/manifest.csv \
  --data-kind synthetic \
  --out outputs/validation_report.md

run_check "Check validation report" bash -lc "test -s outputs/validation_report.md && head -80 outputs/validation_report.md"

run_check "Streamlit import smoke test" python - <<'PY'
import streamlit
print("Streamlit:", streamlit.__version__)
PY

run_check "Compile all Python files" python -m compileall app src scripts tests -q

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Saved log to: $LOG" | tee -a "$LOG"

echo ""
echo "Now inspect failures with:"
echo "grep -n \"❌ FAIL\" $LOG"
