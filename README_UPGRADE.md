# LayerWise-QC Upgrade Patch

This patch upgrades the app from a vague demo dashboard to a guided research prototype.

## Files to copy

Copy these files into your repository:

```text
app/streamlit_app.py
src/am_defect_detection/physics_features.py
src/am_defect_detection/sensor_features.py
src/am_defect_detection/fusion_analysis.py
src/am_defect_detection/explainers.py
src/am_defect_detection/feedforward_control.py
src/am_defect_detection/data_manifest.py
```

## Commands

From your repository root:

```bash
mkdir -p src/am_defect_detection app

cp path/to/patch/app/streamlit_app.py app/streamlit_app.py
cp path/to/patch/src/am_defect_detection/*.py src/am_defect_detection/

python -m compileall app src/am_defect_detection
streamlit run app/streamlit_app.py
```

Then commit and push:

```bash
git add app/streamlit_app.py src/am_defect_detection/
git commit -m "Upgrade dashboard with guided physics-informed sensor fusion workflow"
git push
```

## What the new app adds

- Overview tab
- Process input explanations
- Live decision with uncertainty
- Sensor descriptor table
- Physics-informed features
- Sensor fusion and ablation logic
- Feed-forward advisory control
- Data / manifest readiness checks
- Validation roadmap
