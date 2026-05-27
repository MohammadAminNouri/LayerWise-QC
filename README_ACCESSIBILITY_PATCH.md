# LayerWise-QC Accessibility / Explanation Patch

This patch replaces the Streamlit app with a guided version.

## Main improvement

Every major part of the app now explains:

- what it is,
- what input it uses,
- what output it gives,
- how to read charts,
- how to read tables,
- what is good/bad,
- what is only demo/proxy,
- what is needed for real validation.

## Files included

```text
app/streamlit_app.py
src/am_defect_detection/physics_features.py
src/am_defect_detection/sensor_features.py
src/am_defect_detection/fusion_analysis.py
src/am_defect_detection/explainers.py
src/am_defect_detection/feedforward_control.py
src/am_defect_detection/data_manifest.py
```

## Copy into repo

From your repository root:

```bash
cp app/streamlit_app.py app/streamlit_app.py
cp src/am_defect_detection/*.py src/am_defect_detection/
```

If using this ZIP from outside the repo, copy the files preserving the same folders.

## Test

```bash
python -m compileall app src/am_defect_detection
streamlit run app/streamlit_app.py
```

## Commit

```bash
git add app/streamlit_app.py src/am_defect_detection/
git commit -m "Improve dashboard accessibility and explanations"
git push
```
