# Literature-derived benchmark data

This workflow lets LayerWise-QC test its data/feature/report pipeline using values manually extracted from open-access papers, supplementary datasets, or user-provided files.

It is **not** a substitute for aligned in-situ sensor images and ground-truth validation. Models trained on this data should be described only as **literature-derived baselines**.

Rules:

- Do not bypass paywalls.
- Do not scrape copyrighted full papers.
- Prefer open-access papers, supplementary data, and manually entered values.
- Every row must keep `source_id` and `citation`.
- Mark weak labels using `label_source`.
- Missing spot size / beam diameter must remain unknown unless explicitly imputed.

Template:

```bash
python scripts/build_literature_dataset.py \
  --input data/literature/raw_literature_records_template.csv \
  --out outputs/literature_features.csv \
  --manifest-out outputs/literature_manifest.csv
```
