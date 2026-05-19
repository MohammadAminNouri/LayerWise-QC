# Data folder

`demo_samples/` contains a very small set of simulated OT, MPM and PBI patches. They are included only so the Streamlit app opens with visible examples.

For real work, create a new folder such as:

```text
data/real_build_01/
  manifest.csv
  ot/
  mpm/
  pbi/
```

The manifest should contain paths to the image patches. Relative paths are resolved from the folder where the manifest is stored.

Minimal manifest for one modality:

```text
sample_id,class_idx,class_name,ot_path
```

Recommended manifest for multimodal work:

```text
sample_id,layer,specimen_id,class_idx,class_name,ot_path,mpm_path,pbi_path,laser_power_w,scan_speed_mm_s,hatch_distance_mm,layer_thickness_mm,ved_j_mm3
```
