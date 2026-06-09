import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Sample & Model Requirements",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Sample & Model Requirements")
st.caption(
    "This page lists the minimum LPBF sample data needed for training and explains "
    "which inputs are required for each model type."
)

st.info(
    """
    **Minimum dataset for first model:** 50–80 printed samples  
    **Good dataset:** 100–150 printed samples  
    **Strong research dataset:** 200–300 printed samples  

    Each row in the dataset should represent **one printed sample**.
    """
)

st.header("1. Information needed for each printed sample")

sample_data = pd.DataFrame(
    [
        ["sample_id", "Unique sample name, for example S001", "Required"],
        ["build_id", "Build/job identifier, for example Build_01", "Required"],
        ["machine_id", "LPBF machine name or ID", "Required"],
        ["material", "Material name, for example 316L, Ti64, AlSi10Mg", "Required"],
        ["powder_batch", "Powder batch or supplier batch number", "Recommended"],
        ["sample_geometry", "Cube, cylinder, tensile bar, etc.", "Required"],
        ["sample_position_x_y", "Position of the sample on the build plate", "Required"],
        ["laser_power_W", "Laser power in watts", "Required"],
        ["scan_speed_mm_s", "Scan speed in mm/s", "Required"],
        ["hatch_spacing_um", "Hatch spacing in micrometers", "Required"],
        ["layer_thickness_um", "Layer thickness in micrometers", "Required"],
        ["spot_size_um", "Laser spot size or beam diameter", "Required"],
        ["scan_strategy", "Stripe, chessboard, island, rotation angle, etc.", "Required"],
        ["measured_density_percent", "Final relative density of the printed sample", "Required"],
        ["porosity_percent", "Porosity percentage if available", "Recommended"],
        ["measurement_method", "Archimedes, micro-CT, microscopy, image analysis, etc.", "Required"],
        ["defect_or_failure_note", "Good, lack of fusion, keyhole, crack, failed, etc.", "Required"],
        ["sensor_file_name", "Pyrometer, OT, melt-pool, thermal camera, or layer-image file", "If available"],
        ["general_notes", "Any extra observation during printing or inspection", "Recommended"],
    ],
    columns=["Column name", "What to record", "Priority"]
)

st.dataframe(sample_data, use_container_width=True)

st.header("2. Extra build information if available")

extra_data = pd.DataFrame(
    [
        ["build_plate_temperature_C", "Build plate/preheating temperature"],
        ["oxygen_level_ppm", "Oxygen level during printing"],
        ["gas_flow_condition", "Gas flow condition or machine setting"],
        ["powder_reuse_number", "How many times the powder was reused"],
        ["powder_d10_d50_d90_um", "Powder particle-size distribution"],
        ["contour_power_W", "Contour laser power if used"],
        ["contour_speed_mm_s", "Contour scan speed if used"],
        ["number_of_layers", "Total number of printed layers"],
        ["machine_log_file", "Machine warning/error/build log"],
        ["layer_log_file", "Layer-wise process log if available"],
    ],
    columns=["Extra information", "Why it helps"]
)

st.dataframe(extra_data, use_container_width=True)

st.header("3. Sensor data to collect if available")

sensor_data = pd.DataFrame(
    [
        ["pyrometer_file", "Raw pyrometer or IR signal file"],
        ["thermal_camera_file", "Thermal camera recording or layer images"],
        ["melt_pool_file", "Melt-pool monitoring file"],
        ["powder_bed_images", "Layer-wise powder-bed or optical tomography images"],
        ["thermal_mean", "Average thermal/sensor signal"],
        ["thermal_max", "Maximum thermal/sensor signal"],
        ["thermal_std", "Thermal signal fluctuation"],
        ["thermal_iqr", "Interquartile range; useful for melt-pool stability"],
        ["thermal_mode", "Most frequent thermal/sensor value"],
        ["thermal_skewness", "Asymmetry of thermal signal distribution"],
        ["thermal_kurtosis", "Outlier/sharpness behaviour of thermal signal"],
        ["hotspot_fraction", "Fraction of overheated or hotspot area"],
        ["number_of_anomalous_layers", "Number of suspicious layers"],
    ],
    columns=["Sensor item", "Meaning"]
)

st.dataframe(sensor_data, use_container_width=True)

st.header("4. Physics-informed features calculated by the app")

st.write(
    """
    The lab does not need to manually calculate these values.  
    The app can calculate them from laser power, scan speed, hatch spacing,
    layer thickness, spot size, and material properties.
    """
)

physics_features = pd.DataFrame(
    [
        ["LED", "P / v", "Linear energy density"],
        ["AED", "P / (v × h)", "Areal energy density"],
        ["VED", "P / (v × h × t)", "Volumetric energy density"],
        ["hatch_to_spot_ratio", "h / spot_size", "Track overlap indicator"],
        ["hatch_to_layer_ratio", "h / t", "Spacing compared with layer thickness"],
        ["spot_to_layer_ratio", "spot_size / t", "Beam size compared with layer thickness"],
        ["normalized_VED", "η × VED / (ρ × Cp × ΔT)", "Energy input normalized by material heating need"],
        ["thermal_stability_score", "Function of std, IQR, and mode", "Sensor-based melt-pool stability"],
    ],
    columns=["Feature", "Formula / source", "Physical meaning"]
)

st.dataframe(physics_features, use_container_width=True)

st.header("5. Model types and required inputs")

model_inputs = pd.DataFrame(
    [
        [
            "Model A: Process-only baseline",
            "laser_power_W, scan_speed_mm_s, hatch_spacing_um, layer_thickness_um, spot_size_um",
            "density_percent or porosity_percent",
            "Baseline model using only machine settings."
        ],
        [
            "Model B: Physics-informed model",
            "LED, AED, VED, normalized_VED, hatch_to_spot_ratio, hatch_to_layer_ratio, spot_to_layer_ratio",
            "density_percent or porosity_percent",
            "Explainable model based on energy input, track overlap, and heat-related descriptors."
        ],
        [
            "Model C: Sensor-only model",
            "thermal_mean, thermal_max, thermal_std, thermal_iqr, thermal_mode, hotspot_fraction, anomalous_layers",
            "density_percent, porosity_percent, or defect class",
            "In-situ monitoring model if pyrometry, OT, melt-pool, or thermal-image data is available."
        ],
        [
            "Model D: Hybrid physics + sensor model",
            "process parameters + physics-informed features + sensor descriptors",
            "density_percent, porosity_percent, defect class, or pass/fail",
            "Strongest version because it combines process design, physics meaning, and sensor data."
        ],
        [
            "Model E: Defect classification model",
            "process features, physics features, and sensor features if available",
            "good / lack_of_fusion / keyhole / crack / failed",
            "Classifies defect type, but needs enough labelled examples for each defect class."
        ],
        [
            "Model F: Pass/fail risk model",
            "process features, physics features, and optional sensor features",
            "pass or fail",
            "Simpler first classification model for small datasets such as 50–80 samples."
        ],
    ],
    columns=["Model", "Input features needed", "Target/output needed", "Purpose"]
)

st.dataframe(model_inputs, use_container_width=True)

st.header("6. Recommended training order")

st.markdown(
    """
    1. Train a **process-only baseline model**.  
    2. Train a **physics-informed model** using calculated energy and ratio features.  
    3. If sensor data exists, train a **sensor-only model**.  
    4. Combine everything into a **hybrid model**.  
    5. Compare models using **R², MAE, and RMSE** for density/porosity prediction.  
    6. Use feature importance or SHAP to explain why the model predicts good or poor quality.
    """
)

st.header("7. Simple physical interpretation rules")

rules = pd.DataFrame(
    [
        ["Low VED / low LED", "Possible lack of fusion"],
        ["Very high VED", "Possible keyhole, evaporation, or overheating"],
        ["High thermal IQR or standard deviation", "Unstable melt pool"],
        ["Large hatch-to-spot ratio", "Poor track overlap / lack-of-fusion risk"],
        ["Many anomalous layers", "Higher defect probability"],
        ["High density and stable sensor signal", "Likely good processing condition"],
    ],
    columns=["Observation", "Possible physical meaning"]
)

st.dataframe(rules, use_container_width=True)

st.success(
    """
    Practical goal: collect at least 50–80 samples with process parameters and measured density/porosity.  
    Sensor data is not mandatory for the first model, but it is important for a stronger in-situ quality-control model.
    """
)
