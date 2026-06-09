from __future__ import annotations

import streamlit as st

from app.dashboard_ux import (
    TAB_GUIDES,
    render_claim_level,
    render_data_manifest_answers,
    render_decision_reasons,
    render_feedforward_summary,
    render_fusion_audit,
    render_sensor_evidence_table,
    render_tab_guide,
)

st.set_page_config(page_title="Dashboard Tab Guide", layout="wide")

st.title("Dashboard tab guide")
st.caption("Purpose, interpretation, limits, and next action for each dashboard tab.")

selected = st.selectbox(
    "Select dashboard tab",
    list(TAB_GUIDES.keys()),
    format_func=lambda x: TAB_GUIDES[x].tab,
)

render_tab_guide(selected)

st.divider()
st.subheader("Example content for this tab")

if selected == "overview":
    render_claim_level(
        mode="Demo / proxy mode",
        has_ground_truth=False,
        has_real_sensor_data=False,
        has_group_split=False,
    )
    st.info("Use the overview as a short result page: current state, claim level, main reason, and next action.")

elif selected == "process inputs":
    st.dataframe(
        [
            {"Input": "laser_power_w", "Status": "defined", "Use": "VED, line energy, power density"},
            {"Input": "scan_speed_mm_s", "Status": "defined", "Use": "VED and residence-time proxy"},
            {"Input": "hatch_distance_mm", "Status": "defined", "Use": "VED and hatch/spot ratio"},
            {"Input": "layer_thickness_mm", "Status": "defined", "Use": "VED"},
            {"Input": "spot_size_um", "Status": "strongly recommended", "Use": "beam area and power density"},
            {"Input": "material", "Status": "recommended", "Use": "defines validation domain"},
            {"Input": "machine", "Status": "recommended", "Use": "defines transferability"},
        ],
        use_container_width=True,
        hide_index=True,
    )

elif selected == "live decision":
    render_decision_reasons(
        [
            "VED is above the selected reference.",
            "Power density is high because the beam diameter is small relative to the selected power.",
            "The result is from proxy mode, so it should be treated as a risk indicator.",
        ]
    )

elif selected == "sensor signals":
    render_sensor_evidence_table(
        [
            {
                "sensor": "OT",
                "status": "available",
                "main_indicator": "high-intensity regions",
                "risk_direction": "high-energy",
                "reliability": "demo",
                "limitation": "synthetic or unvalidated source",
            },
            {
                "sensor": "MPM",
                "status": "missing",
                "main_indicator": "not available",
                "risk_direction": "unknown",
                "reliability": "none",
                "limitation": "no aligned file",
            },
            {
                "sensor": "PBI",
                "status": "available",
                "main_indicator": "powder-bed uniformity proxy",
                "risk_direction": "recoating-related",
                "reliability": "demo",
                "limitation": "not ground truth",
            },
        ]
    )

elif selected == "physics features":
    st.dataframe(
        [
            {"Group": "Energy input", "Features": "VED, line energy, areal energy", "Interpretation": "First-order heat input descriptors"},
            {"Group": "Beam concentration", "Features": "spot size, beam area, power density", "Interpretation": "Shows whether energy is concentrated or distributed"},
            {"Group": "Track overlap", "Features": "hatch/spot ratio, spot overlap ratio", "Interpretation": "Helps interpret lack-of-fusion risk"},
            {"Group": "Thermal proxy", "Features": "residence and heat-memory terms", "Interpretation": "Approximate thermal accumulation indicators"},
        ],
        use_container_width=True,
        hide_index=True,
    )

elif selected == "sensor fusion":
    render_fusion_audit(
        [
            {"Source": "Process proxy", "Prediction": "high-energy risk", "Confidence": 0.68, "Weight": 0.40},
            {"Source": "OT", "Prediction": "high-energy risk", "Confidence": 0.71, "Weight": 0.35},
            {"Source": "PBI", "Prediction": "standard", "Confidence": 0.54, "Weight": 0.25},
        ],
        fused_result="high-energy risk",
        uncertainty_note="PBI disagrees with the process and OT indicators, so confidence should be limited.",
    )

elif selected == "feed-forward control":
    render_feedforward_summary(
        current_issue="VED and power density are above the selected reference.",
        recommendation="Reduce laser power slightly or increase scan speed within a conservative adjustment limit.",
        expected_effect="Lower energy input and reduce high-energy/keyhole-spatter tendency.",
    )

elif selected == "data / manifest":
    render_data_manifest_answers(
        can_train=False,
        can_claim_accuracy=False,
        reasons=[
            "No independent ground truth column is available.",
            "No group-wise split by build_id or specimen_id is defined.",
            "Sensor paths are missing or not aligned with layers.",
        ],
    )
