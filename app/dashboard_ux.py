from __future__ import annotations

from dataclasses import dataclass
import html

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class TabGuide:
    tab: str
    purpose: str
    main_output: str
    use_when: str
    limitation: str
    next_action: str


TAB_GUIDES = {
    "overview": TabGuide(
        "Overview",
        "Summarize the workflow status in one place.",
        "Current mode, process state, risk direction, claim level, and missing evidence.",
        "Use this first when presenting the app or checking whether the result is demo-based or validation-based.",
        "A summary is not a validation. It depends on the selected mode, data source, and ground truth.",
        "Check the decision source, then inspect process inputs, physics features, and data-readiness warnings.",
    ),
    "process inputs": TabGuide(
        "Process inputs",
        "Define LPBF process parameters and reference values.",
        "Laser power, scan speed, hatch distance, layer thickness, spot size, VED, power density, and reference ratios.",
        "Use this to check whether the parameter set is physically reasonable.",
        "Process parameters alone cannot prove internal defects.",
        "Confirm material, machine, spot size, reference VED, and operating range.",
    ),
    "live decision": TabGuide(
        "Live decision",
        "Show the current risk direction and the main reasons behind it.",
        "Risk class, uncertainty, decision source, and top contributing reasons.",
        "Use this to see whether the current state is closer to standard, low-energy, or high-energy behavior.",
        "In demo mode this is a proxy risk indicator, not a validated defect prediction.",
        "Read the top reasons and check whether the result is supported by sensor evidence and ground truth.",
    ),
    "sensor signals": TabGuide(
        "Sensor signals",
        "Display the sensor evidence used by the workflow.",
        "Available modalities, previews, descriptors, and sensor-specific warnings.",
        "Use this to check whether OT, MPM, PBI, pyrometry, or logs support the interpretation.",
        "Sensor data is useful only when aligned with build, specimen, layer, process condition, and ground truth.",
        "Check missing modalities, image quality, sensor disagreement, and whether the data is real and aligned.",
    ),
    "physics features": TabGuide(
        "Physics features",
        "Translate process parameters into interpretable physics descriptors.",
        "VED, line energy, areal energy, beam area, power density, hatch/spot ratio, and thermal proxies.",
        "Use this to explain why a parameter set is below, near, or above the selected reference.",
        "Physics descriptors are not labels and do not replace measurement-based validation.",
        "Compare descriptors with the selected reference and focus on the largest deviations.",
    ),
    "sensor fusion": TabGuide(
        "Sensor fusion",
        "Show how process and sensor evidence are combined.",
        "Per-modality scores, fusion weights, fused result, disagreement, and uncertainty.",
        "Use this when several sensor sources are available and must be compared.",
        "Fusion is weak if modalities are missing, poorly aligned, or trained on synthetic data only.",
        "Inspect whether one sensor dominates and whether modalities agree or conflict.",
    ),
    "feed-forward control": TabGuide(
        "Feed-forward control",
        "Provide a conservative process-adjustment suggestion.",
        "Suggested direction of change, expected effect, before/after descriptors, and safety limit.",
        "Use this when the workflow indicates low-energy or high-energy risk.",
        "This is advisory only and must not be treated as automatic machine control.",
        "Check the reason for the suggestion and validate any parameter change experimentally.",
    ),
    "data / manifest": TabGuide(
        "Data / manifest",
        "Check whether the dataset supports training or only workflow demonstration.",
        "Manifest completeness, ground truth, image paths, split quality, class balance, and leakage risk.",
        "Use this before training or reporting accuracy.",
        "A valid manifest does not prove accuracy. It only means the dataset is structured enough to test.",
        "Fix missing build IDs, split columns, ground truth, sensor paths, and class imbalance before training.",
    ),
}


def _styles() -> None:
    st.markdown(
        """
<style>
.lwq-guide {
    border: 1px solid rgba(120,120,120,0.28);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin: 0.6rem 0 1rem 0;
    background: rgba(120,120,120,0.055);
}
.lwq-title {
    font-size: 1.12rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.lwq-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem 1rem;
}
.lwq-item { line-height: 1.45; }
.lwq-item strong { display: block; margin-bottom: 0.15rem; }
.lwq-note {
    border-left: 4px solid rgba(150,150,150,0.7);
    padding: 0.6rem 0.8rem;
    margin-top: 0.75rem;
    background: rgba(120,120,120,0.045);
}
@media (max-width: 900px) { .lwq-grid { grid-template-columns: 1fr; } }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_tab_guide(tab_key: str) -> None:
    _styles()
    guide = TAB_GUIDES.get(tab_key.lower().strip())
    if guide is None:
        return

    st.markdown(
        f"""
<div class="lwq-guide">
  <div class="lwq-title">{html.escape(guide.tab)}</div>
  <div class="lwq-grid">
    <div class="lwq-item"><strong>Purpose</strong>{html.escape(guide.purpose)}</div>
    <div class="lwq-item"><strong>Main output</strong>{html.escape(guide.main_output)}</div>
    <div class="lwq-item"><strong>Use when</strong>{html.escape(guide.use_when)}</div>
    <div class="lwq-item"><strong>Limitation</strong>{html.escape(guide.limitation)}</div>
  </div>
  <div class="lwq-note"><strong>Next action:</strong> {html.escape(guide.next_action)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_claim_level(mode: str, has_ground_truth: bool, has_real_sensor_data: bool, has_group_split: bool) -> None:
    _styles()

    if mode.lower().startswith("demo"):
        level = "Workflow demonstration"
        allowed = "The workflow and interpretation logic can be demonstrated."
        missing = "Real aligned sensor data, ground truth, and group-wise validation."
    elif has_ground_truth and has_real_sensor_data and has_group_split:
        level = "Validation experiment"
        allowed = "Training and validation can be discussed within the stated domain."
        missing = "External validation on unseen builds before broader claims."
    elif has_ground_truth:
        level = "Training experiment"
        allowed = "Model-development experiments can be performed."
        missing = "Aligned sensor data and leakage-safe group splitting."
    else:
        level = "Workflow testing only"
        allowed = "The data structure and software pipeline can be checked."
        missing = "Independent ground truth."

    st.markdown(
        f"""
<div class="lwq-guide">
  <div class="lwq-title">Claim level: {html.escape(level)}</div>
  <div class="lwq-grid">
    <div class="lwq-item"><strong>Supported statement</strong>{html.escape(allowed)}</div>
    <div class="lwq-item"><strong>Still needed</strong>{html.escape(missing)}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_reasons(reasons: list[str], title: str = "Decision reasons") -> None:
    st.subheader(title)
    if not reasons:
        st.info("No decision reasons were provided for this result.")
        return
    st.dataframe(
        pd.DataFrame({"Rank": list(range(1, len(reasons) + 1)), "Reason": reasons}),
        use_container_width=True,
        hide_index=True,
    )


def render_sensor_evidence_table(rows: list[dict]) -> None:
    st.subheader("Sensor evidence")
    if not rows:
        st.info("No sensor evidence is available for the current sample.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_fusion_audit(rows: list[dict], fused_result: str | None = None, uncertainty_note: str | None = None) -> None:
    st.subheader("Fusion audit")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No fusion audit is available.")
    if fused_result:
        st.info(f"Fused result: {fused_result}")
    if uncertainty_note:
        st.warning(f"Uncertainty note: {uncertainty_note}")


def render_feedforward_summary(current_issue: str, recommendation: str, expected_effect: str) -> None:
    _styles()
    st.subheader("Control recommendation summary")
    st.markdown(
        f"""
<div class="lwq-guide">
  <div class="lwq-grid">
    <div class="lwq-item"><strong>Current issue</strong>{html.escape(current_issue)}</div>
    <div class="lwq-item"><strong>Recommended adjustment</strong>{html.escape(recommendation)}</div>
    <div class="lwq-item"><strong>Expected effect</strong>{html.escape(expected_effect)}</div>
    <div class="lwq-item"><strong>Limit</strong>Advisory only. Validate experimentally before applying to a machine.</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_data_manifest_answers(can_train: bool, can_claim_accuracy: bool, reasons: list[str]) -> None:
    st.subheader("Dataset answers")
    c1, c2 = st.columns(2)
    c1.metric("Can train?", "Yes" if can_train else "No")
    c2.metric("Can claim accuracy?", "Yes" if can_claim_accuracy else "No")
    if reasons:
        st.dataframe(pd.DataFrame({"Reason": reasons}), use_container_width=True, hide_index=True)
