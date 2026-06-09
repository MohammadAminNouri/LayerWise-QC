
from __future__ import annotations

import html
import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

.lwq-hero {
    border: 1px solid rgba(120, 120, 120, 0.25);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(120,120,120,0.10), rgba(120,120,120,0.03));
}

.lwq-hero h1 {
    margin: 0 0 0.4rem 0;
    font-size: 2.15rem;
    line-height: 1.15;
}

.lwq-hero p {
    margin: 0.25rem 0;
    font-size: 1.03rem;
    line-height: 1.55;
    opacity: 0.92;
}

.lwq-card {
    border: 1px solid rgba(120, 120, 120, 0.25);
    border-radius: 16px;
    padding: 1.05rem 1.1rem;
    margin-bottom: 0.9rem;
    background: rgba(120, 120, 120, 0.055);
    min-height: 170px;
}

.lwq-card h3 {
    margin-top: 0;
    margin-bottom: 0.45rem;
    font-size: 1.10rem;
}

.lwq-card p {
    margin: 0.25rem 0;
    line-height: 1.48;
    opacity: 0.92;
}

.lwq-label {
    display: inline-block;
    border-radius: 999px;
    padding: 0.22rem 0.65rem;
    margin-bottom: 0.55rem;
    font-size: 0.78rem;
    font-weight: 650;
    letter-spacing: 0.01rem;
    border: 1px solid rgba(120, 120, 120, 0.35);
}

.lwq-label-demo {
    background: rgba(90, 140, 255, 0.13);
}

.lwq-label-training {
    background: rgba(255, 190, 80, 0.14);
}

.lwq-label-validation {
    background: rgba(90, 180, 120, 0.14);
}

.lwq-label-limited {
    background: rgba(255, 110, 90, 0.13);
}

.lwq-step {
    border-left: 4px solid rgba(120, 120, 120, 0.55);
    padding: 0.55rem 0 0.55rem 0.9rem;
    margin: 0.55rem 0;
}

.lwq-step strong {
    display: block;
    margin-bottom: 0.15rem;
}

.lwq-muted {
    opacity: 0.72;
    font-size: 0.92rem;
}

.lwq-table-note {
    font-size: 0.93rem;
    line-height: 1.45;
    opacity: 0.88;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, body: str, note: str | None = None) -> None:
    note_html = f"<p class='lwq-muted'>{html.escape(note)}</p>" if note else ""
    st.markdown(
        f"""
<div class="lwq-hero">
  <h1>{html.escape(title)}</h1>
  <p>{html.escape(body)}</p>
  {note_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str, label: str | None = None, label_kind: str = "demo") -> None:
    label_html = ""
    if label:
        label_html = f"<span class='lwq-label lwq-label-{html.escape(label_kind)}'>{html.escape(label)}</span>"
    st.markdown(
        f"""
<div class="lwq-card">
  {label_html}
  <h3>{html.escape(title)}</h3>
  <p>{html.escape(body)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def step(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="lwq-step">
  <strong>{html.escape(title)}</strong>
  <span>{html.escape(body)}</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def claim_level_box(level: str, allowed: str, missing: str) -> None:
    st.markdown(
        f"""
<div class="lwq-card">
  <span class="lwq-label lwq-label-limited">Claim level</span>
  <h3>{html.escape(level)}</h3>
  <p><strong>Allowed statement:</strong> {html.escape(allowed)}</p>
  <p><strong>Still needed:</strong> {html.escape(missing)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )
