from pathlib import Path
import streamlit as st
import base64

_ASSETS = Path(__file__).resolve().parent / "assets"
_LOGO = _ASSETS / "honeywell_logo.png"

def header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)

def brand_sidebar(logo_path: str | None = None, height: int = 32) -> None:
    """
    Shows the Honeywell logo ABOVE Streamlit's built-in sidebar page list
    and hides the collapse button. Call this AFTER st.set_page_config().
    """
    p = Path(logo_path) if logo_path else _LOGO
    if not p.exists():
        return
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")

    st.markdown(
        f"""
        <style>
        /* Add logo above the built-in sidebar nav */
        [data-testid="stSidebarNav"]::before {{
            content: "";
            display: block;
            height: {height}px;
            margin: 10px 0 8px 12px;
            background-image: url('data:image/png;base64,{b64}');
            background-repeat: no-repeat;
            background-size: auto {height}px;
        }}

        /* Hide the collapse/chevron control (cover multiple Streamlit versions) */
        [data-testid="stSidebar"] [data-testid="collapsedControl"],
        [data-testid="stSidebar"] button[title="Collapse sidebar"],
        [data-testid="stSidebar"] [data-testid="baseButton-header"] {{
            display: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )