import bootstrap
import streamlit as st
from ui import header, brand_sidebar

st.set_page_config(page_title="LuminAI", page_icon="app/assets/honeywell_logo.png",
                   layout="wide", initial_sidebar_state="expanded")
brand_sidebar()

header("LuminAI", "AI assistant for aerospace engineering workflows (RAG • diff • drafts)")

st.divider()

# =========================
# Current use cases
# =========================
st.subheader("Functionality overview")



st.markdown(
        """
- **Search & Chat** – Ask natural-language questions about your ingested PDFs/DOCX.  
  Answers are grounded in retrieved passages and include sources.
- **Change Impact** *(implementation pending)* – Compare two document versions, view a redline, and get quick impact notes.
- **Report Drafts** *(implementation pending)* – Generate first-pass summaries/drafts from the most relevant passages.
- **Meeting Actions** *(implementation pending)* – Extract owners/tasks/dates from meeting notes.
        """
    )



st.divider()

# =========================
# Next steps (roadmap)
# =========================
st.subheader("Next steps")
st.markdown(
    """
1. **Inline citations in replies** – show file + section right in the chat answer.
2. **“Rebuild Index” button** – trigger `python -m luminai.ingest data\\` from the UI.
3. **Better ranking** – enable a reranker for higher-quality top hits.
4. **OCR fallback** – handle scanned PDFs (Tesseract) when text isn’t extractable.
5. **Small evaluation set** – 10–20 Q&A to track answer quality over time.
6. **Exports** – DOCX/PDF for report drafts; CSV for actions.
7. **SharePoint/OneDrive ingestion** – pull docs directly from the project site.
"""
)

# Helpful reminder
st.subheader("Update the knowledge base")
st.code("python -m luminai.ingest data\\", language="bash")
st.caption("Run this after adding or changing files under the `data/` folder.")


