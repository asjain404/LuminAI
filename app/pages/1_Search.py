# app/pages/1_Search.py
from app.ui import header, brand_sidebar
import bootstrap

import time
from time import perf_counter
import streamlit as st
from luminai.ingest import index_ready, INDEX_DIR
from luminai.qa import answer, summarize_evidence

brand_sidebar() 
header("LuminAI — Search & Chat", "Ask questions about your ingested PDFs/DOCX")

if not index_ready():
    st.warning(f"Index not found. Build it first:  python -m luminai.ingest data/")
    st.stop()
else:
    st.success(f"Index ready: {INDEX_DIR}")

with st.sidebar:
    st.subheader("Chat settings")
    top_k = st.slider("Top-k documents", 3, 15, 5, 1)
    use_summary = st.checkbox("Summarize evidence into answer", True)
    show_sources = st.checkbox("Show sources for each reply", True)
    show_timings = st.checkbox("Show processing timings", False)
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me about your PDFs/DOCX—requirements, limits, procedures, etc."}
    ]

for m in st.session_state["messages"]:
    st.chat_message(m["role"]).write(m["content"])

prompt = st.chat_input("Type your question…")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    assistant_box = st.chat_message("assistant")
    out_placeholder = assistant_box.empty()
    progress = st.progress(0)
    status = st.empty()
    t0 = perf_counter()

    try:
        with st.spinner("Searching your documents…"):
            status.info("Step 1/2 • Finding relevant passages and extracting an answer…")
            out = answer(prompt, k=top_k)
            progress.progress(70)

        base = out["result"]["answer"] or "No exact span found—summarizing the most relevant documents."
        reply = base

        if use_summary:
            with st.spinner("Summarizing evidence…"):
                status.info("Step 2/2 • Summarizing evidence…")
                try:
                    summary = summarize_evidence([h["text"] for h in out["hits"]])
                    if summary and summary.strip():
                        reply = f"{base}\n\n{summary}"
                except Exception:
                    pass
                progress.progress(100)

        out_placeholder.write(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})

        if show_timings:
            t_total = perf_counter() - t0
            with st.expander("Timings"):
                st.write(f"End-to-end: {t_total:.2f}s")

        if show_sources:
            with st.expander("Sources (top matches)"):
                for i, h in enumerate(out["hits"][:5], 1):
                    st.markdown(f"**{i}.** `{h['meta']['file']}` • score {h['score']:.3f}")
                    st.write(h["text"])
    finally:
        time.sleep(0.2)
        progress.empty()
        status.empty()
