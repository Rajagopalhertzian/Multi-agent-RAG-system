"""
frontend/app.py
Streamlit UI for the Multi-Agent Document Intelligence Platform.
Run with: streamlit run frontend/app.py
"""
import streamlit as st
import requests
import json
import time
from pathlib import Path

import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card { background: #f8f9fa; border-radius: 8px; padding: 12px; margin: 4px 0; }
.citation-box { background: #e8f4f8; border-left: 3px solid #1a73e8; padding: 10px; border-radius: 4px; margin: 6px 0; font-size: 0.9em; }
.agent-badge { background: #e8eaf6; color: #3949ab; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin: 2px; display: inline-block; }
.score-good { color: #2e7d32; font-weight: bold; }
.score-medium { color: #f57c00; font-weight: bold; }
.score-low { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Doc Intelligence")
    st.caption("Multi-Agent RAG Platform")
    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"✅ API Online")
        st.metric("Chunks indexed", health.get("chunks_indexed", 0))
        st.metric("Documents loaded", health.get("documents_loaded", 0))
    except Exception:
        st.error("❌ API Offline — start the server first")
        st.code("uvicorn api.main:app --reload")

    st.divider()
    st.markdown("**Tech Stack**")
    for tech in ["LangGraph", "ChromaDB + FAISS", "CLIP Vision Agent", "RAGAS Eval", "FastAPI"]:
        st.markdown(f"• {tech}")

    st.divider()
    st.markdown("**Built by** Raja Gopal S")
    st.markdown("[GitHub](https://github.com/Rajagopalhertzian)")

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Ingest Documents", "💬 Query", "📊 Evaluate"])

# ── Tab 1: Ingest ──────────────────────────────────────────────────────────────
with tab1:
    st.header("Document Ingestion")
    st.markdown("Upload PDFs or paste text. The **Vision Agent** (CLIP) will automatically describe any images, charts, and tables found in PDFs.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload PDF")
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded and st.button("Ingest PDF", type="primary"):
            with st.spinner(f"Ingesting {uploaded.name}... extracting text & images..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest/pdf",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        doc = resp.json()
                        st.success(f"✅ Ingested! {doc['num_chunks']} chunks created")
                        if doc.get("has_images"):
                            st.info("🖼️ Vision Agent detected and described images/charts")
                        st.json(doc)
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Failed: {e}")

    with col2:
        st.subheader("📝 Paste Text")
        title = st.text_input("Document title", "My Document")
        text_input = st.text_area("Paste your text here", height=200)
        if st.button("Ingest Text") and text_input.strip():
            with st.spinner("Ingesting text..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest/text",
                        json={"text": text_input, "title": title},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        doc = resp.json()
                        st.success(f"✅ Ingested! {doc['num_chunks']} chunks created")
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()
    st.subheader("📚 Ingested Documents")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=5).json()
        if docs:
            for doc in docs:
                with st.expander(f"📄 {doc['title']} — {doc['num_chunks']} chunks"):
                    cols = st.columns(4)
                    cols[0].metric("Chunks", doc["num_chunks"])
                    cols[1].metric("Has Images", "Yes" if doc.get("has_images") else "No")
                    cols[2].metric("Status", doc["status"])
                    cols[3].metric("Doc ID", doc["doc_id"][:8] + "...")
                    if st.button(f"🗑️ Delete", key=f"del_{doc['doc_id']}"):
                        requests.delete(f"{API_URL}/documents/{doc['doc_id']}")
                        st.rerun()
        else:
            st.info("No documents ingested yet. Upload a PDF to get started.")
    except Exception as e:
        st.warning(f"Could not fetch documents: {e}")


# ── Tab 2: Query ───────────────────────────────────────────────────────────────
with tab2:
    st.header("Multi-Agent Q&A")
    st.markdown("Ask anything about your documents. The **Router Agent** classifies your question, **Retrieval Agent** finds relevant chunks, and **Synthesis Agent** generates a cited answer.")

    query = st.text_area("Your question", placeholder="What are the key findings in the document?", height=80)

    col1, col2, col3 = st.columns(3)
    top_k = col1.slider("Top-K chunks", 3, 15, 5)
    use_reranker = col2.checkbox("Cross-encoder reranker", value=True)
    show_trace = col3.checkbox("Show agent trace", value=True)

    if st.button("🚀 Run Pipeline", type="primary", disabled=not query.strip()):
        with st.spinner("Running multi-agent pipeline..."):
            start = time.time()
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"query": query, "top_k": top_k, "use_reranker": use_reranker},
                    timeout=120,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    elapsed = time.time() - start

                    # Answer
                    st.subheader("💡 Answer")
                    st.markdown(result["answer"])

                    # Agent trace
                    if show_trace and result.get("agent_trace"):
                        st.markdown("**Agent pipeline:**")
                        trace_html = " → ".join(
                            f'<span class="agent-badge">{a}</span>'
                            for a in result["agent_trace"]
                        )
                        st.markdown(trace_html, unsafe_allow_html=True)

                    st.caption(f"⏱️ {result.get('latency_ms', elapsed*1000):.0f}ms")

                    # Citations
                    if result.get("citations"):
                        st.subheader("📎 Citations")
                        for i, cit in enumerate(result["citations"], 1):
                            st.markdown(
                                f'<div class="citation-box"><strong>[{i}]</strong> '
                                f'<code>{Path(cit["source_path"]).name}</code> — '
                                f'Score: {cit["relevance_score"]:.3f}<br>'
                                f'<em>{cit["content_snippet"][:200]}...</em></div>',
                                unsafe_allow_html=True,
                            )

                    # RAGAS scores
                    if result.get("evaluation"):
                        st.subheader("📊 RAGAS Evaluation")
                        eval_data = result["evaluation"]
                        cols = st.columns(len(eval_data))
                        for i, (metric, score) in enumerate(eval_data.items()):
                            if isinstance(score, float):
                                cls = "score-good" if score > 0.7 else ("score-medium" if score > 0.4 else "score-low")
                                cols[i].metric(metric.replace("_", " ").title(), f"{score:.3f}")

                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"Pipeline failed: {e}")


# ── Tab 3: Evaluate ────────────────────────────────────────────────────────────
with tab3:
    st.header("Batch Evaluation")
    st.markdown("Run RAGAS evaluation on multiple Q&A pairs to measure your pipeline quality.")

    st.info("💡 This is your resume differentiator — showing you can measure and improve RAG systems, not just build them.")

    sample_qa = [
        {"question": "What is machine learning?", "expected": "Machine learning is a subset of AI..."},
        {"question": "What are neural networks?", "expected": "Neural networks are computational models..."},
    ]

    st.subheader("Sample Evaluation Dataset")
    st.json(sample_qa)

    if st.button("Run Batch Evaluation"):
        with st.spinner("Running evaluation on all Q&A pairs..."):
            results = []
            progress = st.progress(0)
            for i, qa in enumerate(sample_qa):
                try:
                    resp = requests.post(
                        f"{API_URL}/query",
                        json={"query": qa["question"], "top_k": 5},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        r = resp.json()
                        results.append({
                            "question": qa["question"],
                            "answer": r.get("answer", ""),
                            "eval": r.get("evaluation", {}),
                        })
                    progress.progress((i + 1) / len(sample_qa))
                except Exception as e:
                    st.warning(f"Failed for: {qa['question']}: {e}")

            if results:
                st.success(f"Evaluated {len(results)} questions")
                for r in results:
                    with st.expander(r["question"][:60] + "..."):
                        st.write(r["answer"][:500])
                        if r["eval"]:
                            st.json(r["eval"])
