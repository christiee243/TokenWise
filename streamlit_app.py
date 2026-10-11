"""
TokenWise - Streamlit Frontend
Matches the team architecture diagram:
[UI: Streamlit / Next.js] -> [M4: FastAPI Backend] -> [M1: PDF Processing] & [M2: RAG Retrieval]
"""

import os
import re
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="TokenWise - Token Reducer & RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #101827;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #556070;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .chunk-card {
        background-color: #f8f9f5;
        border: 1px solid rgba(16, 24, 39, 0.1);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .pill {
        display: inline-block;
        border-radius: 9999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get or create embedded RAG retriever
def get_local_retriever():
    if "local_retriever" not in st.session_state:
        from backend.rag import RAGRetriever
        st.session_state["local_retriever"] = RAGRetriever()
    return st.session_state["local_retriever"]

st.sidebar.title("⚡ TokenWise")
st.sidebar.markdown("**Token Optimization Suite**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Tool",
    [
        "📄 PDF Analyzer & Reducer",
        "💬 Prompt Optimizer",
        "🔍 Document Q&A (RAG)",
        "🎯 Context Optimizer (Member 3)",
        "🩺 Backend Health",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Architecture Roles")
st.sidebar.info(
    "**Member 1: PDF & Prompts**\n"
    "- PDF text & table extraction\n"
    "- Boilerplate removal & Markdown\n"
    "- Token reduction measurement\n\n"
    "**Member 2: RAG & Search**\n"
    "- Query analyzer (intent & complexity)\n"
    "- Dense vector embeddings\n"
    "- FAISS cosine vector storage\n"
    "- Semantic chunk retrieval\n\n"
    "**Member 3: Token Optimization**\n"
    "- Query-intent adaptive token budgeting\n"
    "- Redundancy removal & deduplication\n"
    "- Budget-bounded chunk ranking\n"
    "- Structured Markdown context reconstruction\n"
    "- Context efficiency report generation"
)

# -----------------------------------------------------------------------------
# Tool 1: PDF Analyzer & Reducer (Member 1)
# -----------------------------------------------------------------------------
if app_mode == "📄 PDF Analyzer & Reducer":
    st.markdown('<div class="main-header">Less Document. <span style="color: #6d7d17;">More Signal.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract structured Markdown, remove document boilerplate, and cut prompt tokens.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.subheader("1. Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])

        compression = st.select_slider(
            "Compression Level",
            options=["Light", "Balanced", "Maximum"],
            value="Balanced",
            help="Light keeps maximum detail. Balanced removes filler and cleans repetition. Maximum keeps core bullet points and tables.",
        )

        analyze_button = st.button("🚀 Analyze & Compress PDF", type="primary", use_container_width=True, disabled=uploaded_file is None)

    with col2:
        st.subheader("2. Optimized Result & Analytics")

        if analyze_button and uploaded_file is not None:
            with st.spinner("Extracting text, formatting tables, and cutting tokens..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {"compression": compression}
                    resp = requests.post(f"{API_URL}/api/analyze-pdf", files=files, data=data, timeout=60)

                    if resp.status_code == 200:
                        res = resp.json()
                        st.session_state["pdf_result"] = res
                        st.session_state["pdf_filename"] = uploaded_file.name
                        st.success("Extraction complete!")
                    else:
                        st.error(f"Error from backend ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.warning(f"Could not reach FastAPI backend directly: {e}. Running local fallback pipeline...")
                    # Direct module fallback
                    try:
                        from backend.pdf_processor import process_pdf
                        res = process_pdf(uploaded_file.getvalue(), compression_level=compression)
                        res["filename"] = uploaded_file.name
                        st.session_state["pdf_result"] = res
                        st.session_state["pdf_filename"] = uploaded_file.name
                        st.success("Extraction complete via embedded processing pipeline!")
                    except Exception as err:
                        st.error(f"Failed to process PDF: {err}")

        if "pdf_result" in st.session_state:
            res = st.session_state["pdf_result"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Original Tokens", f"{res['original_tokens']:,}")
            m2.metric("Optimized Tokens", f"{res['optimized_tokens']:,}")
            m3.metric("Tokens Saved", f"{res['tokens_saved']:,}")
            m4.metric("Reduction", f"{res['reduction_percent']}%")

            st.caption(f"Pages: {res['page_count']} | Sections: {res['section_count']} | Tables: {res['table_count']} | Semantic Chunks: {res['chunk_count']}")

            st.markdown("### Markdown Output")
            st.text_area("Optimized Markdown", value=res["markdown"], height=320)

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                base_name = st.session_state.get("pdf_filename", "document").replace(".pdf", "")
                st.download_button(
                    label="💾 Download .md File",
                    data=res["markdown"],
                    file_name=f"{base_name}-optimized.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with btn_col2:
                # Button to index into Member 2 RAG
                doc_clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name)[:40]
                if st.button("🔍 Index Chunks for RAG Q&A", use_container_width=True):
                    chunks = res.get("chunks", [])
                    if chunks:
                        with st.spinner("Indexing chunks into vector store..."):
                            indexed = False
                            try:
                                r = requests.post(
                                    f"{API_URL}/api/index-document",
                                    json={"document_id": doc_clean_id, "chunks": chunks},
                                    timeout=30,
                                )
                                if r.status_code == 200:
                                    st.session_state["rag_doc_id"] = doc_clean_id
                                    indexed = True
                            except Exception:
                                pass

                            if not indexed:
                                # Fallback to local
                                rtr = get_local_retriever()
                                rtr.index_document(chunks, document_id=doc_clean_id)
                                st.session_state["rag_doc_id"] = doc_clean_id

                            st.success(f"Indexed {len(chunks)} chunks for document '{doc_clean_id}'! Switch to '🔍 Document Q&A' to ask questions.")

            with st.expander(f"Inspect Chunks for M2 RAG ({res['chunk_count']} chunks)"):
                for c in res.get("chunks", []):
                    st.markdown(f"**Chunk #{c['chunk_id']}** ({c['section']} · {c['tokens']} tokens)")
                    st.code(c["text"], language="markdown")

# -----------------------------------------------------------------------------
# Tool 2: Prompt Optimizer (Member 1)
# -----------------------------------------------------------------------------
elif app_mode == "💬 Prompt Optimizer":
    st.markdown('<div class="main-header">Sharper Prompts. <span style="color: #6d7d17;">Fewer Tokens.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Strip filler phrasing, structure constraints, and reduce LLM input tokens.</div>', unsafe_allow_html=True)

    user_prompt = st.text_area(
        "Enter your rough prompt:",
        placeholder="e.g., Can you please kindly act as an expert in Python and tell me how in order to optimize tokens due to the fact that they cost money...",
        height=140,
    )

    if st.button("✨ Optimize Prompt", type="primary"):
        if not user_prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Refining prompt and trimming conversational fluff..."):
                try:
                    resp = requests.post(f"{API_URL}/api/optimize-prompt", json={"prompt": user_prompt.strip()}, timeout=15)
                    if resp.status_code == 200:
                        p_res = resp.json()
                    else:
                        raise Exception(resp.text)
                except Exception:
                    # Fallback to direct module
                    from backend.prompt_optimizer import optimize_prompt
                    p_res = optimize_prompt(user_prompt.strip())

                st.subheader("Optimized Prompt Directive")
                st.code(p_res["optimized"], language="markdown")

                c1, c2, c3 = st.columns(3)
                c1.metric("Original Tokens", p_res["original_tokens"])
                c2.metric("Optimized Tokens", p_res["optimized_tokens"])
                c3.metric("Token Savings", f"{p_res['reduction_percent']}%")

                if p_res.get("changes"):
                    st.markdown("**Changes applied:**")
                    for ch in p_res["changes"]:
                        st.markdown(f"- {ch}")

# -----------------------------------------------------------------------------
# Tool 3: Document Q&A (Member 2 RAG)
# -----------------------------------------------------------------------------
elif app_mode == "🔍 Document Q&A (RAG)":
    st.markdown('<div class="main-header">Ask Your Document. <span style="color: #6d7d17;">Semantic Search.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Member 2 RAG Retriever: Query understanding, vector similarity search, and passage retrieval.</div>', unsafe_allow_html=True)

    active_doc_id = st.session_state.get("rag_doc_id")
    has_pdf_chunks = "pdf_result" in st.session_state and st.session_state["pdf_result"].get("chunks")

    col_setup, col_search = st.columns([1, 1.8], gap="large")

    with col_setup:
        st.subheader("Index Status")
        if active_doc_id:
            st.success(f"Active Document: `{active_doc_id}`")
        else:
            st.info("No document currently indexed.")

        if has_pdf_chunks:
            chunks = st.session_state["pdf_result"]["chunks"]
            default_name = st.session_state.get("pdf_filename", "uploaded_doc").replace(".pdf", "")
            doc_name_input = st.text_input("Document Identifier", value=re.sub(r"[^a-zA-Z0-9_-]", "_", default_name))
            if st.button("⚡ Index Current Document Chunks", type="primary", use_container_width=True):
                with st.spinner("Generating embeddings and building FAISS index..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/api/index-document",
                            json={"document_id": doc_name_input, "chunks": chunks},
                            timeout=30,
                        )
                        if r.status_code == 200:
                            st.session_state["rag_doc_id"] = doc_name_input
                            st.success(f"Indexed {len(chunks)} chunks via backend API!")
                        else:
                            raise Exception(r.text)
                    except Exception as err:
                        rtr = get_local_retriever()
                        res = rtr.index_document(chunks, document_id=doc_name_input)
                        st.session_state["rag_doc_id"] = doc_name_input
                        st.success(f"Indexed {res['chunks_indexed']} chunks via local engine!")
        else:
            # Load real documents indexed in database
            db_docs = []
            try:
                r_docs = requests.get(f"{API_URL}/api/documents", timeout=5)
                if r_docs.status_code == 200:
                    raw_docs = r_docs.json().get("documents", [])
                    db_docs = [d["document_id"] if isinstance(d, dict) else str(d) for d in raw_docs]
            except Exception:
                pass

            if not db_docs:
                rtr = get_local_retriever()
                raw_docs = rtr.list_documents()
                db_docs = [d["document_id"] if isinstance(d, dict) else str(d) for d in raw_docs]

            if db_docs:
                st.caption("Indexed documents available in SQLite database:")
                selected_doc = st.selectbox("Select document from database:", options=db_docs)
                if st.button("📂 Load Selected Document"):
                    st.session_state["rag_doc_id"] = selected_doc
                    st.success(f"Active document set to '{selected_doc}'!")
            else:
                st.info("No documents indexed in database yet. Upload and analyze a PDF in '📄 PDF Analyzer' first.")

        top_k_override = st.slider("Top K Results", min_value=1, max_value=10, value=3)

    with col_search:
        st.subheader("Semantic Search")
        user_query = st.text_input(
            "Ask a question about the document:",
            placeholder="e.g. How does the attention mechanism work? or What are the BLEU scores?",
        )

        search_btn = st.button("🔎 Search Document", type="primary", disabled=not user_query.strip())

        if search_btn and user_query.strip():
            curr_doc = st.session_state.get("rag_doc_id")
            if not curr_doc:
                st.warning("Please index a document first before querying.")
            else:
                with st.spinner("Analyzing query and searching vector index..."):
                    q_res = None
                    try:
                        resp = requests.post(
                            f"{API_URL}/api/query",
                            json={"document_id": curr_doc, "query": user_query.strip(), "top_k": top_k_override},
                            timeout=15,
                        )
                        if resp.status_code == 200:
                            q_res = resp.json()
                    except Exception:
                        pass

                    if not q_res:
                        rtr = get_local_retriever()
                        q_res = rtr.retrieve(user_query.strip(), document_id=curr_doc, top_k=top_k_override)

                    if q_res and "retrieved_chunks" in q_res:
                        qa = q_res.get("query_analysis", {})
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Intent", str(qa.get("intent", "factual")).capitalize())
                        c2.metric("Complexity", str(qa.get("complexity", "moderate")).capitalize())
                        c3.metric("Passages Retrieved", len(q_res["retrieved_chunks"]))

                        if qa.get("keywords"):
                            st.markdown(f"**Keywords:** " + " · ".join([f"`{k}`" for k in qa["keywords"]]))

                        st.markdown("---")
                        st.subheader("Retrieved Passages (Ranked by Cosine Similarity)")

                        for idx, chunk in enumerate(q_res["retrieved_chunks"], 1):
                            score = chunk["relevance_score"]
                            score_pct = f"{score * 100:.1f}%"
                            color = "#22c55e" if score >= 0.6 else ("#eab308" if score >= 0.3 else "#64748b")
                            sec = chunk.get("metadata", {}).get("section", "General")
                            tokens = chunk.get("metadata", {}).get("tokens", 0)

                            st.markdown(f"""
                            <div class="chunk-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <strong>§ {sec}</strong>
                                    <div>
                                        <span style="color: #64748b; font-size: 0.8rem; margin-right: 8px;">{tokens} tokens</span>
                                        <span class="pill" style="background-color: {color}22; color: {color}; border: 1px solid {color}55;">{score_pct} match</span>
                                    </div>
                                </div>
                                <div style="color: #334155; font-size: 0.92rem; line-height: 1.5;">{chunk['text']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(q_res.get("error", "No results found."))

# -----------------------------------------------------------------------------
# Tool 4: Context Optimizer (Member 3)
# -----------------------------------------------------------------------------
elif app_mode == "🎯 Context Optimizer (Member 3)":
    st.markdown('<div class="main-header">Intent-Aware <span style="color: #6d7d17;">Context Optimization</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Member 3 Engine: Adaptive token budgeting, redundancy removal, and structured context reconstruction.</div>', unsafe_allow_html=True)

    active_doc_id = st.session_state.get("rag_doc_id")
    pdf_tokens = st.session_state.get("pdf_result", {}).get("original_tokens", 0)

    c_opt1, c_opt2 = st.columns([1, 1.8], gap="large")

    with c_opt1:
        st.subheader("Optimization Parameters")
        target_doc = st.text_input("Indexed Document ID", value=active_doc_id or "sample_transformer")
        user_prompt = st.text_area("User Query / Task Prompt", value="Explain the Transformer architecture and attention mechanism in detail.")
        custom_budget = st.number_input("Custom Token Budget Cap (0 = Auto Adaptive)", min_value=0, max_value=10000, value=0, step=100)

        opt_btn = st.button("⚡ Run Context Optimization", type="primary", use_container_width=True, disabled=not user_prompt.strip())

    with c_opt2:
        if opt_btn and user_prompt.strip():
            with st.spinner("Executing Member 3 Optimization Pipeline..."):
                opt_data = None
                try:
                    payload = {
                        "document_id": target_doc.strip(),
                        "query": user_prompt.strip(),
                        "original_doc_tokens": pdf_tokens,
                        "max_budget": custom_budget if custom_budget > 0 else None,
                    }
                    res = requests.post(f"{API_URL}/api/optimize-context", json=payload, timeout=20)
                    if res.status_code == 200:
                        opt_data = res.json()
                except Exception:
                    pass

                if not opt_data:
                    try:
                        rtr = get_local_retriever()
                        if rtr.has_document(target_doc.strip()):
                            rag_res = rtr.retrieve(user_prompt.strip(), document_id=target_doc.strip(), top_k=15)
                            from backend.token_optimizer import token_optimizer
                            opt_data = token_optimizer.optimize_context(
                                retrieved_chunks=rag_res.get("retrieved_chunks", []),
                                query_analysis=rag_res.get("query_analysis", {}),
                                total_analyzed_chunks=len(rag_res.get("retrieved_chunks", [])),
                                original_doc_tokens=pdf_tokens or 2000,
                                max_budget_override=custom_budget if custom_budget > 0 else None,
                                query=user_prompt.strip(),
                            )
                    except Exception as ex:
                        st.error(f"Optimization error: {ex}")

                if opt_data:
                    rep = opt_data.get("report", {})
                    st.subheader("📊 Context Efficiency Report")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Original Tokens", rep.get("original_document_tokens", 0))
                    m2.metric("Optimized Context", rep.get("optimized_context_tokens", 0))
                    m3.metric("Token Reduction", f"{rep.get('token_reduction_percent', 0.0)}%")
                    m4.metric("Tokens Saved", rep.get("tokens_saved", 0))

                    st.markdown("---")
                    col_info1, col_info2, col_info3 = st.columns(3)
                    col_info1.write(f"**Query Intent:** `{str(rep.get('query_intent', 'N/A')).capitalize()}`")
                    col_info2.write(f"**Query Complexity:** `{str(rep.get('query_complexity', 'N/A')).capitalize()}`")
                    col_info3.write(f"**Redundancies Purged:** `{rep.get('redundancy_purged_count', 0)}` chunks")

                    st.markdown("---")
                    st.subheader("📝 Reconstructed Markdown Context")
                    markdown_content = opt_data.get("reconstructed_markdown", "")
                    st.code(markdown_content, language="markdown")

# -----------------------------------------------------------------------------
# Tool 5: Backend Health
# -----------------------------------------------------------------------------
elif app_mode == "🩺 Backend Health":
    st.subheader("Backend Diagnostics")
    st.write(f"Connecting to FastAPI at `{API_URL}`...")
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=5)
        if r.status_code == 200:
            st.success(f"Backend is running healthy: {r.json()}")
        else:
            st.error(f"Backend returned HTTP {r.status_code}")
    except Exception as e:
        st.warning(f"Backend unreachable at {API_URL}: {e}")
        st.info("Start the FastAPI backend with: `python backend/main.py`")
