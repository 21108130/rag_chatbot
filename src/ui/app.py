import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config.settings import settings
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.logger import logger
from src.utils.models import ChatRequest, DocumentStatus


st.set_page_config(
    page_title = settings.app_name,
    page_icon  = "🤖",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #00B8A9, #065A82);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-header { color: #718096; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .chat-user { background:#EBF8FF; border-left:4px solid #3182CE;
                 padding:0.75rem 1rem; border-radius:0.5rem; margin:0.5rem 0; }
    .chat-assistant { background:#F0FFF4; border-left:4px solid #38A169;
                      padding:0.75rem 1rem; border-radius:0.5rem; margin:0.5rem 0; }
    .source-box { background:#FFFFF0; border:1px solid #ECC94B;
                  border-radius:0.35rem; padding:0.5rem 0.75rem;
                  font-size:0.8rem; margin-top:0.25rem; }
    .stat-card { background:#EDF2F7; border-radius:0.5rem;
                 padding:0.75rem 1rem; text-align:center; }
    .stat-value { font-size:1.6rem; font-weight:700; color:#2D3748; }
    .stat-label { font-size:0.8rem; color:#718096; }
    .api-key-box { background:#EBF8FF; border:1px solid #90CDF4;
                   border-radius:0.5rem; padding:0.75rem; margin-bottom:0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = []
    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = False


def get_pipeline() -> RAGPipeline:
    return st.session_state.pipeline


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/chatbot.png",
            width=60,
        )
        st.title("RAG Chatbot")
        st.caption(f"v{settings.app_version} · llama-3.1-8b-instant")
        st.divider()

        # ── API Key Section ───────────────────────────────────────────────────
        st.subheader("🔑 Groq API Key")

        # Check if key already exists in environment
        existing_key = os.environ.get("GROQ_API_KEY", "")
        if existing_key and st.session_state.api_key_set:
            st.success("✅ API Key is set")
            if st.button("🔄 Change Key", use_container_width=True):
                st.session_state.api_key_set = False
                os.environ["GROQ_API_KEY"] = ""
                # Reset pipeline so it picks up new key
                st.session_state.pipeline = RAGPipeline()
                st.rerun()
        else:
            api_key = st.text_input(
                "Enter your Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Your key is used only in this session and never stored.",
            )

            if st.button("✅ Set API Key", type="primary", use_container_width=True):
                if api_key and api_key.startswith("gsk_") and len(api_key) > 20:
                    os.environ["GROQ_API_KEY"] = api_key
                    st.session_state.api_key_set = True
                    # Reset pipeline so it picks up the new key
                    st.session_state.pipeline = RAGPipeline()
                    st.success("✅ API Key set successfully!")
                    st.rerun()
                elif api_key:
                    st.error("❌ Invalid key format. Groq keys start with 'gsk_'")
                else:
                    st.warning("⚠️ Please enter your API key first.")

            st.markdown(
                """
                <div class="api-key-box">
                🆓 Groq is <strong>free</strong> to use.<br>
                <a href="https://console.groq.com/keys" target="_blank">
                👉 Get your free API key here
                </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Document Upload (only show if key is set) ─────────────────────────
        if st.session_state.api_key_set or os.environ.get("GROQ_API_KEY"):
            st.subheader("📄 Upload Documents")
            uploaded_files = st.file_uploader(
                label="Choose files",
                type=settings.allowed_extensions,
                accept_multiple_files=True,
                help=(
                    f"Supported: {', '.join(settings.allowed_extensions).upper()}\n"
                    f"Max size: {settings.max_file_size_mb} MB per file"
                ),
            )

            if uploaded_files:
                if st.button("🚀 Index Documents", type="primary", use_container_width=True):
                    _handle_upload(uploaded_files)

            # Indexed docs list
            if st.session_state.indexed_docs:
                st.divider()
                st.subheader("📚 Knowledge Base")
                for doc in st.session_state.indexed_docs:
                    icon = "✅" if doc.status == DocumentStatus.INDEXED else "❌"
                    st.markdown(
                        f"{icon} **{doc.filename}** "
                        f"<small>({doc.chunk_count} chunks)</small>",
                        unsafe_allow_html=True,
                    )

            # Stats
            st.divider()
            st.subheader("📊 Statistics")
            _render_stats()

            # Controls
            st.divider()
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_id = None
                st.rerun()

            if st.button("⚙️ Settings", use_container_width=True):
                _render_settings_expander()

        else:
            st.info("👆 Set your API key above to get started.")


def _handle_upload(uploaded_files) -> None:
    pipeline = get_pipeline()
    upload_path = settings.upload_path

    with st.spinner("Indexing documents …"):
        for uf in uploaded_files:
            if uf.size > settings.max_file_size_bytes:
                st.sidebar.error(
                    f"❌ '{uf.name}' exceeds the {settings.max_file_size_mb} MB limit."
                )
                continue

            dest = upload_path / uf.name
            dest.write_bytes(uf.getbuffer())

            result = pipeline.ingest_document(dest)
            st.session_state.indexed_docs.append(result)

            if result.status == DocumentStatus.INDEXED:
                st.sidebar.success(f"✅ **{uf.name}** indexed ({result.chunk_count} chunks)")
                logger.info(f"[UI] Indexed: {uf.name}")
            else:
                st.sidebar.error(f"❌ **{uf.name}**: {result.message}")
                logger.error(f"[UI] Failed to index: {uf.name}")


def _render_stats() -> None:
    pipeline = get_pipeline()
    try:
        stats = pipeline.get_kb_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{stats["total_docs"]}</div>'
                f'<div class="stat-label">Documents</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{stats["total_chunks"]}</div>'
                f'<div class="stat-label">Chunks</div></div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("Stats unavailable")


def _render_settings_expander() -> None:
    with st.expander("⚙️ Settings", expanded=True):
        st.markdown(f"**Model:** `llama-3.1-8b-instant`")
        st.markdown(f"**Embedding:** `{settings.embedding_model}`")
        st.markdown(f"**Top-K:** `{settings.top_k_results}`")
        st.markdown(f"**Chunk size:** `{settings.chunk_size}`")
        st.markdown(f"**Chunk overlap:** `{settings.chunk_overlap}`")


# ── Main chat area ─────────────────────────────────────────────────────────────

def render_main() -> None:
    st.markdown(
        '<div class="main-header">🤖 RAG Intelligent Chatbot</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">'
        "Upload your documents in the sidebar, then ask questions below."
        "</div>",
        unsafe_allow_html=True,
    )

    # Show blocked state if no API key
    if not st.session_state.api_key_set and not os.environ.get("GROQ_API_KEY"):
        st.warning(
            "👈 **Please enter your Groq API key in the sidebar to get started.**\n\n"
            "Groq is free — [get your key here](https://console.groq.com/keys).",
            icon="🔑",
        )
        return

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info(
                "👈 Upload a document in the sidebar to get started, "
                "then type your question below.",
                icon="💡",
            )

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 <strong>You:</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-assistant">🤖 <strong>Assistant:</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} source(s) used", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            fname   = src.metadata.get("source", src.doc_id)
                            score   = src.similarity_score
                            preview = src.content[:300].replace("\n", " ")
                            st.markdown(
                                f'<div class="source-box">'
                                f"<strong>Source {i}:</strong> {fname} "
                                f"(similarity: {score:.0%})<br>"
                                f"<em>{preview} …</em></div>",
                                unsafe_allow_html=True,
                            )
                if msg.get("latency_ms"):
                    st.caption(f"⏱ {msg['latency_ms']:.0f} ms")

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                label="Ask a question …",
                placeholder="e.g. What are the main findings in the document?",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send 💬", type="primary")

    if submitted and user_input.strip():
        _handle_chat(user_input.strip())
        st.rerun()


def _handle_chat(query: str) -> None:
    pipeline = get_pipeline()

    if pipeline.get_kb_stats()["total_chunks"] == 0:
        st.warning("⚠️ No documents indexed yet. Please upload a document first.")
        return

    with st.spinner("Thinking …"):
        request = ChatRequest(
            query=query,
            conversation_id=st.session_state.conversation_id,
        )
        response = pipeline.chat(request)

    st.session_state.conversation_id = response.conversation_id
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append(
        {
            "role":       "assistant",
            "content":    response.answer,
            "sources":    response.sources,
            "latency_ms": response.latency_ms,
        }
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()