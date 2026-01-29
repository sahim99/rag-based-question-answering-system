"""
RAG Document Question Answering - Professional Demo UI
Polished design with custom styling for a professional appearance.
"""

import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/api"

# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Subheader styling */
    h3 {
        color: #e0e0e0;
        font-weight: 500;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #1e1e2e;
        border: 1px solid #3d3d5c;
        border-radius: 8px;
        color: #ffffff;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button[data-baseweb="button"] {
        background-color: #6366f1;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #5558e3;
        transform: translateY(-1px);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: #1e1e2e;
        border: 2px dashed #3d3d5c;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Info box styling */
    .stAlert {
        background-color: #1e1e2e;
        border: 1px solid #3d3d5c;
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e1e2e;
        border-radius: 8px;
    }
    
    /* Sidebar styling - distinct from main */
    section[data-testid="stSidebar"] {
        background-color: #0a0a12;
        border-right: 1px solid #2d2d4a;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #8b8b9b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    
    /* Divider styling */
    hr {
        border-color: #2d2d3d;
        margin: 1.5rem 0;
    }
    
    /* Answer container */
    .answer-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Caption styling */
    .stCaption {
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "query_processing" not in st.session_state:
    st.session_state.query_processing = False
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# =============================================================================
# SIDEBAR - System Info
# =============================================================================
with st.sidebar:
    st.markdown("### SYSTEM")
    
    st.markdown("**Backend**")
    st.caption("FastAPI · Python 3.11+")
    
    st.markdown("**Vector Store**")
    st.caption("FAISS (CPU)")
    
    st.markdown("**Embeddings**")
    st.caption("Jina AI v3 (1024 dim)")
    
    st.markdown("**LLM**")
    st.caption("Groq Llama 3.3 70B")
    
    st.divider()
    
    st.markdown("### PIPELINE")
    st.caption("""
    1. Parse & chunk document
    2. Generate embeddings
    3. Store in FAISS index
    4. Semantic search
    5. LLM generation
    """)
    
    st.divider()
    
    st.markdown("### API")
    st.code("POST /api/upload\nPOST /api/query", language=None)

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Centered container
_, center, _ = st.columns([0.5, 4, 0.5])

with center:
    # Header
    st.title("RAG Document Q&A")
    st.caption("Upload documents and ask questions. Answers are grounded strictly in the document content.")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Upload Section
    # -------------------------------------------------------------------------
    st.subheader("Upload Document")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Drag and drop or browse",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            help="Supported: PDF, TXT"
        )
    
    with col2:
        upload_disabled = uploaded_file is None
        if st.button("Upload", use_container_width=True, type="primary", disabled=upload_disabled):
            with st.spinner("Processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                    if response.status_code == 200:
                        st.success(f"Uploaded: {uploaded_file.name}")
                    else:
                        st.error("Upload failed")
                except requests.exceptions.ConnectionError:
                    st.error("Backend unavailable")
                except Exception as e:
                    st.error(str(e))
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Question Section
    # -------------------------------------------------------------------------
    st.subheader("Ask a Question")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Question",
            placeholder="What is the main topic of the document?",
            label_visibility="collapsed",
            disabled=st.session_state.query_processing
        )
    
    with col2:
        ask_disabled = not query or st.session_state.query_processing
        ask_clicked = st.button("Ask", use_container_width=True, type="primary", disabled=ask_disabled)
    
    # Handle query
    if ask_clicked and query:
        st.session_state.query_processing = True
        with st.spinner("Generating answer..."):
            try:
                response = requests.post(f"{API_URL}/query", json={"question": query}, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.last_answer = data.get("answer", "No answer.")
                    st.session_state.last_sources = data.get("sources", [])
                else:
                    st.session_state.last_answer = "Error occurred."
                    st.session_state.last_sources = []
            except requests.exceptions.ConnectionError:
                st.session_state.last_answer = "Cannot connect to backend."
                st.session_state.last_sources = []
            except Exception as e:
                st.session_state.last_answer = str(e)
                st.session_state.last_sources = []
        st.session_state.query_processing = False
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Answer Section
    # -------------------------------------------------------------------------
    st.subheader("Answer")
    
    if st.session_state.last_answer:
        st.markdown(f'<div class="answer-container">{st.session_state.last_answer}</div>', unsafe_allow_html=True)
    else:
        st.info("Your answer will appear here after asking a question.")
    
    # -------------------------------------------------------------------------
    # Sources Section
    # -------------------------------------------------------------------------
    if st.session_state.last_sources:
        with st.expander(f"View Sources ({len(st.session_state.last_sources)})"):
            for i, src in enumerate(st.session_state.last_sources, 1):
                st.caption(f"{i}. {src.get('source_file', 'N/A')} | {src.get('chunk_id', 'N/A')}")
    
    # Footer
    st.divider()
    st.caption("Powered by FastAPI · FAISS · Jina AI · Groq LLM")
