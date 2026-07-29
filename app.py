import os
import io
import re

import streamlit as st
import pandas as pd
import numpy as np
import requests

from groq import Groq
from fastembed import TextEmbedding
from pypdf import PdfReader

st.set_page_config(page_title="RAG Chat AI", page_icon="💬", layout="centered")
st.title("💬 RAG Chat AI")
st.caption("Upload PDFs, Word, Excel, CSV, TXT, or connect Google Sheets → Chat with everything!")

# ========== CONFIG ==========
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3
RELEVANCE_THRESHOLD = 0.55

SUPABASE_URL = "https://hhwjxievppujzlnyqykp.supabase.co"
DB_TABLE_NAME = "feeder_billing_consumption"
DB_PAGE_SIZE = 1000


def get_secret(name):
    """Read a secret from st.secrets, falling back to environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


# ========== API SETUP ==========
groq_key = get_secret("GROQ_API_KEY")
groq_client = None

if groq_key:
    try:
        groq_client = Groq(api_key=groq_key)
    except Exception as e:
        st.sidebar.error(f"Groq Error: {e}")

supabase_anon_key = get_secret("SUPABASE_ANON_KEY")


# ========== LOCAL EMBEDDING MODEL ==========
@st.cache_resource
def load_embedder():
    with st.spinner("Loading embedding model (22MB, one-time)..."):
        return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


embedder = None
try:
    embedder = load_embedder()
except Exception as e:
    st.sidebar.error(f"Embedder Error: {e}")


# ===== COLUMN KNOWLEDGE — edit this to teach the AI your data =====
TABLE_KNOWLEDGE = """
TABLE: feeder_billing_consumption — DISCOM feeder-level billing and consumption data

Column meanings:
- region, circle, division, zone: Administrative hierarchy of MP East DISCOM
- substation, substation_code, feeder, feeder_code: Physical infrastructure identifiers
- meter_serial_number: Unique meter ID on the feeder
- mf: Meter multiplication factor
- category: Feeder category (e.g., "In" = incoming feeder)
- bill_month, billi_year: Billing cycle month and year
- initial_reading_kwh, final_reading_kwh: Meter readings at start/end of billing cycle
- pf: Power factor (0 to 1); low PF indicates inefficient load
- kwh: Active energy consumed (kilowatt-hours) in the billing cycle
- kwh_exp: Exported energy (kWh) — relevant for feeders with reverse power flow
- kvah: Apparent energy (kilovolt-ampere-hours)
- md_kw, md_kva: Maximum demand recorded during the cycle (kW and kVA)
- kvarh_q1 to q4: Reactive energy by quadrant — used for power factor penalty calculations
- pwr_on_dur: Duration the feeder was powered on during the cycle

Use this glossary to correctly interpret any question about this table's data.
"""


# ========== SUPABASE ==========
def _supabase_headers(extra=None):
    headers = {
        "apikey": supabase_anon_key,
        "Authorization": f"Bearer {supabase_anon_key}",
    }
    if extra:
        headers.update(extra)
    return headers


def fetch_all_rows(select="*", page_size=DB_PAGE_SIZE):
    """Fetch every row from the table, paging past PostgREST's default row cap."""
    url = f"{SUPABASE_URL}/rest/v1/{DB_TABLE_NAME}?select={select}"
    rows = []
    offset = 0
    while True:
        headers = _supabase_headers({
            "Range-Unit": "items",
            "Range": f"{offset}-{offset + page_size - 1}",
        })
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        batch = response.json()
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        offset += page_size


def load_db_table_as_text():
    """Fetch the feeder billing table from Supabase and return it as text for chunking."""
    rows = fetch_all_rows()
    df = pd.DataFrame(rows)
    if df.empty:
        return "", 0
    return df.to_string(index=False), len(df)


def get_distinct_values(column_name):
    """Fetch all distinct values for a given column directly from Supabase."""
    rows = fetch_all_rows(select=column_name)
    return sorted({row[column_name] for row in rows if row.get(column_name)})


# ========== SESSION STATE ==========
for key, default in [
    ("messages", []),
    ("embeddings", None),
    ("chunks", []),
    ("file_stats", []),
    ("ready", False),
    ("db_loaded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ========== CHUNKING + EMBEDDING HELPERS ==========
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks


def embed_chunks(chunks, show_progress=False):
    """Embed chunks, returning (kept_chunks, embeddings) so the two stay aligned."""
    kept, vectors = [], []
    progress = st.progress(0) if show_progress else None
    for i, chunk in enumerate(chunks):
        try:
            vector = np.array(list(embedder.embed([chunk[:8000]]))[0], dtype=np.float32)
            kept.append(chunk)
            vectors.append(vector)
        except Exception as e:
            st.warning(f"Embed error on chunk {i}: {e}")
        if progress:
            progress.progress((i + 1) / len(chunks))
    if progress:
        progress.empty()
    return kept, vectors


def add_to_index(chunks, vectors, stat):
    """Append new chunks and their vectors to the existing index."""
    if not vectors:
        return False
    matrix = np.array(vectors, dtype=np.float32)
    if st.session_state.embeddings is None:
        st.session_state.embeddings = matrix
        st.session_state.chunks = list(chunks)
    else:
        st.session_state.embeddings = np.vstack([st.session_state.embeddings, matrix])
        st.session_state.chunks.extend(chunks)
    st.session_state.file_stats.append(stat)
    st.session_state.ready = True
    return True


# ========== TEXT EXTRACTION FUNCTIONS ==========
def extract_pdf_text(file):
    reader = PdfReader(file)
    parts = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text()
        if txt and txt.strip():
            parts.append(f"[Page {i+1}]\n{txt}")
    return "\n\n".join(parts), len(reader.pages)


def extract_word_text(file):
    try:
        from docx import Document
        doc = Document(file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        table_texts = []
        for table in doc.tables:
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            table_texts.append("\n".join(rows))
        all_text = "\n\n".join(paragraphs)
        if table_texts:
            all_text += "\n\n[Tables]\n" + "\n\n".join(table_texts)
        return all_text, 1
    except Exception as e:
        st.error(f"Word read error: {e}")
        return "", 0


def extract_excel_text(file):
    try:
        xls = pd.ExcelFile(file)
        parts = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            parts.append(f"[Sheet: {sheet_name}]\n{df.to_string(index=False)}")
        return "\n\n".join(parts), len(xls.sheet_names)
    except Exception as e:
        st.error(f"Excel read error: {e}")
        return "", 0


def extract_csv_text(file):
    try:
        df = pd.read_csv(file)
        return df.to_string(index=False), 1
    except Exception as e:
        st.error(f"CSV read error: {e}")
        return "", 0


def extract_txt_text(file):
    try:
        return file.read().decode("utf-8"), 1
    except Exception as e:
        st.error(f"TXT read error: {e}")
        return "", 0


def extract_google_sheet(sheet_url):
    """Extract text from a publicly shared Google Sheet URL."""
    try:
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not match:
            st.error("❌ Invalid Google Sheet URL. Expected: https://docs.google.com/spreadsheets/d/XXXX/edit")
            return None, 0

        sheet_id = match.group(1)
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

        resp = requests.get(export_url, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        return df.to_string(index=False), 1
    except Exception as e:
        st.error(f"❌ Google Sheet error: {e}")
        st.info("💡 Make sure the sheet is shared as 'Anyone with the link can view'")
        return None, 0


def process_file(file):
    """Route file to the correct extractor."""
    fname = file.name.lower()
    if fname.endswith(".pdf"):
        return extract_pdf_text(file)
    if fname.endswith(".docx"):
        return extract_word_text(file)
    if fname.endswith((".xlsx", ".xls")):
        return extract_excel_text(file)
    if fname.endswith(".csv"):
        return extract_csv_text(file)
    if fname.endswith(".txt"):
        return extract_txt_text(file)
    return "", 0


# ========== AUTO-LOAD LIVE DATABASE TABLE ON STARTUP ==========
def load_database_into_index():
    db_text, row_count = load_db_table_as_text()
    if not db_text:
        st.sidebar.warning("⚠️ Database table returned no rows.")
        return
    chunks = chunk_text(db_text)
    kept, vectors = embed_chunks(chunks)
    add_to_index(kept, vectors, {
        "name": f"{DB_TABLE_NAME} (live DB)",
        "pages": row_count,
        "chunks": len(kept),
    })


if not st.session_state.db_loaded and embedder and supabase_anon_key:
    with st.spinner("Connecting to live DISCOM database..."):
        try:
            load_database_into_index()
        except Exception as e:
            st.sidebar.error(f"DB connection error: {e}")
        finally:
            st.session_state.db_loaded = True  # don't retry on every rerun


# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🔌 Status")
    if groq_client:
        st.success("✅ Groq Connected")
    else:
        st.error("❌ No Groq API Key")
        st.markdown("Get a free key at [console.groq.com](https://console.groq.com)")

    if embedder:
        st.success("✅ Local Embedder Ready")

    if supabase_anon_key:
        st.success("✅ Live DB Connected")
        if st.button("🔄 Reload Live DB", use_container_width=True):
            with st.spinner("Reloading database..."):
                try:
                    st.session_state.embeddings = None
                    st.session_state.chunks = []
                    st.session_state.file_stats = []
                    st.session_state.ready = False
                    load_database_into_index()
                    st.success("✅ Database reloaded")
                except Exception as e:
                    st.error(f"DB reload error: {e}")
    else:
        st.warning("⚠️ Supabase key not set")

    st.divider()

    # MODE TOGGLE
    st.subheader("🎛️ Answer Mode")
    answer_mode = st.radio(
        "Choose how the AI answers:",
        ["🧠 Hybrid (Docs + General Knowledge)", "📄 Documents Only"],
        index=0,
        help="Hybrid = uses docs when relevant, general knowledge otherwise. Documents Only = strictly from sources.",
    )
    hybrid_mode = answer_mode.startswith("🧠")

    st.divider()

    # GOOGLE SHEETS
    st.subheader("🔗 Google Sheets")
    sheet_url = st.text_input(
        "Paste Google Sheet URL (must be 'Anyone with link' public)",
        placeholder="https://docs.google.com/spreadsheets/d/XXXX/edit",
    )
    if st.button("📥 Fetch Google Sheet", use_container_width=True):
        if not sheet_url:
            st.warning("Paste a sheet URL first.")
        elif not embedder:
            st.error("Embedder not ready.")
        else:
            with st.spinner("Fetching Google Sheet..."):
                text, pages = extract_google_sheet(sheet_url)
                if text:
                    kept, vectors = embed_chunks(chunk_text(text), show_progress=True)
                    if add_to_index(kept, vectors, {
                        "name": "Google_Sheet",
                        "pages": pages,
                        "chunks": len(kept),
                    }):
                        st.success(
                            f"✅ Sheet added: {len(kept)} new chunks. "
                            f"Total: {len(st.session_state.chunks)}"
                        )

    st.divider()

    # FILE UPLOAD
    st.subheader("📄 Upload Files")
    uploaded_files = st.file_uploader(
        "Drop files here (PDF, Word, Excel, CSV, TXT)",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            added = 0
            for f in uploaded_files:
                text, pages = process_file(f)
                if not text:
                    st.warning(f"⚠️ Could not extract text from {f.name}")
                    continue
                kept, vectors = embed_chunks(chunk_text(text), show_progress=True)
                if add_to_index(kept, vectors, {
                    "name": f.name,
                    "pages": pages,
                    "chunks": len(kept),
                }):
                    added += len(kept)
            if added:
                st.success(f"✅ Added {added} chunks. Total: {len(st.session_state.chunks)}")
            else:
                st.error("❌ Nothing was indexed.")

    if st.session_state.file_stats:
        st.divider()
        st.subheader("📊 Sources")
        for s in st.session_state.file_stats:
            st.write(f"📄 {s['name']}")
            st.caption(f"{s['pages']} pages/rows/sheets → {s['chunks']} chunks")

    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown(
        """
        ### ⏱️ Free Tier
        - **Groq**: 30 req/min
        - **Embeddings**: Unlimited (local)
        """
    )


# ========== SETUP CHECK ==========
if not groq_client or not embedder:
    st.warning("⚠️ Setup Required")
    st.markdown(
        """
        ### Step 1: Get a Groq API Key (Free, No Credit Card)
        1. Go to https://console.groq.com
        2. Sign up → API Keys → Create Key

        ### Step 2: Add to Streamlit Cloud Secrets
        ```
        GROQ_API_KEY = "your-key"
        SUPABASE_ANON_KEY = "your-key"
        ```

        ### Step 3: Upload files or connect a Google Sheet in the sidebar
        """
    )
    st.stop()

if not st.session_state.ready:
    st.info("👈 **Upload files or connect a Google Sheet in the sidebar to start chatting.**")
    st.markdown(
        """
        ### 💡 Supported formats:
        - **PDF** — Research papers, notes, reports
        - **Word (.docx)** — Documents with tables
        - **Excel (.xlsx/.xls)** — Multi-sheet spreadsheets
        - **CSV** — Data exports
        - **TXT** — Plain text files
        - **Google Sheets** — Live data (share as 'Anyone with link')
        """
    )
    st.stop()


# ========== RETRIEVAL ==========
def retrieve(query, top_k=TOP_K):
    q_vec = np.array(list(embedder.embed([query[:8000]]))[0], dtype=np.float32)
    matrix = st.session_state.embeddings
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec)
    denom[denom == 0] = 1e-9
    sims = (matrix @ q_vec) / denom
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return top_idx, sims


def build_prompt(prompt, context, history, hybrid, relevant):
    if hybrid and not relevant:
        return f"""You are a helpful assistant. The user asked a question that does not appear related to their indexed sources.
Answer using your general knowledge. Be helpful and accurate.

=== RECENT CONVERSATION ===
{history}

=== USER QUESTION ===
{prompt}

=== YOUR ANSWER ===
Provide a clear, accurate, and helpful answer."""

    if hybrid:
        instruction = (
            "Use the context below to answer if it helps. If the context does not fully "
            "answer the question, supplement with your general knowledge."
        )
        closing = "Provide a clear, accurate, and helpful answer. When using the context, be precise."
    else:
        instruction = (
            "Answer using ONLY the information in the context below. If the answer is not "
            'found there, say: "I don\'t have enough information in the indexed sources to answer this."'
        )
        closing = "Provide a clear, accurate, and concise answer."

    return f"""You are a helpful assistant working with DISCOM data and documents. {instruction}

=== TABLE/COLUMN KNOWLEDGE (use this to interpret any database data correctly) ===
{TABLE_KNOWLEDGE}

=== CONTEXT FROM SOURCES ===
{context}

=== RECENT CONVERSATION ===
{history}

=== USER QUESTION ===
{prompt}

=== YOUR ANSWER ===
{closing}"""


# ========== CHAT INTERFACE ==========
st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("source_type") == "document":
                st.caption("📄 Answered from indexed sources")
            elif message.get("source_type") == "general":
                st.caption("🧠 Answered from general knowledge")
            if message.get("sources"):
                with st.expander("📄 View source chunks"):
                    for src in message["sources"]:
                        st.markdown(f"**Chunk** (score: {src['score']:.3f})")
                        st.text(src["text"][:600])
                        st.divider()

if prompt := st.chat_input("Ask anything about your documents... or anything else!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                top_idx, sims = retrieve(prompt)
                best_score = float(sims[top_idx[0]]) if len(top_idx) else 0.0
                is_relevant = best_score > RELEVANCE_THRESHOLD

                context = "\n\n---\n\n".join(st.session_state.chunks[i] for i in top_idx)

                recent_history = ""
                if len(st.session_state.messages) > 2:
                    recent = st.session_state.messages[-6:-1]
                    recent_history = "\n\n".join(
                        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                        for m in recent
                    )

                system_prompt = build_prompt(
                    prompt, context, recent_history, hybrid_mode, is_relevant
                )
                source_type = "document" if is_relevant else "general"

                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": system_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1024,
                )
                answer = chat_completion.choices[0].message.content

                sources = []
                if is_relevant:
                    sources = [
                        {"text": st.session_state.chunks[i], "score": float(sims[i])}
                        for i in top_idx
                    ]

                st.markdown(answer)

                if sources:
                    st.caption("📄 Answered from indexed sources")
                    with st.expander("📄 View source chunks"):
                        for src in sources:
                            st.markdown(f"**Chunk** (score: {src['score']:.3f})")
                            st.text(src["text"][:600])
                            st.divider()
                else:
                    st.caption("🧠 Answered from general knowledge")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "source_type": source_type,
                })

            except Exception as e:
                if "429" in str(e):
                    st.error("⏳ Rate limit (30/min). Wait a few seconds and try again.")
                else:
                    st.error(f"Error: {e}")
