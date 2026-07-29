import streamlit as st
from groq import Groq
from fastembed import TextEmbedding
from pypdf import PdfReader
import pandas as pd
import numpy as np
import requests
import re

st.set_page_config(page_title="RAG Chat AI", page_icon="💬", layout="centered")
st.title("💬 RAG Chat AI")
st.caption("Upload PDFs, Word, Excel, CSV, TXT, or connect Google Sheets → Chat with everything!")

# ========== API SETUP ==========
import os

groq_key = None
groq_client = None

try:
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_key:
    groq_key = os.getenv("GROQ_API_KEY")

if groq_key:
    try:
        groq_client = Groq(api_key=groq_key)
    except Exception as e:
        st.sidebar.error(f"Groq Error: {e}")

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

# ========== NEW: SUPABASE PERMANENT DATABASE CONNECTION ==========
SUPABASE_URL = "https://hhwjxievppujzlnyqykp.supabase.co"

supabase_anon_key = None
try:
    if "SUPABASE_ANON_KEY" in st.secrets:
        supabase_anon_key = st.secrets["SUPABASE_ANON_KEY"]
except Exception:
    pass
if not supabase_anon_key:
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

DB_TABLE_NAME = "feeder_billing_consumption"

# ===== NEW: COLUMN KNOWLEDGE — edit this to teach the AI your data =====
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

def load_db_table_as_text():
   def load_db_table_as_text():
    """Fetch the feeder billing table from Supabase and return as text for chunking"""
    headers = {
        "apikey": supabase_anon_key,
        "Authorization": f"Bearer {supabase_anon_key}"
    }
    url = f"{SUPABASE_URL}/rest/v1/{DB_TABLE_NAME}?select=*"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    return df.to_string(index=False), len(df)
       def get_distinct_values(column_name):
    """Fetch all distinct values for a given column directly from Supabase"""
    headers = {
        "apikey": supabase_anon_key,
        "Authorization": f"Bearer {supabase_anon_key}"
    }
    url = f"{SUPABASE_URL}/rest/v1/{DB_TABLE_NAME}?select={column_name}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    values = sorted(set(row[column_name] for row in data if row.get(column_name)))
    return values
# ========== END NEW SUPABASE SECTION ==========

# ========== SESSION STATE ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "file_stats" not in st.session_state:
    st.session_state.file_stats = []
if "ready" not in st.session_state:
    st.session_state.ready = False
if "db_loaded" not in st.session_state:   # NEW
    st.session_state.db_loaded = False    # NEW

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
        # Also extract tables
        table_texts = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                rows.append(" | ".join([cell.text.strip() for cell in row.cells]))
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
        content = file.read().decode('utf-8')
        return content, 1
    except Exception as e:
        st.error(f"TXT read error: {e}")
        return "", 0

def extract_google_sheet(sheet_url):
    """Extract text from a publicly shared Google Sheet URL"""
    try:
        # Extract sheet ID from URL
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not match:
            st.error("❌ Invalid Google Sheet URL. Make sure it looks like: https://docs.google.com/spreadsheets/d/XXXX/edit")
            return None, 0
        
        sheet_id = match.group(1)
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        resp = requests.get(export_url, timeout=30)
        resp.raise_for_status()
        
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        text = df.to_string(index=False)
        return text, 1
    except Exception as e:
        st.error(f"❌ Google Sheet error: {e}")
        st.info("💡 Make sure the Google Sheet is shared as 'Anyone with the link can view'")
        return None, 0

def process_file(file):
    """Route file to correct extractor"""
    fname = file.name.lower()
    if fname.endswith('.pdf'):
        return extract_pdf_text(file)
    elif fname.endswith('.docx'):
        return extract_word_text(file)
    elif fname.endswith(('.xlsx', '.xls')):
        return extract_excel_text(file)
    elif fname.endswith('.csv'):
        return extract_csv_text(file)
    elif fname.endswith('.txt'):
        return extract_txt_text(file)
    else:
        return "", 0

# ========== NEW: AUTO-LOAD LIVE DATABASE TABLE ON STARTUP ==========
if not st.session_state.db_loaded and embedder and supabase_anon_key:
    with st.spinner("Connecting to live DISCOM database..."):
        try:
            db_text, row_count = load_db_table_as_text()

            chunks = []
            start = 0
            while start < len(db_text):
                end = min(start + 1000, len(db_text))
                chunk = db_text[start:end].strip()
                if len(chunk) > 50:
                    chunks.append(chunk)
                start = end - 150 if end < len(db_text) else end

            embeddings = []
            for chunk in chunks:
                emb_gen = embedder.embed([chunk[:8000]])
                emb = np.array(list(emb_gen)[0], dtype=np.float32)
                embeddings.append(emb)

            if embeddings:
                if st.session_state.embeddings is None:
                    st.session_state.embeddings = np.array(embeddings)
                    st.session_state.chunks = chunks
                else:
                    st.session_state.embeddings = np.vstack([st.session_state.embeddings, np.array(embeddings)])
                    st.session_state.chunks.extend(chunks)

                st.session_state.file_stats.append({
                    "name": f"{DB_TABLE_NAME} (live DB)",
                    "pages": row_count,
                    "chunks": len(chunks)
                })
                st.session_state.ready = True
            st.session_state.db_loaded = True
        except Exception as e:
            st.sidebar.error(f"DB connection error: {e}")
            st.session_state.db_loaded = True  # don't keep retrying every rerun
# ========== END NEW AUTO-LOAD SECTION ==========

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🔌 Status")
    if groq_client:
        st.success("✅ Groq Connected")
    else:
        st.error("❌ No Groq API Key")
        st.markdown("Get free key at [console.groq.com](https://console.groq.com)")
    
    if embedder:
        st.success("✅ Local Embedder Ready")

    if supabase_anon_key:                              # NEW
        st.success("✅ Live DB Connected")               # NEW
    else:                                                # NEW
        st.warning("⚠️ Supabase key not set")            # NEW
    
    st.divider()
    
    # MODE TOGGLE
    st.subheader("🎛️ Answer Mode")
    answer_mode = st.radio(
        "Choose how the AI answers:",
        ["🧠 Hybrid (Docs + General Knowledge)", "📄 Documents Only"],
        index=0,
        help="Hybrid = uses docs when relevant, general knowledge otherwise. Documents Only = strictly from uploaded sources."
    )
    hybrid_mode = (answer_mode == "🧠 Hybrid (Docs + General Knowledge)")
    
    st.divider()
    
    # GOOGLE SHEETS
    st.subheader("🔗 Google Sheets")
    sheet_url = st.text_input(
        "Paste Google Sheet URL (must be 'Anyone with link' public)",
        placeholder="https://docs.google.com/spreadsheets/d/XXXX/edit",
        help="Make sure the sheet is shared as 'Anyone with the link can view'"
    )
    if st.button("📥 Fetch Google Sheet", use_container_width=True):
        if sheet_url:
            with st.spinner("Fetching Google Sheet..."):
                text, pages = extract_google_sheet(sheet_url)
                if text:
                    # Process as a single document
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = min(start + 1000, len(text))
                        chunk = text[start:end].strip()
                        if len(chunk) > 50:
                            chunks.append(chunk)
                        start = end - 150 if end < len(text) else end
                    
                    if chunks:
                        # If this is the first doc, init; else append
                        if not st.session_state.ready:
                            st.session_state.chunks = []
                            st.session_state.file_stats = []
                        
                        st.session_state.chunks.extend(chunks)
                        st.session_state.file_stats.append({
                            "name": "Google_Sheet",
                            "pages": pages,
                            "chunks": len(chunks)
                        })
                        
                        # Re-embed everything
                        embeddings = []
                        emb_progress = st.progress(0)
                        for i, chunk in enumerate(st.session_state.chunks):
                            try:
                                emb_gen = embedder.embed([chunk[:8000]])
                                emb = np.array(list(emb_gen)[0], dtype=np.float32)
                                embeddings.append(emb)
                            except Exception as e:
                                st.error(f"Embed error: {e}")
                            emb_progress.progress((i + 1) / len(st.session_state.chunks))
                        
                        if embeddings:
                            st.session_state.embeddings = np.array(embeddings)
                            st.session_state.ready = True
                            st.success(f"✅ Google Sheet added! {len(chunks)} new chunks. Total: {len(st.session_state.chunks)}")
    
    st.divider()
    
    # FILE UPLOAD
    st.subheader("📄 Upload Files")
    uploaded_files = st.file_uploader(
        "Drop files here (PDF, Word, Excel, CSV, TXT)",
        type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'txt'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                all_chunks = []
                file_stats = []
                
                for f in uploaded_files:
                    text, pages = process_file(f)
                    if text:
                        chunks = []
                        start = 0
                        while start < len(text):
                            end = min(start + 1000, len(text))
                            chunk = text[start:end].strip()
                            if len(chunk) > 50:
                                chunks.append(chunk)
                            start = end - 150 if end < len(text) else end
                        
                        all_chunks.extend(chunks)
                        file_stats.append({
                            "name": f.name,
                            "pages": pages,
                            "chunks": len(chunks)
                        })
                    else:
                        st.warning(f"⚠️ Could not extract text from {f.name}")
                
                if not all_chunks:
                    st.error("No text extracted from any file.")
                else:
                    embeddings = []
                    emb_progress = st.progress(0)
                    
                    for i, chunk in enumerate(all_chunks):
                        try:
                            emb_gen = embedder.embed([chunk[:8000]])
                            emb = np.array(list(emb_gen)[0], dtype=np.float32)
                            embeddings.append(emb)
                        except Exception as e:
                            st.error(f"Embed error chunk {i}: {e}")
                        emb_progress.progress((i + 1) / len(all_chunks))
                    
                    if embeddings:
                        st.session_state.embeddings = np.array(embeddings)
                        st.session_state.chunks = all_chunks
                        st.session_state.file_stats = file_stats
                        st.session_state.ready = True
                        st.success(f"✅ {len(uploaded_files)} files → {len(all_chunks)} chunks")
                    else:
                        st.error("❌ Failed to create embeddings.")
    
    if st.session_state.file_stats:
        st.divider()
        st.subheader("📊 Sources")
        for s in st.session_state.file_stats:
            st.write(f"📄 {s['name']}")
            st.caption(f"{s['pages']} pages/sheets → {s['chunks']} chunks")
    
    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    st.markdown("""
    ### ⏱️ Free Tier
    - **Groq**: 30 req/min
    - **Embeddings**: Unlimited (local)
    """)

# ========== SETUP CHECK ==========
if not groq_client or not embedder:
    st.warning("⚠️ Setup Required")
    st.markdown("""
    ### Step 1: Get Groq API Key (Free, No Credit Card)
    1. Go to https://console.groq.com
    2. Sign up → API Keys → Create Key
    
    ### Step 2: Add to Streamlit Cloud Secrets
    `GROQ_API_KEY = "your-key"`
    
    ### Step 3: Upload files or connect Google Sheet in the sidebar
    """)
    st.stop()

if not st.session_state.ready:
    st.info("👈 **Upload files or connect a Google Sheet in the sidebar, then click 'Process Documents' to start chatting.**")
    st.markdown("""
    ### 💡 Supported formats:
    - **PDF** — Research papers, notes, reports
    - **Word (.docx)** — Documents with tables
    - **Excel (.xlsx/.xls)** — Multi-sheet spreadsheets
    - **CSV** — Data exports
    - **TXT** — Plain text files
    - **Google Sheets** — Live data (share as 'Anyone with link')
    """)
    st.stop()

# ========== CHAT INTERFACE ==========
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            if message.get("source_type") == "document":
                st.caption("📄 Answered from uploaded documents")
            elif message.get("source_type") == "general":
                st.caption("🧠 Answered from general knowledge")
            
            if "sources" in message and message["sources"]:
                with st.expander("📄 View source chunks"):
                    for src in message["sources"]:
                        st.markdown(f"**Chunk** (score: {src['score']:.3f})")
                        st.text(src["text"][:600])
                        st.divider()

# Chat input at the bottom
if prompt := st.chat_input("Ask anything about your documents... or anything else!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                q_emb_gen = embedder.embed([prompt[:8000]])
                q_vec = np.array(list(q_emb_gen)[0], dtype=np.float32)
                
                sims = []
                for emb in st.session_state.embeddings:
                    sim = np.dot(q_vec, emb) / (np.linalg.norm(q_vec) * np.linalg.norm(emb))
                    sims.append(sim)
                
                top_idx = np.argsort(sims)[-3:][::-1]
                top_scores = [sims[i] for i in top_idx]
                
                best_score = top_scores[0] if top_scores else 0
                is_relevant = best_score > 0.55
                
                relevant_chunks = [st.session_state.chunks[i] for i in top_idx]
                context = "\n\n---\n\n".join(relevant_chunks)
                
                recent_history = ""
                if len(st.session_state.messages) > 2:
                    recent = st.session_state.messages[-6:-1]
                    recent_history = "\n\n".join([
                        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                        for m in recent
                    ])
                
                if hybrid_mode:
                    if is_relevant:
                        system_prompt = f"""You are a helpful assistant. The user has uploaded documents that may contain relevant information. 
Use the document context below to answer if it helps. If the documents don't fully answer the question, supplement with your general knowledge.

=== TABLE/COLUMN KNOWLEDGE (use this to interpret any database data correctly) ===
{TABLE_KNOWLEDGE}

=== RELEVANT DOCUMENT CONTEXT ===
{context}

=== RECENT CONVERSATION ===
{recent_history}

=== USER QUESTION ===
{prompt}

=== YOUR ANSWER ===
Provide a clear, accurate, and helpful answer. When using information from the documents, be precise."""
                        source_type = "document"
                    else:
                        system_prompt = f"""You are a helpful assistant. The user asked a question that doesn't seem related to their uploaded documents. 
Answer using your general knowledge. Be helpful and accurate.

=== RECENT CONVERSATION ===
{recent_history}

=== USER QUESTION ===
{prompt}

=== YOUR ANSWER ===
Provide a clear, accurate, and helpful answer."""
                        source_type = "general"
                else:
                    system_prompt = f"""You are a helpful study assistant. Answer the user's question using ONLY the information provided in the context below.
If the answer is not found in the context, say: "I don't have enough information in the uploaded documents to answer this."

=== TABLE/COLUMN KNOWLEDGE (use this to interpret any database data correctly) ===
{TABLE_KNOWLEDGE}

=== CONTEXT FROM DOCUMENTS ===
{context}

=== RECENT CONVERSATION ===
{recent_history}

=== USER QUESTION ===
{prompt}

=== YOUR ANSWER ===
Provide a clear, accurate, and concise answer."""
                    source_type = "document" if is_relevant else "general"
                
                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": system_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1024
                )
                
                answer = chat_completion.choices[0].message.content
                
                sources = []
                if is_relevant:
                    for i, idx in enumerate(top_idx):
                        sources.append({
                            "text": st.session_state.chunks[idx],
                            "score": sims[idx]
                        })
                
                st.markdown(answer)
                
                if is_relevant and sources:
                    st.caption("📄 Answered from uploaded documents")
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
                    "sources": sources if is_relevant else [],
                    "source_type": source_type
                })
            
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    st.error("⏳ Rate limit (30/min). Wait a few seconds and try again.")
                else:
                    st.error(f"Error: {e}")
