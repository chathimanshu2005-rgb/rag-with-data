import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np

# ========== PAGE SETUP ==========
st.set_page_config(page_title="My RAG App", page_icon="📚", layout="centered")

st.title("📚 My Free RAG Application")
st.markdown("""
**Upload PDF documents → Ask questions → Get AI answers based ONLY on your documents**

*Built with 100% free tools: Streamlit + Gemini API + Python*
""")

# ========== GEMINI SETUP ==========
import os

# Try to get API key from multiple sources
try:
    # Source 1: Streamlit Cloud secrets
    api_key = st.secrets.get("GEMINI_API_KEY", None)

    # Source 2: Environment variable (for local)
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    # Source 3: Direct input (emergency fallback)
    if not api_key:
        st.warning("⚠️ API Key not found in secrets!")
        api_key = st.text_input("Paste your Gemini API Key here (for testing only):", type="password")
        if not api_key:
            st.info("""
            **To fix this permanently:**
            1. Go to your app on [share.streamlit.io](https://share.streamlit.io)
            2. Click ⋮ → **Settings** → **Secrets**
            3. Add: `GEMINI_API_KEY = "your-key"`
            4. Click **Save** and **Reboot**
            """)
            st.stop()

    genai.configure(api_key=api_key)
    st.success("✅ Gemini API connected successfully!")

except Exception as e:
    st.error(f"❌ Error with API Key: {e}")
    st.stop()

# ========== HELPER FUNCTIONS ==========

def get_embedding(text: str):
    """Convert text to vector using Gemini Embedding (FREE tier)"""
    try:
        safe_text = text[:8000] if len(text) > 8000 else text
        result = genai.embed_content(
            model="models/embedding-001",
            content=safe_text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'], dtype=np.float32)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def cosine_similarity(a, b):
    """Calculate similarity between two vectors (no FAISS needed!)"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_answer(question: str, context: str) -> str:
    """Ask Gemini Flash to answer based on retrieved context"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""You are a helpful study assistant. Answer the question using ONLY the information provided in the context below.
If the answer is not found in the context, say: "I don't have enough information in the uploaded documents to answer this."

=== CONTEXT FROM DOCUMENTS ===
{context}

=== USER QUESTION ===
{question}

=== YOUR ANSWER ===
Provide a clear, accurate, and concise answer."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating answer: {e}"

def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list:
    """Split long text into overlapping chunks"""
    if not text or len(text) < chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a natural boundary
        if end < text_len:
            for sep in ['. ', '\n', ' ']:
                pos = text.rfind(sep, start, end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk and len(chunk) > 50:
            chunks.append(chunk)

        start = end - overlap

    return chunks

def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF file + return page count"""
    try:
        reader = PdfReader(pdf_file)
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(f"[Page {i+1}]\n{page_text}")
        full_text = "\n\n".join(text_parts)
        page_count = len(reader.pages)
        return full_text, page_count
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
        return "", 0

# ========== MAIN APP ==========

uploaded_files = st.file_uploader(
    "📄 Upload your PDF files (you can select multiple)",
    type=['pdf'],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 **Get started:** Upload one or more PDF files above")
    st.markdown("""
    ### 💡 What is RAG?
    **RAG** = **R**etrieval **A**ugmented **G**eneration

    1. 📄 You upload documents
    2. 🔍 The app finds the most relevant parts
    3. 🤖 AI reads those parts and answers your question
    4. ✅ You get accurate answers based on YOUR documents
    """)
    st.stop()

# Process documents button
if st.button("🚀 Process Documents", type="primary"):
    with st.spinner("🔍 Reading and analyzing your documents..."):

        all_chunks = []
        file_stats = []

        progress = st.progress(0)
        for idx, pdf_file in enumerate(uploaded_files):
            text, page_count = extract_pdf_text(pdf_file)
            if text:
                chunks = split_text(text)
                all_chunks.extend(chunks)
                file_stats.append({
                    'name': pdf_file.name,
                    'pages': page_count,
                    'chunks': len(chunks)
                })
            progress.progress((idx + 1) / len(uploaded_files))

        if not all_chunks:
            st.error("❌ Could not extract text from the uploaded PDFs. Try different files.")
            st.stop()

        # Create embeddings for each chunk
        st.write(f"🧠 Creating {len(all_chunks)} embeddings...")
        embed_progress = st.progress(0)
        embeddings = []

        for i, chunk in enumerate(all_chunks):
            emb = get_embedding(chunk)
            if emb is not None:
                embeddings.append(emb)
            embed_progress.progress((i + 1) / len(all_chunks))

        if not embeddings:
            st.error("❌ Failed to create embeddings. Check your API key and internet connection.")
            st.stop()

        # Store in session state
        st.session_state['embeddings'] = embeddings
        st.session_state['chunks'] = all_chunks
        st.session_state['file_stats'] = file_stats
        st.session_state['ready'] = True

        st.success(f"✅ Ready! Processed {len(uploaded_files)} file(s) into {len(all_chunks)} searchable chunks.")

# Show file stats in sidebar
with st.sidebar:
    st.header("📊 Document Stats")
    if 'file_stats' in st.session_state:
        for stat in st.session_state['file_stats']:
            st.markdown(f"**{stat['name']}**")
            st.markdown(f"- Pages: {stat['pages']} | Chunks: {stat['chunks']}")
    else:
        st.info("Upload and process documents to see stats")

    st.divider()
    st.markdown("""
    ### 🛠️ Tech Stack
    - **UI**: Streamlit (Free)
    - **AI**: Gemini 2.0 Flash (Free tier)
    - **Embeddings**: Gemini Embedding (Free tier)
    - **Search**: Pure Python (No FAISS needed!)
    - **PDF**: PyPDF (Open source)
    """)

# Question & Answer section
if st.session_state.get('ready', False):
    st.divider()
    st.subheader("❓ Ask Your Documents")

    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What are the main findings? Who is the author?"
    )

    if question:
        with st.spinner("🔎 Searching documents and generating answer..."):
            # 1. Embed the question
            q_emb = get_embedding(question)
            if q_emb is None:
                st.stop()

            # 2. Find top 3 most similar chunks using cosine similarity (no FAISS!)
            similarities = []
            for emb in st.session_state['embeddings']:
                sim = cosine_similarity(q_emb, emb)
                similarities.append(sim)

            # Get indices of top 3
            top_indices = np.argsort(similarities)[-3:][::-1]

            # 3. Retrieve the actual text chunks
            relevant_chunks = [st.session_state['chunks'][i] for i in top_indices]
            context = "\n\n---\n\n".join(relevant_chunks)

            # 4. Ask Gemini to answer
            answer = get_answer(question, context)

            # 5. Display results
            st.markdown("### 💡 Answer")
            st.info(answer)

            with st.expander("📄 View source text chunks used to generate this answer"):
                for i, idx in enumerate(top_indices):
                    chunk = st.session_state['chunks'][idx]
                    score = similarities[idx]
                    st.markdown(f"**Relevant Chunk {i+1}** *(similarity: {score:.3f})*")
                    st.text_area(f"chunk_{i}", chunk[:800] + ("..." if len(chunk) > 800 else ""), 
                                height=120, label_visibility="collapsed")
                    st.divider()
