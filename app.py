import os
import io
import re
import json

import streamlit as st
import pandas as pd
import numpy as np
import requests

from groq import Groq
from fastembed import TextEmbedding
from pypdf import PdfReader

st.set_page_config(page_title="DISCOM Data Chat", page_icon="⚡", layout="centered")
st.title("⚡ DISCOM Data Chat")
st.caption("Ask questions about the live feeder billing table, or upload documents to chat with them.")

# ========== CONFIG ==========
SUPABASE_URL = "https://hhwjxievppujzlnyqykp.supabase.co"
DB_TABLE_NAME = "feeder_billing_consumption"
DB_PAGE_SIZE = 1000

MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3
EMBED_BATCH_SIZE = 64
MAX_RESULT_ROWS = 200          # rows returned from a query
MAX_ROWS_TO_LLM = 60           # rows actually shown to the model for phrasing
MAX_DISTINCT_LISTED = 60       # cardinality cutoff for listing values in the schema

NUMERIC_COLS = [
    "mf", "initial_reading_kwh", "final_reading_kwh", "pf", "kwh", "kwh_exp",
    "kvah", "md_kw", "md_kva", "kvarh_q1", "kvarh_q2", "kvarh_q3", "kvarh_q4",
    "pwr_on_dur", "billi_year",
]

AGG_FUNCS = {"sum", "mean", "min", "max", "count", "nunique", "median", "std"}
OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "isnull", "notnull"}

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
"""


def get_secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


groq_key = get_secret("GROQ_API_KEY")
supabase_anon_key = get_secret("SUPABASE_ANON_KEY")

groq_client = None
if groq_key:
    try:
        groq_client = Groq(api_key=groq_key)
    except Exception as e:
        st.sidebar.error(f"Groq Error: {e}")


@st.cache_resource
def load_embedder():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


# ========== SUPABASE → DATAFRAME ==========
def _supabase_headers(extra=None):
    headers = {
        "apikey": supabase_anon_key,
        "Authorization": f"Bearer {supabase_anon_key}",
    }
    if extra:
        headers.update(extra)
    return headers


@st.cache_data(ttl=3600, show_spinner=False, max_entries=1)
def load_dataframe():
    """Pull the whole table into memory once per hour. 8k rows is a few MB."""
    url = f"{SUPABASE_URL}/rest/v1/{DB_TABLE_NAME}?select=*"
    rows, offset = [], 0
    while True:
        headers = _supabase_headers({
            "Range-Unit": "items",
            "Range": f"{offset}-{offset + DB_PAGE_SIZE - 1}",
        })
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        rows.extend(batch)
        if len(batch) < DB_PAGE_SIZE:
            break
        offset += DB_PAGE_SIZE

    df = pd.DataFrame(rows)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False, max_entries=1)
def build_schema_summary(_df, fingerprint):
    """
    Describe the table to the model: dtypes, ranges, and — critically — the actual
    distinct values of categorical columns, so filters match real data.
    """
    lines = [f"Table: {DB_TABLE_NAME}  ({len(_df)} rows)", "", "Columns:"]
    for col in _df.columns:
        s = _df[col]
        if pd.api.types.is_numeric_dtype(s):
            lines.append(
                f"- {col} (numeric): min={s.min():.4g}, max={s.max():.4g}, "
                f"mean={s.mean():.4g}, nulls={int(s.isna().sum())}"
            )
        else:
            distinct = s.dropna().astype(str).str.strip().unique()
            if len(distinct) <= MAX_DISTINCT_LISTED:
                vals = ", ".join(sorted(distinct)[:MAX_DISTINCT_LISTED])
                lines.append(f"- {col} (text): {len(distinct)} distinct -> {vals}")
            else:
                sample = ", ".join(sorted(distinct)[:8])
                lines.append(f"- {col} (text): {len(distinct)} distinct, e.g. {sample}, ...")
    return "\n".join(lines)


# ========== QUERY SPEC EXECUTION ==========
def _safe_name(name):
    return re.sub(r"\W+", "_", str(name)).strip("_") or "value"


def run_spec(df, spec, max_rows=MAX_RESULT_ROWS):
    """Execute a validated JSON query spec with pandas. No eval, no generated code."""
    out = df
    applied = []

    for f in spec.get("filters") or []:
        col, op, val = f.get("column"), f.get("op"), f.get("value")
        if col not in out.columns or op not in OPS:
            continue
        s = out[col]
        try:
            if op in {">", ">=", "<", "<="}:
                num = pd.to_numeric(s, errors="coerce")
                v = float(val)
                out = out[{">": num > v, ">=": num >= v, "<": num < v, "<=": num <= v}[op]]
            elif op == "==":
                out = out[s.astype(str).str.strip().str.lower() == str(val).strip().lower()]
            elif op == "!=":
                out = out[s.astype(str).str.strip().str.lower() != str(val).strip().lower()]
            elif op in {"in", "not_in"}:
                vals = [str(v).strip().lower() for v in (val if isinstance(val, list) else [val])]
                mask = s.astype(str).str.strip().str.lower().isin(vals)
                out = out[mask if op == "in" else ~mask]
            elif op == "contains":
                out = out[s.astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "isnull":
                out = out[s.isna()]
            elif op == "notnull":
                out = out[s.notna()]
        except Exception:
            continue
        applied.append(f"{col} {op} {val}")

    matched_rows = len(out)

    group_by = [c for c in (spec.get("group_by") or []) if c in out.columns]
    aggs = [a for a in (spec.get("aggregations") or []) if a.get("func") in AGG_FUNCS]

    if aggs:
        named = {}
        for a in aggs:
            col = a.get("column")
            if col in out.columns:
                named[_safe_name(a.get("as") or f"{a['func']}_{col}")] = (col, a["func"])
        if named:
            if group_by:
                out = out.groupby(group_by, dropna=False).agg(**named).reset_index()
            else:
                out = pd.DataFrame({k: [getattr(out[v[0]], v[1])()] for k, v in named.items()})
    elif group_by:
        out = out.groupby(group_by, dropna=False).size().reset_index(name="row_count")
    else:
        cols = [c for c in (spec.get("columns") or []) if c in out.columns]
        if cols:
            out = out[cols]

    sort = spec.get("sort") or {}
    if sort.get("by") in out.columns:
        out = out.sort_values(sort["by"], ascending=not sort.get("desc", True), na_position="last")

    limit = spec.get("limit")
    limit = max_rows if limit is None else min(int(limit), max_rows)
    return out.head(limit), matched_rows, applied


# ========== LLM CALLS ==========
def call_llm(prompt, temperature=0.1, max_tokens=1024):
    resp = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def parse_json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


PLANNER_TEMPLATE = """You convert a user's question into a JSON query specification for a pandas DataFrame.

{knowledge}

=== ACTUAL SCHEMA AND VALUES ===
{schema}

=== RECENT CONVERSATION ===
{history}

=== USER QUESTION ===
{question}

Decide the route:
- "data"    -> the question is about the table above (totals, counts, comparisons, rankings, lookups)
- "docs"    -> the question is about uploaded documents, not the table
- "general" -> neither; general knowledge

If route is "data", build the spec. Rules:
- Use ONLY column names that appear in the schema.
- Filter values MUST match the actual distinct values listed above, including capitalisation.
- Use aggregations for any question about totals, averages, counts, or rankings. Never return raw rows for those.
- func must be one of: sum, mean, min, max, count, nunique, median, std
- op must be one of: ==, !=, >, >=, <, <=, in, not_in, contains, isnull, notnull
- Set "limit" sensibly (e.g. 10 for "top 10", 50 otherwise).

Respond with JSON ONLY, no prose, no markdown fences:
{{
  "route": "data",
  "restated": "plain-English restatement of what will be computed",
  "filters": [{{"column": "bill_month", "op": "==", "value": "Mar"}}],
  "group_by": ["division"],
  "aggregations": [{{"column": "kwh", "func": "sum", "as": "total_kwh"}}],
  "columns": [],
  "sort": {{"by": "total_kwh", "desc": true}},
  "limit": 10
}}"""

ANSWER_TEMPLATE = """You are a power-distribution data analyst. Answer the user's question using the query result below.

{knowledge}

=== WHAT WAS COMPUTED ===
{restated}
Filters applied: {applied}
Rows matching the filters: {matched}
Result rows shown: {shown}

=== QUERY RESULT ===
{result}

=== USER QUESTION ===
{question}

Rules:
- The result table is authoritative. Do not invent, estimate, or extrapolate numbers not present in it.
- Report figures with units (kWh, kVA, etc.) and sensible rounding.
- If the result is empty, say so plainly and suggest which filter may be wrong.
- If the result was truncated, say the list is partial.
- Be concise. Lead with the answer, then supporting detail."""


# ========== DOCUMENT RAG (uploads only) ==========
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if len(piece) > 50:
            chunks.append(piece)
        start = end - overlap if end < len(text) else end
    return chunks


def embed_chunks(embedder, chunks, show_progress=False, batch_size=EMBED_BATCH_SIZE):
    if not chunks:
        return [], []
    kept, vectors = [], []
    progress = st.progress(0) if show_progress else None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        try:
            out = list(embedder.embed([c[:8000] for c in batch], batch_size=batch_size))
            kept.extend(batch)
            vectors.extend(np.asarray(v, dtype=np.float32) for v in out)
        except Exception as e:
            st.warning(f"Embed error near chunk {start}: {e}")
        if progress:
            progress.progress(min((start + batch_size) / len(chunks), 1.0))
    if progress:
        progress.empty()
    return kept, vectors


def add_docs(chunks, vectors, stat):
    if not vectors:
        return False
    matrix = np.array(vectors, dtype=np.float32)
    if st.session_state.doc_embeddings is None:
        st.session_state.doc_embeddings = matrix
        st.session_state.doc_chunks = list(chunks)
    else:
        st.session_state.doc_embeddings = np.vstack([st.session_state.doc_embeddings, matrix])
        st.session_state.doc_chunks.extend(chunks)
    st.session_state.file_stats.append(stat)
    return True


def retrieve_docs(embedder, query, top_k=TOP_K):
    matrix = st.session_state.doc_embeddings
    if matrix is None or not len(matrix):
        return [], 0.0
    q = np.asarray(list(embedder.embed([query[:8000]]))[0], dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    denom[denom == 0] = 1e-9
    sims = (matrix @ q) / denom
    idx = np.argsort(sims)[-top_k:][::-1]
    return [(st.session_state.doc_chunks[i], float(sims[i])) for i in idx], float(sims[idx[0]])


# ========== EXTRACTORS ==========
def extract_pdf_text(file):
    reader = PdfReader(file)
    parts = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text()
        if txt and txt.strip():
            parts.append(f"[Page {i+1}]\n{txt}")
    return "\n\n".join(parts), len(reader.pages)


def extract_word_text(file):
    from docx import Document
    doc = Document(file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    tables = []
    for table in doc.tables:
        tables.append("\n".join(
            " | ".join(cell.text.strip() for cell in row.cells) for row in table.rows
        ))
    text = "\n\n".join(paragraphs)
    if tables:
        text += "\n\n[Tables]\n" + "\n\n".join(tables)
    return text, 1


def extract_excel_text(file):
    xls = pd.ExcelFile(file)
    parts = [
        f"[Sheet: {name}]\n{pd.read_excel(xls, sheet_name=name).to_string(index=False)}"
        for name in xls.sheet_names
    ]
    return "\n\n".join(parts), len(xls.sheet_names)


def extract_google_sheet(sheet_url):
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        st.error("Invalid URL. Expected https://docs.google.com/spreadsheets/d/XXXX/edit")
        return None, 0
    export = f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv"
    resp = requests.get(export, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text)).to_string(index=False), 1


def process_file(file):
    name = file.name.lower()
    try:
        if name.endswith(".pdf"):
            return extract_pdf_text(file)
        if name.endswith(".docx"):
            return extract_word_text(file)
        if name.endswith((".xlsx", ".xls")):
            return extract_excel_text(file)
        if name.endswith(".csv"):
            return pd.read_csv(file).to_string(index=False), 1
        if name.endswith(".txt"):
            return file.read().decode("utf-8"), 1
    except Exception as e:
        st.error(f"Read error ({file.name}): {e}")
    return "", 0


# ========== SESSION STATE ==========
for key, default in [
    ("messages", []),
    ("doc_embeddings", None),
    ("doc_chunks", []),
    ("file_stats", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ========== LOAD TABLE ==========
df, schema_summary, db_error = None, "", None
if supabase_anon_key:
    try:
        df = load_dataframe()
        schema_summary = build_schema_summary(df, fingerprint=(len(df), tuple(df.columns)))
    except Exception as e:
        db_error = str(e)


# ========== SIDEBAR ==========
with st.sidebar:
    st.header("Status")
    if groq_client:
        st.success("Groq connected")
    else:
        st.error("No Groq API key")

    if df is not None:
        st.success(f"Live table: {len(df):,} rows x {len(df.columns)} cols")
        if st.button("Refresh table", use_container_width=True):
            load_dataframe.clear()
            build_schema_summary.clear()
            st.rerun()
        with st.expander("Schema seen by the model"):
            st.code(schema_summary, language="text")
    elif db_error:
        st.error(f"DB error: {db_error}")
    else:
        st.warning("Supabase key not set")

    st.divider()
    st.subheader("Documents (optional)")
    st.caption("PDFs and reports use semantic search. The table above does not.")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files and st.button("Index documents", use_container_width=True):
        embedder = load_embedder()
        with st.spinner("Indexing..."):
            for f in uploaded_files:
                text, pages = process_file(f)
                if not text:
                    continue
                kept, vectors = embed_chunks(embedder, chunk_text(text), show_progress=True)
                add_docs(kept, vectors, {"name": f.name, "pages": pages, "chunks": len(kept)})
        st.success(f"{len(st.session_state.doc_chunks)} chunks indexed")

    sheet_url = st.text_input("Google Sheet URL (public)", placeholder="https://docs.google.com/...")
    if st.button("Fetch sheet", use_container_width=True) and sheet_url:
        embedder = load_embedder()
        with st.spinner("Fetching..."):
            try:
                text, pages = extract_google_sheet(sheet_url)
                if text:
                    kept, vectors = embed_chunks(embedder, chunk_text(text), show_progress=True)
                    add_docs(kept, vectors, {"name": "Google_Sheet", "pages": pages, "chunks": len(kept)})
                    st.success("Sheet indexed")
            except Exception as e:
                st.error(f"Sheet error: {e}")

    if st.session_state.file_stats:
        st.divider()
        for s in st.session_state.file_stats:
            st.caption(f"{s['name']} - {s['chunks']} chunks")

    if st.session_state.messages:
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    show_spec = st.checkbox(
        "Show query spec", value=False,
        help="Display the JSON the model generated for each question.",
    )


# ========== GUARDS ==========
if not groq_client:
    st.warning("Add GROQ_API_KEY and SUPABASE_ANON_KEY to your Streamlit secrets.")
    st.stop()

if df is None:
    st.warning("Could not load the database table. Check SUPABASE_ANON_KEY and table permissions.")
    st.stop()


# ========== CHAT ==========
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("result_table"):
            with st.expander("Result table"):
                st.dataframe(pd.DataFrame(m["result_table"]), use_container_width=True)
        if m.get("spec"):
            with st.expander("Query spec"):
                st.code(json.dumps(m["spec"], indent=2), language="json")

if prompt := st.chat_input("e.g. total kWh by division for March 2026"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            history = "\n".join(
                f"{m['role']}: {m['content'][:300]}" for m in st.session_state.messages[-5:-1]
            )

            with st.spinner("Planning query..."):
                plan_raw = call_llm(PLANNER_TEMPLATE.format(
                    knowledge=TABLE_KNOWLEDGE,
                    schema=schema_summary,
                    history=history,
                    question=prompt,
                ), temperature=0.0, max_tokens=700)
                spec = parse_json(plan_raw) or {"route": "general"}

            route = spec.get("route", "general")
            record = {"role": "assistant", "spec": spec if show_spec else None, "result_table": None}

            if route == "data":
                with st.spinner("Running query..."):
                    result, matched, applied = run_spec(df, spec)
                    shown = result.head(MAX_ROWS_TO_LLM)
                    answer = call_llm(ANSWER_TEMPLATE.format(
                        knowledge=TABLE_KNOWLEDGE,
                        restated=spec.get("restated", "-"),
                        applied="; ".join(applied) or "none",
                        matched=f"{matched:,}",
                        shown=f"{len(shown)} of {len(result)}",
                        result=shown.to_string(index=False) if len(shown) else "(empty)",
                        question=prompt,
                    ), temperature=0.2)
                st.markdown(answer)
                st.caption(f"Computed from {matched:,} matching rows")
                if len(result):
                    with st.expander("Result table"):
                        st.dataframe(result, use_container_width=True)
                    record["result_table"] = result.to_dict("records")

            elif route == "docs" and st.session_state.doc_chunks:
                hits, best = retrieve_docs(load_embedder(), prompt)
                context = "\n\n---\n\n".join(c for c, _ in hits)
                answer = call_llm(
                    f"Answer using the document extracts below.\n\n=== EXTRACTS ===\n{context}\n\n"
                    f"=== QUESTION ===\n{prompt}\n\nIf the extracts do not answer it, say so."
                )
                st.markdown(answer)
                st.caption(f"From documents (top score {best:.2f})")

            else:
                answer = call_llm(
                    f"=== RECENT CONVERSATION ===\n{history}\n\n=== QUESTION ===\n{prompt}\n\n"
                    "Answer clearly using general knowledge."
                )
                st.markdown(answer)
                st.caption("General knowledge")

            if show_spec:
                with st.expander("Query spec"):
                    st.code(json.dumps(spec, indent=2), language="json")

            record["content"] = answer
            st.session_state.messages.append(record)

        except Exception as e:
            if "429" in str(e):
                st.error("Groq rate limit (30/min). Wait a few seconds.")
            else:
                st.error(f"Error: {e}")
