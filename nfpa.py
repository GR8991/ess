import streamlit as st
from io import BytesIO
from typing import List, Dict, Tuple
import re
import numpy as np

# ---------- Optional deps (loaded lazily after indexing) ----------
# sentence-transformers for embeddings, FAISS for vector search, rank_bm25 for BM25

# ---------- PDF utils ----------
def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a PDF using PyPDF2 (pypdf)."""
    try:
        from pypdf import PdfReader
    except Exception as e:
        st.error("Missing dependency 'pypdf'. Install requirements and restart.")
        raise
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def slice_chapter_9(full_text: str) -> str:
    """
    Grab all text from 'Chapter 9' to the next 'Chapter ' heading.
    """
    chap9 = re.search(r'(?is)(chapter\s+9\b.*)', full_text)
    if not chap9:
        return full_text
    start = chap9.start(1)
    nxt = re.search(r'(?is)\n\s*chapter\s+(1[0-9]|[1-8])\b', full_text[start+7:])
    if nxt:
        end = start + 7 + nxt.start(0)
        return full_text[start:end]
    return full_text[start:]

def section_chunker(ch9_text: str) -> List[Dict]:
    """Split Chapter 9 into section-aware chunks."""
    text = re.sub(r'\u00a0', ' ', ch9_text)
    matches = list(re.finditer(r'(?m)^(\s*)(9\.(?:\d+)(?:\.\d+)*)\s*(.*)$', text))
    chunks = []
    if not matches:
        return [{
            "section": "9",
            "title": "Chapter 9 (unstructured)",
            "text": text.strip(),
            "type": "paragraph",
        }]
    for i, m in enumerate(matches):
        sec = m.group(2)
        title_line = m.group(3).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[start:end].strip()
        tables = []
        for t in re.finditer(r'(?mis)(Table\s+9\.[\dA-Za-z]+.*?)(?=\n\s*Table\s+9\.|\Z)', body):
            tables.append(t.group(1).strip())
        if tables:
            body_wo_tables = re.sub(r'(?mis)Table\s+9\.[\dA-Za-z]+.*?(?=\n\s*Table\s+9\.|\Z)', '', body).strip()
        else:
            body_wo_tables = body
        if body_wo_tables:
            chunks.append({
                "section": sec,
                "title": title_line if title_line else "",
                "text": body_wo_tables,
                "type": "paragraph",
            })
        for t in tables:
            mtitle = re.match(r'(Table\s+9\.[\dA-Za-z]+)', t)
            table_id = mtitle.group(1) if mtitle else "Table 9.?"
            chunks.append({
                "section": sec,
                "title": table_id,
                "text": t,
                "type": "table",
                "table_id": table_id,
            })
    # refine long chunks
    refined = []
    for ch in chunks:
        words = ch["text"].split()
        if len(words) <= 900 or ch["type"] == "table":
            refined.append(ch)
        else:
            parts = re.split(r'(?m)\n\s*\(\d+\)\s*', ch["text"])
            if len(parts) > 1:
                for j, part in enumerate(parts):
                    pt = part.strip()
                    if not pt:
                        continue
                    refined.append({
                        **ch,
                        "text": pt,
                        "title": (ch["title"] + f" – item {j+1}")[:120]
                    })
            else:
                for k in range(0, len(words), 800):
                    refined.append({**ch, "text": " ".join(words[k:k+800])})
    return refined

# ---------- Index building ----------
class HybridIndex:
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.bm25 = None
        self.vecs = None
        self.index = None
        self.embedder = None

    def build(self, use_embeddings: bool = True):
        try:
            from rank_bm25 import BM25Okapi
        except Exception:
            st.error("Missing dependency 'rank_bm25'. Install requirements and restart.")
            raise
        corpus = [d["text"] for d in self.docs]
        self.bm25 = BM25Okapi([c.split() for c in corpus])
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
            except Exception:
                st.warning("Embeddings / FAISS not available. Continuing with BM25 only.")
                return
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.vecs = self.embedder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(self.vecs.shape[1])
            self.index.add(self.vecs)

    def search(self, query: str, k_bm25=25, k_vec=25, k_final=10) -> List[Tuple[float, Dict]]:
        tokens = query.split()
        bm_scores = self.bm25.get_scores(tokens)
        bm_top_idx = np.argsort(-bm_scores)[:k_bm25]
        cand = set(bm_top_idx.tolist())
        if self.index is not None and self.embedder is not None:
            qv = self.embedder.encode([query], normalize_embeddings=True)
            sims, ids = self.index.search(qv, k_vec)
            cand |= set(ids[0].tolist())
            cand = list(cand)
            fused = []
            for i in cand:
                sim = float(np.dot(qv, self.vecs[i])) if self.vecs is not None else 0.0
                fused.append((0.55*bm_scores[i] + 0.45*sim, i))
            fused.sort(reverse=True, key=lambda x: x[0])
            top = fused[:k_final]
        else:
            top = sorted([(bm_scores[i], i) for i in cand], reverse=True)[:k_final]
        return [(score, self.docs[i]) for score, i in top]

# ---------- Simple answer composer ----------
def make_answer(query: str, hits: List[Tuple[float, Dict]]):
    if not hits:
        return "No Chapter 9 context found for this query.", []
    bullets = []
    seen = set()
    for _, d in hits[:5]:
        key = (d.get("section"), d.get("title"))
        if key in seen: 
            continue
        seen.add(key)
        snippet = d["text"]
        snippet = re.sub(r'\s+', ' ', snippet).strip()
        snippet = snippet[:400] + ("…" if len(snippet) > 400 else "")
        cite = d.get("title") or d.get("section") or "Chapter 9"
        if d.get("table_id"):
            cite = d["table_id"]
        bullets.append(f"• {snippet}  \n  — *NFPA 855 (2023), §{d.get('section')}*; {cite}")
    lead = "Here are the most relevant Chapter 9 excerpts and citations:"
    return lead, bullets

# ---------- Streamlit UI ----------
st.set_page_config(page_title="NFPA 855 – Chapter 9 Chatbot", layout="wide")
st.title("🔎 NFPA 855 – Chapter 9 Electrochemical ESS Chatbot")

st.sidebar.header("1) Load NFPA 855 PDF")
pdf_file = st.sidebar.file_uploader("Upload NFPA 855 (2023) PDF", type=["pdf"])
use_sample = st.sidebar.checkbox("I don't have the file now (index later)", value=False)

if "index" not in st.session_state:
    st.session_state.index = None

if st.sidebar.button("Build Chapter 9 Index") and (pdf_file or use_sample):
    if pdf_file:
        raw = pdf_file.read()
        full_text = extract_pdf_text(raw)
    else:
        st.warning("No PDF provided. The app will run but won't return grounded answers until you upload the standard.")
        full_text = ""
    ch9_text = slice_chapter_9(full_text) if full_text else ""
    docs = section_chunker(ch9_text) if ch9_text else []
    if not docs:
        st.info("Proceeding with an empty index (you can still upload the PDF later).")
    idx = HybridIndex(docs)
    idx.build(use_embeddings=True)
    st.session_state.index = idx
    st.success(f"Indexed {len(docs)} Chapter 9 chunks.")

st.divider()
st.header("💬 Ask a Chapter 9 question")
q = st.text_input("Your question", placeholder="e.g., Do lithium-ion ESS need explosion control indoors?")

if st.session_state.index is None:
    st.info("➡️ Upload the PDF and click **Build Chapter 9 Index** in the sidebar to enable deep search.")
else:
    if st.button("Search"):
        hits = st.session_state.index.search(q) if q else []
        lead, bullets = make_answer(q, hits)
        st.write(lead)
        for b in bullets:
            st.markdown(b)

st.sidebar.markdown("---")
st.sidebar.header("Tips")
st.sidebar.markdown("""
- Ask **focused** questions: *“What does §9.1.5 require?”*  
- Use keywords: *UL 9540A, explosion control, deflagration, fire barriers, occupied work centers.*  
- If you need cross-references (e.g., **§4.4 HMA**), ask the bot to point them out explicitly.
""")
