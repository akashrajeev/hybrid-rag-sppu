"""Pagewise engine: index any text PDF on the fly and answer from it with the benchmarked retrieval
(clean query -> hybrid BM25 + bge-small, a=0.5, top-20 -> Jina-tiny rerank -> top-3 -> Groq gpt-oss)."""
import os, re, sys, json, time, shutil, threading, uuid
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ragcore as R
from rank_bm25 import BM25Okapi

LIB = os.environ.get("LIBRARY_DIR", os.path.join(R.HERE, "library"))
CHUNK_WORDS, MIN_PAGE_WORDS = 170, 8
SYSTEM = ("You answer questions about the user's uploaded document using ONLY the numbered context passages. "
          "Reply with a short, exact answer (names, codes, dates and numbers exactly as written). "
          "End the answer with the passage numbers you used, like [1] or [1][3]. "
          "If the context does not contain the answer, reply exactly: NOT FOUND.")
_models = {"emb": None, "rr": None}; _mlock = threading.Lock()
def embedder():
    with _mlock:
        if _models["emb"] is None:
            from fastembed import TextEmbedding; _models["emb"] = TextEmbedding(R.EMBEDDER)
        return _models["emb"]
def reranker():
    with _mlock:
        if _models["rr"] is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder; _models["rr"] = TextCrossEncoder(model_name=R.RERANKER)
        return _models["rr"]

def doc_dir(i): return os.path.join(LIB, i)
def read_json(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
def write_json(p, o):
    t = p + ".tmp"; json.dump(o, open(t, "w")); os.replace(t, p)

def extract_chunks(pdf_path):
    import fitz
    d = fitz.open(pdf_path); chunks = []; skipped = []
    for pn in range(len(d)):
        words = d[pn].get_text("text").split()
        if len(words) < MIN_PAGE_WORDS: skipped.append(pn + 1); continue
        for s in range(0, len(words), CHUNK_WORDS):
            part = words[s:s + CHUNK_WORDS]
            if len(part) >= 5 or s == 0: chunks.append({"page": pn + 1, "text": " ".join(part)})
    return len(d), chunks, skipped

def add_document(raw, filename):
    """Save the PDF and start indexing in a thread. Returns the doc id."""
    i = uuid.uuid4().hex[:10]; os.makedirs(doc_dir(i), exist_ok=True)
    open(os.path.join(doc_dir(i), "file.pdf"), "wb").write(raw)
    write_json(os.path.join(doc_dir(i), "meta.json"), dict(id=i, name=filename, status="indexing", added=time.time()))
    threading.Thread(target=_index, args=(i,), daemon=True).start(); return i

def _meta(i, **kw):
    p = os.path.join(doc_dir(i), "meta.json"); m = read_json(p, {}); m.update(kw); write_json(p, m)

def _index(i):
    t0 = time.time()
    try:
        n, chunks, skipped = extract_chunks(os.path.join(doc_dir(i), "file.pdf"))
        if not chunks:
            _meta(i, status="error", pages=n, skipped=skipped, error="No text found. This looks like a scanned PDF (images only). OCR is not included yet."); return
        emb = np.array(list(embedder().embed([c["text"] for c in chunks], batch_size=32)))
        write_json(os.path.join(doc_dir(i), "chunks.json"), chunks); np.save(os.path.join(doc_dir(i), "emb.npy"), emb)
        _meta(i, status="ready", pages=n, chunks=len(chunks), skipped=skipped, seconds=round(time.time() - t0, 1))
    except Exception as e:
        _meta(i, status="error", error=f"Could not read this file: {e}")

def list_docs():
    out = []
    if os.path.isdir(LIB):
        for i in os.listdir(LIB):
            m = read_json(os.path.join(LIB, i, "meta.json"))
            if m:
                if m.get("status") == "indexing" and time.time() - m.get("added", 0) > 1800: m["status"] = "error"; m["error"] = "Indexing was interrupted."
                out.append(m)
    return sorted(out, key=lambda m: m.get("added", 0))

def delete_doc(i):
    if re.fullmatch(r"[0-9a-f]{10}", i) and os.path.isdir(doc_dir(i)): shutil.rmtree(doc_dir(i)); _cache.pop(i, None); return True
    return False

class DocIndex:
    def __init__(self, i):
        self.id = i; self.CH = read_json(os.path.join(doc_dir(i), "chunks.json"))
        self.emb = np.load(os.path.join(doc_dir(i), "emb.npy")); self.bm = BM25Okapi([R.tok(c["text"]) for c in self.CH])
    def retrieve(self, q, K=20, k=3):
        t = {}; t0 = time.time(); cq = R.clean(q)
        b = np.array(self.bm.get_scores(R.tok(cq)), dtype=float); d = self.emb @ np.array(list(embedder().query_embed([q])))[0]
        s = 0.5 * R.Index.mm(d) + 0.5 * R.Index.mm(b); cand = [int(x) for x in np.argsort(-s)[:K]]; t["hybrid"] = time.time() - t0
        t0 = time.time(); sc = list(reranker().rerank(q, [self.CH[x]["text"] for x in cand], batch_size=4))
        order = [cand[x] for x in np.argsort(-np.array(sc))]; t["rerank"] = time.time() - t0
        return order[:k], cq, t
_cache = {}
def get_index(i):
    if i not in _cache: _cache[i] = DocIndex(i)
    return _cache[i]

def highlight(pdf_path, page, answer, question):
    """Find the line(s) on the cited page that best match the answer text (fallback: the question terms)."""
    import fitz
    d = fitz.open(pdf_path); pg = d[page - 1]; W, H = pg.rect.width, pg.rect.height
    lines = {}
    for x0, y0, x1, y1, w, b, l, _ in pg.get_text("words"): lines.setdefault((b, l), []).append((x0, y0, x1, y1, w))
    keep = lambda s: {t for t in R.tok(s) if t not in R.STOP and len(t) > 1}
    at = keep(re.sub(r"\[\d+\]", "", answer)); qt = keep(question)
    scored = []
    for key, ws in lines.items():
        lt = keep(" ".join(w[4] for w in ws))
        if len(ws) < 4 or not lt: continue
        sc = (len(lt & at) + 0.5 * len(lt & qt)) / (len(lt) ** 0.5)
        scored.append((sc, key, ws))
    if not scored: return None
    scored.sort(key=lambda x: -x[0]); best = scored[0][0]
    if best < 0.9: return None
    pick = [x for x in scored[:2] if x[0] >= best * 0.7]
    rects = []
    for _, _, ws in pick:
        rects.append([min(w[0] for w in ws) / W, min(w[1] for w in ws) / H, max(w[2] for w in ws) / W, max(w[3] for w in ws) / H])
    return rects

_llm = None
def llm():
    global _llm
    if _llm is None: _llm = R.Groq()
    return _llm

_alock = threading.Lock()
def ask(i, q):
    ix = get_index(i)
    with _alock:
        order, cq, t = ix.retrieve(q)
        chunks = [ix.CH[x] for x in order]
        src = [dict(n=n + 1, page=c["page"], text=c["text"]) for n, c in enumerate(chunks)]
        res = dict(question=q, cleaned=cq, sources=src, timing={k: round(v, 2) for k, v in t.items()})
        g = llm()
        if not g.key:
            res.update(answer=None, cited=[chunks[0]["page"]], note="No GROQ_API_KEY set, showing the best-matching passages only.")
            res["highlight"] = dict(page=chunks[0]["page"], rects=highlight(os.path.join(doc_dir(i), "file.pdf"), chunks[0]["page"], "", q))
            return res
        ctx = "\n\n".join(f"[{n+1}] (page {c['page']}) {c['text']}" for n, c in enumerate(chunks))
        t0 = time.time()
        text, _ = g.chat(f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:", system=SYSTEM)
        res["timing"]["answer"] = round(time.time() - t0, 2)
    used = [int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)]
    if text.strip().upper().startswith("NOT FOUND"): used = []
    clean_text = re.sub(r"\s*\[\d+\]", "", text).strip()
    cited_pages = []
    for n in (used or ([] if not clean_text or clean_text == "NOT FOUND" else [1])):
        p = chunks[n - 1]["page"]
        if p not in cited_pages: cited_pages.append(p)
    res.update(answer=clean_text, cited=cited_pages)
    if cited_pages:
        res["highlight"] = dict(page=cited_pages[0], rects=highlight(os.path.join(doc_dir(i), "file.pdf"), cited_pages[0], clean_text, q))
    return res

def render_page(i, n, dpi=110):
    import fitz
    d = fitz.open(os.path.join(doc_dir(i), "file.pdf"))
    if not 1 <= n <= len(d): return None
    return d[n - 1].get_pixmap(dpi=dpi).tobytes("png")

def resume_unfinished():
    """A restart kills indexing threads; pick those documents up again."""
    for m in list_docs():
        if m.get("status") == "indexing": threading.Thread(target=_index, args=(m["id"],), daemon=True).start()
