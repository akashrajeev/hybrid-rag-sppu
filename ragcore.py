"""Shared library: corpus, hybrid retrieval, query cleaning, reranking, cached Groq generation."""
import json, os, re, time, hashlib, threading
import numpy as np
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(f"{HERE}/.env"):   # simple KEY=value loader so `python 2_rag_app.py` also works with a local .env
    for _l in open(f"{HERE}/.env"):
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.strip().split("=", 1); os.environ.setdefault(_k.strip(), _v.strip().strip(chr(34)).strip(chr(39)))
DATA = f"{HERE}/data"; CACHE = f"{HERE}/cache"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
RERANKER = "jinaai/jina-reranker-v1-tiny-en"
EMBEDDER = "BAAI/bge-small-en-v1.5"

# ---------------- corpus + retrieval ----------------
def tok(s): return re.findall(r"\w+", s.lower())
STOP = set("which what who whom whose when where how is are was were be the a an of in on at for to by with from and or its it this that does do did has have can may should must as into about than then there their students student listed list included following title course code college institute venue competition recommended taught covered".split())
def clean(q):
    """Drop filler words from the query that goes to BM25 (the dense model still sees the full question)."""
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.&'/]*", q)
    keep = [t for t in toks if t.lower().strip(".") not in STOP]
    return " ".join(keep) if len(keep) >= 2 else q

class Index:
    def __init__(self):
        if not os.path.exists(f"{DATA}/chunks.json"): raise SystemExit("Run `python fetch_data.py` first.")
        self.CH = json.load(open(f"{DATA}/chunks.json"))
        self.bm = BM25Okapi([tok(c["text"]) for c in self.CH])
        self._emb = None; self._m = None; self._rr = None
    def embedder(self):
        if self._m is None:
            from fastembed import TextEmbedding
            self._m = TextEmbedding(EMBEDDER)
        return self._m
    def emb(self):
        if self._emb is None:
            p = f"{DATA}/emb.npy"
            e = np.load(p) if os.path.exists(p) else np.zeros((0, 384))
            if len(e) != len(self.CH):
                e = np.array(list(self.embedder().embed([c["text"] for c in self.CH], batch_size=32))); np.save(p, e)
            self._emb = e
        return self._emb
    def qvec(self, q): return np.array(list(self.embedder().query_embed([q])))[0]
    @staticmethod
    def mm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
    def scores(self, q, use_clean=False):
        b = np.array(self.bm.get_scores(tok(clean(q) if use_clean else q)), dtype=float)
        return b, self.emb() @ self.qvec(q)
    def bm25(self, q, K=20): return [int(i) for i in np.argsort(-self.scores(q)[0])[:K]]
    def dense(self, q, K=20): return [int(i) for i in np.argsort(-self.scores(q)[1])[:K]]
    def hybrid(self, q, K=20, a=0.5, use_clean=False):
        b, d = self.scores(q, use_clean); s = a * self.mm(d) + (1 - a) * self.mm(b)
        return [int(i) for i in np.argsort(-s)[:K]]
    def reranker(self):
        if self._rr is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._rr = TextCrossEncoder(model_name=RERANKER)
        return self._rr
    def rerank(self, q, cands, batch=4):
        sc = list(self.reranker().rerank(q, [self.CH[i]["text"] for i in cands], batch_size=batch))
        return [cands[i] for i in np.argsort(-np.array(sc))]
    def new_pipeline(self, q, K=20):
        """NEW retrieval: cleaned query -> hybrid top-K -> cross-encoder rerank."""
        return self.rerank(q, self.hybrid(q, K, use_clean=True))

# ---------------- generation (Groq, OpenAI-compatible, free tier) ----------------
SYSTEM = ("You answer questions about Savitribai Phule Pune University documents using ONLY the provided context. "
          "Reply with the shortest exact answer (names, codes and numbers exactly as written in the context). "
          "If the context does not contain the answer, reply exactly: NOT FOUND.")
def build_prompt(q, chunks):
    ctx = "\n\n".join(f"[{i+1}] (doc {c['doc']}, page {c['page']}) {c['text']}" for i, c in enumerate(chunks))
    return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"

class Groq:
    """Cached, rate-limited chat client. Free tier is ~8000 tokens/minute, so calls are throttled.
    Every response is stored in cache/llm_cache.jsonl keyed by sha256(model+prompt), so re-runs are free and identical."""
    TPM = 7600
    def __init__(self, key=None):
        import requests; self.rq = requests
        self.key = key or os.environ.get("GROQ_API_KEY")
        self.path = f"{CACHE}/llm_cache.jsonl"; os.makedirs(CACHE, exist_ok=True)
        self.cache = {}
        if os.path.exists(self.path):
            for l in open(self.path): r = json.loads(l); self.cache[r["k"]] = r
        self.win = []   # (time, tokens)
    @staticmethod
    def key_of(model, system, prompt): return hashlib.sha256(json.dumps([model, system, prompt]).encode()).hexdigest()
    def _throttle(self, need):
        while True:
            now = time.time(); self.win = [(t, n) for t, n in self.win if now - t < 62]
            if sum(n for _, n in self.win) + need <= self.TPM: break
            time.sleep(min(5, 62 - (now - self.win[0][0]) + 0.1))
        self.win.append((time.time(), need))
    def chat(self, prompt, system=SYSTEM, max_tokens=300, effort="low"):
        k = self.key_of(MODEL, system, prompt)
        if k in self.cache: return self.cache[k]["text"], True
        if not self.key: raise SystemExit("Set GROQ_API_KEY (free key from console.groq.com) - this prompt is not in the cache.")
        need = int((len(system) + len(prompt)) / 3.6) + 120
        body = dict(model=MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    temperature=0, max_completion_tokens=max_tokens, reasoning_effort=effort)
        for attempt in range(8):
            self._throttle(need)
            r = self.rq.post("https://api.groq.com/openai/v1/chat/completions", json=body, timeout=90,
                             headers={"Authorization": f"Bearer {self.key}"})
            if r.status_code == 200:
                text = (r.json()["choices"][0]["message"].get("content") or "").strip()
                rec = dict(k=k, model=MODEL, text=text, usage=r.json().get("usage", {}).get("total_tokens"))
                open(self.path, "a").write(json.dumps(rec) + "\n"); self.cache[k] = rec; return text, False
            if r.status_code in (429, 500, 502, 503):
                msg = r.text[:300]
                if "per day" in msg or "TPD" in msg: raise SystemExit("Groq daily token limit reached; re-run later (cache keeps progress): " + msg)
                wait = float(r.headers.get("retry-after", 2 ** attempt * 3)); time.sleep(wait + 1); continue
            raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
        raise RuntimeError("Groq: too many retries")

def answer(index, llm, q, order, k=3):
    chunks = [index.CH[i] for i in order[:k]]
    text, cached = llm.chat(build_prompt(q, chunks))
    return text, chunks
