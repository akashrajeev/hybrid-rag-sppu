"""Download the 12 public SPPU PDFs listed in data/docs.json and build data/chunks.json.
Chunking: 170 words, step 120, per page; pages with fewer than min_words words (8, or 30 for the four syllabus PDFs) or <85% ASCII words are skipped
(scanned/Devanagari pages). Verifies the result against the expected checksum."""
import json, re, os, sys, hashlib, requests, pymupdf
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")
docs = json.load(open(f"{D}/docs.json")); os.makedirs(f"{D}/pdf", exist_ok=True)
for d in docs:
    p = f"{D}/pdf/{d['id']}.pdf"
    if os.path.exists(p): continue
    r = requests.get(d["url"], timeout=120); r.raise_for_status()
    assert r.content[:4] == b"%PDF", d["url"]
    open(p, "wb").write(r.content); print("downloaded", d["id"], len(r.content), flush=True)
chunks = []
for d in docs:
    pdf = pymupdf.open(f"{D}/pdf/{d['id']}.pdf")
    for i, pg in enumerate(pdf, 1):
        t = re.sub(r"\s+", " ", pg.get_text()).strip(); w = t.split()
        if len(w) < d.get("min_words", 8): continue
        if sum(bool(re.fullmatch(r"[\x00-\x7f]+", x)) for x in w) / len(w) < 0.85: continue
        for s in range(0, len(w), 120):
            c = w[s:s + 170]
            if len(c) < 25 and s > 0: break
            chunks.append(dict(cid=len(chunks), doc=d["id"], page=i, text=" ".join(c)))
json.dump(chunks, open(f"{D}/chunks.json", "w"))
h = hashlib.sha256(json.dumps([c["text"] for c in chunks]).encode()).hexdigest()
print(len(chunks), "chunks", h)
exp = open(f"{D}/chunks.sha256").read().strip()
print("MATCHES published corpus" if h == exp else "WARNING: differs from the corpus used for the published numbers (PDF changed or parser version differs)")
