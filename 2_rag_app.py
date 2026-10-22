"""File 2 - the new project: a full RAG app on the SPPU corpus using the benchmarked retrieval.
  query cleaning -> hybrid (BM25 + bge-small, a=0.5) top-20 -> jina-reranker-v1-tiny-en -> top-3 -> Groq gpt-oss answer with citations.
Usage:
  export GROQ_API_KEY=...            # free key from console.groq.com
  python 2_rag_app.py "Which college is examination centre College-0002 for the S.Y.B.Sc. electronics practical?"
  python 2_rag_app.py                # interactive
  python 2_rag_app.py --retrieval-only "..."   # no API key needed, shows the cited chunks
  python 2_rag_app.py --serve                  # small web page on http://localhost:7860 (what Docker runs)
"""
import sys, ragcore as R

def ask(ix, llm, q, k=3, retrieval_only=False):
    order = ix.new_pipeline(q)
    chunks = [ix.CH[i] for i in order[:k]]
    cites = [f"[{n+1}] doc {c['doc']} page {c['page']}" for n, c in enumerate(chunks)]
    if retrieval_only: return None, chunks, cites
    text, _ = llm.chat(R.build_prompt(q, chunks))
    return text, chunks, cites

def show(ix, llm, q, ro):
    text, chunks, cites = ask(ix, llm, q, retrieval_only=ro)
    if text is not None: print("\nAnswer:", text)
    print("\nSources:"); [print(" ", c, "-", ch["text"][:160].replace("\n", " "), "...") for c, ch in zip(cites, chunks)]

PAGE = """<!doctype html><meta charset=utf-8><title>Hybrid RAG - SPPU</title>
<style>body{font-family:Arial;max-width:820px;margin:30px auto;color:#222}input{width:78%;padding:8px;font-size:16px}button{padding:8px 14px;font-size:16px}
.a{background:#eaf7ee;padding:12px;border-radius:6px;margin:14px 0;font-size:18px}.s{background:#f4f6f7;padding:8px;border-radius:6px;margin:6px 0;font-size:13px}</style>
<h2>Hybrid RAG on SPPU documents</h2><p>clean query, BM25 + dense, Jina rerank, gpt-oss answer with sources</p>
<form onsubmit="go();return false"><input id=q placeholder="Ask a question about the indexed SPPU documents"> <button>Ask</button></form><div id=o></div>
<script>async function go(){o.innerHTML='Working...';const r=await fetch('/ask?q='+encodeURIComponent(q.value));const d=await r.json();
o.innerHTML=(d.answer?'<div class=a>'+d.answer.replace(/</g,'&lt;')+'</div>':'')+d.sources.map(s=>'<div class=s><b>'+s.cite+'</b> '+s.text.replace(/</g,'&lt;')+'</div>').join('')}</script>"""

def serve(port=7860):
    import json as _j
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse, parse_qs
    ix = R.Index(); llm = R.Groq(); ro = not llm.key
    if ro: print("GROQ_API_KEY not set: serving retrieval only (no generated answer).")
    lock = __import__("threading").Lock()
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/ask":
                q = parse_qs(u.query).get("q", [""])[0].strip()
                with lock:
                    try: text, chunks, cites = ask(ix, llm, q, retrieval_only=ro) if q else (None, [], [])
                    except Exception as e: text, chunks, cites = f"Error: {e}", [], []
                body = _j.dumps(dict(answer=text, sources=[dict(cite=c, text=ch["text"][:300]) for c, ch in zip(cites, chunks)])).encode(); ct = "application/json"
            else: body = PAGE.encode(); ct = "text/html; charset=utf-8"
            self.send_response(200); self.send_header("Content-Type", ct); self.end_headers(); self.wfile.write(body)
        def log_message(self, *a): pass
    print(f"Open http://localhost:{port}"); ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()

if __name__ == "__main__":
    if sys.argv[1:2] == ["--serve"]: serve(); sys.exit()
    args = sys.argv[1:]; ro = "--retrieval-only" in args; args = [a for a in args if a != "--retrieval-only"]
    ix = R.Index(); llm = None if ro else R.Groq()
    if args: show(ix, llm, " ".join(args), ro)
    else:
        while True:
            try: q = input("\nQuestion (empty to quit): ").strip()
            except EOFError: break
            if not q: break
            show(ix, llm, q, ro)
