"""File 2 - the new project: a full RAG app on the SPPU corpus using the benchmarked retrieval.
  query cleaning -> hybrid (BM25 + bge-small, a=0.5) top-20 -> jina-reranker-v1-tiny-en -> top-3 -> Groq gpt-oss answer with citations.
Usage:
  export GROQ_API_KEY=...            # free key from console.groq.com
  python 2_rag_app.py "Which college is examination centre College-0002 for the S.Y.B.Sc. electronics practical?"
  python 2_rag_app.py                # interactive
  python 2_rag_app.py --retrieval-only "..."   # no API key needed, shows the cited chunks
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

if __name__ == "__main__":
    args = sys.argv[1:]; ro = "--retrieval-only" in args; args = [a for a in args if a != "--retrieval-only"]
    ix = R.Index(); llm = None if ro else R.Groq()
    if args: show(ix, llm, " ".join(args), ro)
    else:
        while True:
            try: q = input("\nQuestion (empty to quit): ").strip()
            except EOFError: break
            if not q: break
            show(ix, llm, q, ro)
