"""Runs the Review-1 (DecodX) retrieval code unchanged on the shared corpus; caches ordered chunk ids per question."""
import sys, os, json, time, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "old_decodx"))
import logging; logging.disable(logging.CRITICAL)
from retrieval import build_bm25_index, dual_pass_retrieve
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/results/old_retrieval.jsonl"
def main(budget=100):
    CH = json.load(open(f"{HERE}/data/chunks.json")); Q = json.load(open(f"{HERE}/data/questions.json"))
    texts = [c["text"] for c in CH]; idx = {}
    for i, t in enumerate(texts): idx.setdefault(t, i)
    bm = build_bm25_index(texts); done = set()
    if os.path.exists(OUT): done = {json.loads(l)["id"] for l in open(OUT)}
    t0 = time.time(); f = open(OUT, "a")
    for q in Q:
        if q["id"] in done: continue
        if time.time() - t0 > budget: print("budget reached, re-run to continue", len(done)); return False
        res = asyncio.run(dual_pass_retrieve(q["q"], texts, None, bm))
        f.write(json.dumps(dict(id=q["id"], order=[idx[t] for t in res])) + "\n"); f.flush(); done.add(q["id"])
    print("old retrieval complete", len(done)); return True
if __name__ == "__main__": main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
