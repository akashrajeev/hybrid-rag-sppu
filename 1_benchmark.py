"""File 1 - runs every benchmark.  Steps (each resumable; re-run if it stops at its time budget):
  python 1_benchmark.py retrieval            # retrieval metrics, all 122 questions (no API key needed)
  python 1_benchmark.py rerankers            # optional: compare 4 rerankers on the hybrid top-20 (~25 min CPU)
  python 1_benchmark.py generate             # full RAG answers via Groq gpt-oss (needs GROQ_API_KEY, or the shipped cache)
  python 1_benchmark.py report               # print + write results/benchmark.md
"""
import sys, os, json, re, time
import numpy as np
import ragcore as R
HERE = R.HERE; RES = f"{HERE}/results"; os.makedirs(RES, exist_ok=True)
Q = json.load(open(f"{R.DATA}/questions.json")); QB = {q["id"]: q for q in Q}
BUDGET = float(os.environ.get("BUDGET_S", "100"))
def jload(p, d): return json.load(open(p)) if os.path.exists(p) else d
def tk(s): return re.findall(r"[a-z0-9]+", s.lower())
def f1(p, g):
    p, g = tk(p), tk(g)
    if not p or not g: return 0.0
    c = sum(min(p.count(w), g.count(w)) for w in set(p) & set(g))
    if not c: return 0.0
    pr, rc = c / len(p), c / len(g); return 2 * pr * rc / (pr + rc)
def correct(pred, gold): return f1(pred, gold) >= 0.5 or " ".join(tk(gold)) in " ".join(tk(pred))

# ---------- step: compute ranked lists for every system ----------
def orders():
    ix = R.Index(); path = f"{RES}/orders.json"; O = jload(path, {})
    for k in ["bm25", "dense", "hybrid", "hybrid_clean", "new"]: O.setdefault(k, {})
    t0 = time.time()
    for q in Q:
        i = str(q["id"]); s = q["q"]
        if i not in O["bm25"]:
            O["bm25"][i] = ix.bm25(s); O["dense"][i] = ix.dense(s); O["hybrid"][i] = ix.hybrid(s)
            O["hybrid_clean"][i] = ix.hybrid(s, use_clean=True)
        if i not in O["new"]:
            if time.time() - t0 > BUDGET: break
            O["new"][i] = ix.rerank(s, O["hybrid_clean"][i])
        json.dump(O, open(path, "w"))
    json.dump(O, open(path, "w"))
    # old Review-1 (DecodX) retrieval
    p = f"{RES}/old_retrieval.jsonl"
    if os.path.exists(p): O["old"] = {str(json.loads(l)["id"]): json.loads(l)["order"] for l in open(p)}
    return O, len(O["new"]) == len(Q)

def rr_of(order, q, CH, k=10):
    gold = {tuple(g) for g in q["gold"]}
    pos = next((i for i, c in enumerate(order[:k]) if (CH[c]["doc"], CH[c]["page"]) in gold), None)
    return pos
def boot(a, b, n=4000, seed=1):
    rs = np.random.RandomState(seed); d = np.array(a) - np.array(b)
    bs = [d[rs.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
NAMES = {"bm25": "BM25 only", "dense": "Dense only (bge-small)", "old": "OLD: Review-1 DecodX retrieval (as shipped)",
         "hybrid": "Hybrid a=0.5 (Review-1 prototype)", "hybrid_clean": "Hybrid + query cleaning",
         "new": "NEW: clean + hybrid + jina-tiny rerank"}
def retrieval():
    O, full = orders(); CH = R.Index().CH; out = {}
    for k, nm in NAMES.items():
        if k not in O or len(O[k]) < len(Q): print("incomplete:", k, len(O.get(k, {}))); continue
        rr = []; pos = []
        for q in Q:
            p = rr_of(O[k][str(q["id"])], q, CH); pos.append(p); rr.append(0 if p is None else 1 / (p + 1))
        h = lambda n: float(np.mean([p is not None and p < n for p in pos]))
        out[k] = dict(name=nm, mrr=float(np.mean(rr)), hit1=h(1), hit3=h(3), hit5=h(5), rr=rr)
    if "new" in out and "old" in out: out["paired_new_vs_old_mrr"] = boot(out["new"]["rr"], out["old"]["rr"])
    if "new" in out and "hybrid" in out: out["paired_new_vs_hybrid_mrr"] = boot(out["new"]["rr"], out["hybrid"]["rr"])
    json.dump(out, open(f"{RES}/retrieval.json", "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, dict) and "mrr" in v: print(f"{v['name']:45s} MRR {v['mrr']:.3f} Hit@1 {v['hit1']:.2f} Hit@3 {v['hit3']:.2f} Hit@5 {v['hit5']:.2f}")
        elif isinstance(v, dict): print(k, v)

# ---------- step: full RAG (retrieve -> rerank -> generate) ----------
GEN_SYSTEMS = os.environ.get("GEN_SYSTEMS", "old,new").split(",")
def generate():
    O, _ = orders(); llm = R.Groq(); ix = R.Index(); path = f"{RES}/generations.jsonl"
    done = {(json.loads(l)["sys"], json.loads(l)["id"]) for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a"); t0 = time.time(); todo = 0
    for q in Q:
        if not q.get("answer"): continue
        for s in GEN_SYSTEMS:
            if (s, q["id"]) in done: continue
            if time.time() - t0 > BUDGET: todo += 1; continue
            order = O[s][str(q["id"])]; text, ch = R.answer(ix, llm, q["q"], order)
            f.write(json.dumps(dict(sys=s, id=q["id"], pred=text, ctx=[c["cid"] for c in ch])) + "\n"); f.flush()
    n = sum(1 for q in Q if q.get("answer")) * len(GEN_SYSTEMS)
    print(f"generated {len(done)}+ this run; remaining {todo}; total needed {n}")
def report():
    R_ = json.load(open(f"{RES}/retrieval.json")); CH = R.Index().CH
    rows = [json.loads(l) for l in open(f"{RES}/generations.jsonl")]; G = {}
    for r in rows: G.setdefault(r["sys"], {})[r["id"]] = r
    ids = [q["id"] for q in Q if q.get("answer") and all(q["id"] in G.get(s, {}) for s in GEN_SYSTEMS)]
    out = {"n_answer_questions": len(ids), "model": R.MODEL}; vec = {}
    for s in GEN_SYSTEMS:
        c = []; sup = []; nf = []
        for i in ids:
            r = G[s][i]; q = QB[i]; ctx = " ".join(CH[x]["text"] for x in r["ctx"])
            c.append(correct(r["pred"], q["answer"]))
            sup.append(" ".join(tk(q["answer"])) in " ".join(tk(ctx)))   # gold answer text is inside the retrieved context
            nf.append("not found" in r["pred"].lower())
        vec[s] = np.array(c, float)
        # grounded = correct AND the gold answer appears in the retrieved context (answer is supported by what was retrieved)
        grounded = [bool(a and b) for a, b in zip(c, sup)]
        out[s] = dict(answer_correct=float(np.mean(c)), gold_in_context=float(np.mean(sup)), correct_and_grounded=float(np.mean(grounded)),
                      said_not_found=float(np.mean(nf)), correct_given_gold_in_context=float(np.mean([a for a, b in zip(c, sup) if b])) if any(sup) else None)
    out["paired_new_vs_old_answer_correct"] = boot(vec["new"], vec["old"])
    if "hybrid" in vec: out["paired_new_vs_hybrid_answer_correct"] = boot(vec["new"], vec["hybrid"])
    json.dump(out, open(f"{RES}/generation.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    md = ["# Benchmark results", "", f"Retrieval: {len(Q)} questions. Full RAG: {len(ids)} questions with a gold answer string; LLM {R.MODEL} (Groq free tier), same prompt, top-3 chunks for every system.", "",
          "| System | MRR | Hit@1 | Hit@3 | Hit@5 |", "|---|---|---|---|---|"]
    for k, v in R_.items():
        if isinstance(v, dict) and "mrr" in v: md.append(f"| {v['name']} | {v['mrr']:.3f} | {v['hit1']:.2f} | {v['hit3']:.2f} | {v['hit5']:.2f} |")
    md += ["", "| System | Answer correct | Gold answer in context | Correct and grounded | Said NOT FOUND |", "|---|---|---|---|---|"]
    for s in GEN_SYSTEMS:
        v = out[s]; md.append(f"| {NAMES[s]} | {v['answer_correct']:.3f} | {v['gold_in_context']:.3f} | {v['correct_and_grounded']:.3f} | {v['said_not_found']:.3f} |")
    md += ["", f"Paired bootstrap new vs old, answer correct: {out['paired_new_vs_old_answer_correct']}", f"Paired new vs old MRR: {R_.get('paired_new_vs_old_mrr')}"]
    open(f"{RES}/benchmark.md", "w").write("\n".join(md) + "\n")

# ---------- optional: reranker comparison ----------
def rerankers():
    import fastembed.rerank.cross_encoder as ce
    names = ["jinaai/jina-reranker-v1-tiny-en", "jinaai/jina-reranker-v1-turbo-en", "Xenova/ms-marco-MiniLM-L-6-v2", "Xenova/ms-marco-MiniLM-L-12-v2"]
    ix = R.Index(); CH = ix.CH; path = f"{RES}/rerankers_raw.json"; S = jload(path, {}); t0 = time.time()
    O, _ = orders()
    for nm in names:
        m = ce.TextCrossEncoder(model_name=nm); S.setdefault(nm, {})
        for q in Q:
            i = str(q["id"]); base = ix.hybrid(q["q"], 20)
            if i in S[nm]: continue
            if time.time() - t0 > BUDGET: json.dump(S, open(path, "w")); print("budget reached; re-run"); return
            st = time.time(); sc = [float(x) for x in m.rerank(q["q"], [CH[c]["text"] for c in base], batch_size=4)]
            S[nm][i] = dict(cand=base, s=sc, t=time.time() - st)
    json.dump(S, open(path, "w")); hy = []; res = {}
    for q in Q:
        p = rr_of(ix.hybrid(q["q"], 20), q, CH); hy.append(0 if p is None else 1 / (p + 1))
    for nm in names:
        rr = []; h1 = []; h5 = []
        for q in Q:
            r = S[nm][str(q["id"])]; o = [r["cand"][j] for j in np.argsort(-np.array(r["s"]))]; p = rr_of(o, q, CH)
            rr.append(0 if p is None else 1 / (p + 1)); h1.append(p == 0); h5.append(p is not None and p < 5)
        res[nm] = dict(mrr=float(np.mean(rr)), hit1=float(np.mean(h1)), hit5=float(np.mean(h5)), vs_hybrid=boot(rr, hy), sec_per_query=float(np.mean([S[nm][str(q['id'])]['t'] for q in Q])))
        print(nm, res[nm])
    json.dump(res, open(f"{RES}/rerankers.json", "w"), indent=1)
if __name__ == "__main__":
    {"retrieval": retrieval, "generate": generate, "report": report, "rerankers": rerankers}[sys.argv[1]]()
