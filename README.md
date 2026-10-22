# Hybrid RAG for identifier-heavy university documents (SPPU)

Reproducible benchmark and demo app for the Technical Seminar Review 2 work. Private repo; making it public is the owner's decision.

**Honest framing:** this is a measured comparison on an identifier-heavy domain (circulars, exam timetables, syllabi full of codes and tables), not a new retrieval method. The benchmark is small (122 questions, 31 hand-written, page-level gold) and the generation test uses 83 of them.

Licence: MIT for the code (see LICENSE). The SPPU documents themselves are not covered by it.

## Files
| File | What it does |
|---|---|
| `fetch_data.py` | Downloads the 12 public SPPU PDFs (`data/docs.json`), builds `data/chunks.json` (1,765 chunks), checks it against the published checksum |
| `1_benchmark.py` | **File 1.** Runs every benchmark: retrieval (6 systems), reranker comparison, full RAG old vs new |
| `2_rag_app.py` | **File 2.** The new project: query cleaning -> hybrid BM25 + bge-small -> jina-reranker-v1-tiny rerank -> top-3 -> Groq gpt-oss answer with cited pages |
| `ragcore.py` | Shared library: retrieval, reranking, cached and rate-limited Groq client |
| `old_run.py`, `old_decodx/` | The Review-1 project's own retrieval code (from https://huggingface.co/spaces/akashrajeev/decodx), used as the "old" baseline |
| `data/questions.json` | 122 questions with gold pages (and a gold answer string for 83) |
| `cache/llm_cache.jsonl` | Every Groq response used in the reported numbers, so the generation step reruns identically without a key |
| `results/` | Raw outputs and `benchmark.md` |

## Run the app with Docker (one command)
```bash
cp .env.example .env        # then paste your Groq key after GROQ_API_KEY=
docker compose up --build   # first build downloads the PDFs and models (a few minutes)
```
Open http://localhost:7860 and ask a question. Without a key the page still shows the retrieved, cited chunks (no generated answer). Needs about 1.5 GB of RAM. Without Docker: `python 2_rag_app.py --serve` after the steps below.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU only; torch/faiss are needed only for old_run.py
python fetch_data.py                      # prints "MATCHES published corpus"
python 1_benchmark.py retrieval           # ~2 min, no key needed; numbers below
BUDGET_S=100000 python 1_benchmark.py generate   # uses cache/llm_cache.jsonl; no key needed if cache is complete
python 1_benchmark.py report            # old vs new answer correctness (GEN_SYSTEMS=old,new,hybrid adds the Review-1 hybrid arm, only partly run)
python 1_benchmark.py rerankers           # optional, ~25 min on 1 CPU
python 2_rag_app.py "Which college is examination centre College-0002 for the S.Y.B.Sc. Computer Science electronics practical?"
```
`BUDGET_S` is a per-run time limit (default 100 s) so the steps can be resumed on small machines; set it high on a normal machine.
To regenerate answers instead of using the cache set `GROQ_API_KEY` (free key, no billing). Free tier is 8,000 tokens/min and 200,000 tokens/day, the client throttles itself and retries. Model: `openai/gpt-oss-20b` (override with `GROQ_MODEL`). `old_run.py` (re-running the old retrieval) needs `sentence-transformers` and `faiss-cpu`; its output is already shipped in `results/old_retrieval.jsonl`.

## Results (122 questions, page-level gold)
| System | MRR | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|
| BM25 only | 0.629 | 0.57 | 0.65 | 0.73 |
| Dense only (bge-small) | 0.456 | 0.38 | 0.48 | 0.59 |
| OLD: Review-1 DecodX retrieval (as shipped) | 0.522 | 0.44 | 0.55 | 0.61 |
| Hybrid a=0.5 (Review-1 prototype) | 0.729 | 0.69 | 0.75 | 0.79 |
| Hybrid + query cleaning | 0.767 | 0.70 | 0.80 | 0.88 |
| **NEW: clean + hybrid + jina-tiny rerank** | **0.817** | 0.75 | 0.87 | 0.90 |

New vs old MRR +0.295 (95% paired bootstrap CI 0.22 to 0.37). New vs plain hybrid +0.088 (CI 0.03 to 0.15).

Reranker comparison (rerank hybrid top-20, MRR; earlier run, see `results/rerankers.json` after running the optional step): jina-reranker-v1-tiny 0.806, jina turbo 0.800, MiniLM-L12 0.739, MiniLM-L6 0.724, none 0.729. bge-reranker-v2-m3 was too slow on a 1-CPU box (about 12 s/query) and was not run.

Full RAG, 83 questions with a gold answer, same LLM (gpt-oss-20b), same prompt, top-3 chunks for both:
| | Answer correct | Gold answer in the 3 chunks | Said NOT FOUND |
|---|---|---|---|
| OLD (DecodX retrieval) | 0.40 | 0.52 | 0.51 |
| NEW | 0.66 | 0.82 | 0.16 |

Paired difference +0.27 (CI +0.14 to +0.37). When the answer is in the context both do about equally (0.77 vs 0.81): the gain comes from retrieval.

## What was changed for the old baseline, and limits
- Old retrieval code is unmodified (`old_decodx/`). It runs on the shared chunks (not its own chunker) so page-level gold applies.
- The old app sends up to 50 chunks to the LLM; here both systems send the top 3, because 50 chunks exceed the free-tier token limit. Both use the same prompt (the old app's prompt/model are not used); its config default model is llama-3.1-8b-instant, here both use gpt-oss-20b.
- "Correct" = token F1 >= 0.5 or gold string contained. Gold answer strings are automatic anchors and some are truncated, so absolute numbers are slightly pessimistic for both systems. Faithfulness is only checked as "gold answer text appears in the retrieved chunks"; there is no LLM judge.
- Adaptive alpha weighting gave no gain over fixed a=0.5 and is not used. The reason MiniLM rerankers did not help is a hypothesis, not tested.
- Scanned and Devanagari pages are skipped (no OCR). One corpus, one domain.
