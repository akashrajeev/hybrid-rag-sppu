# Benchmark results

Retrieval: 122 questions. Full RAG: 83 questions with a gold answer string; LLM openai/gpt-oss-20b (Groq free tier), same prompt, top-3 chunks for every system.

| System | MRR | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|
| BM25 only | 0.629 | 0.57 | 0.65 | 0.73 |
| Dense only (bge-small) | 0.456 | 0.38 | 0.48 | 0.59 |
| OLD: Review-1 DecodX retrieval (as shipped) | 0.522 | 0.44 | 0.55 | 0.61 |
| Hybrid a=0.5 (Review-1 prototype) | 0.729 | 0.69 | 0.75 | 0.79 |
| Hybrid + query cleaning | 0.767 | 0.70 | 0.80 | 0.88 |
| NEW: clean + hybrid + jina-tiny rerank | 0.817 | 0.75 | 0.87 | 0.90 |

| System | Answer correct | Gold answer in context | Correct and grounded | Said NOT FOUND |
|---|---|---|---|---|
| OLD: Review-1 DecodX retrieval (as shipped) | 0.398 | 0.518 | 0.398 | 0.506 |
| NEW: clean + hybrid + jina-tiny rerank | 0.663 | 0.819 | 0.663 | 0.157 |

Paired bootstrap new vs old, answer correct: {'mean': 0.26506024096385544, 'lo': 0.14457831325301204, 'hi': 0.37349397590361444}
Paired new vs old MRR: {'mean': 0.2952966432474629, 'lo': 0.22322810954983088, 'hi': 0.3707482760863908}
