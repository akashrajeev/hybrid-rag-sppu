Unmodified copies of retrieval.py, embedding.py, config.py, answer_generation.py from the Review-1 project
(https://huggingface.co/spaces/akashrajeev/decodx, read-only copy, nothing pushed there).
The benchmark calls `dual_pass_retrieve` exactly as the app does (BM25 top-50 -> multilingual-MiniLM re-score of those 50
with 4 query variants -> keyword boost -> top pass2_k). Only differences for benchmarking (see ../README.md):
the chunks come from the shared corpus (data/chunks.json) so page-level gold labels apply, and the LLM context is the
top 3 chunks instead of up to 50, because 50 chunks exceed the Groq free tier's 8000 tokens/min.
