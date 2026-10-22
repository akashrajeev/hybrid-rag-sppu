import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from concurrent.futures import ThreadPoolExecutor

from config import MAX_WORKERS

embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = SentenceTransformer(embedding_model_name)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def embed_one(text: str) -> np.ndarray:
    return embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

async def embed_batch(texts: list, concurrency_limit=20) -> list:
    sem = asyncio.Semaphore(concurrency_limit)

    async def one_chunk(t):
        async with sem:
            return await asyncio.get_event_loop().run_in_executor(executor, embed_one, t)
    return await asyncio.gather(*[one_chunk(t) for t in texts])

def build_faiss_index(embeddings: list):
    matrix = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index
