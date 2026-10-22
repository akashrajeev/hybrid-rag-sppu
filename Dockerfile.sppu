FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.cache FASTEMBED_CACHE_PATH=/app/.cache/fastembed
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY . .
# download the public SPPU PDFs, rebuild + verify the chunks, and pre-download the embedding and reranker models into the image
RUN python fetch_data.py && python -c "import ragcore as R; ix=R.Index(); ix.emb(); ix.reranker(); ix.embedder()"
EXPOSE 7860
CMD ["python", "2_rag_app.py", "--serve"]
