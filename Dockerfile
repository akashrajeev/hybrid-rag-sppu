FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.cache FASTEMBED_CACHE_PATH=/app/.cache/fastembed LIBRARY_DIR=/data/library
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY ragcore.py ./
COPY pagewise ./pagewise
# bake the embedding model and the reranker into the image so the first upload is fast
RUN python -c "from pagewise import engine as E; E.embedder(); E.reranker()"
VOLUME /data
EXPOSE 7860
CMD ["python", "-m", "pagewise"]
