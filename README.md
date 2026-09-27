# H&M Visual Search

Fast image-only retrieval over the local H&M catalog. The API loads the CLIP encoder, FAISS index, and metadata once at startup; requests encode only the uploaded image and search normalized vectors with inner-product similarity.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first model load downloads `openai/clip-vit-base-patch32`. Generate the full catalog assets once:

```powershell
python -m scripts.build_embeddings --batch-size 32
python -m scripts.build_index
uvicorn backend.main:app --reload
```

Embedding generation checkpoints after every batch. You can stop it with `Ctrl+C`, shut down the computer, and rerun the same command later; it resumes from the last completed batch. Use `--reset` only when the dataset or `--limit` changes and you intentionally want to discard the checkpoint.

Open http://127.0.0.1:8000. For a quick smoke test, use `--limit 1000` while developing.

## Performance notes

- Embeddings are generated offline and reused; no product image is encoded during a request.
- The API creates the model and FAISS index once through the FastAPI lifespan.
- `IndexFlatIP` is exact and fast for this catalog size. For larger catalogs, replace it with an IVF or HNSW implementation behind `src/retrieval/faiss_index.py`.
- Requests retrieve only `limit * SEARCH_CANDIDATE_MULTIPLIER` candidates before metadata filtering.
- Set `CLIP_DEVICE=cuda` on a CUDA machine; CPU remains supported.
