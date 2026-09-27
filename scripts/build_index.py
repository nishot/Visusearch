import numpy as np
import faiss

from src.config import settings


vectors = np.load(settings.embeddings_path, mmap_mode="r").astype("float32")
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
settings.index_path.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(settings.index_path))
print(f"Wrote {index.ntotal} vectors to {settings.index_path}")
